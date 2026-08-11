import sqlite3
import json
import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from core.schema import DEFAULT_DEPARTMENT

DB_PATH = "mimir_portal.db"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _connect():
    return sqlite3.connect(DB_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    conn = _connect()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            token_hash TEXT PRIMARY KEY,
            label TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            user_id TEXT PRIMARY KEY,
            history_json TEXT NOT NULL
        )
    ''')

    # Additive migrations: the deployed database predates these columns and holds live
    # tokens, so it is widened in place rather than recreated.
    existing = {row[1] for row in c.execute("PRAGMA table_info(tokens)")}
    if "created_at" not in existing:
        c.execute("ALTER TABLE tokens ADD COLUMN created_at TEXT")
    if "last_used_at" not in existing:
        c.execute("ALTER TABLE tokens ADD COLUMN last_used_at TEXT")
    if "department" not in existing:
        c.execute("ALTER TABLE tokens ADD COLUMN department TEXT")
        # Every token issued before this column existed was implicitly scoped to the
        # reference deployment's own department - it was the only one in the system.
        # Defaulting them here means department scoping is enforced for every existing
        # token immediately, not just newly issued ones.
        c.execute("UPDATE tokens SET department=? WHERE department IS NULL", (DEFAULT_DEPARTMENT,))

    # Append-only access log. Deliberately stores the token label and a short hash prefix
    # rather than the token itself, so the trail stays useful for answering "who saw what"
    # without becoming a second place credentials can leak from.
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT,
            actor_ref TEXT,
            event TEXT NOT NULL,
            ip TEXT,
            detail TEXT
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts DESC)")

    # Thumbs up/down on an answer. citations is a JSON-encoded list of document labels, kept
    # denormalized here rather than joined against anything: feedback needs to remain readable
    # even after the corpus that produced it has changed.
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT,
            actor_ref TEXT,
            verdict TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT,
            citations TEXT,
            model TEXT,
            latency_ms INTEGER,
            comment TEXT
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback(ts DESC)")

    # One row per query, classified answered/refused. Purpose is corpus gap analysis: grouped
    # by query_norm, refused queries become a ranked list of what the corpus is missing,
    # generated from real usage rather than guesswork.
    c.execute('''
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            actor TEXT,
            actor_ref TEXT,
            query TEXT NOT NULL,
            query_norm TEXT NOT NULL,
            outcome TEXT NOT NULL,
            model TEXT,
            latency_ms INTEGER,
            evidence_count INTEGER
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_query_log_norm ON query_log(query_norm)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_query_log_outcome ON query_log(outcome)")

    # Additive: citations weren't tracked when query_log was first added (Phase 4 only needed
    # outcome/query_norm for gap analysis). Phase 7's "most-cited documents" needs document
    # identity per query, not just a count, so this widens the existing table in place.
    existing_ql = {row[1] for row in c.execute("PRAGMA table_info(query_log)")}
    if "citations" not in existing_ql:
        c.execute("ALTER TABLE query_log ADD COLUMN citations TEXT")

    # These are fixed, publicly-known strings in a public repository, and /api/login is an
    # unauthenticated route, so anyone who reads the source can sign in as an officer. They
    # are also re-inserted on every init_db() call, which means revoking one in the admin
    # panel silently un-revokes it at the next restart. Opt in explicitly for local work;
    # a deployment leaves this unset and issues real tokens from the admin panel.
    if os.environ.get("MIMIR_SEED_DEMO_TOKENS", "").strip().lower() in ("1", "true", "yes"):
        seed_tokens = [
            ("OFFICER-TOKEN-1", "Officer 1"),
            ("OFFICER-TOKEN-2", "Officer 2")
        ]

        for raw_token, label in seed_tokens:
            t_hash = hash_token(raw_token)
            c.execute("SELECT token_hash FROM tokens WHERE token_hash=?", (t_hash,))
            if not c.fetchone():
                c.execute(
                    "INSERT INTO tokens (token_hash, label, created_at) VALUES (?, ?, ?)",
                    (t_hash, label, _now()),
                )

    conn.commit()
    conn.close()


def validate_token(token: str) -> bool:
    """Returns True if the token exists in the DB."""
    conn = _connect()
    c = conn.cursor()
    t_hash = hash_token(token)
    c.execute("SELECT token_hash FROM tokens WHERE token_hash=?", (t_hash,))
    row = c.fetchone()
    conn.close()
    return bool(row)


def get_token_label(token: str) -> Optional[str]:
    """Resolve a raw token to its label so the audit trail can name the actor."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT label FROM tokens WHERE token_hash=?", (hash_token(token),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def touch_token(token: str) -> Optional[str]:
    """Stamp last_used_at and return the label, or None when the token is unknown."""
    conn = _connect()
    c = conn.cursor()
    t_hash = hash_token(token)
    c.execute("UPDATE tokens SET last_used_at=? WHERE token_hash=?", (_now(), t_hash))
    updated = c.rowcount
    label = None
    if updated:
        c.execute("SELECT label FROM tokens WHERE token_hash=?", (t_hash,))
        row = c.fetchone()
        label = row[0] if row else None
    conn.commit()
    conn.close()
    return label


def record_audit(event: str, actor: Optional[str] = None, ip: Optional[str] = None,
                 detail: Optional[str] = None, token: Optional[str] = None) -> None:
    """Append one access-log entry. Never raises: auditing must not break the request."""
    try:
        actor_ref = hash_token(token)[:12] if token else None
        conn = _connect()
        conn.execute(
            "INSERT INTO audit (ts, actor, actor_ref, event, ip, detail) VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), actor, actor_ref, event, ip, (detail or "")[:500]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def list_audit(limit: int = 200, event: Optional[str] = None) -> list:
    conn = _connect()
    c = conn.cursor()
    if event:
        c.execute(
            "SELECT ts, actor, actor_ref, event, ip, detail FROM audit WHERE event=? ORDER BY id DESC LIMIT ?",
            (event, limit),
        )
    else:
        c.execute(
            "SELECT ts, actor, actor_ref, event, ip, detail FROM audit ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    rows = c.fetchall()
    conn.close()
    return [
        {"ts": r[0], "actor": r[1], "actor_ref": r[2], "event": r[3], "ip": r[4], "detail": r[5]}
        for r in rows
    ]


def audit_summary() -> dict:
    """Counts for the admin panel header."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM audit")
    total = c.fetchone()[0]
    # strftime, not datetime(): datetime() yields "YYYY-MM-DD HH:MM:SS" while ts is ISO with a
    # 'T', and 'T' sorts above ' ', so the space form would match every row ever written.
    c.execute("SELECT COUNT(*) FROM audit WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-1 day')")
    last_day = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM audit WHERE event='auth.denied'")
    denied = c.fetchone()[0]
    conn.close()
    return {"total": total, "last_24h": last_day, "denied": denied}


def record_feedback(
    verdict: str, query: str, response: Optional[str] = None,
    citations: Optional[list] = None, model: Optional[str] = None,
    latency_ms: Optional[int] = None, comment: Optional[str] = None,
    actor: Optional[str] = None, token: Optional[str] = None,
) -> None:
    """Store one thumbs up/down. Unlike record_audit, this is allowed to raise: feedback is
    the request's actual content, not a side effect, so a write failure must reach the caller
    and the caller must tell the officer rather than silently reporting success."""
    actor_ref = hash_token(token)[:12] if token else None
    conn = _connect()
    conn.execute(
        """INSERT INTO feedback (ts, actor, actor_ref, verdict, query, response, citations,
                                  model, latency_ms, comment)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (_now(), actor, actor_ref, verdict, query, response,
         json.dumps(citations) if citations else None, model, latency_ms, comment),
    )
    conn.commit()
    conn.close()


def list_feedback(limit: int = 200, verdict: Optional[str] = None) -> list:
    conn = _connect()
    c = conn.cursor()
    query = ("SELECT id, ts, actor, verdict, query, response, citations, model, latency_ms, comment "
             "FROM feedback")
    params: tuple = ()
    if verdict:
        query += " WHERE verdict=?"
        params = (verdict,)
    query += " ORDER BY id DESC LIMIT ?"
    c.execute(query, params + (limit,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "ts": r[1], "actor": r[2], "verdict": r[3], "query": r[4],
            "response": r[5], "citations": (json.loads(r[6]) if r[6] else []),
            "model": r[7], "latency_ms": r[8], "comment": r[9],
        }
        for r in rows
    ]


def feedback_summary() -> dict:
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*), SUM(CASE WHEN verdict='up' THEN 1 ELSE 0 END) FROM feedback")
    total, up = c.fetchone()
    total = total or 0
    up = up or 0
    down = total - up
    c.execute("SELECT COUNT(*) FROM feedback WHERE ts >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-7 day')")
    last_7d = c.fetchone()[0]
    conn.close()
    satisfaction = round(100.0 * up / total, 1) if total else None
    return {"total": total, "up": up, "down": down, "satisfaction_pct": satisfaction, "last_7d": last_7d}


def import_legacy_feedback(json_path: str) -> int:
    """One-time migration from the old scratch/feedback.json into this table.

    Idempotent: only runs while the feedback table is still empty, so it is safe to call on
    every startup. Best-effort; a malformed or missing file is not fatal to boot.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM feedback")
    if c.fetchone()[0] > 0:
        conn.close()
        return 0
    conn.close()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            legacy = json.load(f)
    except Exception:
        return 0

    imported = 0
    conn = _connect()
    for entry in legacy:
        verdict = entry.get("feedback")
        query = entry.get("query")
        if not verdict or not query:
            continue
        conn.execute(
            "INSERT INTO feedback (ts, verdict, query, response, comment) VALUES (?, ?, ?, ?, ?)",
            (entry.get("timestamp") or _now(), verdict, query, entry.get("response"),
             "(imported from scratch/feedback.json)"),
        )
        imported += 1
    conn.commit()
    conn.close()
    return imported


def record_query_outcome(
    query: str, query_norm: str, outcome: str, actor: Optional[str] = None,
    token: Optional[str] = None, model: Optional[str] = None,
    latency_ms: Optional[int] = None, evidence_count: Optional[int] = None,
    citations: Optional[list] = None,
) -> None:
    """Log one query's classification. Best-effort like record_audit: a query having already
    streamed its answer to the officer by the time this runs, a logging failure here must not
    turn a successful answer into a 500."""
    try:
        actor_ref = hash_token(token)[:12] if token else None
        conn = _connect()
        conn.execute(
            """INSERT INTO query_log (ts, actor, actor_ref, query, query_norm, outcome, model,
                                       latency_ms, evidence_count, citations)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), actor, actor_ref, query, query_norm, outcome, model, latency_ms, evidence_count,
             json.dumps(citations) if citations else None),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def query_analytics(days: int = 30) -> dict:
    """Volume, refusal rate, latency percentiles and most-cited documents over a window.
    Phase 7: the operational counterpart to list_gaps - that answers "what's missing",
    this answers "how is the system actually performing and what does it get used for".
    """
    conn = _connect()
    c = conn.cursor()
    since = f"strftime('%Y-%m-%dT%H:%M:%S', 'now', '-{int(days)} day')"

    c.execute(f"SELECT COUNT(*), SUM(CASE WHEN outcome='refused' THEN 1 ELSE 0 END) "
             f"FROM query_log WHERE ts >= {since}")
    total, refused = c.fetchone()
    total, refused = total or 0, refused or 0

    c.execute(f"SELECT latency_ms FROM query_log WHERE ts >= {since} AND latency_ms IS NOT NULL "
             f"ORDER BY latency_ms")
    latencies = [r[0] for r in c.fetchall()]

    def _percentile(values, pct):
        if not values:
            return None
        idx = min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1))))
        return values[idx]

    c.execute(f"SELECT citations FROM query_log WHERE ts >= {since} AND citations IS NOT NULL")
    doc_counts: dict = {}
    for (raw,) in c.fetchall():
        try:
            for doc in json.loads(raw):
                doc_counts[doc] = doc_counts.get(doc, 0) + 1
        except Exception:
            continue
    top_documents = sorted(doc_counts.items(), key=lambda kv: -kv[1])[:10]

    c.execute(f"SELECT model, COUNT(*) FROM query_log WHERE ts >= {since} AND model IS NOT NULL "
             f"GROUP BY model ORDER BY COUNT(*) DESC")
    by_model = c.fetchall()

    conn.close()
    return {
        "total": total,
        "refused": refused,
        "refusal_rate_pct": round(100.0 * refused / total, 1) if total else None,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "top_documents": [{"document": d, "count": n} for d, n in top_documents],
        "by_model": [{"model": m, "count": n} for m, n in by_model],
    }


def list_gaps(limit: int = 100, days: int = 30) -> list:
    """Refused queries grouped by normalized text, most frequent first. Each group carries a
    sample of the original (unnormalized) phrasings and who asked, so an admin can see both
    the pattern and the actual questions behind it."""
    conn = _connect()
    c = conn.cursor()
    since = f"strftime('%Y-%m-%dT%H:%M:%S', 'now', '-{int(days)} day')"
    c.execute(f'''
        SELECT query_norm, COUNT(*) as n, MAX(ts) as last_seen
        FROM query_log
        WHERE outcome='refused' AND ts >= {since}
        GROUP BY query_norm
        ORDER BY n DESC, last_seen DESC
        LIMIT ?
    ''', (limit,))
    groups = c.fetchall()

    results = []
    for query_norm, count, last_seen in groups:
        c.execute('''
            SELECT query, actor, ts FROM query_log
            WHERE query_norm=? AND outcome='refused'
            ORDER BY id DESC LIMIT 5
        ''', (query_norm,))
        examples = [{"query": r[0], "actor": r[1], "ts": r[2]} for r in c.fetchall()]
        results.append({"query_norm": query_norm, "count": count, "last_seen": last_seen, "examples": examples})
    conn.close()
    return results


def gaps_summary(days: int = 30) -> dict:
    conn = _connect()
    c = conn.cursor()
    since = f"strftime('%Y-%m-%dT%H:%M:%S', 'now', '-{int(days)} day')"
    c.execute(f"SELECT COUNT(*) FROM query_log WHERE ts >= {since}")
    total = c.fetchone()[0]
    c.execute(f"SELECT COUNT(*) FROM query_log WHERE outcome='refused' AND ts >= {since}")
    refused = c.fetchone()[0]
    c.execute(f"SELECT COUNT(DISTINCT query_norm) FROM query_log WHERE outcome='refused' AND ts >= {since}")
    distinct_gaps = c.fetchone()[0]
    conn.close()
    refusal_rate = round(100.0 * refused / total, 1) if total else None
    return {"total": total, "refused": refused, "distinct_gaps": distinct_gaps, "refusal_rate_pct": refusal_rate}


def save_history(user_id: str, history_list: list):
    conn = _connect()
    c = conn.cursor()
    history_json = json.dumps(history_list)
    c.execute('''
        INSERT INTO history (user_id, history_json)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET history_json=excluded.history_json
    ''', (user_id, history_json))
    conn.commit()
    conn.close()


def get_history(user_id: str) -> list:
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT history_json FROM history WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return []


def generate_officer_token(label: str, department: str = DEFAULT_DEPARTMENT) -> str:
    """Generates a new token, hashes it, stores it, and returns the raw token."""
    raw_token = f"OFFICER-{secrets.token_hex(8).upper()}"
    t_hash = hash_token(raw_token)

    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO tokens (token_hash, label, created_at, department) VALUES (?, ?, ?, ?)",
        (t_hash, label, _now(), department),
    )
    conn.commit()
    conn.close()
    return raw_token


def list_tokens() -> list:
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT token_hash, label, created_at, last_used_at, department FROM tokens")
    rows = c.fetchall()
    conn.close()
    return [
        {"token_hash": r[0], "label": r[1], "created_at": r[2], "last_used_at": r[3], "department": r[4]}
        for r in rows
    ]


def get_token_department(token: str) -> Optional[str]:
    """Resolve a raw token to its department, so retrieval can filter by it.

    None (unknown token, e.g. the legacy shared MIMIR_AUTH_TOKEN) means unrestricted - it
    never has a row here, and there is nothing to scope it to.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT department FROM tokens WHERE token_hash=?", (hash_token(token),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def update_token_label(token_hash: str, new_label: str) -> bool:
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE tokens SET label=? WHERE token_hash=?", (new_label, token_hash))
    rows_affected = c.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def update_token_department(token_hash: str, department: str) -> bool:
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE tokens SET department=? WHERE token_hash=?", (department, token_hash))
    rows_affected = c.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def delete_token(token_hash: str) -> bool:
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM tokens WHERE token_hash=?", (token_hash,))
    rows_affected = c.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0
