import json
import time
import queue
import logging
import asyncio
import threading
from pathlib import Path
import os
import hmac
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from google import genai
import uvicorn
import weaviate
import weaviate.classes as wvc
from core.utils import get_genai_client, get_cerebras_client, get_weaviate_client
from retrieval import run_retrieval
from retrieval.graph import build_citation_graph, load_citation_graph
from core.schema import (
    ensure_collection, ensure_department_property, CORPUS_COLLECTION, QUARANTINE_COLLECTION,
    DEPARTMENTS, DEFAULT_DEPARTMENT,
)

# Single source of truth for which collection is "the corpus" right now. Previously "GovDocs"
# was hardcoded at five separate call sites in this file; flipping to a differently-named
# collection (e.g. after a bulk ingest into GovDocsV2) meant editing all five and hoping none
# were missed. One env var, one rollback step.
ACTIVE_COLLECTION = os.environ.get("CORPUS_COLLECTION", CORPUS_COLLECTION).strip() or CORPUS_COLLECTION
from db import (
    init_db, validate_token, save_history, get_history,
    record_audit, touch_token, get_token_label, get_token_department, list_audit, audit_summary,
    record_feedback, list_feedback, feedback_summary, import_legacy_feedback,
    record_query_outcome, list_gaps, gaps_summary, query_analytics,
)
from core.outcome import classify_outcome, normalize_query
from core.health import component_list, probe as health_probe
import core.deployment as deployment
import ipaddress

logger = logging.getLogger(__name__)

app = FastAPI(title="Mimir")

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
DOCS_DIR = BASE_DIR / "docs"
QUARANTINE_DIR = BASE_DIR / "quarantine"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")


_AUTH_TOKEN = os.environ.get("MIMIR_AUTH_TOKEN", "").strip()
_ADMIN_TOKEN = os.environ.get("MIMIR_ADMIN_TOKEN", "SUPER-SECRET-ADMIN-TOKEN").strip()
_AUTH_OPEN = {"/", "/app", "/health", "/evidence", "/favicon.ico", "/favicon.svg", "/login", "/portal", "/api/login", "/admin"}

_AUTHORIZED_SUBNETS = []
_env_subnets = os.environ.get("MIMIR_ALLOWED_SUBNETS")
if _env_subnets:
    _AUTHORIZED_SUBNETS = [ipaddress.ip_network(s.strip()) for s in _env_subnets.split(",") if s.strip()]
else:
    _AUTHORIZED_SUBNETS = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("172.16.0.0/12"),
    ]

def _is_in_authorized_subnet(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return any(ip in subnet for subnet in _AUTHORIZED_SUBNETS)
    except ValueError:
        return False

def _client_ip(request: Request) -> str:
    """Real client address. Caddy terminates TLS, so request.client is the proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header[len("Bearer "):].strip() if header.startswith("Bearer ") else ""


def _is_authenticated(request: Request) -> bool:
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer "): return False
    token = h[len("Bearer "):].strip()
    
    if _AUTH_TOKEN and hmac.compare_digest(token, _AUTH_TOKEN):
        return True
        
    return validate_token(token)

@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    path = request.url.path
    is_admin_api = path.startswith("/api/admin/")

    # The network gate is the outermost control, so it runs on every path - including the
    # login page and the admin console, and regardless of _AUTH_OPEN. A device outside the
    # permitted range should never be served the form, let alone get to submit a credential
    # to it; being allowed to reach the door and refused at it is the weaker perimeter.
    # /health is the sole exception, so liveness probes and scratch/preflight.py keep
    # working; it discloses nothing beyond whether the process is up.
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        client_host = x_forwarded_for.split(",")[0].strip()
    else:
        client_host = request.client.host if request.client else ""

    if path != "/health":
        if not _is_in_authorized_subnet(client_host):
            record_audit("auth.denied", ip=client_host, detail=f"subnet block on {path}")
            return JSONResponse({
                "detail": "Network Access Denied. Device is outside authorized government intranet."
            }, status_code=403)

    if path not in _AUTH_OPEN or is_admin_api:
        if _AUTH_TOKEN and not is_admin_api:
            if not _is_authenticated(request):
                record_audit("auth.denied", ip=client_host, detail=f"invalid token on {path}")
                return JSONResponse({"detail": "Unauthorized — provide the access token."}, status_code=401)

    return await call_next(request)

gemini_client = None
cerebras_client = None
weaviate_client = None

@app.on_event("startup")
async def startup_event():
    global gemini_client, cerebras_client, weaviate_client
    try:
        init_db()
        legacy_path = BASE_DIR / "scratch" / "feedback.json"
        if legacy_path.exists():
            imported = import_legacy_feedback(str(legacy_path))
            if imported:
                logger.info(f"Imported {imported} legacy feedback entries from {legacy_path}")
        gemini_client = get_genai_client()
        cerebras_client = get_cerebras_client()
        weaviate_client = get_weaviate_client()
        ensure_department_property(weaviate_client, ACTIVE_COLLECTION)
        ensure_department_property(weaviate_client, QUARANTINE_COLLECTION)
        logger.info("Clients initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing clients: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_landing(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html")

@app.get("/app")
async def serve_app():
    """Retired in favour of /portal, which is the maintained officer interface.

    Kept as a redirect rather than removed so existing links and bookmarks still land
    somewhere sensible. templates/app.html is now unused.
    """
    return RedirectResponse(url="/portal", status_code=307)

@app.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/portal", response_class=HTMLResponse)
async def serve_portal(request: Request):
    return templates.TemplateResponse(request=request, name="portal.html")

class LoginRequest(BaseModel):
    token: str

@app.post("/api/login")
async def api_login(req: LoginRequest, request: Request):
    ip = _client_ip(request)
    label = touch_token(req.token)
    if label is None:
        record_audit("auth.denied", ip=ip, detail="officer login rejected")
        return JSONResponse({"error": "Invalid Officer Token."}, status_code=401)
    record_audit("auth.login", actor=label, ip=ip, token=req.token, detail="officer login")
    return {"token": req.token}

class HistorySaveRequest(BaseModel):
    user_id: str = None
    history: List[Dict[str, Any]]

@app.post("/api/history")
async def api_save_history(req: HistorySaveRequest, request: Request):
    h = request.headers.get("authorization", "")
    token = h[len("Bearer "):].strip()
    save_history(token, req.history)
    return {"status": "ok"}

@app.get("/api/history")
async def api_get_history(request: Request, user_id: str = None):
    h = request.headers.get("authorization", "")
    token = h[len("Bearer "):].strip()
    history = get_history(token)
    return {"history": history}

class TokenCreateRequest(BaseModel):
    label: str
    department: str = DEFAULT_DEPARTMENT

@app.get("/api/admin/departments")
async def api_admin_departments(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    # Single source of truth is core.schema.DEPARTMENTS - the admin UI's dropdown is
    # generated from this response rather than hardcoding the list a second time in HTML.
    return {"departments": sorted(DEPARTMENTS - {"ALL"}), "all": "ALL", "default": DEFAULT_DEPARTMENT}

@app.get("/api/admin/tokens")
async def api_admin_list_tokens(request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
    from db import list_tokens
    return {"tokens": list_tokens()}

@app.post("/api/admin/tokens")
async def api_admin_create_token(req: TokenCreateRequest, request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
    if req.department not in DEPARTMENTS:
        return JSONResponse({"error": f"Unknown department '{req.department}'."}, status_code=400)

    from db import generate_officer_token
    new_token = generate_officer_token(req.label, req.department)
    record_audit("token.issued", actor="Administrator", ip=_client_ip(request),
                 detail=f"issued token for '{req.label}' ({req.department})")
    return {"token": new_token, "label": req.label, "department": req.department}

class TokenUpdateRequest(BaseModel):
    label: Optional[str] = None
    department: Optional[str] = None

@app.put("/api/admin/tokens/{token_hash}")
async def api_admin_update_token(token_hash: str, req: TokenUpdateRequest, request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
    if req.department is not None and req.department not in DEPARTMENTS:
        return JSONResponse({"error": f"Unknown department '{req.department}'."}, status_code=400)

    from db import update_token_label, update_token_department
    found = False
    if req.label is not None and update_token_label(token_hash, req.label):
        found = True
        record_audit("token.renamed", actor="Administrator", ip=_client_ip(request),
                     detail=f"{token_hash[:12]} renamed to '{req.label}'")
    if req.department is not None and update_token_department(token_hash, req.department):
        found = True
        record_audit("token.department_changed", actor="Administrator", ip=_client_ip(request),
                     detail=f"{token_hash[:12]} moved to '{req.department}'")
    if found:
        return {"status": "ok"}
    return JSONResponse({"error": "Token not found."}, status_code=404)

@app.delete("/api/admin/tokens/{token_hash}")
async def api_admin_delete_token(token_hash: str, request: Request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or h[len("Bearer "):].strip() != _ADMIN_TOKEN:
        return JSONResponse({"error": "Admin access required."}, status_code=403)
    from db import delete_token
    if delete_token(token_hash):
        record_audit("token.revoked", actor="Administrator", ip=_client_ip(request),
                     detail=f"revoked {token_hash[:12]}")
        return {"status": "ok"}
    return JSONResponse({"error": "Token not found."}, status_code=404)

def _is_admin(request: Request) -> bool:
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer "):
        return False
    return hmac.compare_digest(h[len("Bearer "):].strip(), _ADMIN_TOKEN)


_FORBIDDEN = JSONResponse({"error": "Admin access required."}, status_code=403)

# Single-flight ingestion state. Ingestion is minutes-long and blocking, so it runs on a
# background thread and the panel polls this for progress.
_ingest = {"running": False, "file": None, "log": [], "error": None, "finished_at": None}


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin(request: Request):
    return templates.TemplateResponse(request=request, name="admin.html")


@app.post("/api/admin/login")
async def api_admin_login(req: LoginRequest, request: Request):
    ip = _client_ip(request)
    if not _ADMIN_TOKEN or not hmac.compare_digest(req.token.strip(), _ADMIN_TOKEN):
        record_audit("auth.denied", ip=ip, detail="admin login rejected")
        return JSONResponse({"error": "Invalid admin token."}, status_code=401)
    record_audit("admin.login", actor="Administrator", ip=ip, detail="admin console unlocked")
    return {"token": req.token.strip()}


@app.get("/api/admin/stats")
async def api_admin_stats(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    from db import list_tokens
    stats = {"chunks": None, "documents": 0, "pdfs": 0, "orgpedia": 0, "officers": len(list_tokens())}
    try:
        if DOCS_DIR.exists():
            # One source document is either a PDF or an Orgpedia GR. Orgpedia GRs ship as a
            # pair (<id>.pdf.txt original, <id>.pdf.en.txt translation), so count only the
            # .en.txt side to avoid double-counting one document as two.
            files = [p for p in DOCS_DIR.rglob("*") if p.is_file()]
            stats["files"] = len(files)
            stats["pdfs"] = sum(1 for p in files if p.suffix.lower() == ".pdf")
            # Orgpedia GRs ship as a pair per document (<id>.pdf.en.txt English,
            # <id>.pdf.mr.txt Marathi), so the file count is double the document count.
            stats["orgpedia"] = sum(1 for p in files if p.name.lower().endswith(".en.txt"))
            stats["documents"] = stats["pdfs"] + stats["orgpedia"]
    except Exception as e:
        logger.warning(f"Could not count source documents: {e}")
    try:
        agg = weaviate_client.collections.get(ACTIVE_COLLECTION).aggregate.over_all(total_count=True)
        stats["chunks"] = agg.total_count
    except Exception as e:
        logger.warning(f"Could not read Weaviate stats: {e}")
    return stats


@app.get("/api/admin/topology")
async def api_admin_topology(request: Request):
    """Live deployment map.

    Exists to answer "can this run inside our network" with a screen rather than an
    assertion. Every row states who hosts the component and whether it is reachable right
    now, so the self-hosted majority is visible rather than claimed.

    Probe implementations live in core/health.py, shared with `python deploy.py status` so
    the two never disagree about what "up" means. This endpoint only adds the concurrency
    (asyncio.gather over a thread pool) that a synchronous CLI script doesn't need.
    """
    if not _is_admin(request):
        return _FORBIDDEN

    components = component_list(weaviate_client, cerebras_client)
    loop = asyncio.get_running_loop()
    results = await asyncio.gather(*[
        loop.run_in_executor(None, health_probe, component.pop("fn")) for component in components
    ])
    for component, result in zip(components, results):
        component.update(result)

    local_gen = deployment.gen_provider() == "local"
    return {
        "components": components,
        "self_hosted": sum(1 for c in components if c["hosting"] == "self"),
        "third_party": sum(1 for c in components if c["hosting"] != "self"),
        "generation_provider": "local" if local_gen else "cerebras",
        "deployment": deployment.summary(),
        # Effective configuration, so an administrator can confirm what's actually running
        # without SSH. Model names only, never keys/URLs with embedded credentials.
        "config": {
            "collection": ACTIVE_COLLECTION,
            "embed_model": os.getenv("LOCAL_EMBED_MODEL_NAME", "BAAI/bge-m3"),
            "rerank_model": os.getenv("LOCAL_RERANK_MODEL_NAME", "BAAI/bge-reranker-base"),
            "generation_model": (os.getenv("LOCAL_GEN_MODEL", "qwen3:4b") if local_gen
                                else os.getenv("CEREBRAS_MODELS", "gpt-oss-120b,gemma-4-31b")),
        },
    }


_CHUNK_PROPERTIES = [
    "translated_text", "child_text", "parent_context", "document_title",
    "doc_number", "year", "issuing_authority", "document_category",
    "source_filename", "supersedes", "references", "department",
]


def _quarantine_chunks(filename: str):
    """Every chunk of one quarantined document, vectors included so promotion can copy them.

    Filtered client-side. Quarantine holds at most a handful of documents, and a filtered
    server-side query cannot return vectors through the same call.
    """
    collection = weaviate_client.collections.get(QUARANTINE_COLLECTION)
    return [
        obj for obj in collection.iterator(include_vector=True, return_properties=_CHUNK_PROPERTIES)
        if (obj.properties or {}).get("source_filename") == filename
    ]


@app.get("/api/admin/quarantine")
async def api_admin_quarantine(request: Request):
    """Documents staged but not yet part of the corpus."""
    if not _is_admin(request):
        return _FORBIDDEN

    staged = {}
    if QUARANTINE_DIR.exists():
        for path in sorted(QUARANTINE_DIR.glob("*.pdf")):
            staged[path.name] = {"filename": path.name, "bytes": path.stat().st_size, "chunks": 0}

    try:
        if weaviate_client is not None and weaviate_client.collections.exists(QUARANTINE_COLLECTION):
            collection = weaviate_client.collections.get(QUARANTINE_COLLECTION)
            for obj in collection.iterator(return_properties=["source_filename"]):
                name = (obj.properties or {}).get("source_filename")
                if name in staged:
                    staged[name]["chunks"] += 1
                elif name:
                    staged[name] = {"filename": name, "bytes": None, "chunks": 1}
    except Exception as exc:
        logger.warning(f"Could not count quarantined chunks: {exc}")

    return {"documents": list(staged.values())}


class QuarantineAction(BaseModel):
    filename: str


@app.post("/api/admin/quarantine/promote")
async def api_admin_quarantine_promote(req: QuarantineAction, request: Request):
    """Copy a reviewed document's chunks into the live corpus.

    Copies the stored vectors rather than re-running ingestion: re-embedding would cost
    another pass over the embedding service and could produce slightly different vectors
    than the ones an administrator actually reviewed.
    """
    if not _is_admin(request):
        return _FORBIDDEN
    if weaviate_client is None:
        return JSONResponse({"error": "Vector store unavailable."}, status_code=503)

    name = os.path.basename(req.filename or "").strip()
    if not name:
        return JSONResponse({"error": "No filename given."}, status_code=400)

    try:
        chunks = _quarantine_chunks(name)
        if not chunks:
            return JSONResponse({"error": f"No indexed chunks found for {name}."}, status_code=404)

        ensure_collection(weaviate_client, ACTIVE_COLLECTION)
        corpus = weaviate_client.collections.get(ACTIVE_COLLECTION)
        with corpus.batch.dynamic() as batch:
            for obj in chunks:
                vector = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector
                batch.add_object(properties=obj.properties, vector=vector)

        quarantine = weaviate_client.collections.get(QUARANTINE_COLLECTION)
        quarantine.data.delete_many(
            where=wvc.query.Filter.by_property("source_filename").equal(name)
        )

        staged_file = QUARANTINE_DIR / name
        if staged_file.is_file():
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            staged_file.replace(DOCS_DIR / name)
    except Exception as exc:
        logger.error(f"Promotion failed for {name}: {exc}")
        return JSONResponse({"error": str(exc)[:200]}, status_code=500)

    record_audit("document.promoted", actor="Administrator", ip=_client_ip(request),
                 detail=f"{name} promoted to corpus ({len(chunks)} chunks)")
    return {"status": "promoted", "filename": name, "chunks": len(chunks)}


@app.post("/api/admin/quarantine/discard")
async def api_admin_quarantine_discard(req: QuarantineAction, request: Request):
    """Drop a staged document and its chunks without it ever reaching the corpus."""
    if not _is_admin(request):
        return _FORBIDDEN
    if weaviate_client is None:
        return JSONResponse({"error": "Vector store unavailable."}, status_code=503)

    name = os.path.basename(req.filename or "").strip()
    if not name:
        return JSONResponse({"error": "No filename given."}, status_code=400)

    removed = 0
    try:
        if weaviate_client.collections.exists(QUARANTINE_COLLECTION):
            quarantine = weaviate_client.collections.get(QUARANTINE_COLLECTION)
            result = quarantine.data.delete_many(
                where=wvc.query.Filter.by_property("source_filename").equal(name)
            )
            removed = getattr(result, "successful", 0) or 0
        staged_file = QUARANTINE_DIR / name
        if staged_file.is_file():
            staged_file.unlink()
    except Exception as exc:
        logger.error(f"Discard failed for {name}: {exc}")
        return JSONResponse({"error": str(exc)[:200]}, status_code=500)

    record_audit("document.discarded", actor="Administrator", ip=_client_ip(request),
                 detail=f"{name} discarded from quarantine ({removed} chunks)")
    return {"status": "discarded", "filename": name, "chunks": removed}


@app.get("/api/admin/audit")
async def api_admin_audit(request: Request, limit: int = 200, event: str = ""):
    if not _is_admin(request):
        return _FORBIDDEN
    return {
        "entries": list_audit(limit=max(1, min(limit, 500)), event=(event or None)),
        "summary": audit_summary(),
    }


@app.post("/api/admin/upload")
async def api_admin_upload(request: Request, file: UploadFile = File(...)):
    if not _is_admin(request):
        return _FORBIDDEN
    name = os.path.basename(file.filename or "").strip()
    if not name.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are accepted."}, status_code=400)
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Empty file."}, status_code=400)
    # Uploads land in quarantine, never straight into the corpus. Whoever operates the console
    # is trusted to run it, not trusted to have verified the document, and an unreviewed file
    # silently joining the corpus would be indistinguishable from an authentic circular.
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    (QUARANTINE_DIR / name).write_bytes(data)
    record_audit("document.uploaded", actor="Administrator", ip=_client_ip(request),
                 detail=f"{name} ({len(data)} bytes) to quarantine")
    return {"filename": name, "bytes": len(data), "staged": "quarantine"}


def _ingest_job(filename: str, department: str):
    _ingest.update(running=True, file=filename, log=[f"Starting ingestion of {filename} into quarantine"],
                   error=None, finished_at=None)
    try:
        from ingestion import run_ingestion
        ensure_collection(weaviate_client, QUARANTINE_COLLECTION)
        ensure_department_property(weaviate_client, QUARANTINE_COLLECTION)
        records = run_ingestion(
            gemini_client,
            weaviate_client=weaviate_client,
            collection_name=QUARANTINE_COLLECTION,
            docs_dir=str(QUARANTINE_DIR),
            target_files=[filename],
            department=department,
        )
        _ingest["log"].append(f"Indexed {len(records)} chunks from {filename} into quarantine ({department})")
        _ingest["log"].append("Review it, then Promote to add it to the live corpus.")
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        _ingest["error"] = str(e)
        _ingest["log"].append(f"Failed: {e}")
    finally:
        _ingest["running"] = False
        _ingest["finished_at"] = __import__("datetime").datetime.now().isoformat()


class IngestRequest(BaseModel):
    filename: str
    department: str = DEFAULT_DEPARTMENT


@app.post("/api/admin/ingest")
async def api_admin_ingest(req: IngestRequest, request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    if req.department not in DEPARTMENTS or req.department == "ALL":
        return JSONResponse({"error": f"Invalid department '{req.department}'."}, status_code=400)
    if _ingest["running"]:
        return JSONResponse({"error": f"Ingestion already running for {_ingest['file']}."}, status_code=409)
    name = os.path.basename(req.filename or "").strip()
    if not (QUARANTINE_DIR / name).is_file():
        return JSONResponse({"error": f"{name} not found in quarantine."}, status_code=404)
    record_audit("document.ingested", actor="Administrator", ip=_client_ip(request),
                 detail=f"ingestion started for {name} (quarantine, {req.department})")
    threading.Thread(target=_ingest_job, args=(name, req.department), daemon=True).start()
    return {"status": "started", "filename": name, "department": req.department}


@app.get("/api/admin/ingest/status")
async def api_admin_ingest_status(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    return _ingest


@app.get("/api/admin/documents")
async def api_admin_documents(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    if not DOCS_DIR.exists():
        return {"documents": []}
    docs = [
        {"filename": p.name, "bytes": p.stat().st_size}
        for p in sorted(DOCS_DIR.rglob("*"), key=lambda x: x.name)
        if p.suffix.lower() == ".pdf"
    ]
    return {"documents": docs}


@app.get("/health")
async def health():
    local_gen = deployment.gen_provider() == "local"
    return {
        "auth": bool(_AUTH_TOKEN),
        "demo": False,
        "ready": True,
        "sovereign": deployment.current_mode() == "sovereign",
        "llm": {
            "label": (os.getenv("LOCAL_GEN_MODEL", "qwen3:4b") if local_gen
                     else os.getenv("CEREBRAS_MODELS", "gpt-oss-120b,gemma-4-31b").split(",")[0]),
            "local": local_gen,
        },
    }

@app.get("/workspaces")
async def workspaces():
    return {"workspaces": [{"id": "default", "name": "Mimir Workspace"}]}

@app.get("/systems")
async def systems():
    return {"systems": []}

@app.get("/curation")
async def curation():
    return {"findings": []}

@app.get("/curation/aging")
async def curation_aging():
    return {"aging": []}

@app.get("/timeline")
async def timeline():
    return {"events": []}

@app.get("/graph")
async def graph():
    """Citation graph over the corpus, from the cached build."""
    cached = load_citation_graph()
    if not cached:
        return {"nodes": [], "edges": [], "stats": {}, "built_at": None,
                "detail": "Not built yet. Rebuild it from the admin console."}
    return cached


@app.get("/api/admin/graph")
async def api_admin_graph(request: Request):
    """Same payload as /graph, but under the admin credential.

    /graph is officer-facing and sits behind the officer token; the admin console holds an
    admin token, which is deliberately not an officer token.
    """
    if not _is_admin(request):
        return _FORBIDDEN
    cached = load_citation_graph()
    if not cached:
        return {"nodes": [], "edges": [], "stats": {}, "built_at": None,
                "detail": "Not built yet. Use Rebuild."}
    return cached


@app.post("/api/admin/graph/rebuild")
async def api_admin_graph_rebuild(request: Request):
    if not _is_admin(request):
        return _FORBIDDEN
    if weaviate_client is None:
        return JSONResponse({"error": "Vector store unavailable."}, status_code=503)
    try:
        built = await asyncio.get_running_loop().run_in_executor(
            None, lambda: build_citation_graph(weaviate_client, ACTIVE_COLLECTION)
        )
    except Exception as exc:
        logger.error(f"Citation graph rebuild failed: {exc}")
        return JSONResponse({"error": str(exc)[:200]}, status_code=500)
    record_audit("graph.rebuilt", actor="Administrator", ip=_client_ip(request),
                 detail=f"{built['stats']['edges']} edges over {built['stats']['documents']} documents")
    return {"stats": built["stats"], "built_at": built["built_at"]}

@app.get("/llm-config")
async def llm_config():
    return {"provider": "gemini"}

@app.get("/ingest-status")
async def ingest_status():
    return {"running": False}

@app.get("/download/{filename}")
async def download_file(filename: str):
    for root_dir, _, files in os.walk(DOCS_DIR):
        if filename in files:
            return FileResponse(os.path.join(root_dir, filename))
    return JSONResponse({"error": "File not found"}, status_code=404)

class AskRequest(BaseModel):
    query: str
    history: List[Dict[str, Any]] = []
    workspace: Optional[str] = "default"

def _repair_callout_prefixes(stream):
    """Restore the '> ' blockquote prefixes on a leading [!WARNING] callout.

    The conflict callout only renders as a styled box if every line starts with '> '. Smaller
    local models (gemma3:12b measured at 0/3) reproduce the block's content and drop the
    prefixes, which degrades the most important answer in the product to plain text. Prompt
    emphasis did not fix it, so repair it deterministically instead.

    Only the opening block is buffered, and only when the answer actually starts with the
    marker; everything else streams through untouched.
    """
    buf = ""
    deciding = True
    for chunk in stream:
        if not deciding:
            yield chunk
            continue
        buf += chunk
        stripped = buf.lstrip()
        # Not a callout: flush and stop inspecting. Wait for a full first line before deciding,
        # otherwise a marker split across chunks looks like a miss.
        if stripped and not "[!WARNING]".startswith(stripped[:10]) and not stripped.startswith("[!WARNING]"):
            deciding = False
            yield buf
            buf = ""
            continue
        if "\n\n" in stripped:
            block, _, rest = stripped.partition("\n\n")
            fixed = "\n".join(
                ln if ln.lstrip().startswith(">") else "> " + ln
                for ln in block.split("\n") if ln.strip()
            )
            deciding = False
            yield fixed + "\n\n" + rest
            buf = ""
    if buf:
        stripped = buf.lstrip()
        if stripped.startswith("[!WARNING]"):
            buf = "\n".join(
                ln if ln.lstrip().startswith(">") else "> " + ln
                for ln in stripped.split("\n") if ln.strip()
            )
        yield buf


@app.post("/ask-stream")
async def ask_stream(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    history = data.get("history", [])

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    # Logged before retrieval runs, so an interrupted or failed query still leaves a trace.
    _token = _bearer(request)
    _department = get_token_department(_token) if _token else None
    record_audit("query", actor=(get_token_label(_token) if _token else None),
                 ip=_client_ip(request), token=(_token or None), detail=query)

    formatted_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "text" in msg:
            formatted_history.append({"role": msg["role"], "text": msg["text"]})

    async def event_generator():
        try:
            # run_retrieval is blocking and runs on a worker thread, so its progress callbacks
            # are handed back through a queue and drained here while we wait. Without this the
            # user stares at a static spinner for the whole retrieval.
            status_q: "queue.Queue[str]" = queue.Queue()

            def sync_status(msg: str):
                try:
                    status_q.put_nowait(msg)
                except Exception:
                    pass

            _t_start = time.perf_counter()
            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(
                None,
                lambda: run_retrieval(
                    gemini_client=gemini_client,
                    cerebras_client=cerebras_client,
                    weaviate_client=weaviate_client,
                    query=query,
                    collection_name=ACTIVE_COLLECTION,
                    chat_history=formatted_history,
                    status_callback=sync_status,
                    department=_department,
                )
            )

            while not task.done():
                try:
                    yield json.dumps({"status": status_q.get_nowait()}) + "\n"
                except queue.Empty:
                    await asyncio.sleep(0.05)

            retrieval_result = await task
            while True:
                try:
                    yield json.dumps({"status": status_q.get_nowait()}) + "\n"
                except queue.Empty:
                    break


            status = retrieval_result.get("status")
            if status == "error":
                yield json.dumps({"error": retrieval_result.get("response_text")}) + "\n"
                return
                
            answer_stream = retrieval_result.get("answer_stream")
            evidence = retrieval_result.get("evidence", [])
            recommendations = retrieval_result.get("recommendations", [])
            metrics = dict(retrieval_result.get("metrics") or {})

            first_token_at = None
            full_answer = []
            if answer_stream:
                for chunk in _repair_callout_prefixes(answer_stream):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    full_answer.append(chunk)
                    yield json.dumps({"t": chunk}) + "\n"
                    await asyncio.sleep(0.01)

            if first_token_at is not None:
                metrics["first_token_s"] = round(first_token_at - _t_start, 3)
            metrics["total_s"] = round(time.perf_counter() - _t_start, 3)

            outcome = classify_outcome("".join(full_answer), len(evidence))
            cited_docs = list({e.get("document") for e in evidence if e.get("document")})
            record_query_outcome(
                query=query, query_norm=normalize_query(query), outcome=outcome,
                actor=(get_token_label(_token) if _token else None), token=(_token or None),
                model=metrics.get("model"), latency_ms=round(metrics.get("total_s", 0) * 1000),
                evidence_count=len(evidence), citations=(cited_docs or None),
            )

            yield json.dumps({
                "done": True,
                "citations": evidence,
                "recommendations": recommendations,
                "metrics": metrics,
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error in ask-stream: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
            
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback: str  # 'up' | 'down'
    citations: Optional[List[str]] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    comment: Optional[str] = None

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request):
    if req.feedback not in ("up", "down"):
        return JSONResponse({"error": "feedback must be 'up' or 'down'."}, status_code=400)
    token = _bearer(request)
    try:
        record_feedback(
            verdict=req.feedback, query=req.query, response=req.response,
            citations=req.citations, model=req.model, latency_ms=req.latency_ms,
            comment=(req.comment or "").strip()[:1000] or None,
            actor=(get_token_label(token) if token else None), token=(token or None),
        )
        record_audit("feedback", actor=(get_token_label(token) if token else None),
                     ip=_client_ip(request), token=(token or None), detail=f"{req.feedback}: {req.query[:80]}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/admin/feedback")
async def api_admin_feedback(request: Request, limit: int = 200, verdict: str = ""):
    if not _is_admin(request):
        return _FORBIDDEN
    return {
        "entries": list_feedback(limit=max(1, min(limit, 500)), verdict=(verdict or None)),
        "summary": feedback_summary(),
    }


@app.get("/api/admin/gaps")
async def api_admin_gaps(request: Request, limit: int = 100, days: int = 30):
    """Refused queries grouped by normalized text, most frequent first — a ranked list of
    what the corpus is missing, built from what officers actually asked rather than guessed
    at. See core/outcome.py for how a query is classified and db.py:list_gaps for the grouping."""
    if not _is_admin(request):
        return _FORBIDDEN
    return {
        "groups": list_gaps(limit=max(1, min(limit, 500)), days=max(1, min(days, 365))),
        "summary": gaps_summary(days=max(1, min(days, 365))),
    }


@app.get("/api/admin/query-analytics")
async def api_admin_query_analytics(request: Request, days: int = 30):
    """Volume, refusal rate, latency percentiles and most-cited documents. The operational
    counterpart to /api/admin/gaps: that answers what's missing, this answers how the system
    is actually performing and what it gets used for."""
    if not _is_admin(request):
        return _FORBIDDEN
    return query_analytics(days=max(1, min(days, 365)))


class SummarizeRequest(BaseModel):
    doc_id: str

@app.post("/summarize")
async def summarize_doc(req: SummarizeRequest):
    try:
        weaviate_collection = weaviate_client.collections.get(ACTIVE_COLLECTION)
        res = weaviate_collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("doc_number").equal(req.doc_id),
            limit=50
        )
        pts = res.objects
        if not pts: return JSONResponse({"error": "Document not found."})
        text = "\n".join([(p.properties.get("child_text") or p.properties.get("parent_context") or "") for p in pts])
            
        resp = gemini_client.models.generate_content(
            model=os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash"),
            contents=f"Summarize the following document concisely:\n\n{text[:30000]}"
        )
        return {"summary": resp.text}
    except Exception as e:
        return JSONResponse({"error": str(e)})

class CompareRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str

@app.post("/compare")
async def compare_docs(req: CompareRequest):
    try:
        def fetch(did):
            weaviate_collection = weaviate_client.collections.get(ACTIVE_COLLECTION)
            res = weaviate_collection.query.fetch_objects(
                filters=wvc.query.Filter.by_property("doc_number").equal(did),
                limit=50
            )
            pts = res.objects
            return "\n".join([(p.properties.get("child_text") or p.properties.get("parent_context") or "") for p in pts])
                
        t1, t2 = fetch(req.doc_id_1), fetch(req.doc_id_2)
        resp = gemini_client.models.generate_content(
            model=os.getenv("GENAI_MODEL_NAME", "gemini-2.5-flash"),
            contents=f"Compare these two documents and highlight the differences:\n\nDocument 1:\n{t1[:15000]}\n\nDocument 2:\n{t2[:15000]}"
        )
        return {"comparison": resp.text}
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
