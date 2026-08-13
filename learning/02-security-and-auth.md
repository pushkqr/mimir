# 02 — Security, Auth & Extensibility

**In one line:** Policy documents are highly confidential; Mimir uses a combination of Zero-Trust Intranet Geofencing, isolated Token Identities, and strict API access controls to ensure your data never leaves the network.

---

## 1. Zero-Trust Intranet Geofencing

You cannot leave a government RAG endpoint exposed to the public internet, even with passwords. Mimir checks the network before it checks the credential, at the perimeter of the FastAPI application.

The allowlist is environment-driven through `MIMIR_ALLOWED_SUBNETS`, falling back to loopback and the RFC1918 private ranges when it is unset:

```python
_AUTHORIZED_SUBNETS = [
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),       # RFC1918 private ranges
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
]
```

Every request is checked against that list before authentication is considered. If someone steals a valid token and uses it from a home connection or a coffee shop, the middleware answers `403 Network Access Denied` and no model is ever invoked. Deploying inside a department means setting the variable to the department's range; exposing the system publicly requires deliberately widening it, so the secure posture is what you get by default.

Three details are easy to get wrong and worth stating. Caddy terminates TLS in front of the application, so the client address must be read from `X-Forwarded-For` rather than the socket, which would otherwise report the proxy. The network check applies to `/api/admin/*` as well: an earlier revision exempted admin routes on the grounds that they verify the admin token themselves, which left a hole in a perimeter that claimed to be zero-trust.

The third is a deliberate, narrow exemption. `/assets/*` — the stylesheet and webfonts — sits outside the *token* check, because a `<link>` tag cannot carry a bearer header, and without the exemption every page an unauthenticated visitor is meant to see (the landing page, the login form, the admin gate) would render as unstyled HTML. Those files stay **behind the network gate**, which is the control that actually matters here, and they contain no data: the exemption is scoped by path prefix and grants nothing beyond CSS and fonts.

This is the application-layer half of a pair. In the reference deployment, security groups already refuse the same traffic at the network layer, and only the application instance has a public address at all. Neither layer is asked to be the only one.

---

## 2. Token-Based Multi-Tenancy

Mimir discards vulnerable email/password architectures in favor of a secure, hashed Token Registry stored in an isolated SQLite database (`db.py`).

- **Secure Storage**: Only the SHA-256 hashes of the tokens are stored in the database.
- **Cross-Device Sync**: The frontend passes the token in the `Authorization: Bearer` header. The backend extracts the token, hashes it, and queries the database for that specific officer's chat history.
- **Data Isolation**: This guarantees multi-tenant data isolation. The IDOR vulnerabilities common in client-side architectures are impossible because the server dictates identity purely by the cryptographic token, not client-provided IDs.

---

## 3. The Admin Token CRUD API

Mimir features a fully baked API for IT departments to programmatically provision and manage officer access. Protected by the `MIMIR_ADMIN_TOKEN` environment variable, the backend exposes:

- `POST /api/admin/tokens`: Generates a random secure token, hashes it, and stores it.
- `GET /api/admin/tokens`: Returns a list of all active tokens (hashes only).
- `PUT /api/admin/tokens/{token_hash}`: Renames a token, or moves it to another department.
- `DELETE /api/admin/tokens/{token_hash}`: Instantly revokes an officer's access globally.

Each token carries a **department**, which scopes what that officer can retrieve. The value is one of the departments in `core/schema.py`, or the `ALL` sentinel — not a real department but a supervisor-level marker that skips the filter entirely. This is what makes one deployment serve several departments off a single index without one officer seeing another's corpus.

---

## 4. The Access Log

Identity tells you who someone is. It does not tell you what was done. `record_audit` appends an entry for logins, denied attempts, token issuance and revocation, uploads, promotions, and feedback, each carrying actor, client address, and timestamp. `/api/admin/audit` reads it back.

Two decisions in that function matter more than the feature:

- **Auditing never raises.** It is wrapped so that a logging failure cannot break the request that triggered it. An audit trail that can take down the service it observes will be switched off the first time it does.
- **The actor is a truncated hash of the token, not the token.** The log identifies who acted without becoming a second place the credential is stored.

## 5. Upload Quarantine

The obvious question about a document assistant is what stops someone adding a fabricated circular. The answer is that an administrator uploading a document does not write to the live corpus. Uploads land in a separate quarantine collection and enter retrieval only when an administrator explicitly promotes them, and both the upload and the promotion are recorded.

The honest limit: this establishes provenance, not authenticity. It records who promoted a document. It does not verify that the document is genuine.

## 6. A Failure Worth Recording

Development seeded two fixed officer tokens, `OFFICER-TOKEN-1` and `OFFICER-TOKEN-2`, so a fresh checkout had something to log in with. Three properties combined badly. The strings are in public source. `/api/login` is deliberately unauthenticated, since it is the login route. And the seeding ran on every startup, checking only whether each token was absent and re-inserting it if so.

That last property is the interesting one. Revoking a seeded token in the admin console worked, reported success, and lasted exactly until the next restart. The system had a revocation feature that silently did not persist for these particular tokens, which is worse than not having one.

Seeding is now behind `MIMIR_SEED_DEMO_TOKENS`, off unless explicitly set, and a deployment issues real tokens from the admin console instead. **A convenience default that is public, reachable, and self-healing is not a convenience.**

## 7. An Extensible Engine

Because Mimir is built as an agnostic engine, extending it to a new department (e.g., Department of Finance) requires zero backend security changes. You simply:
1. Spin up a new Weaviate collection with Finance documents.
2. Deploy the Mimir Engine.
3. Provision Finance Officer tokens via the Admin API.

The underlying security, auth, and geofencing work out of the box for any department.
