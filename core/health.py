"""Live component probes, shared between /api/admin/topology and deploy.py status.

Extracted from app.py's api_admin_topology (Phase 5 of the build plan). Two definitions of
"is this component up" would drift the moment one of them changed and the other didn't; this
module is the single implementation both call.

Each probe function is self-contained rather than depending on an already-initialized global
client, because deploy.py runs as a standalone script outside the FastAPI process and has no
such globals — it creates its own short-lived clients via core/utils.py just like this module
does when none is supplied.
"""

import os
import time
from typing import Any, Optional
from urllib.parse import urlparse

import requests

import core.deployment as deployment


def host_of(url: str) -> str:
    try:
        parsed = urlparse(url if "//" in url else f"//{url}", scheme="http")
        return parsed.netloc or url
    except Exception:
        return url


def _service_headers(key_env: str) -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def probe(fn) -> dict:
    """Run one component probe, returning status and latency instead of raising."""
    started = time.perf_counter()
    try:
        detail = fn()
        return {"status": "up", "latency_ms": int((time.perf_counter() - started) * 1000), "detail": detail}
    except Exception as exc:
        return {"status": "down", "latency_ms": int((time.perf_counter() - started) * 1000), "detail": str(exc)[:140]}


def probe_weaviate(weaviate_client: Optional[Any] = None) -> str:
    owns_client = weaviate_client is None
    if owns_client:
        from core.utils import get_weaviate_client
        weaviate_client = get_weaviate_client()
    try:
        return "ready" if weaviate_client.is_ready() else "not ready"
    finally:
        if owns_client:
            weaviate_client.close()


def probe_embeddings() -> str:
    url = os.getenv("LOCAL_EMBED_URL", "")
    if not url:
        raise RuntimeError("LOCAL_EMBED_URL not set")
    r = requests.post(url, json={"input": "probe", "model": os.getenv("LOCAL_EMBED_MODEL_NAME", "BAAI/bge-m3")},
                      headers=_service_headers("LOCAL_EMBED_API_KEY"), timeout=8)
    r.raise_for_status()
    return f"{len(r.json()['data'][0]['embedding'])}-d vector"


def probe_rerank() -> str:
    url = os.getenv("LOCAL_RERANK_URL", "")
    if not url:
        raise RuntimeError("LOCAL_RERANK_URL not set")
    r = requests.post(url, json={"query": "probe", "documents": ["a", "b"],
                                 "model": os.getenv("LOCAL_RERANK_MODEL_NAME", "BAAI/bge-reranker-base"),
                                 "top_n": 2, "return_documents": False},
                      headers=_service_headers("LOCAL_RERANK_API_KEY"), timeout=8)
    r.raise_for_status()
    return f"{len(r.json()['results'])} candidates ranked"


def probe_translation() -> str:
    url = os.getenv("TRANSLATION_SERVICE_URL", "")
    if not url:
        raise RuntimeError("TRANSLATION_SERVICE_URL not set")
    r = requests.post(url, json={"text": "नमस्कार", "source_lang": "mar_Deva", "target_lang": "eng_Latn"}, timeout=12)
    r.raise_for_status()
    out = (r.json().get("translated_text") or "").strip()
    if not out:
        raise RuntimeError("empty translation")
    return out[:40]


def probe_generation(cerebras_client: Optional[Any] = None) -> str:
    local_gen = deployment.gen_provider() == "local"
    if local_gen:
        gen_url = os.getenv("LOCAL_GEN_URL", "http://localhost:11500/v1")
        r = requests.get(f"{gen_url.rstrip('/')}/models", headers=_service_headers("LOCAL_GEN_API_KEY"), timeout=8)
        r.raise_for_status()
        return os.getenv("LOCAL_GEN_MODEL", "qwen3:4b")

    owns_client = cerebras_client is None
    if owns_client:
        from core.utils import get_cerebras_client
        cerebras_client = get_cerebras_client()
    # Metadata call, not a completion: probing must not spend the per-minute chat budget.
    cerebras_client.models.list()
    return ", ".join(m.strip() for m in os.getenv("CEREBRAS_MODELS", "gpt-oss-120b,gemma-4-31b").split(","))


def component_list(weaviate_client: Optional[Any] = None, cerebras_client: Optional[Any] = None) -> list:
    """The six components with their probe functions attached (under the 'fn' key, popped by
    the caller before the result is serialized)."""
    local_gen = deployment.gen_provider() == "local"
    gen_url = os.getenv("LOCAL_GEN_URL", "http://localhost:11500/v1")

    return [
        {"name": "Vector store", "tech": "Weaviate", "role": "Hybrid dense + BM25 search",
         "host": host_of(os.getenv("WEAVIATE_URL", "")), "hosting": "self",
         "fn": lambda: probe_weaviate(weaviate_client)},
        {"name": "Embeddings", "tech": os.getenv("LOCAL_EMBED_MODEL_NAME", "BAAI/bge-m3"),
         "role": "Query and chunk vectors", "host": host_of(os.getenv("LOCAL_EMBED_URL", "")),
         "hosting": "self", "fn": probe_embeddings},
        {"name": "Reranker", "tech": os.getenv("LOCAL_RERANK_MODEL_NAME", "BAAI/bge-reranker-base"),
         "role": "Cross-encoder over candidates", "host": host_of(os.getenv("LOCAL_RERANK_URL", "")),
         "hosting": "self", "fn": probe_rerank},
        {"name": "Translation", "tech": "IndicTrans2", "role": "Marathi to English",
         "host": host_of(os.getenv("TRANSLATION_SERVICE_URL", "")), "hosting": "self", "fn": probe_translation},
        {"name": "Generation", "tech": ("Ollama" if local_gen else "Cerebras"), "role": "Answer synthesis",
         "host": (host_of(gen_url) if local_gen else "api.cerebras.ai"),
         "hosting": ("self" if local_gen else "third-party"), "fn": lambda: probe_generation(cerebras_client)},
        {"name": "Document parsing", "tech": "PyMuPDF", "role": "PDF to structured markdown",
         "host": "in-process", "hosting": "self", "fn": lambda: "local library"},
    ]


def run_all_probes(weaviate_client: Optional[Any] = None, cerebras_client: Optional[Any] = None) -> dict:
    """Synchronous, sequential probe of every component. Used by deploy.py, which has no
    event loop. app.py's /api/admin/topology runs the same component_list() concurrently via
    asyncio instead — see app.py for that variant."""
    components = component_list(weaviate_client, cerebras_client)
    for component in components:
        fn = component.pop("fn")
        component.update(probe(fn))

    local_gen = deployment.gen_provider() == "local"
    return {
        "components": components,
        "self_hosted": sum(1 for c in components if c["hosting"] == "self"),
        "third_party": sum(1 for c in components if c["hosting"] != "self"),
        "generation_provider": "local" if local_gen else "cerebras",
    }
