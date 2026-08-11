"""Citation graph over the indexed corpus.

Government Resolutions cite each other constantly: a circular opens with a "Reference:" block
naming the decisions it builds on, and an amending circular names the one it supersedes. Those
citations are already extracted at ingestion time into the `references` and `supersedes`
fields. This module resolves them into edges between documents that are actually in the
corpus.

Two things are deliberately conservative.

Nodes are keyed on `source_filename`, not `doc_number`. The document-number regex truncates on
messy OCR text, so 308 distinct doc_numbers cover 488 documents and a value like "NGC" spans
hundreds of chunks from unrelated circulars. Filenames are unique and reliable.

An edge is only drawn when a citation resolves to a small number of candidate documents. A
citation key that matches more than `ambiguity_cap` documents is dropped rather than fanned
out, because a wrong lineage edge in a system whose entire pitch is grounding is worse than a
missing one. Edges resolving to exactly one document are marked "resolved"; the rest are
"probable" and the UI distinguishes them.

Most citations in this corpus point at Government Resolutions that predate it and were never
ingested. Those are counted as unresolved rather than silently discarded, because the ratio is
the honest measure of how complete the corpus is.
"""

import json
import re
import collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.log_config import get_logger

logger = get_logger(__name__)

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "citation_graph.json"

# A Maharashtra GR number reduces to a department stem plus a year: "NGC-2010/(193/10)/Mashi-4"
# -> (NGC, 2010). The trailing serial and section suffix are too inconsistently OCR'd to key on.
_STEM_RE = re.compile(r"([A-Za-zऀ-ॿ]{3,})[\s-]*[-–]\s*(\d{4})")


# Titles come from the first markdown heading, which for OCR'd scans is very often the page
# marker rather than the subject line, a generic section header ("Preamble:", "Subject:"), or
# a heading truncated mid-word ("Narrow-", "Validation-"). Those make useless node labels -
# seen live in the citation graph as edges labelled "Preamble:" or "Narrow-" instead of a real
# document name. A short fragment ending in a colon or hyphen is never a real GR title, which
# reads as a full descriptive phrase.
_JUNK_TITLE_RE = re.compile(r"^\s*(page\s*\d+|#+\s*)?\s*$|^.{2,24}[:\-]\s*$", re.IGNORECASE)


def _pick_label(props: dict, filename: str) -> str:
    title = (props.get("document_title") or "").strip()
    if title and not _JUNK_TITLE_RE.match(title) and len(title) > 6:
        return title
    number = (props.get("doc_number") or "").strip()
    if number and len(number) > 3:
        return number
    return filename


def _citation_keys(text: str) -> set:
    if not text:
        return set()
    return {(m.group(1).upper().strip("-"), m.group(2)) for m in _STEM_RE.finditer(text)}


def build_citation_graph(
    weaviate_client: Any,
    collection_name: str = "GovDocs",
    ambiguity_cap: int = 3,
) -> Dict[str, Any]:
    """Scan the collection and resolve citations into a document-level graph."""
    collection = weaviate_client.collections.get(collection_name)

    own_keys = collections.defaultdict(collections.Counter)
    references = collections.defaultdict(list)
    supersedes = {}
    labels = {}
    years = {}
    chunk_counts = collections.Counter()

    for obj in collection.iterator(
        cache_size=200,
        return_properties=[
            "source_filename", "references", "supersedes",
            "child_text", "document_title", "doc_number", "year",
        ],
    ):
        props = obj.properties or {}
        filename = props.get("source_filename")
        if not filename:
            continue

        chunk_counts[filename] += 1
        # A document's own number, when doc_number parses, is authoritative. Falling back to
        # chunk-head text is unreliable on its own: the "Reference:" block sits near the top
        # too, so a cited number can out-vote the document's own.
        for key in _citation_keys(props.get("doc_number") or ""):
            own_keys[filename][key] += 100
        for key in _citation_keys((props.get("child_text") or "")[:900]):
            own_keys[filename][key] += 1

        if filename not in labels:
            labels[filename] = _pick_label(props, filename)
            years[filename] = props.get("year")

        reference_text = (props.get("references") or "").strip()
        if reference_text and len(references[filename]) < 12:
            references[filename].append(reference_text)

        superseded = (props.get("supersedes") or "").strip()
        if superseded:
            supersedes[filename] = superseded

    # Canonical number per document: the key appearing most often in its own chunk heads.
    # Ties break on the key itself, not on insertion order, so two builds over the same corpus
    # produce the same graph. Weaviate's iterator does not guarantee a stable order.
    canonical = {
        f: max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        for f, counter in own_keys.items() if counter
    }
    index = collections.defaultdict(list)
    for filename, key in canonical.items():
        index[key].append(filename)

    edges = {}
    unresolved = 0

    def add_edge(source: str, target: str, kind: str, confidence: str):
        if source == target:
            return
        existing = edges.get((source, target))
        # supersedes outranks references; resolved outranks probable.
        if existing:
            if existing["kind"] == "supersedes" or (
                existing["confidence"] == "resolved" and confidence != "resolved"
            ):
                return
        edges[(source, target)] = {"source": source, "target": target,
                                   "kind": kind, "confidence": confidence}

    for filename, clauses in references.items():
        for key in _citation_keys(" ".join(clauses)):
            candidates = index.get(key, [])
            if not candidates or len(candidates) > ambiguity_cap:
                unresolved += 1
                continue
            confidence = "resolved" if len(candidates) == 1 else "probable"
            for target in candidates:
                add_edge(filename, target, "references", confidence)

    for filename, superseded_number in supersedes.items():
        for key in _citation_keys(superseded_number):
            candidates = index.get(key, [])
            if not candidates or len(candidates) > ambiguity_cap:
                unresolved += 1
                continue
            confidence = "resolved" if len(candidates) == 1 else "probable"
            for target in candidates:
                add_edge(filename, target, "supersedes", confidence)

    edge_list = sorted(edges.values(), key=lambda e: (e["source"], e["target"]))
    degree = collections.Counter()
    for edge in edge_list:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1

    nodes = [
        {
            "id": filename,
            "label": (labels.get(filename) or filename)[:110],
            "number": "-".join(canonical[filename]) if filename in canonical else None,
            "year": years.get(filename),
            "chunks": chunk_counts[filename],
            "degree": degree.get(filename, 0),
        }
        for filename in sorted(chunk_counts)
    ]

    graph = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": collection_name,
        "nodes": nodes,
        "edges": edge_list,
        "stats": {
            "documents": len(nodes),
            "chunks": sum(chunk_counts.values()),
            "edges": len(edge_list),
            "resolved": sum(1 for e in edge_list if e["confidence"] == "resolved"),
            "probable": sum(1 for e in edge_list if e["confidence"] == "probable"),
            "supersedes": sum(1 for e in edge_list if e["kind"] == "supersedes"),
            "connected_documents": sum(1 for n in nodes if n["degree"] > 0),
            "citations_outside_corpus": unresolved,
        },
    }

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(
        f"Citation graph built: {graph['stats']['edges']} edges over "
        f"{graph['stats']['connected_documents']} connected documents"
    )
    return graph


def load_citation_graph() -> Optional[Dict[str, Any]]:
    """Return the cached graph, or None when it has never been built."""
    if not GRAPH_PATH.exists():
        return None
    try:
        return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read citation graph: {exc}")
        return None


def neighbours(graph: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """One-hop neighbourhood of a document, split by direction."""
    if not graph:
        return {"cites": [], "cited_by": []}
    labels = {n["id"]: n for n in graph.get("nodes", [])}
    cites, cited_by = [], []
    for edge in graph.get("edges", []):
        if edge["source"] == filename:
            cites.append({**edge, "node": labels.get(edge["target"])})
        elif edge["target"] == filename:
            cited_by.append({**edge, "node": labels.get(edge["source"])})
    return {"cites": cites, "cited_by": cited_by}
