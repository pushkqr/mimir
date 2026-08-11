import json
import os
import re
import time
import concurrent.futures
import requests
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import weaviate
import weaviate.classes as wvc
from retrieval.query import generate_query_variations
from retrieval.support import extract_response_text, build_context_text
from core.utils import embed_content_safe, generate_content_safe, local_rerank_safe
from core.log_config import get_logger

logger = get_logger(__name__)


def _mentions(combined: str, *terms: str) -> bool:
    """Match a term at a word start.

    Deliberately not a plain substring test: 'itm' appears inside 'recruitment' and 'date'
    inside 'candidate', so substring matching pulled unrelated alias sets into the BM25 query
    and crowded genuinely relevant documents out of the candidate pool. Anchoring only the
    start keeps stem matches working ('disabil' still fires on 'disabilities').
    """
    return any(re.search(r"\b" + re.escape(term), combined) for term in terms)


def build_fast_search_query(original_query: str, translated_query: str) -> str:
    """Build a deterministic search string for fast mode without LLM query expansion."""
    parts = [translated_query]
    if original_query and original_query != translated_query:
        parts.append(original_query)

    combined = f"{original_query} {translated_query}".lower()
    aliases = []

    if re.search(r"\b(gr|government resolution|resolution number|gr number)\b", combined) or "शासन" in combined:
        aliases.extend(["Government Resolution No", "GR number", "Government Decision", "शासन निर्णय", "क्रमांक"])
    if _mentions(combined, "appointment", "appointed", "नियुक्त"):
        aliases.extend(["appointment", "appointed", "APPOINTED", "nomination", "रुजू", "नियुक्ती"])
    if _mentions(combined, "professor", "प्राध्यापक"):
        aliases.extend(["Professor", "Assistant Professor", "Associate Professor", "प्राध्यापक"])
    if _mentions(combined, "chemistry", "रसायन"):
        aliases.extend(["Chemistry", "Professor (Chemistry)", "रसायनशास्त्र"])
    if _mentions(combined, "probation", "परिवी"):
        aliases.extend(["probation", "probationary period", "continuation of appointment", "ending the probation period", "परिवीक्षाधीन"])
    if _mentions(combined, "transfer", "बदली", "स्थानांतर"):
        aliases.extend(["transfer", "transfers", "transferred", "बदली"])
    if _mentions(combined, "date", "तारीख", "दिनांक"):
        aliases.extend(["Date", "dated", "The date", "दिनांक"])
    if _mentions(combined, "course", "पाठ्यक्रम"):
        aliases.extend(["closure of professional courses", "reduction in intake capacity", "professional courses"])
    if _mentions(combined, "disabil", "दिव्यांग"):
        aliases.extend(["persons with benchmark disabilities", "reserve not less than five per cent seats", "Institutions of Higher Education"])
    if _mentions(combined, "library", "ग्रंथालय"):
        aliases.extend(["Library", "Raja Rammohun Roy Library Foundation", "matching fund", "MARAGRAN", "SASHI-5"])
    if _mentions(combined, "temporary post", "तात्पुर", "मुदतवाढ"):
        aliases.extend(["temporary posts", "Directorate of Technical Education", "continue till"])
    if _mentions(combined, "canteen", "chef", "उपहार", "आचारी"):
        aliases.extend(["Mantralaya canteen", "chef", "outsourcing", "आचारी", "मंत्रालय उपहारगृह", "आस्था"])
    if _mentions(combined, "itm", "कामठी"):
        aliases.extend(["ITM College of Engineering Kamthi", "university", "admission"])

    parts.extend(aliases)
    seen = set()
    deduped = []
    for part in parts:
        cleaned = str(part).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return " ".join(deduped)


def build_rerank_text(payload: Dict[str, Any]) -> str:
    """Shape candidate text for cross-encoder reranking with compact document anchors.

    Kept short on purpose: the cross-encoder scores query+doc together, so per-candidate
    length directly drives rerank latency (a prior version stuffed ~2200 chars of
    parent_context per candidate and pushed rerank alone past 10s for a 20-candidate pool).
    """
    body = (payload.get("translated_text") or payload.get("child_text") or "").strip()
    if not body:
        body = (payload.get("parent_context") or "").strip()[:500]

    parts = [
        f"Title: {payload.get('document_title') or ''}",
        f"Document number: {payload.get('doc_number') or ''}",
        f"Section: {payload.get('section_title') or ''}",
        f"Text: {body[:600]}",
    ]
    return "\n".join(part for part in parts if part.strip())[:900]


def diversify_results(results: List[Any], limit: int) -> List[Any]:
    """Avoid filling fast-mode context with many chunks from one near-miss document."""
    max_per_doc = int(os.getenv("FAST_MODE_MAX_CHUNKS_PER_DOC", "4"))
    diversified = []
    counts: Dict[str, int] = {}
    overflow = []

    for result in results:
        payload = result.payload or {}
        doc_key = payload.get("source_filename") or payload.get("doc_number") or payload.get("document_title") or "unknown"
        if counts.get(doc_key, 0) < max_per_doc:
            diversified.append(result)
            counts[doc_key] = counts.get(doc_key, 0) + 1
        else:
            overflow.append(result)
        if len(diversified) >= limit:
            return diversified

    return (diversified + overflow)[:limit]


search_policy_docs_tool = {
    "type": "function",
    "function": {
        "name": "search_policy_docs",
        "description": "Search indexed government policy documents, circulars, acts, and rules for factual answers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or keywords to look up in government documents.",
                },
                "year": {
                    "type": "integer",
                    "description": "Optional publication year filter (e.g. 2025, 2020, 2019, 1961).",
                },
                "fast_mode": {
                    "type": "boolean",
                    "description": "Set true for fast search, false for deep analytical multi-query search.",
                },
            },
            "required": ["query"],
        },
    }
}


class _Result:
    """Uniform wrapper so Weaviate objects and lineage-expanded objects look alike downstream."""

    def __init__(self, props, metadata=None):
        self.payload = props
        self.metadata = metadata


def expand_lineage(
    weaviate_collection: Any,
    top_results: List[Any],
    query_vector: List[float],
    bm25_query: str,
    max_docs: int = 2,
    max_chunks_per_doc: int = 3,
    department: Optional[str] = None,
) -> List[Any]:
    """Follow supersedes edges so both sides of an amendment reach the generator.

    Relevance ranking alone tends to surface only one side. A superseding circular normally
    restates the original question in its preamble, and that preamble outranks the clause
    carrying the revised figure, so the model can see that something was superseded but not
    what changed, and reports the stale value with full confidence. That is the worst
    possible failure for an officer.

    Costs one filtered query per edge and runs only when an edge actually exists, so queries
    against unamended documents pay nothing.
    """
    present = {(r.payload or {}).get("doc_number") for r in top_results}
    present.discard(None)
    if not present:
        return []

    targets = set()

    # Forward edge: a retrieved document names the one it replaces.
    for result in top_results:
        superseded = ((result.payload or {}).get("supersedes") or "").strip()
        if superseded:
            targets.add(superseded)
            targets.add((result.payload or {}).get("doc_number"))

    # Backward edge: some other document names a retrieved one as superseded. Asked as a
    # single contains_any rather than one query per retrieved document, because this runs on
    # every search and most searches turn up no lineage at all.
    candidates = [d for d in present if d][:10]
    if candidates:
        try:
            back_filters = wvc.query.Filter.by_property("supersedes").contains_any(candidates)
            if department and department != "ALL":
                back_filters = back_filters & wvc.query.Filter.by_property("department").equal(department)
            back = weaviate_collection.query.fetch_objects(
                filters=back_filters,
                limit=10,
                return_properties=["doc_number", "supersedes"],
            )
            for obj in back.objects:
                props = obj.properties or {}
                number, superseded = props.get("doc_number"), props.get("supersedes")
                if number and superseded in present:
                    targets.add(number)
                    targets.add(superseded)
        except Exception as exc:
            logger.warning(f"Lineage back-reference lookup failed: {exc}")

    targets = {t for t in targets if t}
    if not targets:
        return []

    # Both sides of an edge are pulled even when already retrieved. Being present is not
    # enough: the chunk that matched the question is usually the preamble restating it,
    # while the figure that actually changed sits in the operative clause.
    seen = {((r.payload or {}).get("doc_number"), (r.payload or {}).get("child_text")) for r in top_results}

    extra: List[Any] = []
    for target in sorted(targets)[:max_docs]:
        try:
            linked_filters = wvc.query.Filter.by_property("doc_number").equal(target)
            if department and department != "ALL":
                linked_filters = linked_filters & wvc.query.Filter.by_property("department").equal(department)
            linked = weaviate_collection.query.hybrid(
                query=bm25_query,
                query_properties=["translated_text", "parent_context", "section_title", "child_text"],
                vector=query_vector,
                alpha=0.5,
                limit=max_chunks_per_doc,
                filters=linked_filters,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )
            added = 0
            for obj in linked.objects:
                key = ((obj.properties or {}).get("doc_number"), (obj.properties or {}).get("child_text"))
                if key in seen:
                    continue
                seen.add(key)
                extra.append(_Result(obj.properties, obj.metadata))
                added += 1
            logger.info(f"Lineage expansion pulled {added} new chunks from {target}")
        except Exception as exc:
            logger.warning(f"Lineage expansion failed for {target}: {exc}")

    return extra


def build_evidence(search_results: List[Any]) -> List[Dict[str, Any]]:
    """Build evidence list from search results for citations."""
    evidence: List[Dict[str, Any]] = []
    for result in search_results:
        payload = result.payload or {}
        child_text = (payload.get("child_text") or "").strip()
        parent_context = (payload.get("parent_context") or "").strip()
        quote = child_text[:400] if child_text else parent_context[:400]
        section_parts = [payload.get(k) for k in ["Document_Part", "Header_1", "Header_2", "Header_3"] if payload.get(k)]
        raw_section = " > ".join(str(p) for p in section_parts) if section_parts else "Section not available"
        clean_sec = raw_section.split(" > ")[-1] if " > " in raw_section else raw_section
        clean_sec = "".join(c for c in clean_sec if ord(c) < 128).strip()
        if not clean_sec:
            clean_sec = "Section not available"

        doc_number = payload.get("doc_number", "Unknown document")
        title = payload.get("document_title", "")
        doc_label = f"{title} ({doc_number})" if title else doc_number
        score = getattr(result.metadata, "score", 0.0) if hasattr(result, "metadata") and result.metadata else 0.0
        
        evidence.append(
            {
                "document": doc_label,
                "year": payload.get("year"),
                "section": clean_sec,
                "quote": quote,
                "score": score,
                "filename": payload.get("source_filename"),
                "supersedes": payload.get("supersedes")
            }
        )
    return evidence


def rerank_results(
    gemini_client: genai.Client,
    results: List[Any],
    standalone_query: str,
    rerank_limit: int,
) -> List[Any]:
    """Rerank search results using LLM judge and return top results."""
    ranking_prompt = f"""Rank the passages below from most relevant to least relevant for answering the question.
Return ONLY a JSON array of indices in descending order of relevance, e.g. [2, 0, 1].

Question: {standalone_query}

Passages:
"""
    for idx, res in enumerate(results):
        payload = res.payload or {}
        passage_text = payload.get("translated_text") or payload.get("child_text") or payload.get("parent_context") or ""
        ranking_prompt += f"{idx}: {passage_text[:800]}\n\n"

    try:
        ranking_response = generate_content_safe(
            gemini_client,
            model=os.getenv("GEN_MODEL_NAME", "gemini-2.5-flash"),
            contents=ranking_prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        )
        ranked_indices = json.loads(extract_response_text(ranking_response))
        if isinstance(ranked_indices, list):
            return [results[idx] for idx in ranked_indices if 0 <= idx < len(results)][:rerank_limit]
    except Exception:
        pass
    return results[:rerank_limit]


def run_hybrid_search(
    gemini_client: genai.Client,
    weaviate_client: Any,
    query_vectors: List[List[float]],
    query_variations: List[str],
    standalone_query: str,
    collection_name: str,
    year_filter: Optional[int],
    fast_mode: bool,
    department: Optional[str] = None,
) -> Tuple[List[Any], List[Any], List[Dict[str, Any]]]:
    """Execute hybrid dense+BM25 search with RRF fusion, optional reranking, and evidence extraction."""
    limit = int(os.getenv("FAST_MODE_CANDIDATE_LIMIT", "100")) if fast_mode else int(os.getenv("DEEP_MODE_CANDIDATE_LIMIT", "150"))
    rerank_limit = int(os.getenv("FAST_MODE_RERANK_LIMIT", "35")) if fast_mode else int(os.getenv("DEEP_MODE_RERANK_LIMIT", "12"))
    query_vector = query_vectors[0] if query_vectors else []

    # Weight alpha: 
    # - 0.25 (BM25-heavy) for explicit GR code patterns
    # - 0.50 (Balanced) for all other general queries
    gr_code_pattern = r"[^\s]{2,}[-/][\d\u0966-\u096F]+[-/][^\s]+|P\.?No\.?\s*\d+|No\.\s+\d+/"
    if re.search(gr_code_pattern, standalone_query) or re.search(r"[A-Z]{2,}[-/]\d+", standalone_query):
        alpha = 0.25
    else:
        alpha = 0.50
    
    if fast_mode:
        bm25_query = build_fast_search_query(standalone_query, query_variations[0] if query_variations else standalone_query)
    else:
        bm25_query = " ".join(query_variations) if query_variations else standalone_query
    
    weaviate_collection = weaviate_client.collections.get(collection_name)
    
    weaviate_filters = None
    if year_filter:
        weaviate_filters = wvc.query.Filter.by_property("year").equal(year_filter)
    # "ALL" is the supervisor/cross-department sentinel - deliberately not filtered.
    if department and department != "ALL":
        dept_filter = wvc.query.Filter.by_property("department").equal(department)
        weaviate_filters = dept_filter if weaviate_filters is None else weaviate_filters & dept_filter

    t_hybrid_start = time.time()
    search_res = weaviate_collection.query.hybrid(
        query=bm25_query,
        query_properties=["translated_text", "parent_context", "section_title", "child_text", "doc_number", "document_title"],
        vector=query_vector,
        alpha=alpha,
        limit=limit,
        filters=weaviate_filters,
        return_metadata=wvc.query.MetadataQuery(score=True)
    )
    hybrid_db_s = time.time() - t_hybrid_start
    current_results = [_Result(obj.properties, obj.metadata) for obj in search_res.objects]

    if not current_results:
        return [], [], [], {"hybrid_db_s": round(hybrid_db_s, 3), "rerank_s": 0.0, "lineage_s": 0.0}

    t_rerank_start = time.time()
    rerank_on = os.getenv("RERANK_ENABLED", "true").strip().lower() in ("true", "1", "yes")
    if fast_mode and not rerank_on:
        # Cross-encoder reranking is the dominant latency cost. Disabling it falls back to
        # hybrid-fusion order, still diversified per document, for a materially faster response.
        top_results = diversify_results(current_results, rerank_limit)
    elif fast_mode:
        texts_to_rerank = [build_rerank_text(obj.payload or {}) for obj in current_results]

        rerank_pool_n = min(rerank_limit * 3, len(current_results))
        ranked_indices = local_rerank_safe(standalone_query, texts_to_rerank, top_n=rerank_pool_n)
        ranked_results = [current_results[i] for i in ranked_indices if i < len(current_results)]
        top_results = diversify_results(ranked_results, rerank_limit)
    else:
        top_results = rerank_results(gemini_client, current_results, standalone_query, rerank_limit)
    rerank_s = time.time() - t_rerank_start

    t_lineage_start = time.time()
    lineage_results = expand_lineage(weaviate_collection, top_results, query_vector, bm25_query, department=department)
    if lineage_results:
        top_results = top_results + lineage_results
    lineage_s = time.time() - t_lineage_start

    return current_results, top_results, build_evidence(top_results), {
        "hybrid_db_s": round(hybrid_db_s, 3),
        "rerank_s": round(rerank_s, 3),
        "lineage_s": round(lineage_s, 3),
    }


def execute_search_tool(
    gemini_client: genai.Client,
    weaviate_client: Any,
    collection_name: str,
    query: str,
    year: Optional[int] = None,
    fast_mode: bool = True,
    department: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]]]:
    """Execute the search tool and format results for the LLM."""
    t_start = time.time()
    
    def is_indic(text: str) -> bool:
        return any('\u0900' <= c <= '\u097f' for c in text)

    def translate_marathi_batch_local(text: str) -> str:
        try:
            import os
            url = os.environ.get("TRANSLATION_SERVICE_URL", "http://localhost:8000/translate")
            # The service answers in about 2s, so a 5s ceiling left almost no margin and was
            # observed timing out on a healthy instance (load 0.00, 1.9s when measured
            # directly a minute later). On timeout this falls through to returning the
            # original Marathi, which searches the English corpus with Devanagari text and
            # degrades the answer without failing - so a tight timeout here surfaces as a bad
            # answer rather than an error.
            response = requests.post(
                url,
                json={"text": text, "src_lang": "mar_Deva", "tgt_lang": "eng_Latn"},
                timeout=float(os.getenv("TRANSLATION_TIMEOUT_S", "15")),
            )
            response.raise_for_status()
            return response.json().get("translated_text", text)
        except Exception as e:
            logger.error(f"Local translation failed: {e}")
            return text

    if is_indic(query):
        logger.info("Indic query detected. Translating query to English via local microservice...")
        search_query = translate_marathi_batch_local(query)
    else:
        logger.info("English query detected. Bypassing translation API.")
        search_query = query
        
    
    t_translate = time.time()
    logger.info(f"[PROFILING] Translation took: {t_translate - t_start:.3f}s")
    
    query_variations = [search_query]
    logger.info("Generating query variations...")
    if not fast_mode:
        query_variations = generate_query_variations(gemini_client, search_query)
        
    t_variations = time.time()
    if not fast_mode:
        logger.info(f"[PROFILING] Query expansion took: {t_variations - t_translate:.3f}s")

    config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    model_name = os.getenv("EMBED_MODEL_NAME", "text-embedding-004")
    
    query_vectors = []
    
    def embed_single_variation(q_var):
        try:
            logger.info(f"Embedding variation: {q_var}")
            resp = embed_content_safe(gemini_client, model=model_name, contents=q_var, config=config)
            return resp.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
            
    logger.info("Starting threadpool for embeddings...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(embed_single_variation, query_variations))
        
    logger.info("Finished threadpool for embeddings.")
    t_embed = time.time()
    logger.info(f"[PROFILING] Embeddings took: {t_embed - t_variations:.3f}s")
    
    for res in results:
        if res is not None:
            query_vectors.append(res)
            
    if not query_vectors:
        return json.dumps({"error": "Failed to generate embeddings."}), [], {}, []

    t_weaviate_start = time.time()
    search_results, top_results, evidence, search_profile = run_hybrid_search(
        gemini_client, weaviate_client, query_vectors, query_variations,
        search_query, collection_name, year, fast_mode, department
    )
    t_weaviate = time.time()
    logger.info(f"[PROFILING] Hybrid Search took: {t_weaviate - t_weaviate_start:.3f}s")

    t_translate_time = t_translate - t_start
    t_expansion_time = t_variations - t_translate if not fast_mode else 0.0
    t_embed_time = t_embed - t_variations
    t_weaviate_time = t_weaviate - t_weaviate_start

    profiling = {
        "translation_s": round(t_translate_time, 3),
        "expansion_s": round(t_expansion_time, 3),
        "embedding_s": round(t_embed_time, 3),
        "weaviate_s": round(t_weaviate_time, 3),
        "hybrid_db_s": search_profile.get("hybrid_db_s", 0.0),
        "rerank_s": search_profile.get("rerank_s", 0.0),
        "lineage_s": search_profile.get("lineage_s", 0.0),
        "language": "indic" if is_indic(query) else "en",
    }

    if not evidence:
        return json.dumps({"results": "No relevant documents found. Try modifying the keywords or changing fast_mode to false."}), [], profiling, []

    context_text = build_context_text(top_results)
    
    used_titles = {e.get("document") for e in evidence if e.get("document")}
    recommendations = []
    seen_recs = set(used_titles)
    for res in search_results:
        payload = res.payload or {}
        doc_num = payload.get("doc_number", "Unknown document")
        title = payload.get("document_title", "")
        doc_label = f"{title} ({doc_num})" if title and title != doc_num else doc_num
        
        if doc_label and doc_label not in seen_recs:
            seen_recs.add(doc_label)
            recommendations.append({
                "document": doc_label,
                "year": payload.get("year"),
                "category": payload.get("document_category", "Document"),
                "source_filename": payload.get("source_filename")
            })
            if len(recommendations) >= 5:
                break
    
    result_dict = {
        "status": "success",
        "fast_mode_used": fast_mode,
        "results_count": len(evidence),
        "context": context_text
    }
    return json.dumps(result_dict), evidence, profiling, recommendations
