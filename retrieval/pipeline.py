import os
import re
import time
import json
import itertools
import concurrent.futures
from typing import Any, Dict, List, Optional, Callable

from google import genai
from cerebras.cloud.sdk import Cerebras

from core.log_config import get_logger
from core.utils import cerebras_chat_completions_create_safe, local_generate_stream
from retrieval.search import search_policy_docs_tool, execute_search_tool
from retrieval.query import contextualize_query
import core.deployment as deployment

logger = get_logger(__name__)


class StreamingResponse:
    """Wraps answer stream to support multiple UI iteration replays."""

    def __init__(self, answer_stream):
        self._answer_stream = answer_stream
        self.captured_parts: List[str] = []
        self._exhausted = False

    def __iter__(self):
        if self._exhausted:
            for part in self.captured_parts:
                yield part
            return

        if isinstance(self._answer_stream, str):
            self.captured_parts = [self._answer_stream]
            self._exhausted = True
            yield self._answer_stream
            return

        for chunk in self._answer_stream:
            text = None
            if hasattr(chunk, "text"):
                text = chunk.text
            elif hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
            
            if text:
                self.captured_parts.append(text)
                yield text
        self._exhausted = True

    @property
    def full_text(self) -> str:
        if isinstance(self._answer_stream, str):
            return self._answer_stream
        return "".join(self.captured_parts)


_MODEL_COUNTER = 0
# Cerebras enforces 5 requests/minute PER MODEL, so the round-robin is the rate-limit
# budget: each model added multiplies available throughput. Keep this list as wide as the
# account allows.
_CEREBRAS_MODELS = [m.strip() for m in os.getenv(
    "CEREBRAS_MODELS", "gpt-oss-120b,gemma-4-31b"
).split(",") if m.strip()]

def run_retrieval(
    gemini_client: genai.Client,
    cerebras_client: Cerebras,
    weaviate_client: Optional[Any] = None,
    query: str = "",
    collection_name: str = "GovDocs",
    chat_history: Optional[List[Dict[str, str]]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    fast_mode: bool = True,
    department: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute direct Weaviate search followed by 1-shot Cerebras synthesis (Round-Robin load balanced)."""
    global _MODEL_COUNTER

    _t_request = time.time()

    if chat_history is None:
        chat_history = []
        
    if status_callback:
        status_callback("Analyzing query intent...")

    # Follow-ups like "what is the number of this GR" carry no antecedent on their own, so
    # searching them verbatim retrieves broadly. Rewrite against recent history first; the
    # helper returns the original query unchanged if rewriting fails or isn't needed.
    search_query = query
    contextualize_s = 0.0
    # Costs one extra Cerebras request against a 5/min per-model budget. Set
    # CONTEXTUALIZE_FOLLOWUPS=false to trade follow-up accuracy back for rate-limit headroom.
    _ctx_on = os.getenv("CONTEXTUALIZE_FOLLOWUPS", "true").strip().lower() in ("true", "1", "yes")
    if chat_history and _ctx_on:
        _t_ctx = time.time()
        try:
            search_query, _ = contextualize_query(gemini_client, query, chat_history, cerebras_client=cerebras_client)
            if search_query != query:
                logger.info(f"Contextualized follow-up: '{query}' -> '{search_query}'")
        except Exception as exc:
            logger.warning(f"Contextualization failed, searching raw query: {exc}")
            search_query = query
        contextualize_s = round(time.time() - _t_ctx, 3)

    if status_callback:
        status_callback("Searching knowledge base...")

    try:
        profiling_metrics = {}
        recommendations = []

        # Extract year filter from the contextualized query for fast_mode (e.g. "GRs from 2018...")
        _year_match = re.search(r'\b(19|20)\d{2}\b', search_query)
        _extracted_year = int(_year_match.group(0)) if _year_match else None

        def search_tool_wrapper(query: str, year: Optional[int] = None, fast_mode: bool = False) -> str:
            # department comes from the authenticated token via run_retrieval's own argument,
            # never from the LLM - the tool schema in search.py deliberately has no such
            # parameter, so the model has no path to widen its own access.
            logger.info(f"LLM called search_tool(query='{query}', year={year}, fast_mode={fast_mode})")
            results, ev, prof, recs = execute_search_tool(
                gemini_client, weaviate_client, collection_name, query, year, fast_mode, department
            )
            evidence.extend(ev)
            profiling_metrics.update(prof)
            recommendations.extend(recs)
            return results

        evidence = []
        search_json = search_tool_wrapper(search_query, year=_extracted_year, fast_mode=fast_mode)
    except Exception as exc:
        logger.error(f"Search tool failed: {exc}")
        return {
            "status": "error",
            "response_text": f"Search failed: {exc}",
            "answer_stream": StreamingResponse(f"Search failed: {exc}"),
            "evidence": [],
        }

    search_context = ""
    try:
        parsed_search = json.loads(search_json) if isinstance(search_json, str) else search_json
        if isinstance(parsed_search, dict):
            search_context = (parsed_search.get("context") or "").strip()
    except Exception as exc:
        logger.warning(f"Could not parse search context: {exc}")

    if search_context:
        context_text = f"Retrieved Evidence:\n{search_context}\n"
    else:
        context_text = "Retrieved Evidence:\n"
        for idx, doc in enumerate(evidence):
            context_text += f"Document: {doc.get('document')} Section: {doc.get('section')}\nQuote: {doc.get('quote')}\n\n"

    # Everything upstream of generation (embeddings, reranking, translation, the vector
    # store) is already self-hosted. Generation is the one hop to a third party, so it is
    # the one that has to be swappable for an air-gapped deployment. GEN_PROVIDER=local
    # points it at any OpenAI-compatible server on the department's own hardware; setting
    # DEPLOYMENT_MODE=sovereign instead flips this and the ingestion-time switches together.
    _local_gen = deployment.gen_provider() == "local"

    if _local_gen:
        target_model = os.getenv("LOCAL_GEN_MODEL", "qwen3:4b")
    else:
        _MODEL_COUNTER += 1
        target_model = _CEREBRAS_MODELS[(_MODEL_COUNTER - 1) % len(_CEREBRAS_MODELS)]
        logger.info(f"Selected model {target_model} for request #{_MODEL_COUNTER}")

    if status_callback:
        status_callback(f"Synthesizing answer using {target_model}...")

    # The full prompt below is roughly 1,850 tokens and is sent with every request. A hosted
    # endpoint reads it in well under a second; a CPU node processes about 17 tokens per
    # second, so the instructions alone cost close to two minutes before the model reaches
    # the evidence, and a default context window then truncates that evidence away.
    #
    # This condensed version keeps every rule that changes what the answer says: ground
    # strictly in context, cite, refuse when unsupported, and the conflict callout, which is
    # the behaviour the system exists to demonstrate. What it drops is elaboration on tone
    # and formatting, which costs polish rather than correctness. It is used only when
    # generation runs locally, so the hosted path is unaffected.
    _LOCAL_SYSTEM_PROMPT = (
        "You are Mimir, a government policy assistant. Answer ONLY from the provided context. "
        "Never use outside knowledge and never invent a document name, number, or section.\n\n"
        "- Lead with the direct answer. Cite the document and section for every claim.\n"
        "- If the context does not answer the question, say so plainly in one sentence and stop.\n"
        "- If it answers only part, answer that part and say what is unsupported.\n"
        "- Preserve exact numbers, dates, GR numbers and thresholds. Never round or paraphrase them.\n"
        "- Reply in the same language as the question.\n\n"
        "If two documents give different values for the same provision, or a block is tagged "
        "[Supersedes: X], you MUST begin with exactly:\n\n"
        "> [!WARNING]\n"
        "> **<the discrepancy in one line>**\n"
        "> - **<Document A> (<year>)**: <value A>\n"
        "> - **<Document B> (<year>)**: <value B>\n"
        "> <which is operative, and why>\n\n"
        "Then answer below it. The later document, or the one carrying the [Supersedes] tag, is "
        "operative unless the context says otherwise. Never silently pick one value and drop the "
        "other: an officer acting on a superseded figure is the failure this exists to prevent.\n\n"
        "Be concise. Prefer short bullets over paragraphs."
    )

    system_prompt = (
        "You are Mimir, an elite Government Policy AI Assistant serving as an instant, reliable decision-support engine for government officials, administrators, and policy experts. Your answers may inform real bureaucratic or legal decisions, so precision and fact-grounding take absolute priority over completeness or fluency.\n\n"
        "## Input Format\n"
        "You will receive a user question along with pre-retrieved context, injected as `Context: [Retrieved Evidence...]`. This context has already been pulled from this deployment's indexed policy corpus by an upstream retrieval system. You do not have a search tool and cannot request additional retrieval — you must work entirely from what is given to you in this single pass.\n\n"
        "## Core Directive: Strict Fact-Grounding\n"
        "- Answer using ONLY the information present in the provided context. Never supplement with outside knowledge, training data, or general assumptions about government policy, even if you believe you know the answer.\n"
        "- If the user's query is highly ambiguous (e.g., a single word like 'fees?' or 'leave?'), provide a brief summary of the various contexts found in the retrieved documents, and then explicitly ask the user to clarify.\n"
        "- If the user uses a demonstrative reference such as 'this GR', 'this circular', 'this document', 'हा शासन निर्णय', or 'हे परिपत्रक' without naming a specific document, treat it as a reference to the single most relevant document in the retrieved context (i.e., the top result). Answer definitively about that document and state its name or number clearly at the start of your answer.\n"
        "- If the context fully answers the question, provide a complete, definitive answer.\n"
        "- If the context partially answers the question, answer only the part that is supported, and explicitly flag what remains unaddressed.\n"
        "- If the user asks which document they should refer to, synthesize a list of ALL highly relevant documents present in the context and briefly summarize what each provides, rather than just picking one.\n"
        "- If the context does not contain the answer, state plainly that the information is not available in the retrieved documents. Do not guess, infer beyond what's written, or pad the response with plausible-sounding filler. A clear \"not found\" is more valuable to a government official than a confident hallucination.\n"
        "- Never fabricate a document name, section number, or citation. Cite only what actually appears in the provided context.\n"
        "- Be exceptionally thorough. Do not omit mentions of specific regulations (such as UGC Regulations), criteria, or governing bodies if they are present in the context.\n\n"
        "## Citation Requirements\n"
        "- The document names provided in the context (e.g., 'MAHENG/2009/35528', 'Manyata-2023...', or Roman numerals) may be internal administrative codes rather than full descriptive titles. Do not refuse to answer simply because these codes don't perfectly match the human-readable name of an Act or Resolution in the user's query. If the text of the context contains the answer, provide the answer and cite the administrative code.\n"
        "- Every factual claim must be tied to its source using the document name and section/clause as given in the context (e.g., \"According to GR-Unaided-30-June-2023, Section 2...\").\n"
        "- If multiple documents are relevant, cite each distinctly rather than blending them into an unattributed summary.\n"
        "- If a section or document name isn't clearly identifiable in the context, cite what identifying information is available (e.g., document title alone) rather than omitting citation entirely.\n\n"
        "## Conflicting and Superseded Provisions\n"
        "This section overrides formatting preferences elsewhere. Apply it whenever EITHER of the following is true:\n"
        "(a) two or more documents in the context state different values for the same provision (different amounts, ages, dates, percentages, deadlines, or eligibility thresholds), or\n"
        "(b) any block in the context carries a `[Supersedes: X]` tag.\n\n"
        "When either holds, your answer MUST begin with a callout block in exactly this form, before any other text:\n\n"
        "> [!WARNING]\n"
        "> **<one-line statement of the discrepancy>**\n"
        "> - **<Document A> (<year>)**: <value A>\n"
        "> - **<Document B> (<year>)**: <value B>\n"
        "> <which one is currently operative, and why>\n\n"
        "Then give the operative answer below the block. Rules for this case:\n"
        "- Never silently pick one value and omit the other. An officer acting on a superseded figure is the specific failure this system exists to prevent.\n"
        "- The later document, or the one carrying the `[Supersedes: X]` tag, is the operative one unless the context says otherwise. State that explicitly.\n"
        "- If a document specifies transitional provisions (for example that processes started before a date remain under the old rule), state them, since they determine which value actually applies to a given case.\n"
        "- If you cannot determine which is operative from the context, say so plainly rather than guessing.\n\n"
        "## Formatting\n"
        "- Lead with the direct answer, not a preamble.\n"
        "- Use bullet points or numbered lists when the answer involves multiple provisions, conditions, steps, or eligibility criteria — bureaucratic content is often enumerable and should be presented that way.\n"
        "- Use short paragraphs for narrative or explanatory context where a list would be unnatural.\n"
        "- Bold key terms, figures, deadlines, or conditions when it aids fast scanning by a busy official.\n"
        "- Keep structure clean and scannable; avoid dense unbroken text blocks.\n\n"
        "## Tone\n"
        "- Professional, objective, and definitive, as befits an assistant used for decisions with real consequences. This doesn't mean cold — you can acknowledge the user's question naturally and respond in clear, direct prose, but every claim still traces back to the provided context.\n"
        "- No hedging language (\"it seems,\" \"possibly,\" \"I think\") unless the context itself is ambiguous or conflicting — in which case state clearly that the ambiguity exists rather than hedging vaguely.\n"
        "- Skip empty filler (\"Great question!\", \"I'd be happy to help\") — get to the substance quickly — but a brief, natural framing sentence before the answer is fine if it helps orient the reader, especially for complex multi-part questions.\n"
        "- When information is unavailable, state this in one direct sentence and stop — do not apologize at length or speculate about where the answer might be found elsewhere.\n\n"
        "## Boundaries\n"
        "- You MUST respond in the exact language the user used in their Latest Question (e.g., if asked in Marathi, reply entirely in Marathi). Strictly preserve official legal terminology (e.g., 'Competent Authority', 'Unaided') as used in the source documents or translate them to their exact official equivalents.\n"
        "- If the user asks for a summary of a specific document, synthesize the key points from all retrieved chunks belonging to that document.\n"
        "- Do not offer legal advice or personal recommendations on how an official should act; present the facts and provisions as documented, and let the reader apply them.\n"
        "- Do not summarize or paraphrase away specific numbers, dates, eligibility thresholds, or procedural steps — these are often the exact details a government decision depends on. Preserve precision over brevity.\n"
        "- If the question itself is ambiguous even with context available (e.g., could refer to multiple distinct policies), note the ambiguity and answer for the most likely interpretation(s) based on what the context actually contains, rather than refusing to answer."
    )
    
    if _local_gen:
        system_prompt = _LOCAL_SYSTEM_PROMPT

    history_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['text']}\n"

    if history_text:
        user_prompt = f"Chat History:\n{history_text}\n\nContext:\n{context_text}\n\nLatest Question: {query}"
    else:
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    try:
        if _local_gen:
            _local_stream = local_generate_stream(system_prompt, user_prompt)
            # Pull the first chunk eagerly so a refused connection or an unknown model name
            # raises here and is reported as a failure, instead of surfacing as a silently
            # empty answer once the UI is already iterating the generator.
            _first = next(_local_stream, None)
            stream = itertools.chain([_first], _local_stream) if _first is not None else iter(())
        else:
            # Cerebras limits are per-minute, so retrying a 429 a few seconds later just fails
            # again while burning the latency budget. Disable SDK retries and surface the
            # failure immediately instead.
            _gen_client = cerebras_client
            try:
                _gen_client = cerebras_client.with_options(max_retries=0)
            except Exception:
                pass
            stream = cerebras_chat_completions_create_safe(
                _gen_client,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=target_model,
                stream=True,
                max_completion_tokens=8192,
                temperature=0.0
            )
    except Exception as exc:
        # Generation fails closed, in both modes.
        #
        # This used to fall back to gemini-2.5-flash. In sovereign mode that silently undid the
        # only guarantee the mode exists to make: the officer's question and the retrieved
        # document text left the network, and the answer came back looking normal, so nobody
        # could tell it had happened. A guarantee that quietly degrades is not a guarantee.
        #
        # It is also the last proprietary model that could produce an answer, so removing it is
        # what makes "every model that can answer is open weight" true rather than nearly true.
        #
        # The cost is honest: a Cerebras rate-limit now surfaces as an error instead of a slower
        # answer from elsewhere. Per-minute limits mean an immediate retry fails again anyway,
        # so telling the officer to try shortly beats spending their latency budget pretending.
        _which = "Local generation" if _local_gen else "Cerebras LLM"
        logger.error(f"{_which} failed ({exc}). Failing closed, no third-party fallback.")
        if _local_gen:
            officer_message = (
                "The answer service on this network is not responding, so no answer was "
                "generated. Nothing was sent outside the network. Please try again shortly."
            )
        else:
            officer_message = (
                "The answer service is temporarily unavailable, so no answer was generated. "
                "Please try again in a minute."
            )
        return {
            "status": "error",
            "response_text": f"{_which} failed: {exc}",
            "answer_stream": StreamingResponse(officer_message),
            "evidence": evidence,
        }

    # Dedup evidence for frontend
    unique_evidence = []
    seen = set()
    for e in evidence:
        k = e.get("quote", "")
        if k not in seen:
            seen.add(k)
            unique_evidence.append(e)

    # Dedup recommendations
    unique_recommendations = []
    seen_recs = set()
    for r in recommendations:
        k = r.get("document", "")
        if k not in seen_recs:
            seen_recs.add(k)
            unique_recommendations.append(r)

    profiling_metrics["contextualize_s"] = contextualize_s
    profiling_metrics["retrieval_s"] = round(time.time() - _t_request, 3)
    profiling_metrics["model"] = target_model

    return {
        "status": "success",
        "response_text": "", # Wil be populated dynamically by app.py if needed
        "answer_stream": StreamingResponse(stream),
        "evidence": unique_evidence,
        "recommendations": unique_recommendations,
        "metrics": profiling_metrics,
    }

