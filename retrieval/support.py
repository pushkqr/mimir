import os
import re
from datetime import datetime
from typing import Any, Dict, List

import core.deployment as deployment


# A Maharashtra GR number reduces to a department stem plus a year:
# "NGC-2010/(193/10)/Mashi-4" -> (NGC, 2010). A value that cannot produce that pair is not a
# GR number; in this corpus it is usually a letterhead or post-box number ("647051", "2023/").
_STEM_RE = re.compile(r"([A-Za-zऀ-ॿ]{3,})[\s-]*[-–]\s*(\d{4})")

# Titles come from the first markdown heading, which for OCR'd scans is very often the page
# marker rather than the subject line, a generic section header ("Preamble:"), or a heading
# truncated mid-word. A short fragment ending in a colon or hyphen is never a real GR title.
_JUNK_TITLE_RE = re.compile(r"^\s*(page\s*\d+|#+\s*)?\s*$|^.{2,24}[:\-]\s*$", re.IGNORECASE)


def _issue_date_from_filename(filename: str) -> str:
    """Orgpedia filenames begin with the issue date: 201903251256356710.pdf.en.txt."""
    stem = os.path.basename(str(filename or ""))
    if len(stem) < 8 or not stem[:8].isdigit():
        return ""
    try:
        return datetime.strptime(stem[:8], "%Y%m%d").strftime("%d %B %Y")
    except ValueError:
        return ""


def document_label(props: Dict[str, Any], filename: str = "") -> str:
    """Human-readable name for a document, for citations and for the prompt's context headers.

    Every call site used to build this as f"{title} ({doc_number})" with no guard, which is how
    officers were shown citations reading "Page 1 (647051)" — 90% of this corpus has a title of
    "Page N" (the extractor takes orgpedia's page marker as the heading) and most doc_numbers
    are letterhead numbers rather than GR numbers. The same string is fed to the model, so it
    also wrote those labels into its own answers.

    Falls back in order of how much an officer can actually do with it: a real subject line, a
    real GR number, the issue date, and only then whatever is left.
    """
    title = str(props.get("document_title") or "").strip()
    if title and len(title) > 6 and not _JUNK_TITLE_RE.match(title):
        return title

    number = str(props.get("doc_number") or "").strip()
    if number and _STEM_RE.search(number):
        return number

    source = filename or props.get("source_filename") or ""
    issued = _issue_date_from_filename(source)
    if issued:
        return f"GR dated {issued}"

    # A doc_number that carries no citation key is still better than a raw filename.
    if number and len(number) > 3:
        return number
    return str(source) or "Unknown document"


def extract_response_text(response: Any) -> str:
    """Safely extract text from either a model response object or a plain string."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _is_mostly_indic(text: str) -> bool:
    """Return True if more than 30% of characters are Devanagari (Marathi/Hindi source)."""
    if not text:
        return False
    indic = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return indic / max(len(text), 1) > 0.3


def _block_char_cap() -> int:
    """Per-block context cap, applied only when generation runs on local CPU.

    Prompt processing on the self-hosted node runs at 40-53 tok/s and the rate decays as the
    window grows (measured: 1024 tokens in 19s, 4971 in 135s), so context length is the
    dominant term in sovereign latency, not thread count or the serving stack.

    The cap is per block rather than on the joined string because trimming the tail would
    drop whole documents, and a contradiction between two documents is only detectable when
    both are still present - which is the behaviour the conflict callout exists to show.

    Unset means unlimited, so the default is byte-for-byte the current output and the hosted
    path never reads this at all.
    """
    if deployment.gen_provider() != "local":
        return 0
    raw = os.getenv("SOVEREIGN_CONTEXT_BLOCK_CHARS", "").strip()
    return int(raw) if raw.isdigit() else 0


def build_context_text(top_results: List[Any]) -> str:
    """Stitch unique parent contexts and child passages into a unified context block.

    Each block is prefixed with a [Document: ... | Section: ...] header so the LLM
    can cite sources precisely. For Marathi-source chunks, English translations are
    preferred over raw Devanagari to reduce LLM reasoning overhead on English queries.
    """
    parent_blocks = {}
    standalone_chunks = []
    char_cap = _block_char_cap()

    for result in top_results:
        payload = getattr(result, "payload", None) or {}
        parent_id = payload.get("parent_id")
        parent_ctx = (payload.get("parent_context") or "").strip()
        child_txt = (payload.get("child_text") or "").strip()
        translated = (payload.get("translated_text") or "").strip()

        if parent_id:
            if parent_id not in parent_blocks:
                parent_blocks[parent_id] = {
                    "doc_number": payload.get("doc_number", "Document"),
                    "document_title": payload.get("document_title", ""),
                    "section_title": payload.get("section_title", ""),
                    "source_filename": payload.get("source_filename", ""),
                    "context": parent_ctx if parent_ctx else child_txt,
                    "children": [child_txt] if child_txt else [],
                    "translated_texts": [translated] if translated else [],
                    "supersedes": payload.get("supersedes"),
                    "references": payload.get("references"),
                }
            else:
                if child_txt and child_txt not in parent_blocks[parent_id]["children"]:
                    parent_blocks[parent_id]["children"].append(child_txt)
                if translated and translated not in parent_blocks[parent_id]["translated_texts"]:
                    parent_blocks[parent_id]["translated_texts"].append(translated)
        elif child_txt:
            standalone_chunks.append({
                "doc_number": payload.get("doc_number", "Document"),
                "document_title": payload.get("document_title", ""),
                "section_title": payload.get("section_title", ""),
                "source_filename": payload.get("source_filename", ""),
                "text": child_txt,
                "translated": translated,
                "supersedes": payload.get("supersedes"),
                "references": payload.get("references"),
            })

    def _make_header(block: Dict[str, Any]) -> str:
        header = f"[Document: {document_label(block, block.get('source_filename', ''))}"
        section_title = block.get("section_title", "")
        if section_title:
            header += f" | Section: {section_title}"
        return header + "]"

    formatted_sections = []

    for pid, block in parent_blocks.items():
        ctx = block["context"]
        if not ctx:
            continue

        # For Marathi-source blocks, prefer joined English translations to reduce
        # LLM reasoning overhead when answering English queries.
        translated_texts = block.get("translated_texts", [])
        if translated_texts and _is_mostly_indic(ctx):
            ctx = "\n\n".join(t for t in translated_texts if t)

        if not ctx:
            continue

        header = _make_header(block)
        meta = header + "\n"
        if block.get("supersedes"):
            meta += f"[Supersedes: {block['supersedes']}]\n"
        if block.get("references"):
            meta += f"[References: {block['references']}]\n"
        formatted_sections.append(meta + (ctx[:char_cap] if char_cap else ctx))

    seen_texts = {block["context"] for block in parent_blocks.values() if block.get("context")}
    for chunk in standalone_chunks:
        txt = chunk["text"]
        translated = chunk.get("translated", "")
        # Prefer English translation for Marathi-source standalone chunks
        display_txt = translated if translated and _is_mostly_indic(txt) else txt
        if not display_txt or display_txt in seen_texts:
            continue
        header = _make_header(chunk)
        meta = header + "\n"
        if chunk.get("supersedes"):
            meta += f"[Supersedes: {chunk['supersedes']}]\n"
        if chunk.get("references"):
            meta += f"[References: {chunk['references']}]\n"
        formatted_sections.append(meta + (display_txt[:char_cap] if char_cap else display_txt))
        seen_texts.add(display_txt)

    return "\n\n---\n\n".join(s for s in formatted_sections if s.strip())
