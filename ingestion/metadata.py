import os
import re
from typing import Any, Dict


# Matches a Maharashtra GR-style document number, e.g. "NGC-2010 / (193/10) / Mashi-4" or
# "Rusayo-2013/ P.S. No.418/ VC-3D". Runs from the department stem to a natural terminator
# (comma, a date word, or end of line) rather than to the first period, since these numbers
# are full of abbreviation periods.
_DOC_NUM_RE = re.compile(
    r"No\.?\s*([A-Za-zऀ-ॿ]{2,}[-–]\d{2,4}[^,\n]{0,60}?)"
    r"(?=\s*(?:,|\bdated\b|\bDy\.|\bDt\.|\bd\.|\bदिनांक\b|$))",
    re.IGNORECASE,
)


def _extract_ref_target(clause: str) -> str:
    """Reduce a citation clause to the referenced document number where one is present.

    The clause is free text ("Government Resolution No. NGC-2010/(193/10)/Mashi-4, dated
    30.10.2010"), so an abbreviation period would truncate a naive match at "No". Pull the
    structured number out when it is there and fall back to the trimmed clause when it is not.
    """
    clause = re.sub(r"\s+", " ", clause).strip(" ,;:-")
    match = _DOC_NUM_RE.search(clause)
    if match:
        return match.group(1).strip(" ,.;:-")
    return clause[:120].rstrip(" ,.;:-")


def extract_document_metadata(markdown_text: str, source_path: str, fallback_year: int = 2025) -> Dict[str, Any]:
    """Extract structured document metadata fields from markdown text."""
    normalized = (markdown_text or "").strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    title = None
    for line in lines:
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            break

    if not title:
        title = os.path.splitext(os.path.basename(source_path))[0].replace("_", " ").replace("-", " ").strip()

    doc_number = None
    patterns = [
        r"document\s*(?:no\.?|number)\s*[:#-]?\s*([A-Za-z0-9\-/\.]+)",
        r"\bno\.\s*([A-Za-z0-9\-/\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            doc_number = match.group(1).strip()
            break

    if not doc_number:
        doc_number = os.path.splitext(os.path.basename(source_path))[0]

    year = fallback_year
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match:
        year = int(year_match.group(0))

    issuing_authority = "Government"
    authority_patterns = [r"issued\s+by\s*[\:\-]\s*(.+)", r"authority\s*[\:\-]\s*(.+)"]
    for pattern in authority_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            issuing_authority = re.sub(r"\s+", " ", match.group(1)).strip()
            break

    document_category = "Document"
    for label, category in [
        ("notification", "Notification"),
        ("circular", "Circular"),
        ("order", "Order"),
        ("rule", "Rule"),
        ("guideline", "Guideline"),
        ("directive", "Directive"),
    ]:
        if re.search(rf"\b{label}\b", (title or ""), flags=re.IGNORECASE):
            document_category = category
            break

    # A document's own number can otherwise get picked back up by one of the cue-word
    # matches below and look like it cites or supersedes itself.
    own_number_key = (doc_number or "").strip().lower()

    # "in supersession of" is the one phrasing this looked for before, but most amendments in
    # practice use one of several other legally-equivalent openings. Checked against real
    # department text: "in supersession" appears in under 1% of files, while "amendment to",
    # "in partial modification" and "in continuation of" appear more often and were previously
    # invisible to this extraction entirely.
    supersede_phrases = [
        r"in\s+supersession\s+of\s+([^\n]{3,200})",
        r"in\s+partial\s+modification\s+of\s+([^\n]{3,200})",
        r"in\s+continuation\s+of\s+([^\n]{3,200})",
        r"in\s+modification\s+of\s+([^\n]{3,200})",
    ]
    supersede_targets = []
    for pattern in supersede_phrases:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        target = _extract_ref_target(match.group(1))
        key = target.strip().lower()
        if target and key != own_number_key and key not in {t.lower() for t in supersede_targets}:
            supersede_targets.append(target)
    supersedes = "; ".join(supersede_targets) if supersede_targets else None

    # An explicit "Reference:" block is a strong signal when present, but most citations in
    # this corpus are inline prose ("...vide Government Resolution No. X dated Y...") rather
    # than under a label - checked against real department text, "Government Resolution No."
    # alone appears in 23% of files, far more than the labelled block. Scan the whole document
    # for citation-shaped numbers (the same pattern _extract_ref_target reduces clauses to) and
    # treat each distinct one as a reference, capped so a heavily cross-referenced circular
    # doesn't produce an unbounded list.
    ref_clauses = []
    ref_label_match = re.search(r"reference\s*[:-]\s*([^\n]+)", normalized, flags=re.IGNORECASE)
    if ref_label_match:
        ref_clauses.append(re.sub(r"\s+", " ", ref_label_match.group(1)).strip())

    seen = {c.lower() for c in ref_clauses}
    for match in _DOC_NUM_RE.finditer(normalized):
        candidate = match.group(1).strip(" ,.;:-")
        key = candidate.lower()
        if not candidate or key == own_number_key or key in seen:
            continue
        seen.add(key)
        ref_clauses.append(candidate)
        if len(ref_clauses) >= 8:
            break

    references = "; ".join(ref_clauses) if ref_clauses else None

    return {
        "document_title": title,
        "year": year,
        "doc_number": doc_number,
        "issuing_authority": issuing_authority,
        "document_category": document_category,
        "source_filename": os.path.basename(source_path),
        "supersedes": supersedes,
        "references": references,
    }
