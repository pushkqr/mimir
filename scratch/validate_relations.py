"""Read-only check: does the supersedes/references extraction actually catch relations on
real department text, before spending hours ingesting it?

ingestion/metadata.py's extraction is pure regex, no LLM. Investigation (documented in
ingestion/metadata.py itself) found the original patterns ("in supersession of", "reference:")
matched under 1% of real department files, while other common phrasings (amendment/
modification language, inline GR-number citations) went completely uncaptured. This reports
the hit rate directly against real .en.txt source files, department by department, so a
department can be checked before it's ingested rather than after.

Touches nothing: no Weaviate connection, no API calls, just extract_document_metadata()
against files already on disk under docs/<department>/*.en.txt.

    python -m scratch.validate_relations
    python -m scratch.validate_relations --department Finance_Department --limit 200
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

from ingestion.metadata import extract_document_metadata

# Higher_and_Technical_Education_Department is the live corpus - included by default as a
# known-good sanity check (its DEMO-2019/DEMO-2022 age-limit pair should still surface a
# supersedes relation). The rest are new departments sampled during investigation.
DEFAULT_DEPARTMENTS = [
    "Higher_and_Technical_Education_Department",
    "Finance_Department",
    "Public_Health_Department",
    "Revenue_and_Forest_Department",
]


def _year_from_filename(filename: str) -> int:
    """OrgPedia files are timestamped YYYYMMDD... - matches ingestion/orgpedia_pipeline.py."""
    prefix = filename[:4]
    return int(prefix) if prefix.isdigit() else 2025


def check_department(docs_dir: str, department: str, limit: int = None) -> dict:
    files = sorted(glob.glob(os.path.join(docs_dir, department, "*.en.txt")))
    if limit:
        files = files[:limit]

    with_supersedes = 0
    with_references = 0
    reference_counts = []
    checked = 0

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception:
            continue
        checked += 1
        metadata = extract_document_metadata(
            text, path, fallback_year=_year_from_filename(os.path.basename(path))
        )
        if metadata["supersedes"]:
            with_supersedes += 1
        if metadata["references"]:
            with_references += 1
            reference_counts.append(len(metadata["references"].split("; ")))

    avg_refs = sum(reference_counts) / len(reference_counts) if reference_counts else 0.0
    return {
        "department": department,
        "total": checked,
        "with_supersedes": with_supersedes,
        "with_references": with_references,
        "avg_references_per_doc_with_any": round(avg_refs, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--department", action="append",
                        help="Department folder name under docs/. Repeatable. Defaults to a fixed sample.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--limit", type=int, default=None, help="Cap files checked per department.")
    args = parser.parse_args()

    departments = args.department or DEFAULT_DEPARTMENTS

    print(f"{'department':45} {'files':>7} {'supersedes':>11} {'references':>11} {'avg refs/doc':>13}")
    for dept in departments:
        dept_dir = os.path.join(args.docs_dir, dept)
        if not os.path.isdir(dept_dir):
            print(f"{dept:45} (not found at {dept_dir})")
            continue
        stats = check_department(args.docs_dir, dept, args.limit)
        if stats["total"] == 0:
            print(f"{dept:45} (no .en.txt files found)")
            continue
        sup_pct = 100 * stats["with_supersedes"] / stats["total"]
        ref_pct = 100 * stats["with_references"] / stats["total"]
        print(f"{dept:45} {stats['total']:>7} {sup_pct:>10.1f}% {ref_pct:>10.1f}% "
              f"{stats['avg_references_per_doc_with_any']:>13}")


if __name__ == "__main__":
    main()
