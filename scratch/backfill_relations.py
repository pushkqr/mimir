"""One-time backfill: re-run the improved supersedes/references extraction against the
already-ingested corpus and patch the stored fields.

ingestion/metadata.py's extraction only runs at ingest time. Broadening its patterns (this
session) has zero effect on documents already in Weaviate until this is run - the citation
graph reads whatever is already stored, it does not re-derive it from source text. Without
this, the live corpus stays on the old, narrower extraction forever while every new
department ingested from here on gets the better one - an inconsistency with no reason to
exist, since the fix is a pure improvement.

Uses collection.data.update() per chunk, same low-risk pattern as scratch/backfill_department
.py: it patches only the properties given and leaves the stored vector untouched, unlike
collection.batch.dynamic()'s add_object() which replaces the whole object and needs the
vector re-supplied.

Only documents whose source .en.txt can still be found on disk are touched - the two curated
DEMO-*.pdf documents are ingested via the PDF path (ingestion/pipeline.py) from raw PDFs, not
this orgpedia .en.txt path, and are deliberately left alone here (their relation was already
verified via the live retrieval test, not through this script).

    python -m scratch.backfill_relations --dry-run
    python -m scratch.backfill_relations --apply
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from core.utils import get_weaviate_client
from core.schema import CORPUS_COLLECTION
from ingestion.metadata import extract_document_metadata

_ACTIVE_COLLECTION = os.environ.get("CORPUS_COLLECTION", CORPUS_COLLECTION).strip() or CORPUS_COLLECTION
SOURCE_DIR = os.path.join("docs", "Higher_and_Technical_Education_Department")


def _year_from_filename(filename: str) -> int:
    prefix = filename[:4]
    return int(prefix) if prefix.isdigit() else 2025


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually patch changed documents.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change, then stop.")
    parser.add_argument("--collection", default=_ACTIVE_COLLECTION)
    parser.add_argument("--source-dir", default=SOURCE_DIR)
    parser.add_argument("--show", type=int, default=8, help="How many before/after diffs to print.")
    args = parser.parse_args()

    if not args.apply and not args.dry_run:
        parser.error("pass --dry-run or --apply")

    weaviate_client = get_weaviate_client()
    try:
        collection = weaviate_client.collections.get(args.collection)

        # One pass to group every chunk's uuid under its document, and capture each
        # document's current stored values (identical across all its chunks - stamped from
        # the same global_metadata dict at ingest time, per ingestion/orgpedia_pipeline.py).
        by_doc = {}
        for obj in collection.iterator(return_properties=["source_filename", "supersedes", "references"]):
            props = obj.properties or {}
            filename = props.get("source_filename")
            if not filename:
                continue
            doc = by_doc.setdefault(filename, {"uuids": [], "old_supersedes": props.get("supersedes"),
                                                "old_references": props.get("references")})
            doc["uuids"].append(obj.uuid)

        print(f"'{args.collection}': {len(by_doc)} distinct documents, "
              f"{sum(len(d['uuids']) for d in by_doc.values())} chunks total.")

        missing_source = 0
        changed = 0
        chunks_patched = 0
        shown = 0

        for filename, doc in by_doc.items():
            source_path = os.path.join(args.source_dir, filename)
            if not os.path.isfile(source_path):
                missing_source += 1
                continue
            try:
                with open(source_path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except Exception as exc:
                print(f"  could not read {source_path}: {exc}")
                continue

            new_metadata = extract_document_metadata(text, source_path, fallback_year=_year_from_filename(filename))
            new_supersedes = new_metadata["supersedes"]
            new_references = new_metadata["references"]

            if new_supersedes == doc["old_supersedes"] and new_references == doc["old_references"]:
                continue

            changed += 1
            if shown < args.show:
                print(f"\n  {filename}")
                print(f"    supersedes : {doc['old_supersedes']!r} -> {new_supersedes!r}")
                print(f"    references : {doc['old_references']!r} -> {new_references!r}")
                shown += 1

            if args.apply:
                for uuid in doc["uuids"]:
                    collection.data.update(
                        uuid=uuid,
                        properties={"supersedes": new_supersedes, "references": new_references},
                    )
                    chunks_patched += 1

        print(f"\n{changed} of {len(by_doc)} documents would change "
              f"({missing_source} had no source file under {args.source_dir}).")
        if args.apply:
            print(f"Patched {chunks_patched} chunks across {changed} documents.")
        else:
            print("Dry run only - re-run with --apply to write.")
    finally:
        weaviate_client.close()


if __name__ == "__main__":
    main()
