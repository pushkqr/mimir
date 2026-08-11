"""Ingest the Higher and Technical Education department into a new collection.

Phase 3 of the build plan. mahGRs-main/GRs/Higher_and_Technical_Education_Department already
holds 4,730 orgpedia-translated .en.txt files (crawled, OCR'd and IndicTrans2-translated
upstream); there is no scraper to write here, just an ingestion run.

Ingests into a NEW collection (GovDocsV2 by default), never into the live one — rollback is
then "unset an env var" rather than "restore from a backup". Uses a separate ingestion-state
file so this run's hash tracking doesn't collide with the live corpus's.

    python -m scratch.ingest_department --measure          # 50 docs, report throughput, stop
    python -m scratch.ingest_department --full              # the rest, resumable
    python -m scratch.ingest_department --full --limit 500   # a bounded subset of the rest

Source files must already be staged (not read directly from mahGRs-main/, which is itself
gitignored and may not be present on every machine that runs this):

    mkdir -p scratch/dept_stage
    cp mahGRs-main/GRs/Higher_and_Technical_Education_Department/*.en.txt scratch/dept_stage/
"""

import argparse
import os
import pathlib
import sys
import time

from dotenv import load_dotenv

load_dotenv()

STAGE_DIR = pathlib.Path(__file__).resolve().parent / "dept_stage"
STATE_PATH = pathlib.Path(__file__).resolve().parent / "ingestion_state_dept.json"
DEFAULT_COLLECTION = "GovDocsV2"

# Set before importing anything that reads it at import or call time.
os.environ.setdefault("INGESTION_STATE_PATH", str(STATE_PATH))

from core.utils import get_genai_client, get_weaviate_client  # noqa: E402
from core.schema import ensure_collection  # noqa: E402
from ingestion.text_ingestion import run_text_ingestion  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true", help="Ingest 50 docs, report throughput, stop.")
    parser.add_argument("--full", action="store_true", help="Ingest everything not yet done (resumable).")
    parser.add_argument("--limit", type=int, default=None, help="Cap how many docs --full processes this run.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    if not args.measure and not args.full:
        parser.error("pass --measure or --full")

    if not STAGE_DIR.exists() or not any(STAGE_DIR.glob("*.en.txt")):
        print(f"No staged files found at {STAGE_DIR}. See the module docstring for the copy step.")
        sys.exit(1)

    all_files = sorted(p.name for p in STAGE_DIR.glob("*.en.txt"))
    print(f"Staged files available: {len(all_files)}")

    target_files = all_files[:50] if args.measure else (all_files[:args.limit] if args.limit else all_files)

    gemini = get_genai_client()
    weaviate_client = get_weaviate_client()
    try:
        ensure_collection(weaviate_client, args.collection)
        before = weaviate_client.collections.get(args.collection).aggregate.over_all(total_count=True).total_count

        print(f"Ingesting {len(target_files)} documents into '{args.collection}' "
              f"(state: {STATE_PATH})...")
        started = time.time()
        records = run_text_ingestion(
            gemini, weaviate_client=weaviate_client, collection_name=args.collection,
            docs_dir=str(STAGE_DIR), target_files=target_files,
        )
        elapsed = time.time() - started

        after = weaviate_client.collections.get(args.collection).aggregate.over_all(total_count=True).total_count
        docs_done = len(records) and len({r["metadata"]["source_filename"] for r in records})

        print()
        print(f"Documents processed this run : {docs_done} (of {len(target_files)} requested; "
              f"the rest may have been skipped as already-ingested per the state file)")
        print(f"Chunks added                 : {after - before}")
        print(f"Elapsed                      : {elapsed:.1f}s")
        if docs_done:
            print(f"Throughput                   : {docs_done / elapsed:.2f} docs/sec, "
                  f"{elapsed / docs_done:.2f}s/doc")
            if args.measure:
                remaining = len(all_files) - docs_done
                est_hours = (remaining * (elapsed / docs_done)) / 3600
                print(f"Extrapolated for remaining {remaining} docs: ~{est_hours:.1f} hours")
        print(f"Collection total now         : {after}")
    finally:
        weaviate_client.close()


if __name__ == "__main__":
    main()
