"""One-time backfill: tag every existing GovDocs chunk with a department.

Department-scoped access (core/schema.py's `department` property, db.py's token column,
the filter in retrieval/search.py) is new. Every chunk already in GovDocs predates it and
has no department set, which the retrieval filter would treat as a non-match - so without
this, department-scoped tokens would get zero results even for their own department's real
data. This tags the existing corpus (today, entirely Higher & Technical Education Department
documents) so it becomes visible to HTE-scoped tokens again.

Uses collection.data.update() rather than batch.add_object(): update() patches only the
properties given and leaves the stored vector untouched, where batch.add_object() replaces
the whole object and requires re-supplying the vector.

    python -m scratch.backfill_department --dry-run
    python -m scratch.backfill_department --apply
    python -m scratch.backfill_department --apply --department Finance_Department --collection GovDocs
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from core.utils import get_weaviate_client
from core.schema import ensure_department_property, DEFAULT_DEPARTMENT, DEPARTMENTS, CORPUS_COLLECTION

# Same override app.py's ACTIVE_COLLECTION uses. The live corpus is not always literally
# "GovDocs" - this deployment's .env points it at GovDocsV2 - and defaulting to the bare
# constant here would silently backfill the wrong (likely empty) collection.
_ACTIVE_COLLECTION = os.environ.get("CORPUS_COLLECTION", CORPUS_COLLECTION).strip() or CORPUS_COLLECTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually write the department tag.")
    parser.add_argument("--dry-run", action="store_true", help="Count objects missing a department, then stop.")
    parser.add_argument("--department", default=DEFAULT_DEPARTMENT, help="Department tag to apply.")
    parser.add_argument("--collection", default=_ACTIVE_COLLECTION,
                        help=f"Defaults to $CORPUS_COLLECTION if set, else '{CORPUS_COLLECTION}'.")
    args = parser.parse_args()

    if not args.apply and not args.dry_run:
        parser.error("pass --dry-run or --apply")
    if args.department not in DEPARTMENTS or args.department == "ALL":
        parser.error(f"--department must be one of: {sorted(DEPARTMENTS - {'ALL'})}")

    weaviate_client = get_weaviate_client()
    try:
        ensure_department_property(weaviate_client, args.collection)
        collection = weaviate_client.collections.get(args.collection)

        missing = 0
        tagged = 0
        total = 0
        for obj in collection.iterator(return_properties=["source_filename", "department"]):
            total += 1
            if (obj.properties or {}).get("department"):
                continue
            missing += 1
            if args.apply:
                collection.data.update(uuid=obj.uuid, properties={"department": args.department})
                tagged += 1
                if tagged % 200 == 0:
                    print(f"  ...tagged {tagged} so far")

        print(f"'{args.collection}': {total} objects, {missing} missing department.")
        if args.apply:
            print(f"Tagged {tagged} objects as '{args.department}'.")
        else:
            print(f"Dry run only - re-run with --apply --department {args.department} to write.")
    finally:
        weaviate_client.close()


if __name__ == "__main__":
    main()
