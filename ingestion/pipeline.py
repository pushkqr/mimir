import glob
import os
import re
from typing import Any, Dict, List, Optional

from google import genai

from ingestion.chunking import chunk_and_embed_circular
from ingestion.state import compute_file_hash, save_ingestion_state, should_skip_file
from core.log_config import get_logger
from core.schema import DEFAULT_DEPARTMENT
from ingestion.metadata import extract_document_metadata
from ingestion.parsers import parse_pdf

logger = get_logger(__name__)


def run_ingestion(
    client: genai.Client,
    weaviate_client: Optional[Any] = None,
    collection_name: str = "GovDocs",
    docs_dir: str = "docs",
    target_files: Optional[List[str]] = None,
    force_reingest: bool = False,
    department: str = DEFAULT_DEPARTMENT,
) -> List[Dict[str, Any]]:
    """Process PDF documents in target directory, upsert immediately to Weaviate per file, and save state."""
    if target_files:
        pdf_files = [os.path.join(docs_dir, f) for f in target_files]
    else:
        pdf_files = sorted(list(set(glob.glob(os.path.join(docs_dir, "*.pdf")) + glob.glob(os.path.join(docs_dir, "*.PDF")))))

    if not pdf_files:
        logger.info(f"No PDF files found in '{docs_dir}/'.")
        return []
    


    logger.info(f"Found {len(pdf_files)} PDF files in '{docs_dir}/'.")
    all_processed_records = []
    state_path = os.getenv("INGESTION_STATE_PATH", os.path.join(os.getcwd(), "scratch", "ingestion_state.json"))

    for idx, target_file in enumerate(pdf_files, 1):
        filename = os.path.basename(target_file)
        if not os.path.exists(target_file):
            continue

        file_hash = compute_file_hash(target_file)
        if not force_reingest and should_skip_file(target_file, file_hash, state_path):
            logger.info(f"[{idx}/{len(pdf_files)}] Skipping {filename} (Unchanged)")
            continue

        logger.info(f"[{idx}/{len(pdf_files)}] Processing {filename}...")
        target_md = parse_pdf(client, target_file)

        if not target_md:
            continue

        doc_year = 2025
        year_match = re.search(r"\b(19|20)\d{2}\b", target_md[:2000])
        if year_match:
            doc_year = int(year_match.group(0))

        extracted_metadata = extract_document_metadata(target_md, target_file, fallback_year=doc_year)
        global_metadata = {
            "doc_type": "PDF Document",
            "issuing_authority": extracted_metadata.get("issuing_authority", "Government"),
            "year": extracted_metadata.get("year", doc_year),
            "doc_number": extracted_metadata.get("doc_number", os.path.basename(target_file)),
            "document_title": extracted_metadata.get("document_title", os.path.splitext(os.path.basename(target_file))[0]),
            "document_category": extracted_metadata.get("document_category", "Document"),
            "source_filename": os.path.basename(target_file),
            # Lineage fields. The orgpedia path already carried these; this path dropped them,
            # so anything ingested through the admin console lost its supersedes/references
            # edges and the conflict warning in the system prompt had nothing to key on.
            "supersedes": extracted_metadata.get("supersedes"),
            "references": extracted_metadata.get("references"),
            "department": department,
        }

        try:
            processed_records = chunk_and_embed_circular(client, target_md, global_metadata)
            all_processed_records.extend(processed_records)

            if weaviate_client and processed_records:
                weaviate_collection = weaviate_client.collections.get(collection_name)
                with weaviate_collection.batch.dynamic() as batch:
                    for record in processed_records:
                        # See orgpedia_pipeline: the id is what makes a re-ingest an update
                        # rather than a duplicate.
                        batch.add_object(
                            uuid=record["id"],
                            properties=record["metadata"],
                            vector=record["vector"]["dense"]
                        )
                logger.info(f"  -> Upserted {len(processed_records)} chunks for {filename} into Weaviate.")

            save_ingestion_state(target_file, file_hash, state_path, global_metadata)
        except Exception as e:
            logger.error(f"Failed to process and embed {filename}: {e}")
            continue

    return all_processed_records
