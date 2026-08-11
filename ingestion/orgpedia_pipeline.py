import glob
import os
import re
from typing import Any, Dict, List, Optional
from google import genai
from ingestion.chunking import chunk_and_embed_circular, PartialEmbeddingError
from ingestion.state import compute_file_hash, save_ingestion_state, should_skip_file
from core.log_config import get_logger
from core.schema import DEFAULT_DEPARTMENT
from ingestion.metadata import extract_document_metadata

logger = get_logger(__name__)

def run_orgpedia_ingestion(
    client: genai.Client,
    weaviate_client: Optional[Any] = None,
    collection_name: str = "GovDocs",
    docs_dir: str = "docs/orgpedia_mahGRs",
    target_files: Optional[List[str]] = None,
    force_reingest: bool = False,
    department: str = DEFAULT_DEPARTMENT,
) -> List[Dict[str, Any]]:
    """Process pre-translated OrgPedia .en.txt files, bypassing OCR and translation."""
    if target_files:
        en_files = [os.path.join(docs_dir, f) for f in target_files]
    else:
        en_files = sorted(glob.glob(os.path.join(docs_dir, "*.en.txt")))

    if not en_files:
        logger.info(f"No OrgPedia .en.txt files found in '{docs_dir}/'.")
        return []
    


    logger.info(f"Found {len(en_files)} OrgPedia text files in '{docs_dir}/'.")
    all_processed_records = []
    state_path = os.getenv("INGESTION_STATE_PATH", os.path.join(os.getcwd(), "scratch", "ingestion_state.json"))

    for idx, target_file in enumerate(en_files, 1):
        filename = os.path.basename(target_file)
        
        file_hash = compute_file_hash(target_file)
        if not force_reingest and should_skip_file(target_file, file_hash, state_path):
            logger.info(f"[{idx}/{len(en_files)}] Skipping {filename} (Unchanged)")
            continue

        logger.info(f"[{idx}/{len(en_files)}] Processing {filename}...")
        
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                target_md = f.read()
        except Exception as e:
            logger.error(f"Failed to read {target_file}: {e}")
            continue

        if not target_md or len(target_md.strip()) < 50:
            continue

        # Basic metadata extraction
        doc_year = 2025
        # OrgPedia files are timestamped YYYYMMDD...
        timestamp_match = re.match(r"^(\d{4})", filename)
        if timestamp_match:
            doc_year = int(timestamp_match.group(1))

        extracted_metadata = extract_document_metadata(target_md, target_file, fallback_year=doc_year)
        
        # Override doc_number if it fails to extract a meaningful one, use timestamp
        doc_number = extracted_metadata.get("doc_number")
        if not doc_number or doc_number == filename:
            doc_number = filename.split(".")[0]

        global_metadata = {
            "doc_type": "OrgPedia GR",
            "issuing_authority": extracted_metadata.get("issuing_authority", "Government of Maharashtra"),
            "year": extracted_metadata.get("year", doc_year),
            "doc_number": doc_number,
            "document_title": extracted_metadata.get("document_title", f"GR {doc_number}"),
            "document_category": extracted_metadata.get("document_category", "Document"),
            "source_filename": extracted_metadata.get("source_filename", filename),
            "supersedes": extracted_metadata.get("supersedes"),
            "references": extracted_metadata.get("references"),
            "language": "en",
            "department": department,
        }

        # Chunk and embed using the English text directly.
        # save_ingestion_state stays inside the try: a document that failed partway through
        # must not be recorded as done, or the next run skips it and the gap is permanent.
        try:
            processed_records = chunk_and_embed_circular(client, target_md, global_metadata)
            all_processed_records.extend(processed_records)

            if weaviate_client and processed_records:
                weaviate_collection = weaviate_client.collections.get(collection_name)
                with weaviate_collection.batch.dynamic() as batch:
                    for record in processed_records:
                        # Passing the id makes a re-ingest replace the chunk in place.
                        # Without it Weaviate assigns a random one and a second run writes a
                        # parallel copy of the whole document.
                        batch.add_object(
                            uuid=record["id"],
                            properties=record["metadata"],
                            vector=record["vector"]["dense"]
                        )
                logger.info(f"  -> Upserted {len(processed_records)} chunks for {filename} into Weaviate.")

            save_ingestion_state(target_file, file_hash, state_path, global_metadata)
        except PartialEmbeddingError as e:
            logger.warning(f"  -> {filename} left for a later run: {e}")
        except Exception as e:
            logger.error(f"Failed to ingest {filename}: {e}")

    return all_processed_records
