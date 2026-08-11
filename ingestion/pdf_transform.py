"""PDF pre-transform: raw PDFs → .en.txt via the transform microservice.

Batch entry point
    run_pdf_transform("docs/raw", "docs/parsed")

Single-file entry point (used by admin console)
    transform_single_pdf("/path/to.pdf", "docs/parsed")
"""

import glob
import os
import requests
from pathlib import Path
from typing import List, Optional

from core.log_config import get_logger
from ingestion.state import compute_file_hash, save_ingestion_state, should_skip_file

logger = get_logger(__name__)

_DEFAULT_TRANSFORM_STATE = os.path.join("scratch", "transform_state.json")

# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def transform_single_pdf(pdf_path: str, output_dir: str) -> Optional[str]:
    """Transform one PDF by calling the transform microservice.

    Returns the absolute path to the produced .en.txt file, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.basename(pdf_path)
    
    # Get the transform service URL from environment
    service_url = os.getenv("TRANSFORM_SERVICE_URL", "http://127.0.0.1:8003/transform")
    
    logger.info(f"Sending {pdf_name} to transform service at {service_url}")
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_name, f, "application/pdf")}
            response = requests.post(service_url, files=files, timeout=300)
            
        if response.status_code == 200:
            en_txt_content = response.text
            
            # Use the .en.txt naming convention expected downstream
            if pdf_name.lower().endswith(".pdf"):
                base_name = pdf_name[:-4]
            else:
                base_name = pdf_name
                
            # The downstream pipeline expects name.pdf.en.txt or name.en.txt.
            # We match what the daemon previously produced: pdf_name + ".en.txt"
            en_txt_dst = os.path.join(output_dir, pdf_name + ".en.txt")
            
            with open(en_txt_dst, "w", encoding="utf-8") as out_f:
                out_f.write(en_txt_content)
                
            return os.path.abspath(en_txt_dst)
        else:
            logger.error(f"Transform service returned error: {response.status_code} - {response.text}")
            return None
    except Exception as exc:
        logger.error(f"Failed to call transform service: {exc}")
        return None


def run_pdf_transform(
    raw_dir: str = "docs/raw",
    output_dir: str = "docs/parsed",
    force: bool = False,
) -> List[str]:
    """Batch-transform all new PDFs in *raw_dir* into .en.txt files in *output_dir*.

    Hash-tracks each source PDF so unchanged files are skipped on re-runs.
    """
    pdf_files = sorted(
        set(glob.glob(os.path.join(raw_dir, "*.pdf")))
        | set(glob.glob(os.path.join(raw_dir, "*.PDF")))
    )

    if not pdf_files:
        logger.info(f"No PDFs found in '{raw_dir}/'.")
        return []

    logger.info(f"Found {len(pdf_files)} PDFs in '{raw_dir}/'.")
    os.makedirs(output_dir, exist_ok=True)

    state_path = os.getenv("TRANSFORM_STATE_PATH", _DEFAULT_TRANSFORM_STATE)
    produced: List[str] = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)

        file_hash = compute_file_hash(pdf_path)
        if not force and should_skip_file(pdf_path, file_hash, state_path):
            logger.info(f"[{idx}/{len(pdf_files)}] Skipping {filename} (already transformed)")
            continue

        logger.info(f"[{idx}/{len(pdf_files)}] Transforming {filename} …")
        en_txt_path = transform_single_pdf(pdf_path, output_dir)

        if en_txt_path:
            save_ingestion_state(pdf_path, file_hash, state_path)
            produced.append(en_txt_path)
            logger.info(f"  → Produced {os.path.basename(en_txt_path)}")
        else:
            logger.error(f"  → Failed to transform {filename}")

    logger.info(
        f"Transform complete: {len(produced)} new .en.txt files "
        f"({len(pdf_files) - len(produced)} skipped)."
    )
    return produced
