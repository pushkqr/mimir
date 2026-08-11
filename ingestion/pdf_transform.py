"""PDF pre-transform: raw PDFs → .en.txt via a warm daemon subprocess.

The daemon loads the docint pipeline (including the 200M IndicTrans2 model)
once and keeps it resident.  Subsequent PDFs are processed without model reload,
which is critical for the admin-console demo flow where single PDFs are uploaded
one at a time.

Batch entry point
    run_pdf_transform("docs/raw", "docs/parsed")

Single-file entry point (used by admin console)
    transform_single_pdf("/path/to.pdf", "docs/parsed")
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

from core.log_config import get_logger
from ingestion.state import compute_file_hash, save_ingestion_state, should_skip_file

logger = get_logger(__name__)

# Vendored pipeline files live next to this module.
_PIPELINE_FILES = Path(__file__).resolve().parent / "transform_pipeline"
_DEFAULT_TRANSFORM_STATE = os.path.join("scratch", "transform_state.json")


# ---------------------------------------------------------------------------
#  Warm daemon
# ---------------------------------------------------------------------------

class _TransformDaemon:
    """Manages a long-lived subprocess that keeps the docint pipeline loaded."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._workdir: Optional[Path] = None

    # -- working directory ---------------------------------------------------

    def _setup_workdir(self) -> Path:
        """Create a persistent runtime directory with the layout docint expects.

        The vendored files are copied (not symlinked) so the layout is
        self-contained and works on every OS without elevated privileges.
        """
        workdir = Path(os.environ.get("TRANSFORM_WORKDIR", "data/transform_work"))
        workdir.mkdir(parents=True, exist_ok=True)

        (workdir / "input").mkdir(exist_ok=True)
        (workdir / "output").mkdir(exist_ok=True)

        # src/writeTxt.yml
        src_dir = workdir / "src"
        src_dir.mkdir(exist_ok=True)
        _copy_if_missing(_PIPELINE_FILES / "src" / "writeTxt.yml", src_dir / "writeTxt.yml")

        # conf/ (glossary + cmaps)
        conf_dst = workdir / "conf"
        if not conf_dst.exists():
            shutil.copytree(str(_PIPELINE_FILES / "conf"), str(conf_dst))

        # word_recognizer.py (must be importable from CWD)
        _copy_if_missing(_PIPELINE_FILES / "word_recognizer.py", workdir / "word_recognizer.py")

        # worker.py (the daemon entry-point)
        _copy_if_missing(_PIPELINE_FILES / "worker.py", workdir / "worker.py")

        self._workdir = workdir
        return workdir

    # -- lifecycle -----------------------------------------------------------

    def _ensure_running(self):
        """Start (or restart) the daemon subprocess if it is not alive."""
        if self._proc is not None and self._proc.poll() is None:
            return  # still alive

        if self._workdir is None:
            self._setup_workdir()

        logger.info("Starting transform daemon (loading pipeline + 200M model) …")

        env = {**os.environ, "PYTHONPATH": str(self._workdir)}
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "worker.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self._workdir),
            text=True,
            env=env,
        )

        # Block until the worker signals readiness (model loaded).
        raw = self._proc.stdout.readline().strip()
        if not raw:
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                f"Transform daemon exited immediately.  stderr:\n{stderr[:2000]}"
            )

        ready = json.loads(raw)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Transform daemon sent unexpected ready signal: {raw}")

        logger.info("Transform daemon ready.")

    # -- public API ----------------------------------------------------------

    def transform(self, pdf_path: str, output_dir: str) -> Optional[str]:
        """Send one PDF to the daemon.  Returns abs path to .en.txt, or None."""
        with self._lock:
            try:
                self._ensure_running()
            except Exception as exc:
                logger.error(f"Could not start transform daemon: {exc}")
                return None

            request = json.dumps({
                "pdf_path": os.path.abspath(pdf_path),
                "output_dir": os.path.abspath(output_dir),
            })

            try:
                self._proc.stdin.write(request + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                logger.warning("Transform daemon pipe broken — will restart on next call.")
                self._proc = None
                return None

            raw = self._proc.stdout.readline().strip()
            if not raw:
                logger.error("Transform daemon returned empty response — marking for restart.")
                self._proc = None
                return None

            result = json.loads(raw)
            if result.get("status") == "ok":
                return result.get("en_txt")

            logger.error(f"PDF transform failed: {result.get('message', 'unknown')}")
            return None

    def shutdown(self):
        """Gracefully stop the daemon."""
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
            logger.info("Transform daemon shut down.")


# Module-level singleton — first call to transform() starts the daemon lazily.
_daemon = _TransformDaemon()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _copy_if_missing(src: Path, dst: Path):
    if not dst.exists():
        shutil.copy2(str(src), str(dst))


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def transform_single_pdf(pdf_path: str, output_dir: str) -> Optional[str]:
    """Transform one PDF through the warm daemon.

    Returns the absolute path to the produced .en.txt file, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    return _daemon.transform(pdf_path, output_dir)


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
