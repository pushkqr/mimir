#!/usr/bin/env python3
"""Embeddings + reranker service (Infinity): standalone deploy script.

Standard library only, deliberately. This directory is meant to work when copied alone to a
machine that has never seen the rest of this repository:

    scp -r microservices/embeddings/ user@node:~/mimir-embeddings/
    ssh user@node 'cd ~/mimir-embeddings && cp .env.example .env && $EDITOR .env && python3 deploy.py up'

So it cannot import anything from core/ or depend on a package installed system-wide beyond
Docker itself.

The embedding model is pinned to BAAI/bge-m3 and is never selected by hardware. Every vector
already in the corpus is 1024-dimensional; swapping the model does not degrade quality, it
makes the entire index unreadable and forces a full re-ingest. Only the reranker tier adapts:

    python deploy.py check   # is docker available, is .env filled in
    python deploy.py tier    # report detected hardware and the reranker tier it implies
    python deploy.py up      # write RERANK_MODEL into .env if unset, then docker compose up -d
    python deploy.py stop    # pause the container, keep it (fast to resume with `up`)
    python deploy.py down    # stop and remove the container
    python deploy.py status
"""

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE_NAME = "infinity"
HEALTH_URL = "http://127.0.0.1:7997/health"

EMBEDDING_MODEL = "BAAI/bge-m3"  # pinned, see module docstring - never changes with hardware
GPU_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CPU_RERANK_MODEL = "BAAI/bge-reranker-base"
GPU_VRAM_THRESHOLD_MB = 6000  # below this, the CPU-sized reranker is the safer default


def _run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=HERE, **kwargs)


def _docker_available():
    if shutil.which("docker") is None:
        return False, "docker CLI not found on PATH. Install Docker first."
    result = _run(["docker", "info"], capture_output=True, text=True)
    if result.returncode != 0:
        if "permission denied" in result.stderr.lower():
            return False, ("permission denied talking to the Docker socket. If this account was "
                           "just added to the docker group, that only takes effect in new "
                           "sessions - run `newgrp docker` or start a new login shell, then retry.")
        return False, "docker daemon not reachable. Is Docker running?"
    return True, "ok"


def _env_ready():
    env_file = HERE / ".env"
    if not env_file.exists():
        return False, f"{env_file.name} missing. Copy .env.example to .env and fill it in."
    text = env_file.read_text(encoding="utf-8")
    if "change-me" in text:
        return False, f"{env_file.name} still has a placeholder value. Edit it before deploying."
    return True, "ok"


def _http_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def detect_gpu_vram_mb():
    """None means no GPU detected, not an error - the CPU tier is the safe default either way."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def detect_reranker_tier():
    vram = detect_gpu_vram_mb()
    if vram is not None and vram >= GPU_VRAM_THRESHOLD_MB:
        return GPU_RERANK_MODEL, f"GPU detected with {vram} MB VRAM"
    if vram is not None:
        return CPU_RERANK_MODEL, f"GPU detected but only {vram} MB VRAM, below the {GPU_VRAM_THRESHOLD_MB} MB threshold"
    return CPU_RERANK_MODEL, "no GPU detected (nvidia-smi not found or returned nothing)"


def _ensure_rerank_model_in_env():
    """Fill RERANK_MODEL into .env if the placeholder default is still sitting there unedited,
    rather than overwriting a value the operator deliberately set."""
    env_file = HERE / ".env"
    text = env_file.read_text(encoding="utf-8")
    if "RERANK_MODEL=" in text and "RERANK_MODEL=BAAI/bge-reranker-base" not in text:
        return  # operator already set something else; leave it alone
    tier_model, reason = detect_reranker_tier()
    print(f"Reranker tier: {tier_model} ({reason})")
    if "RERANK_MODEL=" in text:
        lines = [f"RERANK_MODEL={tier_model}" if line.startswith("RERANK_MODEL=") else line
                 for line in text.splitlines()]
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        with env_file.open("a", encoding="utf-8") as f:
            f.write(f"\nRERANK_MODEL={tier_model}\n")


def cmd_check(_args):
    ok_docker, msg_docker = _docker_available()
    print(f"docker      : {'OK' if ok_docker else 'FAIL'} - {msg_docker}")
    ok_env, msg_env = _env_ready()
    print(f".env        : {'OK' if ok_env else 'FAIL'} - {msg_env}")
    print(f"embedding   : {EMBEDDING_MODEL} (pinned, not hardware-selected)")
    return 0 if (ok_docker and ok_env) else 1


def cmd_tier(_args):
    tier_model, reason = detect_reranker_tier()
    print(f"embedding model : {EMBEDDING_MODEL} (always pinned)")
    print(f"reranker tier   : {tier_model}")
    print(f"reason          : {reason}")
    return 0


def cmd_up(_args):
    ok_docker, msg_docker = _docker_available()
    if not ok_docker:
        print(f"Cannot start: {msg_docker}")
        return 1
    ok_env, msg_env = _env_ready()
    if not ok_env:
        print(f"Cannot start: {msg_env}")
        return 1

    _ensure_rerank_model_in_env()

    print(f"Starting {SERVICE_NAME}...")
    result = _run(["docker", "compose", "up", "-d"])
    if result.returncode != 0:
        return result.returncode

    print("Waiting for it to report healthy...", end="", flush=True)
    for _ in range(90):  # model download + load can take a while on first start
        if _http_ok(HEALTH_URL):
            print(" ready.")
            return 0
        print(".", end="", flush=True)
        time.sleep(3)
    print()
    print(f"Started, but did not report healthy within the timeout. Check: docker compose logs {SERVICE_NAME}")
    return 1


def cmd_stop(_args):
    """Pause the container without removing it - a later `up` just restarts it, no rebuild."""
    return _run(["docker", "compose", "stop"]).returncode


def cmd_down(_args):
    return _run(["docker", "compose", "down"]).returncode


def cmd_status(_args):
    up = _http_ok(HEALTH_URL)
    print(f"{SERVICE_NAME}: {'up' if up else 'down or unreachable'} ({HEALTH_URL})")
    _run(["docker", "compose", "ps"])
    return 0 if up else 1


def main():
    # See the root deploy.py for why this matters: without it, this script's own print()
    # calls can appear out of order relative to subprocess (docker compose) output.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Verify Docker and .env are ready. Changes nothing.")
    sub.add_parser("tier", help="Report detected hardware and the reranker tier it implies. Changes nothing.")
    sub.add_parser("up", help="Start the service and wait for it to become healthy.")
    sub.add_parser("stop", help="Pause the container, keep it - a later `up` just restarts it.")
    sub.add_parser("down", help="Stop the service and remove its container.")
    sub.add_parser("status", help="Check whether it is reachable right now.")
    args = parser.parse_args()

    handlers = {"check": cmd_check, "tier": cmd_tier, "up": cmd_up, "stop": cmd_stop, "down": cmd_down, "status": cmd_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
