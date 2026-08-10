#!/usr/bin/env python3
"""Translation service (IndicTrans2): standalone deploy script.

Standard library only, deliberately. This directory is meant to work when copied alone to a
machine that has never seen the rest of this repository:

    scp -r microservices/translation/ user@node:~/mimir-translation/
    ssh user@node 'cd ~/mimir-translation && cp .env.example .env && python3 deploy.py up'

So it cannot import anything from core/ or depend on a package installed system-wide beyond
Docker itself. No API key is required by this service today; if that changes, keep the
pattern the other services use rather than inventing a new one.

    python deploy.py check   # is docker available, is .env present
    python deploy.py tier    # report detected hardware and the model tier it implies
    python deploy.py up      # write INDICTRANS_MODEL into .env if unset, then docker compose up -d
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
SERVICE_NAME = "translation"
HEALTH_URL = "http://127.0.0.1:8001/health"

CPU_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
GPU_MODEL = "ai4bharat/indictrans2-indic-en-1B"
GPU_VRAM_THRESHOLD_MB = 6000


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
        return False, f"{env_file.name} missing. Copy .env.example to .env."
    return True, "ok"


def _http_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def detect_gpu_vram_mb():
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


def detect_model_tier():
    vram = detect_gpu_vram_mb()
    if vram is not None and vram >= GPU_VRAM_THRESHOLD_MB:
        return GPU_MODEL, f"GPU detected with {vram} MB VRAM"
    if vram is not None:
        return CPU_MODEL, f"GPU detected but only {vram} MB VRAM, below the {GPU_VRAM_THRESHOLD_MB} MB threshold"
    return CPU_MODEL, "no GPU detected (nvidia-smi not found or returned nothing)"


def _ensure_model_in_env():
    env_file = HERE / ".env"
    text = env_file.read_text(encoding="utf-8")
    if "INDICTRANS_MODEL=" in text and f"INDICTRANS_MODEL={CPU_MODEL}" not in text:
        return  # operator already set something else; leave it alone
    tier_model, reason = detect_model_tier()
    print(f"Translation model tier: {tier_model} ({reason})")
    if "INDICTRANS_MODEL=" in text:
        lines = [f"INDICTRANS_MODEL={tier_model}" if line.startswith("INDICTRANS_MODEL=") else line
                 for line in text.splitlines()]
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        with env_file.open("a", encoding="utf-8") as f:
            f.write(f"\nINDICTRANS_MODEL={tier_model}\n")


def cmd_check(_args):
    ok_docker, msg_docker = _docker_available()
    print(f"docker      : {'OK' if ok_docker else 'FAIL'} - {msg_docker}")
    ok_env, msg_env = _env_ready()
    print(f".env        : {'OK' if ok_env else 'FAIL'} - {msg_env}")
    return 0 if (ok_docker and ok_env) else 1


def cmd_tier(_args):
    tier_model, reason = detect_model_tier()
    print(f"model tier : {tier_model}")
    print(f"reason     : {reason}")
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

    _ensure_model_in_env()

    print(f"Starting {SERVICE_NAME}... (first build compiles a PyTorch image, can take a while)")
    result = _run(["docker", "compose", "up", "-d", "--build"])
    if result.returncode != 0:
        return result.returncode

    print("Waiting for it to report healthy (model load can take several minutes)...", end="", flush=True)
    for _ in range(120):
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
    sub.add_parser("tier", help="Report detected hardware and the model tier it implies. Changes nothing.")
    sub.add_parser("up", help="Build and start the service, wait for it to become healthy.")
    sub.add_parser("stop", help="Pause the container, keep it - a later `up` just restarts it.")
    sub.add_parser("down", help="Stop the service and remove its container.")
    sub.add_parser("status", help="Check whether it is reachable right now.")
    args = parser.parse_args()

    handlers = {"check": cmd_check, "tier": cmd_tier, "up": cmd_up, "stop": cmd_stop, "down": cmd_down, "status": cmd_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
