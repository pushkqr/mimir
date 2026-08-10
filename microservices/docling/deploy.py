#!/usr/bin/env python3
"""Docling parsing service: standalone deploy script.

Standard library only, deliberately. Unlike the other services in microservices/, this one's
docker-compose.yml build context reaches back into ../../docling_ingestion (the service source
lives at the repository root, not duplicated in here), so this directory is not fully
self-contained the way the others are - it needs the rest of the repository present, or at
minimum docling_ingestion/ copied alongside it at the same relative path.

    python deploy.py check   # is docker available
    python deploy.py up      # docker compose up -d, then wait for the container to be healthy
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
SERVICE_NAME = "docling"
HEALTH_URL = "http://127.0.0.1:8002/health"


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


def _build_context_present():
    context = (HERE / "../../docling_ingestion").resolve()
    return context.exists(), str(context)


def _http_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def cmd_check(_args):
    ok_docker, msg_docker = _docker_available()
    print(f"docker        : {'OK' if ok_docker else 'FAIL'} - {msg_docker}")
    ok_ctx, ctx_path = _build_context_present()
    print(f"build context : {'OK' if ok_ctx else 'FAIL'} - {ctx_path}")
    return 0 if (ok_docker and ok_ctx) else 1


def cmd_up(_args):
    ok_docker, msg_docker = _docker_available()
    if not ok_docker:
        print(f"Cannot start: {msg_docker}")
        return 1
    ok_ctx, ctx_path = _build_context_present()
    if not ok_ctx:
        print(f"Cannot start: build context not found at {ctx_path}. "
              f"This service needs docling_ingestion/ present two levels up.")
        return 1

    print(f"Starting {SERVICE_NAME}...")
    result = _run(["docker", "compose", "up", "-d", "--build"])
    if result.returncode != 0:
        return result.returncode

    print("Waiting for it to report healthy...", end="", flush=True)
    for _ in range(60):
        if _http_ok(HEALTH_URL):
            print(" ready.")
            return 0
        print(".", end="", flush=True)
        time.sleep(2)
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
    sub.add_parser("check", help="Verify Docker and the build context are ready. Changes nothing.")
    sub.add_parser("up", help="Build and start the service, wait for it to become healthy.")
    sub.add_parser("stop", help="Pause the container, keep it - a later `up` just restarts it.")
    sub.add_parser("down", help="Stop the service and remove its container.")
    sub.add_parser("status", help="Check whether it is reachable right now.")
    args = parser.parse_args()

    handlers = {"check": cmd_check, "up": cmd_up, "stop": cmd_stop, "down": cmd_down, "status": cmd_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
