#!/usr/bin/env python3
"""Generation service (Ollama): standalone deploy script.

Standard library only, deliberately. This directory is meant to work when copied alone to a
machine that has never seen the rest of this repository:

    scp -r microservices/generation/ user@node:~/mimir-generation/
    ssh user@node 'cd ~/mimir-generation && cp .env.example .env && python3 deploy.py up'

So it cannot import anything from core/ or depend on a package installed system-wide beyond
Docker itself.

This is the one service where hardware genuinely changes both which engine runs and which
model it serves. `deploy.py tier` reports the decision and changes nothing; `up` acts on it.

    engine   when                                        models by capacity
    ------   -----------------------------------------   --------------------------------
    sglang   CUDA GPU, compute capability >= 7.5,        Qwen3-1.7B / 4B / 8B / 14B / 32B
             docker nvidia runtime present                by VRAM
    ollama   CUDA GPU older than 7.5 (K80, P100, V100)   qwen3:1.7b .. 30b by VRAM
      +gpu   or nvidia runtime present but SGLang unfit
    ollama   no usable GPU                                qwen3:1.7b, or 4b on a large host

All three bind 127.0.0.1:11500 on the host (11434 inside the container - deliberately not
11434 on the host, since that's Ollama's own default and collides with any bare `ollama
serve` a teammate runs on a shared machine) and speak /v1/chat/completions, so the
application config is identical whichever is chosen and switching engines is not an
application change.

The CPU tier is deliberately conservative. On four vCPUs qwen3:4b was measured at 17.5 tok/s
prompt processing and 3.0 tok/s generation, which is not usable for interactive answers, so
the 4b tier now needs real core count rather than merely enough RAM to hold the weights.

qwen3:4b also failed to reliably produce the conflict-warning callout during a real
deployment (see the officer portal's supersession-warning behaviour) - a model this small is
not just slow, it is untrustworthy for that specific behaviour. gemma4:26b at
reasoning_effort=low/none was the model that actually held up in practice, though it has not
been broken into per-VRAM-tier Ollama tags here, so it is set as a manual LOCAL_GEN_MODEL
override in the application's .env rather than wired into the tables below.

Point the application at this service with GEN_PROVIDER=local, LOCAL_GEN_URL and
LOCAL_GEN_MODEL matching what `deploy.py tier` reports (core/utils.py:local_generate_stream
speaks the standard /v1/chat/completions dialect, so any OpenAI-compatible server works).

    python deploy.py check   # is docker available, is .env present, is a GPU usable
    python deploy.py tier    # report detected hardware and the engine/model it implies
    python deploy.py up      # start the chosen engine, then pull the tiered model
    python deploy.py stop    # pause the container, keep it (fast to resume with `up`)
    python deploy.py down    # stop and remove the container
    python deploy.py status
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE_NAME = "generation"
# Ollama and SGLang both publish on host port 11500 (see docker-compose*.yml) but disagree
# about what a health probe looks like.
HEALTH_URL_OLLAMA = "http://127.0.0.1:11500/api/tags"
HEALTH_URL_SGLANG = "http://127.0.0.1:11500/v1/models"

# SGLang's attention kernels require Turing or newer. Older datacentre cards (K80 3.7,
# P100 6.0, V100 7.0) are common in university clusters and must fall back to Ollama, which
# drives them through its own CUDA build.
SGLANG_MIN_COMPUTE_CAPABILITY = 7.5

# VRAM needed to serve each model at fp16 with room left for the KV cache. SGLang loads
# HuggingFace weights unquantised, so these are roughly 2 bytes per parameter plus headroom.
SGLANG_TIERS = [
    (70000, "Qwen/Qwen3-32B"),
    (36000, "Qwen/Qwen3-14B"),
    (20000, "Qwen/Qwen3-8B"),
    (12000, "Qwen/Qwen3-4B"),
    (8000, "Qwen/Qwen3-1.7B"),
]
# Ollama ships 4-bit quantised GGUF, so the same card holds a considerably larger model.
OLLAMA_GPU_TIERS = [
    (24000, "qwen3:30b"),
    (12000, "qwen3:14b"),
    (8000, "qwen3:8b"),
    (6000, "qwen3:4b"),
    (0, "qwen3:1.7b"),
]

# Nominally-16GB cloud instances (e.g. AWS t3.xlarge) report less than 16 to sysconf due to
# kernel/hypervisor reservation, so a strict >= 16 check misses them. 15 gives headroom.
RAM_THRESHOLD_GB = 15
# On CPU the binding constraint is arithmetic throughput, not capacity: a 16GB four-vCPU host
# holds qwen3:4b comfortably and still generates at 3 tok/s. Require real cores for that tier.
CPU_CORE_THRESHOLD = 16
CPU_RAM_THRESHOLD_GB = 30


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


def detect_ram_gb():
    """Best-effort, stdlib only. Returns None rather than guessing if detection fails on an
    unfamiliar platform - the caller then falls back to the conservative CPU-small tier."""
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError):
        pass
    try:
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / (1024 ** 3)
    except (AttributeError, OSError):
        pass
    return None


def detect_cpu_cores():
    return os.cpu_count()


def detect_gpus():
    """Every CUDA GPU nvidia-smi can see, as {name, vram_total_mb, vram_free_mb, compute_capability}.

    Empty list means no usable GPU, whether because there is none, the driver is missing, or
    nvidia-smi failed. Callers treat all three the same way: use the CPU tier.

    Tiering is done on vram_free_mb, not the card's total capacity: this machine is shared,
    and nvidia-smi's memory.total does not know that another process (an already-running
    Ollama instance, say) is holding a chunk of it. Tiering on total capacity picked a model
    that needed more VRAM than was actually free and failed to load - the card looked capable
    on paper while a third of it was already spoken for.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
    except Exception:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            vram_total = int(float(parts[1]))
            vram_used = int(float(parts[2]))
        except ValueError:
            continue
        # compute_cap is absent on drivers older than 510; unknown means "assume too old
        # for SGLang" rather than "assume fine", so the fallback is the safe direction.
        try:
            cc = float(parts[3]) if len(parts) > 3 and parts[3] not in ("", "[N/A]") else None
        except ValueError:
            cc = None
        gpus.append({
            "name": parts[0],
            "vram_total_mb": vram_total,
            "vram_used_mb": vram_used,
            "vram_free_mb": max(vram_total - vram_used, 0),
            "compute_capability": cc,
        })
    return gpus


def docker_gpu_runtime_available():
    """Whether Docker can actually hand a GPU to a container, as (ok, reason).

    Distinct from "the host has a GPU": a machine can pass nvidia-smi on the host and still
    have no nvidia container runtime installed, in which case the container silently gets no
    device and runs a large tiered model on CPU. That failure is slow rather than loud, so it
    is worth a separate check.

    A permission-denied `docker info` (stale group membership - see _docker_available) looks
    identical to "no nvidia runtime" if not checked for separately, and sends the operator to
    install nvidia-container-toolkit when the runtime was there all along and the real fix was
    `newgrp docker`.
    """
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            if "permission denied" in result.stderr.lower():
                return False, ("permission denied talking to the Docker socket, not confirmed "
                               "missing - run `newgrp docker` or start a new login shell, then "
                               "re-check")
            return False, "docker info failed"
        if "nvidia" not in result.stdout.lower():
            return False, "no nvidia runtime registered with Docker"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _pick(tiers, vram_mb):
    for floor, model in tiers:
        if vram_mb >= floor:
            return model
    return tiers[-1][1]


def detect_plan():
    """Decide engine and model from hardware. Pure inspection - changes nothing."""
    gpus = detect_gpus()
    ram = detect_ram_gb()
    cores = detect_cpu_cores()
    warnings = []

    if gpus:
        vram_free = min(g["vram_free_mb"] for g in gpus)
        vram_total = min(g["vram_total_mb"] for g in gpus)
        caps = [g["compute_capability"] for g in gpus]
        worst_cap = None if any(c is None for c in caps) else min(caps)
        names = ", ".join(sorted({g["name"] for g in gpus}))
        runtime_ok, runtime_reason = docker_gpu_runtime_available()

        # More than a driver's worth of VRAM already gone means something else - another
        # Ollama instance, a display server, another team's job - is holding the card. Tier
        # on what's actually free, not the card's total capacity, or the picked model has
        # nowhere to load and fails at startup while the card looks capable on paper.
        if vram_total - vram_free > 1024:
            warnings.append(
                f"{vram_total - vram_free} MB of {vram_total} MB VRAM is already in use by "
                f"another process (this is a shared machine?) - tiering on the {vram_free} MB "
                "actually free, not the card's total capacity."
            )

        if not runtime_ok:
            if "permission denied" in runtime_reason.lower():
                warnings.append(
                    f"Could not confirm the nvidia runtime: {runtime_reason}. Falling back to "
                    "the CPU tier for now - this is likely a false negative, not a missing runtime."
                )
            else:
                warnings.append(
                    "nvidia-smi sees a GPU but Docker has no nvidia runtime, so a container "
                    "would get no device and run on CPU. Install nvidia-container-toolkit, then "
                    "re-run. Falling back to the CPU tier for now."
                )
        elif worst_cap is None:
            warnings.append(
                "GPU compute capability could not be read (driver older than 510?). "
                "Assuming it predates SGLang's requirement and using Ollama with GPU."
            )

        if runtime_ok:
            if worst_cap is not None and worst_cap >= SGLANG_MIN_COMPUTE_CAPABILITY:
                if len(gpus) > 1:
                    warnings.append(
                        f"{len(gpus)} GPUs present; this serves on one. Add --tp {len(gpus)} "
                        "to the SGLang command to shard across them."
                    )
                return {
                    "engine": "sglang",
                    "model": _pick(SGLANG_TIERS, vram_free),
                    "reason": f"{names}, {vram_free} MB VRAM free of {vram_total} MB, compute capability {worst_cap}",
                    "warnings": warnings,
                    "compose_files": ["-f", "docker-compose.sglang.yml"],
                    "health_url": HEALTH_URL_SGLANG,
                    "needs_pull": False,
                }
            cap_desc = "unknown" if worst_cap is None else str(worst_cap)
            return {
                "engine": "ollama+gpu",
                "model": _pick(OLLAMA_GPU_TIERS, vram_free),
                "reason": (f"{names}, {vram_free} MB VRAM free of {vram_total} MB, compute capability {cap_desc} "
                           f"(below SGLang's {SGLANG_MIN_COMPUTE_CAPABILITY})"),
                "warnings": warnings,
                "compose_files": ["-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml"],
                "health_url": HEALTH_URL_OLLAMA,
                "needs_pull": True,
            }

    ram_desc = f"{ram:.1f} GB RAM" if ram is not None else "RAM undetermined"
    core_desc = f"{cores} logical cores" if cores else "core count undetermined"
    big_enough = (ram is not None and ram >= CPU_RAM_THRESHOLD_GB
                  and cores is not None and cores >= CPU_CORE_THRESHOLD)
    if big_enough:
        model, note = "qwen3:4b", f"CPU only, {core_desc}, {ram_desc}"
    else:
        model, note = "qwen3:1.7b", f"CPU only, {core_desc}, {ram_desc}"
        if ram is not None and ram >= RAM_THRESHOLD_GB:
            warnings.append(
                "Host has memory for a larger model but not the cores to run it at "
                "interactive speed; staying on the small tier. Measured on four vCPUs, "
                "qwen3:4b generates at 3.0 tok/s."
            )
    return {
        "engine": "ollama", "model": model, "reason": note, "warnings": warnings,
        "compose_files": ["-f", "docker-compose.yml"],
        "health_url": HEALTH_URL_OLLAMA, "needs_pull": True,
    }


def _ensure_model_in_env(plan):
    """Write the tiered model into .env unless the operator pinned one themselves.

    qwen3:4b is the placeholder shipped in .env.example, so it counts as "not chosen" and
    gets overwritten; anything else is treated as a deliberate override and left alone. The
    engine matters here too - an Ollama tag left over from a CPU deployment is meaningless
    to SGLang, which wants a HuggingFace repo id.
    """
    env_file = HERE / ".env"
    text = env_file.read_text(encoding="utf-8")
    existing = _read_env_model()
    engine_mismatch = existing is not None and (
        (plan["engine"] == "sglang" and "/" not in existing)
        or (plan["engine"] != "sglang" and "/" in existing)
    )
    if existing and existing != "qwen3:4b" and not engine_mismatch:
        print(f"GEN_MODEL={existing} already set; leaving it alone.")
        return existing
    if engine_mismatch:
        print(f"GEN_MODEL={existing} does not match the {plan['engine']} engine; replacing it.")

    tier_model = plan["model"]
    if "GEN_MODEL=" in text:
        lines = [f"GEN_MODEL={tier_model}" if line.startswith("GEN_MODEL=") else line
                 for line in text.splitlines()]
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        with env_file.open("a", encoding="utf-8") as f:
            f.write(f"\nGEN_MODEL={tier_model}\n")
    return tier_model


def _print_plan(plan):
    print(f"engine     : {plan['engine']}")
    print(f"model      : {plan['model']}")
    print(f"hardware   : {plan['reason']}")
    for warning in plan["warnings"]:
        print(f"WARNING    : {warning}")


def _read_env_model():
    env_file = HERE / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEN_MODEL="):
            return line.split("=", 1)[1].strip()
    return None


def cmd_check(_args):
    ok_docker, msg_docker = _docker_available()
    print(f"docker      : {'OK' if ok_docker else 'FAIL'} - {msg_docker}")
    ok_env, msg_env = _env_ready()
    print(f".env        : {'OK' if ok_env else 'FAIL'} - {msg_env}")
    gpus = detect_gpus()
    if not gpus:
        print("gpu         : none detected, CPU tier")
    else:
        runtime_ok, runtime_reason = docker_gpu_runtime_available()
        runtime = "yes" if runtime_ok else f"NO - {runtime_reason}"
        for gpu in gpus:
            cap = gpu["compute_capability"]
            print(f"gpu         : {gpu['name']}, {gpu['vram_free_mb']} MB free of {gpu['vram_total_mb']} MB, "
                  f"compute capability {cap if cap is not None else 'unknown'}")
        print(f"docker gpu  : {runtime}")
    return 0 if (ok_docker and ok_env) else 1


def cmd_tier(args):
    plan = detect_plan()
    # --model-only exists so the repo-root orchestrator can read the tier without parsing
    # human-readable output, and without importing this module: the tiering rule stays in
    # one place and the two entry points cannot drift.
    if getattr(args, "model_only", False):
        print(plan["model"])
        return 0
    _print_plan(plan)
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

    plan = detect_plan()
    _print_plan(plan)
    model = _ensure_model_in_env(plan) or plan["model"]
    compose = plan["compose_files"]

    print(f"Starting {SERVICE_NAME} on {plan['engine']}...")
    # GEN_MODEL is interpolated into docker-compose.sglang.yml, so it has to be in this
    # process's environment and not only in .env.
    os.environ["GEN_MODEL"] = model
    result = _run(["docker", "compose"] + compose + ["up", "-d"])
    if result.returncode != 0:
        return result.returncode

    # SGLang downloads weights from HuggingFace during startup, so first boot legitimately
    # takes far longer than Ollama's, which serves immediately and pulls afterwards.
    attempts = 300 if plan["engine"] == "sglang" else 60
    print(f"Waiting for the API to come up (up to {attempts * 2}s)...", end="", flush=True)
    for _ in range(attempts):
        if _http_ok(plan["health_url"]):
            print(" ready.")
            break
        print(".", end="", flush=True)
        time.sleep(2)
    else:
        print()
        print(f"Started, but the API did not come up within the timeout. Check: "
              f"docker compose {' '.join(compose)} logs {SERVICE_NAME}")
        return 1

    if plan["needs_pull"]:
        print(f"Pulling {model} (this can take a while on first run)...")
        pull = _run(["docker", "compose"] + compose + ["exec", "-T", SERVICE_NAME, "ollama", "pull", model])
        if pull.returncode != 0:
            print(f"Model pull failed. The service is up, but {model} is not available yet; "
                  f"retry manually with: docker compose {' '.join(compose)} exec "
                  f"{SERVICE_NAME} ollama pull {model}")
            return pull.returncode

    print(f"{model} ready on {plan['engine']}. Point the application at GEN_PROVIDER=local, "
          f"LOCAL_GEN_URL=http://<this-host>:11500/v1, LOCAL_GEN_MODEL={model}")
    return 0


def cmd_stop(_args):
    """Pause the container without removing it - a later `up` just restarts it, no rebuild."""
    return _run(["docker", "compose"] + detect_plan()["compose_files"] + ["stop"]).returncode


def cmd_down(_args):
    return _run(["docker", "compose"] + detect_plan()["compose_files"] + ["down"]).returncode


def cmd_status(_args):
    plan = detect_plan()
    up = _http_ok(plan["health_url"])
    print(f"{SERVICE_NAME}: {'up' if up else 'down or unreachable'} "
          f"({plan['health_url']}, engine {plan['engine']})")
    _run(["docker", "compose"] + plan["compose_files"] + ["ps"])
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
    tier_parser = sub.add_parser("tier", help="Report detected hardware and the engine/model it implies. Changes nothing.")
    tier_parser.add_argument("--model-only", action="store_true",
                             help="Print just the model name, for scripts.")
    sub.add_parser("up", help="Start the service and pull the tiered model.")
    sub.add_parser("stop", help="Pause the container, keep it - a later `up` just restarts it.")
    sub.add_parser("down", help="Stop the service and remove its container.")
    sub.add_parser("status", help="Check whether it is reachable right now.")
    args = parser.parse_args()

    handlers = {"check": cmd_check, "tier": cmd_tier, "up": cmd_up, "stop": cmd_stop, "down": cmd_down, "status": cmd_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
