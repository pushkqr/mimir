#!/usr/bin/env python3
"""Mimir deployment orchestrator.

Brings the whole stack up on one machine, for a fully self-hosted deployment (see
microservices/README.md and scratch/build-plan.md Phase 5/6). Runs from a full checkout of
this repository, unlike the scripts under microservices/*/, which are each standalone and
copyable to a machine that has never seen this repo.

Core principle this script follows: it does not reimplement any service's startup logic. It
shells out to the exact same deploy.py each service directory carries on its own — the one
you would run after `scp -r microservices/embeddings/ node:`. One code path, two entry
points. If this orchestrator had its own copy of that logic, the two would drift, and the
remote-node path is the one that would break, silently, exactly when it's needed on stage or
in front of a department that just handed over a server.

On a machine that has nothing but this checkout and Docker, the whole deployment is:

    python deploy.py up

which writes any missing .env files first, then starts everything. The longer form, when you
want to see each step:

    python deploy.py init                # write every .env: derived URLs, generated secrets
    python deploy.py check               # prerequisites, hardware, per-service readiness
    python deploy.py up                  # bring every service up, in order, then the app
    python deploy.py up --only weaviate  # just one service
    python deploy.py stop                # pause everything, keep containers (fast to resume with `up`)
    python deploy.py stop --only weaviate  # just one service
    python deploy.py down                # tear everything down, reverse order
    python deploy.py status              # live reachability, same probes as the admin panel
    python deploy.py config              # required values, values that must agree, what services report
    python deploy.py logs weaviate       # passthrough to that service's container logs

`init` removes the configuration a deployer would otherwise have to derive by hand. Service
URLs follow from the host and the ports the compose files publish. Secrets shared between the
application and a service exist only to match, so they are generated into both sides at once -
the class of mistake `config` was written to catch is largely a consequence of those two
values being typed twice. Credentials for third parties cannot be invented and are listed.

`check` asks whether services can start, `status` whether they are reachable, and `config`
whether what they were told is coherent. A stack can pass the first two and still be
misconfigured in ways that only appear under real traffic.
"""

import argparse
import ctypes
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MICROSERVICES_DIR = REPO_ROOT / "microservices"

# Host-side ports, as published in each service's docker-compose.yml. These are what the
# application connects to, and they are not all equal to the port inside the container:
# translation and docling both listen on 8000 internally and are published elsewhere.
SERVICE_PORTS = {
    "weaviate": 8080,
    "embeddings": 7997,
    "translation": 8001,
    "docling": 8002,
    "transform": 8003,
    # Not 11434 (Ollama's own default): any bare `ollama serve` a teammate runs on a shared
    # host claims that port with no prompt to change it, colliding with this container.
    "generation": 11500,
}

# Values that ship in the .env.example files and mean "not configured yet". init overwrites
# these and nothing else, so re-running it never disturbs a value someone actually chose.
PLACEHOLDER_VALUES = {
    "change-me-to-a-long-random-string",
    "your_secure_admin_token",
    "your_cerebras_api_key",
    "your-gcp-project-id",
    "your_hf_token",
    "your_gemini_api_key",
}
_PLACEHOLDER_PATTERN = re.compile(r"<[A-Z0-9_]+>")

# Secrets shared between the application and a service. Both sides must carry the identical
# value; when they drift the service starts, answers its health check, and rejects every real
# request. Generating them together is the only way to be sure they agree.
SHARED_SECRETS = [
    # (generated once as, service dir, key in that service's .env, keys in the root .env)
    ("weaviate", "weaviate", "WEAVIATE_API_KEY", ["WEAVIATE_API_KEY"]),
    ("infinity", "embeddings", "INFINITY_API_KEY", ["LOCAL_EMBED_API_KEY", "LOCAL_RERANK_API_KEY"]),
]


def _is_placeholder(value: str) -> bool:
    value = value.strip()
    return (not value) or value in PLACEHOLDER_VALUES or bool(_PLACEHOLDER_PATTERN.search(value))


def _read_env(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        # Strip trailing inline comments; .env.example uses them heavily and dotenv keeps
        # them as part of the value, which is how a URL ends up with a comment glued to it.
        values[key.strip()] = value.split("#", 1)[0].strip()
    return values


def _set_env(path: Path, key: str, value: str, force: bool = False) -> bool:
    """Set key in an env file, preserving order and comments. Returns whether it changed.

    Without force, an existing non-placeholder value is left exactly as it is, which is what
    makes init safe to re-run against a deployment someone has already tuned by hand.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        if not line.strip().startswith(f"{key}=") and line.strip().split("=", 1)[0].strip() != key:
            continue
        current = line.split("=", 1)[1] if "=" in line else ""
        if not force and not _is_placeholder(current.split("#", 1)[0]):
            return False
        lines[index] = f"{key}={value}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True

# Order matters for `up`/`down` only in the sense that infrastructure should be reachable
# before the application starts; the services themselves have no dependencies on each other.
SERVICE_ORDER = ["weaviate", "embeddings", "translation", "generation", "transform"]
OPTIONAL_SERVICES = ["docling"]  # not brought up by default; pass --only docling to include


def _service_dir(name: str) -> Path:
    return MICROSERVICES_DIR / name


def _run_service_script(name: str, command: str) -> int:
    script = _service_dir(name) / "deploy.py"
    if not script.exists():
        print(f"[{name}] no deploy.py found at {script}")
        return 1
    result = subprocess.run([sys.executable, str(script), command])
    return result.returncode


def _detect_ram_gb():
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


def _detect_gpu_vram_mb():
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


def _disk_free_gb():
    try:
        return shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    except OSError:
        return None


def _detected_gen_model() -> str:
    """Ask the generation service's own script which model this hardware implies.

    Shelling out rather than importing keeps the single-source-of-truth rule this file opens
    with: the tiering logic lives in one place and the orchestrator does not carry a copy.
    """
    script = _service_dir("generation") / "deploy.py"
    try:
        result = subprocess.run([sys.executable, str(script), "tier", "--model-only"],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[-1].strip()
    except Exception:
        pass
    return ""


def cmd_init(args):
    """Write every .env this stack needs, deriving what can be derived.

    Three kinds of value exist here and only the third needs a human. URLs follow from which
    host the services run on and which ports their compose files publish. Shared secrets have
    no meaning beyond matching on both sides, so they are generated. Credentials for outside
    services - Cerebras, Google Cloud, HuggingFace - cannot be invented and are reported.

    Re-running is safe: a value that is not one of the shipped placeholders is left alone.
    """
    host = args.host
    print(f"Target host for service URLs: {host}")
    if host not in ("localhost", "127.0.0.1"):
        print("  note: every service's compose file publishes on 127.0.0.1 only. For a\n"
              "        multi-machine deployment, change those port bindings too, or the\n"
              "        application will not reach them from another host.")
    print()

    print("Service env files")
    print("-----------------")
    for name in SERVICE_ORDER + OPTIONAL_SERVICES:
        example = _service_dir(name) / ".env.example"
        target = _service_dir(name) / ".env"
        if not example.exists():
            print(f"  {name:<12} no .env.example, nothing to do")
            continue
        if target.exists():
            print(f"  {name:<12} .env exists, leaving it")
        else:
            shutil.copyfile(example, target)
            print(f"  {name:<12} created .env from .env.example")

    root_env = REPO_ROOT / ".env"
    root_example = REPO_ROOT / ".env.example"
    if not root_env.exists():
        if not root_example.exists():
            print("\nNo .env.example at the repo root; cannot create .env.")
            return 1
        shutil.copyfile(root_example, root_env)
        print(f"  {'application':<12} created .env from .env.example")
    else:
        print(f"  {'application':<12} .env exists, leaving it")

    print()
    print("Shared secrets")
    print("--------------")
    for label, service, service_key, root_keys in SHARED_SECRETS:
        service_env = _service_dir(service) / ".env"
        if not service_env.exists():
            continue
        existing = _read_env(service_env).get(service_key, "")
        if _is_placeholder(existing):
            value = secrets.token_urlsafe(32)
            _set_env(service_env, service_key, value, force=True)
            for key in root_keys:
                _set_env(root_env, key, value, force=True)
            print(f"  {label:<12} generated, written to {service}/.env and "
                  f"{', '.join(root_keys)}")
        else:
            changed = [k for k in root_keys if _set_env(root_env, k, existing, force=True)]
            print(f"  {label:<12} already set in {service}/.env"
                  + (f", copied to {', '.join(changed)}" if changed else ""))

    admin = _read_env(root_env).get("MIMIR_ADMIN_TOKEN", "")
    if _is_placeholder(admin):
        _set_env(root_env, "MIMIR_ADMIN_TOKEN", secrets.token_urlsafe(32), force=True)
        print(f"  {'admin token':<12} generated")
    else:
        print(f"  {'admin token':<12} already set, leaving it")

    # Propagate HF_TOKEN to microservices that need it
    root_env_values = _read_env(root_env)
    hf_token = root_env_values.get("HF_TOKEN", "")
    if not _is_placeholder(hf_token):
        for service in ["translation", "transform"]:
            service_env = _service_dir(service) / ".env"
            if _service_dir(service).exists():
                _set_env(service_env, "HF_TOKEN", hf_token, force=True)
        print(f"  {'HF_TOKEN':<12} propagated to translation and transform microservices")

    print()
    print("Derived service URLs")
    print("--------------------")
    derived = {
        "WEAVIATE_URL": f"http://{host}:{SERVICE_PORTS['weaviate']}",
        "LOCAL_EMBED_URL": f"http://{host}:{SERVICE_PORTS['embeddings']}/embeddings",
        "LOCAL_RERANK_URL": f"http://{host}:{SERVICE_PORTS['embeddings']}/rerank",
        "TRANSLATION_SERVICE_URL": f"http://{host}:{SERVICE_PORTS['translation']}/translate",
        # Absent from .env.example entirely, though config_check requires it whenever
        # ingestion translates locally. Left unset, ingestion quietly uses the smaller
        # query-tier model and nothing reports it.
        "INGEST_TRANSLATION_SERVICE_URL": f"http://{host}:{SERVICE_PORTS['translation']}/translate",
        "DOCLING_SERVICE_URL": f"http://{host}:{SERVICE_PORTS['docling']}/parse",
        "TRANSFORM_SERVICE_URL": f"http://{host}:{SERVICE_PORTS['transform']}/transform",
        "LOCAL_GEN_URL": f"http://{host}:{SERVICE_PORTS['generation']}/v1",
    }
    remote = host not in ("localhost", "127.0.0.1")
    for key, value in derived.items():
        current = _read_env(root_env).get(key, "")
        # .env.example ships literal localhost URLs for docling and generation, which are
        # real values rather than placeholders and so would survive --host. That leaves a
        # multi-machine deployment pointing two of its services at the wrong machine, with
        # nothing to indicate it. A loopback URL is meaningless once a host was named.
        stale_loopback = remote and ("localhost" in current or "127.0.0.1" in current)
        if _set_env(root_env, key, value, force=args.force_urls or stale_loopback):
            print(f"  set   {key} = {value}")
        else:
            print(f"  keep  {key} = {current}")

    # qwen3:4b is the value shipped in .env.example, and microservices/generation/deploy.py
    # already treats it as "nobody chose this" for the same reason. Overwriting it here keeps
    # the two files agreeing about which model the hardware can actually serve; anything else
    # is a deliberate choice and survives.
    gen_model = _detected_gen_model()
    current_gen = _read_env(root_env).get("LOCAL_GEN_MODEL", "")
    if gen_model:
        unchosen = _is_placeholder(current_gen) or current_gen == "qwen3:4b"
        if _set_env(root_env, "LOCAL_GEN_MODEL", gen_model, force=unchosen):
            print(f"  set   LOCAL_GEN_MODEL = {gen_model}  (from detected hardware)")
        else:
            print(f"  keep  LOCAL_GEN_MODEL = {current_gen}")

    print()
    print("Still needs you")
    print("---------------")
    current = _read_env(root_env)
    sovereign = current.get("DEPLOYMENT_MODE", "").lower() == "sovereign"
    outstanding = []
    for key, why in [
        ("CEREBRAS_API_KEY", "hosted generation; not needed if GEN_PROVIDER=local"),
        ("GOOGLE_CLOUD_PROJECT", "Gemini fallback and Document AI OCR"),
        ("HF_TOKEN", "IndicTrans2 checkpoints are gated on HuggingFace"),
    ]:
        if _is_placeholder(current.get(key, "")):
            outstanding.append((key, why))
    if not outstanding:
        print("  nothing - every credential is set")
    for key, why in outstanding:
        print(f"  {key:<22} {why}")

    if current.get("MIMIR_ALLOWED_SUBNETS", "") == "0.0.0.0/0":
        print("\n  WARNING  MIMIR_ALLOWED_SUBNETS is 0.0.0.0/0, which accepts officer logins\n"
              "           from anywhere. Narrow it to your department's range before this is\n"
              "           reachable from an untrusted network. Left as-is because tightening\n"
              "           it blindly can lock you out of your own deployment.")

    print(f"\n.env files written. Next: python {Path(__file__).name} check")
    if sovereign:
        print("DEPLOYMENT_MODE=sovereign, so generation runs locally and Cerebras is unused.")
    return 0


def _docker_install_hint() -> str:
    """The actual command for this machine, rather than a link to a docs page."""
    if sys.platform == "darwin":
        return "brew install --cask docker    (or download Docker Desktop)"
    if sys.platform.startswith("win"):
        return "winget install Docker.DockerDesktop"
    release = ""
    try:
        release = Path("/etc/os-release").read_text(encoding="utf-8").lower()
    except OSError:
        pass
    if any(name in release for name in ("ubuntu", "debian")):
        return ("curl -fsSL https://get.docker.com | sudo sh && "
                "sudo usermod -aG docker $USER   (then log out and back in)")
    if any(name in release for name in ("rhel", "centos", "fedora", "amzn")):
        return ("sudo dnf install -y docker && sudo systemctl enable --now docker && "
                "sudo usermod -aG docker $USER")
    return "https://docs.docker.com/engine/install/"


def cmd_check(_args):
    print("Prerequisites")
    print("-------------")
    docker_ok = shutil.which("docker") is not None
    if docker_ok:
        daemon = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if daemon.returncode == 0:
            print("  docker      : OK")
        else:
            docker_ok = False
            if "permission denied" in daemon.stderr.lower():
                print("  docker      : permission denied talking to the Docker socket")
                print("                if this account was just added to the docker group, that only")
                print("                takes effect in new sessions - try: newgrp docker")
                print("                (or start a new login shell), then re-run this check")
            else:
                print("  docker      : installed but the daemon is not reachable")
                print(f"                try: sudo systemctl enable --now docker")
    else:
        print("  docker      : MISSING")
        print(f"                install with: {_docker_install_hint()}")
    compose_ok = docker_ok and subprocess.run(
        ["docker", "compose", "version"], capture_output=True, text=True).returncode == 0
    print(f"  compose     : {'OK' if compose_ok else 'MISSING (needs the Docker Compose v2 plugin)'}")
    missing_env = [n for n in SERVICE_ORDER if (_service_dir(n) / ".env.example").exists()
                   and not (_service_dir(n) / ".env").exists()]
    if not (REPO_ROOT / ".env").exists():
        missing_env.append("application")
    if missing_env:
        print(f"  env files   : missing for {', '.join(missing_env)}")
        print(f"                run: python {Path(__file__).name} init")
    else:
        print("  env files   : present")

    print()
    print("Hardware")
    print("--------")
    ram = _detect_ram_gb()
    vram = _detect_gpu_vram_mb()
    disk = _disk_free_gb()
    print(f"  CPUs        : {os.cpu_count() or 'unknown'}")
    print(f"  RAM         : {f'{ram:.1f} GB' if ram is not None else 'undetermined'}")
    print(f"  GPU VRAM    : {f'{vram} MB' if vram is not None else 'no GPU detected'}")
    print(f"  Disk free   : {f'{disk:.1f} GB' if disk is not None else 'undetermined'} "
          f"(images ~12GB, models ~5GB, plus your corpus index)")

    print()
    print("Per-service readiness (docker + .env), via each service's own `deploy.py check`")
    print("--------------------------------------------------------------------------------")
    all_ok = True
    for name in SERVICE_ORDER:
        print(f"\n[{name}]")
        rc = _run_service_script(name, "check")
        all_ok = all_ok and (rc == 0)
    return 0 if all_ok else 1


def cmd_up(args):
    targets = [args.only] if args.only else list(SERVICE_ORDER)
    if not args.only and hasattr(args, 'skip') and args.skip:
        if args.skip in targets:
            targets.remove(args.skip)

    unknown = [t for t in targets if t not in SERVICE_ORDER + OPTIONAL_SERVICES]
    if unknown:
        print(f"Unknown service(s): {', '.join(unknown)}. Known: {', '.join(SERVICE_ORDER + OPTIONAL_SERVICES)}")
        return 1

    # Failing with "copy .env.example to .env" is friction with no information in it: the
    # answer is always yes, and the deployer has nothing to decide. Generate them and say so.
    needs_init = [n for n in targets if (_service_dir(n) / ".env.example").exists()
                  and not (_service_dir(n) / ".env").exists()]
    if not args.only and not (REPO_ROOT / ".env").exists():
        needs_init.append("application")
    if needs_init and not args.no_init:
        print(f"No .env for {', '.join(needs_init)}; running init first.\n")
        rc = cmd_init(argparse.Namespace(host=args.host, force_urls=False))
        if rc != 0:
            return rc
        print()

    for name in targets:
        print(f"\n=== {name} ===")
        rc = _run_service_script(name, "up")
        if rc != 0:
            print(f"\n[{name}] failed to come up (exit {rc}). Stopping here rather than "
                  f"starting the application against a stack that isn't fully green.")
            return rc

    if args.only:
        print(f"\n{args.only} is up. Not starting the application (--only was given).")
        return 0

    print("\n=== application ===")
    result = subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

    print("\nAll services and the application are up. Run `python deploy.py status` to verify "
          "the application can actually reach each one (network policy, firewall, etc. can "
          "still block a service that Docker itself reports as running).")
    return 0


def cmd_stop(args):
    """Pause containers without removing them - unlike `down`, a later `up` just restarts
    them (no rebuild, no re-pull). Use this to free RAM/VRAM/CPU between sessions without
    losing the running configuration, e.g. while another process on the same host needs the
    memory back."""
    if not args.only:
        print("=== application ===")
        subprocess.run(["docker", "compose", "stop"], cwd=REPO_ROOT)

    targets = [args.only] if args.only else list(reversed(SERVICE_ORDER))
    all_ok = True
    for name in targets:
        print(f"\n=== {name} ===")
        rc = _run_service_script(name, "stop")
        all_ok = all_ok and (rc == 0)
    return 0 if all_ok else 1


def cmd_down(args):
    if not args.only:
        print("=== application ===")
        subprocess.run(["docker", "compose", "down"], cwd=REPO_ROOT)

    targets = [args.only] if args.only else list(reversed(SERVICE_ORDER))
    all_ok = True
    for name in targets:
        print(f"\n=== {name} ===")
        rc = _run_service_script(name, "down")
        all_ok = all_ok and (rc == 0)
    return 0 if all_ok else 1


def cmd_status(_args):
    """Same probes /api/admin/topology uses, run standalone. Requires this repo's Python
    environment (google-genai, weaviate-client, etc.), unlike the per-service `deploy.py
    status` commands, which only need the standard library."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from core.health import run_all_probes
    except ImportError as exc:
        print(f"Could not import this repo's dependencies ({exc}). "
              f"Run this from an environment with requirements.txt installed, "
              f"or check individual services with e.g. `python microservices/weaviate/deploy.py status`.")
        return 1

    result = run_all_probes()
    print(f"{result['self_hosted']} of {len(result['components'])} components self-hosted "
          f"(generation provider: {result['generation_provider']})\n")
    all_up = True
    for component in result["components"]:
        status = component["status"]
        all_up = all_up and (status == "up")
        marker = "UP  " if status == "up" else "DOWN"
        print(f"  [{marker}] {component['name']:<18} {component['host']:<28} "
              f"{component['latency_ms']:>5} ms   {component['detail']}")
    return 0 if all_up else 1


def cmd_config(args):
    """Check the configuration this process actually holds, not what a file claims.

    Separate from `check`, which asks whether services can start, and from `status`, which
    asks whether they are reachable. A stack can pass both of those and still be misconfigured
    in ways that only surface on real traffic.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from core.config_check import check_config, check_env_file, ERROR, WARN, OK
    except ImportError as exc:
        print(f"Could not import this repo's dependencies ({exc}). "
              f"Run this from an environment with requirements.txt installed.")
        return 1

    findings = check_config(probe_services=not args.offline)
    if args.env_file:
        findings = check_env_file(args.env_file) + findings

    marker = {ERROR: "FAIL", WARN: "WARN", OK: "ok  "}
    errors = warnings = 0
    for level, subject, detail in findings:
        if level == ERROR:
            errors += 1
        elif level == WARN:
            warnings += 1
        if level == OK and not args.verbose:
            continue
        print(f"  [{marker[level]}] {subject:<28} {detail}")

    checked = len(findings)
    if not args.verbose and errors == 0 and warnings == 0:
        print(f"  all {checked} checks passed")
    print(f"\n{checked} checks, {errors} failing, {warnings} warning"
          f"{'' if warnings == 1 else 's'}"
          f"{'' if args.verbose else '  (--verbose to list the ones that passed)'}")
    return 1 if errors else 0


def cmd_logs(args):
    if args.name not in SERVICE_ORDER + OPTIONAL_SERVICES:
        print(f"Unknown service: {args.name}. Known: {', '.join(SERVICE_ORDER + OPTIONAL_SERVICES)}")
        return 1
    return subprocess.run(["docker", "compose", "logs", "-f"], cwd=_service_dir(args.name)).returncode


def main():
    # Without this, this script's own print() calls can appear after a child subprocess's
    # output even though they were written first: Python buffers stdout when it isn't a
    # terminal (e.g. piped, or under this environment's tool wrapper), while the child writes
    # straight to the shared file descriptor. Line-buffering keeps the interleaving honest.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # older Python; harmless to skip

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser(
        "init", help="Write every .env this stack needs: derived URLs, generated secrets, "
                     "and a list of the credentials only you can supply.")
    init_parser.add_argument("--host", default="localhost",
                             help="Host the services will be reached at. Default localhost.")
    init_parser.add_argument("--force-urls", action="store_true",
                             help="Overwrite service URLs even where they were set by hand.")

    sub.add_parser("check", help="Prerequisites, hardware report and per-service readiness. Changes nothing.")

    up_parser = sub.add_parser("up", help="Bring every service up, in order, then the application.")
    up_parser.add_argument("--only", choices=SERVICE_ORDER + OPTIONAL_SERVICES, help="Bring up just one service.")
    up_parser.add_argument("--skip", choices=SERVICE_ORDER + OPTIONAL_SERVICES, help="Skip bringing up one service.")
    up_parser.add_argument("--host", default="localhost",
                           help="Passed to init when env files are missing. Default localhost.")
    up_parser.add_argument("--no-init", action="store_true",
                           help="Fail rather than generating missing .env files.")

    stop_parser = sub.add_parser("stop", help="Pause every container, keep them - a later `up` just restarts.")
    stop_parser.add_argument("--only", choices=SERVICE_ORDER + OPTIONAL_SERVICES, help="Pause just one service.")

    down_parser = sub.add_parser("down", help="Tear everything down, reverse order.")
    down_parser.add_argument("--only", choices=SERVICE_ORDER + OPTIONAL_SERVICES, help="Tear down just one service.")

    sub.add_parser("status", help="Live reachability of every component, same probes as the admin panel.")

    config_parser = sub.add_parser("config", help="Validate configuration: required values, values that must agree, and what the services actually report.")
    config_parser.add_argument("--offline", action="store_true", help="Skip the service probes and check local configuration only.")
    config_parser.add_argument("--verbose", action="store_true", help="List the checks that passed as well as the ones that did not.")
    config_parser.add_argument("--env-file", help="Also check this env file for duplicate keys and for values that never reached the process.")

    logs_parser = sub.add_parser("logs", help="Follow one service's container logs.")
    logs_parser.add_argument("name", choices=SERVICE_ORDER + OPTIONAL_SERVICES)

    args = parser.parse_args()
    handlers = {"init": cmd_init, "check": cmd_check, "up": cmd_up, "stop": cmd_stop, "down": cmd_down,
                "status": cmd_status, "config": cmd_config, "logs": cmd_logs}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
