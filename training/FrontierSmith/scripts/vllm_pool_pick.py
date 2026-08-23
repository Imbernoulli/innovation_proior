#!/usr/bin/env python3
"""Pick a live vLLM backend from the pool registry and print its base URL.

Used by CPU-only eval jobs to find a GPU server:

    export VLLM_BASE_URL="$(python3 scripts/vllm_pool_pick.py --tag my-model)"

Design notes:

* Liveness is decided by the entry's mtime, which the serving job refreshes every
  60s. A backend whose job was SIGKILLed at the wall clock never runs its cleanup
  trap, so a stale file would otherwise be handed out forever.
* Every candidate is probed with a real HTTP request before being returned. The
  heartbeat proves the wrapper script is alive; only /v1/models proves vLLM is
  actually answering.
* Spreading matters: if 30 CPU jobs all start within a minute and each picks
  "the first healthy backend", they pile onto one server while the others idle.
  Selection is therefore randomised over the healthy set, seeded per-caller, and
  --least-loaded can instead consult each backend's running-request count so the
  choice reflects actual load rather than luck.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REGISTRY = Path(
    os.environ.get(
        "VLLM_POOL_REGISTRY",
        "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/.cache/vllm_pool",
    )
)
STALE_AFTER = 300.0  # 5 min: 5 missed 60s heartbeats


def _probe(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _running_requests(host: str, port: int, timeout: float) -> float:
    """vLLM's Prometheus endpoint: current running requests. Lower is better.

    Returns +inf when unreadable so an unscrapable backend loses to any backend
    we can actually measure, rather than winning with a default of 0.
    """
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=timeout) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith("vllm:num_requests_running"):
                    return float(line.rsplit(" ", 1)[-1])
    except (urllib.error.URLError, OSError, ValueError, IndexError):
        pass
    return float("inf")


def load(registry: Path, tag: str | None, timeout: float) -> list[dict]:
    if not registry.is_dir():
        return []
    now = time.time()
    out = []
    for f in sorted(registry.glob("*.json")):
        try:
            if now - f.stat().st_mtime > STALE_AFTER:
                continue
            entry = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if tag and entry.get("tag") != tag:
            continue
        if _probe(entry["url"], timeout):
            out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="only backends serving this model tag")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--least-loaded", action="store_true",
                    help="pick by vllm:num_requests_running instead of at random")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="seconds to wait for a backend to appear (0 = fail fast)")
    ap.add_argument("--list", action="store_true", help="print all healthy backends and exit")
    args = ap.parse_args()

    deadline = time.time() + args.wait
    while True:
        alive = load(args.registry, args.tag, args.timeout)
        if alive or time.time() >= deadline:
            break
        time.sleep(10)

    if not alive:
        print(
            f"ERROR: no live vLLM backend in {args.registry}"
            + (f" for tag={args.tag}" if args.tag else ""),
            file=sys.stderr,
        )
        return 1

    if args.list:
        for e in alive:
            print(f"{e['tag']}\t{e['url']}\tjob={e.get('job_id')}")
        return 0

    if args.least_loaded and len(alive) > 1:
        pick = min(alive, key=lambda e: _running_requests(e["host"], e["port"], args.timeout))
    else:
        # Seed per caller so concurrently-starting jobs diverge instead of all
        # drawing the same "random" backend from an identical default seed.
        random.seed(f"{os.environ.get('SLURM_JOB_ID', os.getpid())}")
        pick = random.choice(alive)
    print(pick["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
