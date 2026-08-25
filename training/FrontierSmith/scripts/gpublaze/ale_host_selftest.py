#!/usr/bin/env python3
"""ALE-Bench no-docker (host) backend validation suite -- gpublaze.

Repeatable acceptance tests for the bwrap host backend
(scripts/gpublaze/pysite/ale_host_backend.py). NOTHING here touches the GPU.

Subcommands:
  compile      One-shot host compile selftest through the REAL scoring path
               (ale_compile_selftest -> run_compile_container -> bwrap+g++12.2).
  compare      Run compute_score over the fixed 6-problem x 2-sample matrix
               (scripts/gpublaze/ale_host_test_assets/samples.json) with the
               CURRENT backend and dump metrics to JSON.
  report       Diff a host vs docker compare-JSON per problem/sample/field.
  fault        Failure-semantics suite (dockerd dead -> AleInfraError; host
               rootfs missing -> AleInfraError; bwrap missing -> AleInfraError;
               host healthy -> no raise). Each subcase runs in a subprocess.
  concurrency  4-way x 8 samples through the host backend; asserts no fd leak,
               no leftover /tmp ale artifacts, no zombie child processes.
  all          compile -> fault -> concurrency -> compare(host) ->
               compare(docker, subprocess) -> report. The one-command recheck:

    cd /srv/home/bohanlyu/innovation_proior/training/FrontierSmith
    .venv-gpublaze/bin/python scripts/gpublaze/ale_host_selftest.py all

Env honored: ALE_BENCH_DATA / ALE_BENCH_CACHE (default to the in-repo paths),
ALEBENCH_LITE (default true = production default), ALEBENCH_NUM_WORKERS
(default 4 = production), ALE_BENCH_HOST_CPUS / ALE_BENCH_HOST_ROOTFS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FS_ROOT = SCRIPT_DIR.parents[1]
PYSITE = SCRIPT_DIR / "pysite"
ASSETS = SCRIPT_DIR / "ale_host_test_assets"
SAMPLES_JSON = ASSETS / "samples.json"
RESULTS_DIR = ASSETS / "results"

PROBLEMS = ["ahc008", "ahc015", "ahc011", "ahc046", "ahc024", "ahc039"]
SCORE_FIELDS = ["score", "performance", "rank", "overall_absolute_score", "overall_relative_score"]


def base_env() -> dict:
    env = dict(os.environ)
    env.setdefault("ALE_BENCH_DATA", str(FS_ROOT / "data" / "alebench" / "local_data"))
    env.setdefault("ALE_BENCH_CACHE", str(FS_ROOT / ".cache" / "ale-bench"))
    env.setdefault("ALEBENCH_LITE", "true")
    env.setdefault("ALEBENCH_NUM_WORKERS", "4")
    # pysite on PYTHONPATH so subprocess sitecustomize can activate backends.
    pp = env.get("PYTHONPATH", "")
    if str(PYSITE) not in pp:
        env["PYTHONPATH"] = f"{pp}:{PYSITE}" if pp else str(PYSITE)
    return env


def activate_host_backend_in_process() -> None:
    """sitecustomize only auto-runs when pysite is on PYTHONPATH at interpreter
    start; when THIS script is invoked directly it was not, so install now."""
    if str(PYSITE) not in sys.path:
        sys.path.insert(0, str(PYSITE))
    from verl.utils.reward_score.ale_host import install_host_backend

    install_host_backend()


def load_samples() -> dict:
    return json.loads(SAMPLES_JSON.read_text())


# ---------------------------------------------------------------- compile --

def cmd_compile(_args) -> int:
    os.environ.update({k: v for k, v in base_env().items() if k.startswith("ALE")})
    activate_host_backend_in_process()
    from verl.utils.reward_score.ale_selftest import ale_compile_selftest, reset_selftest_cache

    reset_selftest_cache()
    t0 = time.time()
    ale_compile_selftest(RuntimeError, force=True)
    print(f"[compile] HOST compile selftest PASSED in {time.time()-t0:.1f}s "
          "(real g++ via bwrap sandbox, run_compile_container path)")
    return 0


# ---------------------------------------------------------------- compare --

def cmd_compare(args) -> int:
    env = base_env()
    for k, v in env.items():
        if k.startswith("ALE"):
            os.environ[k] = v
    backend = args.backend
    if backend == "host":
        os.environ["ALE_BENCH_CONTAINER_BACKEND"] = "host"
        activate_host_backend_in_process()
    else:
        os.environ.pop("ALE_BENCH_CONTAINER_BACKEND", None)
        os.environ["ALE_BENCH_DOCKER_ROOT_USER"] = "1"  # rootless remap patch

    from verl.utils.reward_score.alebench import compute_score

    samples = load_samples()
    results = {}
    for prob in PROBLEMS:
        results[prob] = {}
        for kind in ("good", "broken"):
            text = samples[prob][kind]["text"]
            t0 = time.time()
            try:
                r = compute_score("alebench", text, prob)
                err = None
            except Exception as exc:  # comparison run must not die mid-matrix
                r, err = None, f"{type(exc).__name__}: {exc}"
            dt = time.time() - t0
            results[prob][kind] = {"metrics": r, "error": err, "seconds": round(dt, 1)}
            tag = "ERR " if err else "ok  "
            abs_s = r["overall_absolute_score"] if r else "-"
            perf = r["performance"] if r else "-"
            print(f"[compare:{backend}] {tag}{prob} {kind:6s} abs={abs_s} perf={perf} ({dt:.0f}s)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out or RESULTS_DIR / f"compare_{backend}.json")
    out.write_text(json.dumps({"backend": backend, "results": results}, indent=1))
    print(f"[compare:{backend}] -> {out}")
    return 0


# ----------------------------------------------------------------- report --

def _fmt(v):
    return "-" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))


def cmd_report(args) -> int:
    host = json.loads(Path(args.host).read_text())["results"]
    docker = json.loads(Path(args.docker).read_text())["results"]
    tol = args.tol
    rows, bad = [], 0
    for prob in PROBLEMS:
        for kind in ("good", "broken"):
            h, d = host[prob][kind], docker[prob][kind]
            if h["error"] or d["error"]:
                ok = (h["error"] is not None) == (d["error"] is not None)
                rows.append((prob, kind, "errors", h["error"], d["error"], ok))
                bad += 0 if ok else 1
                continue
            for field in SCORE_FIELDS:
                hv, dv = h["metrics"].get(field), d["metrics"].get(field)
                if hv is None or dv is None:
                    ok = hv is None and dv is None
                else:
                    ok = abs(float(hv) - float(dv)) <= tol
                if not ok:
                    bad += 1
                rows.append((prob, kind, field, _fmt(hv), _fmt(dv), ok))
    print(f"{'problem':8s} {'sample':7s} {'field':24s} {'host':>14s} {'docker':>14s}  match")
    for prob, kind, field, hv, dv, ok in rows:
        print(f"{prob:8s} {kind:7s} {field:24s} {str(hv):>14s} {str(dv):>14s}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n[report] {'ALL EQUAL' if bad == 0 else f'{bad} MISMATCHES'} (tol={args.tol})")
    return 0 if bad == 0 else 1


# ------------------------------------------------------------------ fault --

_CHILD_SNIPPET = r"""
import sys
from verl.utils.reward_score.alebench import compute_score, AleInfraError
code = "```cpp\n#include <bits/stdc++.h>\nint main(){std::cout<<1<<std::endl;return 0;}\n```"
try:
    r = compute_score("alebench", code, "ahc046")
except AleInfraError as e:
    print("ALEINFRA_RAISED")
    sys.exit(0)
except Exception as e:
    print(f"OTHER_EXCEPTION: {type(e).__name__}: {e}")
    sys.exit(3)
print("SCORED_OK", r["overall_absolute_score"] if r else None)
sys.exit(0)
"""


def _run_child(name: str, env_overrides: dict, expect: str, timeout: int = 600) -> bool:
    env = base_env()
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SNIPPET],
        env=env, capture_output=True, text=True, timeout=timeout, cwd=FS_ROOT,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    got = "ALEINFRA_RAISED" if "ALEINFRA_RAISED" in out else (
        "SCORED_OK" if "SCORED_OK" in out else f"unexpected(rc={proc.returncode})")
    ok = expect in out
    print(f"[fault] {name:34s} expect={expect:18s} got={got:18s} {'OK' if ok else 'FAIL'}")
    if not ok:
        tail = "\n".join(out.splitlines()[-15:])
        print(f"  --- child output tail ---\n{tail}\n  -------------------------")
    return ok


def cmd_fault(_args) -> int:
    results = []
    # 1) docker backend, dead daemon -> AleInfraError (loud, not zero).
    results.append(_run_child(
        "docker-dead-daemon",
        {"ALE_BENCH_CONTAINER_BACKEND": "docker",
         "ALE_BENCH_DOCKER_ROOT_USER": "1",
         "DOCKER_HOST": "unix:///nonexistent-ale-host-test.sock"},
        "ALEINFRA_RAISED"))
    # 2) host backend, rootfs missing -> AleInfraError (loud, not zero).
    results.append(_run_child(
        "host-missing-rootfs",
        {"ALE_BENCH_CONTAINER_BACKEND": "host",
         "ALE_BENCH_HOST_ROOTFS": "/nonexistent-ale-host-rootfs"},
        "ALEINFRA_RAISED"))
    # 3) host backend, bwrap missing (empty PATH) -> AleInfraError.
    results.append(_run_child(
        "host-missing-bwrap",
        {"ALE_BENCH_CONTAINER_BACKEND": "host",
         "PATH": "/nonexistent-path-dir"},
        "ALEINFRA_RAISED"))
    # 4) host backend, healthy -> scores without raising.
    results.append(_run_child(
        "host-healthy",
        {"ALE_BENCH_CONTAINER_BACKEND": "host"},
        "SCORED_OK"))
    ok = all(results)
    print(f"[fault] {'ALL OK' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


# ------------------------------------------------------------ concurrency --

def _tmp_ale_entries() -> set:
    """/tmp entries owned by us (the shared box has other users' churn)."""
    out = set()
    tmp = Path("/tmp")
    uid = os.getuid()
    try:
        for e in tmp.iterdir():
            try:
                if e.lstat().st_uid == uid and (
                    "ale" in e.name.lower() or e.name.startswith("tmp")
                ):
                    out.add(e.name)
            except OSError:
                pass
    except OSError:
        pass
    return out


def _fd_count() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


def _child_processes() -> list:
    me = os.getpid()
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            rest = stat.rpartition(")")[2].split()
            state, ppid = rest[0], int(rest[1])
            if ppid == me:
                out.append((p.name, state))
        except (OSError, ValueError, IndexError):
            pass
    return out


def cmd_concurrency(args) -> int:
    env = base_env()
    for k, v in env.items():
        if k.startswith("ALE"):
            os.environ[k] = v
    os.environ["ALE_BENCH_CONTAINER_BACKEND"] = "host"
    activate_host_backend_in_process()

    from concurrent.futures import ThreadPoolExecutor

    from verl.utils.reward_score.alebench import compute_score

    samples = load_samples()
    # 8 cheap samples: ahc046 (batch 15 cases) good/wa, ahc024 (batch 15) good/tle,
    # ahc039 (batch 15) good/wa, ahc008 CE, ahc011 CE. CE samples return fast.
    plan = [
        ("ahc046", "good"), ("ahc046", "broken"), ("ahc024", "good"), ("ahc024", "broken"),
        ("ahc039", "good"), ("ahc039", "broken"), ("ahc008", "broken"), ("ahc011", "broken"),
    ]

    tmp_before, fd_before = _tmp_ale_entries(), _fd_count()
    t0 = time.time()
    errors = []

    def work(item):
        prob, kind = item
        return prob, kind, compute_score("alebench", samples[prob][kind]["text"], prob)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(work, it) for it in plan]
        for f in futs:
            try:
                prob, kind, r = f.result()
                print(f"[conc] ok {prob} {kind} abs={r['overall_absolute_score']} perf={r['performance']}", flush=True)
            except Exception as exc:
                errors.append(repr(exc))
                print(f"[conc] ERR {exc!r}", flush=True)
    dt = time.time() - t0

    time.sleep(2)  # let any straggler cleanup land
    tmp_after, fd_after = _tmp_ale_entries(), _fd_count()
    leaked_tmp = tmp_after - tmp_before
    children = _child_processes()
    zombies = [c for c in children if c[1] == "Z"]
    ok = not errors and not leaked_tmp and not zombies and fd_after <= fd_before + 8
    print(f"[conc] 8 samples / 4 workers in {dt:.0f}s | errors={len(errors)} "
          f"fd {fd_before}->{fd_after} tmp_leak={sorted(leaked_tmp) or 'none'} "
          f"children={children or 'none'} zombies={zombies or 'none'}")
    print(f"[conc] {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


# -------------------------------------------------------------------- all --

def cmd_all(args) -> int:
    rc = 0
    rc |= cmd_compile(args)
    rc |= cmd_fault(args)
    rc |= cmd_concurrency(args)

    # host comparison in-process (backend already installable here)
    class _A:
        backend = "host"
        out = str(RESULTS_DIR / "compare_host.json")
    rc |= cmd_compare(_A())

    # docker comparison in a subprocess (cannot un-install the host patch in
    # this interpreter; a fresh one with backend=docker uses native docker).
    env = base_env()
    env["ALE_BENCH_CONTAINER_BACKEND"] = "docker"
    env["ALE_BENCH_DOCKER_ROOT_USER"] = "1"
    print("[all] running docker-backend comparison in a subprocess (native rootless docker)...")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "compare", "--backend", "docker",
         "--out", str(RESULTS_DIR / "compare_docker.json")],
        env=env, cwd=FS_ROOT,
    )
    rc |= proc.returncode

    class _R:
        host = str(RESULTS_DIR / "compare_host.json")
        docker = str(RESULTS_DIR / "compare_docker.json")
        tol = args.tol
    rc |= cmd_report(_R())
    print(f"[all] {'ALL GREEN' if rc == 0 else 'FAILURES (see above)'}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=1e-6, help="score equality tolerance (default 1e-6)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compile")
    c = sub.add_parser("compare")
    c.add_argument("--backend", choices=["host", "docker"], required=True)
    c.add_argument("--out", default=None)
    r = sub.add_parser("report")
    r.add_argument("--host", required=True)
    r.add_argument("--docker", required=True)
    sub.add_parser("fault")
    sub.add_parser("concurrency")
    sub.add_parser("all")
    args = ap.parse_args()
    return {
        "compile": cmd_compile,
        "compare": cmd_compare,
        "report": cmd_report,
        "fault": cmd_fault,
        "concurrency": cmd_concurrency,
        "all": cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
