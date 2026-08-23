"""nb_run.py — Apptainer+Slurm port of NatureBench's Docker orchestration.

Needs NO root, NO user namespaces, NO docker daemon.

  docker run/exec <img>              -> apptainer exec --overlay <dir> <sif>
  mutable image layers               -> apptainer writable overlay dir (per task)
  -v problem:/task/problem:ro        -> --bind ...:ro
  -v ws:/workspace                   -> --bind ws:/workspace (+ identity bind)
  host.docker.internal               -> 127.0.0.1 (apptainer shares host netns)
  --gpus device=N                    -> --nv inside a Slurm GPU allocation
  4h autonomous CLI agent            -> bounded in-container loop (nb_agent.py)

Scoring is 100% official: the repo's unmodified eval_service.py runs the task's
own evaluation/evaluator.py (hidden ground truth, never mounted into the
container) and computes g = dir*(m - m_sota)/|m_sota| from metadata.json.

This module is both a CLI (one task, own eval service) and a library used by
nb_batch.py (many tasks, one shared vLLM + one shared eval service).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NB_ROOT = Path(__file__).resolve().parent.parent
REPO = NB_ROOT / "repo"
DEFAULT_SIF = NB_ROOT / "containers" / "naturebench-base.sif"
DEFAULT_DATA = NB_ROOT / "data" / "naturebench_data" / "tasks"
DEFAULT_EVAL_PY = NB_ROOT / "envs" / "naturebench-eval" / "bin" / "python"


def log(msg: str) -> None:
    print(f"[nb_run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Dockerfile parsing — behavior-equivalent to solve.py:_parse_dockerfile
# (FrontisAI, Apache-2.0), so setup matches the official --skip-build path.
# ---------------------------------------------------------------------------
@dataclass
class DockerfileSpec:
    run_commands: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    copy_srcs: List[Tuple[str, str]] = field(default_factory=list)


def parse_dockerfile(dockerfile: Path) -> DockerfileSpec:
    spec = DockerfileSpec()
    if not dockerfile.exists():
        return spec
    text = dockerfile.read_text(encoding="utf-8").replace("\\\n", " ")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("FROM"):
            continue
        if upper.startswith("RUN "):
            cmd = line[4:].strip()
            if cmd:
                spec.run_commands.append(cmd)
        elif upper.startswith("ENV "):
            rest = line[4:].strip()
            if "=" in rest:
                key, _, val = rest.partition("=")
                spec.env_vars[key.strip()] = val.strip().strip('"')
            else:
                parts = rest.split(None, 1)
                if len(parts) == 2:
                    spec.env_vars[parts[0]] = parts[1].strip('"')
        elif upper.startswith("COPY ") or upper.startswith("ADD "):
            rest = line[5 if upper.startswith("COPY ") else 4:].strip()
            parts = rest.split()
            if len(parts) >= 2:
                spec.copy_srcs.append((parts[0], parts[-1]))
    return spec


def build_setup_script(task_root: Path, setup_extra: Optional[str] = None,
                       use_pip_cache: bool = True) -> Tuple[str, int]:
    """Official skip-build setup script for a task, plus optional repair."""
    spec = parse_dockerfile(task_root / "environment" / "Dockerfile.v3")
    patched = []
    for c in spec.run_commands:
        if "pip install" in c and "--no-build-isolation" not in c:
            c = c.replace("pip install", "pip install --no-build-isolation", 1)
        if use_pip_cache:
            # infra-only: allow the shared wheel cache. Does NOT change which
            # versions pip resolves, only whether wheels are re-downloaded.
            c = c.replace("--no-cache-dir ", "").replace(" --no-cache-dir", "")
        patched.append(c)
    env_exports = [f'export {k}="{v}"' for k, v in spec.env_vars.items()]
    script = " && ".join(env_exports + patched)
    if setup_extra:
        script = (f"({script}) ; echo '--- SETUP_EXTRA ---' ; {setup_extra}"
                  if script else setup_extra)
    return script, len(patched)


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 60.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
class ApptainerTask:
    """One task environment: read-only base SIF + a per-task writable overlay.

    The overlay replaces the Docker image's mutable layers: the task's
    Dockerfile RUN pip-installs land in <overlay>/upper and cost only the files
    they actually add (vs ~8 GB / 90k inodes for a full /opt/py311 copy).
    """

    def __init__(self, sif: Path, task: str, task_root: Path, workspace: Path,
                 overlay_root: Path, gpu: bool, pip_cache: Optional[Path] = None):
        self.sif = sif
        self.task = task
        self.task_root = task_root
        self.workspace = workspace
        self.overlay = overlay_root / task
        self.marker = overlay_root / f"{task}.setup_done"
        self.gpu = gpu
        self.pip_cache = pip_cache

    def ensure_overlay(self) -> None:
        (self.overlay / "upper").mkdir(parents=True, exist_ok=True)
        (self.overlay / "work").mkdir(parents=True, exist_ok=True)

    def _base_cmd(self, extra_binds: List[str] | None = None) -> List[str]:
        cmd = ["apptainer", "exec", "--cleanenv",
               # --env HOME=... is refused by apptainer; host $HOME is quota-tight
               "--home", "/workspace/.home",
               "--overlay", str(self.overlay)]
        if self.gpu:
            cmd.append("--nv")
        binds = [
            f"{self.task_root / 'problem'}:/task/problem:ro",
            f"{self.workspace}:/workspace",
            # identity bind: the host workspace path resolves identically inside
            # the container, so the host output_dir posted to /evaluate is valid
            # on both sides.
            f"{self.workspace}:{self.workspace}",
            f"{NB_ROOT / 'harness'}:/nbharness:ro",
        ] + (extra_binds or [])
        if self.pip_cache:
            self.pip_cache.mkdir(parents=True, exist_ok=True)
            binds.append(f"{self.pip_cache}:/pipcache")
        for b in binds:
            cmd += ["--bind", b]
        cmd += ["--env", "VIRTUAL_ENV=/opt/py311",
                "--env", "PATH=/opt/py311/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "--env", "PYTHONUNBUFFERED=1",
                # base image pip.conf points at a dead NGC index -> 30s retries
                "--env", "PIP_CONFIG_FILE=/dev/null",
                "--env", "PIP_INDEX_URL=https://pypi.org/simple",
                "--env", "NONINTERACTIVE=1",
                "--env", "CI=1",
                ]
        if self.pip_cache:
            cmd += ["--env", "PIP_CACHE_DIR=/pipcache", "--env", "PIP_NO_CACHE_DIR=0"]
        else:
            cmd += ["--env", "PIP_NO_CACHE_DIR=1"]
        cmd.append(str(self.sif))
        return cmd

    def run(self, argv: List[str], timeout: int, log_path: Path | None = None,
            extra_env: Dict[str, str] | None = None,
            extra_binds: List[str] | None = None) -> subprocess.CompletedProcess:
        cmd = self._base_cmd(extra_binds)
        if extra_env:
            flat: List[str] = []
            for k, v in extra_env.items():
                flat += ["--env", f"{k}={v}"]
            cmd = cmd[:-1] + flat + cmd[-1:]   # env flags go before the sif path
        cmd += argv
        log(f"apptainer exec ... {' '.join(shlex.quote(c) for c in argv)[:110]}")
        if log_path:
            with open(log_path, "a") as fh:
                return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                      text=True, timeout=timeout)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def setup(self, script: str, timeout: int, log_path: Path,
              force: bool = False) -> str:
        """Replay the task's Dockerfile RUN commands into the overlay.

        Returns "skipped" | "ran". NOTE: compute nodes have no outbound DNS, so
        this only works on a login node; a matching marker lets compute jobs
        skip it.
        """
        self.ensure_overlay()
        if not script:
            return "skipped"
        if not force and self.marker.exists() and self.marker.read_text() == script:
            log("setup marker matches; skipping Dockerfile RUN replay")
            return "skipped"
        proc = self.run(["/bin/bash", "-c", script], timeout=timeout, log_path=log_path)
        if proc.returncode != 0:
            raise RuntimeError(f"task setup failed (exit {proc.returncode}); see {log_path}")
        self.marker.write_text(script)
        return "ran"


# ---------------------------------------------------------------------------
# Eval service (official, unmodified)
# ---------------------------------------------------------------------------
def start_eval_service(port: int, log_path: Path, eval_python: str,
                       timeout: int = 3600) -> subprocess.Popen:
    log(f"starting official eval_service.py on :{port}")
    proc = subprocess.Popen(
        [eval_python, str(REPO / "eval_service.py"),
         "--host", "127.0.0.1", "--port", str(port), "--timeout", str(timeout)],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(90):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"eval service did not come up; see {log_path}")


def stop_eval_service(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# One task, against an already-running eval service
# ---------------------------------------------------------------------------
def run_task(task: str, *, eval_url: str, batch: str, out_dir: Path,
             data_dir: Path = DEFAULT_DATA, sif: Path = DEFAULT_SIF,
             mode: str = "agent", model: str = "", openai_base_url: str = "",
             ref_solver: Optional[str] = None, gpu: bool = False,
             timeout: int = 1800, setup_timeout: int = 1800,
             max_rounds: int = 6, run_timeout: int = 900,
             setup_extra: Optional[str] = None, skip_setup: bool = False,
             force_setup: bool = False, setup_only: bool = False,
             overlay_root: Optional[Path] = None,
             pip_cache: Optional[Path] = None) -> dict:
    t_task = time.time()
    task_root = data_dir / task
    if not (task_root / "metadata.json").exists():
        raise FileNotFoundError(f"task not downloaded: {task_root}")
    workspace = out_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "output").mkdir(exist_ok=True)
    (workspace / ".home").mkdir(exist_ok=True)
    overlay_root = overlay_root or (out_dir.parent.parent / "_overlays")

    result: dict = {"task": task, "batch": batch, "mode": mode,
                    "model": model if mode == "agent" else "reference",
                    "setup_extra": setup_extra, "result_dir": str(out_dir)}

    if not setup_only:   # setup needs no eval service (login-node phase)
        st, resp = http_json("POST", f"{eval_url}/register", {
            "task_name": task, "data_dir": str(task_root.resolve()),
            "out_dir": str(out_dir.resolve()), "batch_name": batch, "timeout": timeout,
        })
        if st != 200:
            raise RuntimeError(f"/register failed: {resp}")
        log(f"[{task}] /register -> instances={resp.get('instances')}")
        result["instances"] = resp.get("instances")

    ct = ApptainerTask(sif, task, task_root, workspace, overlay_root, gpu, pip_cache)
    script, n_cmds = build_setup_script(task_root, setup_extra,
                                        use_pip_cache=pip_cache is not None)
    t0 = time.time()
    if skip_setup:
        ct.ensure_overlay()
        setup_state = "skipped(flag)"
    else:
        log(f"[{task}] setup: {n_cmds} Dockerfile RUN commands (timeout={setup_timeout}s)")
        setup_state = ct.setup(script, setup_timeout, out_dir / "setup.log", force=force_setup)
    result["setup_state"] = setup_state
    result["setup_seconds"] = round(time.time() - t0, 1)
    log(f"[{task}] setup {setup_state} in {result['setup_seconds']}s")
    if setup_only:
        result["status"] = "setup_only"
        return result

    http_json("POST", f"{eval_url}/start_timer", {"task_name": task, "batch_name": batch})

    if mode == "probe":
        # Cheap per-task validation of the OFFICIAL scoring path: submit the
        # (empty) output dir and see what the task's own evaluator does.
        # Expected: a finite failure penalty (g=-1.0) => registration, metadata
        # SOTA table, evaluator import and scoring all work. An exception or
        # g=None means the task is not usable yet (missing evaluator dep, etc).
        st, ev = http_json("POST", f"{eval_url}/evaluate", {
            "task_name": task, "batch_name": batch,
            "output_dir": str(workspace / "output")}, timeout=3600)
        result["probe_http"] = st
        result["probe_response"] = ev
        log(f"[{task}] probe /evaluate -> status={st} g={ev.get('aggregate_improvement')}")
    elif mode == "reference":
        if not ref_solver:
            raise ValueError("reference mode needs ref_solver")
        shutil.copy(ref_solver, workspace / "run.py")
        env = {"DATA_DIR": "/task/problem/data", "OUTPUT_DIR": str(workspace / "output")}
        proc = ct.run(["python", "/workspace/run.py"], timeout=run_timeout,
                      log_path=out_dir / "reference_run.log", extra_env=env)
        log(f"[{task}] reference run.py exit={proc.returncode}")
        result["ref_exit"] = proc.returncode
        if proc.returncode == 0:
            st, ev = http_json("POST", f"{eval_url}/evaluate", {
                "task_name": task, "batch_name": batch,
                "output_dir": str(workspace / "output")}, timeout=3600)
            log(f"[{task}] /evaluate -> g={ev.get('aggregate_improvement')}")
    else:
        if not openai_base_url:
            raise ValueError("agent mode needs openai_base_url")
        env = {
            "OPENAI_BASE_URL": openai_base_url,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "NB_MODEL": model, "EVAL_SERVICE_URL": eval_url,
            "TASK_NAME": task, "BATCH_NAME": batch,
            "DATA_DIR": "/task/problem/data",
            "OUTPUT_DIR": str(workspace / "output"),
            "NB_MAX_ROUNDS": str(max_rounds), "NB_RUN_TIMEOUT_S": str(run_timeout),
            "NB_AGENT_BUDGET_S": str(timeout),
        }
        log(f"[{task}] bounded agent (model={model}, rounds<={max_rounds}, budget={timeout}s)")
        try:
            proc = ct.run(["python", "/nbharness/nb_agent.py"], timeout=timeout + 300,
                          log_path=out_dir / "agent.log", extra_env=env)
            result["agent_exit"] = proc.returncode
        except subprocess.TimeoutExpired:
            log(f"[{task}] agent hard timeout")
            result["agent_exit"] = "timeout"

    st, best = http_json("GET", f"{eval_url}/best_score?task_name={task}&batch_name={batch}")
    result["best_score_response"] = best
    result["g"] = best.get("best_aggregate_improvement")
    result["total_attempts"] = best.get("total_attempts")
    result["task_seconds"] = round(time.time() - t_task, 1)
    result["status"] = "done"
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    log(f"[{task}] OFFICIAL g={result['g']} attempts={result['total_attempts']} "
        f"({result['task_seconds']}s)")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--sif", default=str(DEFAULT_SIF))
    ap.add_argument("--results-dir", default=str(NB_ROOT / "results"))
    ap.add_argument("--batch", default=None)
    ap.add_argument("--eval-python", default=str(DEFAULT_EVAL_PY))
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--setup-timeout", type=int, default=1800)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--run-timeout", type=int, default=900)
    ap.add_argument("--mode", choices=["agent", "reference", "probe"], default="agent")
    ap.add_argument("--setup-extra", default=None)
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument("--force-setup", action="store_true")
    ap.add_argument("--setup-only", action="store_true")
    ap.add_argument("--ref-solver", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-pip-cache", action="store_true")
    ap.add_argument("--model", default=os.environ.get("NB_MODEL", "qwen35-9b"))
    ap.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    args = ap.parse_args()

    batch = args.batch or time.strftime("nb%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir)
    out_dir = results_dir / batch / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_proc = start_eval_service(args.port, out_dir / "eval_service.log",
                                   args.eval_python, timeout=args.timeout)
    try:
        res = run_task(
            args.task, eval_url=f"http://127.0.0.1:{args.port}", batch=batch,
            out_dir=out_dir, data_dir=Path(args.data_dir), sif=Path(args.sif),
            mode=args.mode, model=args.model, openai_base_url=args.openai_base_url,
            ref_solver=args.ref_solver, gpu=args.gpu, timeout=args.timeout,
            setup_timeout=args.setup_timeout, max_rounds=args.max_rounds,
            run_timeout=args.run_timeout, setup_extra=args.setup_extra,
            skip_setup=args.skip_setup, force_setup=args.force_setup,
            setup_only=args.setup_only, overlay_root=results_dir / "_overlays",
            pip_cache=None if args.no_pip_cache else results_dir / "_pipcache",
        )
        print(f"NB_SCORE {args.task} {res.get('g')}")
        return 0
    finally:
        stop_eval_service(eval_proc)


if __name__ == "__main__":
    sys.exit(main())
