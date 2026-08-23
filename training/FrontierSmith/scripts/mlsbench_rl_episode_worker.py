#!/usr/bin/env python3
"""Per-episode MLS-Bench environment worker (JSON-RPC over stdio).

One process == one episode == one task. This mirrors how the eval harness runs
`mlsbench agent` in its own subprocess per task, which matters because task
mid_edit.py / parser.py files do `sys.path.insert + import dgp` from per-task
holdout dirs — running two different tasks in one process poisons sys.modules
(verified: causal-* / ml-* cross-contamination). Process-per-episode also gives
the RL agent loop a clean kill path for hung Apptainer tests.

Run under a python that has mlsbench deps (default: the eval conda python).
Protocol: one JSON object per line on stdin -> one `@@RSP@@{json}` line on the
RESERVED protocol fd (original stdout). All prints (mlsbench chatter) are
redirected to stderr so they can never corrupt the protocol stream.

Requests:
  {"cmd":"init","task":T,"model_name":M,"max_steps":N,"max_tests":N,
   "save_path":P,"workspace_root":P,"data_root":P,"use_replace":true,
   "budget_banner":true}
      -> {"ok":true,"system_prompt":S,"initial_prompt":S,"tool_schemas":[...],
          "workspace_dir":P,"log_dir":P,"max_steps":N,"max_tests":N}
      max_steps/max_tests are PER-EPISODE (the RL loop passes the row's
      extra_info.budget here). A "BUDGET (HARD-ENFORCED)" banner is prepended
      to initial_prompt (budget_banner=false disables). The same numbers are
      also woven into the prompt's "## Your Budget" section by mlsbench itself.
  {"cmd":"dispatch","name":TOOL,"args":{...}}
      -> {"ok":true,"result":S,"step_count":N,"test_count":N,"done":B,
          "budget_exhausted":B}
      HARD enforcement: once step_count >= max_steps every further dispatch is
      REJECTED (not executed) with budget_exhausted=true; test() calls beyond
      max_tests are rejected inside mlsbench ToolManager.test itself.
  {"cmd":"finalize"}   -> {"ok":true,"reward":F,"detail":{...}}
  {"cmd":"cleanup","keep_workspace":B} -> {"ok":true}
  {"cmd":"quit"}       -> {"ok":true}  (then exits)

The WorkerClient class at the bottom is the blocking client used by both the
verl agent loop and the parquet prep script (imported via importlib from this
file's path so neither needs mlsbench on its own import path).
"""
from __future__ import annotations

import json
import os
import shutil
import sys


def _serve() -> int:
    # Host-RAM guard (2026-08-11): every test() runs MODEL-GENERATED code (e.g. a
    # dimred solution doing O(n^2) pairwise distances on MNIST) with no memory
    # bound -- caught red-handed at 140G+ PER bench process, 89G->374G in 45s,
    # the true killer behind all six rlv3 host OOMs. RLIMIT_AS set here is
    # inherited by every descendant (mlsbench ToolManager.test children included),
    # so a runaway solution dies with MemoryError (test fails, episode scores
    # low) instead of OOM-killing the training node. 0 disables.
    _mem_mb = int(os.environ.get("MLS_RL_EPISODE_MEM_MB", "16384"))
    if _mem_mb > 0:
        import resource

        _lim = _mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (_lim, _lim))

    # Reserve original stdout for the protocol; everything print()ed after this
    # (mlsbench chatter, warnings) goes to stderr.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    mls_root = os.environ.get("MLS_RL_MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev")
    sys.path.insert(0, os.path.join(mls_root, "src"))
    os.environ.setdefault("MLSBENCH_SCHEDULER_MANAGED", "1")
    os.environ.setdefault("MLSBENCH_NO_PREBUILT", "1")

    agent = None
    max_steps_lim = 8  # per-episode action budget, set at init (HARD-enforced in dispatch)
    max_tests_lim = 1  # per-episode test budget, enforced inside ToolManager.test()

    def respond(obj: dict) -> None:
        proto.write("@@RSP@@" + json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            respond({"ok": False, "error": f"bad request json: {e}"})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "init":
                from mlsbench.agent.interactive import InteractiveAgent

                class _Headless(InteractiveAgent):
                    def get_action(self, messages):  # pragma: no cover
                        raise RuntimeError("worker is driven externally")

                max_steps_lim = int(req.get("max_steps", 8))
                max_tests_lim = int(req.get("max_tests", 1))
                global_config = {
                    "model": req["model_name"],
                    "max_steps": max_steps_lim,
                    "max_tests": max_tests_lim,
                    "save_path": req.get("save_path", ""),
                    "data_root": req.get("data_root", ""),
                    "seeds": [42],
                    "container_runtime": "apptainer",
                    "use_replace": bool(req.get("use_replace", True)),
                    "providers": {"vllm": {"api_key": "EMPTY", "base_url": "http://127.0.0.1:9/v1"}},
                    "thinking": {"enabled": False},
                }
                agent = _Headless(req["task"], global_config, workspace_root=req.get("workspace_root") or None)
                agent.setup_workspace()
                agent.tools.capture_template_state()
                initial_prompt = agent.build_initial_prompt()
                if req.get("budget_banner", True):
                    # Prominent, agent-visible budget statement at the very top of
                    # the task prompt (in addition to mlsbench's own "## Your
                    # Budget" section, which uses the same numbers).
                    banner = (
                        "== BUDGET (HARD-ENFORCED) ==\n"
                        f"You may take at most {max_steps_lim} actions (tool calls) in "
                        f"total and call the test tool at most {max_tests_lim} time(s). "
                        "Exceeding either limit ends the episode immediately. "
                        "Plan accordingly.\n\n"
                    )
                    initial_prompt = banner + initial_prompt
                try:
                    agent.logger.log_initial_prompt(initial_prompt)
                except Exception:
                    pass
                respond({
                    "ok": True,
                    "system_prompt": agent.system_prompt,
                    "initial_prompt": initial_prompt,
                    "tool_schemas": agent._tool_schemas,
                    "workspace_dir": str(agent.tools.workspace_task_dir),
                    "log_dir": str(agent.logger.log_dir),
                    "max_steps": max_steps_lim,
                    "max_tests": max_tests_lim,
                })

            elif cmd == "dispatch":
                assert agent is not None, "init first"
                tools = agent.tools
                # HARD per-episode action-budget guard: once the budget is spent,
                # reject EVERY further tool call without executing it (mirrors the
                # eval harness stopping its loop at step_count >= max_steps; the
                # final submission happens in finalize's force-submit).
                if tools.step_count >= max_steps_lim:
                    respond({
                        "ok": True,
                        "result": (
                            f"ERROR: Action budget exhausted "
                            f"({tools.step_count}/{max_steps_lim}). No further tool "
                            "calls are allowed; the episode is over."
                        ),
                        "step_count": tools.step_count,
                        "test_count": tools.test_count,
                        "done": bool(tools.done),
                        "budget_exhausted": True,
                    })
                    continue
                step_num = tools.step_count + 1
                try:
                    agent.logger.log_assistant(step_num, {"name": req["name"], "input": req.get("args", {})})
                except Exception:
                    pass
                try:
                    result = tools.dispatch(req["name"], req.get("args", {}))
                except Exception as e:  # noqa: BLE001 — one bad tool call must not kill the episode
                    result = f"ERROR: tool '{req.get('name')}' crashed: {e}"
                    try:
                        tools.step_count += 1
                    except Exception:
                        pass
                try:
                    agent.logger.log_tool_result(step_num, str(result))
                except Exception:
                    pass
                respond({
                    "ok": True,
                    "result": str(result),
                    "step_count": tools.step_count,
                    "test_count": tools.test_count,
                    "done": bool(tools.done),
                    "budget_exhausted": tools.step_count >= max_steps_lim,
                })

            elif cmd == "finalize":
                assert agent is not None, "init first"
                tools = agent.tools
                try:
                    if not tools.done and tools._test_history:
                        tools.submit(n=-1, _force=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[worker] force submit failed: {e!r}")
                try:
                    tools.record_zero_if_no_finals()
                except Exception as e:  # noqa: BLE001
                    print(f"[worker] record_zero_if_no_finals failed: {e!r}")
                try:
                    agent._write_run_summary()
                except Exception:
                    pass

                from mlsbench.scoring.evaluate import evaluate_task
                model_row = tools._decorated_model_name()
                results = evaluate_task(agent.task_name, model=model_row)
                if not results:
                    rsp = {"ok": True, "reward": 0.0, "detail": {"reason": "no_leaderboard_row"}}
                else:
                    tr = results[0]
                    score = float(tr.score) if tr.score is not None else 0.0
                    rsp = {
                        "ok": True,
                        "reward": score,
                        "detail": {
                            "task_score": score,
                            "settings": [{"name": s.name, "score": s.score} for s in tr.settings],
                            "warnings": list(tr.warnings or [])[:4],
                        },
                    }
                # Prune this episode's rows from the shared per-task leaderboard.csv
                # AFTER scoring (RL generates thousands of episodes; each add()
                # rewrites the whole CSV, so unbounded growth would be O(n^2) and
                # bloat the dev checkout). Only rows whose model EXACTLY matches
                # this episode's unique name are touched; baselines are untouched.
                if os.environ.get("MLS_RL_PRUNE_LEADERBOARD", "1").strip().lower() in ("1", "true", "yes", "on"):
                    try:
                        _prune_leaderboard_rows(agent.leaderboard, model_row)
                    except Exception as e:  # noqa: BLE001
                        print(f"[worker] leaderboard prune failed (non-fatal): {e!r}")
                respond(rsp)

            elif cmd == "cleanup":
                if agent is not None and not req.get("keep_workspace", False):
                    try:
                        wtd = agent.tools.workspace_task_dir
                        if wtd.exists():
                            shutil.rmtree(wtd, ignore_errors=True)
                    except Exception:
                        pass
                respond({"ok": True})

            elif cmd == "quit":
                respond({"ok": True})
                return 0

            else:
                respond({"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            respond({"ok": False, "error": repr(e)})
    return 0


def _prune_leaderboard_rows(leaderboard, model_row: str) -> None:
    """Remove all rows with model == model_row from the per-task leaderboard CSV.

    Uses the Leaderboard's own fcntl lock; WAL is compacted inside the same lock
    so pruned rows can't be resurrected by WAL replay on the next add().
    """
    import csv
    import fcntl

    with leaderboard._locked(fcntl.LOCK_EX):
        fieldnames, rows = leaderboard._read_existing()
        if not rows:
            return
        kept = [r for r in rows if r.get("model") != model_row]
        if len(kept) == len(rows):
            return
        with open(leaderboard.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in kept:
                writer.writerow(row)
        # Drop WAL entries no longer present in the CSV (rule 1 of compact_wal).
        leaderboard.compact_wal()
        print(f"[worker] pruned {len(rows) - len(kept)} leaderboard row(s) for {model_row}")


# ---------------------------------------------------------------------------
# Blocking client (imported via importlib from this file — stdlib only)
# ---------------------------------------------------------------------------


class WorkerClient:
    """Blocking JSON-RPC client for the episode worker subprocess."""

    RSP_PREFIX = "@@RSP@@"

    def __init__(self, worker_python: str, worker_script: str, env: dict | None = None,
                 stderr_path: str | None = None):
        import subprocess

        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        self._stderr_f = open(stderr_path, "a", buffering=1) if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [worker_python, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_f,
            text=True,
            bufsize=1,
            env=full_env,
            start_new_session=True,  # own process group -> killpg reaps apptainer children
        )

    def call(self, **req) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"episode worker exited rc={self.proc.returncode}")
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"episode worker EOF (rc={self.proc.poll()})")
            if line.startswith(self.RSP_PREFIX):
                rsp = json.loads(line[len(self.RSP_PREFIX):])
                if not rsp.get("ok"):
                    raise RuntimeError(f"worker error for cmd={req.get('cmd')}: {rsp.get('error')}")
                return rsp
            # anything else on the protocol fd is unexpected chatter; ignore

    def kill(self) -> None:
        import signal

        try:
            if self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        if self._stderr_f not in (None,) and hasattr(self._stderr_f, "close"):
            try:
                self._stderr_f.close()
            except Exception:
                pass

    def quit(self) -> None:
        try:
            self.call(cmd="quit")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            self.kill()
            return
        if hasattr(self._stderr_f, "close"):
            try:
                self._stderr_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(_serve())
