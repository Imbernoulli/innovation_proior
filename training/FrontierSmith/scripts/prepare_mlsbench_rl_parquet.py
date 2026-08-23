#!/usr/bin/env python3
"""Prepare the MLS-Bench dev/test CPU tasks as a VERL RL-training parquet.

Each task's initial conversation is built in an ISOLATED subprocess (the same
episode worker the RL agent loop uses — scripts/mlsbench_rl_episode_worker.py,
run under the eval conda python) because MLS task mid_edit.py/parser.py files
`import dgp` from per-task holdout dirs and poison sys.modules across tasks.

Each row (mirroring prepare_frontiersmith_synth_parquet.py conventions):
  prompt       = [{"role":"system",...},{"role":"user",...}]  — the EXACT initial
                 conversation the MLS InteractiveAgent sees (use_replace edit
                 schema + build_initial_prompt over a fresh workspace). Used by
                 the dataset for length filtering; the agent loop rebuilds the
                 identical prompt from its own fresh per-episode workspace.
  data_source  = "mlsbench_rl"
  agent_name   = "mlsbench_agent"     — selects MLSBenchAgentLoop per sample
  reward_model = {"ground_truth": <task>}   (unused; reward comes from the loop)
  extra_info   = {"task": <task>, "index": i, "uid": "<task>::<tier>",
                  "budget": {"max_tests": N, "max_actions": N, "tier": T}}

TWO-TIER BUDGETS (default): every task is emitted TWICE — once per budget tier
  tight: max_tests=1, max_actions=5      loose: max_tests=3, max_actions=20
The per-row budget is (a) stated in the prompt (worker prepends a
"BUDGET (HARD-ENFORCED)" banner and mlsbench weaves the same numbers into its
"## Your Budget" section, so tiers have DIFFERENT prompts) and (b) hard-enforced
per episode by MLSBenchAgentLoop.resolve_budget -> episode worker.
--single-budget restores the legacy 1-row-per-task output (no extra_info.budget;
episodes then fall back to the MLS_RL_MAX_STEPS/MLS_RL_MAX_TESTS env knobs).

Usage:
  python scripts/prepare_mlsbench_rl_parquet.py                          # 20 CPU tasks x 2 tiers -> train.parquet
  python scripts/prepare_mlsbench_rl_parquet.py --tasks a b --suffix _smoke
  python scripts/prepare_mlsbench_rl_parquet.py --single-budget          # legacy rows
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "mlsbench_rl_episode_worker.py")

# Canonical two-tier budget classes (user directive). Imported by
# scripts/mlsbench_rl_episode_test.py so smokes exercise the SAME dicts.
BUDGET_TIERS = {
    "tight": {"max_tests": 1, "max_actions": 5, "tier": "tight"},
    "loose": {"max_tests": 3, "max_actions": 20, "tier": "loose"},
}

DEFAULT_CPU_TASKS = [
    "causal-discovery-discrete",
    "causal-observational-linear-gaussian",
    "causal-observational-linear-non-gaussian",
    "causal-observational-nonlinear",
    "causal-treatment-effect",
    "ml-active-learning",
    "ml-anomaly-detection",
    "ml-calibration",
    "ml-clustering-algorithm",
    "ml-dimensionality-reduction",
    "ml-ensemble-boosting",
    "ml-missing-data-imputation",
    "ml-selective-deferral",
    "ml-subgroup-calibration-shift",
    "ml-symbolic-regression",
    "mlsys-moe-load-balance",
    "optimization-evolution-strategy",
    "optimization-hyperparameter-search",
    "optimization-multi-objective",
    "optimization-nas",
]


def _load_worker_client_cls():
    spec = importlib.util.spec_from_file_location("mlsbench_rl_episode_worker", WORKER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WorkerClient


def build_row(WorkerClient, task: str, index: int, tmp: Path, args, budget: dict | None = None) -> dict:
    """Build one parquet row; budget=None -> legacy env-knob row (no override)."""
    tier = (budget or {}).get("tier", "global")
    max_steps = int(budget["max_actions"]) if budget else args.max_steps
    max_tests = int(budget["max_tests"]) if budget else args.max_tests
    client = WorkerClient(
        worker_python=args.worker_python,
        worker_script=WORKER_SCRIPT,
        env={"MLS_RL_MLSBENCH_ROOT": args.mlsbench_root},
        stderr_path=str(tmp / f"{task}.{tier}.worker.log"),
    )
    try:
        init = client.call(
            cmd="init",
            task=task,
            model_name="vllm/parquet-prep",
            max_steps=max_steps,
            max_tests=max_tests,
            save_path=str(tmp / "saves"),
            workspace_root=str(tmp / "ws"),
            data_root=args.data_root,
            use_replace=True,
        )
        client.call(cmd="cleanup", keep_workspace=False)
    finally:
        client.quit()

    prompt = [
        {"role": "system", "content": init["system_prompt"]},
        {"role": "user", "content": init["initial_prompt"]},
    ]
    extra_info = {"task": task, "index": index}
    if budget is not None:
        extra_info["uid"] = f"{task}::{budget['tier']}"
        extra_info["budget"] = dict(budget)
    return {
        "prompt": prompt,
        "data_source": "mlsbench_rl",
        "agent_name": "mlsbench_agent",
        "reward_model": {"ground_truth": task},
        "extra_info": extra_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "mlsbench_rl"))
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _smoke")
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("MLS_RL_MAX_STEPS", "8")))
    ap.add_argument("--max-tests", type=int, default=int(os.environ.get("MLS_RL_MAX_TESTS", "1")))
    ap.add_argument("--mlsbench-root", default=os.environ.get(
        "MLS_RL_MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev"))
    ap.add_argument("--data-root", default=os.environ.get(
        "MLS_RL_DATA_ROOT", "/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench/vendor/data"))
    ap.add_argument("--worker-python", default=os.environ.get(
        "MLS_RL_WORKER_PYTHON", "/home/bl3615/miniconda3/bin/python"))
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "models" / "Qwen3.5-9B"),
                    help="tokenizer dir for prompt-token accounting ('' to skip)")
    ap.add_argument("--single-budget", action="store_true",
                    help="legacy 1 row/task without extra_info.budget (env-knob episodes)")
    args = ap.parse_args()

    import pandas as pd

    tasks = list(args.tasks) if args.tasks else list(DEFAULT_CPU_TASKS)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="mls_rl_prep_"))

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"[prep] tokenizer load failed ({e}); skipping token counts")

    # Row plan: (task, budget|None). Default = 2 rows/task (tight + loose).
    if args.single_budget:
        row_specs = [(task, None) for task in tasks]
    else:
        row_specs = [(task, BUDGET_TIERS[tier]) for task in tasks for tier in ("tight", "loose")]

    WorkerClient = _load_worker_client_cls()
    rows, report = [], []
    for i, (task, budget) in enumerate(row_specs):
        tier = (budget or {}).get("tier", "global")
        try:
            row = build_row(WorkerClient, task, i, tmp, args, budget=budget)
        except Exception as e:  # noqa: BLE001
            print(f"[prep] FAILED {task} [{tier}]: {e!r}  (worker log: {tmp / f'{task}.{tier}.worker.log'})")
            continue
        n_chars = sum(len(m["content"]) for m in row["prompt"])
        n_tok = None
        if tok is not None:
            text = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
            n_tok = len(tok.encode(text, add_special_tokens=False))
        rows.append(row)
        report.append({"task": task, "tier": tier, "chars": n_chars, "tokens": n_tok})
        print(f"[prep] {task:45s} [{tier:6s}] chars={n_chars:7d} tokens={n_tok}")

    if not rows:
        print(f"[prep] no rows built; worker logs under {tmp}")
        return 1

    df = pd.DataFrame(rows)
    train_path = out_dir / f"train{args.suffix}.parquet"
    df.to_parquet(train_path)
    df.to_parquet(out_dir / f"full{args.suffix}.parquet")
    (out_dir / f"report{args.suffix}.json").write_text(json.dumps(report, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    toks = [r["tokens"] for r in report if r["tokens"]]
    tier_counts: dict[str, int] = {}
    for r in report:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
    print(f"[prep] wrote {len(df)} rows -> {train_path}  (by tier: {tier_counts})")
    if toks:
        print(f"[prep] prompt tokens: max={max(toks)} min={min(toks)}  (MAX_PROMPT_LENGTH must exceed max; "
              f"note: +~2k tokens for tool schemas at rollout time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
