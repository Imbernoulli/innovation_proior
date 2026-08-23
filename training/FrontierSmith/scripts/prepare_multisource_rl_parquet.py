#!/usr/bin/env python
"""Build the MULTI-SOURCE RL train parquet: synth + research + MLS in ONE file.

Sources (all individually verified single-source RL arms):
  1. frontiersmith_synth  data/frontiersmith_synth_dedup1300/train.parquet (1300 rows)
     single-turn; reward = sandbox harness (isorun bwrap/apptainer), data_source dispatch.
  2. frontiercs_research  data/frontiercs/research.parquet (64 CPU rows)
     single-turn; reward = official evaluator subprocess, data_source dispatch.
  3. mlsbench_rl          data/mlsbench_rl/train_full84.parquet (168 rows = 84 tasks x
     tight/loose budget tiers); agentic episodes via agent_name=mlsbench_agent
     (MLSBenchAgentLoop); reward computed in-loop -> AgentLoopOutput.reward_score.

MIXING MECHANISM (zero verl changes needed):
  verl's rollout is ALWAYS the AgentLoopManager (async mode; sync is deprecated).
  Routing is PER ROW via the `agent_name` non-tensor column
  (agent_loop.py:_run_agent_loop asserts agent_name in the registry). When the
  column is absent verl fills `single_turn_agent` for every row -- so a MIXED
  parquet only needs an explicit agent_name per row:
    * synth/research rows -> "single_turn_agent"  (built-in @register)
    * MLS rows            -> "mlsbench_agent"     (registered via
      config/mlsbench_agent_loop.yaml passed as agent_loop_config_path)
  Rewards: MLS rows are scored in-loop (reward_score set); single-turn rows have
  reward_score=None so the streaming RewardLoopWorker computes them via
  default_compute_score, dispatching on data_source (frontiersmith_synth /
  frontiercs_research). agent_loop.py:_postprocess writes ALL scores into
  rm_scores (one scalar at each row's last response token). With
  FS_PERTASK_REWARD_NORM=1 the single-turn scorers' 0-100 scale is normalized to
  [0,1], matching the MLS task-score scale.

RATIO: natural counts by default (1300 + 64 + 168 = 1532). GRPO samples batches
uniformly over rows, so the per-batch source mix follows row proportions
(~85% / 4% / 11%). Pass --balance to upsample research x N / MLS x N
(defaults research-x=20 -> 1280, mls-x=8 -> 1344: roughly equal thirds).

Also emits report.json (rows per source, multipliers) and, with --smoke, an
8-row train_smoke.parquet (4 synth + 2 research + 2 cheap tight-budget MLS) so a
TRAIN_BATCH_SIZE=8 one-step smoke covers every source deterministically.

Usage:
  .venv-vllm023/bin/python scripts/prepare_multisource_rl_parquet.py [--balance]
      [--research-x N] [--mls-x N] [--smoke] [--seed 0] [--out-dir data/multisource_rl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "frontiersmith_synth": "single_turn_agent",
    "frontiercs_research": "single_turn_agent",
    "mlsbench_rl": "mlsbench_agent",
}

# smoke picks: research problems with replay-verified nonzero scores from prior
# 9B evals; MLS rows = the two cheapest tight-tier prompts (7.4k / 6.4k tokens).
SMOKE_RESEARCH_GT = [
    "cant_be_late/low_availability_loose_deadline_small_overhead",
    "cant_be_late/low_availability_loose_deadline_large_overhead",
]
SMOKE_MLS_UIDS = [
    "ml-clustering-algorithm_tv4::tight",
    "optimization-convex-concave_train::tight",
]


def load(path: Path, expect_source: str) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    srcs = set(df["data_source"].unique())
    assert srcs == {expect_source}, f"{path}: data_source {srcs} != {expect_source}"
    for i, rm in enumerate(df["reward_model"]):
        assert rm and rm.get("ground_truth"), f"{path} row {i}: empty ground_truth"
    # fill agent_name for single-turn sources; MLS parquet already carries it
    agent = EXPECTED[expect_source]
    if "agent_name" not in df.columns:
        df["agent_name"] = agent
    else:
        got = set(df["agent_name"].unique())
        assert got == {agent}, f"{path}: agent_name {got} != {agent}"
    df["extra_info"] = df["extra_info"].apply(lambda e: dict(e) if e is not None else {})
    return df[["prompt", "data_source", "agent_name", "reward_model", "extra_info"]]


def finalize(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle and assign a GLOBAL unique extra_info.index (verl uses it for
    trajectory/trace grouping; per-source indices would collide, and rows whose
    extra_info lacks the key would read back None once the struct schemas union)."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["extra_info"] = [dict(e, index=i) for i, e in enumerate(df["extra_info"])]
    return df


def report_mix(df: pd.DataFrame, label: str) -> dict:
    counts = df["data_source"].value_counts().to_dict()
    total = int(len(df))
    mix = {k: {"rows": int(v), "frac": round(v / total, 4)} for k, v in counts.items()}
    print(f"[{label}] {total} rows")
    for k, v in sorted(mix.items()):
        print(f"    {k:24s} {v['rows']:5d}  ({v['frac']:.1%})")
    agents = df["agent_name"].value_counts().to_dict()
    print(f"    agent_name: { {k: int(v) for k, v in agents.items()} }")
    return {"total": total, "per_source": mix, "agent_name": {k: int(v) for k, v in agents.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", default=str(ROOT / "data/frontiersmith_synth_dedup1300/train.parquet"))
    ap.add_argument("--research", default="", help="DISABLED by default 2026-08-04 (user): research is EVAL-ONLY -- its 64 problems ARE the eval set (no held-out), so training on them would poison the Research benchmark. Pass an explicit path to re-enable.")
    ap.add_argument("--mls", default=str(ROOT / "data/mlsbench_rl/train_full84.parquet"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/multisource_rl"))
    ap.add_argument("--balance", action="store_true",
                    help="upsample research/MLS (default multipliers 20x/8x ~= equal thirds)")
    ap.add_argument("--research-x", type=int, default=None, help="research upsample factor (implies nothing alone; default 1, or 20 with --balance)")
    ap.add_argument("--mls-x", type=int, default=None, help="MLS upsample factor (default 1, or 8 with --balance)")
    ap.add_argument("--smoke", action="store_true", help="also emit 8-row train_smoke.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    research_x = args.research_x if args.research_x is not None else (20 if args.balance else 1)
    mls_x = args.mls_x if args.mls_x is not None else (8 if args.balance else 1)

    synth = load(Path(args.synth), "frontiersmith_synth")
    # research is EVAL-ONLY (user directive 2026-08-04): its 64 problems ARE the
    # Research benchmark with no held-out split, so training on them would make
    # that benchmark uninterpretable. Included only if a path is passed explicitly.
    research = load(Path(args.research), "frontiercs_research") if args.research else None
    mls = load(Path(args.mls), "mlsbench_rl")

    parts = [synth] + ([research] * research_x if research is not None else []) + [mls] * mls_x
    train = finalize(pd.concat(parts, ignore_index=True), args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    train.to_parquet(train_path, index=False)
    rep = {
        "train": str(train_path),
        "research_x": research_x,
        "mls_x": mls_x,
        "seed": args.seed,
        "mix": report_mix(train, "train"),
        "note": "GRPO samples batches uniformly over rows -> per-batch source mix follows row proportions.",
    }

    if args.smoke:
        rng = np.random.RandomState(args.seed)
        s_pool = synth[synth["extra_info"].apply(lambda e: e.get("lang") == "py")]
        if len(s_pool) < 4:
            s_pool = synth
        s_rows = s_pool.iloc[rng.choice(len(s_pool), 4, replace=False)]
        smoke_parts = [s_rows]
        if research is not None:
            r_rows = research[research["reward_model"].apply(lambda r: r["ground_truth"] in SMOKE_RESEARCH_GT)]
            if len(r_rows) < 2:
                r_rows = research.iloc[:2]
            smoke_parts.append(r_rows.iloc[:2])
        m_rows = mls[mls["extra_info"].apply(lambda e: e.get("uid") in SMOKE_MLS_UIDS)]
        if len(m_rows) < 2:
            m_rows = mls.iloc[:2]
        # research disabled -> take 2 more synth rows so the smoke keeps 8 rows
        if research is None and len(s_pool) >= 6:
            smoke_parts.append(s_pool.iloc[rng.choice(len(s_pool), 2, replace=False)])
        smoke_parts.append(m_rows.iloc[:2])
        smoke = finalize(pd.concat(smoke_parts, ignore_index=True), args.seed)
        smoke_path = out_dir / "train_smoke.parquet"
        smoke.to_parquet(smoke_path, index=False)
        rep["smoke"] = {"path": str(smoke_path), "mix": report_mix(smoke, "smoke")}
        rep["smoke"]["rows"] = [
            {"data_source": r["data_source"], "agent_name": r["agent_name"],
             "ground_truth": r["reward_model"]["ground_truth"]}
            for _, r in smoke.iterrows()
        ]

    # read-back sanity: schema union must preserve per-source fields
    back = pq.read_table(train_path).to_pandas()
    assert len(back) == len(train)
    mls_back = back[back["data_source"] == "mlsbench_rl"].iloc[0]["extra_info"]
    assert mls_back.get("budget") and mls_back["budget"].get("tier") in ("tight", "loose"), "MLS budget lost in write"
    syn_back = back[back["data_source"] == "frontiersmith_synth"].iloc[0]["extra_info"]
    assert syn_back.get("checker"), "synth checker field lost in write"
    assert all(a in {"single_turn_agent", "mlsbench_agent"} for a in back["agent_name"].unique())
    print("[read-back] OK: budgets + synth fields survive the struct union; agent_name valid")

    rep_path = out_dir / "report.json"
    rep_path.write_text(json.dumps(rep, indent=2))
    print(f"[report] {rep_path}")


if __name__ == "__main__":
    main()
