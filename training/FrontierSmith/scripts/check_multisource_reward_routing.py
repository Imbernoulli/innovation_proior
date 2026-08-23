#!/usr/bin/env python
"""OFFLINE (no GPU) reward-routing check for the multi-source RL parquet.

For one row of each source, exercise the exact path the trainer uses:
  * synth/research rows: verl.utils.reward_score.default_compute_score dispatch on
    data_source -- the same call RewardLoopWorker/NaiveRewardManager.run_single makes
    for rows whose agent loop left reward_score=None. Payloads with KNOWN scores:
      - synth: the problem's own reference solutions (strong.py should be > trivial.py > 0)
      - research: a stored eval response that previously scored 65.0 (replay must match)
  * MLS rows: verify agent_name routing exists -- load config/mlsbench_agent_loop.yaml
    into the agent-loop registry exactly like AgentLoopWorker.__init__ does, import
    MLSBenchAgentLoop cleanly, resolve the row's budget, and confirm the in-loop
    reward -> AgentLoopOutput.reward_score -> rm_scores wiring statically.

Run under the training venv from the project root:
  .venv-vllm023/bin/python scripts/check_multisource_reward_routing.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verl"))

# --- trainer-equivalent env (mirrors slurm/cc_rl_multisource.sh exports) ---
os.environ.setdefault("FRONTIERSMITH_SYNTH_ROOT", str(ROOT.parent / "innovation_prior/frontiersmith_synth"))
os.environ.setdefault("FRONTIERSMITH_SYNTH_FAIL_SOFT", "1")
os.environ.setdefault("FRONTIERCS_RESEARCH_PYTHON", "/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python")
os.environ.setdefault("JULIA_DEPOT_PATH", "/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_depot")
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", "/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/julia_env")
os.environ.setdefault("FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB", "64")
os.environ.setdefault("FRONTIERCS_RESEARCH_CPU_TIMEOUT", "300")
os.environ.setdefault("MLS_RL_MLSBENCH_ROOT", "/scratch/gpfs/CHIJ/bohan/MLS-Bench-train")

import pyarrow.parquet as pq  # noqa: E402

PARQUET = ROOT / "data/multisource_rl/train.parquet"
df = pq.read_table(PARQUET).to_pandas()
print(f"=== multisource routing check on {PARQUET} ({len(df)} rows) ===\n")

ok = True

# ---------------------------------------------------------------- registry ---
print("[1] agent_name routing registry")
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry  # noqa: E402
import verl.experimental.agent_loop  # noqa: E402,F401  (registers single_turn_agent, tool_agent)
from omegaconf import OmegaConf  # noqa: E402

# same load AgentLoopWorker.__init__ performs when agent_loop_config_path is passed
for cfg in OmegaConf.load(ROOT / "config/mlsbench_agent_loop.yaml"):
    _agent_loop_registry[cfg.name] = cfg
print(f"    registry after loading config/mlsbench_agent_loop.yaml: {sorted(_agent_loop_registry)}")
used = set(df["agent_name"].unique())
missing = used - set(_agent_loop_registry)
print(f"    agent_name values in parquet: {sorted(used)}  missing from registry: {sorted(missing) or 'NONE'}")
ok &= not missing

# ---------------------------------------------------------------- synth ------
print("\n[2] frontiersmith_synth reward (default_compute_score dispatch, sandbox harness)")
from verl.utils.reward_score import default_compute_score  # noqa: E402

srow = df[df["data_source"] == "frontiersmith_synth"].iloc[0]
gt = srow["reward_model"]["ground_truth"]
pdir = Path(os.environ["FRONTIERSMITH_SYNTH_ROOT"]) / "problems" / gt
for name, expect in (("strong", "high"), ("trivial", "low/nonzero")):
    sol = (pdir / "solutions" / f"{name}.py").read_text()
    score = default_compute_score(
        data_source=srow["data_source"],
        solution_str=f"```python\n{sol}\n```",
        ground_truth=gt,
        extra_info=dict(srow["extra_info"]),
    )
    print(f"    {gt} reference={name}.py -> score={score} (expected {expect})")
    if name == "strong":
        ok &= float(score if not isinstance(score, dict) else score["score"]) > 0
print(f"    sandbox backend: ISORUN_BACKEND={os.environ.get('ISORUN_BACKEND', '(auto)')}")

# ---------------------------------------------------------------- research ---
print("\n[3] frontiercs_research reward (default_compute_score dispatch, evaluator subprocess)")
import json  # noqa: E402

REPLAY_GT = "cant_be_late/low_availability_loose_deadline_small_overhead"
REPLAY_EXPECT = 65.0
replay_text = None
shards = sorted((ROOT / "outputs/cc_eval_clean_clean_full_wd01_a10_research_thinking_32k_vllm").glob("shard_*/samples.jsonl"))
for shard in shards:
    with open(shard) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ground_truth") == REPLAY_GT and (d.get("metrics") or {}).get("score") == REPLAY_EXPECT:
                replay_text = d["text"]
                break
    if replay_text:
        break
rrow = df[df["reward_model"].apply(lambda r: r["ground_truth"] == REPLAY_GT)]
assert len(rrow) > 0, f"{REPLAY_GT} not in mixed parquet"
rrow = rrow.iloc[0]
assert replay_text, "stored replay sample not found"
score = default_compute_score(
    data_source=rrow["data_source"],
    solution_str=replay_text,
    ground_truth=REPLAY_GT,
    extra_info=dict(rrow["extra_info"]),
)
sval = float(score if not isinstance(score, dict) else score["score"])
print(f"    {REPLAY_GT}: replayed stored response -> score={sval} (stored eval score={REPLAY_EXPECT})")
ok &= abs(sval - REPLAY_EXPECT) < 1e-6

# ---------------------------------------------------------------- pertask ----
print("\n[4] per-task reward normalization (FS_PERTASK_REWARD_NORM=1 in launcher)")
os.environ["FS_PERTASK_REWARD_NORM"] = "1"
from verl.utils.reward_score import pertask_norm  # noqa: E402

for dsrc, raw in (("frontiersmith_synth", 37.5), ("frontiercs_research", REPLAY_EXPECT)):
    n = pertask_norm.maybe_normalize(dsrc, raw, {})
    print(f"    {dsrc}: raw {raw} -> normalized {n['reward']} (bounds 0-100 -> [0,1]; MLS in-loop scores are already [0,1])")

# ---------------------------------------------------------------- MLS --------
print("\n[5] mlsbench_rl agent-loop wiring (no episode; import + budget + reward path)")
from verl.experimental.agent_loop.mlsbench_agent_loop import MLSBenchAgentLoop, resolve_budget  # noqa: E402
import inspect  # noqa: E402

mrow = df[df["data_source"] == "mlsbench_rl"].iloc[0]
budget = resolve_budget(dict(mrow["extra_info"]), default_max_steps=8, default_max_tests=1)
print(f"    MLSBenchAgentLoop imported from {inspect.getfile(MLSBenchAgentLoop)}")
print(f"    row task={mrow['extra_info']['task']} uid={mrow['extra_info']['uid']} -> resolved budget={budget}")
src = inspect.getsource(MLSBenchAgentLoop.run)
assert "reward_score=state.reward" in src, "in-loop reward -> AgentLoopOutput.reward_score wiring missing"
print("    MLSBenchAgentLoop.run sets AgentLoopOutput.reward_score=state.reward  [OK]")
from verl.experimental.agent_loop import agent_loop as AL  # noqa: E402

post = inspect.getsource(AL.AgentLoopWorker._postprocess)
assert 'batch["rm_scores"] = rm_scores' in post, "rm_scores assembly missing"
print('    AgentLoopWorker._postprocess writes batch["rm_scores"] when all reward_scores set  [OK]')
comp = inspect.getsource(AL.AgentLoopWorker._compute_score)
assert "output.reward_score is None" in comp
print("    single-turn rows (reward_score=None) -> RewardLoopWorker.compute_score (data_source dispatch)  [OK]")

# worker-side import under the episode worker python + train root
import subprocess  # noqa: E402

wp = os.environ.get("MLS_RL_WORKER_PYTHON", "/home/bl3615/miniconda3/bin/python")
r = subprocess.run(
    [wp, "-c", "import mlsbench.agent.interactive, mlsbench.agent.tools; print('mlsbench worker imports OK')"],
    env={**os.environ, "PYTHONPATH": os.path.join(os.environ["MLS_RL_MLSBENCH_ROOT"], "src")},
    capture_output=True, text=True, timeout=300,
)
print(f"    worker python import (root={os.environ['MLS_RL_MLSBENCH_ROOT']}): "
      f"{r.stdout.strip() or r.stderr.strip()[:400]}")
ok &= r.returncode == 0

print(f"\n=== RESULT: {'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} ===")
sys.exit(0 if ok else 1)
