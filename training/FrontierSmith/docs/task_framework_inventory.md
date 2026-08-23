# Task Framework Inventory

This repo currently runs FrontierSmith-style RL through VERL with task routing
driven by the parquet `data_source` column. New tasks should be added by
providing:

- a parquet builder that writes `prompt`, `reward_model.ground_truth`, and
  `data_source`
- a reward implementation under `verl/verl/utils/reward_score/`
- a `default_compute_score` branch for the new `data_source`
- any offline cache/container preflight needed on compute nodes
- an eval wrapper that can emit the same `summary.json` schema used by
  `scripts/collect_reproduction_results.py`

## Current Data Sources

| data_source | Split/use | Rows | Reward backend | External runtime |
| --- | ---: | ---: | --- | --- |
| `frontiercs` | train/eval | 172 official + 10 local FrontierSmith train | `frontiercs.compute_score` | local FrontierCS judge via Apptainer |
| `alebench` | eval | 10 lite eval problems | `alebench.compute_score` | ALE-Bench Apptainer images + Rust tool cache |
| `alebench_full` | train | 40 full public ALE problems | `alebench_full.compute_score` | ALE-Bench Apptainer images + Rust tool cache |

Current parquet inventory:

| parquet | rows | data_source counts |
| --- | ---: | --- |
| `data/frontiercs/full.parquet` | 172 | `frontiercs=172` |
| `data/frontiercs/train.parquet` | 172 | `frontiercs=172` |
| `data/frontiercs/train_synthetic.parquet` | 10 | `frontiercs=10` |
| `data/alebench/val.parquet` | 10 | `alebench=10` |
| `data/alebench_full/val.parquet` | 40 | `alebench_full=40` |
| `data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet` | 222 | `frontiercs=182`, `alebench_full=40` |

## Reward Routing

`verl/verl/utils/reward_score/__init__.py` routes by exact `data_source`:

- `frontiercs` -> `frontiercs.compute_score`
- `alebench` -> `alebench.compute_score`
- `alebench_full` -> `alebench_full.compute_score`

For future ThetaEvolve or TTT-Discover-style tasks, prefer one `data_source`
per evaluator contract, for example:

- `circle_packing`
- `autocorr_faci`
- `autocorr_saci`
- `autocorr_taci`
- `hadamard_matrix`
- `atcoder_ahc039`
- `atcoder_ahc058`

Each reward module should return a dict with at least `score` and any
task-native metrics needed for summaries.

## Offline Runtime Requirements

Compute nodes should not download or build dependencies.

- FrontierCS uses `.cache/apptainer/frontiercs-judge.sif`.
- ALE-Bench uses `.cache/apptainer/alebench` images.
- ALE-Bench Rust tools must be prebuilt under
  `.cache/ale-bench/rust-tool-builds`.
- `ALE_BENCH_REQUIRE_TOOL_CACHE=1` should remain enabled in Slurm jobs so a
  missing tool cache fails immediately instead of trying `cargo build`.

Preflight command:

```bash
source .venv/bin/activate
python scripts/prepare_alebench_tool_cache.py --no-lite --check-only
```

## Eval Summary Contract

`scripts/collect_reproduction_results.py` expects each eval output directory to
eventually contain `summary.json`. Pending sharded evals may contain
`shards/shard_*/samples.jsonl`; the collector reports partial sample and error
counts until the summary exists.

New task evals should either reuse `scripts/eval_qwen35_base_vllm_request.py`
where possible or write a compatible `summary.json` containing:

- `config`
- `complete_problem_count`
- `scored_sample_count`
- `metrics`
- `oracle_best` where applicable
