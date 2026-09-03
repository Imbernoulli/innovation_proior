# MLS-Bench — 13 recovered tasks

This directory holds **13** MLS-Bench tasks developed by **bl3615** and **st3812** that are **not** among the 140 published tasks on `Imbernoulli/MLS-Bench` `main` (commit `4b1fd673`). They were collected on **2026-09-02** for upstreaming. Each task was taken from the newest complete copy found across the three colleagues' working trees, checking **both** `tasks/<name>` and `tasks/deprecated/<name>` in every tree, with git history of the two git repositories used to cross-check that no newer or richer version was being missed. Symlinks were dereferenced; `__pycache__/`, `*.pyc`, `.git/`, `workspaces/`, `runs/`, `logs/`, `.ipynb_checkpoints/` and `*.sif` images were excluded; no single file over 5 MB was copied. All reads from the colleagues' trees were read-only — nothing was checked out, reset or written there.

> ## ⚠️ These tasks are marked DEPRECATED upstream
>
> **11 of the 13** were moved by their authors from `tasks/<name>` to `tasks/deprecated/<name>` (the deprecation commit `38673b830a`, 2026-05-15, moved the whole batch), and the two that still live at the active path — `libero-lifelong` and `llm-hybrid-posttraining` — also have deprecated siblings. The `tasks/<name>` directory that remains behind for most of these is a **hollow stub** holding nothing but stale `__pycache__` or an empty `edits/`, which is why a naive scan of `tasks/<name>` finds them empty. Anyone upstreaming this work needs to know these were deliberately retired upstream and decide whether to revive them.

## Leaderboard usability

The leaderboards are append-only run logs: each row is one run, and wide schemas union the columns of several sub-benchmarks, so a row legitimately fills only its own group's columns. Every `leaderboard.csv` here was parsed and each data row checked for populated **metric** cells (bookkeeping columns — `timestamp`, `model`, `is_final`, `seed`, `elapsed_*`, `*_std`, `*_n_total`, `*_n_samples`, `*_n_correct` — are excluded from that test). Results:

- **9 of 13** ship a usable leaderboard with at least one row of real numbers.
- **3 of 13** ship a leaderboard file that is **not usable**: `llm-pretrain-weight-averaging` and `robot-imitation-objective` (no metric columns in the header at all) and — for the rejected deprecated copy only — `llm-hybrid-posttraining`.
- **1 of 13** (`llm-pretrain-packing`) has **no leaderboard at all**, and never did in any commit of either repository.

All 13 task directories were shipped in full regardless of leaderboard quality.

## Tasks

| task | chosen source (`deprecated/`?) | leaderboard.csv | usable content | files |
| --- | --- | --- | --- | --- |
| `llm-pretrain-packing` | `bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-packing` **(deprecated/)** | **none** | no leaderboard | 10 |
| `llm-pretrain-precision` | `bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-precision` **(deprecated/)** | 23 rows, 19 with data, 4 blank | 16 complete baseline (`is_final=true`) row(s) | 19 |
| `llm-pretrain-sparse-attention` | `bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-sparse-attention` **(deprecated/)** | 25 rows, 10 with data, 15 blank | 4 complete baseline (`is_final=true`) row(s) | 13 |
| `llm-pretrain-weight-averaging` | `bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-weight-averaging` **(deprecated/)** | 6 rows / **0 usable** | **NOT usable** (no metric columns at all) | 15 |
| `graph-temporal` | `bohan/MLS-Bench-dev/tasks/deprecated/graph-temporal` **(deprecated/)** | 12 rows, 12 with data, 0 blank | 12 complete baseline (`is_final=true`) row(s) | 18 |
| `robot-imitation-objective` | `bohan/MLS-Bench-dev/tasks/deprecated/robot-imitation-objective` **(deprecated/)** | 4 rows / **0 usable** | **NOT usable** (no metric columns at all) | 14 |
| `ar-video-kv-temporal-policy` | `bohan/MLS-Bench-dev/tasks/deprecated/ar-video-kv-temporal-policy` **(deprecated/)** | 4 rows, 4 with data, 0 blank | 4 complete baseline (`is_final=true`) row(s) | 23 |
| `cv-flowmaps-training` | `bohan/MLS-Bench-dev/tasks/cv-flowmaps-training` (active) | 15 rows, 15 with data, 0 blank | 3 complete baseline (`is_final=true`) row(s) | 17 |
| `humanoid-ppo-extractor` | `bohan/MLS-Bench-dev/tasks/deprecated/humanoid-ppo-extractor` **(deprecated/)** | 3 rows, 3 with data, 0 blank | 0 complete baseline (`is_final=true`) row(s) | 17 |
| `jepa-mask-strategy` | `bohan/MLS-Bench-dev/tasks/deprecated/jepa-mask-strategy` **(deprecated/)** | 15 rows, 15 with data, 0 blank | 0 complete baseline (`is_final=true`) row(s) | 14 |
| `libero-lifelong` | `bohan/MLS-Bench-dev/tasks/libero-lifelong` (active) | 34 rows, 4 with data, 30 blank | 0 complete baseline (`is_final=true`) row(s) | 11 |
| `llm-hybrid-posttraining` | `bohan/MLS-Bench/tasks/llm-hybrid-posttraining` (active) | 3 rows, 3 with data, 0 blank | 3 complete baseline (`is_final=true`) row(s) | 23 |
| `llm-on-policy-distillation` | `bohan/MLS-Bench-dev/tasks/llm-on-policy-distillation` (active) | 5 rows, 5 with data, 0 blank | 5 complete baseline (`is_final=true`) row(s) | 16 |

## Per-task detail: other locations seen, divergences, merges

### `llm-pretrain-packing`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-packing`
- **Also present in:** byte-identical copy also at st3812 and bohan-MLS-Bench `tasks/deprecated/`; hollow stub (only stale `__pycache__`) at st3812 `tasks/`
- **Notes:** **No `leaderboard.csv` has ever existed for this task** — verified exhaustively against the full object database of both git repos (`git rev-list --all --objects`, 337 refs in st3812 / 131 in dev): no blob at `tasks/llm-pretrain-packing/leaderboard.csv` or `tasks/deprecated/.../leaderboard.csv` in any commit. Git history also shows an *earlier* variant of the task at the active path with different edits (`first_fit`, `greedy_concat`, `sorted_packing`) plus `related_work.json`; the chosen deprecated copy is the later, fuller one (adds `parser.py`, `scripts/train.sh`, `mid_edit.py`, `custom_template.py` and 5 baseline edits).

### `llm-pretrain-precision`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-precision`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stub at st3812 `tasks/`
- **Notes:** Wide union-schema run log: rows are per-model-size groups (124m / 345m / 1.5b), so no single row fills every column — "fully populated" is not meaningful here. 19 of 23 rows carry numbers; the 4 blank rows are failed agent attempts (deepseek-reasoner, claude-opus-4.6, gemini).

### `llm-pretrain-sparse-attention`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-sparse-attention`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stubs at st3812 and bohan `tasks/`
- **Notes:** 10 of 25 rows carry numbers, including 4 fully-populated `is_final=true` baseline rows (dense, moba, nsa_full, reformer). The 15 blank rows are failed baseline retries and failed agent attempts (gpt-5.4, qwen3.6-plus, gemini-3.1-pro).

### `llm-pretrain-weight-averaging`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/llm-pretrain-weight-averaging`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stub at st3812 `tasks/`
- **Notes:** **Leaderboard NOT usable.** Its only non-bookkeeping column is `elapsed_gpt-762m` (8-10 s) — there is not a single metric column. All 6 rows are timing-only stubs from aborted runs. No better copy exists in any tree or in either git repo (same 421-byte blob everywhere).

### `graph-temporal`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/graph-temporal`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stubs at st3812 and bohan `tasks/`
- **Extra file preserved:** `leaderboard.superset-git-31e4675f09.csv` — blob `31e4675f09:tasks/graph-temporal/leaderboard.csv`
- **Notes:** **DIVERGENCE — needs human adjudication.** The shipped `leaderboard.csv` (12 rows, all fully populated) covers only 3 baselines (gwnet, mtgnn, staeformer). Git commit `31e4675f09` (bohan-MLS-Bench-dev, 2026-05-05, path `tasks/graph-temporal/leaderboard.csv`) holds a **24-row strict superset** covering all 6 baselines the task ships edits for (adds astgcn, dcrnn, stgcn) with identical numbers for the 3 shared ones. That richer file is preserved here as `leaderboard.superset-git-31e4675f09.csv`. It was NOT promoted to `leaderboard.csv` because the chosen deprecated snapshot is structurally later (it is the only copy carrying `score_spec.py`), so the scoring spec may have changed between the two.

### `robot-imitation-objective`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/robot-imitation-objective`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stub at st3812 `tasks/`
- **Notes:** **Leaderboard NOT usable.** The header is literally `timestamp,model,is_final,seed` — zero metric columns — and all 4 rows are `is_final=false`. No better copy exists anywhere (same 278-byte blob in both repos).

### `ar-video-kv-temporal-policy`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/ar-video-kv-temporal-policy`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stubs at st3812 and bohan `tasks/`
- **Notes:** Git commit `31e4675f09` holds a 12-row file at the active path, but it is the same four baselines in an older long/tidy schema (baseline x workload rows). The shipped 4-row wide-schema file is the later, harness-conformant pivot of the identical numbers — no data lost.

### `cv-flowmaps-training`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/cv-flowmaps-training`
- **Also present in:** st3812 `tasks/deprecated/` holds an older 6-row copy; bohan-MLS-Bench `tasks/` is byte-identical to the chosen dev copy
- **Merged in:** `scripts/bench_env.sh` (from `st3812/projects/MLS-Bench/tasks/deprecated/cv-flowmaps-training`)
- **Notes:** **DIVERGENCE resolved.** st3812 `tasks/deprecated/` has a 6-row leaderboard; the chosen active copy is a strict superset (same first 6 rows plus 9 newer runs from 2026-05-13/15/17). **MERGE APPLIED:** `scripts/bench_env.sh` was missing from the chosen copy but is referenced by `task_description.md`, `scripts/train_{small,medium,large}.sh` and three baseline edits — it was merged in from `st3812/tasks/deprecated/cv-flowmaps-training/`.

### `humanoid-ppo-extractor`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/humanoid-ppo-extractor`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stubs at st3812 and bohan `tasks/`
- **Notes:** All 3 rows fully populated. Identical in all three trees.

### `jepa-mask-strategy`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/deprecated/jepa-mask-strategy`
- **Also present in:** byte-identical copies at all three trees under `tasks/deprecated/`; hollow stub at bohan `tasks/`
- **Notes:** 14 of 15 rows fully populated; 1 partial. Identical in all three trees.

### `libero-lifelong`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/libero-lifelong`
- **Also present in:** st3812 `tasks/deprecated/` holds a header-only (0-row) leaderboard; bohan-MLS-Bench `tasks/` is byte-identical to the chosen dev copy
- **Notes:** Leaderboard is thin: of 34 rows only 4 carry any numbers (3 of them complete with fwt/nbt/auc) and all rows are `is_final=false`; the other 30 are empty harness stubs. Still the best copy by far — the st3812 `tasks/deprecated/` copy is header-only (0 rows).

### `llm-hybrid-posttraining`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench/tasks/llm-hybrid-posttraining`
- **Also present in:** all three trees hold a `tasks/deprecated/` copy (22 files, 9-row leaderboard with NO numbers); bohan-MLS-Bench-dev `tasks/` holds an older complete 3-row run (2026-04-27)
- **Merged in:** `data/eval_subsets_20260426/AIME24_eval.parquet` (from `bohan/MLS-Bench-dev/tasks/deprecated/llm-hybrid-posttraining`), `data/eval_subsets_20260426/AMC23_eval.parquet` (from `bohan/MLS-Bench-dev/tasks/deprecated/llm-hybrid-posttraining`), `data/eval_subsets_20260426/MATH-500_eval.parquet` (from `bohan/MLS-Bench-dev/tasks/deprecated/llm-hybrid-posttraining`), `scripts/select_eval_subset.py` (from `bohan/MLS-Bench-dev/tasks/deprecated/llm-hybrid-posttraining`)
- **Notes:** **DIVERGENCE adjudicated — the 9-row copy is NOT a superset, it is unusable.** The `tasks/deprecated/` leaderboard has 9 rows that are *entirely blank* (every metric cell empty, all `is_final=false`, 2026-04-28/30). The chosen copy (bohan-MLS-Bench `tasks/`) has 3 fully-populated `is_final=true` baseline rows (sft/grpo/hpt) from 2026-06-01 and 2026-06-05 — the newest real content in any tree; its `scripts/train_shared_hpt.sh` is also substantially more developed (2026-06-03 "soft" pivot, in-job retry/resume loop, failure guards) than the dev-tree version. **MERGE APPLIED:** 4 files present only in the deprecated snapshot were merged in — `data/eval_subsets_20260426/{AIME24,AMC23,MATH-500}_eval.parquet` and `scripts/select_eval_subset.py`. Note these are *not referenced* by any file in the chosen copy, so they may be stale; they are preserved rather than dropped.

### `llm-on-policy-distillation`

- **Chosen:** `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/tasks/llm-on-policy-distillation`
- **Also present in:** bohan-MLS-Bench `tasks/` differs only in `task_description.md`; not present in st3812 at all
- **Notes:** Chosen the dev copy: the only difference from bohan-MLS-Bench is `task_description.md`, where dev carries the later deliberate fix (commit `322b922362`, 2026-06-24, "make task-body edit path package-prefixed to match the editable contract") changing `trl/experimental/...` to `trl/trl/experimental/...`.

## Recovered from git history

No task directory needed to be reconstructed from git history — every one of the 13 was present as real files in at least one working tree. Git history was used **read-only** for verification and for one artifact:

- `graph-temporal/leaderboard.superset-git-31e4675f09.csv` — extracted with `git cat-file blob 31e4675f09:tasks/graph-temporal/leaderboard.csv` in `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev`.
- `llm-pretrain-packing` — absence of any leaderboard confirmed by scanning the complete object database of both repositories.

## Oversized files skipped (>5 MB)

**0** — no file in any chosen copy exceeded 5 MB. (The largest merged binaries, the three `llm-hybrid-posttraining` eval parquets, are 8.7-11.6 KB each.)

## Needs human adjudication

1. **`graph-temporal`** — promote the 24-row 6-baseline superset over the shipped 12-row 3-baseline file? Requires checking whether `score_spec.py` (present only in the newer deprecated snapshot) changed the scoring.
2. **`llm-hybrid-posttraining`** — the 4 merged `data/eval_subsets_20260426/*` + `scripts/select_eval_subset.py` files come from the older deprecated snapshot and are referenced nowhere in the chosen copy. Keep or drop?
3. **`llm-pretrain-weight-averaging`**, **`robot-imitation-objective`** — leaderboards are structurally empty (no metric columns). The baselines were evidently never scored. Decide whether to upstream these tasks without results.
4. **`llm-pretrain-packing`** — never had a leaderboard; also has an alternative earlier baseline set in git history. Decide which baseline set is intended.

## Totals

- Tasks: **13 / 13** requested (none missing)
- Files: **210**
- Total size on disk: **637 KB** (652094 bytes)
- Tasks with a usable leaderboard: **9**; unusable: **3**; absent: **1**
- Oversized files skipped: **0**
