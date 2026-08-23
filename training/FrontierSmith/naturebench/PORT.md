# NatureBench on Della: recon + Apptainer port plan

Date: 2026-08-15. Status: pilot implementation in `naturebench/`.

## 1. Recon: how NatureBench actually launches tasks

Official entry points: `run_naturebench.py` (dataset download + batch) and
`solve.py` (per-task orchestration). Repo: https://github.com/FrontisAI/NatureBench
(cloned at `naturebench/repo`).

### Container orchestration (solve.py)
- Base image `naturebench-base:v3` is **built locally** from
  `docker/Dockerfile.base` (`scripts/ensure_naturebench_base.sh`):
  `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` + Python 3.11 venv at
  `/opt/py311` + pinned scientific stack (torch 2.6.0+cu118, numpy 1.26.4,
  scanpy, rdkit, ...) + node22 + agent CLIs.
- Per-task images are **built locally at runtime** from
  `tasks/<id>/environment/Dockerfile.v3` (`solve.py:_ensure_task_image`,
  `docker build`). There is **no official registry of prebuilt task images**.
- **But** the default config path is `--skip-build` (`config.example.yaml`:
  `skip_build: true`): solve.py parses Dockerfile.v3 (`_parse_dockerfile`),
  starts the **base** image with `sleep N`, and replays the task's RUN
  commands inside via `docker exec` (setup phase, own `--setup-timeout`),
  then starts the timer (`POST /start_timer`) and `docker exec`s the agent
  with `--timeout` (default 14400 s = 4 h). So per-task image builds are
  already optional in the official code.
- Container mounts: `problem/` read-only at `/task/problem`, a per-run
  workspace at `/workspace`. `evaluation/` (evaluator + ground truth) is
  **never mounted** — the agent cannot see it.
- Eval endpoints: the container reaches a **host-side** HTTP service through
  `host.docker.internal` (`EVAL_SERVICE_URL`).

### Scoring (eval_service.py, host-side, unmodified in our port)
- Single conda env `naturebench-eval` (`conda_env_eval.yml`) serves all 90
  tasks (`eval_env_mapping.json` routes everything to it).
- `POST /register` loads `metadata.json` per-instance primary metric +
  `sota_score` into a per-(task,batch) table.
- `POST /evaluate {task_name, batch_name, output_dir}` runs
  `tasks/<id>/evaluation/evaluator.py` in a subprocess with `OUTPUT_DIR` set
  (`_evaluator_runner.py`), then computes per-instance
  `g = dir*(m - m_sota)/|m_sota|` and the aggregate (mean over primary
  instances). Best attempt across submissions is the task score
  (`/best_score`). The judge (`judge.py`, `JUDGE_MODEL`) is a post-hoc
  anti-cheat review, NOT part of g.

### Agent plug-in surface
- `agent/adapter.py` `AgentAdapter` registry; built-ins: claude/codex/gemini
  CLIs. `agent/base.py` has a single-shot "write run.py" agent with an
  OpenAI-compatible backend; its SYSTEM_PROMPT scaffold is reused by our
  harness agent verbatim.

## 2. Environment verdicts (measured, not assumed)

| Constraint | Verdict |
|---|---|
| user namespaces | DISABLED (`unshare -U true` / `bwrap` both fail) |
| docker daemon | NONE (client exists, no socket; rootless impossible w/o userns) |
| `apptainer build x.sif docker://...` | **WORKS unprivileged** (apptainer 1.5.0; verified with `docker://python:3.11-slim` and the 9.1 GB NatureBench base) |
| `apptainer build` from def w/ `%post` as root | blocked (needs root/fakeroot) — NOT needed by this port |
| HF dataset total size | **1.25 TB / 90 tasks** (computed from repo metadata, no blobs pulled) |
| outbound network from login node | works (HF, PyPI, Docker Hub reachable) |

### Prebuilt images found on Docker Hub (user `t2ance`)
- `t2ance/naturebench-base:5c8dc8803edb` (10.3 GB compressed -> 9.1 GB .sif).
  Verified contents: py3.11.15, torch 2.6.0+cu118, numpy 1.26.4,
  /opt/py311 = 7.5 GB, node present. Matches `docker/Dockerfile.base`.
- 6 prebuilt task images (~10.3-11.1 GB each): s43588-025-00872-z,
  s41467-025-65557-7, s42256-024-00956-x, s42256-022-00464-w,
  s42256-024-00790-1, s43588-025-00903-9. These are base + task pip deps
  baked in; with them, even the setup phase can be skipped for those tasks.
- The other 84 tasks have NO prebuilt image -> we replay Dockerfile.v3 RUN
  commands at setup time (the official `--skip-build` path).

### Dockerfile.v3 RUN-command audit (all 90 tasks, fetched metadata-only)
- 198 RUN commands total: **82/90 tasks are pip-only**.
- 8 tasks need `apt-get` (s41467-025-63418-x, s41587-025-02654-4,
  s41592-023-02148-8, s41592-024-02316-4, s41592-025-02870-5,
  s42256-023-00712-7, s42256-024-00795-w, s42256-024-00815-9) — the hard
  tail; 2 of those also use `R -e install.packages(...)`; 1 downloads a
  binary. See "Cost model / what breaks first".

### Per-task data sizes (HF metadata; full table in `naturebench/task_sizes.json`)
- Smallest: s43588-024-00730-4 (~0 MB), s43588-025-00872-z (0.2 MB),
  s42256-022-00468-6 (2.6 MB) ... 60 tasks are < 1 GB.
- Largest: s41592-025-02826-9 **532 GB**, s41592-025-02854-5 124 GB,
  s41592-023-02032-5 88 GB. Top 5 tasks = ~55% of the dataset.
- **CPU-only tasks (3)**: s43588-024-00689-2 (85 MB), s41592-022-01709-7
  (287 MB), s41592-025-02870-5 (2.4 GB, but it is an apt task).
- Cheapest pilot = **s43588-024-00689-2** (CPU, 85 MB, pip-only Dockerfile,
  evaluator needs only numpy+sklearn). Chosen for the pilot.

## 3. The port (what we built)

`naturebench/harness/nb_run.py` + `nb_agent.py`, Slurm wrapper
`naturebench/slurm/nb_pilot.sh`. Docker -> Apptainer mapping:

| Official (docker) | Port (apptainer) |
|---|---|
| `docker build` base image | `apptainer build naturebench-base.sif docker://t2ance/naturebench-base:5c8dc8803edb` (once, unprivileged) |
| `docker build` per task | skipped — official `--skip-build` semantics: replay Dockerfile.v3 RUN cmds at setup |
| image is mutable | per-task **writable copy of `/opt/py311`** bind-mounted over `/opt/py311` (one `cp -a` out of the SIF per task, cached in `results/_venvs/`); RUN pip-installs land there. Zero changes to RUN commands; `VIRTUAL_ENV`/`PATH` already point at `/opt/py311` |
| `docker run -d ... sleep N` + `docker exec` | no long-lived container needed: each phase is an `apptainer exec` with identical binds |
| `-v problem:/task/problem:ro`, `-v ws:/workspace` | `--bind ...:ro` / `--bind`; plus an **identity bind** of the workspace so the host path sent to `/evaluate` resolves inside the container too |
| `--add-host host.docker.internal` | not needed — apptainer shares the host netns; `EVAL_SERVICE_URL=http://127.0.0.1:<port>` |
| `--gpus device=N` | `apptainer exec --nv` inside a Slurm GPU allocation |
| eval service in conda env | unchanged: official `eval_service.py` run by `naturebench/envs/naturebench-eval` python; task `evaluator.py` + ground truth stay on the host, never mounted |
| 4 h autonomous CLI agent | bounded in-container loop (`nb_agent.py`): <= N rounds of (write run.py -> run -> official /evaluate -> feedback), hard wall budget; scoring path untouched |

Ground-truth hiding: identical to official — only `problem/` (ro) and the
workspace enter the container; `evaluation/` is read only by the host-side
eval service subprocess.

## 4. Pilot status (updated during bring-up)

**Scorer sanity check: PASSED.** Reference mode (bypasses the LLM; a fixed
competent solver: scanpy PCA + KMeans on spliced+unspliced, no ground truth)
on `s43588-024-00689-2` ran end-to-end on the login node through the FULL
official path (venv-copy setup -> Dockerfile.v3 RUN replay -> in-container
execution -> official eval_service /evaluate -> metadata.json SOTA
normalization):

    NB_SCORE s43588-024-00689-2 0.05173509062121387

raw ARI: cl3 0.9933, cl5 0.9844, mop_sc 0.6725, mop_sn 0.7160;
per-instance g: +0.0033 / +0.0004 / +0.1594 / +0.0438 -> aggregate g = +0.0517
(positive = above the paper's SOTA reference on the primary metric).
This proves the ported scorer returns sensible NONZERO g for a good
submission, and that the whole Apptainer path is faithful.

Porting pitfalls found and fixed in the harness (all logged):
1. Base image `/etc/pip.conf` adds dead extra index `pypi.ngc.nvidia.com` ->
   every pip lookup retried for ~30 s. Fix: `PIP_CONFIG_FILE=/dev/null` +
   `PIP_INDEX_URL=https://pypi.org/simple` (empty `PIP_EXTRA_INDEX_URL` does
   NOT override pip.conf).
2. Apptainer refuses `--env HOME=...`; host home is quota-tight (48.8/50 GiB)
   -> in-container writes to $HOME fail. Fix: `apptainer exec --home
   /workspace/.home`.
3. venv-presence check must use `pyvenv.cfg`, not `bin/python` (absolute
   symlink that only resolves inside the container).
4. Task bitrot (upstream, NOT a port bug): the task's Dockerfile today
   resolves numpy 2.2.6, which breaks its own pinned loompy 3.0.7
   (`np.string_` removed in numpy 2). The official docker path would fail
   identically if built today. Fix: `--setup-extra 'pip install numpy==1.26.4'`
   (explicit, logged in setup.log and result.json).
5. conda must not use $HOME pkgs cache (`CONDA_PKGS_DIRS` on scratch) — home
   quota overflow corrupted the first env build.
6. **Compute nodes have NO outbound network/DNS** (pypi.org unresolvable on
   della-l07g2). Consequences: (a) the setup phase (pip installs) must run on
   a LOGIN node once per task — `nb_run.py --setup-only` does this and writes
   a `.setup_done` marker (content = exact setup script) into the task venv;
   compute-node runs with a matching marker skip setup entirely; (b) the
   agent inside the container cannot reach the internet either, consistent
   with the benchmark's offline task semantics — our agent only needs the
   node-local vLLM endpoint and the node-local eval service.

**Agent pilot: PIPELINE WORKS END-TO-END, WITH A POSITIVE MODEL SCORE.**
Two agent-mode runs on gpu-test (1x A100, vLLM Qwen3.5-9B served as
`qwen35-9b`, official eval service, bounded in-container agent):

- Job 12431192 (4 rounds): rounds 1-3 crashed and were fixed via error
  feedback; round 4 ran clean and the official /evaluate returned
  **g = -1.0** — the benchmark's own validation-failure penalty (9B predicted
  per-gene instead of per-cell labels). Wall: 4 min total.
- Job 12431631 (6 rounds, gpu-test): rounds 1-5 crashed, round 6 ran clean;
  official **g = +0.0652** (cl3 +0.0033, cl5 -0.0010, mop_sc +0.2267,
  mop_sn +0.0317; raw ARI 0.9933/0.9830/0.7115/...). POSITIVE = the 9B's own
  submission surpassed the paper's SOTA reference aggregate on this task.
  Wall: ~5 min.
- Job 12432118 (6 rounds, **gpu-ee** — the coordinator-sanctioned lane):
  scheduled in 23 s, same flow, round 6 ran clean; official **g = +0.0436**.
  gpu-ee works end-to-end and is the recommended partition going forward.

All three scoring regimes are thus demonstrated through the untouched
official path (eval_service.py + task evaluator.py + metadata.json SOTA
normalization): good submission -> +0.0517 (reference) / +0.0652 (9B agent),
invalid submission -> -1.0 penalty.

Note: the first agent pilot attempt (12430611) was cancelled by the
coordinator (gpu-test slot conflict, not a failure) — resubmission ran to
completion. gpu-test is capped at 3 jobs/user and contended by the cce-rlv10
campaign; per coordinator guidance, use `--partition=gpu-ee --gres=gpu:a100:1`
(2 nodes x 2 A100-40G, 9B bf16 fits) or `--qos=gpu-medium` for future runs.

## 4b. Phase 2 — from one task to a rankable subset

Goal: a NatureBench SUBSET that yields a comparable per-model score cheap
enough to run per RL checkpoint.

### Overlay instead of venv copies (the scaling unlock)
Phase 1 copied `/opt/py311` out of the SIF per task: **8.0 GB and ~90k inodes
each** — 15 tasks would have cost 120 GB and 1.35M inodes against a fileset
with only ~3.3M inodes free. Apptainer's writable **overlay directory** works
unprivileged on GPFS (verified), so the base SIF stays read-only and each
task's Dockerfile pip-installs land in `results/_overlays/<task>/upper`:

| | venv copy (phase 1) | overlay (phase 2) |
|---|---:|---:|
| disk / task | 8.0 GB | **0.37 GB avg** |
| inodes / task | ~90,000 | **~5,300 avg** |
| prep time / task | ~60 s copy + setup | setup only (8-78 s) |

A shared pip wheel cache (`results/_pipcache`) makes repeat installs fast;
`--no-cache-dir` is stripped from task RUN commands for this (infra-only — it
changes nothing about which versions pip resolves).

### Harness files (all under naturebench/)
| file | role |
|---|---|
| `harness/nb_fetch.py` | throttled per-task HF download (+ official archive materialization) |
| `harness/nb_setup.py` | LOGIN-NODE phase: build each task's overlay, record cost, emit `task-sets/working.txt` |
| `harness/nb_run.py` | library + CLI: one task (agent / reference / probe) |
| `harness/nb_batch.py` | one Slurm job = one shard: **vLLM loaded once**, one eval service, N tasks, resume, per-task locks |
| `harness/nb_aggregate.py` | official Match-SOTA (g>=0) / Surpass-SOTA (g>0.1); refuses incomplete batches |
| `harness/nb_report.py` | per-task cost + score table (`SUBSET.md`) |
| `slurm/nb_batch.sh` | gpu-ee / pli launcher |

### probe mode (cheap per-task validation, no GPU)
`--mode probe` submits the empty output dir to the official `/evaluate`. A
finite failure penalty (g = -1.0) proves registration, the metadata SOTA table,
the evaluator import (host conda env) and scoring all work for that task; an
exception or `g=None` means the task is not usable yet. ~1 s per task on a
login node. **All 17 prepared tasks pass.**

### Subset status
17/17 attempted tasks prepared and probe-validated, 0 setup failures. See
`SUBSET.md` for the per-task table (tier, data size, setup seconds, overlay
MB/inodes, repair flag, probe g, agent g). Only 1 of 17 needed a repair
(`s43588-024-00689-2`, the numpy/loompy pin).

### VERDICT: this subset canNOT yet rank 9B checkpoints
Two full 12-task batches of the SAME model (Qwen3.5-9B) were run end-to-end
through the official scorer, differing only in agent round budget:

| | 5 rounds | 8 rounds |
|---|---:|---:|
| tasks scored | 6/12 | 9/12 |
| of those, at the -1.0 failure floor | 3 | 6 |
| **graded (informative) scores** | **3** | **3** |
| Match-SOTA (g>=0) | 0.0% | 0.0% |
| mean g | -0.7988 | -0.8416 |

**11 of 12 tasks changed materially between the two runs**, and the 3 graded
tasks were not the same 3. The dominant variable is not science quality but
"did the 9B emit runnable, format-valid code this time" — a high-variance
binary that pins most tasks to `none` or the -1.0 floor. A single run per
checkpoint would rank on noise.

This is a model-capability ceiling, not a harness defect: with the identical
scorer, hand-written non-LLM baselines score `s43588-024-00689-2` **g = +0.0517**
and `s41592-022-01709-7` **g = -0.4698**, and an empty submission scores -1.0
everywhere — the tasks grade smoothly across the whole range.

To make it rankable, in order of expected value:
1. **k>=3 seeds per task, averaged** (the harness already shards/resumes;
   cost multiplies by k).
2. **Select tasks with graded partial credit**, dropping all-or-nothing output
   formats where a single shape error yields -1.0.
3. **Score validity separately** (a "produced a valid submission" rate is a
   lower-variance signal than g at this capability level, and is free —
   `total_attempts` is already recorded per task).
4. A stronger model; 9B is below the threshold where g differences are the
   dominant signal.

## 5. Cost model

Measured on the pilot (task s43588-024-00689-2):

| Item | One-time | Per task |
|---|---|---|
| HF data download | — | 0-532 GB (60/90 tasks < 1 GB; full 90 = 1.25 TB; a sensible 40-task subset < 60 GB) |
| base .sif | 9.1 GB (+9.6 GB apptainer cache, clearable) | — |
| eval conda env | ~6 GB | — |
| venv copy (/opt/py311) | — | 7.5 GB, ~60 s (GPFS) |
| setup (pip replay) | — | 12 s warm / 5-15 min cold, LOGIN NODE ONLY (compute nodes have no DNS) |
| agent wall time (bounded) | — | 4 min measured (4 rounds) to ~40 min budget |
| GPU-hour (9B vLLM + agent) | — | ~0.1-0.7 h on 1x A100 per attempt; CPU tasks still want a GPU for the serving model |

Scaling to N=40 pip-only tasks: disk ~60-350 GB depending on subset
(dominated by data + 7.5 GB venv each; venvs are deletable/rebuildable),
GPU-hours ~4-30 per full pass at bounded budgets, wall time ~1-2 days at
3-way gpu-test concurrency.

What breaks first, in order:
1. **gpu-test contention** — the cce-rlv10 campaign holds all 3 slots with
   cycling 1 h jobs; pilot jobs queue tens of minutes and one was externally
   scancelled. Mitigation: gpu-ee partition (worked) or off-hours.
2. **8 apt tasks** — no root anywhere in our path; options: (a) use the 6
   prebuilt task images on Docker Hub when they cover the task (1 does:
   s41467-025-65557-7 is apt-free though...), (b) extract .debs with
   `dpkg -x` into a user prefix + LD_LIBRARY_PATH via setup-extra, (c) skip.
3. **GPU tasks (70+17)** — need `--nv` (already wired, untested here) and
   24 GB/80 GB cards; A100-80GB covers both tiers. vLLM shares the card with
   task compute — either lower gpu-memory-utilization or run vLLM on a
   separate job/node for GPU-heavy tasks.
4. **1 h gpu-test cap** — fine for the bounded agent; NOT fine for the
   official 4 h budget. Use gpu-ee/gpu-medium (15d/4h+) for long runs.
5. **Inode quota** (96.2M/100M used fileset-wide) — the venv copies add
   ~50-100k files each; 40 tasks ~ +3M inodes. Watch it.
6. **Judge** (judge.py, post-hoc anti-cheat) needs an LLM with a large
   context; wire JUDGE_BASE_URL to local vLLM if wanted — it is NOT part of
   the official g score and is off by default in the harness.

