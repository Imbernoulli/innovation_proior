# decontam/ — SFT data-leakage audit & decontaminated copies

Audit of the `innovation_proior` SFT data for **data leakage / contamination** against the five
eval benchmarks the training will be graded on: **FrontierCS (FCS), ALE-Bench (ALE), ThetaEvolve
(THETA), TTT-Discover (TTT), MLS-Bench (MLS)**.

This extends the earlier **n-gram** audit (`experiments/DATA_CONTAMINATION_AUDIT_zh.md`), which by
its own caveat cannot catch *semantic paraphrase / classic-problem reconstruction*. Most of the real
leakage here is exactly that: training examples that **reconstruct an eval task** (often citing its
published record) without copying its prompt verbatim, so n-gram sees nothing.

**Originals are never modified.** Everything here is additive: annotations + duplicated clean copies.

## What leaks (three classes)

1. **Direct eval-task reconstructions** — training methods/trajectories built *around a specific eval
   task*, frequently reaching its published record. Smoking-gun slug suffixes: `-autoevolver-record`,
   `-frontier-largeN`, `-record`, `-goldberg-optimal`, `-shinka-targeted-sa`.
   - THETA/TTT math-discovery family (AlphaEvolve problem set): circle-packing n=26, Erdős
     minimum-overlap C5, autocorrelation inequalities AC1/AC2/AC3, Hadamard max-det, Heilbronn
     triangle, cap set, kissing number, fast matrix multiplication.
   - ALE: `ahc039-*` + `ale-atcoder-ahc039` = the real AtCoder AHC039 problem.
   - FCS-research / TTT (~20 methods, semantic — n-gram blind): `trimul`; the `denoising` reference
     methods `ttt-discover-denoise`/`bio-scrna-denoise`/`magic-imputation`/`magic-diffusion`/`knn-smoothing`;
     FlashAttention (`flashattention`, `flash-attention-2`, `flash-v1/2/3` = `flash_attention`); MLA
     (`mla`, `mla-attention`); `simp` (=`topology_optimization`); `proton-therapy-impt`; `malloc-allocator`.
     (FCS *algorithmic* track is essentially clean — 0 confident hits across the 40-statement shard.)
2. **MLS-Bench same-task** — 139 trajectory slugs + their agentic mirrors are *literally MLS-Bench
   task slugs* (same research question, fixed interface, baseline ladder). Not clean held-out.
   - **Type-1 sub-case (user's concern):** 51 of these trajectories inject a **non-native, stronger
     baseline** as a `finale` endpoint (a real published method absent from the MLS task's native
     baselines — e.g. jSO, SOAP, UniPC, diff-transformer, gated-DeltaNet, CrossQ). This is extra
     capability beyond what MLS discloses at inference. See `mls_type1_nonnative_finale.json`.
     - **Reference = the 140 PUBLISHED tasks** (`public_140_tasks.json`, github.com/Imbernoulli/MLS-Bench).
       Only the **46** offenders whose task is in that release have their finale dropped; the **5** on
       *unpublished* MLS tasks are spared (不用管 — keep finale).
3. **Register-matched synthetic** (annotated, *not* auto-dropped) — `v4` (346 synthetic FCS/ALE-style
   single-file C++, "造题 not 抄题") and `wave2` code/math/fcs_codex. Constructed, not copied, but drawn
   from the *same distribution* as FCS/ALE. Kept in the clean copy on purpose (dropping in-distribution
   delivery data defeats its training purpose) but flagged for per-item verification.

## Decontamination policy (2026-07-08 user directive — surgical, not blanket-delete)

- **`drop_row`** (65 rows) — discovery **heuristic/evolutionary-search & record constructions** (slug
  suffixes `-record`/`-autoevolver-record`/`-frontier-largeN`/`-funsearch-evolved`/`-slsqp`/`-sa-*`/
  `-basinhop`/`-multistart`/`-grid-baseline`/`-goldberg-optimal` …) **+ AHC039** (method + trajectory).
  These are overfit to the eval instance → removed entirely.
- **`drop_finale_turn`** (46 trajectories, **on the public-140 only**) — MLS same-task trajectories that
  inject a **non-native stronger baseline** as a `finale` rung: keep the trajectory (baseline ladder,
  which the public task discloses), drop only that `多出来的更强的` rung. Offenders on unpublished MLS
  tasks (5) are spared. Reference: `public_140_tasks.json`.
- **`keep`** — MLS baseline ladders (same-task but the ladder the public task itself discloses); **paper** discovery
  methods (`circle-packing-in-square`, `cap-set`, `kissing-number`, `fast-matrix-multiplication`,
  `erdos-minimum-overlap`, `autocorrelation-inequalities`, `heilbronn-triangle`, LABS); the 39 standalone
  finale methods; **FCS-research** reconstructions (per directive, may not be evaluated); **v4/wave2**
  synthetic (n-gram re-check hit only boilerplate). Kept examples that still *leak* are `keep_flagged`.

## Files

| file | what |
|---|---|
| `eval_registry.json` | Curated map: eval-task families → the training slugs that reconstruct them. Edit to tune the audit. |
| `audit_leakage.py` | Deterministic auditor. Writes annotations + `decontam_rules.json` + `benchmark_denylist.txt`. **Re-run after editing the registry.** |
| `decontam_rules.json` | **The gate `build_sft.py` reads.** `drop_method_slugs`(28) / `drop_traj_slugs`(8) / `type1_finale_traj`(46) / `keep_paper_methods`. |
| `leakage_tags_sft.jsonl` | Per-row annotation, line-aligned with `sft/innovation_sft.jsonl(.gz)` (2698). Fields incl. `leak, decontam_action, type1_nonnative_finale, exclude_from_summary, benchmarks[], family, severity, reason`. |
| `leakage_tags_wave2.jsonl` / `leakage_tags_v4.jsonl` | " for `innovation_wave2_sft` (758) / `innovation_v4_sft` (346). |
| `mls_type1_nonnative_finale.json` | The 51 MLS non-native-finale trajectories tagged on_public_140 (46 drop_finale / 5 spared), with the injected method. |
| `benchmark_denylist.txt` | The 36 slugs removed entirely (28 method + 8 traj). |
| `clean_rebuilt/innovation_sft.jsonl(.gz)` | **Decontaminated duplicate** (2587 rows), produced by the `build_sft.py` gate. **Clean training set = this + original `sft/innovation_wave2_sft` + original `sft/innovation_v4_sft`.** |
| `summary.json` | Rollup + `decontam_action` distribution. |

## Usage

```bash
# regenerate annotations + rules (safe; touches nothing under sft/)
python3 decontam/audit_leakage.py

# build the clean SFT (gate is ON by default; writes to a review path, NOT over sft/)
SFT_OUT=decontam/clean_rebuilt/innovation_sft.jsonl \
SFT_TAGS_OUT=decontam/clean_rebuilt/_sft_tags.jsonl \
  python3 sft/build_sft.py            # -> 2587 rows; INNOVATION_DECONTAM=0 disables the gate

# once happy, the plain `python3 sft/build_sft.py` writes the clean version to sft/ (gate default-on).
# summarizing training data: skip rows with decontam_action in {drop_row, drop_finale_turn},
# or exclude_from_summary == true.
```

## Caveats

- ALE-Bench eval-set membership of **AHC039** was not locally confirmable (no manifest on disk);
  dropped per the "delete if identical to ALE" directive since it's a faithful AHC039 reconstruction.
- Kept families (FCS-research, discovery-paper, v4, wave2) are kept **per the 2026-07-08 directive**. If an
  eval scope changes (e.g. you decide to run FCS-research), move that family from keep→drop in
  `eval_registry.json` and re-run.
- The register-matched `v4`/`wave2` synthetic sets need per-item verification (a subagent/LLM pass), not
  blanket removal; the annotations mark them for that.
