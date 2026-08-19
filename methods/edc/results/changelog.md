# Changelog

## 2026-08-18 — svfix(W3_reconstructed)
- **reasoning.md**: the opacity-factor (`0.6`) decisive step was framed as a closed-form
  compromise between two computed objectives, landing at "around 0.6" with logical-derivation
  tone. The primary paper (src/sec/4_experiments.tex) explicitly states 0.85/0.6 were
  "determined as extrema through testing" (an empirical sweep), not solved for. Restaged the
  step: keep the honest math that bounds the two competing pulls (transmittance-matching
  ~0.51-0.65 vs spatial-mass ~0.8-1.0), then explicitly hand off to a sweep over trained runs
  and report the real ablation numbers from the primary (SSIM 0.821->0.823, PSNR 27.67->27.72,
  LPIPS 0.217->0.216; 0.6/0.85 as extrema, +-0.05 robust), then interpret which of the two
  competing arguments the empirical winner actually favors. Same tightening applied to the
  0.85 short-axis-factor passage (already honestly hedged, now explicitly tied to the same
  sweep result instead of left open-ended).
- **Factual error fixed** (found while checking the above): answer.md and train_answer.md both
  stated 0.6 "is the interior minimum of the mismatch between the parent's unimodal density and
  the children's bimodal density" — this contradicts reasoning.md's own computation, which
  finds that spatial-mismatch minimum at f ≈ 0.84, not 0.6 (0.6 sits in the *transmittance*-
  matching band, a different, competing objective). Corrected in both files, and in answer.md's
  RevisingGS comparison paragraph which had inherited the same mix-up.
- No new non-primary source was needed or added; a real multi-venue search (project page,
  GitHub issues, alphaxiv, arXiv v1-vs-current diff) found no author self-account material for
  EDC (see notes/sources.md for the search log). Long-axis-split geometry (halving/offset/
  split-only/recovery-aware-pruning) reasoning was already sound and is unchanged.

## 2026-08-18 — svfix(epistemic)
- The previous pass (above) had the narrator report the sweep's own real ablation numbers
  (SSIM/PSNR/LPIPS, "the sweep below confirms `0.85`/`0.6` as the extremum") as an
  already-observed outcome in reasoning.md/answer.md/train_answer.md. At that point in the
  frame the proposal's own experiments haven't happened yet, so a narrator that reports its own
  method's results is out of voice regardless of whether the numbers are real. Removed the
  reported numbers and the outcome-dependent conclusions ("0.6 lands at the low end, telling me
  which argument the renderer cares about") from all three files.
- Kept: the math bounding the two competing objectives (spatial-mass minimum at `f ≈ 0.84` vs.
  transmittance band `0.51-0.65`), the discriminating-experiment design (matched trained runs,
  everything but the factor held fixed), each hypothesis's prediction (winner near 0.5-0.6 vs.
  near 0.8-1.0), and the decision rule (whichever value survives the sweep is the one to ship).
  `0.6`/`0.85` remain in the text as the narrator's working proposal — a reasoned bet toward the
  transmittance side, justified only by the on-page transmittance-vs-full-opacity algebra, not
  by a reported result — pending that sweep.
- The landing is honestly under-justified now that the observation is gone; this method unit is
  queued for the trajectory-conversion pass to supply the actual sweep as an observation turn.

## 2026-08-18 — svfix(epistemic) repair
- The prior pass rewrote the prose but missed a matching claim in the Python code comment
  shipped identically in all three files: `# Long-Axis Split constants (sweep extrema; +-0.05
  [is] robust)`. That comment asserted, as an already-established fact, that 0.6/0.85 were the
  measured outcome of a sweep and that a +-0.05 robustness band had already been checked —
  contradicting the prose immediately above it, which (post-fix) says the sweep hasn't happened
  yet and 0.6/0.85 are only the working proposal pending it. Reworded the comment in
  reasoning.md, answer.md, and train_answer.md to `# Long-Axis Split constants (proposed
  values; exact extrema pending a sweep)` — no numeric or behavioral change, comment-only.
- Re-checked all three files end to end for any other surviving "already observed" language;
  none found. `tools/lint_inframe.py` reports zero hits for methods/edc/.
