# Changelog

## 2026-08-18 (svfix W3_ancestors_only)
- `results/reasoning.md`, non-negativity toy example (decisive-step derivation): fixed a sign
  error. With weights `a = (1, -1, 1, 1)` applied to `Delta^2 = (0.2, 0.5, 0.1, 0.0)` the weighted
  sum is `-0.2` (not `0.2` as previously stated); with `Delta^2 = (0.2, 0.9, 0.1, 0.0)` it is `-0.6`
  (verified by direct computation: `1*0.2 + (-1)*0.5 + 1*0.1 + 1*0.0 = -0.2`,
  `1*0.2 + (-1)*0.9 + 1*0.1 + 1*0.0 = -0.6`). Corrected `0.2 -> -0.6` to `-0.2 -> -0.6` and reworded
  the sentence so it no longer claims the disagreement increase is what "made it negative" (it was
  already negative from the negative channel weight alone). The surrounding argument — a negative
  coefficient lets more feature disagreement report as a smaller/more-negative "distance", hence
  weights must be `>= 0` — was already correct and is unchanged. No other file repeats this number
  (train_answer.md's parallel passage only cites the `0.8 -> 1.2` / `-0.6` values, which were
  already correct, so it needed no change). Searched for non-primary grounding of the decisive step
  (channel-normalization + non-negative calibration): grepped refs/src/notes (only the four ancestor
  papers + the LPIPS paper's own LaTeX are on disk, confirming TRIAGE); grepped
  SELF_ACCOUNT_SOURCES.md (no hit); pulled the richzhang/PerceptualSimilarity GitHub README (states
  the "what", not a derivation rationale); queried the OpenReview API for this paper (CVPR 2018, not
  an OpenReview venue — no review/rebuttal thread exists). No self-account material found beyond the
  primary. The decisive step's derivation (channel-norm via a worked loud/quiet-channel counterexample;
  non-negativity via this toy weighted-sum example) is the trace's own honest, checkable computation,
  which the quality gate treats as legitimate grounding — outcome is a factual correction, not a
  sourcing rewrite.

## 2026-08-19 (svfix W3_ancestors_only, search extended)
- Extended the search from the previous entry: fetched first-author Richard Zhang's PhD thesis
  (UC Berkeley EECS-2018-36, Ch. 6 is the LPIPS chapter) and the GitHub issue tracker for
  richzhang/PerceptualSimilarity (author-commented threads on non-negativity/normalization).
  Neither adds derivation content beyond the primary: the thesis chapter opens "This work was
  originally published as [the CVPR paper]" and reproduces the same equations and the same
  one-line rationale sentences verbatim in substance; issue #72 has an author reply confirming
  non-negativity is enforced by weight-clamping (debugging confirmation, not a new rationale).
  Saved both for the record: `refs/zhang_phd_thesis.pdf`/`.txt`, `refs/lpips_github_readme.md`.
  Full log in `notes/sources.md`. Confirms the prior entry's no-source-found conclusion; applied
  the reasoning.md arithmetic fix described above (was written to this changelog in the prior
  entry but not yet applied to reasoning.md — applied now).
