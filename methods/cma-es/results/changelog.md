# CMA-ES changelog

## 2026-08-18 — obs-fix (repair pass, reasoning.md)
- The prior obs-fix pass on this slug (VERDICT=R) had adjudicated the flagged old-mean-vs-EMNA
  passage as `kept_desk_check` and made no edit; a repair pass rejected that call. The passage
  ("Let me actually run it: sample λ=40 isotropic points ... averaged over a few thousand
  trials" → "Old-mean reference gives variance ≈1.82 ... Own-mean (EMNA) reference gives ≈0.23")
  is a stochastic Monte-Carlo simulation reporting specific empirical numbers that validate the
  method's own core design choice (old-mean vs EMNA reference for the rank-μ update) — a
  narrator-run experiment outcome, not a deterministic on-page check. Rewrote it as: the two
  candidate references stated as competing hypotheses → a discriminating test design (isotropic
  offspring on a linear slope, matched λ/μ/selected-points, only the reference differing) →
  each hypothesis's prediction derived analytically from the reference's construction (EMNA is
  by construction the minimizer of within-cloud spread, so it must shrink variance along the
  descent axis; the old-mean reference references selected steps against a stale center, so it
  must inflate variance there) → the decision rule (ship the reference predicted to inflate
  along the productive direction; the online check is whether the estimated variance along the
  true descent direction grows rather than shrinks during a real run). Numbers removed.
- A broader grep of `reasoning.md` (prompted by the same repair pass) found two more unaddressed
  violations the first obs-fix pass never touched:
  - `L87`: "Running the code below on both: the plain ellipsoid reaches f ≈ 7.7e-11 and the
    rotated one reaches f ≈ 3.8e-11" — a full end-to-end run of the finished CMA-ES algorithm on
    a benchmark (rotated vs. unrotated ill-conditioned ellipsoid, n=10) reporting quantitative
    performance, i.e. own-method result reporting. Rewrote as a discriminating test design
    (matched start/seed/budget/target, rotation the only difference) → the falsifiable prediction
    (rotation-invariance ⇒ evaluation count to target stays flat across the rotation; its absence
    would show up as the rotated run needing asymptotically more evaluations or missing the
    target within budget) → the decision rule (accept the affine-adapting design if the two
    evaluation counts land in the same ballpark; otherwise revisit C's learning rate). Numbers
    removed.
  - `L208` (causal-chain summary): repeated both of the above as settled outcomes ("the side
    experiment showed ... ≈1.8× ... ≈0.23×" and "as the rotated-vs-unrotated ellipsoid run
    confirmed empirically"). Reworded to match the design+prediction framing above — construction
    *predicts* the variance-inflation direction (checked online), and affine invariance is a
    property the matched-rotation comparison is designed to test, not one already confirmed.
- Left untouched (adjudicated ALLOWED, not violations, under the rule's desk-scale carve-out):
  `L69` ("I also checked it numerically on a random 3×3 C with three random steps — the two sides
  agree to ~10⁻¹⁵") is a tiny deterministic cross-check that a symbolic rewrite of the rank-μ
  update is an algebraic identity — no method-performance claim, squarely the allowed "tiny
  deterministic code check" category. `L79` ("I re-ran the same Monte-Carlo as before ... the
  empirical E‖p_σ‖² came out 4.99 for n = 5") is the same category as the untouched, unflagged
  p_c-stationarity check earlier in the file (`diag ≈ (1.00, 4.01, 0.25, 8.99, 1.01)` against a
  target derived in closed form): a small simulation confirming an already analytically-derived
  formula (E‖p_σ‖² = n by the whitening argument), not a claim about the method's real-world
  performance. Both fall under "small Monte-Carlo over a handful of draws" / "on-page
  computation" per the obs-fix rule's allowed list, so kept as-is.
- `answer.md` and `train_answer.md` were checked and contain no instance of the removed numbers
  or of "our experiments/ablations show ..."-style claims; `context.md` does not carry the
  removed numbers as pre-existing facts either, so no changes were needed in those channels.
