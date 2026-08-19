# sac-rnd changelog

## 2026-08-18 — mechanism correction + epistemic fix (svfix pass, W3_notes_unclear)
- `results/reasoning.md` (the FiLM-vs-concat decisive step): the decisive step had two
  compounding defects. (1) Self-supplied observation: the narrator claimed to have
  "trained both a concat RND and a FiLM RND ... to convergence" and reported invented
  numeric results mid-reasoning (`‖∂b/∂a‖ ≈ 0.0062` vs `≈ 0.0103`; a gradient-descent
  "escape" trajectory from action norm `7.03 → 1.62` for concat, `6.92 → 0.86` for FiLM;
  bonus values `0.266 → 0.010` / `0.915 → 0.008`) — the exact "narrator claims to run an
  experiment mid-reasoning and states its result" pattern the epistemic rule bans,
  regardless of whether the numbers are real. (2) Factual/mechanistic error: the
  underlying story it invented to justify those numbers — that `[s,a]` concatenation
  *dilutes* the action at the input so the bonus becomes a flat, weakly-discriminating
  function of the action ("not discriminative enough") — directly contradicts the primary
  source. Section 3 of the actual paper ("Random Network Distillation is Discriminative
  Enough") reproduces the prior "not discriminative" claim using exactly `[s,a]`
  concatenation and refutes it: concat-conditioned RND already separates ID from OOD
  actions about as well as a trained critic-ensemble. The primary source's real diagnosis
  (Section 4, "Concatenation Prior Hinders Bonus *Minimization*") is that the actor
  cannot gradient-descend the concat-conditioned bonus back toward the data — an
  optimization-landscape problem, traced in Section 6.3 to the anti-gradient field being
  "noisy" under concatenation and "smooth ... over the entire available action space"
  under FiLM (Figure 4 caption). Rewrote the decisive step to: split "not discriminative
  enough" into two testable hypotheses (bad detector vs. bad-for-gradient-descent
  landscape); describe the frozen-snapshot detection probe (graded distributional-shift
  dial, ensemble-disagreement as the separation bar) and the critic-free actor-only
  minimization test (matching the primary source's actual Algorithm 2 ablation) as
  hypothesis → design → prediction → decision rule only, with no narrated numeric
  outcome; ground the FiLM-over-concat resolution in the anti-gradient-field
  noisy-vs-smooth structural argument instead of an invented magnitude measurement.
  Also fixed a smaller instance of the same self-supplied-observation pattern in the
  running-std normalization paragraph (fabricated "toy" bonus values `O(0.3–0.9) →
  O(10⁻⁵)`), replaced with the structural/definitional fact that MSE-regression error
  falls toward its own training objective's minimum on the training distribution.
- Propagated the same mechanism correction to `results/answer.md` ("Key idea" section)
  and `results/train_answer.md` (the FiLM-conditioning paragraph), which had asserted the
  same "concatenation is an escapable/featureless detector → not discriminative enough"
  claim as settled fact. Both now state: concat RND detects OOD actions fine in
  isolation; the real failure is that gradient descent through the concat-conditioned
  bonus does not reliably navigate back to the data because the anti-gradient field over
  the action is locally inconsistent, and FiLM's multiplicative conditioning gives nearby
  actions nearby gradients instead.
- No change to `results/context.md` (background/setup section was already neutral —
  states the RND-with-concatenation baseline and the concat/gated/bilinear/FiLM ablation
  axis as given facts, not as a mechanism claim) or to the code (architecture/algorithm
  unchanged; only the prose justification for the FiLM choice was corrected).
- Sources: `methods/sac-rnd/refs/primary/sac_rnd.txt` (arXiv 2301.13616v2, pdftotext
  -layout), see `notes/sources.md` for quotes and the self-account search log (OpenReview
  API, GitHub issues/code, HuggingFace papers page, OpenAlex — no genuine author
  self-account surfaced; WebSearch quota was exhausted for the session).
