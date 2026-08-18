# Changelog

- 2026-08-17: `notes/`, `refs/`, `src/` did not exist for this method at all. Fetched the primary
  (arXiv 2012.12821, `refs/primary/ffl_2012.12821.pdf` + pdftotext `.txt`) and the official code repo
  (github.com/EndlessSora/focal-frequency-loss: `refs/primary/focal_frequency_loss.py`,
  `refs/self_accounts/README.md`). Added `notes/sources.md`.
- 2026-08-17: `methods/focal-frequency-loss/results/reasoning.md:80-88` cross-checked the claimed
  `(alpha+2)/2` gradient-inflation-from-not-detaching factor two ways: (1) against the primary's own
  statement (Section 3.3, immediately before Eq. 10) that "the gradient through the spectrum weight
  matrix is locked, so it only serves as the weight for each frequency" — confirms the detach itself,
  though the paper does not state the `(alpha+2)/2` consequence; (2) by hand (chain rule) and with
  independent `torch.autograd` runs, both on the 1D scalar residual already in the file and on the
  actual two-real-parameter complex object `(a_f, b_f)` (not previously checked). Both give exactly
  `1.5` at `alpha=1` — the existing claim was correct; no numeric error found. Extended the passage
  (added 2 sentences, deleted nothing) to report the complex-object check alongside the existing 1D
  check, so the constant-factor claim is no longer resting on the 1D proxy alone.
- 2026-08-17: Verified the official code's `loss_formulation` (`focal_frequency_loss.py:60-94`,
  in particular the `weight_matrix = matrix_tmp.clone().detach()` line) matches
  `results/reasoning.md`'s and `results/answer.md`'s/`results/train_answer.md`'s landing code
  structurally line-for-line; no code changes needed in any of the four results files.
