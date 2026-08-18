# concat-film (FiLM) changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, the FiLM-generator passage: the trace previously asserted that "the
  modulation layer itself should not care" which network emits `(gamma, beta)` and moved on. That is
  precisely the belief the authors' own retrospective records as the one that had to be corrected.
  The passage now runs through the documented obstacle and its resolution, grounded in Ethan Perez's
  first-person retrospective (`refs/self_accounts/film_retrospective.html`, ML Retrospectives @
  NeurIPS 2019; quotes in `notes/sources.md`): every variant overfits with mediocre validation; the
  capacity lives in the layers *after* the affine, not in the affine (one early modulation layer
  ≈ four spread through the network); so the free parameters that matter are the generator's; the
  fix is a linear projection in place of the recurrent decoder plus real L2 weight decay, whose
  removal costs roughly ten points of answer accuracy. The primary mentions weight decay only as a
  buried hyperparameter (`src/Perez-Strub.tex:128`, `1e-5`) and never as load-bearing.
- No factual errors found elsewhere; the unification derivation, the landing and the code are
  unchanged.
