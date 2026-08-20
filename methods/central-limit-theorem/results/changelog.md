# Changelog

## 2026-08-19 — svfix(W3_primary_plus_ancestors) quality-gate audit

- Reviewed the decisive step (discovery of the Lindeberg small-tail condition
  via the self-built bimodal counterexample: a variance-1/2 point mass at
  `+-1/sqrt(2)` plus a variance-1/2 block of `n-1` shrinking `+-m` terms).
  This step is genuinely derived on the page via a worked, checkable
  counterexample (reproduced independently: `P(|S|<0.3)` from the described
  Monte Carlo setup matches the trace's reported `0.2073` to four significant
  figures) followed by an on-page analytic explanation of the failure. No
  external source citation is required or added; grafting a Lindeberg/Durrett/
  Polya/De Moivre name-drop onto this self-derived computation would be
  decorative, not load-bearing. Outcome: sound_as_is.
- Fixed one factual error found while reviewing the file: the Taylor-bound
  passage claimed the branches of
  `min(|x|^3/6, x^2)` cross "near `|x|=3`"; the actual crossing is exactly
  `|x|=6` (`|x|^3/6 = x^2` iff `|x| = 6`), which is also why the trace's own
  numeric check range is `x in [-6,6]`. Corrected the prose to state the
  exact crossing point. Verified independently: `max_{x in [-6,6]} (|e^{ix} -
  (1+ix-x^2/2)| - min(|x|^3/6, x^2)) = 0.0`, consistent with the trace's own
  check. No other file references the old claim.
