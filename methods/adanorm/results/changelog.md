# Changelog

## 2026-08-19 — svfix(W3_primary_plus_ancestors) verification pass
- Quality-gate review of the decisive step TRIAGE flagged (class B): the
  claim that a zero-initialized **gate**, not `scale = 0` alone, is what
  makes a residual block the identity function at init. Confirmed genuinely
  derived: reasoning.md works a concrete counterexample by hand
  (`x = [1,2,3,4]` -> `LN(x)` nonzero -> `scale=0,shift=0` leaves `LN(x)`
  live -> a random sublayer maps it to a nonzero branch -> `x + branch !=
  x`; but `gate=0` zeroes the whole branch -> `x + 0 = x` exactly), and the
  arithmetic checks out to the stated precision.
- Cross-checked the same distinction against `src/main.tex` (the DiT paper's
  own LaTeX source, already on disk): its "adaLN-Zero block" paragraph
  states directly that regressing `gamma, beta` alone (vanilla adaLN, no
  gate) does not give identity init, and that the separate zero-initialized
  `alpha` ("applied immediately prior to any residual connections... We
  initialize the MLP to output the zero-vector for all alpha; this
  initializes the full DiT block as the identity function") is what does.
  This is the primary's own version of the exact gamma/beta-vs-alpha
  distinction reasoning.md re-derives by hand — confirmation, not
  contradiction.
- Also verified word-for-word: Goyal et al.'s zero-gamma claim
  (`refs/goyal_1hour.txt` lines 409-415) and the ADM AdaGN formula
  (`refs/adm.txt` line 292) both match reasoning.md exactly.
- Checked for a self-account per TRIAGE and `notes/synthesis.md`'s existing
  note ("no author Nobel/Turing-style self-account exists"); confirmed no
  hit in `SELF_ACCOUNT_SOURCES.md`. No missing struggle-account to graft in.
- No rewrite: the step is already correctly self-derived and is
  corroborated, not contradicted, by the primary source already on disk.
  Grafting a citation onto it would be decorative (the derivation does not
  depend on external material to hold). No factual errors found. See
  `notes/sources.md` for the full write-up and quotes.
