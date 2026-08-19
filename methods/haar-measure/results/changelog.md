# haar-measure changelog

## 2026-08-18 — svfix W3_notes_unclear
- Retrieved and verified the two sources `notes/sources.md` had cited without any file on disk:
  Moritz Tornier's arXiv:2006.10956 and (the actual paper, not "lecture notes") Floris van Doorn's
  "Formalized Haar Measure" (ITP 2021 / arXiv:2102.07636). Saved to `refs/` with `.txt` extracts.
- Corrected a factual error in the modular-function convention, present identically in
  `results/reasoning.md`, `results/answer.md`, and `results/train_answer.md`: all three asserted
  "with the common convention, `mu(Ea) = Delta(a)^{-1} mu(E)`". Tornier's own definition (Eq. (M),
  p.9) is `mu(Eg) = Delta_G(g) mu(E)` -- no inverse. Independently re-derived on the classical
  `ax+b` affine group (left Haar measure `da db / a^2`) before editing: right-translating by
  `(a0,b0)` scales measure by `1/a0`, confirming the scalar itself (no inversion) is the standard
  `Delta`. Also cross-checked against Wikipedia's raw-formula definition of the modular function,
  converted from right- to left-Haar-measure form. Changed all three occurrences to
  `mu(Ea) = Delta(a) mu(E)`; no other content in any file depended on the old sign, so nothing else
  changed.
- Corrected `notes/sources.md`'s prior mischaracterization of the van Doorn source (it does not
  ground "the passage from covering numbers to positive functionals on C_c(G)" -- van Doorn's own
  existence proof stays entirely at the level of compact-set covering numbers via a Tychonoff
  compactness limit and never invokes C_c(G) or Riesz representation; that is the classical
  Weil/Halmos functional route the trace actually uses, a different, equally standard, proof of the
  same theorem).
- Decisive step (covering gauge `[f:g]` -> normalized `I_g(f)` -> compactness limit -> Riesz
  representation) judged sound as is and left unchanged: it is genuinely derived (obvious first
  move -- raw covering numbers on compact sets -- fails for the stated, checkable reason that
  covering numbers are subadditive, not additive), and that exact obstacle is now independently
  corroborated by van Doorn's Lemma 4 (`hU(K u K') <= hU(K) + hU(K')`, equality only under a
  disjointness-with-margin condition) via a genuinely different construction route. No source
  citation was grafted onto that passage.
