# MI-FGSM changelog

## 2026-08-17 — source-value recheck
Two author self-accounts were located that the earlier pass wrongly recorded as non-existent
(Yinpeng Dong's Tsinghua PhD dissertation, Ch. 2; and §5.2 of the NIPS 2017 adversarial-competition
report, written by the same team). They confirm the trace's existing chain almost verbatim — greedy
per-iteration updates fall into poor local optima and therefore overfit the white-box model; the L1
normalization is there "because we noticed that the scale of the gradients varies in magnitude
between iterations"; the "holes" are the boundary-alignment ancestor's anomalous decision regions.
Quotes in `notes/source_matrix.md`. Two things they add, now in `results/reasoning.md`:
- A second mechanism for why the momentum version transfers better, stated in the dissertation and
  in neither the primary nor anywhere else on file: a stable update direction *increases the ℓ2 norm
  of the perturbation*. Written into the trace with the arithmetic that makes it checkable — a
  zig-zagging direction leaves each pixel at a random-walk displacement of order ε/√T while a
  consistent one saturates the ε clip, so ‖δ‖₂ goes from about ε√d/√T to ε√d at the same L∞ budget —
  and with the honest note that the two mechanisms are confounded in any success-rate number.
- The documented negative result absent from the primary: the targeted variant's examples showed no
  transferability at all, so that member of the family is a white-box / known-ensemble tool. The
  trace previously presented targeted attacks as a clean extension.
Also noted: `src/egpaper_final.tex:276` carries a commented-out co-author objection to the decisive
choice — "any reason for the L1 distance? looks arbitrary. how about L2/L_inf distance?" — which the
trace already answers ("the particular norm isn't sacred"), matching the competition report's
"however other norms will work too". No factual errors found; landing and code unchanged.
