# Changelog

## 2026-08-19 — svfix(W3_ancestors_only)
Fixed a factual/procedural error in the 4-city unit-square worked example in
`results/reasoning.md` (the numeric check of the gain bookkeeping, right after the
6-city forced-`x_i` hand trace). The trace had picked `y_1 = (2,1)`, but `(2,1)` was
still a tour link at `t_2 = 2` at that point — the primary source is explicit that
`y_1` "is not allowed to be either of the edges already connected to `t_2`" (Lin &
Kernighan 1973, p.505, "An Example" section; `refs/lk1973.pdf`), so that choice
duplicates an existing tour edge instead of adding a link from outside the tour.
The only valid target was city 3. Corrected `y_1 = (2,3)`, `t_3 = 3`, and the forced
`x_2 = (1,3)` (the tour's other link at `t_3`, `(3,0)`, runs straight back to `t_1`
and gives nothing to close up, so it was never a live second candidate). The
close-up gain, `G*`, and the final rebuilt tour (`0-1-2-3`, length 4) are numerically
unchanged since the unit square is symmetric under the 1<->3 relabeling; only the
city labels and the edge set of the cashed exchange (`(2,3)`/`(1,0)` in place of the
erroneous `(2,1)`/`(3,0)`) were corrected.

Sourcing verdict for the decisive step (variable-depth sequential exchange: forced
`x_i` via close-up feasibility + positive-partial-sum gain criterion): left as-is.
The gain criterion, its cyclic-permutation-lemma proof, and the forced-`x_i`
feasibility argument are all derived and proved on the page, straight from the
primary (Lin & Kernighan 1973 §1, pp.502-503), which the trace's own 6-city hand
trace and cyclic-permutation numeric check (`g = (3,-5,4,-1,2)`) independently
verify. `refs/` also holds `helsgaun_general_kopt.pdf`, `kopt_report.pdf`, and
`lkh_report.pdf`, but these describe LKH's later 5-opt/α-nearness machinery — a
documented descendant, not evidence for *this* method's own decisive step — and the
trace already treats them correctly as descendant-context only, never as the
source of the derivation. No external material was grafted on.
