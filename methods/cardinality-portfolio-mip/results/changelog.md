# cardinality-portfolio-mip changelog

## 2026-08-20 — reconcile lost edit (svfix W3_reconstructed, re-pass)
The 2026-08-18 entry below was researched and written correctly (this file + `notes/sources.md`
committed to disk), but the corresponding `results/reasoning.md` rewrite was never committed and
was subsequently overwritten by an unrelated recovery commit (`528dc3b7e`, "recover audit-edit
method:cardinality-portfolio-mip") that reconstructed the file from an older session snapshot —
so the file on disk still carried the pre-fix "Pivoting methods such as Lemke's method are
attractive there" sentence, attached directly to the older branch-on-`x_s` scheme. Re-applied the
rewrite described below (verified independently against the same two source files) and committed
`reasoning.md` + `notes/sources.md` + this changelog together in one commit this time.

## 2026-08-18 — source-value fix (svfix W3_reconstructed)
- `results/reasoning.md` (paragraph beginning "There is an older continuous-variable way..."): the
  decisive step (binary indicator `y_i` + `alpha_i y_i <= x_i <= u_i y_i` linking, `sum y_i <= K`)
  had no source grounding anywhere in the trace. Verified it against Chang, Meade, Beasley & Sharaiha
  2000's canonical cardinality+quantity binary-linking MIP (as restated verbatim in
  `refs/chang_localsearch.txt`, Schaerf 2001, eqs 4-6) and against Bertsimas & Shioda 2009's problem
  statement (`refs/bertsimas_shioda.txt`, problem (1)) — the trace's landing matches the documented
  formulation term for term; see `notes/sources.md`.
- Same paragraph, the Bienstock branch-on-x_i aside: fixed a factual conflation. The text attached
  "Pivoting methods such as Lemke's method" directly to the older/original branch-on-x_i scheme, but
  per Bertsimas & Shioda's account of Bienstock 1996 (`refs/bertsimas_shioda.txt` lines 71-82), the
  original scheme's node warm-start was a primal-feasible descent method (Newton / steepest descent /
  Frank-Wolfe) with a quadratic-penalty term, not pivoting — Lemke's pivoting is Bertsimas-Shioda's own
  later refinement on top of that scheme (lines 81-83), a separate idea. Rewrote the passage to
  attribute the descent+penalty warm-start to the older scheme and introduce pivoting as a distinct
  alternative, rather than merging the two solvers into one. Landing (binary MIQP + code) unchanged.
- `refs/bienstock.pdf` remains a broken placeholder (16-byte Cloudflare error page, not a real PDF);
  Bienstock 1996 (Math. Program. 74:121-140) is Springer-paywalled and no OA copy was found this pass
  either (search log in `notes/sources.md`). The aside is grounded via Bertsimas-Shioda's documented
  account of it, which the method's own primary source explicitly cites and builds on.
