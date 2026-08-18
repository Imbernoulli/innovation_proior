# ISTA / FISTA changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, two insertions grounded in Beck & Teboulle's own later expository chapter
  (`refs/self_accounts/beck-teboulle-2010-gradient-based-algorithms-chapter.pdf`; quotes in
  `notes/source_matrix.md`), which the earlier passes had not located:
  (1) before the composite rewrite — why the route in must be the *model* reading of the gradient step
  rather than the *direction* reading, since `F = f + g` has no steepest-descent direction exactly at
  the points the `L1` term drives the solution toward. This is the authors' own stated reason for
  switching interpretations, and the trace previously jumped straight to the majorization.
  (2) before the acceleration — the alternatives actually on the table for "use memory" (conjugate
  gradients, which already builds steps from the two previous iterates, and Shor's R-algorithm for the
  nonsmooth case) and the checkable reasons they were rejected: neither is proven to beat `O(1/k)`, and
  both carry per-iteration matrix operations, which is the same cost that ruled out the Newton-type
  route on the first page. This is why the extrapolation had to stay a closed-form vector operation.
- No factual errors found; the `t_k` recurrence, its numeric iterates, the anchor formula, the
  backtracking argument, the landing and the code are unchanged.
