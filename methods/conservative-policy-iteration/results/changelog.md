# Conservative Policy Iteration changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, the max-norm/greedy-degradation passage: previously "a greedy swap commits
  to its value estimate at every state at once ... locks in that error everywhere". Rewritten to carry
  the two sharper statements from Kakade's PhD thesis (a non-primary self-account already on disk but
  unused for this step; quotes in `notes/synthesis.md`):
  (1) the selection effect from the thesis's Discussion chapter — the greedy policy "may visit only
  those states where the worst case errors have occurred ... the reason it visits these states might
  be due to the errors themselves", i.e. an overestimate at a state is exactly what sends the new
  policy there, which is what turns a per-state ε into a horizon-compounding penalty; and
  (2) from the thesis's CPI chapter intro, that the difficulty is specific to *stationary* policies —
  a non-stationary update alters one timestep only and avoids max-norm bounds entirely, so the problem
  actually being solved is narrower than "approximate policy iteration degrades": it is that replacing
  the policy at all timesteps at once is what lets the error propagate.
- No factual errors found. The performance-difference identity, the α/α² split, the coupling on
  "no switch yet", the exact all-α bound, α* = (1−γ)𝔸/(4R), the landing and the code are unchanged;
  the thesis's own stated proof intuition for Lemma 7.2.2 matches the trace's derivation.
