# Changelog — kernel-ridge

## 2026-08-19 — svfix(W3_primary_plus_ancestors) — numeric error fix
- results/reasoning.md, rank-2 singular-`K` illustration (duplicate row 3 = row 1 of `Phi`,
  `a = 0.5`): the stated coefficient vector `c = [-0.067, -0.4406, 0.533]` and the resulting
  predictions `k(x)^T c = k(x)^T c2 = 0.51675` did not satisfy `(K + aI)c = y` for the `K`
  built from that `Phi` (residual ~0.7 on the third row of the linear system). Recomputed
  directly from the stated `Phi`/`y`/`a`: `c = [-0.7208, -0.4975, 1.2792]`,
  `k(x)^T c = k(x)^T c2 = 0.6802`. The qualitative claim the passage makes (the fitted
  values `Kc`/`Kc2` and the held-out prediction are unchanged by adding a nullspace vector
  to `c`) still holds with the corrected numbers; only the printed constants were wrong.
  Eigenvalues `[0, 2.60, 15.40]` were already correct and unchanged.
- Grounding/decisive-step assessment: the push-through identity
  `(PQ+aI)^-1 P = P(QP+aI)^-1` is derived on the page from first principles (`PQP+aP`
  equality) and independently verified against two numerically distinct solves
  (`max|diff| ~ 4e-16`) before being trusted — this is the trace's own honest computation,
  not an assertion, so it passes the quality gate without grafting a citation. Checked
  against the primary (Saunders/Gammerman/Vovk 1998): the primary reaches the same
  feature-dim -> example-dim result (Lemma 1 / eq. 8) but via Lagrangian/KKT duality, not
  via this push-through identity — the identity itself is the route used by the modern
  explainer already on disk (refs/explainers/khan_2015_kernel_ridge_regression.txt), not by
  the primary. No source added to the trace since the step needs none: it is self-derived
  and self-verified in-frame.
