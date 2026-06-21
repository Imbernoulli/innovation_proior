# Sophia Synthesis

This file is retained as a pointer for older workflows. The strict, source-grounded synthesis for the repaired deliverables is `methods/sophia/notes/discovery_synthesis.md`; the source-by-source evidence table is `methods/sophia/notes/source_matrix.md`.

Key audit corrections from the prior draft:

- Negative curvature is made safe by the positive denominator floor or PSD estimator plus clipping; clipping a raw negative-curvature Newton ratio alone would only cap the wrong-sign update.
- The convex theorem is for exact full-Hessian clipped Newton in the Hessian eigenbasis, not for the stochastic diagonal implementation.
- The official SophiaG implementation stores unscaled sampled-label `grad * grad` in `hessian` and restores the GNB batch/token factor through `bs` in `rho * bs * hessian + 1e-15`.
- Paper defaults and code defaults differ slightly: the paper reports `beta1=0.96` and `epsilon=1e-12`, while the official code/configs use `beta1=0.965` and `1e-15`.
