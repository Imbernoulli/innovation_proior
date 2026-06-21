## Research question

For a differentiable map `f : U -> R^m`, with `U` open in `R^n`, the points where `Df_x` fails to be onto are precisely the places where the usual regular-level-set picture can break. The direct set of such points can be huge: for a constant map, every point is critical. The useful question is therefore not whether the critical points themselves are rare, but whether their images in the target are rare. Can differentiability force the set of critical values to have Lebesgue measure zero, even when the critical set in the domain is large?

The target-space formulation matters because regular values are the values for which inverse images have controlled geometry. If almost every target value is regular, then the pathological values are negligible for integration, perturbation, generic-position arguments, and level-set constructions.

## Background

A differentiable map is locally governed first by its derivative. When `Df_x` has full rank `m`, the map is a submersion near `x`, and the implicit/submersion theorem gives a clean local model: after coordinates are chosen, the map looks like projection onto `m` coordinates. Fibers over nearby regular values are smooth manifolds of dimension `n-m`.

At a critical point the derivative has rank `< m`, so the first-order image lies in a proper linear subspace of the target. The pointwise rank defect alone is not enough: critical points can form an open set for a constant map, or a large closed set for smoother examples. The crucial analytic resource is stronger than pointwise linear algebra: Taylor expansion says that, on a small box, a function whose low-order derivatives are constrained cannot spread freely in all target directions. The local image is flattened to first order, and higher-order remainders give quantitative thickness estimates.

The proof problem is globalizing that local squeeze. One has to cover critical points by small boxes or charts, estimate how much target volume their images can occupy, and sum the estimates over a countable cover. The delicate part is that different points may have different ranks or different orders of vanishing, so the argument must separate the critical set into pieces where either rank can be straightened or higher-order Taylor control applies.

The downstream geometric payoff is regular-value abundance. If critical values have measure zero, then almost every value is regular. This supports the regular level-set theorem and, in manifold language, the common transversality pattern: after excluding a negligible set of bad parameters or values, inverse images behave as smooth submanifolds.

## Baselines

- **Inverse and implicit function theorems.** These give complete local structure at points where the derivative has maximal rank. In coordinates, a full-rank map becomes a projection, and fibers are smooth manifolds. Gap: the theorems say nothing quantitative about the target image of the rank-deficient locus.

- **Constant-rank theorem.** If the rank is locally constant, the map can be straightened so that its image locally lies in a lower-dimensional coordinate plane. That immediately suggests zero target volume when the local rank is `< m`. Gap: critical sets need not have constant rank, and the rank can change on complicated subsets.

- **Naive small-critical-set strategy.** One might try to prove the critical points themselves are small. This is false: a constant map has every point critical, and rank-zero behavior can persist on large subsets. Gap: the theorem must measure the image, not the domain set.

- **Change-of-variables intuition.** For maps `R^m -> R^m`, a nonzero Jacobian determinant preserves positive `m`-dimensional volume locally, while determinant zero suggests volume collapse. Gap: determinant language only covers square maps and does not by itself handle maps between different dimensions or the global union of critical pieces.

- **Continuity alone.** Purely continuous maps can behave too wildly for a volume-squeezing conclusion; differentiability is the load-bearing hypothesis that converts rank deficiency into local flattening. Gap: without Taylor control, there is no reliable estimate on how much target volume a small domain box can fill.

## Evaluation settings

The artifact is a theorem and proof, so evaluation is logical rather than empirical. Natural checks include:

- `f` constant: all points are critical, but the critical value set is a single point.
- `f : R^n -> R^m` with `n < m`: every point is critical, yet a differentiable image of `n`-dimensional pieces should occupy no `m`-dimensional volume.
- Fold-type maps such as `(x,y) -> (x,y^2)`: the critical set is a line and its image is a line, not an area-filling set.
- Projection maps with full rank: regular values dominate and fibers have the expected dimension.
- Smooth maps on manifolds: local coordinate charts should reduce the statement to the Euclidean result, and regular values should give smooth inverse-image manifolds.

The proof must account for compact localization, countable covers, variable rank, Taylor remainders, and the passage from Euclidean charts to manifolds.

## Code framework

There is no computational implementation to reproduce. The field-appropriate scaffold is a proof skeleton whose empty slots are analytic estimates and induction steps.

```text
Input:
  smooth map f : U subset R^n -> R^m
  compact set K inside U

Definitions:
  critical point: rank Df_x < m
  critical value: y = f(x) for at least one critical point x

Proof scaffold:
  1. Localize K by finitely many coordinate boxes.
  2. Separate critical points by local rank.
  3. Where positive rank is present, straighten the nondegenerate coordinates.
  4. Reduce the remaining critical-value estimate to lower-dimensional slices.
  5. Where rank is zero, separate by order of derivative vanishing.
  6. Use Taylor expansion on small boxes to bound image thickness.
  7. Sum the covering estimates and pass to countable unions.
  8. Transfer the Euclidean statement to manifold charts.
```
