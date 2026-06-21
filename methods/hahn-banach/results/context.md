# Context: extending linear functionals without losing domination

## Research question

A linear functional is easy to understand on a small subspace: assign values consistently with addition and scalar multiplication. The question is whether such a functional can be continued to a larger vector space while still respecting the same bound that controlled it on the subspace.

The clean problem is this. Let `E` be a real vector space, let `M` be a subspace, let `p:E->R` be sublinear (`p(x+y)<=p(x)+p(y)`, `p(alpha x)=alpha p(x)` for `alpha>=0`), and let `f:M->R` be linear with `f(x)<=p(x)` on `M`. Is there a linear `F:E->R` extending `f` and still satisfying `F(x)<=p(x)` on all of `E`?

In a normed space this asks for a norm-preserving extension of a continuous linear functional from a subspace. In geometric language it asks whether a convex set can be separated from a point or subspace by a linear functional. The setting offers local linear information and asks for global linear information, with no formula for the missing values and no finite basis available in general.

## Background

Finite-dimensional linear algebra supplies extensions by choosing a basis. Once a value is assigned to a new basis vector `z`, every vector on the enlarged subspace has the form `x+t z`, so the single new scalar interacts with infinitely many vectors at once, and any domination constraint must hold for all of them.

Earlier functional analysis already treated special extension and moment problems. Banach's 1929 paper explicitly cites earlier special cases by F. Riesz, Hahn, and Helly in a footnote to its main extension theorem, and stresses that the proof is simple and does not assume completeness or separability of the ambient normed vector space.

The basic step is one-dimensional. If `z notin M`, an extension to `M+R z` has the form `F(x+t z)=f(x)+t c`, so the task is to choose one real number `c` so that `F<=p`. The choice of `c` is constrained by the inequalities `F(x+t z)<=p(x+t z)` for every `x in M` and every real `t`.

In an infinite-dimensional space there may be no countable sequence of missing directions. Banach's original proof uses transfinite induction after well-ordering the missing elements; the modern formulation phrases the same passage to the whole space through Zorn's lemma.

## Baselines

- **Basis extension without bounds.** Extend a Hamel basis of `M` to a basis of `E`, assign arbitrary values on the new basis vectors, and extend linearly. This solves the algebraic extension: it produces a linear functional on `E` agreeing with `f`.

- **Finite-dimensional separating hyperplanes.** Convex geometry in finite dimensions separates a point from a convex set by a hyperplane, using compactness and finite-dimensional geometry.

- **Special function-space extension theorems.** The earlier results cited by Banach cover particular classes of spaces or systems, extending linear data within those settings.

- **Normed-space continuity estimates.** If a linear functional is already defined on the whole normed space, continuity is equivalent to a bound `|f(x)|<=C||x||`.

## Evaluation settings

This is a theorem, so success is logical rather than empirical.

- **Analytic setting.** Real vector spaces with a sublinear dominating functional; normed spaces as the symmetric special case `p(x)=C||x||`; subspaces of arbitrary dimension.

- **Geometric setting.** Open convex sets disjoint from a subspace, or a point outside a closed subspace, where success means producing a nonzero linear functional whose kernel is a separating hyperplane.

- **Stress cases.** Infinite-dimensional spaces; nonseparable spaces; subspaces with no closed complement; normed spaces where the extension must preserve the exact norm.

- **Standard of success.** The one-step extension inequalities must be correct for all `x+t z`; chains of partial extensions must have valid upper bounds; the construction must reach all of `E`; and the separation corollary must arise from a genuine linear functional, not from a finite-dimensional picture.

## Code framework

The artifact is a theorem and proof. The reusable proof scaffold is:

1. Express the desired bound as domination by a sublinear functional.
2. Establish a one-dimensional extension step that chooses the new value while preserving domination.
3. Partially order all dominated extensions and pass to a global extension.
4. Read the resulting global functional geometrically as a separating hyperplane through its kernel or a level set.
