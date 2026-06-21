# Context: extending linear functionals without losing domination

## Research question

A linear functional is easy to understand on a small subspace: assign values consistently with addition and scalar multiplication. The hard question is whether such a functional can be continued to a larger vector space without breaking the same bound that made it continuous or controlled in the first place.

The clean problem is this. Let `E` be a real vector space, let `M` be a subspace, let `p:E->R` be sublinear (`p(x+y)<=p(x)+p(y)`, `p(alpha x)=alpha p(x)` for `alpha>=0`), and let `f:M->R` be linear with `f(x)<=p(x)` on `M`. Must there be a linear `F:E->R` extending `f` and still satisfying `F(x)<=p(x)` on all of `E`?

In a normed space this asks for a norm-preserving extension of a continuous linear functional from a subspace. In geometric language it asks whether a convex set can be separated from a point or subspace by a linear functional. The same result would turn local linear information into global linear information, but with no formula for the missing values and no finite basis available in general.

## Background

Finite-dimensional linear algebra supplies extensions by choosing a basis, but that procedure does not protect a norm or domination constraint. Once a value is assigned to a new basis vector, every vector on the enlarged subspace changes by `x+t z`, so the single new scalar must satisfy infinitely many inequalities at once.

Earlier functional analysis already treated special extension and moment problems. Banach's 1929 paper explicitly cites earlier special cases by F. Riesz, Hahn, and Helly in a footnote to its main extension theorem, and stresses that the proof is simple and does not assume completeness or separability of the ambient normed vector space. The point is not merely to add coordinates: the extension must preserve the least possible bound.

The decisive local observation is one-dimensional. If `z notin M`, an extension to `M+R z` has the form `F(x+t z)=f(x)+t c`. The problem is to choose one real number `c` that keeps `F<=p`. The inequalities for positive and negative `t` reduce to an interval:

`sup_{x in M} (f(x)-p(x-z)) <= c <= inf_{y in M} (p(y+z)-f(y))`.

Sublinearity is exactly what proves this interval is nonempty, because

`f(x)+f(y)=f(x+y)<=p(x+y)<=p(x-z)+p(y+z)`.

So the infinite extension problem has a finite local engine: adjoining one new direction is possible whenever domination is expressed by a sublinear function.

The remaining obstacle is not algebraic but set-theoretic. In an infinite-dimensional space there may be no countable sequence of missing directions. Banach's original proof uses transfinite induction after well-ordering the missing elements; the modern equivalent is Zorn's lemma. The nonconstructive step is not decoration: it is what converts the one-dimensional extension lemma into an extension on the whole space.

## Baselines

- **Basis extension without bounds.** Extend a Hamel basis of `M` to a basis of `E`, assign arbitrary values on the new basis vectors, and extend linearly. This solves only algebraic extension. It gives no reason the resulting functional remains bounded, continuous, or dominated by `p`.

- **Finite-dimensional separating hyperplanes.** Convex geometry in finite dimensions can often separate a point from a convex set by a hyperplane. The limitation is that compactness or finite-dimensional geometry is doing the work; the argument does not automatically survive arbitrary normed spaces or topological vector spaces.

- **Special function-space extension theorems.** The earlier results cited by Banach cover particular classes of spaces or systems. They show the need for extending linear data, but they do not isolate the general mechanism that works without completeness or separability.

- **Normed-space continuity estimates.** If a linear functional is already defined on the whole normed space, continuity is equivalent to a bound `|f(x)|<=C||x||`. This does not explain how to define the missing values while preserving the best constant.

## Evaluation settings

This is a theorem, so success is logical rather than empirical.

- **Analytic setting.** Real vector spaces with a sublinear dominating functional; normed spaces as the symmetric special case `p(x)=C||x||`; subspaces of arbitrary dimension.

- **Geometric setting.** Open convex sets disjoint from a subspace, or a point outside a closed subspace, where success means producing a nonzero linear functional whose kernel is a separating hyperplane.

- **Stress cases.** Infinite-dimensional spaces where no finite basis calculation is informative; nonseparable spaces where sequential construction is not enough; subspaces with no closed complement; normed spaces where the extension must preserve the exact norm.

- **Standard of success.** The one-step extension inequalities must be correct for all `x+t z`; chains of partial extensions must have valid upper bounds; maximality must force the domain to be all of `E`; and the separation corollary must arise from a genuine linear functional, not from a finite-dimensional picture.

## Code framework

The artifact is a theorem and proof. The reusable proof scaffold is:

1. Express the desired bound as domination by a sublinear functional.
2. Prove the one-dimensional extension lemma by deriving the admissible interval for the new value.
3. Partially order all dominated extensions and use a maximal-extension principle.
4. Show any maximal dominated extension that misses a vector can be enlarged, contradicting maximality.
5. Read the resulting global functional geometrically as a separating hyperplane through its kernel or a level set.
