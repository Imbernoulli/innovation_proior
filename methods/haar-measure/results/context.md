## Research question

Abstract groups encode symmetry, but analysis needs more than algebraic motion: it needs a way to measure sets and integrate functions without choosing coordinates that destroy the symmetry. On `R^n`, Lebesgue measure makes translation invisible to volume. On a discrete group, counting measure does the same. On a compact Lie group, invariant volume forms suggest the same behavior. The question is whether this is a coincidence of familiar examples or a theorem about topological groups themselves.

The object sought is a measure on the Borel sets of a topological group `G` that is compatible with the topology and that satisfies

`mu(xE) = mu(E)`

for every group element `x` and every measurable set `E`. One also wants to know how much choice remains: if two such objects exist, how are they related?

## Background

The topological assumptions carry weight alongside the algebraic ones. Continuity of multiplication and inversion lets translates of neighborhoods remain neighborhoods, while local compactness supplies compact neighborhoods and finite subcovers. Those compactness facts are what make a finite local mass plausible.

Borel regularity is the measure-theoretic interface. One works with a measure defined at least on the Borel sigma-algebra, finite on compact sets, outer regular on Borel sets, and inner regular in the usual Radon sense. This keeps the object tied to the topology rather than being merely finitely additive or purely algebraic.

The natural test functions are `C_c(G)`, the continuous compactly supported functions. If a positive linear functional on `C_c(G)` is invariant under left translation, the Riesz representation theorem can turn it into a regular Borel measure. Conversely, a regular Borel measure finite on compact sets integrates functions in `C_c(G)`. This makes the functional and measure formulations two faces of the same problem.

Left and right translations need not be interchangeable. In abelian and compact cases the distinction disappears; for a general locally compact group a left-invariant measure can transform by a scalar under right translation.

## Baselines

**Counting measure on a discrete group.** Every singleton is open, compact sets are finite when the group is discrete and locally compact in the usual examples, and left translation is a bijection. Counting measure is invariant and unique after choosing the mass of one point.

**Lebesgue measure on Euclidean groups.** On `R^n`, translation invariance and regularity are familiar, and linear changes of variables explain how volume scales under automorphisms.

**Invariant volume on compact manifolds and compact groups.** Smooth compact groups often carry invariant differential forms, and compactness allows normalization to total mass one.

**Ad hoc averaging over neighborhoods.** One can approximate volume by covering compact sets with translates of a fixed neighborhood. This uses only topology and group multiplication.

## Evaluation settings

The theorem should recover the expected objects in standard cases: counting measure on discrete groups, Lebesgue measure on `R^n`, normalized invariant probability on compact groups, and the usual invariant volume on Lie groups after choosing a normalization.

The formal checks are existence, nontriviality, regularity, finiteness on compact sets, positivity on nonempty open sets, and invariance under every left translation. Uniqueness should be tested by comparing any two candidate regular left-invariant Borel measures and asking how their values are related.

The non-unimodular check is also part of the picture. Right translation of a left-invariant measure should produce another left-invariant measure, whose relationship to the original is to be determined.

## Code framework

The field-natural scaffold is analytic rather than computational. The primitive objects are a locally compact Hausdorff topological group `G`, its Borel sigma-algebra, compact neighborhoods, the test-function space `C_c(G)`, left translation operators

`(L_x f)(t) = f(x^{-1}t)`,

and the Riesz representation theorem connecting positive linear functionals on `C_c(G)` with regular Borel measures.

The empty slot is a construction that takes a locally compact Hausdorff topological group and produces an invariant measure of the kind described, together with an account of the relationship between any two such measures and of the behavior under right translation.
