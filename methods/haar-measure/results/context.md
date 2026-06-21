## Research question

Abstract groups encode symmetry, but analysis needs more than algebraic motion: it needs a way to measure sets and integrate functions without choosing coordinates that destroy the symmetry. On `R^n`, Lebesgue measure makes translation invisible to volume. On a discrete group, counting measure does the same. On a compact Lie group, invariant volume forms suggest the same behavior. The question is whether this is a coincidence of familiar examples or a theorem about topological groups themselves.

The desired object should live on the Borel sets of a topological group `G`, be finite on compact sets and compatible with approximation by compact and open sets, assign positive mass to nonempty open sets, and satisfy

`mu(xE) = mu(E)`

for every group element `x` and every measurable set `E`. A successful answer must also explain how much choice remains: if two such objects exist, are they genuinely different, or only different normalizations of the same volume scale?

## Background

The algebraic operation alone is not enough. A purely abstract group can be too large or too wild for useful measure theory, and arbitrary subsets need not behave well. The topological assumptions are doing real work: continuity of multiplication and inversion lets translates of neighborhoods remain neighborhoods, while local compactness supplies compact neighborhoods and finite subcovers. Those compactness facts are exactly what make a finite local mass plausible.

Borel regularity is the right measure-theoretic interface. One wants a measure defined at least on the Borel sigma-algebra, finite on compact sets, outer regular on Borel sets, and inner regular in the usual Radon sense. This keeps the object tied to the topology instead of being merely finitely additive or purely algebraic.

The natural test functions are `C_c(G)`, the continuous compactly supported functions. If a positive linear functional on `C_c(G)` is invariant under left translation, the Riesz representation theorem can turn it into a regular Borel measure. Conversely, a regular Borel measure finite on compact sets integrates functions in `C_c(G)`. This makes the functional and measure formulations two faces of the same problem.

Left and right translations need not be interchangeable. In abelian and compact cases the distinction disappears, but for a general locally compact group a left-invariant measure can transform by a scalar under right translation. That scalar behavior is a structural caveat, not a defect in the left-invariant construction.

## Baselines

**Counting measure on a discrete group.** Every singleton is open, compact sets are finite when the group is discrete and locally compact in the usual examples, and left translation is a bijection. Counting measure is invariant and unique after choosing the mass of one point. Its limitation is that it uses discreteness completely; it gives no recipe for a nondiscrete group where singletons should have zero mass.

**Lebesgue measure on Euclidean groups.** On `R^n`, translation invariance and regularity are familiar, and linear changes of variables explain how volume scales under automorphisms. Its limitation is coordinate dependence. It does not by itself handle a topological group with no vector-space chart or no global differentiable structure.

**Invariant volume on compact manifolds and compact groups.** Smooth compact groups often carry invariant differential forms, and compactness allows normalization to total mass one. The limitation is smoothness and compactness. A general locally compact group may be neither a manifold nor compact, so the construction cannot depend on a global volume form or finite total mass.

**Ad hoc averaging over neighborhoods.** One can try to approximate volume by covering compact sets with translates of a fixed neighborhood. This uses only topology and group multiplication, and it is close to the right raw material. Its limitation is additivity: covering numbers are naturally subadditive and scale-dependent, so a proof must extract a genuinely linear limiting object rather than merely a rough size.

## Evaluation settings

The theorem should recover the expected objects in standard cases: counting measure on discrete groups, Lebesgue measure on `R^n`, normalized invariant probability on compact groups, and the usual invariant volume on Lie groups after choosing a normalization.

The formal checks are existence, nontriviality, regularity, finiteness on compact sets, positivity on nonempty open sets, and invariance under every left translation. Uniqueness should be tested by comparing any two candidate regular left-invariant Borel measures and proving that their ratio is a single positive scalar independent of the set or test function.

The non-unimodular check is also essential. A left-invariant measure should not be silently assumed right invariant. Right translation should produce another left-invariant measure, hence by uniqueness must rescale the original one through a continuous positive character.

## Code framework

The field-natural scaffold is analytic rather than computational. The primitive objects are a locally compact Hausdorff topological group `G`, its Borel sigma-algebra, compact neighborhoods, the test-function space `C_c(G)`, left translation operators

`(L_x f)(t) = f(x^{-1}t)`,

and the Riesz representation theorem connecting positive linear functionals on `C_c(G)` with regular Borel measures.

The empty slot is a construction that starts from local compactness and group translation, produces a nonzero positive left-invariant linear functional on `C_c(G)`, and then converts that functional into a regular Borel measure. The same scaffold must also compare two such functionals and isolate the possible right-translation scaling.
