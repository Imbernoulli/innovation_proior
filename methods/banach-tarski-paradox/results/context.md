## Research question

Volume behaves reliably for ordinary regions: if a solid ball is cut into finitely many measurable pieces and each piece is moved by a rigid motion, finite additivity and invariance under rigid motions force the total volume to stay the same. The question is what remains of that intuition when the pieces are arbitrary subsets, not measurable solids.

The goal is to determine whether a bounded set in three-dimensional Euclidean space can be finitely equidecomposed with another bounded set of different ordinary volume, using only congruent moves of pieces. The construction would operate on arbitrary subsets whose existence depends on the axiom of choice.

## Background

Finite equidecomposition means that two sets `A` and `B` are partitioned into the same finite number of disjoint pieces,

`A = A_1 sqcup ... sqcup A_n`, `B = B_1 sqcup ... sqcup B_n`,

with each `A_i` congruent to `B_i`. In Euclidean space the congruences are rigid motions, and on the sphere they are rotations.

The measure-theoretic situation for measurable pieces is settled. If a finitely additive measure is defined on every piece, invariant under the allowed motions, and normalized so that the original ball has positive finite measure, then equidecomposition preserves measure. So a finite equidecomposition between one ball and two balls would contain nonmeasurable pieces; measurable decompositions preserve measure.

The group-theoretic source of the phenomenon is nonamenability. A group action on a set is paradoxical when the set can be partitioned into pieces that can be moved by group elements to produce two copies of the same set. Amenability is the relevant property in Tarski's alternative: an invariant mean prevents `1 = 2`, while the absence of such a mean allows a paradoxical decomposition.

The three-dimensional rotation group contains a free subgroup on two generators. A free group on two generators is itself paradoxical under left multiplication: its Cayley tree branches enough that words beginning with selected first letters can be shifted to cover the whole group twice. This is the algebraic engine. The geometric problem is to relate that engine to points on the sphere.

## Baselines

- **Measurable equidecomposition.** If every piece is Lebesgue measurable, rigid-motion invariance and finite additivity preserve total volume. This explains why ordinary cutting arguments cannot change the volume of a ball.

- **Two-dimensional measure intuition.** In one and two dimensions, invariant finitely additive extensions exist for bounded sets in the form needed to block finite paradoxical decompositions of measurable planar regions.

- **Hausdorff-style sphere phenomenon.** Earlier work on the sphere shows that the surface of a sphere can exhibit decomposition behavior unlike planar figures, up to a countable exceptional set.

- **Free group paradox on itself.** The free group `F_2=<a,b>` can be split by initial letters so that left multiplication by `a` and `b` makes selected parts cover the whole group.

- **Orbit decomposition under a group action.** A free group action partitions a set into orbits, each orbit being a copy of the group; selecting one representative from each orbit is governed by the axiom of choice.

## Evaluation settings

The artifact is a theorem and proof. The natural setting is the unit sphere `S^2` and the closed unit ball `B^3` in `R^3`, with rotations from `SO(3)` as the main motions. The construction may set aside and later restore countable exceptional sets fixed by nonidentity rotations.

Success means producing finite partitions of the sphere or ball into arbitrary subsets such that two disjoint unions of pieces are each equidecomposable with the original set, together with an account of why the pieces cannot all be measurable.

## Proof artifact

The final theorem should state that a three-dimensional ball can be partitioned into finitely many subsets that, after rigid motions, form two balls congruent to the original. A sphere version is the natural intermediate statement, with the ball obtained by working outward from the sphere.

The proof should close by showing that a rotation-invariant finitely additive volume on all subsets of the ball cannot extend ordinary volume: if it did, two equidecomposable copies would each carry the original ball's volume while their disjoint union would still be the original ball.
