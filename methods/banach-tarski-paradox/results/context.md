## Research question

Volume behaves reliably for ordinary regions: if a solid ball is cut into finitely many measurable pieces and each piece is moved by a rigid motion, finite additivity and invariance under rigid motions force the total volume to stay the same. The question is what remains of that intuition when the pieces are arbitrary subsets, not measurable solids.

The goal is to understand whether a bounded set in three-dimensional Euclidean space can be finitely equidecomposed with another bounded set of different ordinary volume, using only congruent moves of pieces. A satisfactory theorem has to separate two issues that are easy to confuse: no physical material is being recombined, and no volume-preserving theorem is being violated for measurable pieces. The construction operates on arbitrary subsets whose existence depends on choice.

## Background

Finite equidecomposition means that two sets `A` and `B` are partitioned into the same finite number of disjoint pieces,

`A = A_1 sqcup ... sqcup A_n`, `B = B_1 sqcup ... sqcup B_n`,

with each `A_i` congruent to `B_i`. In Euclidean space the congruences are rigid motions, and on the sphere they are rotations.

The measure-theoretic obstruction is immediate whenever the pieces are measurable. If a finitely additive measure is defined on every piece, invariant under the allowed motions, and normalized so that the original ball has positive finite measure, then equidecomposition preserves measure. Therefore any finite equidecomposition between one ball and two balls must contain nonmeasurable pieces. The primary source explicitly notes that the decompositions supplied by the fundamental theorems necessarily involve non-Lebesgue-measurable sets, because measurable decompositions preserve measure.

The group-theoretic source of the phenomenon is nonamenability. A group action on a set is paradoxical when the set can be partitioned into pieces that can be moved by group elements to produce two copies of the same set. Amenability is exactly the obstruction to this behavior in Tarski's alternative: an invariant mean prevents `1 = 2`, while the absence of such a mean allows a paradoxical decomposition.

The three-dimensional rotation group contains a free subgroup on two generators. A free group on two generators is already paradoxical under left multiplication: its Cayley tree branches enough that words beginning with selected first letters can be shifted to cover the whole group twice. This is the algebraic engine. The geometric problem is to transfer that engine from the free group to the sphere.

## Baselines

- **Measurable equidecomposition.** If every piece is Lebesgue measurable, rigid-motion invariance and finite additivity preserve total volume. This explains why ordinary cutting arguments cannot change the volume of a ball. Gap: the obstruction applies only to measurable pieces, while arbitrary subsets are not guaranteed to carry volume.

- **Two-dimensional measure intuition.** In one and two dimensions, invariant finitely additive extensions exist for bounded sets in the form needed to block finite paradoxical decompositions of measurable planar regions. Gap: the same measure problem changes in dimension three, where the rotation group has enough algebraic complexity to support paradoxical behavior.

- **Hausdorff-style sphere phenomenon.** Earlier work on the sphere shows that the surface of a sphere can exhibit decomposition behavior unlike planar figures. Gap: the countable exceptional set and the passage from a surface statement to a full ball still have to be handled carefully.

- **Free group paradox on itself.** The free group `F_2=<a,b>` can be split by initial letters so that left multiplication by `a` and `b` makes selected parts cover the whole group. Gap: this is initially only an algebraic decomposition of group elements, not a decomposition of geometric points in space.

- **Orbit decomposition under a group action.** A free group action partitions a set into orbits. Gap: moving from orbit structure to an actual subset decomposition requires selecting one representative from each orbit, which is exactly where the axiom of choice enters.

## Evaluation settings

The artifact is a theorem and proof. The natural setting is the unit sphere `S^2` and the closed unit ball `B^3` in `R^3`, with rotations from `SO(3)` as the main motions. The construction may ignore and later restore countable exceptional sets fixed by nonidentity rotations.

Success means producing finite partitions of the sphere or ball into arbitrary subsets such that two disjoint unions of pieces are each equidecomposable with the original set. The proof should identify the free subgroup of `SO(3)`, transfer the free-group paradox through orbits, use choice only where orbit representatives are selected, and explain why the pieces cannot all be measurable.

## Proof artifact

The final theorem should state that a three-dimensional ball can be partitioned into finitely many subsets that, after rigid motions, form two balls congruent to the original. A sphere version should appear first: outside a countable exceptional set, the free subgroup action decomposes the sphere into orbits; a choice set of representatives turns subsets of `F_2` into subsets of `S^2`.

The proof should proceed through these objects:

- a free subgroup `F=<a,b>` of `SO(3)`;
- the countable exceptional set `D` of points fixed by nonidentity elements of `F`;
- a choice set `M` containing one representative from each `F`-orbit in `S^2 \ D`;
- the four initial-letter pieces of `F_2`, arranged into two sets each equidecomposable with `F_2`;
- their transported versions `A M`, `B M`, and related subsets of the sphere;
- radial extension from `S^2` to `B^3 \ {0}`;
- a final removal/restoration of the center.

The proof should close by showing that a rotation-invariant finitely additive volume on all subsets of the ball cannot extend ordinary volume: if it did, the two equidecomposable copies would each have the original ball's volume while their disjoint union would still be the original ball.
