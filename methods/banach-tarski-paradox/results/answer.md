# Banach-Tarski Paradox

In three-dimensional Euclidean space, a solid ball can be partitioned into finitely many arbitrary subsets and those subsets can be moved by rigid motions to form two balls congruent to the original. The pieces are not measurable solids; they are choice-built subsets. This is why the result does not describe a physical recombination process and does not contradict ordinary volume for measurable regions.

## Theorem

Let `B^3` be a solid ball in `R^3`. There exist disjoint subsets

`E_1, ..., E_n subset B^3`

whose union is `B^3`, and rigid motions `g_1, ..., g_n`, `h_1, ..., h_n`, such that the moved pieces can be grouped into two disjoint balls congruent to `B^3`.

Equivalently, `B^3` is equidecomposable with two copies of itself.

## Proof

Work first on the unit sphere `S^2`. Choose two rotations in `SO(3)` that generate a free subgroup `F=<a,b>` of rank two. Such rotations exist; for example, rotations about two perpendicular axes through an angle with cosine `3/5` give a standard explicit construction.

The free group `F_2=<a,b>` is paradoxical under left multiplication. Let words be classified by their first letter. The sets of words beginning with `a`, `a^{-1}`, `b`, and `b^{-1}`, with the identity allocated to one side, can be arranged into two disjoint subsets `P_a` and `P_b` such that

`P_a sqcup P_b = F_2`,

and both `P_a` and `P_b` are equidecomposable with all of `F_2`. Algebraically, left multiplication by `a` shifts the `a^{-1}`-initial words so that they fill the complement of the `a`-initial words; left multiplication by `b` does the analogous job for the `b`-initial words.

Let

`D = {x in S^2 : g x = x for some g in F, g != e}`.

Each nonidentity rotation fixes exactly two points on the sphere, and `F` is countable, so `D` is countable. On `S^2 \ D`, the action of `F` is free. By the axiom of choice, choose a set `M subset S^2 \ D` containing exactly one representative from each `F`-orbit. Then every point of `S^2 \ D` has a unique representation `g m`, with `g in F` and `m in M`.

For `Q subset F`, write

`Q M = {q m : q in Q, m in M}`.

Uniqueness of orbit representation makes this operation preserve disjoint unions. Therefore `P_a M` and `P_b M` partition `S^2 \ D`. The free-group equidecompositions transport through the action: `P_a M` is rotation-equidecomposable with `F M = S^2 \ D`, and `P_b M` is also rotation-equidecomposable with `S^2 \ D`.

The countable exceptional set can be absorbed. Choose a rotation `rho` such that `D, rho D, rho^2 D, ...` are pairwise disjoint. Then the union of this countable chain is congruent to the same union with `D` removed, while the complement is left in place. Hence `S^2` is rotation-equidecomposable with `S^2 \ D`.

Define

`S_1 = P_a M union D`, `S_2 = P_b M`.

These two sets partition `S^2`. The first is equidecomposable with `S^2`, and the second is equidecomposable with `S^2 \ D`, hence with `S^2`. Thus the sphere is paradoxically decomposed into two parts, each equivalent to the whole sphere.

Extend radially to the punctured ball. For `E subset S^2`, set

`C(E) = {r x : 0<r<=1, x in E}`.

Rotations commute with radial scaling, so the equidecompositions of `S_1` and `S_2` extend to equidecompositions of `C(S_1)` and `C(S_2)`. These two cones partition `B^3 \ {0}` and are each equidecomposable with `B^3 \ {0}`. Finally, the center point is absorbed by the same countable-shift idea, so `B^3` and `B^3 \ {0}` are equidecomposable. Therefore the full ball has a finite paradoxical decomposition.

## Why Volume Does Not Survive

If all pieces were measurable, finite additivity and rigid-motion invariance would preserve volume. Let `mu` be a hypothetical finitely additive rigid-motion-invariant volume defined on every subset of the ball and agreeing with ordinary volume on balls. If the original ball has volume `V>0`, the two equidecomposable parts would each have measure `V`, while their disjoint union is the original ball. Finite additivity would give

`V = V + V = 2V`,

which is impossible. Hence the construction necessarily uses nonmeasurable pieces. The paradox shows that volume cannot be extended to all subsets while preserving the usual invariance and additivity properties.
