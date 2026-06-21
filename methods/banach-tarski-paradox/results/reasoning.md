I start with the thing that ordinary volume says I am not allowed to do. If I cut a ball into finitely many decent pieces, move each piece rigidly, and put the pieces back together, the volume must be unchanged. That argument is too quick only because it hides a hypothesis: each piece has to be in the domain of the volume function. If I insist on Lebesgue-measurable pieces, finite additivity and invariance under rotations and translations end the discussion. So the only place the surprise can live is in arbitrary subsets, where choice can produce sets that volume is not allowed to measure.

I should keep the physical picture out of the proof. This is not a recombination of matter. There are no shaped fragments with boundaries that can be machined or moved in space. I am asking whether the set-theoretic relation "finite partition plus congruent motion" can behave differently once the pieces are allowed to be completely nonmeasurable.

What algebraic shape can force that behavior? A group action can duplicate a set if the group itself already has a duplication pattern. The free group on two generators is the clean model because every nonidentity element is a reduced word. Let `F_2=<a,b>`. Split the nonidentity words by their first letter: words beginning with `a`, with `a^{-1}`, with `b`, and with `b^{-1}`, plus the identity. Left multiplication by `a` cancels the initial `a^{-1}` when present and otherwise prefixes an `a`; that is enough to make a shifted piece together with the `a`-side piece cover all of `F_2`. Similarly left multiplication by `b` makes the `b`-side cover all of `F_2`. With a small allocation of the identity and the positive powers of `a`, I can arrange an actual partition of `F_2` into two disjoint blocks `P_a` and `P_b` such that `P_a` is equidecomposable with `F_2` and `P_b` is also equidecomposable with `F_2`.

If I try the same idea with an amenable group, an invariant mean blocks me immediately. A finitely additive invariant probability measure would assign the whole set mass `1`; two disjoint pieces that each move to cover the whole set would force `1=2`. So the group action must be nonamenable. The free subgroup is not decoration. It is the mechanism that makes the finite additive intuition fail.

Now I need geometry. I need the free group to act by rotations of the sphere, because rotations are the congruences I want. In three dimensions `SO(3)` contains a free subgroup on two generators. One explicit route is to take rotations about two perpendicular axes through an angle whose cosine is `3/5`; the standard matrix calculation shows that no nontrivial reduced word in those two rotations is the identity. The important point is that a nonabelian free group sits inside the rotation group of the sphere. This is exactly what fails in the lower-dimensional intuition: the available rigid motions are not algebraically rich enough in the same way.

Let `F=<a,b>` be such a free subgroup of `SO(3)`. It acts on `S^2`. There is an annoyance: a nonidentity rotation fixes the two endpoints of its axis, so the action is not free everywhere. I remove the bad points. Let

`D = {x in S^2 : g x = x for some g in F, g != e}`.

The group `F` is countable, and each nonidentity rotation fixes exactly two points on the sphere, so `D` is countable. Away from `D`, no nonidentity element fixes a point. That means if `g m = h m` for `m in S^2 \ D`, then `h^{-1} g m = m`, hence `h^{-1}g=e`, so `g=h`. On `S^2 \ D`, each orbit is a clean copy of the free group.

The next step is exactly where ordinary constructive geometry runs out. The orbits of `F` partition `S^2 \ D`, and I need one seed point from each orbit. There is no canonical rule for selecting all of them. By the axiom of choice, choose a set `M subset S^2 \ D` containing exactly one point from each orbit. Then every point of `S^2 \ D` has a unique form `g m` with `g in F` and `m in M`.

Now the algebraic partition of `F_2` becomes a geometric partition. If `Q subset F`, define

`Q M = {q m : q in Q, m in M}`.

Because the representation `q m` is unique, disjoint subsets of `F` give disjoint subsets of the sphere outside `D`, and unions are preserved: `(Q_1 union Q_2)M = Q_1M union Q_2M`. So the two blocks `P_a` and `P_b` transported from the free group partition `S^2 \ D`.

The equidecomposition also transports. Suppose `P_a` is split into pieces such that rotating the first by `a` and leaving or rotating the others according to the free-group pattern covers all of `F`. Multiplying every group piece by `M` gives sphere pieces. Applying the same rotations to those sphere pieces covers `F M = S^2 \ D`. Therefore `P_a M` is equidecomposable with `S^2 \ D`. The same argument gives `P_b M` equidecomposable with `S^2 \ D`.

I still have the countable set `D`. A countable subset of the sphere can be absorbed by a rotation trick. Choose an axis missing `D`, then choose a rotation `rho` around that axis so that the sets `D, rho D, rho^2 D, ...` are pairwise disjoint. This is possible because only countably many rotation angles cause collisions among countably many points. Then

`D union rho D union rho^2 D union ...`

is congruent, after applying `rho`, to the same set with `D` removed. Its complement stays fixed. So `S^2` is equidecomposable with `S^2 \ D`.

Combining these facts, one part of the sphere can include the exceptional set and the other can avoid it. For instance, set

`S_1 = P_a M union D`, `S_2 = P_b M`.

They are disjoint and cover `S^2`. The first is equidecomposable with `(S^2 \ D) union D = S^2`, because `P_a M` gives `S^2 \ D` and `D` is carried along. The second is equidecomposable with `S^2 \ D`, and the countable-set absorption gives `S^2 \ D` equidecomposable with `S^2`. So the sphere has been partitioned into two subsets, each equidecomposable with the entire sphere.

To pass from the sphere to the ball, I extend radially. Every nonzero point of the unit ball is uniquely `r x` with `0<r<=1` and `x in S^2`. For a subset `E subset S^2`, define its cone

`C(E) = {r x : 0<r<=1, x in E}`.

Rotations commute with radial scaling, so any rotational equidecomposition of `E` becomes a rotational equidecomposition of `C(E)`. The two cones `C(S_1)` and `C(S_2)` partition `B^3 \ {0}`; each is equidecomposable with `B^3 \ {0}`. The missing center is harmless, because the ball is equidecomposable with the punctured ball by the same kind of countable-orbit shift along a ray or by the standard absorption argument. Thus the full ball can be partitioned into finitely many arbitrary subsets that can be moved by rigid motions to form two balls congruent to the original.

The measure conclusion is now forced. Suppose there were a finitely additive volume `mu` defined on every subset of the ball, invariant under rigid motions, and agreeing with ordinary volume on balls. Let the original ball have volume `V>0`. Since the two disjoint parts are each equidecomposable with the whole ball, invariance and finite additivity give `mu(B_1)=V` and `mu(B_2)=V`. But `B_1 sqcup B_2` is the original ball, so finite additivity gives `V=2V`. That is impossible. Therefore at least some pieces are nonmeasurable, and the lesson is not that volume has been physically doubled. The lesson is that ordinary volume intuition cannot be extended to all subsets while preserving finite additivity and rotation invariance in the presence of these nonamenable group actions.
