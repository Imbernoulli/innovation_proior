I want a large cap set in `F_3^n`, and the very first thing I should settle is the simplest
constructive idea that is guaranteed to produce a *valid* cap, so I have a floor to stand on
before I get clever. The defining constraint is local in a useful way: a set fails to be a cap
exactly when some three distinct points lie on a line, and over `F_3` a line is just `{a, b,
−a−b}` — given any two points, there is exactly one third point that would complete a forbidden
triple. That suggests an incremental rule that can never produce an invalid set: walk through the
points of `F_3^n` in some fixed order, and admit a point only if it is not the completion of a
line through two points I have already admitted. Whatever order I use, the set I end with is a cap
by construction, because every point was checked against all its predecessors at the moment it was
added. So the question is not validity — that is free — the question is *which order* to walk, and
how good the resulting cap is.

The most obvious order, the one requiring no thought at all, is lexicographic: list the `3^n`
vectors as if counting in base 3, `00…0, 00…1, 00…2, 00…10, …`, and greedily take each one if it
keeps the set cap-valid. This is the natural baseline because it is deterministic, parameter-free,
and trivially correct. Let me reason about what it should give before I run it, because I suspect
the answer is rigid.

Think about what the greedy lexicographic rule does on the last coordinate first. It will take
`0…00` and `0…01` immediately — two points, no line yet. The third point `0…02` would complete the
line `{0…00, 0…01, 0…02}` (their sum is `0…0(0+1+2) = 0…00 ≡ 0`), so it is rejected. So already on
`n = 1` the cap is `{0, 1}`, size `2`, which is the known optimum there. Lift to general `n` and
the same logic cascades: lexicographic order fills the low-order coordinates first, and the
two-symbols-per-coordinate pattern that the `n = 1` case forces seems to propagate. My instinct is
that the lexicographic greedy cap is essentially the set of vectors that, coordinate by coordinate,
avoid the third symbol once two are pinned — a set whose size is governed by powers of two, not by
the richer algebraic structure that the strong caps exploit. If that instinct is right, the cap
will come out around `2^n`: `2, 4, 8, 16, …`. That is a clean, honest baseline, but it is also a
*weak* one — at `n = 3` it would give `8` against the optimum `9`, at `n = 4` it would give `16`
against `20`, and the gap to the optimum should widen as `n` grows, because `2^n` falls further and
further below the true growth rate (which sits near `2.756^n`, not `2^n`).

Why does lexicographic order leave so much on the table? Because it commits early and locally. The
first points it grabs are all clustered at the low-index end of the space, and the line-blocking
those points induce is concentrated there too — the greedy rule never gets to make a *global*
choice about which points to spend its budget on; it just takes whatever comes next in the
arbitrary counting order and lives with the blocked-out consequences. There is no reason the
counting order should be aligned with the geometry of `F_3^n`; it is an artifact of how I happen to
enumerate tuples. A cap that reaches the optimum has to place its points so that the lines they
generate fall *outside* the cap as efficiently as possible, packing the space tightly, and a fixed
lexicographic walk has no mechanism to do that — it cannot look ahead, cannot reconsider, cannot
prefer a point that blocks fewer future candidates over one that blocks many.

So I will run the lexicographic greedy as the floor and read off exactly how far below the optima
it lands. I expect the pattern `2, 4, 8, 16, 32, 64, 128` for `n = 1..7`: matching the optimum only
at `n = 1, 2` (where `2^n` *is* the optimum), and falling progressively short everywhere after —
`8` vs `9`, `16` vs `20`, `32` vs `45`, `64` vs `112`, `128` vs `236`. The whole value of this rung
is to make that gap concrete and to confirm the mechanism: a fixed, geometry-blind order is correct
but rigid, and every point of the gap to the optimum is a point that a *better order* would have to
buy. The single lever this rung exposes and refuses to pull is the ordering itself. That is exactly
what the next rung has to attack — if a fixed order is the weakness, then trying *many* orders and
keeping the best is the first, cheapest way to do better, and it is where I go next.
