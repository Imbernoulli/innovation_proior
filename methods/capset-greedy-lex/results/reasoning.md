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
and trivially correct. Before I run anything I should figure out what this actually produces,
because the value of a floor is only as honest as my understanding of how high it sits.

Let me just walk it by hand on small `n`. For `n = 1` the order is `0, 1, 2`. Take `0`; take `1`
(no line yet); now `2` — the line `{0, 1, 2}` has sum `0 + 1 + 2 = 3 ≡ 0`, so `2` is the
completion and is rejected. Cap is `{0, 1}`, size `2`. For `n = 2` the order is `00, 01, 02, 10,
11, 12, 20, 21, 22`. Take `00`; take `01`; `02` completes `{00, 01, 02}` (coordinatewise sums `0`
and `0+1+2=3≡0`), rejected. Take `10`; check it against `00` — the completion of `{00, 10}` is
`(−0−1, −0−0) = (2, 0) = 20`, not `10`, and against `01` the completion is `(2, 2) = 22`, not
`10`, so `10` is admitted. Take `11`: completions with `00, 01, 10` are `22, 21, 12` — none is
`11` — so admitted. Now `12`: it completes `{10, 11, 12}` (the first coordinate is constant `1`,
the second runs `0,1,2`), rejected. `20` completes `{00, 10, 20}`, rejected; `21` completes `{01,
11, 21}`, rejected; `22` completes `{00, 11, 22}`, rejected. So `n = 2` gives `{00, 01, 10, 11}`,
size `4`.

There is a pattern staring at me in those two cases that I did not expect to be this clean: the
admitted vectors are exactly the ones whose every coordinate is `0` or `1`. At `n = 1` that is
`{0, 1}`; at `n = 2` that is `{00, 01, 10, 11}`. Every vector that got *rejected* is one that
contains a `2`. That is a much sharper statement than "roughly powers of two," so let me see
whether it is forced rather than coincidental.

First, why would greedy never admit a vector containing a `2`? Suppose the walk has so far
admitted exactly the `0/1`-vectors lexicographically below the current one, and the current vector
`v` has a `2` in some coordinate. I can try to write `v` as the completion of a line through two
earlier `0/1`-vectors. Pick `a` and `b` in `{0,1}^n` and ask when `−a − b ≡ v (mod 3)`, i.e.
`a + b ≡ −v ≡ 2v (mod 3)` coordinatewise. In a coordinate where `v_d = 0` I need `a_d + b_d ≡ 0`,
so `(a_d, b_d) = (0, 0)`. Where `v_d = 1` I need `a_d + b_d ≡ 2`, so `(a_d, b_d) = (1, 1)`. Where
`v_d = 2` I need `a_d + b_d ≡ 1`, so `(a_d, b_d) ∈ {(0,1), (1,0)}` — and there is at least one such
coordinate by assumption. So I *can* solve it: set `a` to copy `v` on the `0` and `1` coordinates,
and on each `2`-coordinate independently pick `(0,1)` for `a, b`; let `b` be forced by `a` and `v`.
Both `a` and `b` are then `0/1`-vectors, they are distinct from each other (they disagree on every
`2`-coordinate, and there is at least one) and from `v` (which has a `2`, they do not), and `{a, b,
v}` is a line. Are `a` and `b` lexicographically below `v`? They agree with `v` on every coordinate
before the first `2`-coordinate and are `0/1 < 2` there, so yes, both precede `v` in counting
order and were therefore already admitted. So `v` was blocked the moment its two cube-predecessors
went in. Every vector with a `2` is rejected — not just some.

That argument says the cap is contained in `{0,1}^n`. The other half is whether *all* of `{0,1}^n`
gets admitted, i.e. whether the Boolean cube is itself a cap (greedy admits a `0/1`-vector exactly
when it does not complete a line with two earlier admitted points, so if the cube has no internal
lines at all, nothing in it is ever blocked). Take three distinct vectors `a, b, c ∈ {0,1}^n` and
ask whether they can sum to `0 (mod 3)` coordinatewise. In a single coordinate the three entries
are each `0` or `1`, so their sum is one of `0, 1, 2, 3`; it is `≡ 0 (mod 3)` only when the sum is
`0` (all three are `0`) or `3` (all three are `1`). Either way the three entries are *equal* in
that coordinate. If this held in every coordinate, `a, b, c` would be identical — contradicting
distinctness. So no three distinct cube points sum to zero: `{0,1}^n` is a cap, every cube vector
survives greedy, and the output is exactly the cube.

Let me sanity-check the cap claim by brute force rather than trusting the coordinatewise argument
alone. For `n = 3` the cube is the eight vectors `000 … 111`; scanning all `C(8,3) = 56` triples
for a coordinatewise-zero sum should return none, and the same for `n = 4, 5, 6`. Running the
greedy walk and an independent triple scan agrees: the admitted set equals `{0,1}^n` and is a
valid cap at `n = 1` through `7`, with sizes `2, 4, 8, 16, 32, 64, 128`. So the floor is not
"around `2^n`" — it is *exactly* `2^n`, the order of the Boolean cube, with no slack to argue
about.

Now I can read the deficit off cleanly instead of guessing it. The known optima are `2, 4, 9, 20,
45, 112, 236` for `n = 1..7`. Lexicographic greedy matches at `n = 1, 2` — there `2^n` *is* the
optimum — and then falls progressively short: `8` vs `9`, `16` vs `20`, `32` vs `45`, `64` vs
`112`, `128` vs `236`. The ratio optimum / `2^n` climbs monotonically, `1.000, 1.000, 1.125,
1.250, 1.406, 1.750, 1.844`, and it keeps climbing because the true growth rate sits near
`2.756^n` (the Croot–Lev–Pach / Ellenberg–Gijswijt regime), and `2.756^n / 2^n = 1.378^n` blows up
— already a factor of `9` by `n = 7` and `~47` by `n = 12`. So the Boolean cube is a correct cap
but an exponentially loose one.

Why does lexicographic order strand the construction on the cube? Because it commits early and
locally, and the cube is exactly the fixed point of that myopia: the first thing greedy does is
fill out the `0/1` corner of the space, and the lines those points generate are precisely the
ones that knock out every vector with a `2`. The walk never gets to make a *global* choice about
which points to spend its budget on; it takes whatever comes next in the arbitrary counting order
and lives with the blocked-out consequences. There is no reason the counting order should be
aligned with the geometry of `F_3^n` — it is an artifact of how I happen to enumerate tuples — and
a cap that reaches the optimum has to place its points so the lines they generate fall *outside*
the cap as efficiently as possible, packing the space tightly. A fixed lexicographic walk has no
mechanism to do that: it cannot look ahead, cannot reconsider, cannot prefer a point that blocks
fewer future candidates over one that blocks many.

So the single lever this rung exposes and refuses to pull is the ordering itself — the rest of the
construction (admit-if-no-line, block-the-completion) is already forced and already correct. The
whole gap to the optimum, every point of it, is attributable to the geometry-blind order, and that
is exactly what the next rung has to attack. If a fixed order is the weakness, then trying *many*
orders and keeping the best is the first, cheapest way to do better, and it is where I go next.
