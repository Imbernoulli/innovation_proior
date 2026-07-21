The polish landed at `0.037032`, five parts in a million below the conjectured optimum
`1/27 = 0.037037...`, and the residual is not a geometric gap more search could close: fresh restarts
clustered lower, and the best only reached `0.99986` by polishing a configuration already at the edge
of the optimum's basin. The gap is the soft-min ladder converging within its own tolerance — the
surrogate tracks the true minimum only up to a bias `(log k)/β ≈ 10⁻⁵`, and L-BFGS-B stops on its
gradient and function tolerances, not at the algebraic point. The search has told me the answer is
`1/27` to five figures and then run out of the ability to say more, because it is asking a
floating-point objective to certify an exact rational, which it cannot. To actually *reach* `1/27` I
have to stop climbing and reproduce the optimum as what it is: an exact, structured configuration with
a closed form.

So the question changes shape — not "how do I optimize harder" but "what is the exact configuration,
and can I write it down and confirm it on the nose." A clean rational like `1/27` does not fall out of
unstructured search: a generic local optimum would be a messy algebraic number of high degree, not a
fraction with denominator `27`. A fraction that clean is the signature of a configuration symmetric and
rigid enough that a whole family of binding triangles collapses to the *same* exact value. And the
polished points already whisper it — they sat on multiples of `1/3` along the boundary and on `2/9` and
`4/9` in the interior, mirror-symmetric left-to-right. That is the fingerprint of a hand construction,
and it says the honest endpoint is not a better optimizer but the exact arrangement the search was
groping toward.

I can reason about that arrangement before writing it down. A max-min maximum has many triangles
simultaneously tight — if only one or two sat at the minimum I could nudge the points to grow just
those and raise the score, so a genuine maximum has its small triangles pinned in a rigid,
over-determined web (the "binding set spread across many points" that made single-point moves and even
the soft-min polish struggle). The more triangles share the exact same area, the more the minimum is
forced onto a rational value. The natural symmetry is a mirror about `x = 1/2`, which the polished
points already show; with eleven points — odd — that forces pairs straddling the axis plus a small odd
remainder *on* the axis. And the points should be boundary-heavy: pushing a point onto an edge gives
its triangles the most room, so I expect most of the eleven on the boundary with a few interior points
breaking up the collinearities the boundary points would otherwise create.

That structure is exactly Goldberg's, coordinates rational with denominators `3, 9, 6, 2`: `(1/3, 0)`
and `(2/3, 0)` on the bottom; `(0, 2/9), (1, 2/9), (0, 2/3), (1, 2/3)` on the sides; `(1/6, 1)` and
`(5/6, 1)` on top; an interior pair `(1/3, 4/9)` and `(2/3, 4/9)`; and the single axis apex
`(1/2, 7/9)`. Eight points on the boundary, three interior, mirror-symmetric about `x = 1/2` — the
boundary-heavy symmetric pattern predicted.

The vertical structure is what makes the areas resolve so cleanly. The distinct heights are
`0, 2/9, 4/9, 2/3, 7/9, 1` — in ninths, `0, 2, 4, 6, 7, 9`. The lower run `0, 2/9, 4/9, 6/9` steps by
`2/9`, and `2/9` is exactly the height that turns a horizontal base of length `1/3` into area
`½·(1/3)·(2/9) = 1/27`. So the bottom edge, the side heights, and the interior rail are stacked at
intervals engineered so any horizontal base of the right length, paired with a point one rung up or
down, makes the minimum triangle; the horizontal ladder of thirds and sixths `0, 1/6, 1/3, 1/2, 2/3,
5/6, 1` does the same for vertical bases, where a width of `1/6` turns a vertical base of length `4/9`
into `½·(4/9)·(1/6) = 1/27`. The coordinates are chosen so many different base-and-offset pairs all
multiply to the same `2/27` cross product — which is why so many triangles tie.

The `1/27` comes out cleanly in exact arithmetic. The two bottom points and a
lower-side point, `(1/3,0), (2/3,0), (0,2/9)`: base `1/3` along `y = 0`, third point at height `2/9`,
area `½·(1/3)·(2/9) = 1/27` — and via the cross product, `b−a = (1/3,0)`, `c−a = (−1/3,2/9)`, cross
`= (1/3)(2/9) = 2/27`, half is `1/27`. The interior pair `(1/3,4/9), (2/3,4/9)` are `1/3` apart at
height `4/9`; any third point at vertical distance `2/9` — that is at `y = 2/9` or `y = 2/3` — makes
`1/27`, and the four side points sit exactly at those heights, so all four interior-pair-with-side
triangles bind. A vertical base: `(0,2/9), (0,2/3)` on the left edge are `4/9` apart, and `(1/6,1)` is
`1/6` away horizontally, area `½·(4/9)·(1/6) = 1/27`. And mixing edge and apex: `(1/3,0), (1/3,4/9)`
share `x = 1/3`, `4/9` apart, apex `(1/2,7/9)` is `1/6` away, `1/27` again. Flat base, interior rail,
tall side pair, edge-plus-apex — all land on the identical `1/27`, the over-determined binding web a
true max-min maximum must have.

Counting the ones I can reach by hand, organized by base: a horizontal base of `1/3` needs height
`2/9` — the bottom pair with the two `y = 2/9` side points (two triangles), the interior rail with
points at `y = 2/9` or `2/3` (four); a vertical base of `4/9` needs width `1/6` — the two side-edge
pairs each with a top point (two), and the `x = 1/3` and `x = 2/3` rails each with a top point and the
apex (four). That is `2 + 4 + 2 + 4 = 12` from axis-aligned bases alone, before the diagonal bases,
which by the density of coincidences here surely add more — the total is well past a dozen, and the
exact evaluator will report it.

But showing several triangles *equal* `1/27` does not show that *none* is smaller, and that is where
the rung's discipline lives. Every coordinate is a multiple of `1/18`: the `x`-values are
`0, 3, 6, 9, 12, 15, 18` eighteenths, the `y`-values `0, 4, 8, 12, 14, 18` eighteenths. So every
coordinate difference is a multiple of `1/18`, every cross product a multiple of `1/324`, every area a
multiple of `1/648`, and `1/27 = 24/648`. The areas live on a grid of spacing `1/648` — which is what
makes an exact answer possible — but the quantization does *not* by itself force the minimum to `1/27`:
the smallest nonzero grid value is `1/648`, and twenty-three grid values lie strictly between `0` and
`24/648`, so nothing prevents some triangle from landing on `23/648` or `7/648`. The geometry has to
dodge every one of those, and only checking all `165` triples in exact arithmetic certifies that it
does. Floating point cannot: `0.037032` is consistent both with a true value of exactly `1/27` masked
by tolerance and with a genuine value slightly below, and double precision cannot distinguish `24/648`
from `24/648 − 10⁻¹²`. Exact rational arithmetic can, because each coordinate is a `Fraction`, each
area an exact `k/648`, and the minimum an exact reduction.

So the method is: write down the eleven exact fractions and evaluate the minimum over all `165` triples
in rational arithmetic — each cross product an exact rational, each area half its absolute value, the
minimum an exact reduction. If the construction is right that minimum is the literal `1/27`, and a
count of how many triangles achieve it reports the binding web's size. I also recompute the minimum in
double precision, to confirm the evaluator the rest of the ladder uses agrees to machine epsilon — so
the `5×10⁻⁶` residual is revealed as pure optimizer tolerance against a target that was `1/27` all
along. The constructor returns floats for the harness; the exact fractions live in the verification. I
can also confirm the mirror symmetry directly: reflect every point `(x, y)` to `(1−x, y)` and check the
set is unchanged — the bottom pair `1/3 ↔ 2/3` swaps, the side points swap edges, the top pair
`1/6 ↔ 5/6` swaps, the rail swaps, the apex `1/2` is fixed. The symmetry is why the binding triangles
come in left-right pairs, and it is exactly what the polish kept rediscovering when its returned points
came out balanced — the search was converging on this specific symmetric, rational optimum, and now I
write it down in closed form.

This rung is the opposite of everything before it: no search, no randomness, no temperature, no seed —
eleven fixed fractions and a single deterministic exact sweep that finishes in a fraction of a second,
against the previous rung's six minutes. The whole ladder spent its budget *finding* this
configuration; this rung spends almost nothing because it already knows the answer and only certifies
it. What the sequence of rungs bought was monotone not in method sophistication but in whether each
supplied the one capability the objective was withholding — naming, then memory-plus-a-move-rule, then a
gradient that could *coordinate* the near-tight triangles, then exact reproduction. The one telling
anomaly is that blind sampling *lost* ground to the named baseline: more compute without the right
mechanism moved *backward*.

I do not beat the record, and am not meant to: `1/27` is the conjectured optimum at `n = 11` in the
square — believed but not proven — so reaching it *exactly* is the ceiling of any construction. My
exact evaluation certifies one point in the twenty-two-dimensional configuration space — that *this*
arrangement scores `1/27`. Optimality is the claim that no *other* point in that uncountable continuum
scores higher, and no amount of evaluating configurations, exact or not, can exhaust it. The search
rungs gave strong circumstantial evidence — heavy annealing from many starts plus polish kept
converging to this same basin and never found anything above `1/27` — but "never found better" over a
finite search is not "none exists." A real proof would have to argue globally, capping the best
achievable minimum over every region of the domain — a fundamentally different enterprise from writing
down one optimal-looking point. So the ladder stops with the exact value in hand and the optimality
proof left open.
