The polish landed at `0.037032`, a whisker — five parts in a million — below the conjectured optimum
`1/27 = 0.0370370...`, and the feedback is candid about why. That residual `5×10⁻⁶` is not a geometric
gap that more search could close: fresh restarts cluster lower still, around `0.958` of the record, and
the best only reached `0.99986` by polishing the annealing best already sitting at the edge of the
optimum's basin. The gap is the soft-min ladder converging to within its own tolerance — the surrogate
tracks the true minimum only up to a bias of order `(log k)/β ≈ 10⁻⁵`, exactly the size I predicted, and
L-BFGS-B stops when its gradient and function tolerances are met, not when it reaches the algebraic
point. In other words the search has told me the answer is `1/27` to five significant figures and then
run out of the ability to say more, because it is asking a floating-point objective to certify an exact
rational, which floating point cannot do. To actually *reach* `1/27` rather than approximate it, I have
to stop treating the optimum as something to be found by climbing and instead reproduce it as what it
is: an exact, structured configuration with a closed form.

So the question changes shape. It is no longer "how do I optimize harder" but "what is the exact
configuration Goldberg found, and can I write it down and confirm it on the nose." A clean rational like
`1/27` does not fall out of an unstructured search — a generic local optimum of the max-min objective
would be some messy algebraic number of high degree, not a fraction with denominator `27`. A fraction
that clean is the signature of a configuration with enough symmetry and rational structure that a whole
family of binding triangles collapses to the *same* exact value. And my own polished points already
whisper exactly this structure: the feedback noted they sit on multiples of `1/3` along the boundary and
on `2/9` and `4/9` in the interior, and the whole arrangement is mirror-symmetric left-to-right. That is
the fingerprint of a hand construction, and it tells me the honest endpoint is not a better optimizer
but the exact arrangement the search was groping toward.

I can reason about what that arrangement must be before I look it up. A configuration whose minimum is
*maximal* is one where many triangles are simultaneously tight: if only one or two triangles sat at the
minimum, I could nudge the points to grow just those and raise the score, so a genuine max-min maximum
has its small triangles pinned in a rigid, over-determined web — the very "binding set spread across many
points" that made single-point moves and even the soft-min polish struggle. The more triangles that
share the exact same area, the more the minimum is forced onto a rational value, and the cleaner and
more symmetric the layout, the larger that shared family. The natural symmetry is a mirror about the
vertical centreline `x = 1/2`, which the boundary-heavy, left-right-symmetric shape of my polished points
already exhibits. With eleven points — an odd number — that mirror forces pairs straddling the axis plus
a small odd remainder of points sitting *on* the axis `x = 1/2`. And the points should be boundary-heavy:
pushing a point onto an edge of the square gives its triangles the most room, so I expect most of the
eleven on the boundary with only a few interior points breaking up the collinearities the boundary
points would otherwise create.

That structure is exactly Goldberg's, and the coordinates are rational with denominators `3, 9, 6, 2`:
two points `(1/3, 0)` and `(2/3, 0)` on the bottom edge; four on the two side edges at `(0, 2/9)`,
`(1, 2/9)`, `(0, 2/3)`, `(1, 2/3)`; two on the top at `(1/6, 1)` and `(5/6, 1)`; an interior pair
`(1/3, 4/9)` and `(2/3, 4/9)`; and the single axis apex `(1/2, 7/9)`. That is eight points on the
boundary and three in the interior, mirror-symmetric about `x = 1/2` with the apex on the axis — the
boundary-heavy, symmetric pattern the reasoning predicted, and every coordinate a simple fraction.

The vertical structure is worth reading, because it is what makes the areas resolve so cleanly. The
distinct heights are `0, 2/9, 4/9, 2/3, 7/9, 1` — in ninths, `0, 2, 4, 6, 7, 9`. The lower run
`0, 2/9, 4/9, 6/9` is an arithmetic progression in steps of `2/9`, and `2/9` is exactly the height that
turns a horizontal base of length `1/3` into a triangle of area `½ · (1/3) · (2/9) = 1/27`. So the
bottom edge, the two side heights, and the interior rail are stacked at intervals engineered so that any
horizontal base of the right length, paired with a point one rung up or down, produces the minimum
triangle. The apex at `7/9` and the top edge at `1 = 9/9` cap the ladder, and the horizontal ladder is
the mirror pair of thirds and sixths `0, 1/6, 1/3, 1/2, 2/3, 5/6, 1` that does the same job for vertical
bases, where a width of `1/6` turns a vertical base of length `4/9` into `½ · (4/9) · (1/6) = 1/27`. The
coordinates are not arbitrary rationals; they are chosen so that many different base-and-offset pairs
all multiply out to the same `2/27` cross product, which is exactly why so many triangles tie.

Before I trust it I want to *see* the `1/27` come out, so let me hand-compute a few triangles in exact
arithmetic, because this is the whole point of the rung and I refuse to eyeball it. Take the two bottom
points and one lower-side point, `(1/3, 0)`, `(2/3, 0)`, `(0, 2/9)`. The base along `y = 0` has length
`1/3`; the third point sits at height `2/9`; the area is `½ · (1/3) · (2/9) = 1/27` — on the nose. I can
confirm it through the cross product too: with `a = (1/3, 0)`, `b − a = (1/3, 0)`, `c − a = (−1/3, 2/9)`,
the cross is `(1/3)(2/9) − (−1/3)(0) = 2/27`, and half of that is `1/27`. Its mirror image
`(1/3, 0), (2/3, 0), (1, 2/9)` gives the same `1/27` by symmetry. Now the interior pair: `(1/3, 4/9)`
and `(2/3, 4/9)` are `1/3` apart at height `4/9`, and any third point at vertical distance `2/9` from
that line — that is, at `y = 2/9` or `y = 2/3` — makes a triangle of area `½ · (1/3) · (2/9) = 1/27`. The
side points `(0, 2/9)`, `(1, 2/9)`, `(0, 2/3)`, `(1, 2/3)` are exactly at those heights, so all four
of the interior-pair triangles with those side points bind at `1/27`. And a vertical base: `(0, 2/9)`
and `(0, 2/3)` on the left edge are `4/9` apart, and the point `(1/6, 1)` sits at horizontal distance
`1/6` from that edge, giving area `½ · (4/9) · (1/6) = 1/27`; the cross-product check with `a = (0, 2/9)`,
`b − a = (0, 4/9)`, `c − a = (1/6, 7/9)` gives `(0)(7/9) − (1/6)(4/9) = −2/27`, half-magnitude `1/27`,
confirmed. One more, mixing an edge and the apex: `(1/3, 0)` and `(1/3, 4/9)` share `x = 1/3` and are
`4/9` apart, and the apex `(1/2, 7/9)` is `1/6` away horizontally, so `½ · (4/9) · (1/6) = 1/27` again.
Every one of these different-looking triangles — flat base on the bottom, the interior rail, a tall side
pair, an edge-plus-apex — lands on the identical value `1/27`, which is precisely the over-determined
binding web a true max-min maximum must have.

Let me count the ones I can reach by hand, organizing by the base — two points on a common horizontal or
vertical line, with a third point at the perpendicular distance that makes base times height equal
`2/27`. Horizontal base of length `1/3` needs height `2/9`. The bottom pair `(1/3,0),(2/3,0)` with a
point at `y = 2/9` gives the two side points `(0,2/9)` and `(1,2/9)`: two triangles. The interior rail
`(1/3,4/9),(2/3,4/9)` with a point at `y = 2/9` or `y = 6/9 = 2/3` gives all four of `(0,2/9),(1,2/9),
(0,2/3),(1,2/3)`: four triangles. Vertical base of length `4/9` needs width `1/6`. The left-edge pair
`(0,2/9),(0,2/3)` with a point at `x = 1/6` gives `(1/6,1)`, and its mirror the right-edge pair with
`(5/6,1)`: two triangles. The `x = 1/3` rail `(1/3,0),(1/3,4/9)` with a point at `x = 1/6` or `x = 1/2`
gives `(1/6,1)` and the apex `(1/2,7/9)`, and its mirror the `x = 2/3` rail with `(5/6,1)` and the apex:
four triangles. That is `2 + 4 + 2 + 4 = 12` binding triangles just from axis-aligned bases, and I have
not touched the diagonal bases (two points on a slanted line with a third at the matching offset), which
by the density of coincidences here surely contribute more. So the mirror symmetry and the shared
`2/27` cross-value are producing binding triangles in bulk; the total is well past a dozen, likely
around two dozen or more. I will not pin the exact number
by hand; the exact rational evaluator will return it, and that count is itself the measure of how rigid
the optimum is.

It also helps to check a couple of *non*-binding triangles, to see that the ones above the floor sit
comfortably above it rather than crowding down toward `1/27` — a genuine max-min optimum should have its
binding family exactly at the floor and a clear gap up to everything else, not a smear of near-ties just
above. The interior pair with the apex, `(1/3,4/9),(2/3,4/9),(1/2,7/9)`, has base `1/3` and height
`7/9 − 4/9 = 1/3`, area `½ · (1/3) · (1/3) = 1/18` — half again larger than the floor. A bottom point
with a side point and an interior point, `(1/3,0),(0,2/9),(1/3,4/9)`, computes to cross `(−1/3)(4/9) −
(0)(2/9) = −4/27`, area `2/27` — double the floor. So the triangles I sampled that are *not* on the
binding web land at `1/18` and `2/27`, well clear of `1/27`, which is the margin a rigid optimum should
show. That is reassuring but still only a sample; the point of the exact scan is to prove *every* one of
the `165` is at least `24/648`, which is what turns "I believe" into "I have certified."

What I have established by hand is enough to believe the minimum is `1/27` — but believing
is not certifying, and there is a subtle trap I have to close: showing several triangles *equal* `1/27`
does not show that *none* is smaller. That is where the whole rung's discipline lives, and it needs exact
arithmetic for a reason I can make precise.

Look at the denominators. Every coordinate here is a multiple of `1/18`: the `x`-values `0, 1/6, 1/3,
1/2, 2/3, 5/6, 1` are `0, 3, 6, 9, 12, 15, 18` eighteenths, and the `y`-values `0, 2/9, 4/9, 2/3, 7/9, 1`
are `0, 4, 8, 12, 14, 18` eighteenths. So every coordinate difference is a multiple of `1/18`, every
cross product — a sum of products of two such differences — is a multiple of `1/324`, and every triangle
area, half a cross product, is a multiple of `1/648`. The areas live on a grid of spacing `1/648`, and
`1/27 = 24/648`. This is what makes an *exact* answer possible: `1/27` is representable on the grid, and
the whole minimum-area question reduces to integer arithmetic over multiples of `1/648`. But — and this
is the trap — the quantization does *not* by itself force the minimum to be `1/27`. The smallest nonzero
value the grid allows is `1/648 ≈ 0.00154`, and there are twenty-three grid values strictly between `0`
and `24/648`; nothing about "coordinates are multiples of `1/18`" prevents some triangle from landing on
`23/648` or `7/648` or any of them. The specific geometry has to *dodge every one* of those smaller
values, and only checking all `165` triples in exact arithmetic can certify that it does. Floating point
cannot: the polish's `0.037032` is consistent both with a true value of exactly `1/27` masked by
optimizer tolerance and with a genuine value slightly below it, and double precision has no way to
distinguish `24/648` from `24/648 − 10⁻¹²`. Exact rational arithmetic does, because each coordinate is a
`Fraction`, each area is an exact `k/648`, and the minimum over all triples is an exact fraction with no
rounding anywhere.

So the method is: write down the eleven exact fractions, and evaluate the minimum over all `C(11,3) =
165` triples in rational arithmetic — each cross product an exact rational, each area half its absolute
value, the minimum an exact reduction. If the construction is right, that minimum is the literal
`1/27`, and a count of how many triangles achieve it reports the size of the binding web. I also
recompute the same minimum in double precision, to confirm that the evaluator the rest of the ladder
uses agrees with the exact value to machine epsilon — so the number the harness reports and the true
value coincide, and the `5×10⁻⁶` residual of the previous rung is revealed as nothing but optimizer
tolerance against a target that was `1/27` all along. The constructor itself returns the eleven points
as floats for the harness; the exact fractions live in the verification block, which is where the
certification happens.

I can also check the mirror symmetry directly rather than assert it, and it is a one-line test in exact
arithmetic: reflect every point `(x, y)` to `(1 − x, y)` and confirm the resulting set equals the
original. The bottom pair `1/3, 2/3` swaps with itself, the side points `0 ↔ 1` swap edges at each
height, the top pair `1/6 ↔ 5/6` swaps, the interior rail `1/3 ↔ 2/3` swaps, and the apex `1/2 ↔ 1/2`
is fixed — the set is invariant, so the configuration is exactly mirror-symmetric about `x = 1/2`. That
matters as more than aesthetics: the symmetry is why the binding triangles come in left-right pairs
(making the count nearly even, with the axis-touching ones as the odd exceptions), and it is exactly the
symmetry the soft-min polish kept rediscovering when its returned points came out left-right balanced.
The search was not stumbling toward an arbitrary optimum; it was converging on this specific symmetric,
rational one, and now I am writing that one down in closed form.

The nature of this rung's computation is worth naming, because it is the opposite of everything before
it. There is no search, no randomness, no temperature, no tolerance, no seed — the constructor returns
eleven fixed fractions, and the verification is a single deterministic sweep over `165` triples in exact
rational arithmetic that finishes in a fraction of a second, against the previous rung's six minutes of
annealing and polishing. The whole ladder spent its budget *finding* this configuration; this rung
spends essentially nothing because it already knows the answer and only has to certify it. That
inversion — expensive search replaced by cheap exact confirmation — is the signature of having crossed
from "approximating the optimum" to "reproducing it."

Standing at the top of the ladder, the sequence of fractions tells a clean story about which capability
each rung actually bought, and it is not the story of "steadily more sophisticated methods do steadily
better." Naming one clever symmetric shape — the inscribed eleven-gon — reached `0.579` of the record.
Then blind sampling of four million random configurations *lost* ground, down to `0.294`, below the
named baseline, because for all its brute force it had no mechanism to *improve* a configuration and the
extreme-value ceiling pinned it in the low `0.01`s. Adding the single missing capability — memory plus a
move rule that could climb, with annealing to escape the knife-edge traps — jumped to `0.962`. Adding
the next missing capability — a smooth gradient that could *coordinate* the several near-tight triangles
a single-point move could not grow together — closed almost the whole remaining gap to `0.99986`. And
only exact reproduction reaches `1.000`. The ladder is monotone not in method complexity but in whether
each rung supplied the one capability the objective was actually withholding: a positivity guarantee,
then improvement, then coordination, then exact rational structure. The random rung's dip below the
baseline is the sharpest evidence of that — more compute without the right mechanism moved *backward* —
and it is why the endpoint is a closed-form construction rather than a bigger search: past `0.99986`,
what is missing is not more optimization but the recognition that the target is an exact fraction to be
written down, not a floating-point summit to be climbed.

What I expect: the exact minimum equals `1/27` exactly, with a large family of triangles — the couple
dozen my hand count already reaches toward — all tied at exactly that value, forming the rigid,
over-determined binding web that pins the configuration as a genuine max-min optimum; the double-
precision minimum agreeing with `1/27` to around `10⁻¹⁷`; and every one of the eleven points sitting
inside the closed unit square, which the fractions plainly satisfy. This rung does not beat the record,
and it is not meant to: `1/27` is the conjectured optimum at `n = 11` in the square — believed but not
proven — so reaching it *exactly* is the true ceiling of any construction. The search-and-polish ladder
got within five parts in a million by groping toward this arrangement from scratch, which is the strong
statement that the arrangement is a real attractor of the objective and not an accident; this rung closes
that last gap the only way it can be closed, by writing down the exact configuration and confirming, in
arithmetic that cannot lie, that its smallest triangle is exactly `1/27`. What remains open is what was
open before I started: the *proof* that no eleven points do better. A construction can hit the record and
exact arithmetic can certify the construction, but neither certifies optimality — that is a statement
about all configurations at once, and only a proof supplies it. It is worth being precise about why the
gap between "certified construction" and "proof of optimality" is unbridgeable by anything I have done.
My exact evaluation certifies one point in the twenty-two-dimensional configuration space — that *this*
arrangement scores exactly `1/27`. Optimality is the claim that no *other* point in that continuum scores
higher, and the domain is uncountable, so no amount of evaluating configurations, exact or not, can
exhaust it. The search rungs gave strong circumstantial evidence — heavy annealing from many random
starts plus soft-min polish kept converging to this same basin and never found anything above `1/27` —
but "never found better" over a finite search is not "none exists," and the objective is a maximin over a
non-convex landscape riddled with the near-degenerate slivers that made the whole problem hard. A real
proof would have to argue globally: partition the configuration space, bound the best achievable minimum
in each region, and show every region is capped at `1/27` — the kind of case analysis or interval
arithmetic over the whole domain that certifies a global maximum, which is a fundamentally different
enterprise from writing down one optimal-looking point. That is why `n = 11`'s value is *conjectured*
optimal rather than *proven*: the construction is old and the number is clean, but the certificate that
nothing beats it has never been supplied. The ladder reaches the record and stops there, honestly, at the
edge of what construction and search can establish — the exact value in hand, the optimality proof left
open, and no self-deception about which is which.
