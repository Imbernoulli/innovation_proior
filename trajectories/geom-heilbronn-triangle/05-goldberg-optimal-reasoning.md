The polish landed at `0.037032`, a whisker — five parts in a million — below the conjectured optimum
`1/27 = 0.0370370...`, and the feedback is candid about why: the soft-min gradient ascent snaps onto
the *basin* of the optimum and stops a hair short, because L-BFGS-B on a floating-point surrogate
converges to within its own tolerance, not to the algebraic point. That residual `5×10^-6` is not a
geometric gap that more search could close — fresh restarts cluster lower still, around `0.958` of the
record — it is pure optimizer tolerance against a target that the floating-point objective can only
approach. To actually *reach* `1/27`, not approximate it, I have to stop treating the optimum as
something to be found by climbing and instead reproduce it as what it is: an exact, structured
configuration with a closed form.

So the question changes shape. It is no longer "how do I optimize harder" but "what is the exact
configuration Goldberg found in 1972, and can I write it down and confirm it on the nose." The record
`Δ(11) = 1/27` is a clean rational, and a clean rational does not come out of an unstructured search —
it comes out of a configuration with enough symmetry and rational structure that the binding triangles
all collapse to the *same* exact value. My own polished points already whisper this: they sit on
multiples of `1/3` along the boundary and on `2/9` and `4/9` in the interior, and the arrangement is
mirror-symmetric left-to-right. That is the fingerprint of a hand construction, and it tells me the
honest endpoint is not a better optimizer but the exact arrangement the search was groping toward.

I reason about what that arrangement must be. A configuration whose minimum is a single clean fraction,
and whose minimum is *maximal*, is one where many triangles are simultaneously tight — if only a few
triangles were at the minimum, I could perturb the points to grow them and raise the score, so a true
local maximum of the max-min objective has the small triangles pinned in a rigid, over-determined web.
The cleaner and more symmetric the layout, the more triangles share the exact same area, and the more
the minimum is forced onto a rational value. The natural symmetry here is a mirror about the vertical
centerline `x = 1/2`, which Friedman's tables confirm Goldberg's `n=11` configuration has: points come
in left-right pairs, plus points that sit on the axis itself. With eleven points — an odd number — I
expect pairs straddling the axis and a small number of points exactly on `x = 1/2`. The boundary of the
square should carry most of the points, because pushing a point to an edge gives its triangles the most
room; only a few points sit in the interior to break up the collinearities the boundary points would
otherwise create.

That structure is exactly what Goldberg published, and the coordinates are rational with denominators
`3`, `9`, `6`, and `2`: points at `(1/3, 0)` and `(2/3, 0)` on the bottom edge; `(0, 2/9)` and
`(1, 2/9)` and `(0, 2/3)` and `(1, 2/3)` on the two side edges; `(1/6, 1)` and `(5/6, 1)` on the top
edge; and three interior/axis points at `(1/3, 4/9)`, `(2/3, 4/9)`, and the apex `(1/2, 7/9)`. Eight
points on the boundary, three in the interior — the boundary-heavy, mirror-symmetric pattern the
reasoning predicted. The denominators `9` are what make the area come out to `1/27`: a triangle with
vertices on a `1/9`-style grid has area a multiple of a small fraction, and the maximal max-min layout
forces the smallest such multiple, `(1/3)·(1/9) = 1/27` after the factor of one half, to be shared by a
large family of triangles.

The remaining work is verification, and here I refuse to trust floating point, because the whole point
of this rung is to land *exactly* on `1/27`, not `0.037037...` to some number of digits. I evaluate the
minimum triangle area in exact rational arithmetic: each coordinate is a fraction, each triangle's area
is half the absolute cross product of integer-denominator differences, and the minimum over all `165`
triples is therefore an exact fraction with no rounding anywhere. If the construction is right, that
minimum is the rational `1/27` literally, and a count of how many triangles achieve it tells me how
over-determined — how rigid — the optimum is. I also recompute in double precision to confirm the
evaluator the rest of the ladder uses agrees to machine epsilon, so the number the harness reports and
the exact value coincide.

What I expect: the exact minimum equals `1/27` on the nose, with a large family of triangles all tied
at exactly that value — the binding web that pins the configuration as a rigid max-min optimum — and
every point sitting inside the closed unit square. This rung does not beat the record and it is not
meant to: `1/27` is the conjectured optimum at `n = 11` in the square, believed but unproven, so
reaching it *exactly* is the true ceiling. The search-and-polish ladder got within five parts in a
million by groping toward this arrangement from scratch; this rung closes that last gap the only way it
can be closed — by writing down Goldberg's exact configuration and confirming, in arithmetic that
cannot lie, that its smallest triangle is exactly `1/27`. The still-open part is the same as it was: the
*proof* that no configuration of eleven points does better. The construction can hit the record; only a
proof can certify it, and that is not something a construction or a search supplies.
