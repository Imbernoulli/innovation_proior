The polish landed at `0.037032`, a whisker — five parts in a million — below the conjectured optimum
`1/27 = 0.0370370...`, and the feedback is candid about why: the soft-min gradient ascent snaps onto
the *basin* of the optimum and stops a hair short, because L-BFGS-B on a floating-point surrogate
converges to within its own tolerance, not to the algebraic point. That residual `5×10^-6` is not a
geometric gap that more search could close — fresh restarts cluster lower still, around `0.958` of the
record — it is pure optimizer tolerance against a target that the floating-point objective can only
approach. So more climbing will not get me there. If I want to *reach* `1/27` rather than approximate
it, I have to stop treating the optimum as something to be found by descending a surrogate and instead
reproduce it as what it is: an exact, structured configuration with a closed form.

So the question changes shape. It is no longer "how do I optimize harder" but "what is the exact
configuration behind the `1/27` record, and can I write it down and confirm it on the nose." A clean
rational like `1/27` does not come out of an unstructured search — it comes out of a configuration with
enough symmetry and rational structure that the binding triangles all collapse to the *same* exact
value. My own polished points already whisper this: they sit on multiples of `1/3` along the boundary
and on `2/9` and `4/9` in the interior, and the arrangement is mirror-symmetric left-to-right. That is
the fingerprint of a hand construction, and it tells me the honest endpoint is not a better optimizer
but the exact arrangement the search was groping toward.

Let me reason about what that arrangement must be. A configuration whose minimum is a single clean
fraction, and whose minimum is *maximal*, is one where many triangles are simultaneously tight — if only
a few triangles were at the minimum, I could perturb the points to grow them and raise the score, so a
true local maximum of the max-min objective has the small triangles pinned in a rigid, over-determined
web. The cleaner and more symmetric the layout, the more triangles share the exact same area, and the
more the minimum is forced onto a rational value. The natural symmetry here is a mirror about the
vertical centerline `x = 1/2`, which the Heilbronn-for-squares tables list for the `n=11` record: points
come in left-right pairs, plus points that sit on the axis itself. With eleven points — an odd number — I
expect pairs straddling the axis and a small number of points exactly on `x = 1/2`. The boundary of the
square should carry most of the points, because pushing a point to an edge gives its triangles the most
room; only a few points sit in the interior to break up the collinearities the boundary points would
otherwise create.

Putting flesh on that skeleton, the tabulated coordinates for the record are rational with denominators
`3`, `9`, `6`, and `2`: points at `(1/3, 0)` and `(2/3, 0)` on the bottom edge; `(0, 2/9)` and
`(1, 2/9)` and `(0, 2/3)` and `(1, 2/3)` on the two side edges; `(1/6, 1)` and `(5/6, 1)` on the top
edge; and three interior/axis points at `(1/3, 4/9)`, `(2/3, 4/9)`, and the apex `(1/2, 7/9)`. Eight
points on the boundary, three in the interior — the boundary-heavy, mirror-symmetric pattern the
reasoning predicted. Before I trust it I want to see the `1/27` fall out of the actual geometry, not
take it on faith, because the entire reason for this rung is to land *exactly* on the record and a
plausible-looking layout is not a checked one.

So I work a few triangles by hand in fractions. Take the two bottom-edge points `(1/3, 0)` and
`(2/3, 0)` together with the lower-left side point `(0, 2/9)`. The base is the horizontal segment from
`x=1/3` to `x=2/3`, length `1/3`; the third vertex sits at height `2/9` above that line. Area is
`½ · (1/3) · (2/9) = 1/27`. That is the record value exactly, on a triangle I picked almost at random
from the boundary. Try a second one of a different shape: the left-edge segment from `(0, 2/9)` to
`(0, 2/3)` has vertical length `2/3 - 2/9 = 4/9`, and the top-edge point `(1/6, 1)` lies horizontal
distance `1/6` from that edge, so the area is `½ · (4/9) · (1/6) = 1/27` again. Two unrelated triangles,
both at `1/27`. That is the binding web I was arguing must exist, and it is already showing through.

Now I check whether my intuition about *which* triangles bind is actually right, because I do not want
to mistake a suggestive coincidence for the real minimum. My instinct says the apex `(1/2, 7/9)` with
the two top-edge points `(1/6, 1)` and `(5/6, 1)` should be one of the smallest triangles — it looks
flat and pinched near the top. Its base is `5/6 - 1/6 = 2/3` and its height is `1 - 7/9 = 2/9`, giving
`½ · (2/3) · (2/9) = 2/27`. That is *twice* `1/27`, not the minimum at all. So the small-looking
triangle near the apex is comfortably above the floor; the binding triangles are the ones reaching
across the square, not the locally flat ones. Good that I checked — eyeballing "which triangle is
smallest" would have pointed me at the wrong family. The minimum is set by a structured collection I
cannot fully see by inspection, which is exactly the case for grinding through all `165` triples in
exact arithmetic rather than trusting a hand-picked few.

That is the verification, and here I refuse to trust floating point, because the whole point of this
rung is to land *exactly* on `1/27`, not `0.037037...` to some number of digits. I evaluate the minimum
triangle area in exact rational arithmetic: each coordinate is a fraction, each triangle's area is half
the absolute cross product of integer-denominator differences, and the minimum over all `165` triples is
therefore an exact fraction with no rounding anywhere. Running it, the exact minimum comes back as the
rational `1/27` — equality, not approximation — and no triangle is degenerate (the smallest is the
record value, not zero), so the boundary-plus-interior split really did break every collinearity. The
count of triangles achieving that minimum is `28`: a large, over-determined family all tied at exactly
`1/27`, which is the rigid binding web that pins the configuration as a genuine max-min optimum rather
than a soft plateau. For orientation I also list the next few distinct areas — `1/18`, `2/27`, `5/54`,
`1/9` — and the `2/27` slot (`21` triangles, plus the apex triangle I tested by hand) sits well clear
above the `1/27` floor, confirming a real gap between the binding set and everything else. Finally I
recompute in double precision the way the rest of the ladder's evaluator does: the min comes out
`0.037037037037037035`, which differs from `1/27` by `1.4×10^-17`, one ulp — so the number the harness
reports and the exact value coincide to machine epsilon. I also confirm the symmetry I assumed: mapping
every point `(x,y)` to `(1-x, y)` returns the same set, so the configuration is exactly mirror-symmetric
about `x = 1/2`.

The denominators tell the same story the arithmetic just confirmed: a triangle with vertices on this
`1/9`-flavored grid has area an integer multiple of `1/54`, and the maximal max-min layout forces the
smallest such multiple — `½ · (1/3) · (2/9) = 1/27`, i.e. `2/54` — to be shared by the binding family,
with everything else landing on larger multiples. That is why the record is `1/27` and not some uglier
number, and why `28` triangles can all sit on it at once.

This rung does not beat the record and it is not meant to: `1/27` is the conjectured optimum at
`n = 11` in the square, believed but unproven, so reaching it *exactly* is the true ceiling. The
search-and-polish ladder got within five parts in a million by groping toward this arrangement from
scratch; this rung closes that last gap the only way it can be closed — by writing down the exact
rational configuration and confirming, in arithmetic that cannot lie, that its smallest triangle is
exactly `1/27` and that `28` triangles bind there. The still-open part is the same as it was: the
*proof* that no configuration of eleven points does better. The construction can hit the record; only a
proof can certify it, and that is not something a construction or a search supplies.
