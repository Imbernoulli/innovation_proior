Random multi-start saturated at `0.0109`, a third of the record, and the diagnosis was clean: it
finds a decent configuration and then forgets it. It has no memory and no way to *improve* a
promising point set. So the move now is to add exactly those two things — keep the current
configuration, and nudge it toward a larger minimum triangle. The hard part is the nudging rule,
because the naive version fails in a way that is specific to this objective.

Consider plain greedy hill-climbing: repeatedly pick a point, move it a little, and keep the move
only if the minimum triangle got bigger. The trouble is that the score is the *minimum* over all
`165` triangles, which makes the landscape full of shallow traps. At any configuration there is one
(or a few) "worst" triangles pinning the score. To raise the score I must enlarge *that* worst
triangle — but moving the point that fixes it almost always shrinks some *other* triangle, which
then becomes the new worst, often no better than before. So the minimum sits on a knife-edge: most
single nudges either leave the worst triangle untouched (no improvement) or trade one sliver for
another (no improvement). Let me make sure this isn't just hand-waving. I take the regular inscribed
`11`-gon — the prior rung's structured baseline, whose minimum triangle is `0.021456` — and ask what
a greedy climber does from there. In that configuration the worst triangles are the three nearly
collinear consecutive-vertex slivers, all close to equal. Pushing the offending middle vertex
outward to fatten one of those slivers immediately thins its neighbor on the other side, because the
two share that vertex; the minimum doesn't move. That is the trap in miniature, and it is structural,
not bad luck: any single-point move that helps the current-worst triangle is geometrically forced to
hurt an adjacent one of comparable area. Greedy climbing stalls almost immediately in a configuration
where every small move makes the current worst triangle worse, even though a coordinated rearrangement
two or three moves away would help. The minimum-of-many structure manufactures these traps.

So I need a search that can take a *worsening* move now to reach a better basin later. The cleanest
way to do that without abandoning the cheap single-point move is to keep the move and change only the
acceptance rule: propose moving one random point a little; if the minimum triangle improves, take it;
if it worsens, take it anyway with a probability that depends on how much worse and on a temperature
I cool over time. That is simulated annealing, and the reason it fits here is precisely the trap
structure above — early, when the temperature is high, the search accepts moves that shrink the worst
triangle, so it can walk *out* of a shallow trap and cross a ridge to a different basin; late, when
the temperature is low, only improving moves survive and the search settles into whatever basin it
has wandered into. The bet is that with enough wandering it lands in a basin whose floor is far
higher than anything greedy could reach from a random start.

A few design decisions are specific to the geometry. First, the move. I move one point per step by
adding a small Gaussian displacement, then clip the result back into the square so the configuration
stays legal. I tie the step size loosely to the temperature: when it is hot I allow larger jumps, so
the search relocates points across the square and reshapes the configuration globally; as it cools I
shrink the jumps so the late phase is fine local adjustment. There is always a small floor on the
step so that even when cold the search keeps making real, if tiny, moves.

Second, what the acceptance rule compares. The objective is already on a sane scale — areas are
numbers like `0.01` to `0.04` — so unlike a determinant problem I do not need to take a logarithm; I
can anneal directly on the change in the minimum triangle area. The temperature then lives on the
same scale as the score, of order `0.01`. Let me check that the schedule this implies is actually
reasonable rather than just asserting it is. Take a move that shrinks the worst triangle by `0.005`
at temperature `0.01`: the Metropolis probability is `exp(-0.005/0.01) = exp(-0.5)`, which evaluates
to `0.607`. So early on a move that costs half a percent of the whole record's worth of area is still
a near coin-flip — the search really is willing to step down. Now cool the temperature to `1e-4` and
keep the same `0.005` drop: `exp(-0.005/1e-4) = exp(-50) ≈ 2e-22`, effectively never. So the same
move that was a coin-flip when hot is forbidden when cold, which is exactly the hot-explore /
cold-commit behavior I want, and it confirms the warm start should be around `0.02` (a couple of
record-fractions of slack) cooling to a tiny floor, over a few hundred thousand steps per run.

Third — and this is what makes the inner loop affordable — incremental scoring. Moving one point
only changes the triangles that *involve* that point. Of the `C(11,3) = 165` triples, the ones
containing a fixed index number `C(10,2) = 45`; I checked this directly by counting, for several
choices of the moved point, how many of the `165` triples contain it, and it is `45` every time. So I
keep the full vector of `165` current triangle areas, and on each proposed move I recompute only the
`45` affected areas, splice them in, and take the minimum of the spliced vector. The thing that could
silently break here is the splice: if I get the index bookkeeping wrong, the candidate score is
quietly wrong and the whole search optimizes a phantom objective. So before trusting it I ran a
direct equality test — make a random configuration, move one point, build the score both ways (splice
the `45` recomputed areas into the kept vector vs. recompute all `165` from scratch) and compare. The
two vectors agreed to `maxdiff 0.0`, exact equality, so the incremental score is the true score and I
can rely on it. Each step is then a handful of array operations on `45` numbers instead of `165`;
timing a tight loop of these incremental steps clocks about `6e4` per second in plain NumPy, so a run
of a few hundred thousand steps is a second or two, and dozens of restarts fit a second-scale budget
comfortably.

Fourth, restarts. A single annealing run still gambles on its initial random configuration and its
particular sequence of accepted moves; some runs land in good basins, some in mediocre ones. I want
to know how real that spread is before committing to multi-restart, so I ran eight independent
annealing runs (`200k` steps each, the schedule above) and looked at the best minimum triangle each
reached: `0.0277, 0.0277, 0.0266, 0.0291, 0.0311, 0.0316, 0.0332, 0.0333` — as fractions of the
record `1/27`, a spread from `0.72` to `0.90`. So the run-to-run variation is large, a full `0.18` of
the record between a bad run and a good one, and a single run genuinely cannot be trusted to land
high. Taking the best across the eight pulls the outcome up to `0.0333`, the `0.90`-of-record run,
for free. That settles it: run annealing many times from independent random starts and keep the best
minimum triangle found, which the incremental scoring makes cheap.

So where does this land, measured rather than hoped? Best-of-eight reached `0.0333`, about `0.90` of
the record — a huge jump over the random-multi-start `0.0109` and comfortably past the eleven-gon
`0.0215`, which is what annealing buys by actually *improving* configurations instead of sampling
them. With the full `30` restarts I would expect the best to creep a little higher than `0.90`, since
the tail of good basins gets sampled more, but the runs are clearly clustering in the high `0.0`s and
not pressing against `1/27`. The interesting part is *why* they stop short, and the eight runs show
it: even the best run plateaus around `0.90` and the worst sit near `0.72-0.75`, none of them closing
the final tenth onto the optimum.

Looking at what the good runs converge to explains the plateau. Near the optimum the worst triangle
is set by several triples simultaneously, all nearly equal, and squeezing the final percent of area
requires nudging the configuration so that all of those near-tight triangles grow together. A random
single-point Gaussian move rarely does that — it tends to grow one tight triangle while slightly
shrinking another, so the minimum hovers just below the true optimum and the annealing plateaus a
hair short, which is exactly the `0.90`-not-`1.0` ceiling I measured. That plateau is the opening for
the endpoint. If the only thing keeping me from the record is the inability to *coordinate* the final
adjustment of the several simultaneously-tight triangles, then what I want after annealing is a
*smooth* objective whose gradient pushes all the tight triangles up at once — a differentiable
soft-minimum of the triangle areas that I can hand to a real optimizer to polish the annealed
configuration the rest of the way onto the exact optimum.
