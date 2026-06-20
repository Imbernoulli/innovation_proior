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
another (no improvement). Greedy climbing stalls almost immediately in a configuration where every
small move makes the current worst triangle worse, even though a coordinated rearrangement two or
three moves away would help. The minimum-of-many structure manufactures these traps.

The standard cure is simulated annealing: keep the single-point nudge as the move, but change the
acceptance rule. Propose moving one random point a little; if the minimum triangle improves, take
it; if it worsens, take it anyway with a probability that depends on how much worse and on a
temperature I cool over time. Early, when the temperature is high, the search accepts moves that
shrink the worst triangle, so it can walk *out* of a shallow trap and cross a ridge to a different
basin; late, when the temperature is low, only improving moves survive and the search settles into
whatever basin it has wandered into. The bet is that with enough wandering it lands in a basin whose
floor is far higher than anything greedy could reach from a random start.

A few design decisions are specific to the geometry. First, the move. I move one point per step by
adding a small Gaussian displacement, then clip the result back into the square so the configuration
stays legal. I tie the step size loosely to the temperature: when it is hot I allow larger jumps, so
the search relocates points across the square and reshapes the configuration globally; as it cools I
shrink the jumps so the late phase is fine local adjustment. There is always a small floor on the
step so that even when cold the search keeps making real, if tiny, moves.

Second, what the acceptance rule compares. The objective is already on a sane scale — areas are
numbers like `0.01` to `0.04` — so unlike a determinant problem I do not need to take a logarithm; I
can anneal directly on the change in the minimum triangle area. The temperature then lives on the
same scale as the score, of order `0.01`, which makes the schedule easy to reason about: a move that
shrinks the worst triangle by `0.005` at temperature `0.01` is accepted with probability about
`e^{-0.5}`, a coin-flip-ish chance early on, vanishing late. I cool geometrically from a warm start
of about `0.02` down to a tiny floor, over a few hundred thousand steps per run.

Third — and this is what makes the inner loop affordable — incremental scoring. Moving one point
only changes the triangles that *involve* that point: of the `165` triples, the ones containing the
moved index, which is `C(10,2) = 45` of them. So I keep the full vector of `165` current triangle
areas, and on each proposed move I recompute only the `45` affected areas, splice them in, and take
the minimum of the spliced vector. That is the score of the candidate without recomputing the other
`120` triangles. Each step becomes a handful of array operations on `45` numbers instead of `165`,
so I can run hundreds of thousands of steps per second-scale budget and afford many restarts.

Fourth, restarts. A single annealing run still gambles on its initial random configuration and its
particular sequence of accepted moves; some runs land in good basins, some in mediocre ones. The
spread across runs is real, so I run annealing many times from independent random starts and keep the
best minimum triangle found across all of them. Multi-restart annealing turns a noisy single-run
outcome into a reliable best-of-many, and it is cheap given the incremental scoring.

What do I expect? Annealing with the right move set should clear the random-multi-start wall by a
large margin, because it actually *improves* configurations instead of just sampling them. I expect
it to climb well past the eleven-gon and into the neighborhood of the record — a configuration whose
worst triangle is most of the way to `1/27`. How close depends on luck across restarts: the record
configuration is a specific, somewhat structured arrangement, and annealing on continuous
coordinates should be able to drift into its basin and approximate it. My honest guess is the
mid-`0.03`s, somewhere around `0.95` of the record on the best of a few dozen restarts — a big jump,
landing just short of the optimum.

What I expect to *limit* it is the last little bit of precision. Annealing is a stochastic local
search; near the optimum the worst triangle is set by several triples simultaneously, all nearly
equal, and squeezing the final percent of area requires nudging the configuration so that all of
those near-tight triangles grow together. A random single-point Gaussian move rarely does that — it
tends to grow one tight triangle while slightly shrinking another, so the minimum hovers just below
the true optimum and the annealing plateaus a hair short. That plateau is the opening for the
endpoint. If the only thing keeping me from the record is the inability to *coordinate* the final
adjustment of the several simultaneously-tight triangles, then what I want after annealing is a
*smooth* objective whose gradient pushes all the tight triangles up at once — a differentiable
soft-minimum of the triangle areas that I can hand to a real optimizer to polish the annealed
configuration the rest of the way onto the exact optimum.
