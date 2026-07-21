Random multi-start saturated at `0.0109`, a third of the record, and the diagnosis was clean: it finds
a decent configuration and then forgets it. With no memory and no way to *improve* a promising point
set, all it can do is wait for a luckier independent draw, which the extreme-value law makes
exponentially rare. So the move now is to add exactly the two things it lacks — keep the current
configuration, and nudge it toward a larger minimum triangle. If memory-plus-improvement is the right
idea it should not creep, it should jump clean past the `0.0215` eleven-gon in a single run, because it
climbs a landscape instead of resampling one. The hard part is the nudging rule, because the naive
version fails in a way specific to this objective, and I want that failure precise before reaching for
the cure.

Consider plain greedy hill-climbing: pick a point, move it, keep the move only if the minimum triangle
grew. The trouble is that the score is the `min` over `165` triangles. At any configuration a small
handful of "worst" triangles pin the score, and to raise it I must enlarge *every* currently-binding
triangle at once — the moment one stays put the minimum has not moved. But a single-point move only
touches the `C(10,2) = 45` triples containing that point; if the binding triangles are spread across
several different points — exactly what happens at a good, tightly-packed configuration — no
single-point move can even reach them all. And when a move does touch a binding triangle, enlarging it
almost always drags the point in a direction that thins some other triangle through the same point,
which becomes the new worst. So most single nudges either leave an untouched binding triangle pinning
the score or trade one sliver for another, and greedy climbing freezes almost immediately, in a
configuration where a coordinated rearrangement two or three moves away would help. The
minimum-of-many structure manufactures these traps, and the trap depth is set by how many distinct
points the binding triangles span.

I can make that concrete with a configuration I already understand — the inscribed eleven-gon. Its
minimum `0.0215` is achieved by *eleven* triangles at once, one per vertex: the consecutive triple
`(v−1, v, v+1)` is tight for every `v`, and those eleven binding triangles involve all eleven points.
Move one vertex `v`: it sits in only three of those triples, so the *other eight*, which do not contain
`v`, stay pinned at exactly `0.0215`. The global minimum cannot rise from any single-point move — the
eleven-gon is a fixed point of greedy climbing even though it is barely half the record. That is the
trap in its purest form, and it is exactly why annealing has to accept *downhill* moves: to break the
deadlock the search must be willing to shrink the current worst triangle, walk several points to a
different arrangement, and only then find a basin where the minimum climbs.

So the cure is simulated annealing: keep the single-point nudge as the move, but accept a worsening
move with a probability that depends on how much worse it is and on a temperature I cool over time. Hot
early, it accepts moves that shrink the worst triangle and can walk out of a shallow trap; cold late,
only improving moves survive and it settles. The design comes down to a few decisions, and I want each
pinned with arithmetic rather than taste, because a schedule chosen by feel is the hidden tuning I have
been avoiding.

The move: add a small Gaussian displacement to one point, then clip back into the square. I tie the
step loosely to temperature, `σ = 0.05 + 0.5·T`, but the numbers show what that actually does — at the
warm start `T = 0.02`, `σ = 0.06`; at the cold floor `T = 10⁻⁵`, `σ = 0.05005`. Across the whole run
the step slides only from `0.06` to `0.05`, a `20%` modulation dominated by the fixed floor. So the
step is essentially constant at `~0.05`, five percent of the square, big enough to relocate a point but
small enough that a cold move need not wreck the configuration; the temperature-linked bonus is a minor
hot-phase sweetener. What carries the annealing is the *acceptance* rule, and I keep the floor at
`0.05` precisely so the late phase still makes real (usually-rejected) `5%` proposals and can keep
testing escapes when cold.

The acceptance rule: areas are already on an `O(0.01)` scale, so unlike a log-determinant problem I do
not need a logarithm — I anneal directly on `d = cand − cur`, accepting a worsening move with
probability `exp(d/T)`. The temperature then lives on the score's own scale, which makes the schedule
transparent. At the warm start `T = 0.02`, a move shrinking the worst triangle by `0.005` is accepted
with probability `exp(−0.25) = 0.78`; by a full `0.02` — nearly the entire score — `exp(−1) = 0.37`. At
the cold floor `T = 10⁻⁵`, a worsening of even `10⁻⁴` is accepted with probability `exp(−10) ≈
4.5×10⁻⁵`, effectively never. So the schedule transitions cleanly from "cross ridges up to `~0.02`
deep" to "only improving moves survive." The `O(0.01)` scale is a gift: a move worsening the score by
`0.005` means roughly the same whether the score is `0.02` or `0.035`, so a single linear temperature
is well-behaved across the whole run and a logarithm would only add a nonlinearity and a near-zero
floor I would have to guard.

The cooling schedule is geometric from `T₀ = 0.02` to `T_end = 10⁻⁵` over `200,000` steps, so
`cool = (5×10⁻⁴)^{1/200000} ≈ 0.99996` — each step shaves four thousandths of a percent. The
temperature halves about every `18,240` steps, eleven halvings over the run, and it crosses the score
scale `0.01` at only `~9%` of the run. So the first ninth is the genuinely hot, basin-hopping phase
where the configuration reshapes globally, and the remaining `91%` is a long cool settling in which
acceptances progressively narrow to only-improving. That split falls out of the schedule constants
rather than being imposed. And `T₀ = 0.02` is set to the score's scale on purpose: at the start a move
destroying nearly the entire current score is still accepted with probability `exp(−1) = 0.37` — hot
enough that essentially any ridge can be crossed early, but not so hot the search degenerates into the
pure random sampling the previous rung proved worthless.

The fourth decision makes the inner loop affordable: incremental scoring. Moving one point changes only
the `45` triples that involve it, so I keep the full vector of `165` current areas, recompute only
those `45` per step, splice them into a copy, and take the minimum. The splice is *exact*, not an
approximation — the other `120` triangles have all three vertices untouched, so carrying their old
areas forward is an identity, and the min of the spliced vector equals a full recompute exactly. On
reject I restore the old point and the old vector, so no drift accumulates over hundreds of thousands
of steps. Each step becomes array work on `45` numbers instead of `165`, which matters because the
budget is `30 × 200,000 = 6×10⁶` steps and the dominant cost at that scale is the Python per-step
overhead; keeping the array work small stops that overhead from being compounded. One clipping
subtlety: a clipped point can land collinear with edge-neighbours, but that is safe because I score
*after* clipping — a clip that makes a sliver lowers the candidate's minimum, so the Metropolis test
rejects it (or accepts only when hot, where escaping is the point). Clipping cannot silently corrupt
the score.

Finally, restarts, justified by a small probability argument. A single run gambles on its initial
configuration and its accepted-move sequence; some land in good basins, some mediocre. If a single run
has probability `p` of finding a near-record basin, best-of-`R` misses only with `(1−p)^R`; even a
pessimistic `p = 0.1` gives best-of-`30` a hit probability `1 − 0.9³⁰ = 0.96`. So thirty restarts turn
a one-in-ten single-run chance into `96%`, and I expect the per-restart spread to be *wide* — if it
were narrow, multi-restart would be doing no work. Each restart begins from a *fresh* uniform random
configuration, not from the previous rung's best draw: seeding all thirty from one good configuration
would collapse their spread into one basin, whereas independent random starts sample thirty different
regions, and the hot phase of each dissolves its start anyway (a random draw is a bad final
configuration but a fine *starting* one). The division of labour is clean — fresh starts supply
diversity, the hot-then-cool schedule supplies the climb.

What do I expect? Annealing actually *improves* configurations instead of sampling them, so it steps
outside the `ln N` regime entirely. My honest, falsifiable guess is the mid-`0.03`s — around `0.95` of
the record on the best of thirty restarts, with a wide per-restart spread beneath that best — a big
jump from `0.0109`, landing just short of the optimum. That is a change of regime, not a marginal gain:
random search's log law needed `~8×10¹²` draws merely to reach `0.0215`, while annealing sails past it
and most of the way to `0.037` in `6×10⁶` scored steps — not because it evaluates faster but because
its accepted moves accumulate along a connected uphill path instead of resampling.

And I already expect *what* will limit it, from the same knife-edge analysis. Near a good configuration
several triangles are simultaneously near-tight, and — by the argument that made greedy fail — spread
across many of the eleven points. A single-point Gaussian move can only touch the `45` triples through
one point, so it grows some binding triangles while shrinking or leaving others; as the configuration
improves the binding set spreads over *more* points, so the fraction of moves that can lift the whole
minimum shrinks toward zero, and the search plateaus a hair below the optimum — not in the wrong basin,
but because the *move set* cannot coordinate the final squeeze. That plateau is the precise opening for
the next rung: if the only obstruction is the inability to grow several simultaneously-tight triangles
*together*, what I want next is a *smooth* objective whose gradient pushes all the tight triangles up
at once — a differentiable soft-minimum of the triangle areas handed to a real optimizer to polish the
annealed configuration onto the exact optimum.
