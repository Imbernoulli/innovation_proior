Random multi-start saturated at `0.0109`, a third of the record, and the diagnosis was clean: it
finds a decent configuration and then forgets it. It has no memory and no way to *improve* a
promising point set, so all it can do is wait for an even luckier independent draw, and the
extreme-value law made that luck exponentially rare — the best-so-far only crawls up like the
logarithm of the budget. So the move now is to add exactly the two things it lacks — keep the current
configuration, and nudge it toward a larger minimum triangle. If memory-plus-improvement is the right
idea it should not creep, it should jump clean past the `0.0215` eleven-gon in a single run, because
it climbs a landscape instead of resampling one. The hard part is the nudging rule, because the naive
version fails in a way that is specific to this objective, and I want to understand that failure
precisely before I reach for the cure.

Consider plain greedy hill-climbing: repeatedly pick a point, move it a little, and keep the move
only if the minimum triangle got bigger. The trouble is that the score is the *minimum* over all
`165` triangles, which makes the landscape full of shallow traps, and I can say exactly why. At any
configuration there is one — or, more often near a good configuration, a small handful — of "worst"
triangles pinning the score. To raise the score I must enlarge *every* currently-binding triangle at
once, because the moment one of them stays put the minimum has not moved. But a single-point move only
touches the `C(10,2) = 45` triples that contain that one point; if the binding triangles are spread
across several different points — which is exactly what happens at a good, tightly-packed
configuration — then no single-point move can even *reach* all of them. I lift the binding triangles
through the point I moved, and a binding triangle that does not involve that point sits unchanged, so
the minimum is unchanged. And when a move *does* touch a binding triangle, enlarging it almost always
drags the moved point in a direction that thins some other triangle through the same point, which then
becomes the new worst, no better than before. So the minimum sits on a knife-edge: most single nudges
either leave an untouched binding triangle pinning the score (no improvement) or trade one sliver for
another (no improvement). Greedy climbing freezes almost immediately, in a configuration where every
small move makes the current worst triangle worse, even though a coordinated rearrangement two or
three moves away would help. The minimum-of-many structure manufactures these traps, and the trap
depth is set by how many distinct points the binding triangles span.

I can make that failure completely concrete with a configuration I already understand — the inscribed
eleven-gon from the baseline. Its minimum `0.0215` is achieved by *eleven* triangles at once, one per
vertex: the three-consecutive triple `(v−1, v, v+1)` is tight for every `v`, and those eleven binding
triangles collectively involve all eleven points. Now try to greedily improve it by moving one vertex
`v`. That vertex sits in exactly three of the binding triples — it is the apex of `(v−1, v, v+1)` and a
base-endpoint of `(v−2, v−1, v)` and `(v, v+1, v+2)` — so whatever I do to `v`, the *other eight*
consecutive triangles, which do not contain `v`, stay pinned at exactly `0.0215`. The global minimum
cannot rise above `0.0215` from any single-point move, full stop: the eleven-gon is a fixed point of
greedy single-point climbing even though it is barely half the record. That is the trap in its purest
form — a binding set spread across all `n` points makes the configuration immovable under one-point
moves — and it is exactly why annealing has to accept *downhill* moves: to break the eleven-gon-like
deadlock, the search must be willing to *shrink* the current worst triangle, walk several points to a
different arrangement, and only then find a basin where the minimum climbs. A move set that only ever
improves can never leave such a configuration; a move set that sometimes worsens can.

The standard cure is simulated annealing: keep the single-point nudge as the move, but change the
acceptance rule. Propose moving one random point a little; if the minimum triangle improves, take it;
if it worsens, take it anyway with a probability that depends on how much worse the move is and on a
temperature I cool over time. Early, when the temperature is high, the search accepts moves that
shrink the worst triangle, so it can walk *out* of a shallow trap and cross a ridge into a different
basin; late, when the temperature is low, only improving moves survive and the search settles into
whatever basin it has wandered into. The bet is that with enough wandering it lands in a basin whose
floor is far higher than anything greedy could reach from a random start. The design then comes down
to four decisions, and each one I want to pin with arithmetic rather than taste, because a schedule
chosen by feel is exactly the kind of hidden tuning I have been trying to avoid.

First, the move. I move one point per step by adding a small Gaussian displacement, then clip the
result back into the square so the configuration stays legal. I tie the step size loosely to the
temperature — `σ = 0.05 + 0.5·T` — so that in principle it jumps farther when hot and finer when cold.
But let me actually put in the numbers, because the phrase "tie the step to the temperature" can
promise more than it delivers. At the warm start `T = 0.02`, `σ = 0.05 + 0.01 = 0.06`; at `T = 0.01`,
`σ = 0.055`; at the cold floor `T = 10⁻⁵`, `σ = 0.05005`. So across the *entire* run the step size
only slides from `0.06` down to `0.05` — a `20%` modulation, dominated entirely by the fixed floor of
`0.05`. The honest reading is that the step is essentially a constant `~0.05`, five percent of the
square's width, big enough to relocate a point meaningfully but small enough that a cold move is not
guaranteed to wreck the configuration; the temperature-linked bonus is a minor hot-phase sweetener
that costs nothing but is *not* what carries the annealing. What carries the annealing is the
*acceptance* rule, and I keep the floor at `0.05` precisely because I do not want the late phase to
freeze its moves to nothing — I want it still making real, if usually-rejected, `5%` proposals so it
can keep testing escapes even when cold.

Second, what the acceptance rule compares. The objective is already on a sane scale — areas are
numbers like `0.01` to `0.04` — so unlike a log-determinant problem I do not need to take a logarithm;
I can anneal directly on the change in the minimum triangle area, `d = cand − cur`, accepting a
worsening move with probability `exp(d / T)`. The temperature then lives on the same scale as the
score, of order `0.01`, which makes the schedule transparent, and I can read the acceptance
probabilities straight off. At the warm start `T = 0.02`: a move that shrinks the worst triangle by
`0.005` is accepted with probability `exp(−0.005/0.02) = exp(−0.25) = 0.78`; by `0.01`, `exp(−0.5) =
0.61`; by a full `0.02` — nearly the entire score — `exp(−1) = 0.37`. So early on the search freely
crosses ridges up to about one whole score-unit deep, which is exactly the freedom it needs to leave a
shallow trap. At the cold floor `T = 10⁻⁵`: a move worsening by even `10⁻⁴` is accepted with
probability `exp(−10) ≈ 4.5×10⁻⁵` — effectively never. So the schedule cleanly transitions from "accept
ridge crossings up to `~0.02` deep" to "only improving moves survive," which is precisely the
early-explore, late-refine behaviour annealing is supposed to have.

The reason I can get away with annealing on the raw area difference is that the score barely spans one
order of magnitude — the interesting range is `0.01` to `0.04`, a factor of four. In a problem where the
objective spanned many orders of magnitude — a log-determinant, say, running from `10⁻¹⁰` to `10⁰` — a
single temperature could not simultaneously be "warm" for the small values and "cold" for the large
ones, and I would have to anneal on `log`(objective) so that a *relative* change of fixed size mapped
to a fixed acceptance probability everywhere. Here there is no such stretch: a move worsening the score
by `0.005` means roughly the same thing whether the current score is `0.02` or `0.035`, so a single
temperature on the linear area difference is well-behaved across the whole run, and taking a logarithm
would only add a nonlinearity I do not need and a numerical floor near zero I would have to guard. The
sane scale of the objective is a small gift this problem hands me, and I take it.

Third, the cooling schedule itself, geometric from `T₀ = 0.02` to `T_end = 10⁻⁵` over `iters =
200{,}000` steps. The per-step factor is `cool = (T_end/T₀)^{1/iters} = (5×10⁻⁴)^{1/200000}`. Taking
logs, `ln(cool) = ln(5×10⁻⁴)/200000 = −7.60/200000 = −3.80×10⁻⁵`, so `cool ≈ 0.99996` — each step
shaves the temperature by four thousandths of a percent. The temperature's half-life is `ln(2)/(3.80×
10⁻⁵) ≈ 18{,}240` steps, so over the `200{,}000`-step run the temperature halves about
`200000/18240 ≈ 11` times (`2¹¹ = 2048`, and `0.02/2048 ≈ 9.8×10⁻⁶ ≈ T_end`, which checks). And I can
locate the crossover where `T` falls to the score scale `0.01`: `0.02 · cool^k = 0.01` at `k =
ln(0.5)/ln(cool) ≈ 18{,}240` steps, which is only `9%` of the run. So the first ninth of the schedule
is the genuinely hot, basin-hopping phase where moves worsening the score by a full `0.01`–`0.02` are
routinely accepted and the configuration reshapes globally; the remaining `91%` is a long, slow
settling in which the temperature drops through eleven halvings and the acceptances progressively
narrow to only-improving. That split — a short hot reshaping followed by a long cool refinement — is
what I want, and it falls out of the schedule constants rather than being imposed by hand.

The choice of `T₀ = 0.02` is not arbitrary either; it is set to the scale of the whole score. The
eleven-gon scores `0.0215` and the record is `0.037`, so a temperature of `0.02` means that at the
start a move which *destroys nearly the entire current score* is still accepted with probability
`exp(−0.02/0.02) = exp(−1) = 0.37`. That is the right amount of heat: hot enough that essentially any
ridge in the landscape can be crossed early (the search is nearly free to relocate the configuration),
but not so hot that it degenerates into pure random sampling, which the previous rung already proved
worthless. Warmer than the score scale would waste the early budget wandering with no memory; cooler
would leave the search trapped near its random start. Setting `T₀` equal to the score's own scale is
the principled middle. It is worth tracing the "how deep a downhill move survives" threshold across the
run, since that is what the schedule really controls. A move worsening the score by `d` is accepted
half the time when `exp(d/T) = 0.5`, i.e. at ridge depth `d = 0.693·T`. At step `0` (`T = 0.020`) that
half-accept depth is `0.014`, about a third of the record; by step `18{,}000` (`T = 0.010`) it is
`0.0069`; by step `50{,}000` (`T ≈ 0.0030`) it is `0.0021`; by step `100{,}000` (`T ≈ 0.00045`) it is
`0.00031`; by step `150{,}000` (`T ≈ 6.7×10⁻⁵`) it is `4.6×10⁻⁵`; and by the final step (`T = 10⁻⁵`)
it is `6.9×10⁻⁶`. So the search's tolerance for crossing a ridge narrows by more than three orders of
magnitude, from a third of the record down to a few parts in a million — it starts by reshaping the
configuration wholesale and ends by accepting only microscopic backward steps, which is precisely the
funnel from global exploration to local refinement I am buying with the geometric cool.

Fourth, and this is what makes the inner loop affordable at all — incremental scoring. Moving one
point only changes the triangles that *involve* that point: of the `165` triples, the ones containing
the moved index, which is `C(10,2) = 45` of them, since fixing the moved point the other two vertices
are any two of the remaining ten. So I keep the full vector of `165` current triangle areas, and on
each proposed move I recompute only the `45` affected areas, splice them into a copy of the vector, and
take the minimum of the spliced vector — that is the exact score of the candidate without touching the
other `120` triangles. The splice is exact, and I want to be sure of that because a subtle bug here
would silently corrupt every score: the `120` triangles not involving the moved point are *literally
unchanged* — their three vertices are all untouched — so carrying their old areas forward is an
identity, not an approximation, and the minimum of the spliced vector equals the minimum of a full
`165`-triangle recompute exactly. If I accept the move I keep the spliced vector as the new state; if I
reject it I restore the old point and the old vector is still valid, so no drift accumulates over
hundreds of thousands of steps. Each step becomes a handful of array operations on `45` numbers instead
of `165`, a `165/45 = 3.67×` cut in the arithmetic per step. That matters because the budget is large:
`30` restarts × `200{,}000` steps = `6×10⁶` steps, and the dominant cost at that scale is the Python
per-step overhead of the loop itself (six million iterations, each proposing, scoring, and deciding),
not the array math — so keeping the per-step array work to `45` numbers is what stops the overhead from
being compounded by big-array cost, and the whole thing runs in a minute or two rather than much
longer. There is one thing to check about clipping: when a proposed move would push a point outside the
square, I clip it to the edge, and a clipped point can land collinear with edge-neighbours, making a
thin triangle. But that is safe precisely because I score after clipping — a clip that creates a sliver
lowers the candidate's minimum, so the acceptance rule rejects it (or accepts it only at high
temperature, where escaping is the point). Clipping cannot silently corrupt the score; it can only
propose a bad move that the Metropolis test then judges honestly.

Fifth, restarts, which I justify with a small probability argument rather than just "run it a few
times." A single annealing run still gambles on its initial random configuration and its particular
sequence of accepted moves; some runs land in good basins, some in mediocre ones, and near-optimal
basins are a minority of where a random start plus a stochastic walk ends up. Suppose a single run has
some modest probability `p` of finding a near-record basin. Then the best of `R` independent runs
misses it only with probability `(1−p)^R`. Even a pessimistic `p = 0.1` gives best-of-`30` a hit
probability `1 − 0.9³⁰ = 1 − 0.042 = 0.96` — so thirty restarts turn a one-in-ten single-run chance
into a `96%` chance that at least one run reaches the good basin. That is why `30` is enough and why I
expect the per-restart spread to be *wide*: individual runs will scatter across basins of visibly
different depth, and the reported number is the best of that scatter, not a typical run. If the spread
across restarts were narrow, multi-restart would be doing no work and I would be wasting the budget;
I expect it to be doing real work.

Each restart begins from a *fresh* uniform random configuration rather than from the previous rung's
best draw, and that is deliberate. Seeding every restart from one good configuration would bias all
thirty runs into the same basin, so their "spread" would collapse and the best-of-`30` would gain
nothing over a single run — I would be paying thirty times the compute to explore one basin thirty
times. Independent random starts, by contrast, let the thirty runs sample thirty different regions of
the twenty-two-dimensional space, and it is the hot phase of each that then reshapes its start into
whatever basin it can reach. The random draw is a bad *final* configuration but a perfectly good
*starting* configuration precisely because annealing does not care where it begins — the first
eighteen thousand hot steps dissolve the initial arrangement anyway. So the division of labour is
clean: the fresh random start supplies diversity across restarts, and the hot-then-cool schedule
within each restart supplies the climb.

What do I expect the number to be? Annealing with this move set should clear the random-multi-start
wall by a large margin, because it actually *improves* configurations instead of just sampling them —
it steps outside the `ln N` regime entirely. I expect it to climb well past the eleven-gon's `0.0215`
and into the neighbourhood of the record: a configuration whose worst triangle is most of the way to
`1/27`. My honest, falsifiable guess is the mid-`0.03`s, somewhere around `0.95` of the record on the
best of thirty restarts, with a wide per-restart spread beneath that best — a big jump from `0.0109`,
landing just short of the optimum. If instead it stalls near the eleven-gon, then accepting downhill
moves is not escaping the traps I think it is, and the whole annealing premise is wrong for this
objective.

The efficiency gap over random sampling, if this prediction holds, is worth stating in the same units,
because it is the whole justification for adding machinery. Random search's log law said it would take
about `8×10¹²` draws merely to *reach* the eleven-gon's `0.0215`, and effectively infinite compute to
approach the record. Annealing, if it lands in the mid-`0.03`s, sails past `0.0215` and most of the way
to `0.037` using `30 × 200{,}000 = 6×10⁶` scored steps — six orders of magnitude fewer evaluations
than random would need for a far worse target. The reason is not that annealing evaluates faster; it is
that it *reuses* information. Random draws are independent, so best-of-`N` only improves as `ln N`;
annealing's accepted moves accumulate, so a run walks a connected path uphill through the landscape
rather than sampling it afresh each time. That is the entire payoff of adding memory and a move rule,
and it is why the jump from `0.0109` to the mid-`0.03`s — if I get it — is not a marginal gain but a
change of regime.

And I already expect *what* will limit it, from the same knife-edge analysis that motivated annealing
in the first place. Near a good configuration several triangles are simultaneously near-tight, all
roughly equal to the current minimum, and — by the very argument that made greedy climbing fail — those
binding triangles are spread across many of the eleven points. A random single-point Gaussian move can
only touch the `45` triples through one point, so it grows the binding triangles that involve that
point while, generically, shrinking another binding triangle it also touches, or leaving an untouched
binding triangle pinning the score. As the configuration gets better the binding set spreads over
*more* points, so the fraction of single-point moves that can lift the whole minimum shrinks toward
zero, and the search plateaus a hair below the true optimum — not because it is in the wrong basin, but
because the *move set* cannot coordinate the final squeeze. That plateau is the precise opening for the
next rung. If the only thing keeping me from the record is the inability to grow several
simultaneously-tight triangles *together*, then what I want after annealing is not a better random
kick but a *smooth* objective whose gradient pushes all the tight triangles up at once — a
differentiable soft-minimum of the triangle areas that I can hand to a real optimizer to polish the
annealed configuration the rest of the way onto the exact optimum. Annealing's job is to deliver a
configuration sitting right at the edge of the record's basin; the next rung's job is to make the
coordinated move annealing structurally cannot.
