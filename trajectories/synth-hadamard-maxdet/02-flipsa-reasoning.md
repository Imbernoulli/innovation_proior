The Jacobsthal matrix gave me multiplier `49`, and the feedback told me something sharp about it:
not a single one of the `841` single-entry flips increases the determinant. So `Q + I` is a
*strict* local maximum under the most basic move I have — flip one entry's sign. That is the wall.
If I run any greedy hill-climb from this seed, it terminates immediately, on the first step, at
`49`, because there is no uphill direction. The symmetry of the construction has parked me at the
bottom of a basin that looks like a peak to a greedy eye, and the record at `320` is somewhere
else entirely on the landscape. To get anywhere I have to be willing to walk *downhill* — to
accept moves that make the determinant temporarily worse — so that I can cross the ridge out of
this basin and into a region greedy could never reach.

That is exactly what simulated annealing is for. Keep the single-entry flip as the move, but
change the acceptance rule. Propose flipping a random entry; if it improves `|det|`, take it; if
it worsens `|det|`, take it anyway with a probability that depends on how much worse and on a
temperature that I cool over time. Early, when the temperature is high, almost any downhill move
is accepted and the search wanders freely, shaking loose from the Jacobsthal basin. Late, when
the temperature is low, only improving moves survive and the search settles into whatever basin it
has wandered into. The whole bet is that with enough wandering it finds a basin deeper than `49`
before it freezes.

Two design decisions need care, and both come from the geometry of this particular objective.
First, what quantity should the acceptance rule compare? The raw determinant is astronomically
large — multiplier times `2^28 · 7^12`, a twenty-one-digit integer — and its *changes* under a
single flip can be enormous in absolute terms while being modest in relative terms. If I anneal on
the raw difference `|det'| − |det|`, the temperature has to span twenty orders of magnitude and
the schedule becomes impossible to tune. The natural scale is multiplicative: a single flip
multiplies the determinant by some ratio, and what I care about is whether that ratio is near `1`,
above it, or below it. So I should anneal on `log|det|`, not `|det|`. A flip that takes the
determinant from `m = 49` to `m = 48` is a `log` step of about `−0.02`; one that doubles it is
`+0.69`. On the log scale the steps are `O(0.01)` to `O(1)`, a temperature of order `0.01–0.1`
covers the dynamics, and the schedule is sane. This is the key reframing: *maximize `log|det|` by
annealing, and read the exact integer determinant only at the end.*

Second, how do I evaluate a candidate flip? The honest, exact thing is the Bareiss integer
determinant, but that is `O(n³)` of big-integer arithmetic per candidate and I will propose tens
of thousands of candidates. I do not need exactness *during* the search — I only need a faithful
ranking of which configurations have larger `|det|`, and the final answer's determinant I will
compute exactly. So inside the loop I use floating-point `log|det|` via a sign-and-logdet routine
(`slogdet`): one LU factorization, `O(n³)` floats, but fast and accurate enough to compare
configurations at order `29`. Each accepted flip just recomputes `slogdet` from scratch. It is not
clever, but it is correct, and at `n = 29` a float factorization is cheap — I will get tens of
thousands of evaluated flips per second, which is enough to escape the basin and climb a long way.

So the rung is: seed at the Jacobsthal `Q + I`; set a starting temperature warm enough that the
first downhill ridge out of the basin is crossable; propose random single-entry flips; accept on
the Metropolis rule in `log|det|`; cool geometrically toward a small floor; keep the best matrix
ever seen; at the end recompute its determinant exactly with Bareiss and report the multiplier.
The seed matters — starting from `Q + I` rather than from a random sign matrix means I begin in a
structured, already-good region (`m = 49`) instead of the random-matrix swamp (where the typical
determinant multiplier is a fraction of one), so the annealing spends its budget improving a good
configuration rather than first clawing up from noise.

What do I expect? The annealing should clear the Jacobsthal wall easily — that is the entire
reason to accept downhill moves — and climb well past `49`. How far depends on the budget and the
schedule, but a single-seed run of a few tens of thousands of flips should land somewhere in the
low hundreds of multiplier, a large fraction of the way from the baseline to the LLM-evolution
frontier near `197`. What I should be honest about up front is the ceiling of *this* move set. A
single-entry flip is a local move; annealing with local moves explores one connected basin
structure and tends to plateau once the temperature is low and the remaining improvements require
coordinated multi-entry changes that a single flip cannot make in one step. So I expect this rung
to prove the principle — that breaking the symmetry by accepting downhill moves buys a large jump,
say `49 → ~150` — and then to stall short of the frontier, limited not by the idea but by how few
flips I can afford to evaluate when each one costs a full `slogdet`. That stall is the opening for
the next rung: if the only thing holding me back is the per-flip cost, then making each flip
dramatically cheaper lets me run far longer, restart from many seeds, and push the multiplier up
the part of the curve this rung can only start.
