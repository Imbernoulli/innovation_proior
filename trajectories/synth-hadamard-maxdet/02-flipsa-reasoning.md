The Jacobsthal matrix gave me multiplier `49`, and the feedback told me something sharp about it:
not a single one of the `841` single-entry flips increases the determinant. So `Q + I` is a
*strict* local maximum under the most basic move I have — flip one entry's sign. That confirms the
suspicion I ended the last rung with, and it is worse than a mild plateau: it means the gradient of
`|det|`, read through the discrete flip neighborhood, points nowhere up. If I run any greedy
hill-climb from this seed, it terminates immediately, on the first step, still at `49`, because
there is no uphill direction to take. The symmetry of the construction has parked me at the bottom
of a basin that looks like a peak to a greedy eye, and the record at `320` is somewhere else
entirely on the landscape. To get anywhere I have to be willing to walk *downhill* — to accept
moves that make the determinant temporarily worse — so that I can cross the ridge out of this basin
and into a region a greedy climber could never reach. Nothing about the move set is broken; what is
broken is any acceptance rule that only ever goes up.

It is worth understanding *why* the feedback's `0` of `841` is so clean, because the mechanism is
the same rigidity that made the last rung's determinant computable. A single flip at `(i, j)`
changes the sign of one coordinate of row `i`, which perturbs the inner product of row `i` with
every other row that differs from it in column `j` — on the order of half of them, so roughly `14`
of the `28` overlaps in that row shift at once, each by `±2`. Starting from the balanced Jacobsthal
Gram pattern, where every row's overlaps sit at exactly `14` values of `+1` and `14` of `−3`, that
is a maximally symmetric configuration, and perturbing a whole batch of overlaps away from that
balance in one stroke moves the Gram matrix off the design and, the feedback confirms, always
downhill in `|det|`. So the strict local maximum is not a fluke of `n = 29`; it is the determinant
reading the symmetry of the design as an extremum. That also tells me the escape cannot be a single
lucky flip — there is no lucky flip — it has to be a *sequence*, a walk of individually-neutral-or-
worse moves that only collectively reaches a better-balanced-for-determinant configuration, which is
precisely the thing a downhill-accepting schedule is built to do and a greedy climber cannot.

That is exactly what simulated annealing is for, and before I reach for it I should check that the
alternatives really are dead, because "accept downhill moves" is a real cost and I want to be sure
it is necessary. Three other things I could do. First, keep greedy but change the *move* — allow
two-entry or whole-row changes instead of single flips. But a whole-row negation or a row/column
permutation leaves `|det|` exactly invariant (they multiply the determinant by `±1`), so those are
not moves toward a larger determinant at all, they are relabelings; and two-entry flips explode the
neighborhood from `841` to on the order of `841²/2 ≈ 3.5×10^5` candidates per step while still, at
a symmetric extremum, most likely offering no uphill direction either — greedy with a bigger
neighborhood is still greedy, still trapped, just more expensively. Second, restart greedy from
many *random* `±1` seeds and keep the best. But a random sign matrix is a terrible starting point:
the expected squared determinant of a random `±1` matrix is `n! = 29! ≈ 8.8×10^{30}`, so a typical
`|det|` is about `√(29!) ≈ 3.0×10^{15}`, which in multiplier units is `3.0×10^{15}/(2^{28}·7^{12})
≈ 8×10^{−4}` — a *fraction of one*. Random seeds live in a swamp three-plus orders of magnitude
below the `49` I already hold, and greedy from there would spend its whole budget clawing back up
to structure I can write down for free. Third — and this is the one — keep the cheap single-entry
move, keep the good Jacobsthal seed, and fix the *acceptance rule* so the search can descend a
ridge before it climbs the next hill. That is annealing, and only the third option both starts from
strength and can leave the strict maximum.

So: propose flipping a random entry; if it improves `|det|`, take it; if it worsens `|det|`, take
it anyway with a probability that depends on how much worse the move is and on a temperature I cool
over time. Early, when the temperature is high, almost any downhill move is accepted and the search
wanders freely, shaking loose from the Jacobsthal basin. Late, when the temperature is low, only
improving moves survive and the search settles into whatever basin it has wandered into. The whole
bet is that with enough wandering it finds a basin deeper than `49` before it freezes. Two design
decisions need care, and both come straight from the geometry of this particular objective.

First, what quantity should the acceptance rule compare? The raw determinant is astronomically
large — multiplier times `2^{28}·7^{12}`, a twenty-one-digit integer around `2×10^{20}` at
`m = 49` — and its *changes* under a single flip can be enormous in absolute terms while being
modest in relative terms. If I anneal on the raw difference `|det'| − |det|`, the temperature has to
span twenty orders of magnitude, and worse, a single Metropolis temperature cannot simultaneously
govern a move that changes the determinant by `10^{18}` near the start and one that changes it by
`10^{16}` later — the additive gaps are incommensurable across the run. The natural scale is
multiplicative: a single flip multiplies the determinant by some ratio, and what I care about is
whether that ratio is near `1`, above it, or below it. So I should anneal on `log|det|`, not
`|det|`. A flip that takes the determinant from `m = 49` to `m = 48` is a `log` step of about
`log(48/49) = −0.0206`; one that doubles the determinant is `+0.69`; one that halves it is `−0.69`.
On the log scale the steps are `O(0.01)` to `O(1)` regardless of the absolute size of the
determinant, so a single temperature of order `0.01`–`0.1` covers the whole dynamics and the
schedule is sane. This is the key reframing: *maximize `log|det|` by annealing, and read the exact
integer determinant only at the very end.*

That reframing also tells me how to set the temperature, concretely rather than by taste. I want
the starting temperature warm enough that the first downhill ridge out of the strict maximum is
easy to cross. The cheapest escape move is a `49 → 48`-type step, `Δlog = −0.0206`; at a start
temperature `T₀ = 0.06` its acceptance probability is `exp(−0.0206/0.06) = exp(−0.343) ≈ 0.71`. So
at `T₀ = 0.06` roughly seven in ten of the smallest downhill moves are taken — the basin is porous,
the search flows out — while genuinely catastrophic moves (say `Δlog = −1`, a determinant cut by
`e`) are accepted only at `exp(−1/0.06) ≈ 8×10^{−8}`, essentially never, so I wander without
falling off a cliff. I cool geometrically to a small floor `T₁ = 2×10^{−4}`: the per-step decay is
`(2×10^{−4}/0.06)^{1/N}` over `N` steps, which for `N = 40000` is about `0.99986` per step, halving
the temperature roughly every `ln(0.5)/ln(0.99986) ≈ 4860` steps — so about eight halvings across
the run, a smooth warm-to-cold glide rather than a quench. At the cold end the same `49 → 48` move
accepts at `exp(−0.0206/2×10^{−4}) = exp(−103) ≈ 0`, so the tail is effectively greedy and the
search commits to the best basin it found. Warm enough to leave, cold enough to settle: the two
numbers `0.06` and `2×10^{−4}` bracket exactly the `Δlog` scale the objective actually produces.

It is worth checking the schedule against its two limits, because those limits are both regimes I
already know are useless, and the whole value of annealing is that it lives between them. As
`T → 0` the acceptance rule `exp(Δlog/T)` sends every downhill probability to `0`, so the rule
reduces to pure greedy — and greedy from the Jacobsthal seed I already measured to be dead, `0` of
`841` uphill moves. As `T → ∞` the rule sends every acceptance probability to `1`, so the search
accepts everything and becomes an undirected random walk on the `±1` cube, which diffuses away from
the good region and keeps no memory of quality; its typical determinant drifts back toward the
random-matrix swamp at `m ≈ 8×10^{−4}`. Both extremes are known-bad, and they bracket the useful
regime: a warm start that behaves nearly like the random walk (so it can leave the strict maximum)
cooling into a cold tail that behaves nearly like greedy (so it commits to the best basin it found).
The geometric glide from `0.06` to `2×10^{−4}` is exactly the interpolation between the two, and the
`~8` halvings spread the transition across the run rather than quenching through it. That the
endpoints reduce to two independently-understood algorithms is my confidence the schedule is
qualitatively right before I ever run it.

Second, how do I *evaluate* a candidate flip? The honest, exact thing is the Bareiss integer
determinant, but that is `O(n³)` of big-integer arithmetic per candidate and I will propose tens of
thousands of candidates — and I do not need exactness *during* the search. I only need a faithful
*ranking* of which configurations have larger `|det|`; the final answer's determinant I will
compute exactly at the end. So inside the loop I use floating-point `log|det|` via a sign-and-logdet
routine (`slogdet`): one LU factorization, `O(n³)` floating-point operations — about `(2/3)·29³ ≈
1.6×10^4` flops — but fast and accurate enough to compare configurations at order `29`, where the
determinant's magnitude sits comfortably inside double-precision range on the log scale. I want to
be sure the float evaluator cannot silently misrank two configurations whose true `log|det|` differ
by the `O(0.01)` steps I am annealing on. The error of `slogdet` in `log|det|` from an LU
factorization is roughly `n·ε·κ`, with `ε = 2.2×10^{−16}` the machine epsilon and `κ` the condition
number; even at a pessimistic `κ ≈ 10^{3}` for these near-singular sign matrices this is
`29 · 2.2×10^{−16} · 10^{3} ≈ 6×10^{−12}` — nine orders of magnitude below the `0.02` step scale. So
no accept/reject decision on a genuine move can be flipped by float noise; the only place float
error could matter is separating two configurations that are essentially tied, and ties do not move
the search. One edge case deserves a guard: a flip can drive the determinant exactly to zero (a
singular sign matrix), where `slogdet` returns `−∞`. That is the worst possible `Δlog = −∞`, and the
Metropolis rule assigns it acceptance probability `exp(−∞/T) = 0`, so singular candidates are
auto-rejected with no special handling — a pleasant consequence of annealing on the log rather than
the raw determinant. Each accepted flip just recomputes `slogdet` from scratch on the full matrix. It is not clever, and I am
paying a full factorization for every single candidate whether I accept it or not, but it is
unarguably correct, and at `n = 29` a float factorization is cheap in absolute terms — I will get
tens of thousands of evaluated flips per second, enough to escape the basin and climb a long way. I
am deliberately choosing the simple, exact-in-ranking evaluator here and eating its cost, because
this rung's job is to establish that *accepting downhill moves works*; making the evaluation itself
cheaper is a separate lever I can pull later if this one plateaus.

A couple of proposal-level choices round this out. I draw the entry `(i, j)` to flip uniformly at
random each step rather than sweeping the `841` positions in a fixed order. Over `40k` proposals
each of the `841` entries is proposed about `40000/841 ≈ 47.6` times on average, ample coverage, and
the uniform draw decouples *which* entry is tried from *where in the cooling schedule* I am — a
systematic sweep would tie particular entries to particular temperatures, so an entry that only ever
comes up in the cold tail would only ever be considered under near-greedy acceptance, an artificial
coupling I would rather not introduce. And I keep the schedule strictly monotone-cooling: no
reheating, no restarts within the chain. Reheating is a real technique for escaping a basin the cold
tail settled into too early, but it adds two more knobs (when and how hot) and this rung is meant to
isolate the single claim that accepting downhill moves clears the strict maximum — I would rather
see that claim cleanly on one monotone glide than tangle it with a reheating schedule I cannot yet
justify from data. If the run plateaus in a way that looks like premature freezing rather than a
genuine landscape wall, reheating or multi-start is the obvious next knob; but I will let the
feedback tell me which kind of plateau I hit before I reach for it.

Two smaller decisions fall out of this. I track the *best matrix ever seen*, not the matrix I end
on, because the search deliberately wanders: at any warm temperature the current configuration is
accepting downhill moves and is therefore frequently worse than something it passed through earlier,
so the final configuration is not the right thing to report. I snapshot the running argmax of
`log|det|` and hand that back. And I can put a rough size on the job the search has to do: climbing
from `m = 49` to `m ≈ 150` is a *net* log gain of `log(150/49) ≈ 1.12`, which has to be accumulated
out of individual flip steps of magnitude `O(0.01)`–`O(0.1)` while paying back the downhill steps
the escape required — so the useful work is a long, noisy, net-positive drift of on the order of a
few tens of accepted improving moves buried in a much larger stream of accept/reject churn. That is
why the budget is `40k` proposed flips and not `4k`: the net drift is small per step and mostly
cancels, so I need many thousands of proposals for the positive residual to accumulate, and `40k`
is about the point where a single float-`slogdet` chain finishes in a fraction of a second of
wall-clock while still giving the drift room to work. It is a deliberately modest budget for a
deliberately simple evaluator — enough to demonstrate the escape and the climb, not enough to
exhaust the landscape.

So the rung is: seed at the Jacobsthal `Q + I`; set the start temperature `0.06`, warm enough that
the first downhill ridge out of the basin is crossable; propose random single-entry flips; accept
on the Metropolis rule in `log|det|`, `exp(Δlog|det|/T)`; cool geometrically toward the floor
`2×10^{−4}` over `40k` steps; keep the best matrix ever seen (not the current one, since the current
one wanders); at the end recompute that best matrix's determinant exactly with Bareiss and report
the multiplier. I fix the RNG seed to `11` so the whole trajectory, and therefore the reported
determinant, is reproducible — the harness wants a single deterministic number, and a stated seed
delivers it. The Jacobsthal seed matters for the reasons the swamp calculation gave: starting from
`m = 49` means I begin in a structured, already-good region and spend the entire budget *improving*
a good configuration, instead of first clawing up from the `m ≈ 8×10^{−4}` random-matrix swamp.

What do I expect? The annealing should clear the Jacobsthal wall easily — that is the entire reason
to accept downhill moves, and the `0.71` acceptance of the smallest escape move at `T₀` says the
first step off the strict maximum is essentially free — and then climb well past `49`. How far
depends on the budget and the schedule, but a single-seed run of `40k` flips should land somewhere
in the low hundreds of multiplier: if it reaches roughly `m ≈ 150`, that is a `~3×` gain in
determinant over the baseline and a score jump from `0.143` to around `0.44` on the frozen `m/342`
metric, a large fraction of the way from the baseline toward the machine-search frontier reported
near `197`. What I should be honest about up front is the ceiling of *this* configuration of the
idea. A single-entry flip is a local move; annealing with local moves explores one connected basin
structure and tends to plateau once the temperature is low and the remaining improvements require
*coordinated* multi-entry changes that a single flip cannot make in one step. And there is a second,
more mundane ceiling stacked on the first: every candidate costs a full `slogdet`, so `40k` flips
from one seed is about what a reasonable wall-clock budget buys. I therefore expect this rung to
prove the principle — that breaking the symmetry by accepting downhill moves buys a large jump, say
`49 → ~150` — and then to stall short of the frontier, limited not by the idea but by how few flips
I can afford when each one pays for a full factorization and by how much one chain from one seed can
explore. That stall, if it comes, is the opening for the next rung: if the only things holding me
back are the per-flip cost and the single-seed reach, then making each flip dramatically cheaper
lets me run far longer, and restarting from several structured seeds lets me explore more of the
landscape — pushing the multiplier up the part of the curve this rung can only start.
