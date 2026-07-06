The flat floor gave me a map, not just a number: the objective is a redistribution game under a fixed
total overlap `n²/4`, with a hard conservation floor near `0.25` (precisely `0.2553` at `n = 24`), a
symmetric starting point at `0.5` sitting at twice that floor, and a target band near `0.38` where the
worst shift stands about `1.5×` the mean. The only lever is the *shape* of the heights, and the only way
down is to break the self-complementary symmetry — push cells toward `0` and `1` so the zero-shift
self-overlap `c_0 = n/2 − Σ v_i²` collapses — while arranging things so the mass conservation dumps onto
does not simply re-pile at a new worst shift. So now I actually have to optimize. The first decision is the
piece count, and I deliberately keep it small — two dozen cells — because I want to *understand* what an
optimized profile looks like before I commit to a long vector. With `24` heights the minimax has `2n − 1 =
47` shifts, restarts are cheap, and I can read the shape off the result. The published `~600`-piece records
are the destination, but I should not start there; I should start where the search is fast and the
structure is legible.

Before I reach for an optimizer, though, I should test the cheapest idea the rung-1 analysis hands me,
because if it works I do not need SLSQP at all. Rung 1 showed that a *balanced binary* profile — exactly
`n/2` ones and `n/2` zeros — drives `c_0` to zero, killing the self-overlap that pins the flat profile at
`1/2`. So the tempting move is to just hand-write a balanced binary pattern and read off the bound. Let me
actually do that and not assume. The alternating pattern `1,0,1,0,…` at `n = 24` scores `C = 1.0`. A
half-block, twelve `1`s then twelve `0`s, scores `C = 1.0`. A period-six `1,1,1,0,0,0` pattern scores `C =
1.0`. Every clean hand pattern is *catastrophic* — twice as bad as the flat floor it was supposed to
improve on. The mechanism is exactly the conservation law biting back: binarizing empties `c_0`, but the
`n²/4` that lived there has to reappear, and a regular binary pattern lines its block of `1`s up perfectly
against the complement's block of `1`s at some shift, piling the entire half-mass into that one shift. So I
try randomness instead of structure: `2000` random balanced binary arrangements at `n = 24`. The best of
them is `0.5`; the mean is `0.628`; not a single one beats the flat floor. Binarity alone, whether
structured or random, is not a descent — it is usually a disaster. And the spread is enormous: those random
binaries ran from a best of `0.5` up to a worst near `0.92`, so the binary landscape is not merely bad on
average but wildly variable, a jagged terrain where neighboring arrangements can differ by half in the bound.
That variance is a second, independent argument for a continuous gradient-guided optimizer over blind
sampling — a gradient method can *descend* from wherever it starts, following the smooth surrogate downhill,
instead of gambling on a random draw landing near one of the rare good arrangements. When the terrain is this
rough, sampling loses and descent wins.

It is worth tracing exactly why the half-block detonates, because the mechanism is the whole difficulty in
miniature. Take `v = (1,…,1,0,…,0)`, twelve ones then twelve zeros, so `1 − v = (0,…,0,1,…,1)`. The ones of
`v` sit at positions `0…11`; the ones of the complement sit at positions `12…23`. The overlap `c_k = Σ_i
v_i(1 − v_{i−k})` is nonzero only where a `v`-one meets a complement-one, i.e. `i ∈ [0,11]` and `i − k ∈
[12,23]`, which happens for every one of the twelve `v`-ones simultaneously at the single shift `k = −12`,
giving `c_{−12} = 12` and `C = 12·(2/24) = 1.0`. So the entire half-mass collides at one shift. This is the
conservation law at its most hostile: `c_0` is empty, but the `n²/4 = 144` of total overlap is not gone, it
is concentrated into a single spike of height `12` instead of the flat triangle's `6`. Killing the zero-shift
peak without controlling where the mass lands can *double* the worst overlap. That is the precise thing the
optimizer has to avoid, and the reason a good profile must spread its `1`s and `0`s so no shift ever lines
up a large block against a large complementary block.

That is a genuine wall, and it teaches me two things I did not get from rung 1. First, the *arrangement* of
the corners is everything, and it is not something I can guess or stumble into by sampling — the good
arrangements are a vanishing fraction that a search has to find deliberately. Second, and more subtly, pure
binary is too rigid to flatten the envelope: with only `0`s and `1`s I cannot finely *detune* the shifts
that are competing to be worst, because I cannot make small height adjustments. The sub-`0.5` profiles must
therefore be *near*-binary — mostly at the corners to keep `c_0` small, but with a minority of interior
cells that fine-tune the balance and shave the tied shifts apart. That immediately rules out a
combinatorial search over binary strings and points me at a *continuous* constrained optimizer over the box
`[0,1]`, which can both explore arrangements and place the interior heights that binary cannot. So the wall
does real work: it converts "optimize the heights" into a specific shape of problem — continuous, boxed,
sum-constrained, and non-convex enough that structure and randomness both fail.

I should be clear-eyed that there is no shortcut hiding here. The continuous Erdős problem — the infimum of
the worst overlap over all step functions — is itself unsolved after seventy years, so I cannot expect an
analytic optimum to fall out at any finite `n`; the discrete instance inherits that hardness. And the
landscape is genuinely non-convex, not merely awkward: I just watched profiles scoring `1.0` sit adjacent to
profiles scoring under `0.4`, so the objective has deep bad basins and (presumably) many shallow good ones,
which is the defining situation where local descent from a single start is unreliable and global search is
mandatory. That is what justifies the whole shape of this rung — a small, legible instance attacked by a
smooth-surrogate local solver wrapped in multi-start — and it is why I treat the coarse result as a
*measurement* of what this resolution can reach rather than as a solved optimum. The strategy of the ladder
is to make an intractable global optimization tractable by solving a small instance well, understanding its
shape, and then lifting that understood shape to finer resolution, rather than launching a blind search
directly at hundreds of variables.

Now, what am I actually minimizing, and why is it hard? The score is `max_k c_k` over `47` integer shifts,
rescaled — a *minimax*. The objective is the pointwise maximum of `47` smooth functions of the heights (one
`c_k` per shift), so it is piecewise-smooth with kinks exactly where two shifts tie for the worst overlap.
A plain gradient method chatters at those kinks, and — worse — a minimizer that only sees the hard `max`
gets no gradient information about the *other* near-worst shifts that are one step away from becoming
binding, so it lowers today's worst shift straight into tomorrow's. I saw the germ of this at the flat
point: a small perturbation lowers `c_0` until a neighboring shift climbs past it and the score bounces
back up. Optimizing a hard `max` directly is brittle for exactly this reason. The fix is to replace the
hard `max` with a smooth surrogate — a log-sum-exp soft-max over the shifts with a sharpness `β`,

```
C̃_β(v) = ( m + (1/β) log Σ_k exp(β (c_k − m)) ) · 2/n,     m = max_k c_k,
```

written with the `m` shift for numerical stability. For moderate `β` this surrogate is smooth and
differentiable, *feels* all the near-worst shifts at once and pushes them down together, and converges to
the true `max` as `β → ∞`. So I will minimize the soft-max overlap, not the hard one.

But I have to be careful about how faithfully the surrogate tracks the number I actually report, and I can
quantify it. The log-sum-exp overshoots the true max by at most `log(#shifts)/β` in raw units, so in the
rescaled score the surrogate sits below the true worst overlap by at most `(2/n)·log(2n−1)/β`. At `n = 24`
even the sharpest level I plan to use, `β = 1200`, leaves a gap of about `(2/24)·log(47)/1200 ≈ 2.7×10⁻⁴`.
That is *the same order* as the entire distance I expect to have left between a coarse solution around
`0.3812` and the published record `0.38087` — so I cannot just optimize the surrogate and quote it; the
surrogate would flatter me by a couple of ten-thousandths, exactly the scale that matters. Two consequences
follow. I *anneal* `β` up a ladder `(60, 150, 300, 600, 1200)`, solving at each level: the early soft
levels let the whole profile reorganize without locking onto a single binding shift, and the late sharp
levels make the surrogate hug the constant I care about. And I always *score the true hard-max* overlap of
whatever vector comes back, never the surrogate value — the honest number is the hard one, and the
surrogate is only a vehicle to get a good vector.

The second issue is the constraints, and they are what make this the Erdős problem rather than a trivial
"set everything to zero." The heights must lie in the box `[0,1]` and sum to exactly `n/2`. The sum
equality is load-bearing: rung 1 already showed that without it the minimizer drives every height to `0`,
zeroing every product and certifying a meaningless bound of `0`; `Σ v = n/2` is the discrete image of the
`A`/`B` balance that forbids that escape. So I need a solver that handles a box *plus* a linear equality
natively, and the natural choice is SLSQP — sequential least-squares quadratic programming — whose
wheelhouse is exactly box-plus-equality, and which the agentic-search record (AutoEvolver) reports using on
this very problem. Its internal quadratic subproblem is roughly cubic in the number of variables, which at
`n = 24` is nothing — a few hundred iterations cost milliseconds — so I can afford to run the full annealed
ladder many times over.

A word on how SLSQP sees the objective, because it shapes what I can afford. I do not hand it an analytic
gradient of the soft-max; at `24` variables scipy's SLSQP will finite-difference the surrogate itself, which
costs one extra objective evaluation per variable per gradient — a couple dozen extra cross-correlations
per step, trivial here. That convenience is quietly resolution-dependent: a numeric gradient costs `O(n)`
objective evaluations, each an `O(n²)` correlation, so `O(n³)` per gradient step, and the QP subproblem is
itself super-linear in `n`. At two dozen cells none of that registers; I note it now only because it is
exactly the cost that will eventually make this solver the bottleneck at record resolution, and I would
rather see the wall coming than be surprised by it.

The `β` ladder itself deserves a design choice rather than a guess. I want each level roughly `2`–`2.5×`
sharper than the last — `60 → 150 → 300 → 600 → 1200` — and I warm-start each solve from the previous
level's output, so that the surrogate's landscape deforms *gradually* from a soft, nearly-convex bowl at
`β = 60` toward the kinked true minimax at `β = 1200`. If I jumped straight to a sharp `β`, the surrogate
would already be nearly as kinked as the hard max and SLSQP would stall at the first tie it meets; if I
stopped at a soft `β`, the surrogate's peak would sit the full `(2/n)log(2n−1)/β` below the true worst
overlap and I would be optimizing a systematically wrong objective. Annealing threads between the two: soft
enough early to move freely, sharp enough late that the last solve is genuinely descending the number I
report. Five levels is enough to keep any single step's change in `β` small relative to the spread of the
`c_k`, which is what keeps the warm start useful.

There is a subtlety in keeping the scored vector *exactly* feasible, because SLSQP's equality is only
satisfied to its tolerance and the box clip can nudge the sum. After each solve I re-project onto the
constraint set: clip to `[0,1]`, measure the residual `n/2 − Σ v`, and spread it across the strictly
*interior* cells, iterating until the sum is restored, with a final clip. The choice to redistribute only
over free (non-saturated) cells matters — pushing mass into a cell already pinned at `1` would just get
clipped straight back off on the next pass, so the residual would never close. Spreading over the interior
cells is a Michelot-style projection onto the box-and-hyperplane intersection, and it converges in a
handful of passes because each pass either closes the residual or saturates another cell, and there are
only finitely many cells to saturate.

The third issue is local minima, and the binary catastrophe already warned me how vicious this landscape
is: profiles scoring `1.0` sit right next to profiles scoring `0.38`, so the basins are many and some are
terrible. One SLSQP ladder from one start finds one basin, and there is no reason to think it is a good
one. At `n = 24` the decisive, cheap remedy is *multi-start*: run the full annealed ladder from `12` random
feasible initializations under a fixed seed and keep the vector with the best *true* overlap. Each start is
a fresh random height vector projected to sum `n/2`, and the basin SLSQP settles into depends sharply on
where it begins. A dozen starts at two dozen cells is a trivial budget — `12` restarts × `5` `β`-levels ×
`250` iterations, all at the millisecond scale of a `24`-variable QP — and it gives a reliable read on the
best the coarse resolution can reach. I can put a crude number on why a dozen suffices at this scale: if
some fraction `p` of random starts flow into the best basin, then `12` independent starts miss it with
probability `(1 − p)¹²`, which is already under `7%` at `p = 0.2` and under `1%` at `p = 0.32`. I do not
know `p` in advance, but the point of keeping `n` small is that the basins are few enough and wide enough
that a modest `p` is plausible, so a dozen starts is a sensible bet — and if the returned value looks
basin-limited rather than resolution-limited, the cheap fix is simply more starts, not a longer vector. At
larger `n` this arithmetic turns against me: more cells means more and narrower basins, `p` shrinks, and a
fixed dozen starts stops being enough — another reason this careful, legible pass belongs at small `n`,
where I can trust that what I measure is the resolution ceiling and not a missed basin.



What do I expect from all this? Rung 1 predicted a big first drop off `0.5` followed by a long grind, and
this rung is the test of the big first drop. Breaking the flat symmetry with a genuinely optimized
near-binary profile should be worth most of the descent: the literature's earliest optimized step functions
already crossed `0.5 → 0.4 → 0.38`, and even a coarse profile captures the gross structure that flattens the
envelope. So I expect to land in the low `0.38`s — somewhere a touch above the Haugland-2016 landmark
`0.380927` and the AlphaEvolve `0.380924`, and within a few ten-thousandths of the AutoEvolver record
`0.38086945` — but *not* at the frontier, because two dozen wide cells cannot resolve the fine structure a
near-optimal profile needs. Reading it through the conservation lens: flat sits at peak-to-mean `1.958` at
`n = 24`, the geometric floor is `1.0`, and a landing near `0.381` would mean peak-to-mean `≈ 1.49`, so the
coarse optimizer covers roughly half of the reachable peakedness range in a single rung — a big first drop,
exactly as predicted, with the remaining half being the long grind the later rungs must do. The shape I
expect to see is asymmetric and near-binary, with a substantial fraction of cells pinned to the box corners
and a minority at interior values doing the fine detuning — I will check that pinned fraction in the
returned vector, since the whole argument for a continuous optimizer rests on those interior cells
existing.

It is worth measuring the big first drop against the *provable* window rather than just the flat floor,
because that is the honest scale of what one coarse optimization buys. The whole descent from flat to White's
lower bound is `0.5 − 0.379005 = 0.120995`. A landing at `0.3812396` would sit `0.1187604` below the flat
floor, which is `0.9815` of that total window — a single coarse rung closing more than `98%` of the distance
from the trivial `0.5` down to the provable floor `C5 ≥ 0.379005`. That is the sense in which breaking the
symmetry is worth "most" of the descent: `98%` of the gap falls in one step, and the entire remaining
enterprise of this ladder — coarse `0.3812` down toward the record `0.38087`, and the still-open sliver
below that to `0.379005` — lives in the last `~2%`. Every later rung is fighting over ten-thousandths inside
that final slice, which is exactly why the surrogate-vs-max gap of `2.7×10⁻⁴` I computed is not a rounding
detail but the same size as the prize.

I can also read off why two dozen cells caps me here, in terms of the interior heights the wall forced me to
allow. To keep `c_0` small most cells want to be pinned at `0` or `1`; the sum constraint `Σ v = n/2` is
then carried by the minority of interior cells, and those interior cells are the *only* knobs I have for
detuning the shifts that compete to be worst — nudging one of them slightly separates two tied `c_k`. If, as
I expect, something like a quarter to a third of the `24` cells are pinned, I have on the order of a dozen or
so interior detuning knobs to flatten a `47`-shift envelope. That is a coarse instrument: each knob moves
several shifts at once and I cannot address the tied shifts individually, so the worst overlap bottoms out at
a granularity floor. More cells would mean more interior knobs and finer detuning — which is precisely the
lift the next rung buys. I will check the pinned fraction in the returned vector, since the entire argument
for a continuous optimizer, and this reading of the resolution cap, rests on those interior cells existing.

The honest number I report is the true `max_k`-overlap of the best returned vector under the frozen
evaluator, and I expect it to come in around `0.3812`. This sets up a clean, falsifiable test for the next rung, phrased in the two quantities the evaluation
actually tracks — the piece count and the upper bound. If the cap at `~0.3812` is genuinely *resolution*,
then adding pieces should lower the bound (a little), while adding more restarts at `n = 24` should not,
because I would already be at the coarse basin floor. If instead I have merely *missed a basin*, then more
starts at the same `n = 24` would find something lower without any lift at all. The two mechanisms make
opposite predictions about which knob helps, and the next rung — which lifts the piece count — is the test:
a modest drop on lifting, and none from extra coarse restarts, would confirm the resolution reading; a
large drop from either would refute it. I expect the former, precisely because the interior-knob argument
says `24` cells simply cannot represent the fine detuning a lower envelope needs.

The limitation this rung will expose is resolution:
`24` wide cells give only `47` shifts and a coarse set of interior heights, so the worst overlap cannot be
shaved as finely as a long vector allows — the tied shifts are few and each carries a lot of mass, so
detuning them apart hits a granularity wall. The next rung has to lift this optimized coarse profile to many
more pieces and refine it there, where the extra degrees of freedom let the optimizer carve the finer
structure that brings the bound down toward the published step-function frontier — and the flat baseline's
control still holds over that lift: because a bare piece count buys nothing on its own, any drop I see as the
cells multiply will be the genuine work of the finer shape, not an artifact of resolution granting free
progress.
