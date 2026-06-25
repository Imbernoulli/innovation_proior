Warmup + cosine is the strongest rung, and reading its numbers tells me both why it won and where it is
still leaving accuracy behind. It landed {92.71, 72.43, 94.83}: the short five-epoch ramp recovered
ResNet-56 / CIFAR-100 from cosine's 71.07 all the way to 72.43 — the biggest move of any setting, exactly
the init-protection the deep net needed — and FashionMNIST nudged up to 94.83, while ResNet-20 / CIFAR-10
slipped a hair to 92.71 from cosine's 93.03 (a five-epoch sub-`base_lr` ramp is a near-no-op on the easy
net, and here it cost it a fraction). So the warmup hypothesis was right and the schedule strictly improved
the hard net. But step back and look at what warmup+cosine — and cosine, and one-cycle before it — all have
in common, because that shared property is the next thing to attack. Every rung so far starts *cooling the
rate almost immediately*. Cosine is at its peak `base_lr` for a single instant, epoch 0, then descends the
whole way. Warmup+cosine is at `base_lr` for one instant, epoch 5 (the top of the ramp), then the cosine
takes over and descends. One-cycle held `base_lr` only momentarily at the top of its 30% ramp. None of
them *stays* at the high rate; they all treat the peak as a point to pass through on the way down. The
question I have not asked yet is whether passing through the peak so quickly is itself the limitation —
whether the schedules are tuned for *getting to a minimum fast* when what actually decides test accuracy is
*which* minimum they get to.

That reframing is the move. In these nets, generalization is governed by the *geometry* of the minimum SGD
lands in: wide, flat minima generalize well, sharp, narrow ones do not. And the learning rate is the lever
that controls which kind SGD finds, through the noise it injects. To first order the stochastic-gradient
noise scale is `g ≈ eps·N/(B·(1−m))` — it rises with the rate `eps`. A higher rate means a noisier descent,
and noise is exactly what ejects SGD from a narrow basin: a sharp minimum is a small target with steep
walls that a noisy step bounces out of, while a wide minimum is a large flat region the same noise cannot
escape. So a high learning rate is a *wide-minimum-seeking* device — it keeps the search hot enough to be
thrown out of narrow basins and to settle only where the basin is wide. The learning rate is the
*temperature* of the search, and "decay the rate" is really "cool the search." Now the limitation of every
prior rung is visible: they all cool almost from the start, so they barely search at high temperature
before they begin settling — they find *a* minimum fast, but they are biased toward whichever basin is
nearby when they start cooling. That is a very different objection from the ones the ladder has raised so
far. One-cycle's problem was amputated machinery; cosine's was a too-high floor and a missing warmup;
warmup+cosine fixed both ends. This objection is about the *middle*, which no rung has touched.

Which basin is nearby when cooling starts? That depends on the *density* of wide versus narrow minima in
the landscape, and the hypothesis I will design from is that wide minima are *rarer* — the landscape is
full of narrow basins and only sparsely dotted with wide ones. If that is true, a search that cools quickly
almost certainly falls into one of the abundant narrow basins long before it stumbles onto a rare wide one.
The only way to find a rare wide basin is to *stay hot for a long time* — keep the rate high long enough
that the rare wide basins get sampled — and only *then* cool to settle into the one you found. I can
sanity-check the hypothesis against something the field knows directly and that the ladder hints at:
large-batch training generalizes worse. Under the noise scale, a larger batch `B` means *less* noise, a
colder search, which by this argument settles into a narrow basin sooner — and large-batch training is
indeed known to find sharper minima and generalize worse. The pieces line up, so I will take "wide minima
are lower-density, so you must explore long to find them" as the working hypothesis.

This is precisely the diagnosis that warmup+cosine's win sharpens rather than answers. The warmup fixed the
*start* (init curvature); the cosine fixed the *finish* (smooth settle to near-zero). But neither addressed
the *middle* — and the middle is where the wide-minimum search has to happen. The cosine body begins
descending the instant warmup ends, so the high-temperature explore phase is essentially zero epochs long.
If wide minima are rare, warmup+cosine is cooling before it has searched, and the accuracy it leaves on the
table is exactly the gap between the narrow basin it settles into quickly and the wide basin it would have
found if it had stayed hot. The 72.43 on CIFAR-100 is the best the ladder has reached, but a deep,
overparameterized net on a hard 100-class problem is precisely the setting with the richest wide-minimum
structure to find — and the most to lose by cooling before finding it. So the next move is forced by the
hypothesis: insert a deliberate, sustained *explore* phase — hold the rate flat at the peak for a long
stretch — *before* cooling. Explore, then exploit.

Design the two phases. The explore phase I want *constant at the peak* — not a ramp, not a slow cosine
decay, just held at `base_lr` — because the entire point is to keep the temperature maximal and steady so
the search keeps sampling basins for the full explore budget; any decay during explore is premature
cooling, the disease I am curing. So explore is a flat plateau at `base_lr`. The peak is `base_lr` itself,
the same rate the winning baselines already use as their high value — chosen to be as high as these nets
tolerate, which the whole ladder has confirmed is safe *after* the five-epoch warmup. I do not need a
larger peak: one-cycle already showed the harness neuters the range-test / super-convergence regime (no
momentum cycling, no weight-decay rebalancing, so I cannot safely push the peak past what the fixed weight
decay can absorb), and this schedule does not want a bigger peak anyway — it wants to *hold* the peak I
already trust, longer. That is the crucial distinction from one-cycle: one-cycle tried to get more
regularization from a *higher* peak (and the harness blocked it); the explore-exploit schedule gets more
wide-minimum search from a *longer* time at the *same* peak, which the harness allows freely.

Now the exploit phase. Once the plateau has parked the iterate in or near a wide basin, I cool the search
to settle into it — drive the rate down so the noise shrinks and SGD descends to the bottom of the wide
region and stays. What cooling shape? Here I deliberately do *not* reach for the cosine, even though the
cosine won the last two rungs — because on this schedule the explore plateau is carrying the generalization
load, and the exploit phase only has to bring the rate monotonically to zero over the remaining epochs. A
plain *linear* decay from `base_lr` to 0 is the honest minimal choice: it is the strongest simple monotone
budgeted decay, it has one obvious slope, and it reaches exactly 0 at the budget boundary so the final
steps vanish and the basin fully settles. The slope is set so the line hits 0 at the end:
`-base_lr / decay_epochs`, and the rate is `base_lr + slope × (epochs into exploit)`, clamped at 0 so it
never goes negative. No milestones, no shape parameter — the explore plateau, not the exploit curve, is
where the design lives, so I keep the curve as plain as possible.

The split between explore and exploit is the one real knob, and the density hypothesis tells me how to set
it. Too little explore and I cool before finding a wide basin — straight back to warmup+cosine's failure
mode, the immediately-descending body. Too much explore and I leave too few epochs to settle, so I end
*near* a wide basin but never descend into it and the training loss stays high. So there is an interior
optimum, and the hypothesis says it should be *large*, because rare wide basins need a long search. The
principled way to pick it is to sweep a few explore fractions and watch both the accuracy *and* the width
of the minimum reached; the expectation is that more explore monotonically widens the minimum and raises
accuracy until too little exploit budget remains. The value that balances "enough search to find a rare
wide basin" against "enough settling to descend into it" lands at half the budget — so explore = 50% of
`total_epochs`, exploit = the rest. That single fraction is the whole new design surface.

Now compose it with the warmup the last rung proved I need, and land it in the edit surface. I keep the
five-epoch linear ramp from `base_lr/5` to `base_lr` in front — it is pure init hygiene, complementary to
explore-exploit, and the ladder already showed the deep net needs it (it was the entire reason CIFAR-100
went from 71.07 to 72.43). Then the three segments tile the 200-epoch budget: warmup (epochs 0–4, linear
ramp to `base_lr`), explore (the next 50% of the budget, 100 epochs held flat at `base_lr`), exploit (the
remaining 95 epochs, linear decay from `base_lr` to 0 with slope `-base_lr / decay_epochs` and
`decay_epochs = total_epochs − warmup − explore_epochs = 95`). The edit surface is the same single `get_lr`
float per epoch — and notably this schedule needs *no* momentum or weight-decay change, unlike one-cycle:
it earns its regularization by holding the existing peak longer, entirely within the lever the harness
exposes. I check the seams: warmup ends at `base_lr`, explore opens at `base_lr` (continuous), explore ends
at `base_lr`, exploit opens at `base_lr` and the clamp keeps it non-negative through to 0 at the boundary —
only the slope changes at each seam, never the value, so there is no shock to the dynamics or the momentum
velocity. The three-phase math I will call the explore-exploit (Knee) schedule, built on the wide-minima
density hypothesis; the scaffold fill expresses its `get_lr` as — warmup
`peak_lr·step/warmup`, explore constant `peak_lr`, exploit `max(0, peak_lr − peak_lr/decay·(step − warmup −
explore))` — in this task's per-epoch contract, mapping the 0-indexed harness epoch to the canonical
1-indexed `global_step = epoch + 1` so the branch boundaries match and the final epoch lands exactly at 0,
with `peak_lr = base_lr` and explore = 50%. (The full schedule body is in the answer.)

So the delta from warmup+cosine is one targeted insertion at the place the ladder never touched — the
middle. Keep the warmup that fixed the start and keep an anneal-to-zero finish, but between them replace the
*immediately-descending cosine body* with a *sustained flat explore plateau at `base_lr`* for half the
budget, then a linear cool to zero. The bar this has to clear is warmup+cosine's measured {92.71, 72.43,
94.83}, and the falsifiable expectations follow directly from the wide-minimum mechanism. If the density
hypothesis is right, the explore plateau should land the nets in *wider* minima than the early-cooling
cosine did, so I expect the gains to show up where there is the most generalization headroom and the most
overparameterization to exploit: ResNet-56 / CIFAR-100 should clear 72.43 — the deep, hard, 100-class net
has the richest wide-minimum structure to find, and it is where holding the rate high longest should help
most. ResNet-20 / CIFAR-10 should recover the fraction warmup+cosine lost (back toward or above 92.71,
since the explore plateau gives the easy net a longer productive high-rate phase than the cosine's instant
descent), and FashionMNIST should hold near 94.83 or edge above it. The mechanism is also falsifiable in
the *other* direction: if the explore phase is too long for a 200-epoch budget, the exploit phase (95
epochs) will be too short to settle and the training loss — and test accuracy — will *drop* below
warmup+cosine, most visibly on the hardest net; that would say 50% is too much explore for this budget and
the fraction should come down. There is no leaderboard row for this schedule to confirm it against; what
the trajectory ends on is the construction whose explicit explore phase is the one thing every prior rung —
tuned for speed, cooling from the start — structurally lacked.

The causal chain in one breath: warmup+cosine won by fixing the start (the five-epoch ramp took CIFAR-100
from 71.07 to 72.43) and the finish (cosine's anneal to zero), but it — like every prior rung — *cools the
rate almost immediately*, holding the peak for a single instant → reframe the rate as the *temperature* of
the search via the noise scale `g ≈ eps·N/(B(1−m))`: a high rate ejects SGD from narrow basins and settles
only in wide ones, and generalization is set by *which* minimum is found → under the wide-minima-density
hypothesis (wide minima are rare, consistent with large-batch/low-noise training generalizing worse), early
cooling lands in an abundant narrow basin, so the missing ingredient is a sustained high-temperature
*explore* phase the ladder never tried → hold the rate flat at `base_lr` for ~50% of the budget (constant,
since any decay there is premature cooling), then *linearly* cool to 0 over the rest (the exploit settle),
keeping the warmup the ladder proved the deep net needs in front, and needing no momentum/weight-decay
change because the regularization comes from holding the *existing* peak longer (not a higher peak, which
the harness neutered for one-cycle) → land it as the three-phase Knee `get_lr` (warmup ramp, explore
plateau, linear exploit, clamped at 0, tiling the budget exactly) in the same per-epoch edit surface →
expect it to clear warmup+cosine's {92.71, 72.43, 94.83} most on the deep CIFAR-100 net (richest
wide-minimum structure) while recovering CIFAR-10 and holding FashionMNIST, with the falsifiable failure
being an over-long explore phase starving the 95-epoch settle.
