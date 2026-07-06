Warmup + cosine is the strongest rung, and reading its numbers against the other two tells me both why it
won and where it is still leaving accuracy behind. It landed {92.71, 72.43, 94.83}. Lay the three rungs
side by side per setting. On ResNet-56 / CIFAR-100 the ladder went 71.57 → 71.07 → 72.43: the short five-
epoch ramp recovered the deep net from cosine's regression all the way past one-cycle, `+1.36` over cosine
and `+0.86` over one-cycle — the single biggest metric move anywhere on the ladder, and exactly the init-
protection I predicted the deep net needed. On MobileNetV2 / FashionMNIST it went 93.93 → 94.70 → 94.83, a
steady climb, the warmup adding a final `+0.13`. On ResNet-20 / CIFAR-10 it went 92.31 → 93.03 → 92.71 —
warmup+cosine slipped `0.32` back from cosine's peak, the small tax I predicted a five-epoch sub-`base_lr`
ramp would levy on the easy net that did not need it. The means track the story: 85.94 → 86.27 → 86.66,
each rung up, the last one up despite the CIFAR-10 give-back because the CIFAR-100 recovery dwarfs it. So
the warmup hypothesis was right and the schedule strictly improved the hard net. But step back and look at
what warmup+cosine — and cosine, and one-cycle before it — all have in common, because that shared property
is the next thing to attack. Every rung so far starts *cooling the rate almost immediately*. I can make
that precise from the schedules themselves: the pure cosine sits at its peak `base_lr` for exactly one
epoch (epoch 0) and descends every epoch after; warmup+cosine holds `base_lr` for two epochs (the seam at
epochs 4 and 5, which I traced last rung) and then the cosine takes over and descends; one-cycle held
`base_lr` only momentarily at the top of its 30% ramp. None of them *stays* at the high rate — one or two
epochs out of two hundred — they all treat the peak as a point to pass through on the way down. The
question I have not asked yet is whether passing through the peak so quickly is itself the limitation:
whether these schedules are all tuned for *getting to a minimum fast* when what actually decides test
accuracy is *which* minimum they get to.

That reframing is the move, and it is a different kind of objection from any the ladder has raised. In
these nets, generalization is governed by the *geometry* of the minimum SGD lands in: wide, flat minima
generalize well, sharp, narrow ones do not. And the learning rate is the lever that controls which kind
SGD finds, through the noise it injects. To first order the stochastic-gradient noise scale is
`g ≈ eps·N/(B·(1−m))` — it rises linearly with the rate `eps`. Put the fixed numbers on the other factors:
`N = 50000` training images, `B = 128` batch, and the frozen `m = 0.9` gives `1/(1−m) = 10`, so
`g ≈ eps · 50000 · 10 / 128 ≈ 3900 · eps` — every term except `eps` is nailed down by the harness, so on
this problem the noise scale *is* the learning rate up to a fixed multiplier. A higher rate means a noisier
descent, and noise is exactly what ejects SGD from a narrow basin: a sharp minimum is a small target with
steep walls that a noisy step bounces out of, while a wide minimum is a large flat region the same noise
cannot escape. So a high learning rate is a *wide-minimum-seeking* device — it keeps the search hot enough
to be thrown out of narrow basins and to settle only where the basin is wide. The learning rate is the
*temperature* of the search, and "decay the rate" is really "cool the search." Now the limitation of every
prior rung is visible in one sentence: they all cool almost from the start, so they barely search at high
temperature before they begin settling — they find *a* minimum fast, but they are biased toward whichever
basin is nearby when they start cooling. One-cycle's problem was amputated machinery; cosine's was a too-
high floor and a missing warmup; warmup+cosine fixed both ends. This objection is about the *middle*, which
no rung has touched — the entire high-temperature search phase is one or two epochs long.

Which basin is nearby when cooling starts? That depends on the *density* of wide versus narrow minima in
the landscape, and the hypothesis I will design from is that wide minima are *rarer* — the landscape is
full of narrow basins and only sparsely dotted with wide ones. If that is true, a search that cools quickly
almost certainly falls into one of the abundant narrow basins long before it stumbles onto a rare wide one.
The only way to find a rare wide basin is to *stay hot for a long time* — keep the rate high long enough
that the rare wide basins get sampled — and only *then* cool to settle into the one you found. I can sanity-
check the hypothesis against something the field knows directly, and the noise-scale formula makes the
check quantitative: large-batch training generalizes worse. Under `g ≈ eps·N/(B(1−m))`, holding the rate
fixed and raising `B` *lowers* `g` — a bigger batch is a colder search — which by this argument settles
into a narrow basin sooner, and large-batch training is indeed known to find sharper minima and generalize
worse. The same formula says I could recover the lost noise by scaling `eps` up with `B`, and the linear
scaling rule is exactly that move — which is further evidence the noise scale, not the batch or the rate
alone, is the governing quantity. The pieces line up, so I will take "wide minima are lower-density, so you
must explore long to find them" as the working hypothesis.

This is precisely the diagnosis that warmup+cosine's win sharpens rather than answers. The warmup fixed the
*start* (init curvature); the cosine fixed the *finish* (smooth settle to near-zero). But neither addressed
the *middle* — and the middle is where the wide-minimum search has to happen. The cosine body begins
descending the instant warmup ends, so the high-temperature explore phase is essentially zero epochs long.
If wide minima are rare, warmup+cosine is cooling before it has searched, and the accuracy it leaves on the
table is exactly the gap between the narrow basin it settles into quickly and the wide basin it would have
found if it had stayed hot. The 72.43 on CIFAR-100 is the best the ladder has reached, but a deep,
overparameterized net on a hard 100-class problem is precisely the setting with the richest wide-minimum
structure to find — the most parameters, the most basins, the most generalization headroom — and the most
to lose by cooling before finding it. So the next move is forced by the hypothesis: insert a deliberate,
sustained *explore* phase — hold the rate flat at the peak for a long stretch — *before* cooling. Explore,
then exploit.

Design the two phases. The explore phase I want *constant at the peak* — not a ramp, not a slow cosine
decay, just held at `base_lr` — because the entire point is to keep the temperature maximal and steady so
the search keeps sampling basins for the full explore budget; any decay during explore is premature
cooling, the disease I am curing. So explore is a flat plateau at `base_lr`. The peak is `base_lr` itself,
the same rate the winning baselines already use as their high value — chosen to be as high as these nets
tolerate, which the whole ladder has confirmed is safe *after* the five-epoch warmup. Should I push the
peak higher to make the search hotter still? I already have the arithmetic that says no: back at the first
rung I found that the frozen `m = 0.9` pins the effective-step amplifier at `10*` with no way to drop it at
the peak, so a peak above `base_lr` runs the unstable configuration one-cycle warned about, with none of
the momentum/weight-decay compensations the harness froze away. And this schedule does not *want* a bigger
peak: it wants to *hold* the peak I already trust, longer. That is the crucial distinction from one-cycle.
One-cycle tried to get more regularization from a *higher* peak (and the harness blocked it, because a
higher peak needs momentum cycling and weight-decay rebalancing to survive); the explore-exploit schedule
gets more wide-minimum search from a *longer* time at the *same* peak — a change in the *duration* of the
high-temperature phase, not its temperature — which the harness allows freely, since holding `base_lr` for
a hundred epochs needs no lever beyond the per-epoch float I already control. And the temperature at
`base_lr` is not modest to begin with: the frozen `m = 0.9` already multiplies the noise by `1/(1−m) = 10`,
so the plateau's effective search temperature `g ≈ 3900·base_lr` is high even without a bigger peak — the
momentum the harness froze is, for once, working *for* this schedule, keeping the plateau genuinely hot.
Duration is the free lever and temperature is already ample, so I spend the whole rung on duration.

Now the exploit phase. Once the plateau has parked the iterate in or near a wide basin, I cool the search
to settle into it — drive the rate down so the noise `g` shrinks and SGD descends to the bottom of the wide
region and stays. What cooling shape? Here I deliberately do *not* reach for the cosine, even though the
cosine won the last two rungs, and the reason is specific rather than contrarian. The cosine's signature
property is a *flat top* — zero slope at the start of its descent — which is exactly what made it a good
*whole-run* curve, lingering high before falling. But on this schedule I have *already* allocated the
lingering-high to an explicit explore plateau; if I then cool with a cosine, its flat top would keep the
rate near `base_lr` for the first many epochs of the exploit phase, smuggling *more* plateau in through the
back door and stealing budget from the actual settling. I do not want the cooling curve to secretly extend
explore — I have set explore's length on purpose and I want exploit to *start cooling immediately*. A plain
*linear* decay from `base_lr` to `0` does exactly that: constant negative slope from the first exploit
epoch, the strongest simple monotone budgeted descent, one obvious slope, reaching exactly `0` at the
budget boundary so the final steps vanish and the basin fully settles. The slope is set so the line hits
`0` at the end, `−base_lr/decay_epochs`, and the rate is `base_lr + slope·(epochs into exploit)`, clamped
at `0` so it never goes negative. And it cools to `0`, not to a small floor: the same stationary-jitter
argument that took the cosine to zero two rungs ago applies unchanged here — the residual variance scales
with the rate, so only `0` fully quiets the finish, and there is even less reason to leave a floor now that
the plateau, not the tail, is carrying the regularization. The tail's only job is to be quiet, and `0` is
the quietest. No milestones, no shape parameter — the explore plateau, not the exploit curve, is where the
design lives, so I keep the curve as plain as possible.

The split between explore and exploit is the one real knob, and the density hypothesis tells me how to set
it. Too little explore and I cool before finding a wide basin — straight back to warmup+cosine's failure
mode, the immediately-descending body. Too much explore and I leave too few epochs to settle, so I end
*near* a wide basin but never descend into it and the training loss stays high — I would have found the
wide region but never dropped to its floor, and best test accuracy would come in *below* warmup+cosine
because the model never converged. So there is an interior optimum, and the hypothesis says it should be
*large*, because rare wide basins need a long search. The principled way to pick it is to sweep a few
explore fractions and watch both the accuracy *and* the sharpness of the minimum reached; the expectation
is that more explore widens the minimum and raises accuracy until too little exploit budget remains, then
falls off as the settle starves. I can see the starvation coming in the settle slope: with explore at 50%
the exploit phase has 95 epochs to bring `base_lr` to `0`, a gentle `−0.00105` per epoch; push explore to
60% and exploit shrinks to 75 epochs at `−0.00133`; push it to 75% and exploit has only 45 epochs at
`−0.00222`, twice as steep a cool as the 50% split. A steeper settle means the rate is dropping fast while
the iterate is still far from the wide basin's floor, so it freezes partway down — the "end near a wide
basin but never descend into it" failure, made quantitative. Half the budget keeps the settle gentle enough
(the slope stays near `0.001`, the same order as the cosine's own late-epoch changes) while still giving
the search a hundred epochs at full temperature. The value that balances "enough search to find a rare wide
basin" against "enough settling to descend into it" lands at half the budget — so explore = 50% of
`total_epochs`, exploit = the rest. That single fraction is the whole new design surface.

Now compose it with the warmup the last rung proved I need, and land it in the edit surface. I keep the
five-epoch linear ramp from `base_lr/5` to `base_lr` in front — it is pure init hygiene, complementary to
explore-exploit, and the ladder already showed the deep net needs it (it was the entire reason CIFAR-100
went from 71.07 to 72.43). Then the three segments tile the 200-epoch budget: warmup (epochs 0–4, linear
ramp to `base_lr`), explore (`int(0.5·200) = 100` epochs held flat at `base_lr`), exploit (the remaining
`200 − 5 − 100 = 95` epochs, linear decay from `base_lr` to `0` with slope `−base_lr/95`). Let me trace the
code to be sure the phase boundaries and the endpoint land where I claim, since three stitched branches on
an index is where an off-by-one hides. I map the 0-indexed harness `epoch` to the canonical 1-indexed
`global_step = epoch + 1` so the branch boundaries match. Warmup is `global_step <= 5`: epoch 0 gives
`base_lr·1/5 = 0.02`, epoch 4 gives `base_lr·5/5 = 0.10`. Explore is `global_step <= 105`: epoch 5
(`global_step 6`) gives `base_lr`, epoch 104 (`global_step 105`) gives `base_lr`. Exploit is the rest: epoch
105 (`global_step 106`) gives `base_lr − base_lr/95·(106−105) = 0.09895`, epoch 152 gives `≈ 0.0495` (about
half, the midpoint of the decay), epoch 198 (`global_step 199`) gives `base_lr − base_lr/95·94 = 0.00105`,
and epoch 199 (`global_step 200`) hits the `global_step >= total_epochs` guard and returns exactly `0.0`.
So the seams are clean: warmup ends at `base_lr`, explore opens and closes at `base_lr`, exploit opens at
`base_lr` and the clamp carries it non-negative to `0` at the boundary — only the slope changes at each
seam, never the value, so there is no shock to the dynamics or the `10*`-amplified momentum velocity. And
one number captures how different this is from every prior rung: `base_lr` is held at exactly its full
value from epoch 4 (top of the ramp) through epoch 104 (end of explore) — 101 consecutive epochs at the
peak — against the pure cosine's one epoch and warmup+cosine's two. That hundred-fold-longer high-
temperature phase, needing no momentum or weight-decay change, is the entire content of the rung. And it
is a genuinely *hotter* run, not a rearrangement: summing the whole schedule's per-epoch rate gives `15.0`
in total travel against the cosine's `10.05` — about `49%` more accumulated stepping over the same 200
epochs. That is a sharp contrast with the earlier rungs, where cosine and one-cycle spent nearly identical
total travel and differed only in *where* they spent it; here the plateau genuinely adds heat, half again
as much total stepping, concentrated in the sustained high-temperature search the wide-minimum hypothesis
asks for. The
three-phase math I will call the explore-exploit (Knee) schedule, built on the wide-minima density
hypothesis.

So the delta from warmup+cosine is one targeted insertion at the place the ladder never touched — the
middle. Keep the warmup that fixed the start and keep an anneal-to-zero finish, but between them replace the
*immediately-descending cosine body* with a *sustained flat explore plateau at `base_lr`* for half the
budget, then a linear cool to zero. The bar this has to clear is warmup+cosine's measured {92.71, 72.43,
94.83}, and the falsifiable expectations follow directly from the wide-minimum mechanism. If the density
hypothesis is right, the explore plateau should land the nets in *wider* minima than the early-cooling
cosine did, so I expect the gains to show up where there is the most generalization headroom and the most
overparameterization to exploit: ResNet-56 / CIFAR-100 should clear 72.43 — the deep, hard, 100-class net
has the richest wide-minimum structure to find, and it is where holding the rate high longest should help
most. ResNet-20 / CIFAR-10 should recover the `0.32` fraction warmup+cosine gave back (toward or above
92.71), since the explore plateau gives the easy net a longer productive high-rate phase than the cosine's
instant descent — 101 epochs at `base_lr` versus one. The give-back came from spending five early epochs
below `base_lr`; the plateau more than repays that by handing the easy net a hundred epochs of full-rate
stepping it never had, so if anything CIFAR-10 could edge past cosine's 93.03 rather than merely returning
to it — though the easy net has the least wide-minimum headroom, so I expect a small recovery, not a large
one. FashionMNIST should hold near 94.83 or edge above it.
The mechanism is also falsifiable in the *other* direction, and I want that failure mode explicit: if the
explore phase is too long for a 200-epoch budget, the exploit phase (95 epochs) will be too short to settle
and the training loss — and best test accuracy — will *drop* below warmup+cosine, most visibly on the
hardest net, where descending into a wide basin takes the most settling; that would say 50% is too much
explore for this budget and the fraction should come down. There is no leaderboard row for this schedule to
confirm it against — it is the finale, and no measurement follows it in this trajectory. What the ladder
ends on is the construction whose explicit explore phase is the one thing every prior rung — tuned for
speed, cooling from the start — structurally lacked.

The causal chain in one breath: warmup+cosine won by fixing the start (the five-epoch ramp took CIFAR-100
from 71.07 to 72.43, `+1.36`, the biggest move on the ladder) and the finish (cosine's anneal to zero), but
it — like every prior rung — *cools the rate almost immediately*, holding the peak for one or two epochs out
of two hundred → reframe the rate as the *temperature* of the search via the noise scale
`g ≈ eps·N/(B(1−m))`, which on this frozen harness (`N=50000, B=128, m=0.9`) is just `≈3900·eps`, so a high
rate ejects SGD from narrow basins and settles only in wide ones, and generalization is set by *which*
minimum is found → under the wide-minima-density hypothesis (wide minima are rare, consistent with large-
batch/low-noise training generalizing worse, which the same formula predicts) early cooling lands in an
abundant narrow basin, so the missing ingredient is a sustained high-temperature *explore* phase the ladder
never tried → hold the rate flat at `base_lr` for 50% of the budget (constant, since any decay there is
premature cooling), then *linearly* cool to `0` over the rest (linear, not cosine, because a cosine's flat
top would smuggle more plateau into the settle), keeping the warmup the ladder proved the deep net needs in
front, and needing no momentum/weight-decay change because the regularization comes from holding the
*existing* peak longer (101 epochs at `base_lr` versus one), not a higher peak (which the frozen `m=0.9`
would make unstable, as one-cycle showed) → land it as the three-phase Knee `get_lr` (warmup ramp, explore
plateau, linear exploit, clamped at 0, tiling the budget exactly, traced clean at every seam) in the same
per-epoch edit surface → expect it to clear warmup+cosine's {92.71, 72.43, 94.83} most on the deep CIFAR-100
net (richest wide-minimum structure) while recovering CIFAR-10 and holding FashionMNIST, with the
falsifiable failure being an over-long explore phase starving the 95-epoch settle.
