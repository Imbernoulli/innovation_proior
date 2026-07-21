Warmup + cosine is the strongest schedule so far, and reading its numbers against the other two tells me
both why it won and where it is still leaving accuracy behind. It landed {92.71, 72.43, 94.83}. Lay the
three side by side per setting. On ResNet-56 / CIFAR-100 the sequence went 71.57 → 71.07 → 72.43: the short
five-epoch ramp recovered the deep net from cosine's regression all the way past one-cycle, `+1.36` over
cosine and `+0.86` over one-cycle — the single biggest metric move anywhere, exactly the init-protection I
predicted the deep net needed. On MobileNetV2 / FashionMNIST it went 93.93 → 94.70 → 94.83, a steady climb.
On ResNet-20 / CIFAR-10 it went 92.31 → 93.03 → 92.71 — warmup+cosine slipped `0.32` back from cosine's
peak, the small tax I predicted a five-epoch sub-`base_lr` ramp would levy on the easy net that did not need
it. The means track the story: 85.94 → 86.27 → 86.66, each step up, the last despite the CIFAR-10 give-back
because the CIFAR-100 recovery dwarfs it. So the warmup hypothesis was right and the schedule strictly
improved the hard net. But step back and look at what warmup+cosine, cosine, and one-cycle all share,
because that shared property is the next thing to attack: every one of them starts cooling the rate almost
immediately. The pure cosine sits at its peak for exactly one epoch (epoch 0) and descends after;
warmup+cosine holds `base_lr` for two epochs (the seam at epochs 4 and 5) and then descends; one-cycle held
the peak only momentarily. None of them stays at the high rate — one or two epochs out of two hundred —
they all treat the peak as a point to pass through on the way down. The question I have not asked is whether
passing through so quickly is itself the limitation: whether these schedules are all tuned for getting to a
minimum fast when what actually decides test accuracy is which minimum they get to.

That reframing is the move, and it is a different kind of objection from any I have raised. In these nets,
generalization is governed by the geometry of the minimum SGD lands in: wide, flat minima generalize well,
sharp narrow ones do not. And the learning rate is the lever that controls which kind SGD finds, through
the noise it injects. To first order the stochastic-gradient noise scale is `g ≈ eps·N/(B·(1−m))`, rising
linearly with the rate. Put the fixed numbers on the other factors: `N = 50000` training images, `B = 128`
batch, and `m = 0.9` gives `1/(1−m) = 10`, so `g ≈ eps · 50000 · 10 / 128 ≈ 3900 · eps` — every term
except `eps` is nailed down, so on this problem the noise scale is the learning rate up to a fixed
multiplier. Higher rate, noisier descent, and noise is what ejects SGD from a narrow basin: a sharp minimum
is a small target with steep walls a noisy step bounces out of, while a wide minimum is a large flat region
the same noise cannot escape. So a high learning rate is a wide-minimum-seeking device — it keeps the
search hot enough to be thrown out of narrow basins and settle only where the basin is wide. The rate is
the temperature of the search, and "decay the rate" is "cool the search." Now the limitation of every prior
schedule is visible in one sentence: they all cool almost from the start, so they barely search at high
temperature before they begin settling — they find a minimum fast but are biased toward whichever basin is
nearby when cooling starts. One-cycle's problem was amputated machinery; cosine's was a too-high floor and
missing warmup; warmup+cosine fixed both ends. This objection is about the middle, which nothing has
touched — the entire high-temperature search phase is one or two epochs long.

Which basin is nearby when cooling starts depends on the density of wide versus narrow minima, and the
hypothesis I design from is that wide minima are rarer — the landscape is full of narrow basins and only
sparsely dotted with wide ones. If so, a search that cools quickly almost certainly falls into an abundant
narrow basin long before it stumbles onto a rare wide one. The only way to find a rare wide basin is to stay
hot for a long time — keep the rate high long enough that the rare wide basins get sampled — and only then
cool to settle into the one you found. I can check the hypothesis against something the field knows:
large-batch training generalizes worse. Under `g ≈ eps·N/(B(1−m))`, holding the rate fixed and raising `B`
lowers `g` — a bigger batch is a colder search — which by this argument settles into a narrow basin sooner,
and large-batch training is indeed known to find sharper minima. The same formula says I could recover the
lost noise by scaling `eps` with `B`, which is exactly the linear scaling rule — further evidence the noise
scale is the governing quantity. So I take "wide minima are lower-density, so you must explore long to find
them" as the working hypothesis.

This is what warmup+cosine's win sharpens rather than answers. The warmup fixed the start (init curvature);
the cosine fixed the finish (smooth settle). Neither addressed the middle — and the middle is where the
wide-minimum search has to happen. The cosine body begins descending the instant warmup ends, so the
high-temperature explore phase is essentially zero epochs long. If wide minima are rare, warmup+cosine is
cooling before it has searched, and the accuracy it leaves on the table is the gap between the narrow basin
it settles into quickly and the wide basin it would have found if it had stayed hot. The 72.43 on CIFAR-100
is the best so far, but a deep, overparameterized net on a hard 100-class problem is precisely the setting
with the richest wide-minimum structure to find, and the most to lose by cooling early. So the next move is
forced: insert a deliberate, sustained explore phase — hold the rate flat at the peak for a long stretch —
before cooling. Explore, then exploit.

Design the two phases. The explore phase I want constant at the peak — not a ramp, not a slow decay, just
held at `base_lr` — because the point is to keep the temperature maximal and steady so the search keeps
sampling basins for the full explore budget; any decay during explore is premature cooling. So explore is a
flat plateau at `base_lr`. The peak is `base_lr` itself, the same high value the winning schedules already
use, which the whole sequence has confirmed is safe after the five-epoch warmup. Push the peak higher? The
arithmetic from the first attempt says no: the frozen `m = 0.9` pins the effective-step amplifier at `10*`
with no way to drop it at the peak, so a peak above `base_lr` runs the unstable configuration one-cycle
warned about, with none of the momentum/weight-decay compensations. And this schedule does not want a
bigger peak; it wants to hold the peak I already trust, longer. That is the crucial distinction from
one-cycle: one-cycle tried to get more regularization from a higher peak (which needs momentum cycling and
weight-decay rebalancing to survive, both frozen away); explore-exploit gets more wide-minimum search from
a longer time at the same peak — a change in the duration of the high-temperature phase, not its
temperature — which needs no lever beyond the per-epoch float. And the temperature at `base_lr` is already
ample: `m = 0.9` multiplies the noise by `10`, so the plateau's `g ≈ 3900·base_lr` is high even without a
bigger peak — the frozen momentum, for once, works for this schedule. Duration is the free lever,
temperature is already ample, so I spend the whole design on duration.

Now the exploit phase. Once the plateau has parked the iterate in or near a wide basin, cool the search to
settle into it — drive the rate down so `g` shrinks and SGD descends to the bottom of the wide region. What
cooling shape? Here I deliberately do not reach for the cosine, even though it won the last two runs, for a
specific reason. The cosine's signature is a flat top — zero slope at the start of its descent — which made
it a good whole-run curve, lingering high before falling. But here I have already allocated the
lingering-high to an explicit explore plateau; if I then cool with a cosine, its flat top keeps the rate
near `base_lr` for the first many exploit epochs, smuggling more plateau in through the back door and
stealing budget from the actual settling. I have set explore's length on purpose and want exploit to start
cooling immediately. A plain linear decay from `base_lr` to `0` does exactly that: constant negative slope
from the first exploit epoch, reaching exactly `0` at the budget boundary so the final steps vanish and the
basin fully settles. The slope is `−base_lr/decay_epochs`, clamped at `0`. And it cools to `0`, not a
floor: the stationary-jitter argument that took the cosine to zero applies unchanged, and there is even less
reason to leave a floor now that the plateau, not the tail, carries the regularization. The tail's only job
is to be quiet, and `0` is quietest.

The split between explore and exploit is the one real knob, and the density hypothesis sets it. Too little
explore and I cool before finding a wide basin — back to warmup+cosine's immediately-descending failure.
Too much and I leave too few epochs to settle, ending near a wide basin but never descending into it, so
best test accuracy comes in below warmup+cosine because the model never converged. So there is an interior
optimum, and the hypothesis says it should be large. I can see the starvation coming in the settle slope:
with explore at 50% the exploit phase has 95 epochs to bring `base_lr` to `0`, a gentle `−0.00105` per
epoch; at 60% it shrinks to 75 epochs at `−0.00133`; at 75% only 45 epochs at `−0.00222`, twice as steep. A
steeper settle means the rate drops fast while the iterate is still far from the wide basin's floor, so it
freezes partway down. Half the budget keeps the settle gentle (slope near `0.001`, the order of the
cosine's own late-epoch changes) while still giving the search a hundred epochs at full temperature. So
explore = 50% of `total_epochs`, exploit the rest.

Compose it with the warmup the last run proved I need, and land it. I keep the five-epoch linear ramp from
`base_lr/5` to `base_lr` in front — pure init hygiene, complementary to explore-exploit, and the deep net's
`71.07 → 72.43` recovery showed it is needed. Then three segments tile the 200-epoch budget: warmup (epochs
0–4, ramp to `base_lr`), explore (`int(0.5·200) = 100` epochs flat at `base_lr`), exploit (the remaining
`95` epochs, linear decay from `base_lr` to `0`, slope `−base_lr/95`). Mapping the 0-indexed `epoch` to a
1-indexed `global_step = epoch + 1` so the phase boundaries line up: warmup is `global_step <= 5` (epoch 0 →
`0.02`, epoch 4 → `0.10`); explore is `global_step <= 105` (epochs 5–104 all at `base_lr`); exploit is the
rest (epoch 105 → `0.09895`, epoch 152 → `≈0.0495`, epoch 198 → `0.00105`), and epoch 199 hits the
`global_step >= total_epochs` guard and returns exactly `0.0`. Only the slope changes at each seam, never
the value. One number captures how different this is: `base_lr` is held at full value from epoch 4 through
epoch 104 — 101 consecutive epochs at the peak — against the pure cosine's one and warmup+cosine's two. And
it is genuinely hotter, not a rearrangement: summing the per-epoch rate gives `15.0` in total travel against
the cosine's `10.05`, about `49%` more stepping over the same 200 epochs — where cosine and one-cycle spent
nearly identical total travel and differed only in where, the plateau adds real heat, concentrated in the
sustained search the wide-minimum hypothesis asks for. I will call this the explore-exploit (Knee) schedule.

So the delta from warmup+cosine is one insertion at the place nothing has touched — the middle. Keep the
warmup that fixed the start and the anneal-to-zero finish, but between them replace the immediately-
descending cosine body with a sustained flat explore plateau at `base_lr` for half the budget, then a
linear cool to zero. The bar is warmup+cosine's {92.71, 72.43, 94.83}. If the density hypothesis is right,
the plateau lands the nets in wider minima than the early-cooling cosine did, so the gains should show where
there is the most generalization headroom: ResNet-56 / CIFAR-100 should clear 72.43 — the deep, hard,
100-class net has the richest wide-minimum structure and is where holding the rate high longest helps most.
ResNet-20 / CIFAR-10 should recover the `0.32` it gave back, since the plateau hands the easy net 101 epochs
at full rate against the cosine's one; the easy net has the least wide-minimum headroom, so I expect a small
recovery, not a large one, though it could edge past cosine's 93.03. FashionMNIST should hold near 94.83 or
edge above. The mechanism is falsifiable the other way too: if the explore phase is too long for a
200-epoch budget, the 95-epoch exploit is too short to settle and best test accuracy drops below
warmup+cosine, most visibly on the hardest net where descending into a wide basin takes the most settling —
that would say 50% is too much explore for this budget. There is no measurement after this one; what the
search ends on is the construction whose explicit explore phase is the one thing every prior schedule, tuned
for speed and cooling from the start, structurally lacked.
