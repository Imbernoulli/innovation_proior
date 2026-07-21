The cosine run did two things at once and the split is the whole story. On ResNet-20 / CIFAR-10 it jumped
to 93.03, a clean lift over one-cycle's 92.31; on MobileNetV2 / FashionMNIST it rose to 94.70 from 93.93.
Both confirm the last diagnosis: dropping the sixty-epoch warmup and annealing to zero reclaimed the
productive high-rate epochs and removed the floor jitter, and the shallow net and the mobile net rewarded
it immediately. But the deltas' pattern is what I have to explain. Against one-cycle, cosine gained `+0.72`
on CIFAR-10 and `+0.77` on FashionMNIST — nearly the same lift on the two easy settings — but on
ResNet-56 / CIFAR-100 it went the other way, `71.07` against `71.57`, a loss of `0.50`. Two settings up by
about three-quarters of a point and one down by half; the mean crept from `85.94` to `86.27`, a `+0.33`
that hides a regression. That negative delta is not noise, because it is exactly the sign I told myself to
watch for. When I stripped the warmup I wrote the failure down in advance: if the bare full-rate start
re-exposed an init instability on the deep net, CIFAR-100 would fail to clear 71.57 despite the cleaner
finish. It did precisely that. The prediction fired, so I take its mechanism seriously.

Read the three settings together and the pattern is coherent under one cause. The two nets that improved are
the shallow ResNet-20 and the inverted-residual MobileNetV2; the one that regressed is the 56-layer ResNet.
Depth is the axis that separates them. ResNet-56 has `2.8×` more composed nonlinear maps between input and
loss than ResNet-20, and curvature at initialization compounds with depth: the loss Hessian's top
eigenvalue is built from products of per-layer Jacobians, so a deeper stack has a sharper, higher-`lambda_max`
region right after Kaiming init before BatchNorm's statistics settle. From the stability arithmetic,
`base_lr = 0.1` diverges along any direction with curvature past `20`; the deeper the net, the more likely
`lambda_max` at init sits above that line. So `0.1` from epoch 0 is most likely above the ceiling on
ResNet-56 specifically, where the first few full-rate steps overshoot, knock the net into a worse basin
before it settles, and the clean cosine anneal that follows cannot fully recover — the deepest net loses the
accuracy its better finish should have bought. The shallow ResNet-20 and the mobile net have gentler
initial curvature, tolerate `0.1` from the start, and keep the gains. So the deep net is telling me warmup
was doing real curvature-protection work — just not sixty epochs of it.

I have to be careful, because the last run changed two things relative to one-cycle: it removed the warmup
and it dropped the floor from `base_lr/25` to `0`. Before I build a whole schedule on "restore the warmup,"
rule out the floor. Could annealing to `0` have hurt the deep net? That runs against the mechanism: a lower
floor shrinks the stationary jitter monotonically, so it can only make the finish cleaner, and it
demonstrably helped the two easy nets. There is no mechanism by which a smaller final rate damages a deep
net specifically. The regression, by contrast, is concentrated where the start is hardest — the deepest,
highest-`lambda_max` net — the signature of an init overshoot, not a finish problem; and the two easy nets,
which got the same floor change, did not regress. So the floor is exonerated and the warmup removal
implicated. One loose end is MobileNetV2, which is not shallow yet improved rather than regressing. I read
that as its setting: inverted-residual blocks with linear bottlenecks on grayscale FashionMNIST are an
easier optimization than a 56-layer stack on 100-way CIFAR, so its init curvature stays under the ceiling
even without warmup. Depth alone is not the trigger; depth times task difficulty is, and ResNet-56 /
CIFAR-100 is the only setting that maxes both.

That points at a precise fix. The mistake the first time was not having a warmup; it was making it sixty
epochs and forcing the rate below `base_lr` for nearly a third of the run, which cost the easy nets half
their early step-distance. The mistake the second time was throwing the warmup out entirely and re-exposing
the deep net's init. The two failures bracket the answer: one had too much warmup, the other too little, so
the right move is a short warmup that protects the sharp initial region for a handful of epochs, then hands
off to the same cosine anneal that already won on the easy nets. I split the problem: a start-of-training
problem (survive the sharp init) and a body problem (anneal smoothly to settle), handled separately, then
stitched.

Take the start, the one cosine got wrong on CIFAR-100. The disease is that `base_lr` is above
`2/lambda_max` while the deep net is in its sharp initial region. The cure is to not use `base_lr` at the
very start — use something smaller while curvature is high, then raise it. Why does small-then-raise buy
anything rather than just delaying the problem? Because the sharp initial region is transient: even a few
small, safe steps move the weights into a flatter region where `lambda_max` has come down and
`2/lambda_max` has risen, so the full `base_lr` becomes admissible where it was not at epoch 0. Starting
small is to survive long enough for the curvature to relax under its own descent, then step up to the rate I
want for the body. So at the very start the curve should rise, from a safe small value to `base_lr`, over a
brief opening — the same rise-then-run shape one-cycle had, compressed from sixty epochs to a handful and
with no aggressive peak on the far side.

How to climb? The laziest version is a constant low prefix — hold `base_lr/5 = 0.02` for a few epochs, then
snap to `base_lr`. But the snap recreates the disease: after the prefix I have no guarantee the curvature
has come down enough that `base_lr` is now below `2/lambda_max`, and if it has not, the five-fold jump from
`0.02` to `0.1` leaps straight back over the ceiling, the same loss spike merely postponed. The
discontinuity is the problem, so climb gradually — increase the rate by a constant increment each epoch. As
`eta` rises the weights keep moving into flatter regions and `2/lambda_max` keeps rising; a linear ramp
keeps `eta` below that rising ceiling, tracking the relaxation rather than betting it has finished. So the
opening is a linear ramp: the k-th warmup epoch is `k*base_lr/warmup`, reaching `base_lr` at the last
warmup epoch. I write it as `base_lr*(epoch+1)/warmup` so the 0-indexed first epoch already does a little
work at `base_lr/warmup` rather than sitting at zero.

Linear, not cosine — one-cycle's up-leg was a cosine climb and I am deliberately not reusing it. Over a
five-epoch window the shape barely exists: a cosine warmup would put the samples at about
`0.028, 0.048, 0.072, 0.092, 0.100` against the linear `0.020, 0.040, 0.060, 0.080, 0.100` — a maximum
difference of `0.012`, one epoch's worth of curvature no run will resolve. A cosine ramp buys a smoother
slope at the bottom of the climb for a shape parameter and a second formula, a difference smaller than the
ramp's own step size. The linear ramp is standard, carries no free knob, and over five epochs is
indistinguishable from any smooth alternative, so I spend my attention on the length instead.

How long is `warmup`? The two failed runs pin it between them. Sixty epochs was far too long — 30% of the
run, half the early step-distance thrown away. Zero was too short for the deep net. I want the smallest
warmup that still relaxes the deep net's initial curvature: long enough to matter, short enough to be a
negligible prefix. Five epochs — `2.5%` of the run, one twelfth of one-cycle's warmup — gives the sharp
region a few steps to relax while costing the body almost nothing. The five ramp epochs accumulate
`0.02 + 0.04 + 0.06 + 0.08 + 0.10 = 0.30` in step-distance, against `0.50` at full rate, so the ramp costs
`0.20` of travel against the run's `~10` total — under `2%`, versus one-cycle's warmup which cost the easy
nets roughly half their early step-distance. And it is a one-parameter bridge between the last two runs:
`warmup = 0` recovers the pure cosine, `warmup ~ 60` recovers a one-cycle-length opening, and I place the
knob at `5`, near the cosine end, close enough to keep the body cosine won and far enough from zero to
shield the deep net.

Now the body — epochs 5 through 199 — starting at the full `base_lr` (continuous with the top of the ramp)
and annealing to zero. This is the cosine I already derived and validated on the easy nets. The only real
question is the clock. The ramp ends at exactly `base_lr` at epoch 4, so the cosine body must begin at
`base_lr` at the seam, which means measuring its progress from the end of warmup, spanning
`total_epochs - warmup = 195` epochs, not from epoch 0. Measured from the end of warmup,
`progress = (epoch - warmup)/(total - warmup)`: at `epoch = 5` progress is `0`, `eta = base_lr` —
continuous. Measured from epoch 0 instead, the cosine at `epoch = 5` reads `0.09985`, a `1.5e-4` drop at
the handoff and — worse — five epochs of the descent range used up on the warmup window, shortening the
anneal. The value jump is negligible but there is no reason to accept it when the warmup-relative clock
removes it exactly and hands the body its full 195-epoch budget.

Trace the seam. Epoch 4 (last warmup): `base_lr*5/5 = 0.10`. Epoch 5 (first body): progress `0`,
`base_lr*0.5*(1+cos 0) = 0.10`. So both epochs 4 and 5 sit at exactly `base_lr` — a two-epoch plateau at
the top — then epoch 6 begins the descent at `0.099994`. That plateau is a value-continuous handoff with
only a slope change (from the ramp's constant positive slope to the cosine's flat tangent), so nothing
shocks the dynamics, and it does not re-trigger the overshoot the ramp prevents: by epoch 4 the net has
taken four gradually-rising steps that already moved it into a flatter region where `2/lambda_max` has
risen above `base_lr`, so arriving at `0.10` lands on a ceiling that has already risen to admit it. The
ramp's samples `0.02, 0.04, 0.06, 0.08, 0.10` are the largest epoch-to-epoch changes anywhere in the
schedule (the body's are on the order of `base_lr/100`), but they are upward and gradual by construction,
the opposite of step decay's downward cliffs.

Stitch the two: for `epoch < warmup`, `base_lr*(epoch+1)/warmup`; for `epoch >= warmup`, the cosine body
with the warmup-relative clock. At `epoch = 199`, progress `= 194/195 = 0.99487`, `eta ≈ 6.6e-6`, just
above zero, reaching exactly zero at the boundary — the same settling convention the plain cosine had. It
conditions on nothing in `config`. The asymmetry is the whole point of putting a short warmup back rather
than a long one: nearly costless on the easy nets, decisive on the hard one. Summing the whole schedule's
per-epoch rate gives `10.10` in total travel against the pure cosine's `10.05` — a difference of `0.05` out
of ten, the `0.20` ramp tax at the front almost exactly offset by handing the body its full 195-epoch
descent. So on the nets that did not need init protection, this and the cosine are near-identical objects,
which is why I expect CIFAR-10 and FashionMNIST within a hair of their cosine values while the deep net,
which alone cared about the first five epochs, gets its overshoot fixed.

So the delta from the last run is one targeted addition: keep cosine's body verbatim — the part that won
CIFAR-10 (93.03) and FashionMNIST (94.70) — and prepend a five-epoch linear ramp from `base_lr/5` to
`base_lr`, with the cosine clock shifted to start at the end of warmup. The decisive claim is on CIFAR-100:
if the short warmup is the right fix, ResNet-56 / CIFAR-100 should clear both one-cycle's 71.57 and cosine's
71.07, and by the biggest margin of any setting, because that is where the init instability lived — I
expect it back into the 72s, recovering a genuine overshoot rather than shaving a floor. On the easy nets
the prediction is the opposite kind: CIFAR-10 and FashionMNIST should stay essentially where cosine put
them, a five-epoch ramp being nearly a no-op on nets that already tolerated the full rate. That is a real
risk, not a hedge: if either easy net drops noticeably, then even five epochs of sub-`base_lr` opening is
costing them early step-distance and five is too long. This attempt succeeds only if it buys CIFAR-100
while leaving CIFAR-10 and FashionMNIST untouched.
