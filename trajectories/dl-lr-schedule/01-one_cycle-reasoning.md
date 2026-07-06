The one knob I am allowed to move is the per-epoch learning rate, and the scaffold hands it to me as a
constant: `get_lr` returns `base_lr` for all 200 epochs. So before anything else I should be honest about
why a flat rate is the wrong place to stop, because the shape of its failure is what every schedule on
this ladder is trying to fix. Model the loss near the current weights as a quadratic with Hessian `H`.
One gradient step along an eigendirection of curvature `lambda` multiplies the distance to the minimum in
that direction by `(1 - eta*lambda)`, which contracts only while `|1 - eta*lambda| < 1`, i.e. while
`eta < 2/lambda`. So the admissible rate is capped by `2/lambda_max(H)`. Put my actual number on it: with
`eta = base_lr = 0.1`, the step diverges along any direction whose curvature exceeds `2/0.1 = 20`. That
threshold is not exotic — curvature is largest right after random init, because a freshly initialized net
sits in a sharp, badly-conditioned region: Kaiming-normal weights make the forward activations
unit-scaled but say nothing about the loss curvature, and before BatchNorm's running statistics have
settled the effective Hessian can have directions well past `lambda = 20`. Along those directions `0.1`
is above the ceiling, `(1 - eta*lambda)` has magnitude bigger than one, and the first steps *amplify* the
error instead of shrinking it — the loss spikes rather than descends. That is the early end. The late end
is the mirror image, but the mechanism is stochastic rather than curvature: near a minimum the *true*
gradient has shrunk toward zero, but the *minibatch* gradient has not. Each step still injects a
displacement whose size does not vanish with the true gradient, and treating SGD near the minimum as an
Ornstein–Uhlenbeck process, the stationary variance of the iterate around the optimum scales like `eta`
times the gradient-noise covariance — linear in the rate. So a fixed `0.1` parks the weights in a cloud
of residual jitter whose variance is set by the rate, and it keeps kicking them around the minimum at
that noise floor instead of settling into it. A single constant rate is therefore structurally
impossible: the start wants `eta < 2/lambda_max` to avoid the overshoot and the end wants a rate small
enough to shrink the stationary jitter, and those are different numbers separated by orders of magnitude.
The rate has to change over the run.

That tells me the curve must be high in the middle and small at the ends, but it does not yet tell me the
*shape*, and I want to choose the first rung deliberately rather than reach for the obvious. Lay out what
is actually on the table. The conventional answer is step decay — hold a high plateau, divide by a fixed
factor at hand-picked milestones. Concretely that is `0.1` until epoch 60, then `0.02` to 120, then
`0.004` to 160, then `0.0008` to the end: three milestones and a drop factor, four numbers all glued to
the 200-epoch budget (60/120/160 are really 30/60/80 percent of `T`, so change `T` and I re-pick every
one). It works, but it is a staircase — a discontinuity at each drop that shocks the dynamics and lurches
the loss, a rate that between drops is systematically mismatched to wherever the iterate now sits, and
nothing at all about the unstable start (it holds the full `0.1` from epoch 0, straight into the
curvature ceiling I just computed). The second option is the triangular cyclical rate — ramp linearly up
to a `max_lr` and back down, repeatedly, between two fixed bounds — which at least says something the
staircase does not: that going *up* before coming down can help. The third is to take that one idea and
commit to it once. I want to start the ladder from a schedule that is already *smooth* and that probes
the most interesting claim in this lineage — that a rise before the fall is beneficial overall even
though each rise temporarily hurts — so that when it underperforms I learn something sharp about what to
keep and what to drop, rather than learning only that staircases are ugly, which I already know.

The justification for the rise is the loss topology, and it is worth walking rather than asserting.
Training makes steep, fast progress in the first handful of epochs, then enters a long nearly-flat valley
where the slope is tiny and per-iteration progress is tiny, and only at the end has to thread a narrow
trough to the minimum. Overlay the rate on that topology. In the steep early part a big step overshoots
the walls of the ravine, so I want a *small* rate. In the long flat valley a small rate is exactly wrong:
the step is `eta` times the slope, and a small rate times a tiny slope is a microscopic step, so I stall,
crawling over saddle plateaus for epochs instead of crossing them — here I want a *large* rate to punch
through fast. At the very end, threading the trough, a large rate bounces me out of it, so I want a
*small* rate again. The rate read straight off the topology is small, then large, then small: one rise
and one fall. Not the triangular policy's perpetual oscillation between two bounds — the topology wants
one fall that *settles*, not a band held forever — so my first rung is a single cycle, ramp up to a peak,
then ramp down. This is the one-cycle shape.

But I have to be careful here, because the *name* "one-cycle" carries a much more aggressive method than
what this harness can actually run, and starting the ladder honestly means being exact about which
version I am fitting. The full one-cycle policy is built around three moves the topology alone does not
give me, and each one leans on a lever this task freezes. First, the peak is meant to be *enormous* — an
order of magnitude above the usual `0.1` — found by sweeping the rate upward in a short pre-run (the LR
range test) and reading off where accuracy turns ragged; the whole "super-convergence" claim rests on the
network tolerating rates I would normally rule out, because the large rate in the valley is itself a
strong regularizer. Second, *because* the large rate carries so much regularization, the other
regularizers have to be turned *down* to keep the total balanced — reduce weight decay, maybe dropout — or
the network over-regularizes and cannot use the large rate at all. Third, momentum is cycled *inversely*
to the rate, high when the rate is low and dropped to a floor at the peak, and this is the one I want to
put a number on, because it is the load-bearing stability move. In the SGD-with-momentum update the
velocity accumulates as `v <- m*v + g` and the step is `eta*v`, so under a sustained gradient the
displacement per step settles to `eta*g/(1 - m)`: the momentum multiplies the effective step by
`1/(1 - m)`. With the frozen `m = 0.9` that factor is `1/0.1 = 10` — every nominal `eta` is really a
`10*eta` displacement. The full method's inverse cycling swings momentum from about `0.95` at the low-rate
ends to `0.85` at the peak, which swings the amplifier from `1/0.05 = 20` down to `1/0.15 = 6.67`: it
*shrinks* the effective-step multiplier by a factor of three exactly where the rate is largest, so the
rising `eta` and the falling `1/(1-m)` partly cancel and the effective step stays inside the stability
region. That cancellation is what lets the peak go to `10*base_lr` at all.

Now hold each of those against this task's edit surface, because the surface decides what "one-cycle" even
means here. I get to edit exactly one thing: the body of `get_lr`, which returns a single float per epoch.
Momentum is fixed at `0.9` by the frozen optimizer and the loop never re-sets it, so I *cannot* cycle it —
the amplifier is pinned at `10*` for the whole run, and the inverse-momentum machinery that keeps the
large-rate phase stable is simply unavailable. Weight decay is fixed at `5e-4` and I cannot touch it, so
the rebalancing principle is unavailable too. And the LR range test is not part of the contract: `get_lr`
is a pure function of `epoch`, with no pre-run hook and no way to read a tolerated peak off the network.
So let me actually try the aggressive port and watch it fail on the arithmetic rather than on a feeling.
Suppose I chase super-convergence and set the peak to `10*base_lr = 1.0` with momentum frozen at `0.9` and
weight decay frozen high. The effective step at the peak is `eta/(1-m) = 1.0/0.1 = 10`, against
`0.1/0.1 = 1.0` for a well-behaved `base_lr` run — a tenfold-larger effective step, and a full hundred
times the plain-GD step `0.1` I would take at `base_lr` with the momentum stripped out — and with none of
the compensations the full method uses to survive it: no dropped momentum (the amplifier is pinned at `10`
where the real method would have dropped it to `6.67` at its peak), no reduced weight decay. That is precisely the configuration the method itself warns is unstable, reproduced with the safety
systems removed. So the aggressive peak is ruled out not by taste but by the effective-step count: the
harness has amputated exactly the two levers (momentum cycling, weight-decay rebalancing) that make a
large peak survivable, so I must not use a large peak. The honest port into *this* harness is the
deliberately tame one: keep the rise-then-fall *shape*, but peak at `base_lr` itself rather than far above
it, leave momentum and weight decay exactly where the loop fixes them, and replace the range-test peak
with the reference rate I am given.

With the peak pinned at `base_lr`, the remaining design is the shape of the two legs and how far below the
start the tail goes. For the legs I want a smooth curve, not the triangular policy's linear corners,
because a cosine eases into and out of the extremes — it spends a little more time near the top (sustained
exploration at the peak) and near the bottom (a gentle landing) and, most importantly, has zero slope at
the peak, which is the most dangerous moment to jolt the dynamics. A cosine from a start value to an end
value across a fraction `pct` running 0→1 is `end + (start - end)/2 * (cos(pi*pct) + 1)`: it equals
`start` at `pct=0` and `end` at `pct=1`, gliding monotonically between, with zero slope at both ends. I
use that for both legs, and I should check the two branches actually compose into a continuous rise-then-
fall rather than trusting the formula. Take the up-leg written as `min_lr + (base_lr - min_lr)*0.5*(1 +
cos(pi*(1 - t)))` with `t` the fraction through warmup. At `t=0` the argument is `pi`, `cos(pi) = -1`, so
it returns `min_lr` — the climb genuinely opens at the floor. At `t=1` the argument is `0`, `cos(0) = 1`,
so it returns `base_lr` — the climb tops out at the peak. Monotone increasing between, since `1 - t` runs
`1→0` and `cos` rises on that interval. The down-leg `final_lr + (base_lr - final_lr)*0.5*(1 + cos(pi*t))`
at `t=0` gives `base_lr` and at `t=1` gives `final_lr`, monotone decreasing. So the two legs meet at
`base_lr` at the seam with no value jump, which is what I wanted.

How long is the up-leg versus the down-leg, and where does it start? The topology says the steep early
region is short and the flat valley plus final descent is most of the run, so a short ramp up and a long
ramp down — the first 30% climbing (`pct_start = 0.3`, sixty of two hundred epochs), the last 70%
descending. The climb does not open at the peak; I parameterize the start as the peak divided by a
division factor of 25, so it opens at `base_lr/25 = 0.004` — comfortably below the `2/lambda = 20`
ceiling, since a rate of `0.004` only diverges along directions with curvature past `500`, which init is
very unlikely to have — and cosine-climbs to `base_lr` over the first 30%. Then the down-leg cosine-
descends, and the question is to *what* floor. The full method drives the tail orders of magnitude below
the start (a `final_div` of `1e4`, ending near `1e-5`) — that deep "annihilation" is the move that lets
SGD drop into a steep narrow minimum inside the wide flat region the large-rate phase found. But that move
only earns its keep when there *was* a large-rate phase to carry the iterate into a wide region; with my
peak capped at the ordinary `base_lr`, there is no wide-region-then-annihilate story to honor, so I keep
the floor modest at the same factor, `final_div = 25`, ending at `base_lr/25 = 0.004`. So the schedule is:
cosine warmup from `0.004` to `0.1` over the first 30%, then cosine anneal from `0.1` back to `0.004` over
the remaining 70%. Both cosine, both anchored at `base_lr` and `base_lr/25`.

Before I sample the curve I want to check the parameterization does not do something silly at its
extremes, because a good parameterization should degrade into a schedule I already understand when its
knobs go to their limits. Take `div_factor -> 1`: then `min_lr = base_lr`, the up-leg starts *at* the
peak, and the warmup collapses — the schedule is a pure cosine decay from `base_lr` to `base_lr/25` with a
flat 30% prefix at the top. Take `pct_start -> 0`: the up-leg vanishes and the whole run is one cosine
decay from `base_lr` to `base_lr/25`, which is the plain single-cycle anneal (bar the non-zero floor).
Take `final_div -> infinity`: the tail goes to `0`, the deep-annihilation limit. So the three knobs
interpolate exactly between the shapes I want to be able to name — a flat-top decay, a pure anneal, a
full annihilation — and my chosen `(0.3, 25, 25)` sits at the tame interior of that box, with a real
warmup and a modest floor. That is the reassurance I wanted: I am not choosing an arbitrary curve, I am
choosing a point in a family whose boundaries are all schedules with clear meanings.

There is one more shape decision hiding in "cosine legs," and it is worth being explicit about why I did
*not* take the triangular policy's straight legs. A linear up-then-down has constant slope on each leg and
therefore a *corner* at the peak: the derivative jumps from `+s` to `-s` in one epoch. That corner is a
discontinuity in the rate of change right at the moment the effective step is largest — the velocity the
momentum has accumulated over the climb suddenly finds itself paired with a rate that has stopped rising
and started falling, and the mismatch between the `10*`-amplified velocity and the abruptly-turning rate
is exactly the kind of jolt I flagged as dangerous. The cosine's zero slope at the peak removes that
corner: the rate flattens as it approaches `base_lr`, holds near it for a beat, and eases into the
descent, so the velocity and the rate turn together. The same argument applies at the floor, where the
cosine flattens instead of hitting `base_lr/25` at full speed. So the cosine is not decoration over the
triangle; it is specifically the corner-free version, and on a momentum optimizer with a `10*` effective-
step amplifier the corners are where I would expect a linear cycle to lose ground.

Let me sample the actual curve to see what these choices produce, because the sampled values are what the
loop will really apply. With `progress = epoch/199`: epoch 0 gives `0.004`; epoch 10 gives `0.0105`;
epoch 20 gives `0.0282`; epoch 30 gives `0.0524`; the peak arrives around epoch 59–60 at `0.0999`; then
the descent, epoch 100 at `0.0815`, epoch 129 at `0.0524`, down to epoch 199 at `0.0040`. Reading that
column tells me exactly what I have built and — before any measurement — where I expect it to leak. Two
features of this constrained one-cycle make me expect it to be the *weakest* rung, not because the shape
is wrong but because the harness amputated the parts that made it strong. The first is the warmup-shaped
*start*: the sampled values say that at epoch 10 the rate is still only `0.0105`, at epoch 20 only
`0.028`, at epoch 30 only `0.052` — the net spends the first thirty-to-sixty epochs, a long slice of the
run, stepping at a fraction of the `0.1` it could safely use after the first handful of epochs, so a large
part of the most productive phase is spent under-stepping. Sixty epochs is a very long warmup; a net whose
curvature has relaxed within the first few epochs is then made to crawl. I can put a number on how much
stepping that costs: summing the sampled rate over epochs 0–59 gives about `3.09` in accumulated `eta`,
against `6.0` if those same sixty epochs had run at the full `0.1`. So the climb spends only about half the
step-distance a full-rate opening would have covered — it throws away close to `49%` of the "travel"
available in the first thirty percent of the run, and it does so precisely in the phase where the true
gradient is largest and most informative. The second is the *end*: the down-
leg stops at `0.004`, not at zero. The late-training argument was that the stationary jitter scales like
`eta`, so a floor of `0.004` leaves a residual cloud whose variance is `0.004` times the noise covariance
rather than shrinking it toward zero — there is jitter at the finish that a schedule annealing all the way
to `0` would have removed. So I expect this one-cycle to leave accuracy on the table at both ends: a too-
long sub-`base_lr` warmup eating the productive middle, and a non-zero floor that never fully settles the
trough.

Which of the three settings should feel that most? The deeper, harder net — ResNet-56 on CIFAR-100 — is
where a long warmup and a too-high floor should both bite hardest, because a 56-layer net on 100 classes
needs both the productive high-rate middle (to make progress at all on a hard target) and a clean low-rate
finish (to resolve 100-way boundaries) more than a shallow 10-class net does. The shallow ResNet-20 on
CIFAR-10 should be the most forgiving — easy enough that even an under-stepped middle and a residual-jitter
finish still land near where a cleaner schedule would. MobileNetV2 on FashionMNIST is the one I am least
sure about: an inverted-residual net on grayscale replicated to three channels, where the floor and the
warmup length interact with a different architecture, but I expect the same two leaks to show. My
falsifiable expectation, then, is concrete: this constrained one-cycle should come in at or below a plain
smooth anneal on every setting, and the gap should be widest where the amputations cost most — the deep
CIFAR-100 net and the grayscale mobile net — with CIFAR-10 closest. I cannot pin exact numbers before I
run it; the three-setting accuracy table will tell me the magnitudes, but the *ordering* of the gaps is a
real prediction I can be wrong about.

Whatever the exact numbers, the diagnosis is already pointed at the next rung, and that is the entire
reason to run this shape first. The two leaks are the sub-`base_lr` warmup and the non-zero floor, so the
natural next move is to *drop the warmup entirely* and let a single smooth curve start at the full rate
and anneal all the way to zero — which is exactly the cosine schedule, the down-leg of this one-cycle
stretched over the whole run with the climb removed and the floor taken to `0`. If that beats this one-
cycle, it confirms the warmup and the floor, not the cosine shape, were the cost; if it does *not*, the
warmup was doing real curvature-protection work and I keep it. Either way the next rung is the experiment
this rung is designed to set up.

The causal chain in one breath: a constant rate fails at both ends because the admissible step is
curvature-bounded (`eta < 2/lambda_max`, and `0.1` diverges past `lambda = 20`, which sharp Kaiming init
can reach) and because the minibatch gradient does not vanish at the minimum (stationary jitter scales
like `eta`), so the rate must be small-high-small; the cyclical-LR lineage says a *rise* before the fall
helps, by crossing the flat valley and saddle plateaus fast, which argues for a single rise-then-fall
cycle rather than a staircase or a perpetual band; but the full one-cycle's power comes from a huge range-
test peak, rebalanced weight decay, and inverse-momentum cycling — and putting the arithmetic on it, the
frozen `m = 0.9` pins the effective-step amplifier at `10*` with no way to drop it at the peak, so a
`10*base_lr` peak would be a `100*` effective step with every safety system removed — so the faithful port
peaks at `base_lr`, keeps the modest `base_lr/25` floor, and cosine-shapes a 30%-up, 70%-down cycle;
sampling the curve shows it crawling at `0.01`–`0.05` through the first thirty epochs and floored at
`0.004`, so I expect it to be the weakest rung, worst on the deep CIFAR-100 net, and I expect the next
rung — drop the warmup, anneal smoothly to zero — to close exactly those two leaks.
