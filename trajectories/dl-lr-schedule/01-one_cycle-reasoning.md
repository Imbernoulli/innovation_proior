The one knob I am allowed to move is the per-epoch learning rate, and the default I am handed is a
constant: `get_lr` returns `base_lr` for all 200 epochs. So before anything else I should be honest about
why a flat rate is the wrong place to stop, because the shape of its failure is what every schedule I try
is trying to fix. Model the loss near the current weights as a quadratic with Hessian `H`. One gradient
step along an eigendirection of curvature `lambda` multiplies the distance to the minimum in that
direction by `(1 - eta*lambda)`, which contracts only while `|1 - eta*lambda| < 1`, i.e. while
`eta < 2/lambda`. So the admissible rate is capped by `2/lambda_max(H)`. With `eta = base_lr = 0.1`, the
step diverges along any direction whose curvature exceeds `2/0.1 = 20`. That threshold is not exotic:
curvature is largest right after random init, because a freshly initialized net sits in a sharp,
badly-conditioned region — Kaiming-normal weights make the forward activations unit-scaled but say nothing
about the loss curvature, and before BatchNorm's running statistics have settled the effective Hessian can
have directions well past `lambda = 20`. Along those directions `0.1` is above the ceiling,
`(1 - eta*lambda)` has magnitude bigger than one, and the first steps amplify the error instead of
shrinking it — the loss spikes rather than descends. That is the early end. The late end is the mirror
image, but the mechanism is stochastic rather than curvature: near a minimum the true gradient has shrunk
toward zero while the minibatch gradient has not. Each step still injects a displacement whose size does
not vanish with the true gradient, and treating SGD near the minimum as an Ornstein–Uhlenbeck process, the
stationary variance of the iterate around the optimum scales like `eta` times the gradient-noise
covariance — linear in the rate. So a fixed `0.1` parks the weights in a cloud of residual jitter and
keeps kicking them around the minimum at that noise floor instead of settling into it. A single constant
rate is structurally impossible: the start wants `eta < 2/lambda_max` to avoid the overshoot and the end
wants a rate small enough to shrink the stationary jitter, and those are different numbers separated by
orders of magnitude. The rate has to change over the run.

That tells me the curve must be high in the middle and small at the ends, but not yet the shape. The
conventional answer is step decay — hold a high plateau, divide by a fixed factor at hand-picked
milestones: `0.1` until epoch 60, then `0.02` to 120, `0.004` to 160, `0.0008` to the end, four numbers
all glued to the 200-epoch budget (60/120/160 are 30/60/80 percent of `T`, so change `T` and I re-pick
every one). It works, but it is a staircase — a discontinuity at each drop that shocks the dynamics, a
rate that between drops is mismatched to wherever the iterate now sits, and it holds the full `0.1` from
epoch 0 straight into the curvature ceiling I just computed. The triangular cyclical rate — ramp linearly
up to a `max_lr` and back down, repeatedly — at least says something the staircase does not: going up
before coming down can help. I want to start from a schedule that is already smooth and that probes the
most interesting claim in this lineage — that a rise before the fall is beneficial overall even though each
rise temporarily hurts — so that when it underperforms I learn something sharp about what to keep, rather
than only that staircases are ugly.

The justification for the rise is the loss topology. Training makes steep, fast progress in the first
handful of epochs, then enters a long nearly-flat valley where the slope is tiny and per-iteration progress
is tiny, and only at the end has to thread a narrow trough to the minimum. Overlay the rate on that
topology. In the steep early part a big step overshoots the walls of the ravine, so I want a small rate. In
the long flat valley a small rate is exactly wrong: the step is `eta` times the slope, and a small rate
times a tiny slope is a microscopic step, so I stall, crawling over saddle plateaus for epochs — here I
want a large rate to punch through. At the very end, threading the trough, a large rate bounces me out of
it, so I want a small rate again. Small, then large, then small: one rise and one fall — not the triangular
policy's perpetual oscillation, but a single cycle that settles. This is the one-cycle shape.

But the name "one-cycle" carries a much more aggressive method than what I can actually run here, and being
honest means being exact about which version I am fitting. The full one-cycle policy is built around three
moves, each leaning on a lever this task freezes. First, the peak is meant to be enormous — an order of
magnitude above `0.1` — found by sweeping the rate upward in a short pre-run (the LR range test); the whole
"super-convergence" claim rests on the large valley rate being itself a strong regularizer. Second,
because that large rate carries so much regularization, weight decay has to be turned down to keep the
total balanced. Third, momentum is cycled inversely to the rate, high when the rate is low and dropped to a
floor at the peak — and this is the load-bearing stability move, so let me put a number on it. In
SGD-with-momentum the velocity accumulates as `v <- m*v + g` and the step is `eta*v`, so under a sustained
gradient the displacement per step settles to `eta*g/(1 - m)`: momentum multiplies the effective step by
`1/(1 - m)`. With `m = 0.9` that factor is `1/0.1 = 10` — every nominal `eta` is really a `10*eta`
displacement. The full method's inverse cycling swings momentum from about `0.95` at the low-rate ends to
`0.85` at the peak, swinging the amplifier from `20` down to `6.67`: it shrinks the effective-step
multiplier by a factor of three exactly where the rate is largest, so the rising `eta` and the falling
`1/(1-m)` partly cancel and the effective step stays inside the stability region. That cancellation is what
lets the peak reach `10*base_lr` at all.

Now hold those against my edit surface, because it decides what "one-cycle" even means here. I edit exactly
one thing: the body of `get_lr`, which returns a single float per epoch. Momentum is fixed at `0.9` and the
loop never re-sets it, so I cannot cycle it — the amplifier is pinned at `10*` for the whole run, and the
inverse-momentum machinery that keeps the large-rate phase stable is unavailable. Weight decay is fixed at
`5e-4`, so the rebalancing is unavailable too. And there is no pre-run hook for a range test. Price the
aggressive port rather than dismiss it: set the peak to `10*base_lr = 1.0` with momentum frozen at `0.9`.
The effective step at the peak is `eta/(1-m) = 1.0/0.1 = 10`, against `0.1/0.1 = 1.0` for a well-behaved
`base_lr` run — tenfold larger, and a full hundred times the plain-GD step at `base_lr` — with none of the
compensations: no dropped momentum (pinned at `10` where the real method would have dropped it to `6.67`),
no reduced weight decay. That is exactly the configuration the method warns is unstable, reproduced with
the safety systems removed. So the honest port is the deliberately tame one: keep the rise-then-fall shape,
peak at `base_lr` itself, leave momentum and weight decay where they are fixed, and replace the range-test
peak with the reference rate I am given.

With the peak pinned at `base_lr`, the remaining design is the shape of the two legs and how far below the
start the tail goes. For the legs I want a smooth curve, not the triangular policy's linear corners,
because a cosine eases into and out of the extremes — a little more time near the top and near the bottom
and, most importantly, zero slope at the peak, the most dangerous moment to jolt the dynamics. A cosine
from `start` to `end` across a fraction `pct` in `0→1` is `end + (start - end)/2 * (cos(pi*pct) + 1)`:
`start` at `pct=0`, `end` at `pct=1`, gliding monotonically with zero slope at both ends. I use it for both
legs, anchored so the up-leg tops out at `base_lr` exactly where the down-leg begins, meeting at the seam
with no value jump.

How long is the up-leg versus the down-leg, and where does it start? The topology says the steep early
region is short and the flat valley plus final descent is most of the run, so a short ramp up and a long
ramp down — the first 30% climbing (`pct_start = 0.3`, sixty epochs), the last 70% descending. The climb
does not open at the peak; I parameterize the start as the peak over a division factor of 25, opening at
`base_lr/25 = 0.004` — comfortably below the ceiling, since `0.004` only diverges along curvature past
`500`, which init is very unlikely to have — and cosine-climbs to `base_lr`. Then the down-leg
cosine-descends to a floor. The full method drives the tail orders of magnitude below the start (a
`final_div` of `1e4`, ending near `1e-5`) — that deep "annihilation" lets SGD drop into a steep narrow
minimum inside the wide flat region the large-rate phase found. But that move only earns its keep when
there was a large-rate phase to carry the iterate into a wide region; with my peak capped at ordinary
`base_lr`, there is no such story to honor, so I keep the floor modest, `final_div = 25`, ending at
`base_lr/25 = 0.004`. So: cosine warmup from `0.004` to `0.1` over the first 30%, then cosine anneal from
`0.1` back to `0.004` over the remaining 70%.

One shape decision I should be explicit about is why I did not take the triangular policy's straight legs. A
linear up-then-down has constant slope on each leg and a corner at the peak: the derivative jumps from `+s`
to `-s` in one epoch. That corner is a discontinuity in the rate of change right when the effective step is
largest — the velocity the momentum accumulated over the climb suddenly finds itself paired with a rate
that has stopped rising and started falling, and the mismatch between the `10*`-amplified velocity and the
abruptly-turning rate is the jolt I flagged as dangerous. The cosine's zero slope at the peak removes that
corner: the rate flattens as it approaches `base_lr`, holds near it for a beat, and eases into the descent,
so velocity and rate turn together. The same holds at the floor. So the cosine is specifically the
corner-free version, and on a momentum optimizer with a `10*` amplifier the corners are where I would
expect a linear cycle to lose ground.

Let me sample the curve, since the sampled values are what the loop will really apply. With
`progress = epoch/199`: epoch 0 gives `0.004`; epoch 10 `0.0105`; epoch 20 `0.0282`; epoch 30 `0.0524`;
the peak arrives around epoch 59–60 at `0.0999`; then epoch 100 at `0.0815`, epoch 129 at `0.0524`, down to
epoch 199 at `0.0040`. That column tells me where I expect it to leak. Two features make me expect this to
be the weakest schedule I try, not because the shape is wrong but because the missing levers force it. The
first is the warmup-shaped start: at epoch 10 the rate is still only `0.0105`, at epoch 30 only `0.052` —
the net spends the first thirty-to-sixty epochs, a long slice of the run, stepping at a fraction of the
`0.1` it could safely use after the first few epochs, so a large part of the most productive phase is spent
under-stepping. Summing the sampled rate over epochs 0–59 gives about `3.09` in accumulated `eta`, against
`6.0` if those sixty epochs had run at full `0.1` — the climb covers only about half the step-distance a
full-rate opening would have, throwing away close to `49%` of the "travel" available in the first thirty
percent, precisely where the true gradient is largest. The second is the end: the down-leg stops at
`0.004`, not zero, so by the stationary-jitter argument the finish carries a residual cloud of variance
proportional to `0.004` rather than shrinking toward zero — jitter at the finish a schedule annealing to
`0` would have removed.

Which setting should feel that most? The deeper, harder net — ResNet-56 on CIFAR-100 — is where a long
warmup and a too-high floor should both bite hardest, because a 56-layer net on 100 classes needs both the
productive high-rate middle and a clean low-rate finish more than a shallow 10-class net does. The shallow
ResNet-20 on CIFAR-10 should be the most forgiving. MobileNetV2 on FashionMNIST is the one I am least sure
about, but I expect the same two leaks. So my falsifiable expectation is concrete: this constrained
one-cycle should come in at or below a plain smooth anneal on every setting, and the gap should be widest
on the deep CIFAR-100 net and the grayscale mobile net, with CIFAR-10 closest. I cannot pin exact numbers
before I run it, but the ordering of the gaps is a real prediction I can be wrong about.

The diagnosis already points at the next move, which is the reason to run this shape first: the two leaks
are the sub-`base_lr` warmup and the non-zero floor, so the natural next step is to drop the warmup
entirely and let a single smooth curve start at the full rate and anneal all the way to zero — the cosine
schedule, this down-leg stretched over the whole run with the climb removed and the floor taken to `0`. If
it beats this, the warmup and floor were the cost, not the cosine shape; if not, the warmup was doing real
curvature-protection work and I keep it.
