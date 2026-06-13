The one knob I am allowed to move is the per-epoch learning rate, and the scaffold hands it to me as a
constant: `get_lr` returns `base_lr` for all 200 epochs. So before anything else I should be honest about
why a flat rate is the wrong place to stop, because the shape of its failure is what every schedule on
this ladder is trying to fix. Model the loss near the current weights as a quadratic with Hessian `H`.
One gradient step along an eigendirection of curvature `lambda` multiplies the distance to the minimum in
that direction by `(1 - eta*lambda)`, which contracts only while `eta < 2/lambda`. So the admissible rate
is capped by `2/lambda_max(H)`, and curvature is largest right after random init — a freshly initialized
net sits in a sharp region, `lambda_max` is big, `2/lambda_max` is small, and 0.1 can easily be above
that ceiling, so the first steps overshoot and the loss spikes instead of descending. That is the early
end. The late end is the mirror image: near a minimum the *true* gradient has shrunk but the *stochastic*
gradient has not — minibatch noise gives every step a component that does not vanish — so a step of 0.1
keeps kicking the weights around the minimum at the noise floor instead of settling. A single constant
rate is therefore structurally impossible: the start wants a rate below the curvature ceiling and the end
wants a tiny rate to beat the noise floor, and those are different numbers. The rate has to change.

That tells me the curve must be high in the middle and small at the ends, but it does not yet tell me the
*shape*, and I want to choose the first rung deliberately. The conventional answer is step decay — hold a
high plateau, divide by a fixed factor at hand-picked milestones — and it works, but it is a staircase: a
discontinuity at each drop that shocks the dynamics, milestones glued to the 200-epoch budget, and
nothing at all about the unstable start. I want to start the ladder from a schedule that is already
*smooth* and that probes the most interesting idea in this lineage — that going *up* before coming down
can help — so that when it underperforms I learn something sharp about what to keep and what to drop. The
cyclical-learning-rate observation is exactly that idea made concrete: letting the rate rise and fall,
rather than only fall, is beneficial overall even though each rise temporarily hurts. The justification is
the loss topology. Training makes steep, fast progress in the first handful of epochs, then enters a long
nearly-flat valley where the slope is tiny and per-iteration progress is tiny, and only at the end has to
thread a narrow trough to the minimum. Overlay the rate on that. In the steep early part a big step
overshoots, so I want a *small* rate. In the long flat valley a small rate is exactly wrong — a small
rate times a tiny slope is a microscopic step, and I stall in the plateau — so I want a *large* rate to
cross the flat region fast and punch through the saddle plateaus rather than crawl over them. At the very
end, threading the trough, a large rate would bounce me out, so I want a *small* rate again. The rate read
straight off the topology is small, then large, then small: one rise and one fall.

So my first rung is a single cycle — ramp up to a peak, then ramp down — rather than the triangular
policy's many oscillations between two fixed bounds, because the topology wants one fall that settles, not
a perpetual band. This is the one-cycle shape. But I have to be careful here, because the *name*
"one-cycle" carries a much more aggressive method than what this harness can actually run, and starting
the ladder means being exact about which version I am fitting. The full one-cycle policy is built around
three moves the topology alone does not give me. First, the peak is meant to be *enormous* — an order of
magnitude above the usual 0.1 — found by sweeping the rate upward in a short pre-run (the LR range test)
and reading off where accuracy turns ragged; the whole "super-convergence" claim rests on the network
tolerating rates I would normally rule out, because the large rate in the valley is itself a strong
regularizer (across a band of large rates the training loss rises while the test loss falls, the gap
shrinks). Second, *because* the large rate carries so much regularization, the other regularizers have to
be turned *down* to keep the total balanced — reduce weight decay, maybe dropout — or the network
over-regularizes and cannot use the large rate at all. Third, momentum is cycled *inversely* to the rate
(high when the rate is low, dropped to a floor at the peak), because in the SGD-with-momentum update the
displacement scales with both `eta` and `m`, so a rising rate and a high momentum stack two amplifiers and
the effective step blows past stability.

Now hold each of those against this task's edit surface, because the surface is narrow and it decides what
"one-cycle" even means here. I get to edit exactly one thing: the body of `get_lr`, which returns a single
learning-rate float per epoch. Momentum is fixed at 0.9 by the frozen optimizer and the loop never
re-sets it, so I *cannot* cycle momentum — the inverse-momentum machinery, the part that keeps the
large-rate phase stable, is simply unavailable. Weight decay is fixed at 5e-4 and I cannot touch it, so
the balance principle — turn the other regularizers down so a large rate fits — is also unavailable; if I
pushed the rate to ten times 0.1 with weight decay frozen high and momentum frozen at 0.9, I would be
running the exact configuration the full method warns is unstable, with none of the compensations. And the
LR range test is not part of the contract either: `get_lr` is a pure function of `epoch`, with no pre-run
hook and no way to read a tolerated peak off the network. So the honest port of one-cycle into *this*
harness is the deliberately tame one: keep the rise-then-fall *shape*, but peak at `base_lr` itself
rather than far above it, leave momentum and weight decay exactly where the loop fixes them, and replace
the range-test peak with the reference rate I am given. The signature super-convergence regime — huge
peak, rebalanced regularization, cycled momentum — is exactly the part this task does not expose, and
pretending otherwise would just be the unstable configuration in disguise.

With the peak pinned at `base_lr`, the remaining design is the shape of the two legs and how far below the
start the tail goes. For the legs I want a smooth curve, not the triangular policy's linear corners,
because a cosine eases into and out of the extremes — it spends a little more time near the top (sustained
exploration at the peak) and near the bottom (a gentle landing) and avoids an abrupt slope change at the
peak, which is the most dangerous moment to jolt the dynamics. A cosine from a start value to an end value
across a fraction `pct` running 0→1 is `end + (start - end)/2 * (cos(pi*pct) + 1)`: it equals `start` at
`pct=0` and `end` at `pct=1`, gliding monotonically between, with zero slope at both ends. I use that for
both legs. How long is the up-leg versus the down-leg? The topology says the steep early region is short
and the flat valley plus the final descent is most of the run, so a short ramp up and a long ramp down —
the first 30% climbing, the last 70% descending. Where does the up-leg start? Not at the peak; well below
it, so the warmup genuinely begins small while curvature is high. I parameterize the start as the peak
divided by a division factor of 25, so the climb opens at `base_lr/25 = 0.004` — comfortably below the
curvature ceiling at init — and cosine-climbs to `base_lr` over the first 30% of the run. Then the
down-leg cosine-descends from `base_lr`, and the question is to *what* floor. The full method drives the
tail orders of magnitude below the start (a `final_div_factor` of 1e4) — that deep "annihilation" is the
move that lets SGD drop into a steep narrow minimum inside the wide flat region the large-rate phase found.
But that move only earns its keep when there *was* a large-rate phase to carry the iterate into a wide
region; with my peak capped at the ordinary `base_lr`, there is no wide-region-then-annihilate story to
honor, so I keep the floor modest: `final_div` of 25, the same factor, ending the run at `base_lr/25`
rather than near zero. So the schedule is: cosine warmup from `base_lr/25` to `base_lr` over the first
30%, then cosine anneal from `base_lr` back down to `base_lr/25` over the remaining 70%. Two phases, both
cosine, both anchored at `base_lr` and `base_lr/25`. (The full schedule body is in the answer.)

Now I should reason about what this floor is likely to do, because that is the entire point of running it
first. Two features of this constrained one-cycle make me expect it to be the *weakest* rung, not because
the shape is wrong but because the harness amputates the parts that made it strong. The first is the
warmup-shaped *start*: the up-leg spends the first 30% of the run — sixty epochs out of two hundred — with
the rate *below* `base_lr`, climbing from 0.004 and only reaching the full 0.1 at epoch 60. Sixty epochs
is a long warmup; a net whose curvature has relaxed within the first handful of epochs is then made to
crawl at a fraction of the rate it could safely use, so a large slice of the most productive phase is
spent under-stepping. The second is the *end*: the down-leg stops at `base_lr/25 = 0.004`, not at zero.
The late-training argument was that I need the rate small enough to settle below the stochastic-gradient
noise floor, and 0.004 is small but it is not zero — there is residual jitter at the finish that a
schedule annealing all the way to 0 would have removed. So I expect this one-cycle to leave accuracy on
the table at both ends: a too-long sub-`base_lr` warmup eating the productive middle, and a non-zero floor
that never fully settles the trough.

Which settings should feel that most? The deeper, harder net — ResNet-56 on CIFAR-100 — is where a long
warmup and a too-high floor should both bite hardest, because a 56-layer net on 100 classes needs both
the productive high-rate middle and a clean low-rate finish more than a shallow 10-class net does. The
shallow ResNet-20 on CIFAR-10 should be the most forgiving — easy enough that even an under-stepped middle
and a residual-jitter finish still land near the others. MobileNetV2 on FashionMNIST is the unknown: an
inverted-residual net on grayscale, where the floor and the warmup length interact with a different
architecture, but I expect the same two leaks. My falsifiable expectation, then, is concrete: this
constrained one-cycle should come in at or below a plain smooth anneal on every setting, and the gap
should be widest where the amputations cost most — the deep CIFAR-100 net and the grayscale mobile net —
with CIFAR-10 closest. Whatever the exact numbers, the diagnosis is already pointed at the next rung: the
two leaks are the sub-`base_lr` warmup and the non-zero floor, so the natural next move is to *drop the
warmup entirely* and let a single smooth curve start at the full rate and anneal all the way to zero —
which is exactly the cosine schedule. If that beats this one-cycle, it tells me the warmup and the floor,
not the cosine shape, were the cost; if it does not, the warmup was doing real work and I keep it. Either
way the next rung is the experiment this rung sets up.

The causal chain in one breath: a constant rate fails at both ends because the admissible step is
curvature-bounded at init and the stochastic gradient does not vanish at the minimum, so the rate must be
small-high-small; the cyclical-LR lineage says a *rise* before the fall helps, by crossing the flat valley
and saddle plateaus fast, which argues for a single rise-then-fall cycle; but the full one-cycle's power
comes from a huge range-test peak, rebalanced weight decay, and inverse-momentum cycling, and *this*
harness exposes none of them — only a per-epoch `get_lr` float, with momentum and weight decay frozen — so
the faithful port peaks at `base_lr`, keeps a modest `base_lr/25` floor, and cosine-shapes a 30%-up,
70%-down cycle; that leaves two leaks (a sixty-epoch sub-`base_lr` warmup eating the productive middle, a
non-zero floor that never settles the trough), so I expect it to be the weakest rung, worst on the deep
CIFAR-100 net, and I expect the next rung — drop the warmup, anneal smoothly to zero — to close exactly
those two leaks.
