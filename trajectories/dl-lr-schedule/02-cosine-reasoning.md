The one-cycle run came in low and it told me exactly which parts of it were dead weight. On ResNet-20 /
CIFAR-10 it managed 92.31, on ResNet-56 / CIFAR-100 it managed 71.57, on MobileNetV2 / FashionMNIST it
managed 93.93. Read those against what the schedule actually did and the diagnosis is sharp. The two
features I flagged as leaks — a sixty-epoch sub-`base_lr` warmup, and a floor that stops at `base_lr/25`
instead of zero — are precisely the two ends where a 56-layer net on 100 classes should suffer most, and
71.57 is the lowest of the three relative to what the setting can reach. The schedule spent the first 30%
of training, sixty epochs, crawling at a fraction of 0.1 — climbing from 0.004 and only touching the full
rate at epoch 60 — so the most productive high-rate phase, the one that does the bulk of the work on a
deep net, was eaten by an over-long warmup. And it never annealed below 0.004, so the trough was never
fully settled: there is residual minibatch jitter at the finish that a schedule going to zero would have
removed. CIFAR-10 at 92.31 is the most forgiving setting and even it is unremarkable; the grayscale mobile
net at 93.93 is the weakest of the three on its own scale. Nothing here says the rise-then-fall *cosine
shape* is wrong. It says the warmup is too long and the floor is too high — the two things the harness
forced me to keep because I had no range-test peak to justify a deep tail and no momentum cycling to make
an aggressive cycle safe. So the next move is forced: strip the warmup entirely, let a single smooth curve
start at the full `base_lr` and anneal all the way to zero, and see whether those two amputations were the
whole cost.

Stripping the warmup is a real decision, not a default, so let me check it against the curvature argument
that motivated warmup in the first place. The worry was that `base_lr = 0.1` exceeds the admissible step
`2/lambda_max(H)` at the sharp initial region, so the first steps overshoot. But one-cycle's evidence cuts
the other way: it spent sixty epochs below `base_lr` and still underperformed, which means whatever
instability `base_lr` causes at init on *these* nets is either mild or transient — Kaiming init plus
BatchNorm keeps the initial curvature bounded enough that 0.1 is survivable from epoch 0. A handful of
possibly-rough early steps that recover is a far cheaper price than sixty epochs of under-stepping. So for
the second rung I drop warmup and start the curve at `base_lr`. If I am wrong and the start truly is
unstable, the cosine will show it as a CIFAR-100 collapse and I will put a short warmup back at the next
rung — but the one-cycle numbers strongly suggest the warmup was the leak, not the cure.

That leaves the shape of a single monotone decay from `base_lr` to a floor, and the floor itself. Take the
floor first because one-cycle already answered it: stop at `base_lr/25` and you leave the trough jittering;
the late-training argument is that I need the step small enough to settle below the stochastic-gradient
noise floor, and the only floor that fully removes that residual is zero. With no plan to ever raise the
rate again, zero is the natural minimum — the smallest possible steps at the very end, the cleanest fit.
So this rung anneals to 0. Now the shape. The crude smooth option is a straight line from `base_lr` to 0,
and I can rule it out by thinking about where I want to spend the rate budget. Early in the run the model
is far from any good minimum and the gradient is large and informative — I want to *stay* high here and
make fast progress, and dropping the rate quickly would waste the most productive phase (exactly the phase
one-cycle wasted at the *other* end, by warming up). Late in the run I am closing on a minimum and need
the rate to come down slowly and spend a long time *small*, to fine-tune below the noise floor. So the
ideal curve is flat-ish near the top (linger high), falls through the middle, and flat-ish near the bottom
(linger low). A straight line is wrong in both tails: it leaves the top immediately and reaches the bottom
only at the last instant. I want a curve whose slope is near zero at both ends and steepest in the middle.

What function has zero slope at both ends of an interval and a single steep middle? A half-period of a
cosine. Over the interval where its argument runs 0→pi, cosine goes from +1 to −1, decreases the whole
way, and its derivative −sin is zero at both ends and most negative at the midpoint. Map the run onto that:
let progress go 0→1 across the 200 epochs, feed `pi*progress` into cosine to get a value running +1→−1,
and rescale that to `[base_lr, 0]`. The affine map sending +1→`base_lr` and −1→0 is
`base_lr*(1 + cos(pi*progress))/2`: at progress 0 it is `base_lr`, at progress 1 it is 0. So the schedule
is `eta = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))`, a single half-cosine from the full rate
at epoch 0 to zero at the budget boundary. Let me verify the slope claim, since it is the whole reason I
picked cosine over a line: the derivative in progress is `-(base_lr*pi/2) * sin(pi*progress)`, which is
zero at progress 0 and 1 (the rate barely changes near the top and the bottom — linger high, then linger
low) and most negative at progress 1/2 (the fast decay happens in the middle). That is exactly the "stay
high early, settle low late" profile, and it needs no milestones — the only inputs are `base_lr` and the
budget, both of which I already have. No drop epochs, no drop factor, no warmup length.

I want to be honest that the four conditions I am leaning on — value `base_lr` at the start, 0 at the end,
zero slope at both — do not *uniquely* force the cosine; a cubic smoothstep satisfies them too. But the
cosine is the parameter-free member of that family — one term, `cos(pi*progress)`, no shape exponent to
pick, no degree to choose — and it is the canonical way to interpolate between two levels with zero slope
at both ends. Picking a polynomial would just be choosing an equivalent S-curve with an extra arbitrary
knob attached, against the stated goal of *few* hyperparameters. So I take the half-cosine.

There is a deeper way to see why this single cosine is the right second rung, and it reframes what
one-cycle was. The cosine schedule is the no-restart special case of cosine annealing *with warm restarts*
— and the restart idea is what one-cycle was a crippled imitation of. In momentum optimization a restart
is a reset of the acceleration phase: stop refining in the current basin, explore broadly again. You can
emulate that without touching the momentum vector at all, just by raising the learning rate back up — a
*warm* restart that keeps the hard-won velocity and is *scheduled* rather than triggered, so it needs no
noisy stochastic test to decide when to fire. Within each run between restarts, you anneal along the same
half-cosine from a high `eta_max` to a low `eta_min`, and you can grow the run length across restarts for
good anytime behavior. One-cycle's "rise then fall" was an attempt at the restart kick (the up-leg) glued
to a single anneal, but with the kick capped at `base_lr` and the floor stuck at `base_lr/25` it got
neither the genuine high-rate exploration nor the clean low-rate finish. The pure cosine drops the kick
entirely — `T_mult = 1`, one cycle over the whole budget, `eta_max = base_lr`, `eta_min = 0` — and keeps
only the part the topology actually rewards on a fixed budget: a smooth coarse-to-fine anneal that starts
at the full rate and ends at zero. So I am not abandoning one-cycle's idea; I am taking the half of it
that the harness can actually run well and discarding the half (the aggressive kick) that the harness
neutered.

Now make this concrete in the edit surface, because the constraints decide what is even expressible. I
edit only the body of `get_lr`, which returns one float per epoch; the loop calls it with the integer
`epoch`, writes the float into every parameter group, and never touches momentum or weight decay. So the
schedule is sampled once per epoch — within an epoch the rate is constant and it steps at epoch
boundaries. Is that granularity fine enough for a smooth cosine? The curve is sampled at 200 points across
the run; the cosine's maximum slope is `base_lr*pi/2` spread over 200 epochs, so each epoch-to-epoch change
is on the order of `base_lr/100` — tiny, no shock. A per-batch evaluation would be even smoother but the
pipeline gives me per-epoch and it is more than enough. The function conditions on nothing in `config` —
the same curve serves ResNet-20 / CIFAR-10, ResNet-56 / CIFAR-100, and MobileNetV2 / FashionMNIST, because
the two things the cosine does (stay high while the gradient is informative, then anneal smoothly to
settle) are architecture-agnostic; I would only reach for `arch`/`dataset`-specific tweaks if a particular
setting showed it needed them, and the point of this curve is to not need them. One subtlety worth naming:
at `epoch = total_epochs - 1` the rate is `base_lr*0.5*(1 + cos(pi*199/200))`, just above 0 — it reaches
exactly 0 at the boundary *after* the last training epoch, the usual scheduler convention, so the last
trained epoch runs at a tiny but nonzero rate, which is what I want for the final settling step. (The full
schedule body is in the answer.)

So the delta from rung one is concrete and it is mostly subtraction: where one-cycle spent sixty epochs
warming up below `base_lr` and floored at `base_lr/25`, the cosine starts at the full `base_lr` from epoch
0 and anneals all the way to 0, with the same smooth flat-topped, flat-bottomed S-shape but spanning the
whole run as one descending half-period. The falsifiable expectations follow directly from the one-cycle
numbers. If the over-long warmup was the dominant leak, the gain should be largest exactly where I argued
warmup hurt most — ResNet-56 / CIFAR-100 — so I expect cosine to clear one-cycle's 71.57 there by the
biggest margin, recovering the productive high-rate epochs the warmup ate. On ResNet-20 / CIFAR-10, where
one-cycle was most forgiving at 92.31, I expect a clear lift too — the full-rate start plus the
anneal-to-zero finish should both help on the easy net — and I would not be surprised to see CIFAR-10
become cosine's *best-relative* setting, since a shallow 10-class net rewards a clean smooth anneal cheaply.
On MobileNetV2 / FashionMNIST, one-cycle's 93.93 should lift toward the mid-94s as the non-zero floor that
left jitter at the finish is removed. The one place I am genuinely unsure is the *start*: if dropping
warmup re-exposes an init instability on the deep CIFAR-100 net, I will see it as a CIFAR-100 result that
*fails* to clear 71.57 despite the cleaner finish — and that would tell me the warmup was doing real
curvature-protection work after all, sending me to put a short warmup back. That single comparison — does
CIFAR-100 rise or stall when warmup is removed — is the experiment this rung is designed to run.

The causal chain in one breath: one-cycle's {92.31, 71.57, 93.93} confirmed its two leaks — a
sixty-epoch sub-`base_lr` warmup eating the productive middle and a `base_lr/25` floor that never settles
the trough, worst on the deep CIFAR-100 net → so strip both: start at the full `base_lr` (the long warmup
underperformed, so init instability on these BatchNorm nets is mild enough to skip) and anneal to 0 (the
only floor that removes the noise-floor jitter) → choose the shape from where the rate budget should go —
linger high (informative gradients), fall through the middle, linger low (settle the minimum) — which the
half-cosine `base_lr*0.5*(1+cos(pi*epoch/total))` matches exactly with zero slope at both ends and
steepest descent at the midpoint, a parameter-free member of the flat-ended S-curve family → recognize
this as the no-restart special case of warm-restart cosine annealing, keeping the part of one-cycle's
restart idea the harness can run (the smooth anneal) and dropping the part it neutered (the capped kick) →
expect the biggest gain on ResNet-56 / CIFAR-100, a clear lift on CIFAR-10, and the FashionMNIST floor
jitter removed, with the one open risk being a re-exposed init instability on the deep net that would
argue for putting warmup back.
