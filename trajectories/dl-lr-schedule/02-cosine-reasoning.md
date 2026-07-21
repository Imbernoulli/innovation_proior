The one-cycle run came in low and told me exactly which parts of it were dead weight: 92.31 on
ResNet-20 / CIFAR-10, 71.57 on ResNet-56 / CIFAR-100, 93.93 on MobileNetV2 / FashionMNIST. One seed, so I
cannot read variance; what I can do is read each number against what its setting can plausibly reach and
against the two leaks I predicted. The two features I flagged — a sixty-epoch sub-`base_lr` warmup, and a
floor that stops at `base_lr/25` instead of zero — are exactly the two ends where a 56-layer net on 100
classes should suffer most. CIFAR-100 with a ResNet-56 and this augmentation typically lands in the
low-to-mid seventies, so 71.57 is at the floor of that band, not the top; the deep net reads as the weakest
of the three relative to its own ceiling, just where I said the missing levers would cost most. CIFAR-10 at
92.31 is the most forgiving setting and even it is unremarkable; the grayscale mobile net at 93.93 is the
weakest on its own scale. Nothing here says the rise-then-fall cosine shape is wrong — it says the warmup
is too long and the floor too high, the two things I was forced to keep because I had no range-test peak to
justify a deep tail and no momentum cycling to make an aggressive cycle safe. So the next move is forced:
strip the warmup, let a single smooth curve start at the full `base_lr` and anneal all the way to zero, and
see whether those two were the whole cost.

Stripping the warmup is a real decision, not a default, so let me check it against the curvature argument
that motivated warmup in the first place — if I am wrong here I break the deep net. The worry was that
`base_lr = 0.1` exceeds `2/lambda_max(H)` at the sharp initial region, where `lambda_max` could sit above
`20`. But one-cycle's evidence cuts the other way: it spent sixty epochs below `base_lr`, gave the
curvature every chance to be protected, and still came in last — which means whatever instability `base_lr`
causes at init on these nets is either mild or transient. Kaiming plus BatchNorm keeps the initial
curvature bounded enough that `0.1` is survivable from epoch 0, even if the first few steps are rough.
Weigh the two costs. If I skip warmup and the start is unstable, I pay for a handful of rough early steps
the rest of the run recovers from; if I keep a sixty-epoch warmup, I pay with half the step-distance of the
most productive third of training, every run, guaranteed. A possibly-rough recovery is far cheaper than a
certain sixty-epoch tax. So I drop warmup and start the curve at `base_lr`. This is a genuine gamble and I
want it falsifiable: if I am wrong and the start truly is unstable on the deepest net, the cosine will show
it as a CIFAR-100 result that fails to clear one-cycle's 71.57 despite the cleaner finish, and I will put a
short warmup back next.

That leaves the shape of a single monotone decay from `base_lr` to a floor, and the floor itself. Take the
floor first, because one-cycle already answered it: stop at `base_lr/25` and you leave the trough
jittering, since the stationary variance scales like the rate, so the only floor that fully removes the
residual is `0`. With no plan to raise the rate again — this is a single descent, not a cycle I restart —
`0` is the natural minimum, the cleanest fit, and it puts the finish strictly below one-cycle's `0.004`, so
if the jitter story is right the FashionMNIST and CIFAR-10 finishes should firm up. Now the shape. A
straight line from `base_lr` to `0` I can rule out by asking where I want to spend the rate budget. Early,
the model is far from any good minimum and the gradient is large and informative — I want to stay high and
make fast progress, and dropping the rate quickly wastes the most productive phase (exactly what one-cycle
wasted at the other end by warming up). Late, I am closing on a minimum and need the rate to come down
slowly and spend a long time small, to fine-tune below the noise floor. So the ideal curve is flat-ish near
the top, falls through the middle, flat-ish near the bottom. A straight line is wrong in both tails: it
leaves the top immediately and reaches the bottom only at the last instant.

Before committing to a fresh curve, the cheaper question: could I just retune one-cycle's three dials —
shorten the warmup, drop the floor to zero via `final_div -> infinity`, keep the two-phase body? That would
fix both leaks with the smallest edit, but it confounds the experiment. If I change warmup length and floor
at once and CIFAR-100 improves, I cannot tell which did it — and the sharpest thing one-cycle told me is
that I do not yet know whether warmup helps at all on these nets. The clean test is the extreme: remove
warmup entirely, take the floor to zero, and read whether the deep net rises or stalls. A ten-epoch warmup
would blur exactly that signal. So I collapse the two-phase shape to a single monotone curve — which also
happens to be the simpler object.

So I write the four conditions the curve should meet: value `base_lr` at progress `0`, value `0` at
progress `1`, zero slope at both ends, monotone decreasing. A linear decay `base_lr*(1 - p)` has slope
`-base_lr` everywhere — both tails wrong. An exponential `base_lr*c^p` is steepest at `p=0` and flattest at
`p=1`, backwards — it abandons the high rate almost immediately and never reaches `0`. A power
`base_lr*(1-p)^k` with `k>1` gives zero slope at the bottom but slope `-k*base_lr` at the top, fixing only
one tail. What has zero slope at both ends and a single steep middle? A half-period of a cosine: over
`0→pi` it goes `+1→-1`, decreasing throughout, derivative `-sin` zero at both ends and most negative at the
midpoint. The affine map sending `+1→base_lr` and `−1→0` is `base_lr*(1 + cos(pi*p))/2`, so
`eta = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))`, `base_lr` at epoch 0, zero at the boundary.
The derivative `-(base_lr*pi/2)*sin(pi*p)` is zero at both ends and `-0.157` at the midpoint, and
`sin(pi*p) > 0` on `(0,1)` so it is strictly monotone — all four conditions with no free coefficient. These
four do not uniquely force the cosine; the cubic smoothstep `1 - (3p^2 - 2p^3)` satisfies them too and
tracks the cosine to about one percent (at `p=1/4`, `0.844` vs `0.854`). But that just means the conditions
pin a family of flat-ended S-curves and the cosine is the parameter-free member — one term, no shape
exponent — so picking the polynomial would only add an arbitrary knob against the goal of few
hyperparameters. I take the half-cosine.

Does starting at `base_lr` actually reclaim the productive middle one-cycle wasted? Put the two side by side
at the same early epochs: epoch 10, cosine `0.0994` vs one-cycle `0.0105` — `9.5×` more step; epoch 20,
`0.0976` vs `0.028`, `3.5×`; epoch 30, `0.0946` vs `0.052`, `1.8×`. Through the whole steep, informative
first thirty epochs the cosine steps two to nine times larger, doing the productive high-rate work
one-cycle postponed, and it spends nothing extra — the only inputs are `base_lr` and the budget.

I want to catch myself before saying "cosine takes bigger steps overall," because summing the curves says
something more precise. Total accumulated `eta` is `10.05` for the cosine and `10.35` for one-cycle —
within three percent, essentially the same total step-distance. The mean rate is `0.0503`, almost exactly
`base_lr/2` (the integral of `0.5*(1+cos)` over a half-period is `0.5`). The two spend the same budget of
travel in different places: the cosine puts about 82% of its travel (`8.21` of `10.05`) into the first
hundred epochs and only `1.84` into the last, front-loading into the steep phase and easing to near-zero at
the end, while one-cycle back-loads the opening behind the warmup and refuses to go all the way down. So the
mechanism is redistribution, not more stepping — same step-distance, allocated where the gradient is large
early and shrunk to nothing late.

There is a deeper way to see why this single cosine is the right second attempt, and it reframes what
one-cycle was. The cosine schedule is the no-restart special case of cosine annealing with warm restarts —
and the restart idea is what one-cycle was a crippled imitation of. A restart resets the acceleration
phase: stop refining in the current basin, explore broadly again. You can emulate that without touching the
momentum vector, just by raising the rate back up — a warm restart that keeps the hard-won velocity and is
scheduled rather than triggered. Within each cycle you anneal the same half-cosine from `eta_max` to
`eta_min`, and you can grow the cycle length for good anytime behavior. One-cycle's rise-then-fall was an
attempt at the restart kick glued to a single anneal, but with the kick capped at `base_lr` and the floor
stuck at `base_lr/25` it got neither the high-rate exploration nor the clean finish. The pure cosine drops
the kick entirely — `T_mult = 1`, one cycle over the whole budget, `eta_max = base_lr`, `eta_min = 0` —
keeping only the part the topology rewards on a fixed budget. And I do not want restarts here: their payoff
is anytime performance, a decent model early and better ones as cycles lengthen, which matters when the
budget is open-ended or I might stop early. My budget is fixed at 200 epochs with one number reported at
the end, so every restart would spend budget re-heating instead of pushing one clean descent as deep as it
goes. `T_mult = 1` is the right specialization, not a reluctant simplification.

Now the edit surface. `get_lr` returns one float per epoch; the loop writes it into every parameter group
and never touches momentum or weight decay. So the rate is constant within an epoch and steps at epoch
boundaries — fine enough for a smooth cosine, since the maximum per-epoch jump is `base_lr*pi/2/200 =
0.00079`, under one part in a hundred of `base_lr`, and only at the steepest point. The function conditions
on nothing in `config`: the same curve serves all three settings, because staying high while the gradient
is informative and then annealing to settle are architecture-agnostic. One subtlety: at `epoch = 199` the
rate is `base_lr*0.5*(1 + cos(pi*199/200)) ≈ 6e-6`, just above `0`, reaching exactly `0` only at the
boundary `epoch = 200` after the last training epoch — so the last trained epoch runs at a tiny but nonzero
settling step, which is what I want.

The delta from the last run is mostly subtraction: where one-cycle spent sixty epochs warming up below
`base_lr` and floored at `base_lr/25`, the cosine starts at the full `base_lr` from epoch 0 and anneals all
the way to `0`. The falsifiable expectations follow from the one-cycle numbers. If the over-long warmup was
the dominant leak, the gain should be largest exactly where warmup hurt most — ResNet-56 / CIFAR-100 — so I
expect cosine to clear 71.57 there by the biggest margin. On ResNet-20 / CIFAR-10 (one-cycle's most
forgiving at 92.31) I expect a clear lift from the full-rate start plus the anneal-to-zero finish. On
MobileNetV2 / FashionMNIST, 93.93 should lift toward the mid-94s as the non-zero floor is removed. The one
place I am genuinely unsure is the start: if dropping warmup re-exposes an init instability on the deep
CIFAR-100 net, I will see it as a CIFAR-100 result that fails to clear 71.57 despite the cleaner finish —
and that would tell me warmup was doing real curvature-protection work, sending me to put a short warmup
back. Does CIFAR-100 rise or stall when warmup is removed — that single comparison is the experiment this
run is designed to run.
