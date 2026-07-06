The uncertainty run confirmed the diagnosis and then drew the next line for me. Learning the weights
gave 66.81 on ResNet-20, 70.94 on ResNet-56, 72.67 on VGG-16-BN. Let me difference these against the
PCGrad row I already have, 64.31 / 70.20 / 74.17, because the per-backbone deltas are where the
mechanism hides. ResNet-20 moved `66.81 − 64.31 = +2.50`; ResNet-56 moved `70.94 − 70.20 = +0.74`; and
VGG moved `72.67 − 74.17 = −1.50`. The headline test passed exactly where I bet it would: ResNet-20
jumped two and a half points on the capacity-scarce backbone, which is the strongest evidence that the
small-backbone collapse really was a *weighting* problem, not a direction problem, and that picking up
the magnitude lever was the right move. ResNet-56 nudged up a little, the modest gain I expected on a
trunk that was less starved — and the *ordering* of the two gains, `+2.50` on the small trunk against
`+0.74` on the deeper one, is itself confirmation, because the recovery is largest exactly where the
capacity contention was tightest. So the magnitude story holds on two of three backbones.

But VGG-16-BN went the *other* way — down `1.50` — and that sign flip is the tell I want to chase.
First let me make sure it actually costs me in aggregate, because a drop on one backbone could be
noise swamped by gains elsewhere. The task score is the geometric mean, so uncertainty scores
`(66.81 · 70.94 · 72.67)^{1/3} ≈ 70.10` against PCGrad's `69.44`, a net `+0.66`. So the move up the
ladder is real but *thin*, and it is thin precisely because the method is not Pareto — it did not lift
all three backbones, it *traded* VGG for ResNet-20. Summed raw, the changes are `+2.50 + 0.74 − 1.50 =
+1.74` points spread across three backbones, and a full point and a half of that is a regression I paid
to buy the ResNet-20 recovery. A method that could recover ResNet-20 *without* giving back VGG would
clear a full point of headroom the current one is leaving on the table, so the VGG regression is not a
cosmetic blemish, it is where the next point of task score is sitting. That makes explaining it the
whole job of this rung.

So why VGG, and why down? On the large-headed VGG the auxiliary signal matters least — big heads, ample
trunk, the coarse task is the least needed of the three backbones — and the `σ → 1` approximation in the
classification derivation is only first-order once the scales drift. But the deeper reason, and the one
that generalizes, is *what* uncertainty weighting learns: a single *static* scale per task, with a
fixed point tied to the loss *level*, `σ_i² = L_i`, that drifts only slowly because it is one scalar
crawling under the same SGD learning rate as ten million network weights. If the fine and coarse tasks
are learning at genuinely different *rates* over the 200-epoch cosine schedule, a slow-moving learned
scale can lag the actual balance the network needs at each phase. And VGG is where that lag should bite
hardest: it masters the easy 20-way coarse task *quickly*, after which the coarse head mostly emits a
stale, well-fit gradient into the trunk, and the level-based weight — which reacts to how big the loss
*is*, not to whether it is still *moving* — does something quietly perverse as the coarse loss falls.
Its weight is `1/σ² → 1/L`, so a *shrinking* coarse loss drives the coarse weight *up*: uncertainty
ramps the mastered coarse task's multiplier higher exactly as that task stops contributing anything
useful, amplifying its stale near-zero-signal gradient into the shared trunk right when it should be
receding. On a big-headed backbone that needed the auxiliary least, spending a growing share of the
loss budget on a dead coarse task is precisely the way to leak a point of fine accuracy. So the
limitation of the uncertainty
rung is now legible: it weights by a learned *level* of each loss, with a fixed point tied to the loss
*value*, and it has almost no notion of the *speed* at which each task is currently improving. The next
lever is rate, not level.

Before I build the rate rule, let me be sure it is the right next move and not just the most obvious
one, because there are cheaper things I could try first. I could keep uncertainty weighting and just
*schedule* the log-variances — anneal them, or re-initialize them — but that is still a level-based
rule wearing a schedule; it does not introduce any signal about how fast each loss is descending, so it
cannot fix a lag that is fundamentally about rate. I could reach again for a gradient-magnitude
balancer of the GradNorm family, which does carry a rate target — but that drags me back into per-task
gradient norms on the shared trunk, which means the autograd-graph walk and the extra backward passes I
was so glad to shed at the uncertainty rung, plus its restoring-force hyperparameter I have no budget to
tune. What I want is a rate signal I can compute from the *only* things the interface hands me cheaply:
the two scalar losses, the epoch counters, and whatever small state I choose to carry between calls. And
that is enough, because the rate of a loss is visible in the loss itself over time — I do not need its
gradient, I only need its value now and its value a moment ago. That is the one piece of information the
`epoch`/`total_epochs` arguments and a stored history buffer let me compute, and the two earlier rungs
threw it away: PCGrad read the instantaneous gradient *geometry*, uncertainty read the instantaneous
loss *level*; neither looked at the loss's *change over time*.

Now derive the form. What I actually care about, for keeping the two tasks balanced, is not how big a
loss is in absolute terms — that is exactly the units-and-scale problem uncertainty weighting already
tried to neutralize — but how *fast* each task is still descending. A task whose loss is plummeting is
learning fine and does not need more weight; a task whose descent has stalled relative to the other is
the one being neglected by the shared trunk and the one I want to *up-weight* so it catches up. The
cleanest scale-free measure of "how fast is task `k` descending" is the ratio of its loss now to its
loss one step ago, `r_k = L_k(now) / L_k(prev)`. A ratio is unit-free, so it already solves the
incommensurability that plagued the bare sum — a 100-way fine cross-entropy sitting near `log 100 ≈ 4.6`
early and a 20-way coarse cross-entropy near `log 20 ≈ 3.0` live at different magnitudes, but their
*ratios* are pure numbers and directly comparable. Read what `r_k` says: `r_k ≈ 1` means task `k`'s loss
is barely moving — it has stalled, it is the slow learner; `r_k < 1` means the loss is still dropping
fast — it is learning well, the fast learner; `r_k > 1` means the loss is *rising*. So a *larger* `r_k`
corresponds to a *slower* (or worsening) learner, and a larger `r_k` is exactly the task I want to give
*more* weight. That monotonicity — bigger ratio gets bigger weight — is the core of the rule. Contrast
uncertainty weighting, where a bigger *loss* got a bigger `σ` and therefore *less* weight; here a bigger
*ratio* gets more weight. The two rules push on different quantities and in the opposite direction,
which is the point, and it is why this rule can catch a task that uncertainty misses. Take the easy
coarse task in the phase where the weighting actually matters — it is descending *fast*, so it carries a
*small* loss and a *low* ratio at the same moment, and the two rules read those two facts in opposite
directions. Uncertainty sees the small loss and drives the coarse weight *up* (`1/L` grows as `L`
shrinks); DWA sees the fast descent (low ratio, well below the harder fine task's ratio) and drives the
coarse weight *down*. On a task I only ever wanted as an auxiliary, DWA's direction is the one that
protects the fine head's share of the trunk, and it never performs uncertainty's runaway up-ramp on a
task that is busy making itself irrelevant.

Let me put numbers on that "opposite direction" so it is not just a slogan. Pick a mid-training moment
where the easy coarse task has descended well while the fine task lags: `L_coarse = 0.8`, `L_fine = 2.2`,
with the coarse loss still dropping fast, `r_coarse = 0.70`, and the fine loss creeping, `r_fine = 0.92`.
Uncertainty weights by level, at fixed point `1/L`, so it wants `w_coarse = 1/0.8 = 1.25` against
`w_fine = 1/2.2 = 0.45` — a coarse-to-fine weight ratio of `2.75`, nearly three times as much budget on
the auxiliary as on the objective I actually care about. DWA weights by rate: `ratios/T = (0.92, 0.70)/2
= (0.46, 0.35)`, `exp` values `1.584` and `1.419` summing to `3.003`, softmax `(0.528, 0.472)`, weights
`(1.055, 0.945)` for `(fine, coarse)` — a coarse-to-fine ratio of `0.90`, *less* budget on the auxiliary
than on the fine task. So at the identical training moment the two rules do not merely differ in degree;
they *reverse the ordering* of the two tasks — `2.75×` on coarse under uncertainty versus `0.90×` under
DWA — and DWA's ordering is the one that keeps the shared trunk working for the head the metric reads.

I need to turn the two ratios `r_fine`, `r_coarse` into two weights, and I have two requirements: the
map should be smooth and monotone (bigger ratio, bigger weight), and the two weights should sum to a
fixed budget so the overall loss scale does not drift out from under the fixed SGD learning rate and
cosine schedule. The natural map from a set of scores to a normalized set is the softmax, and it comes
with a temperature knob that controls how sharply the weighting reacts. So weight task `k` by
`softmax(r_k / T)` over the two tasks, then scale the softmax — which sums to 1 — up by the number of
tasks `K = 2` so the weights sum to `K` and the *mean* weight stays 1. Let me check that budget claim,
since it is the whole reason for the `K` factor. Softmax over two entries always sums to exactly 1, so
`K · softmax` sums to exactly `K = 2`, identical to the equal-weighted sum's `(1, 1)` which also sums to
2 — so the total loss magnitude DWA hands back is, on average, the same magnitude the plain sum hands
back, and the learning rate and schedule keep seeing a sensibly-sized loss. This is a real structural
advantage over the uncertainty rung, whose `1/σ²` weights are unbounded and whose extra `+s` penalty
shifts the loss scale around; DWA is *budget-preserving by construction*. And the per-weight range is
bounded too: each softmax entry lies in `(0, 1)`, so each weight lies in `(0, 2)`, meaning DWA can never
starve a task to zero or blow one up — it reweights only within a bounded band around equal weighting,
which is a feature, not a limitation, on a deep net I do not want to destabilize.

The temperature `T` sets how wide that band is, and I should pin down its two limits before choosing a
value. As `T → ∞`, every `r_k/T → 0`, the softmax flattens to `(0.5, 0.5)`, and `K · softmax → (1, 1)`:
DWA reduces to *equal weighting*, a soft rule that barely reweights. As `T → 0`, the softmax sharpens to
a hard argmax that puts `(1, 0)` on the single largest ratio, and `K · softmax → (2, 0)`: DWA dumps the
entire budget on the slowest task and zeroes the other, which is far too aggressive and would whipsaw a
deep net batch to batch. So I want a middle value, and let me actually compute what a given `T` does
rather than guess. Take a representative mid-training moment where the coarse task is mostly mastered and
still creeping down while the fine task has stalled: say `r_fine = 1.0` (stalled) and `r_coarse = 0.8`
(still descending). At `T = 2.0`, `r_k/T = (0.5, 0.4)`, so `exp(0.5) = 1.649` and `exp(0.4) = 1.492`
sum to `3.141`, the softmax is `(0.525, 0.475)`, and the weights are `(1.050, 0.950)` — a gentle nudge
toward the stalled fine task, exactly the direction I want, and small enough not to jolt the trunk.
Drop to `T = 0.5` and the same ratios give `r_k/T = (2.0, 1.6)`, `exp` values `7.389` and `4.953`
summing to `12.342`, softmax `(0.599, 0.401)`, weights `(1.197, 0.803)` — noticeably sharper, starting
to press. Push to `T = 100` and I get `r_k/T = (0.01, 0.008)`, weights `(1.001, 0.999)` — essentially
equal weighting, confirming the `T → ∞` limit numerically. So `T = 2.0` is the conservative middle:
enough contrast to favor the lagging task by a few percent, not so much that it starves the other, and
the numeric trace shows the reweighting it produces is a light touch that stays close to the equal-sum
budget while still leaning toward whichever task has stalled. One more property falls out of the trace
that I like: if the two ratios are *equal*, say both `0.9`, the softmax is `(0.5, 0.5)` and the weights
are `(1, 1)` regardless of the common value — DWA responds only to the *gap* between the two descent
rates, not to their absolute level, which is precisely the scale-freedom I was after and another way of
seeing that it is a genuinely different signal from the level-based uncertainty rule.

There is a bootstrap question — what are the weights on the very first update, when there is no
"previous" loss to form a ratio against? On the first step (or whenever no history exists yet) I have no
rate information, so I fall back to the most neutral choice: equal weights, `w_fine = w_coarse = 1`,
which is also exactly the `T → ∞` limit, so the method starts life *as* the equal sum and only departs
once it has a rate to read. After that, each step I read the current losses, form the ratios against the
stored previous losses, compute the softmax weights, then overwrite the stored previous losses with the
current ones for next time. That is the whole state: one buffer of the previous losses, plus the fixed
`T`.

One subtlety I have to get exactly right, and it is where this differs from a textbook description of the
method — and it is a real design fork, not a formality. In the original formulation the ratio is taken
between two *epoch-averaged* losses, `r_k(t) = L̄_k(t−1)/L̄_k(t−2)`, so the rate is smoothed over a whole
epoch before it drives the weight, and the weight is held fixed across that epoch. To reproduce that
here I would have to accumulate a running sum of each loss and a sample count over the epoch, watch the
`epoch` argument for a boundary, compute the two epoch means, form the ratio only at the boundary, and
hold the resulting weight fixed for the next epoch — more state (two accumulators, a count, a last-epoch
marker, two held weights) and a weight that updates only 200 times over the whole run, once per epoch,
which is very coarse. The alternative the interface makes natural is to keep the single cheapest piece of
state, the loss from the *previous call*, and form the ratio `L_k(this batch) / L_k(previous batch)`: a
*per-batch* rate, recomputed and the buffer overwritten every step. I want to be honest that this is a
noisier signal than the epoch-averaged version — batch-to-batch loss ratios fluctuate with minibatch
composition, and if the per-batch loss has some coefficient of variation, the ratio of two consecutive
batches fluctuates by roughly `√2` times that. But two things make me take the per-batch form anyway.
The `T = 2.0` softmax damps the noise heavily — I just saw it turn a `0.2` gap in the raw ratios into a
five-percent nudge in the weights, and it squashes fluctuations the same way — and the buffer's constant
overwriting means the rate always reflects the *most recent* descent, the freshest possible read on
which task is currently lagging, where the epoch-averaged form is always a full epoch stale. The `epoch
== 0` guard pins the weights to uniform for the entire first epoch, so the rate signal only switches on
once the losses have a meaningful recent history and the network is past its most chaotic phase. These
are the concrete shapes the edit takes, and they are not the canonical epoch-loop form — they are what
the per-batch scalar interface allows, and I am making the tradeoff with my eyes open: I am betting the
freshness plus the softmax damping beats the epoch-averaged smoothing, and if I am wrong the failure
will be legible as noise swamping the signal.

The last detail is keeping the gradient honest, and it is a correctness point I want to actually work
out rather than assert. The returned loss is `Σ_k w_k L_k`. If I computed `w_k` from the *live* losses
— left them attached to the graph — then `w_k` would be a function of `L_k(now)`, which is a function of
the network parameters `θ`, and differentiating the product would give
`∂/∂θ (w_k L_k) = w_k ∂L_k/∂θ + L_k ∂w_k/∂θ`. That second term is poison: `w_k` runs through the
softmax and the ratio, and `∂w_k/∂L_k(now) = ∂w_k/∂r_k · (1/L_k(prev))` is plainly nonzero, so the
network would receive a gradient telling it to move `θ` so as to *game the weighting rule* — for
instance to inflate its own current loss to grab a larger `r_k` and thus a larger weight — a perverse
incentive that has nothing to do with lowering either loss. The cure is to compute the weights from the
*detached* losses (`ratios = losses.detach() / (prev + 1e-8)`, so the softmax is a function of
constants) and to store the buffer detached, so no gradient ever flows *through* the weighting
computation: `∂w_k/∂θ = 0` by construction, the second term vanishes, and
`∂/∂θ (Σ_k w_k L_k) = Σ_k w_k ∂L_k/∂θ` — the honest weighted gradient with `w_k` a constant multiplier
at each step. That is correct precisely because DWA is a *scheduling* rule on the loss weights, not a
learnable parameter like uncertainty's log-variances; it registers nothing in the optimizer, it is pure
state (a previous-loss buffer) plus arithmetic, and the `1e-8` floor on the denominator keeps the ratio
finite if a previous loss ever underflows toward zero. This is again simpler than PCGrad's machinery and
even lighter than uncertainty's two learnable scalars — no parameters at all, just a buffer and a
softmax. (The full scaffold module is in the answer.)

So the delta across the three rungs is a clean progression of *what signal drives the weighting*: PCGrad
used the instantaneous gradient geometry and ignored magnitude; uncertainty used the instantaneous loss
level and learned a slow static scale; DWA uses the recent loss *rate of change* and pushes weight
toward whichever task is descending more slowly, recomputed every step. It directly targets the
limitation the uncertainty run exposed — a static, slow-drifting weight that cannot track tasks learning
at different speeds — by making the weight a fast, bounded, budget-preserving function of the live
descent rate.

Now the falsifiable expectations against the numbers I have. Uncertainty was 66.81 / 70.94 / 72.67,
gmean 70.10; PCGrad was 64.31 / 70.20 / 74.17, gmean 69.44. My sharpest claim is on **VGG-16-BN**,
because that is where I located the whole failure: the uncertainty rung *regressed* there to 72.67, and
if that regression was caused by uncertainty ramping the coarse weight *up* as the coarse loss fell,
then a rate-driven weight — which reads the coarse task's fast descent as a low ratio and holds its
share *down*, never performing that runaway up-ramp, and returning to neutral only once both tasks stall
together — should *recover* VGG. I expect it back above uncertainty's 72.67, and plausibly at or above
PCGrad's 74.17, since 74.17 is roughly what the coarse task's *un*-over-weighted contribution looked
like on that backbone. If
DWA does *not* lift VGG, my "rate vs level" story is wrong and I will have to conclude the VGG drop was
something other than lag. On **ResNet-20** I expect a further small gain over uncertainty's 66.81 — the
rate signal should help the starved small trunk allocate weight to whichever task is currently lagging,
on top of the level-based recovery uncertainty already bought, so I would be disappointed if it did not
clear 67 — though ResNet-20 is also where I most fear the per-batch noise, being the smallest model with
the least averaging in its features, so if any backbone shows the ratio noise swamping the signal it is
this one. On **ResNet-56** I expect another modest step over 70.94, the same "deep trunk was less
starved, less to gain" logic as before. If the pattern comes out as "VGG recovered, ResNet-20 up past
67, ResNet-56 modestly up" — a Pareto improvement over uncertainty rather than uncertainty's trade — that
confirms the *rate* of learning is the signal this hierarchical fine/coarse task most rewards, and DWA
sits at the top of the ladder. If instead the per-batch noise in the ratio swamps the signal and DWA
lands no better than uncertainty, the read would be that the epoch-averaged form the per-batch interface
cannot cheaply express is what actually matters, and the static learned weight is as far as a
scalar-loss rule can climb here.
