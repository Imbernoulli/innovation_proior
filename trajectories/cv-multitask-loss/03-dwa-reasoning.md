The uncertainty run confirmed the diagnosis and then drew the next line for me. Learning the weights
gave 66.81 on ResNet-20, 70.94 on ResNet-56, 72.67 on VGG-16-BN. The headline test passed exactly where
I bet it would: ResNet-20 jumped from PCGrad's 64.31 to 66.81 — a clear two-and-a-half point recovery on
the capacity-scarce backbone — which is the strongest evidence that the small-backbone collapse really was
a *weighting* problem, not a direction problem, and that picking up the magnitude lever was the right move.
ResNet-56 nudged up a little (70.20 → 70.94), the modest gain I expected on a trunk that was less starved.
But VGG-16-BN went the *other* way — 74.17 down to 72.67 — and that drop is the tell I want to chase. On
the large-headed VGG the auxiliary signal matters least, the `σ → 1` approximation in the classification
derivation is only first-order, and, more importantly, the thing uncertainty weighting learns is a single
*static* scale per task that drifts only slowly toward its `σ_i² = L_i` fixed point. If the fine and coarse
tasks are learning at genuinely different *rates* over the 200-epoch cosine schedule, a slow-moving learned
scale can lag the actual balance the network needs at each phase — and on VGG, where the coarse task is
mastered quickly and then mostly contributes a stale, mis-sized gradient, that lag costs accuracy. So the
limitation of the uncertainty rung is now legible: it weights by a learned *level* of each loss, with a
fixed point tied to the loss *value*, and it has almost no notion of the *speed* at which each task is
currently improving. The next lever is rate, not level.

Let me make "rate" precise, because that is the whole design. What I actually care about, for keeping the
two tasks balanced, is not how big a loss is in absolute terms — that is exactly the units-and-scale problem
uncertainty weighting already tried to neutralize — but how *fast* each task is still descending. A task
whose loss is plummeting epoch over epoch is learning fine and does not need more weight; a task whose loss
has plateaued, whose descent has stalled relative to the other, is the one being neglected by the shared
trunk and the one I want to *up-weight* so it catches up. So I want a weighting that reads the recent
*trajectory* of each loss and pushes weight toward whichever task is currently the slower learner. That is a
fundamentally different signal from both prior rungs: PCGrad read the instantaneous gradient *geometry*,
uncertainty read the instantaneous loss *level*; this reads the loss's *change over time*. It is the one
piece of information the `epoch`/`total_epochs` arguments and a stored history buffer let me compute, and
the two earlier rungs threw it away.

Now derive the form. The cleanest scale-free measure of "how fast is task `k` descending" is the ratio of
its loss now to its loss one step ago: `r_k = L_k(now) / L_k(prev)`. A ratio is unit-free, so it already
solves the incommensurability that plagued the bare sum — a 100-way fine cross-entropy and a 20-way coarse
cross-entropy live at different magnitudes, but their *ratios* are directly comparable. Read what `r_k`
says: `r_k ≈ 1` means task `k`'s loss is barely moving — it has stalled, it is the slow learner; `r_k < 1`
means the loss is still dropping fast — it is learning well, the fast learner; `r_k > 1` means the loss is
*rising*. So a *larger* `r_k` corresponds to a *slower* (or worsening) learner, and a larger `r_k` is
exactly the task I want to give *more* weight. That monotonicity — bigger ratio gets bigger weight — is the
core of the rule. (Contrast uncertainty weighting, where a bigger *loss* got a bigger `σ` and therefore
*less* weight; here a bigger *ratio* gets more weight. The two rules push on different quantities and in the
opposite direction, which is the point.)

I need to turn the two ratios `r_fine`, `r_coarse` into two weights that sum to a fixed budget so the
overall loss scale does not drift. The natural map from a set of scores to a normalized set of weights is
the softmax, and it has a temperature knob that controls how sharply the weighting reacts. So weight task
`k` by `softmax(r_k / T)` over the two tasks, then scale the softmax (which sums to 1) up by the number of
tasks `K = 2` so the weights sum to `K` and the *mean* weight stays 1 — that keeps the total loss roughly
the same magnitude as the equal-weighted sum, so the fixed SGD learning rate and cosine schedule still see
a sensibly-sized loss. The temperature `T` controls the contrast: as `T → ∞` the softmax flattens toward
uniform and DWA reduces to equal weighting (weights → 1 each), so a large `T` is a *soft* version that
barely reweights; as `T → 0` it becomes a hard argmax that dumps almost all the budget on the single
slowest task, which is too aggressive and unstable on a deep net. `T = 2.0` is the conservative middle the
method uses — enough contrast to favor the lagging task, not so much that it starves the other. So the
weight is `w_k = K · softmax(r_k / T)`, with the ratios as the only inputs.

There is a bootstrap question — what are the weights on the very first update, when there is no "previous"
loss to form a ratio against? On the first step (or whenever no history exists yet) I have no rate
information, so I fall back to the most neutral choice: equal weights, `w_fine = w_coarse = 1`. After that,
each step I read the current losses, form the ratios against the stored previous losses, compute the
softmax weights, then overwrite the stored previous losses with the current ones for next time. That is the
whole state: one buffer of the previous losses, plus the fixed `T`.

One subtlety I have to get exactly right, and it is where this differs from a textbook description of the
method. In the original formulation the ratio is taken between two *epoch-averaged* losses, `w_k(t−1) =
L_k(t−1)/L_k(t−2)`, so the rate is smoothed over a whole epoch before it drives the weight, and the weight
is held fixed across that epoch. But the interface here is invoked *per batch* — `forward` runs every
minibatch — and the only per-call state I can cheaply keep is the loss from the *previous call*. So in this
task's implementation the ratio is `L_k(this batch) / L_k(previous batch)`: a *per-batch* rate, recomputed
and the buffer overwritten every step, not a per-epoch one. I want to be honest that this is a noisier
signal than the epoch-averaged version — batch-to-batch loss ratios fluctuate with minibatch composition —
but the `T = 2.0` softmax damps that noise heavily (it is a soft, near-uniform reweighting), and the
buffer's constant overwriting means the rate always reflects the *most recent* descent, which is the
freshest possible read on which task is currently lagging. The `epoch == 0` guard pins the weights to
uniform for the entire first epoch, so the rate signal only switches on once the losses have a meaningful
recent history. These are the concrete shapes the edit takes, and they are not the paper's epoch-loop
form — they are what the per-batch scalar interface allows.

The last detail is keeping the gradient honest. The weights are computed from the *detached* losses
(`ratios = losses.detach() / (prev + 1e-8)`, so the softmax is a function of constants), and the buffer
stores detached losses, so no gradient ever flows *through* the weighting computation — `w_k` is treated as
a constant multiplier at each step, and the gradient flows only through the live `w_k · L_k` product into
the network and heads. That is correct: DWA is a *scheduling* rule on the loss weights, not a learnable
parameter like uncertainty's log-variances, so it registers nothing in the optimizer; it is pure state
(a previous-loss buffer) plus arithmetic. This is again simpler than PCGrad's machinery and even lighter
than uncertainty's two learnable scalars — no parameters at all, just a buffer and a softmax. (The full
scaffold module is in the answer.)

So the delta across the three rungs is a clean progression of *what signal drives the weighting*: PCGrad
used the instantaneous gradient geometry and ignored magnitude; uncertainty used the instantaneous loss
level and learned a slow static scale; DWA uses the recent loss *rate of change* and pushes weight toward
whichever task is descending more slowly, recomputed every step. It directly targets the limitation the
uncertainty run exposed — a static, slow-drifting weight that cannot track tasks learning at different
speeds — by making the weight a fast function of the live descent rate.

Now the falsifiable expectations against the numbers I have. Uncertainty was 66.81 / 70.94 / 72.67; PCGrad
was 64.31 / 70.20 / 74.17. My sharpest claim is on **VGG-16-BN**: the uncertainty rung *regressed* there to
72.67, and if that regression was caused by a stale, slow-drifting static weight on a quickly-mastered
coarse task, then a rate-driven weight that backs off the coarse task as soon as its loss stops dropping
should *recover* VGG — I expect it back above uncertainty's 72.67 and plausibly at or above PCGrad's 74.17.
If DWA does *not* lift VGG, my "rate vs level" story is wrong. On **ResNet-20** I expect a further small gain
over uncertainty's 66.81 — the rate signal should help the starved small trunk allocate weight to whichever
task is currently lagging, on top of the level-based recovery uncertainty already bought, so I would be
disappointed if it did not clear 67. On **ResNet-56** I expect another modest step over 70.94, the same
"deep trunk was less starved, less to gain" logic as before. If the pattern comes out as "VGG recovered,
ResNet-20 up past 67, ResNet-56 modestly up," that confirms the *rate* of learning is the signal this
hierarchical fine/coarse task most rewards, and DWA sits at the top of the ladder; if instead the per-batch
noise in the ratio swamps the signal and DWA lands no better than uncertainty, the read would be that the
epoch-averaged form the per-batch interface cannot express is what actually matters, and the static learned
weight is as far as a scalar-loss rule can climb here.
