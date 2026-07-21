The dueling head told me exactly what I suspected and where its limit is, and it told me in the
LunarLander column. CartPole sat at 500 on two of three seeds and 461.6 on the third — pinned, as
expected, since the cap was already reached by plain DQN and the architecture only had to hold it.
Acrobot came in at a mean of −82.6, right in the low −80s I guessed, essentially on par with what a
plain value head gets — also as expected, because Acrobot's reward is a dense negative time-to-goal where
the action matters at almost every step, so the value/advantage split has very little redundancy to
exploit. The signal is LunarLander: mean 89.06, and the per-seed numbers are the whole story —
{127.35, 229.07, **−89.25**}. One seed found a genuinely good policy (229), one found a mediocre one
(127), and one fell into the crash basin and came out *negative*. That −89.25 seed is the classic
failure tell: the agent's greedy policy on that seed is systematically landing in the deceptive crash
region, and the mean is being dragged down by it.

Let me read those three LunarLander numbers as mechanically as I can, because the shape of the failure
should dictate the fix. The spread {127.35, 229.07, −89.25} has a range of 318.32 between best and worst
seed — enormous against a mean of 89.06; the seed-to-seed standard deviation is about 133, larger than
the mean itself, so this is not "mediocre on average," it is *bimodal across seeds*. Pull the one crash
seed out and the remaining two average 178.2, so that single basin seed drags the reported mean down by
roughly 89 points — it nearly halves it. And the drag is localized: CartPole and Acrobot are essentially
flat across all three seeds (Acrobot mean −82.6; CartPole 487.2, with only the 461.6 seed off the cap),
so seed 456 did not learn a uniformly worse agent — it learned one specific bad behavior on one specific
task. That localization is a strong hint about what to change. The fix should leave CartPole and Acrobot
alone, where nothing is broken, and target precisely whatever lets LunarLander's greedy policy commit to
the crash region on some seeds and not others. A change that improved the mean estimate uniformly
everywhere would be the wrong shape; I want something that specifically stops the argmax from being
fooled on the one task whose return is bimodal. The dueling architecture did
its job — it improved how the *mean* state value is estimated and shared across actions — but the failure
that survives is not about how I estimate the mean. It is that I am estimating *only* a mean, and on
LunarLander the return is sharply bimodal: a safe landing scores a few hundred, a crash scores a large
negative, and a single scalar `Q = E[Z]` averages those two worlds into a number that corresponds to
neither and gives the greedy argmax no way to tell "this action has a high mean because it sometimes
lands and sometimes crashes" from "this action has a high mean because it reliably lands." That is the
seed-456 failure in one sentence. So the next move is not a better head for the mean — it is to stop
collapsing the return to its mean at all.

Let me sit with that. We train the agent to maximize expected return, and so we learn `Q = E[Z]`, where
`Z(s,a)` is the actual random return from `(s,a)`. The Bellman recursion the scalar obeys is
`Q(s,a) = E R + γ E Q(s',a')`, a `γ`-contraction in `L∞`, unique fixed point — clean, and it is exactly
what the dueling head still relied on. But `Z` itself is a random variable, and on LunarLander it is
visibly bimodal. The scalar throws that away. In supervised learning I would never hesitate to model the
full conditional distribution when I can — it strictly carries more information. The catch in RL is that
there are no given targets; I bootstrap, "learning a guess from a guess," so the question is whether the
*machinery* survives in distribution space. Three things make mean-value learning work: a Bellman
recursion the object obeys, a contraction so iterating it converges, and a loss I can train from sampled
transitions by SGD. I need all three in distribution space, and the representation and loss should fall
out of what the math will and will not allow.

The recursion is the easy part. Peel the first reward off the return: `Z(s,a) =D R(s,a) + γ Z(s',a')`,
an equality *in distribution* — the law of the left side equals the law of the compound variable on the
right, built from the reward, the random next state-action, and the next return. So a distributional
Bellman operator exists; it scales the next-state distribution by `γ` (shrinking it toward 0), shifts by
the reward, and mixes over transitions. Now the contraction, and here is the subtlety that decides the
whole design: in *which metric* does this operator contract? My deep-learning instinct is KL — that is
what a softmax minimizes. But watch the `γ`-scaling alone. Take two distributions on `{1, 2}`; scale
both by `γ = 0.5` to `{0.5, 1}`. The probabilities are untouched, so KL is exactly unchanged — no
contraction. Worse, two distributions on *disjoint* supports stay KL-infinite (or total-variation 1) no
matter how much I shrink them toward 0. KL, total variation, Kolmogorov sup-CDF distance are all
*vertical* distances — they compare mass at matched locations and are blind to how far apart the
locations are. But the Bellman update is a *horizontal* operation. So none of the metrics I know how to
minimize with a softmax can even see the contraction. The metric that does see it is Wasserstein:
`d_p(F,G) = (∫₀¹ |F⁻¹(u) − G⁻¹(u)|ᵖ du)^{1/p}`, a transport distance — how far mass has to slide. It is
finite for disjoint supports and it scales: multiply the variables by `γ` and the transport multiplies
by `γ`. In the maximal Wasserstein metric `d̄_p = sup_{s,a} d_p` the distributional operator is a
`γ`-contraction (the common reward shift drops out, the `γ` scaling factors out, the mixture over
successors is bounded by the worst case), with the true return distribution as its unique fixed point.
So the metric the theory loves is Wasserstein — and the metric I can minimize with a softmax is not.

Now the wall. Can I just minimize Wasserstein between my predicted distribution and the bootstrapped
target, from sampled transitions, by SGD? No — and this is a real theorem, not a nuisance. The sample
gradient of Wasserstein is *biased*: if I form an empirical target distribution from samples and minimize
the sample `W_p`, the minimizer of the expected sample loss is not the minimizer of the true `W_p`. The
intuition is that `W_p` is built from the quantile function `F⁻¹`, and a single sample is a draw, not an
observation of a quantile; the optimal transport reshuffles which sample pairs with which prediction, and
the gradient of that matching, averaged over sample sets, does not equal the population transport
gradient. So the metric the theory wants is the one I cannot descend from single transitions. Let me see
whether I can find a representation that is genuinely Wasserstein-aware *and* trainable from samples — and
ideally one whose support is not fixed in advance but slides to wherever the returns actually live, since
these three tasks span wildly different return ranges and I would rather not hand-set a range per task.

Here is the lever. Wasserstein is built entirely from the quantile function `F⁻¹` — the inverse CDF, a
map from probability level to value — so if I want a representation natively aligned with the metric the
operator actually contracts in, its free parameters should be *points on that curve*: values indexed by
probability level, not probabilities indexed by value. That is the transpose of an ordinary histogram.
Fix the probabilities to be uniform, `q_i = 1/N`, and make the *locations* `θ_i` — the values at those
levels — the learnable thing:
`Z_θ(s,a) = (1/N) Σ_i δ_{θ_i(s,a)}`. I am no longer learning how much mass sits at fixed heights; I am
learning *where* `N` equal lumps of mass should sit. And "where the `i`-th of `N` equal lumps sits" is
exactly a **quantile** of the distribution. So this transposed parametrization estimates quantiles of
the return. Three things immediately look better: the support is not pinned in advance — the locations
slide to wherever the returns actually live, with per-state adaptive resolution (this matters here, where
LunarLander spans roughly −400 to +300 and Acrobot −500 to −60 and CartPole 0 to 500, three very
different ranges that any single prearranged range would have to straddle); the Bellman target's shifted
atoms are just numbers I compare directly against my locations, with no intermediate step needed to
realign them; and — the part I must verify — estimating quantiles may be doable from samples *without* a
biased gradient.

Which quantiles? Minimize `W_1` between an arbitrary target `Y` and a uniform-`N`-Dirac distribution on
ordered locations `θ_1 ≤ … ≤ θ_N`. With cumulative levels `τ_i = i/N`, the inverse-CDF of the uniform
comb is the staircase equal to `θ_i` on `(τ_{i-1}, τ_i]`, so
`W_1 = Σ_i ∫_{τ_{i-1}}^{τ_i} |F_Y⁻¹(ω) − θ_i| dω`. The cells decouple — each `θ_i` appears in only its
own integral — so minimize each separately. The subgradient in `θ` of `∫_τ^{τ'} |F⁻¹(ω) − θ| dω` is
`(F(θ) − τ) − (τ' − F(θ)) = 2F(θ) − (τ + τ')`, zero when `F(θ) = (τ + τ')/2`. So the `W_1`-optimal
location is the quantile at the **midpoint** of the cell: `θ_i = F_Y⁻¹(τ̂_i)` with
`τ̂_i = (2i − 1)/(2N)`. Not the cell edges `i/N` — the cell centers. That tells me precisely which
quantiles my `N` locations should chase.

Make that concrete with `N = 50`. The cumulative edges are `τ_i = i/50 ∈ {0.02, 0.04, …, 1.00}`, but the
locations sit at the midpoints `τ̂_i = (2i−1)/100 ∈ {0.01, 0.03, 0.05, …, 0.99}`. The distinction is not
pedantic. If I had naively parked location `i` at the edge `τ_i`, location 50 would chase the `1.00`
quantile — the essential supremum of the return, an infinitely-sensitive statistic on a heavy tail — and
location 0 would chase `0.00`, the infimum. The midpoint convention instead sends the extreme locations
to `0.99` and `0.01`, one full half-cell in from the edges, which is exactly the reachable-from-samples
region: the `0.99` quantile is estimable, the `1.00` "quantile" is not. So the `W_1`-optimal placement
is also the *statistically well-posed* placement, and I did not have to impose that separately — it fell
out of decoupling the transport integral cell by cell.

Can I hit those midpoint quantiles from samples without bias? A quantile parametrization alone does *not*
unbias Wasserstein — minimizing sample-`W_p` is still biased even here. The unbiasedness has to come
from the *loss I use to hit each quantile*, and that loss is quantile regression. To estimate the
`τ`-quantile of a distribution from samples, use the asymmetric loss `ρ_τ(u) = u(τ − 1{u<0})` with
`u = Ẑ − θ`: it charges `τ|u|` when I underestimate and `(1−τ)|u|` when I overestimate. Its subgradient
in `θ` is `Pr(Ẑ < θ) − τ`, zero exactly when `θ = F⁻¹(τ)`. Crucially the gradient depends only on the
*sign* of `u` — `τ − 1{u<0}` — so a single sample gives an **unbiased** stochastic gradient. That is the
whole escape: I cannot descend `W_p`, but I can descend the quantile-regression loss whose minimizers
are the very locations that minimize `W_1`. End-to-end Wasserstein, by way of quantile regression on the
midpoint quantiles.

The contrast with what I gave up is sharp: minimizing sample-`W_1` needs the sample's *value* to locate
a quantile, and one draw is not an observation of a quantile, so its gradient does not average to the
population one; here the sign is a Bernoulli whose mean is precisely the CDF I am matching, so the
single-sample gradient is exactly the population gradient. That sign-only structure is the whole
difference between a biased and an unbiased estimator.

One wrinkle before a deep net: `ρ_τ` is kinked at `u = 0` and its gradient magnitude stays constant
(`τ` or `1−τ`) as `u → 0`, so there is no shrinking of the step as the error gets small and the locations
jitter. Round the kink with a **Huber** loss — quadratic inside `|u| ≤ κ`, linear outside — and multiply
by the asymmetric weight, giving the *quantile Huber* loss
`ρ_τ^κ(u) = |τ − 1{u<0}| · L_κ(u)`. With `κ = 1` the inner piece is `½u²` inside `[−1, 1]` and
`|u| − ½` outside — exactly the gradient-clipped squared error a scalar agent already uses — so I am
swapping its Huber for an *asymmetric* one. Control stays mean-greedy, because the objective is still to
maximize expected return: the greedy action is `argmax_a (1/N) Σ_j θ_j(s,a)`, the argmax of the
per-action location average — a drop-in for DQN's `argmax_a Q`. The bootstrapped target locations are
`Tθ_j = r + γ θ_j(s', a*)` (with `γ` zeroed at terminals), and each predicted location `θ_i(s,a)` is
regressed, at its own level `τ̂_i`, against *all* `N` target locations: the all-pairs quantile Huber
loss `(1/N) Σ_i Σ_j ρ_{τ̂_i}^κ(Tθ_j − θ_i)`. No target realignment, no range to set by hand; the only
new knob over DQN is `N`.

Now land it in *this* task's edit surface, and note where it departs from the generic recipe. The torso
is the **fixed MLP encoder** (`obs_dim → 120 → 84`), not a conv stack, so the only change to `QNetwork`
is the head: a linear map `84 → |A|·N` reshaped to `(batch, |A|, N)`, and `forward` returns the
per-action mean over the `N` quantiles so the evaluation harness still argmaxes a clean `(batch, |A|)`.
I set **`N = 50`** quantiles, not the larger value the generic Atari recipe uses — on classic-control
tasks with a single environment and a 500k-step budget, 50 locations already resolve the bimodality I
care about (safe-landing mass vs crash mass on LunarLander) without inflating the head past the
parameter budget, and a coarser comb trains faster and more stably here. Put numbers to that. The head
is `84 → |A|·N`, so on LunarLander (`|A| = 4`) at `N = 50` it is `84·200 + 200 = 17 000` parameters —
already comparable to the ~11 000-parameter frozen encoder, which is as far as I want to push before the
capacity check starts to frown. Take the generic Atari `N = 200` instead and it becomes
`84·800 + 800 = 68 000`, roughly six times the encoder — that is a head that dwarfs the trunk it reads
from, exactly the "capacity, not algorithm" smell I am supposed to avoid. And the loss cost is worse than
linear: the all-pairs quantile-Huber term is `O(N²)` per transition, `50² = 2 500` pairwise errors at
`N = 50` versus `200² = 40 000` at `N = 200` — a 16× compute multiplier on every one of the ~50 000
updates over the run. So both the parameter budget and the per-update FLOP budget point the same way, and
50 quantiles is where "enough resolution to see the LunarLander bimodality" meets "cheap and stable
enough to fit on a single 500k-step run." The midpoint levels are
`τ̂_i = (2i − 1)/(2N)`, fixed buffers. `κ = 1`. For the bootstrap I select `a*` greedily on the
target network's next-state quantile means and take *that* network's quantiles for `Tθ_j` — note this is
the scaffold's plain-DQN style target (select and evaluate both on the **target** net), *not* the
double-DQN split (select on online, evaluate on target) that the generic recipe sometimes pairs with;
the frozen loop only lets me change the head and the loss, and decoupling overestimation is a different
concern. The target uses `(1 − dones)` to zero `γ` at terminals exactly as the scalar loop did. I drop
the dueling head: distributional learning is a change to the *output object and loss*, and stacking it on
the two-stream head would conflate two changes — I want to isolate the effect of modeling the distribution.
Adam at the scaffold `lr`, no grad clip needed beyond what the Huber already provides. (The full scaffold
module is in the answer.)

One alternative is tempting: instead of a *fixed* comb of `N` midpoint levels, sample the quantile levels
`τ` at random each forward pass and condition the head on the sampled `τ` — a continuous quantile
function with, in principle, infinite effective resolution. But it needs a `τ`-embedding (a cosine
feature bank feeding an extra learned layer) grafted onto the head — capacity and structure beyond the
single `84 → |A|·N` linear map the edit surface cheaply allows — and it turns the clean
`forward(obs) → (batch, |A|)` interface into something that must sample and average internally to hand
the harness a deterministic argmax. For classic control on a single 500k-step run I want the *same* `N`
quantiles every step, so the bootstrap target is a stable function of the network rather than a fresh
random draw; the sampled-`τ` variant buys resolution I do not need at a stability cost I cannot afford. I
keep the fixed `N = 50` comb.

So what do I expect against the dueling numbers? CartPole is already saturated, so this can only match
500 — the test is that the 50-way head does not *destabilize* a solved task, a real risk since it is more
to fit than a linear one. Acrobot I expect on par again, low −80s, because its return is fairly unimodal
and there is little distributional structure for quantiles to exploit beyond the mean. The bet is on
**LunarLander**: the dueling failure was the crash-basin seed pulling the mean to −89. If modeling the
return distribution is the right fix, the quantile head lets the greedy policy distinguish a high-mean
action that sometimes crashes from one that reliably lands, so I expect the worst seed to climb out of
the negative basin and the mean to rise well above 89 with a tighter spread. I am less sure it reaches
the very top: 50 equal-mass quantiles spend the same one-in-50 budget on the thin crash tail as on the
dense middle, so if the low-probability crash tail is where resolution is thinnest, that is where this
representation stays weakest. Should the mean and worst seed lift but the tail stay ragged, the reading
is that spreading resolution uniformly across the probability axis is the wrong allocation for a heavy
tail — and the remaining gain would have to come from putting resolution where the returns actually
live.
