The dueling head told me exactly what I suspected and where its limit is, and it told me in the
LunarLander column. CartPole sat at 500 on two of three seeds and 461.6 on the third — pinned, as
expected, since the cap was already reached by plain DQN and the architecture only had to hold it.
Acrobot came in at a mean of −82.6, right in the low −80s I guessed, essentially on par with what a
plain value head gets — also as expected, because Acrobot's reward is a dense negative time-to-goal where
the action matters at almost every step, so the value/advantage split has very little redundancy to
exploit. The signal is LunarLander: mean 89.06, and the per-seed numbers are the whole story —
{127.35, 229.07, **−89.25**}. One seed found a genuinely good policy (229), one found a mediocre one
(127), and one fell into the crash basin and came out *negative*. That −89.25 seed is the −1000-style
tell from the bottom of any RL ladder: the agent's greedy policy on that seed is systematically landing
in the deceptive failure region, and the mean is being dragged down by it. The dueling architecture did
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
what the dueling rung still relied on. But `Z` itself is a random variable, and on LunarLander it is
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
gradient. So the metric the theory wants is the one I cannot descend from single transitions. The
categorical approach (the next rung) dodges this by fixing the atom *locations* on a predetermined grid,
making the *probabilities* learnable, projecting the shifted target back onto the grid, and minimizing
KL — but that needs me to supply `[v_min, v_max]` as prior knowledge and the projection exists only
because fixed atoms force off-grid collisions. Let me see whether I can find a representation that is
genuinely Wasserstein-aware *and* trainable from samples, with no projection and no support bounds.

The categorical agent's free variables are the *probabilities* on fixed locations — it learns the
*vertical* axis. So turn the parametrization on its side. Fix the probabilities to be uniform,
`q_i = 1/N`, and make the *locations* `θ_i` the learnable thing:
`Z_θ(s,a) = (1/N) Σ_i δ_{θ_i(s,a)}`. I am no longer learning how much mass sits at fixed heights; I am
learning *where* `N` equal lumps of mass should sit. And "where the `i`-th of `N` equal lumps sits" is
exactly a **quantile** of the distribution. So this transposed parametrization estimates quantiles of
the return. Three things immediately look better: the support is not pinned to any `[v_min, v_max]` — the
locations slide to wherever the returns actually live, with per-state adaptive resolution (this matters
here, where LunarLander spans roughly −400 to +300 and Acrobot −500 to −60 and CartPole 0 to 500, three
very different ranges that one fixed grid would have to straddle); there is no projection, because when
the Bellman target's atoms move they are just numbers I compare directly to my locations; and — the part
I must verify — estimating quantiles may be doable from samples *without* a biased gradient.

Which quantiles? Minimize `W_1` between an arbitrary target `Y` and a uniform-`N`-Dirac distribution on
ordered locations `θ_1 ≤ … ≤ θ_N`. With cumulative levels `τ_i = i/N`, the inverse-CDF of the uniform
comb is the staircase equal to `θ_i` on `(τ_{i-1}, τ_i]`, so
`W_1 = Σ_i ∫_{τ_{i-1}}^{τ_i} |F_Y⁻¹(ω) − θ_i| dω`. The cells decouple — each `θ_i` appears in only its
own integral — so minimize each separately. The subgradient in `θ` of `∫_τ^{τ'} |F⁻¹(ω) − θ| dω` is
`(F(θ) − τ) − (τ' − F(θ)) = 2F(θ) − (τ + τ')`, zero when `F(θ) = (τ + τ')/2`. So the `W_1`-optimal
location is the quantile at the **midpoint** of the cell: `θ_i = F_Y⁻¹(τ̂_i)` with
`τ̂_i = (2i − 1)/(2N)`. Not the cell edges `i/N` — the cell centers. That tells me precisely which
quantiles my `N` locations should chase.

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
loss `(1/N) Σ_i Σ_j ρ_{τ̂_i}^κ(Tθ_j − θ_i)`. No projection, no `[v_min, v_max]`; the only new knob over
DQN is `N`.

Now land it in *this* task's edit surface, and note where it departs from the generic recipe. The torso
is the **fixed MLP encoder** (`obs_dim → 120 → 84`), not a conv stack, so the only change to `QNetwork`
is the head: a linear map `84 → |A|·N` reshaped to `(batch, |A|, N)`, and `forward` returns the
per-action mean over the `N` quantiles so the evaluation harness still argmaxes a clean `(batch, |A|)`.
I set **`N = 50`** quantiles, not the larger value the generic Atari recipe uses — on classic-control
tasks with a single environment and a 500k-step budget, 50 locations already resolve the bimodality I
care about (safe-landing mass vs crash mass on LunarLander) without inflating the head past the
parameter budget, and a coarser comb trains faster and more stably here. The midpoint levels are
`τ̂_i = (2i − 1)/(2N)`, fixed buffers. `κ = 1`. For the bootstrap I select `a*` greedily on the
target network's next-state quantile means and take *that* network's quantiles for `Tθ_j` — note this is
the scaffold's plain-DQN style target (select and evaluate both on the **target** net), *not* the
double-DQN split (select on online, evaluate on target) that the generic recipe sometimes pairs with;
the frozen loop only lets me change the head and the loss, and decoupling overestimation is a different
concern. The target uses `(1 − dones)` to zero `γ` at terminals exactly as the scalar loop did. I drop
the dueling head: distributional learning is a change to the *output object and loss*, and stacking it on
the two-stream head would conflate two rungs — I want to isolate the effect of modeling the distribution.
Adam at the scaffold `lr`, no grad clip needed beyond what the Huber already provides. (The full scaffold
module is in the answer.)

So what do I expect against the dueling numbers? CartPole is already saturated, so QR-DQN can only match
500 — the test is that the quantile head does not *destabilize* a solved task (a real risk: a 50-way
head is more to fit than a linear one, and on an easy task that can add variance). Acrobot I expect to
land in the low −80s again, roughly on par, because its return is fairly unimodal (negative
time-to-goal) and there is little distributional structure for quantiles to exploit beyond the mean. The
bet is entirely on **LunarLander**, and it is a sharp, falsifiable one: the dueling failure was the
seed-456 crash basin pulling the mean to −89 and the spread {−89, 127, 229}. If modeling the return
distribution is the right fix, the quantile head should let the greedy policy distinguish a high-mean
action that sometimes crashes from one that reliably lands, so I expect the **worst seed to climb out of
the negative basin** and the mean to rise well above 89 with a tighter spread. I am less sure it reaches
the very top — 50 quantiles on a single 500k-step run is a coarse picture of a heavy-tailed return, and
if the low-probability crash tail is exactly where the resolution is thinnest, a categorical
representation that spends its capacity on *probabilities at fixed locations* might capture that tail
mass more faithfully than uniform-mass quantiles do. If QR-DQN lifts LunarLander's mean and worst seed
but still leaves the tail ragged, that is the precise opening for the next rung: fix the support to the
known classic-control range and learn the *mass* on it instead of the locations.
