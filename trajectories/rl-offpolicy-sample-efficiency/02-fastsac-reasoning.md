PPO landed where the structure said it would: weakest of the ladder, because it spends the budget in
the one currency this task cannot afford. On-policy means every one of the 12.8M frames feeds at most
a few gradient steps and is then discarded, and the consensus from the HumanoidBench source is
unambiguous — off-policy actor-critics beat PPO under exactly this budget, and PPO falls furthest
behind on the harder-to-explore behaviors (walk, run). So the diagnosis is precise, and it is not
"PPO is unstable" — the clip makes it reliable — it is "PPO's learning throughput per frame is too
low," a hard reuse ceiling of about ten passes before the trust region flattens the gradient and the
batch must be thrown away. The fix is not a better policy-gradient trick; it is to start reusing
transitions, which means moving onto the editable off-policy surface: keep everything the fixed loop
gives me (128 parallel envs, the GPU replay buffer, observation normalization, the distributional
critic substrate) and fill the algorithm contract with an off-policy actor-critic that revisits each
transition many times.

The first design question is which off-policy actor-critic, and it is a genuine fork with two live
branches. The deterministic line — DDPG and its fix TD3 — reuses experience well and gives the
cleanest exploitation: a deterministic actor `μ(s)` pushed straight uphill on the critic by the
deterministic policy gradient, no score-function variance, the actor moving directly toward the
critic's argmax. That is attractive precisely because PPO's other failure was undirected,
high-variance exploration. But it inherits a specific weakness worth worrying about here: a
deterministic actor explores by a thin band of additive noise bolted onto the action, and on a
humanoid with tens of contact-rich action dimensions, "the current policy plus a little Gaussian
noise" is a narrow tube of data around one trajectory — PPO's isotropic-cloud problem in a different
dress. PPO's failure was throughput; I do not want to trade it for a different exploration failure. So
I reach for the maximum-entropy line, where exploration is not a noise process bolted on but a term
the policy *optimizes*: maximize reward plus policy entropy,
`J(π) = Σ_t E[r(s_t,a_t) + α H(π(·|s_t))]`. A policy paid for entropy keeps mass on every action that
looks comparably good rather than collapsing prematurely — directed-by-the-objective exploration that
needs no hand-tuned per-task noise schedule. I am not yet certain the deterministic branch is wrong —
its cleaner exploitation is a real pull, and if a fleet of parallel envs could manufacture breadth for
it the calculus might flip — so I hold it open as the thing to test if entropy leaves exploitation on
the table.

Crucially the entropy must live *inside* the value, not be sprinkled on the actor loss. The soft
state value is `V(s) = E_{a~π}[Q(s,a) − α log π(a|s)]`, and the soft Bellman target bootstraps on it:
`y = r + γ E_{a'~π}[Q(s',a') − α log π(a'|s')]`. Written that way the future entropy I will collect
shows up in the value of acting now, so it shapes long-horizon behavior — exactly the regime (walk,
run) where PPO struggled. Added only to the actor loss, `−α log π` would be a one-step nudge with no
memory; folding it into the bootstrap is what makes it a value the agent plans around.

Now the substrate constraint: the task's SAC fill is not the textbook scalar-critic SAC, because the
critic here is **categorical distributional** — 101 atom logits over a fixed support `[−250,250]`,
trained by cross-entropy against a *projected* distribution. So I cannot use the usual soft-Bellman
MSE target `y = r + γ(min Q' − α log π')`. The clean way to fold max-entropy into that machinery is to
push the entropy bonus into the reward the projection operates on. The projection computes, for each
support atom `z`, the shifted location `Tz = reward + bootstrap·discount·z`, clamps it to the support,
and splits its mass onto the two neighboring grid atoms by linear interpolation; cross-entropy then
trains the current critic toward that target. If I replace `reward` with `reward − α·log π(a'|s')` —
the entropy-augmented reward at the *next* action sampled from the current stochastic actor — then the
soft value's entropy term enters the distributional bootstrap exactly as the theory demands, and
everything downstream is unchanged. That is the precise edit: sample the next action and its log-prob,
subtract `α·next_log_prob` from the reward, project as usual.

I should confirm this reproduces the scalar soft Bellman target and is not just a plausible shuffle of
terms, because that is the whole claim. The categorical projection is mean-preserving: away from the
clamp boundaries the two-atom split places mass so that the expected support value of the projected
distribution equals the shifted location it was built from. So the mean of my augmented target under
the support `z` is `(r − α log π') + γ E[Z'|s',a']`, and with clipped double-Q selecting the
smaller-mean distribution, `E[Z'] = min(Q1', Q2')`. That is `r − α log π' + γ min(Q1', Q2')` —
precisely the SAC soft-Bellman target rearranged, with the substrate's per-step formulation folding
`−α log π'` at the step it is sampled. The distributional fill therefore trains toward the same
expectation scalar SAC would, while additionally carrying the *shape* of the return distribution — the
win of going distributional at no change to the entropy accounting. That is what convinces me the
one-line reward substitution is faithful rather than a reward-shaping hack.

The clipped double-Q logic survives the move to distributions: read each target critic's scalar mean
`Σ_i p_i z_i` and keep the *whole distribution* of whichever has the smaller mean, as the cross-entropy
target for both critics. Underestimation is the safe direction — a stochastic actor sampling around
its mean will not systematically chase a value biased low, whereas an over-estimated atom mass gets
sampled toward and amplified.

The actor is a tanh-squashed Gaussian: the net outputs a mean and a log-std (clamped to `[−5,2]`), I
draw a reparameterized `u ~ N(μ,σ)`, squash to `a = tanh(u)`, and correct the log-prob for the squash.
That correction is a change of variables I must get exactly right, since a wrong Jacobian silently
biases the entropy estimate: under `a = tanh(u)`, `da/du = 1 − tanh²(u)`, so
`log π(a) = log N(u;μ,σ) − Σ_i log(1 − tanh²(u_i) + ε)`, the sum over action dimensions and `ε` a
numerical floor for when tanh saturates near `±1`. The actor objective is the reparameterized
`E[α log π(a|s) − min(Q1,Q2)]` — ascend the value, pay the entropy price. Because exploration now comes
from the stochastic policy itself, the actor updates *every* gradient step rather than on a delayed
schedule. At evaluation the loop calls the deterministic readout `tanh(μ)`, the mode of the squashed
Gaussian.

The temperature `α` decides whether this is usable across stand, walk, and run without per-task tuning,
so I do not hand-set it. Recast it as a dual variable servoing expected entropy onto a target:
constrain `E[−log π] ≥ H̄` and solve the Lagrangian, giving the update
`α ← argmin_α E[−α(log π + H̄)]`. Parameterize `log α` for positivity; the gradient of
`−log α·(log π + H̄)` w.r.t. `log α` is `−(log π + H̄)`, so when the policy is too deterministic
(entropy below target) the gradient drives `α` up and forces more exploration, and when it is too
random it drives `α` down. That sign makes it a thermostat rather than a free coefficient. The target
entropy is the scale-aware `H̄ = −dim(A)`, one nat per action dimension, so it scales with the action
count and needs no sweep. The fill initializes the entropy coefficient at `0.2` and adapts `log α` from
there with its own Adam optimizer.

Two substrate-specific choices lean toward the fast-and-stable philosophy the loop is built for.
First, the architecture is not a bare MLP — it uses LayerNorm and SiLU in both actor and critic (a
descending `512→256→128` actor, `1024→512→256` critic, each Linear followed by LayerNorm and SiLU).
This is deliberate stabilization for off-policy value learning: LayerNorm controls feature magnitudes
so the bootstrapped critic does not amplify its own scale drift, and SiLU is smoother than ReLU. PPO
never had to worry about the deadly triad — function approximation, bootstrapping, off-policy data —
because it never bootstrapped through a replayed target; the moment I do, controlling feature scale
becomes load-bearing, and I am about to push far more gradient signal through this critic. Second, the
optimizer and reuse rate stay matched to the substrate: AdamW with weight decay `0.1`, cosine LR
annealing, `num_updates=2` per env step, a fast target update `tau=0.1`. These are the fast-and-stable
settings — a fast `tau` moves the target toward the online critic quickly, which pairs with the
low-variance gradients a large replayed batch provides — not SAC's classic small-tau single-update
regime.

One clamp check, because it is where this fill diverges most from the textbook. The entropy-augmented
reward `reward − α·next_log_prob` is what gets clamped to `[−250,250]` before the atom split, so a
badly-scaled `α` could push it outside the support and silently distort the target. At the chosen scale
this does not bite: for a tanh-Gaussian near its target entropy `log π` is order `dim(A)`, so
`α·log π ≈ 0.2 × (tens) ≈` single-to-low-double digits — small against the `250` half-width. The danger
only appears if `α` runs away, which is exactly what the dual update prevents by pulling it down as the
policy sharpens. One more reason the temperature must self-tune rather than be hand-set. And keeping the
two entropy channels straight — one in the bootstrap at the next action, one in the actor's own loss —
is what makes this a faithful soft actor-critic.

Where does this land relative to the prior method and the branch I held open? Against PPO the advantage
is structural and should be decisive: FastSAC reuses every transition out of the buffer many times
instead of discarding it after ten epochs — the loop runs `num_updates=2` sampled batches per env step
over 100,000 steps, so a transition is eligible for resampling across its whole buffer residency,
orders of magnitude more than PPO's ceiling of ten — and its exploration is the entropy-driven kind
that suits long-horizon coordination rather than undirected Gaussian wandering. So I expect FastSAC to
clear PPO on all three tasks and open the largest margin on walk and run, closing the shortfall the PPO
feedback diagnosed. Where it might fall short is against a deterministic off-policy method on this
substrate: maximizing entropy over a high-dimensional humanoid action is genuinely hard, and a
deterministic actor whose exploration is supplied for free by 128 parallel environments could exploit
the value function more aggressively — especially on stand, where exploration is cheap and exploitation
is what matters. That is precisely the deterministic branch I held open at the fork, and it is the next
rung. The distilled FastSAC fill of the editable contract is in the answer.
