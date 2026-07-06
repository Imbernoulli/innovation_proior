PPO landed where the structure said it would: weakest of the ladder, because it spends the
budget in the one currency this task cannot afford. On-policy means every one of the 12.8M frames
feeds at most a few gradient steps and is then discarded, and the consensus from the
HumanoidBench source is unambiguous — off-policy actor-critics beat PPO under exactly this budget,
and PPO falls furthest behind on the harder-to-explore behaviors (walk, run) where long-horizon
gait coordination needs the sample efficiency that experience reuse provides and on-policy
learning lacks. So the diagnosis is precise and it is not "PPO is unstable" — the clip makes PPO
reliable — it is "PPO's learning throughput per frame is too low," a hard reuse ceiling of about
ten passes per frame before the trust region flattens the gradient and the batch must be thrown
away. The fix is not a better policy-gradient trick; it is to start reusing transitions, which
means moving onto the editable off-policy surface and putting the replay buffer to work. That is
the entire content of this step: keep everything the fixed loop gives me (128 parallel envs, the
GPU replay buffer, observation normalization, the distributional critic substrate) and fill the
algorithm contract with an off-policy actor-critic that revisits each transition many times.

The first design question is which off-policy actor-critic, and it is a genuine fork with two live
branches. The deterministic line — DDPG and its fix TD3 — reuses experience well and gives the
cleanest exploitation: a deterministic actor `μ(s)` pushed straight uphill on the critic by the
deterministic policy gradient, no score-function variance, the actor moving directly toward the
critic's argmax. That is attractive precisely because PPO's other failure was undirected, high-
variance exploration, and a deterministic actor has none of that noise in its *learning* signal.
But it inherits a specific weakness that I should worry about here: a deterministic actor explores
by a thin band of additive noise bolted onto the action, and on a humanoid with tens of contact-
rich action dimensions, "the current policy plus a little Gaussian noise" is a narrow tube of data
around one trajectory. PPO already showed me what an isotropic noise cloud does in high dimensions
— it mostly samples off the coordinated-gait manifold — and a single deterministic actor with one
global noise scale is that same undirected-exploration risk in a different dress. PPO's failure was
throughput; I do not want to trade it for a different exploration failure. So I reach for the
maximum-entropy line instead, where exploration is not a noise process bolted on but a term the
policy *optimizes*: maximize reward plus the entropy of the policy,
`J(π) = Σ_t E[r(s_t,a_t) + α H(π(·|s_t))]`. A policy paid for entropy keeps probability mass on
every action that looks comparably good rather than collapsing onto one prematurely, and that is
directed-by-the-objective exploration that does not need me to hand-tune a noise schedule per task.
I am not yet certain the deterministic branch is wrong — its cleaner exploitation is a real pull,
and if a fleet of parallel envs could manufacture breadth for it, the calculus might flip — but on
the argument I have now, entropy-as-objective is the safer answer to PPO's exploration problem, and
I will hold the deterministic branch open as the thing to test if this one leaves exploitation on
the table.

Crucially the entropy must live *inside* the value, not be sprinkled on the actor loss. The soft
state value is `V(s) = E_{a~π}[Q(s,a) − α log π(a|s)]`, and the soft Bellman target bootstraps on
that `V`: `y = r + γ E_{a'~π}[Q(s',a') − α log π(a'|s')]`. Written that way the future entropy I
will collect shows up in the value of acting now, so it shapes long-horizon behavior — which is
exactly the regime (walk, run) where PPO struggled. If I had instead only added `−α log π` to the
actor's loss, the entropy would be a one-step nudge on the policy with no memory; folding it into
the bootstrap is what makes it a value the agent plans around.

Now I have to be careful, because the task's SAC fill is not the textbook scalar-critic SAC from
the standard derivation — it is SAC re-expressed on *this* substrate, and the substrate's critic
is **categorical distributional**. So I cannot use the usual soft-Bellman MSE target
`y = r + γ(min Q' − α log π')`. The critic here emits 101 atom logits over a fixed support
`[−250, 250]`, and the target is a *projected distribution* trained against by cross-entropy. I
need to fold the max-entropy idea into that distributional machinery, and the clean way to do it is
to push the entropy bonus into the reward that the projection operates on. The projection computes,
for each support atom `z`, the shifted location `Tz = reward + bootstrap·discount·z`, clamps it to
the support, and splits its probability mass onto the two neighboring grid atoms by linear
interpolation; cross-entropy then trains the current critic toward that projected target. If I
replace `reward` in that projection with `reward − α·log π(a'|s')` — the entropy-augmented reward
at the *next* action sampled from the current stochastic actor — then the soft value's entropy
term enters the distributional bootstrap exactly as the theory demands, and everything downstream
(clamp, floor/ceil split, cross-entropy) is unchanged. That is the precise edit: sample the next
action and its log-prob from the actor, subtract `α·next_log_prob` from the reward, and project as
usual.

I should verify this actually reproduces the scalar soft Bellman target and is not just a plausible
shuffle of terms, because that is the whole claim. The categorical projection has a mean-preserving
property: away from the clamp boundaries, the two-atom linear split places mass so that the expected
support value of the projected distribution equals the shifted location it was built from. Take the
mean of my entropy-augmented projected target under the support `z`:
`E[Z_target] = E_z[(reward − α log π') + bootstrap·discount·z] = (r − α log π') + γ E[Z'|s',a']`,
where `E[Z'|s',a']` is the target critic's mean at `(s', a')`. With clipped double-Q selecting the
distribution whose mean is smaller, `E[Z'] = min(Q1', Q2')`. So the scalar mean of my distributional
target is `r − α log π' + γ min(Q1', Q2')`, which is *exactly* the SAC soft-Bellman target rearranged
(`r + γ(min Q' − α log π')` differs only in whether the discount multiplies the entropy of the next
step, and here the substrate's per-step formulation folds `−α log π'` at the same step it is sampled).
The distributional fill therefore trains toward the same expectation the scalar SAC would, while
additionally carrying the *shape* of the return distribution — the win of going distributional at no
change to the entropy accounting. That check is what convinces me the one-line reward substitution is
faithful rather than a reward-shaping hack.

The clipped double-Q logic survives the move to distributions the same way a distributional TD3
would do it — read each target critic's scalar mean `Σ_i p_i z_i`, and keep the *whole distribution*
belonging to whichever critic has the smaller mean; the min selects a distribution by its
expectation and that selected distribution is the cross-entropy target for both critics.
Underestimation is the safe direction: a stochastic actor sampling around its mean will not
systematically chase a value that is biased *low*, whereas an over-estimated atom mass would get
sampled toward and amplified, so taking the pessimistic critic is the right regularizer even in the
soft setting.

The actor side mirrors this. The stochastic actor is a tanh-squashed Gaussian: the net outputs a
mean and a log-std (clamped to a sane band, here `[−5, 2]`), I draw a reparameterized
pre-activation `u ~ N(μ, σ)`, squash with tanh to get the bounded action `a = tanh(u)`, and correct
the log-prob for the squash. That correction is a change-of-variables I should get exactly right,
because a wrong Jacobian silently biases the entropy estimate. Under `a = tanh(u)`,
`da/du = 1 − tanh²(u)`, so `log π(a) = log N(u; μ, σ) − Σ_i log(1 − tanh²(u_i) + ε)`, the sum over
action dimensions and the `ε` a numerical floor for when `tanh` saturates near `±1`. Without that
subtraction the log-prob would be the unbounded-Gaussian one and the entropy term would be wrong by
a state-dependent amount, so I keep it exact. The actor objective is then the reparameterized
`E[α log π(a|s) − min(Q1, Q2)]`, where the two `Q` values are the support-weighted means of the two
critics' predicted distributions at the actor's sampled action — ascend the value, pay the entropy
price. And because exploration now comes from the stochastic policy itself, the actor updates
*every* gradient step rather than on a delayed schedule; the substrate's loop runs the actor update
inside the same inner loop as the critic for SAC. At evaluation the loop calls the deterministic
readout `tanh(μ)`, the mean action, which is the right deterministic projection of an entropy-trained
policy — the mode of the squashed Gaussian, not a fresh sample.

The temperature `α` is the knob that decides whether this is usable across stand, walk, and run
without per-task tuning, so I do not hand-set it. Recast it as a dual variable that servos the
expected entropy onto a target: constrain `E[−log π] ≥ H̄` and solve the Lagrangian, which gives
the update `α ← argmin_α E[−α(log π + H̄)]`. Parameterize `log α` for positivity and take one
gradient step per update; the gradient of `−log α·(log π + H̄)` with respect to `log α` is
`−(log π + H̄)`, so when the policy's current `log π` sits above `−H̄` (entropy below target, too
deterministic) the gradient drives `log α` *up*, raising `α` and forcing more exploration, and when
the policy is too random it drives `α` down and lets the policy commit. That sign check is what
makes it a thermostat rather than a free-floating coefficient. The target entropy is the scale-aware
heuristic `H̄ = −dim(A)`, one nat per action dimension, so it scales with the humanoid's action count
and needs no sweep; the intuition is that each independent action coordinate should retain about one
nat of residual stochasticity, so a wider action space is granted proportionally more total entropy.
The fastsac fill initializes the entropy coefficient at `0.2` and adapts `log α` from there with its
own Adam optimizer.

Two substrate-specific choices remain, and they are where this differs from a vanilla SAC and lean
toward the fast-and-stable design philosophy the loop is built for. First, the architecture: the
fill does *not* use a bare MLP — it uses LayerNorm and SiLU in both the actor and the critic (a
descending stack `512→256→128` for the actor, `1024→512→256` for the critic, each Linear followed by
LayerNorm and SiLU). This is a deliberate stabilizer for off-policy value learning: LayerNorm
controls feature magnitudes so the bootstrapped critic does not amplify its own scale drift, and
SiLU is a smoother nonlinearity than ReLU. I keep it because the whole point of moving off-policy is
to push more gradient signal through the critic, and that is exactly the regime where the deadly
triad — function approximation, bootstrapping, off-policy data — bites without normalization. PPO
never had to worry about the triad because it never bootstrapped through a replayed off-policy
target; the moment I do, controlling feature scale becomes load-bearing. Second, the optimizer and
reuse rate stay matched to the substrate: AdamW with weight decay `0.1`, cosine LR annealing,
`num_updates=2` gradient steps per env step, a fast target update `tau=0.1`. These are not SAC's
classic small-tau, single-update settings — they are the fast-and-stable settings, and SAC inherits
them here because the substrate is built for them; the fast `tau=0.1` moves the target toward the
online critic quickly, which pairs with the low-variance gradients a large replayed batch provides.

Before I trace the expectations, one more substrate detail deserves a careful sentence because it is
where this fill diverges most from the textbook and it bears on stability, and it is checkable with
arithmetic. The classic SAC evaluates the soft value by sampling the next action from the current
policy and subtracting its entropy; here that sampling happens *inside* the distributional projection,
and the entropy-augmented reward `reward − α·next_log_prob` is what gets clamped to the `[−250, 250]`
support and split across atoms. That means a poorly-scaled `α` can push the augmented reward outside
the support and get clamped, silently distorting the target. Is that a real risk at the chosen scale?
The action space is tens of dimensions, and for a tanh-Gaussian near its target entropy `log π` is
order `dim(A)` in magnitude, so `α·log π ≈ 0.2 × (tens) ≈ single-to-low-double digits` — small against
the `250` half-width of the support. So at initialization the augmented reward shift is comfortably
inside the window and the clamp does not bite; the danger only appears if `α` runs away, which is
exactly what the dual update prevents by pulling it down as the policy sharpens. That is one more
reason the temperature must self-tune rather than be hand-set, and why initializing it modestly at
`0.2` and letting the dual update pull it down is the safe regime. The entropy bonus also enters only
at the next action, not the current one, so the critic's cross-entropy target carries the future-
entropy credit exactly as the soft-value theory requires, and the actor's own entropy term is
separate, in its loss. Keeping those two entropy channels straight — one in the bootstrap, one in the
policy objective — is what makes this a faithful soft actor-critic rather than a reward-shaping hack.

It helps to put the reuse advantage in the same arithmetic I used to indict PPO, so the "materially
more sample-efficient" claim is a number and not a mood. PPO's reuse factor was a hard 10 — ten
epochs, then delete. Here the buffer holds transitions across their whole residency and the loop
runs `num_updates=2` sampled batches per env step for 100,000 env steps, i.e. 200,000 gradient
updates over the run, each drawing a fresh minibatch from the buffer. A transition is therefore
eligible to be resampled on every update between the moment it is stored and the moment it is
evicted, which for a buffer that spans a large fraction of the run is orders of magnitude more than
ten expected touches. That is the mechanical reason the same 12.8M frames turn into far more usable
gradient here: I have lifted the reuse factor off PPO's ceiling and let it float up with buffer
residency. The distributional critic then converts each of those touches into a richer target — a
whole return distribution rather than a scalar — so the extra gradient signal is also higher-quality,
which is what lets the critic keep the survive-versus-fall return modes distinct instead of blurring
them into one averaged value the way a scalar critic would on a task where a fall is a cliff in return.

The LayerNorm+SiLU body is not free, and I want to confirm its cost is proportionate before I commit
to it as a stabilizer rather than a reflex. Count the critic body: `1024→512→256` is
`(n_in·1024) + (1024·512) + (512·256)` ≈ `n_in·1024 + 524k + 131k` weights, plus the atom head
`256·101 ≈ 26k`, and the LayerNorms add only `2·(1024+512+256) ≈ 3.6k` affine parameters — a rounding
error against the linear layers. So normalization buys me feature-scale control for well under one
percent extra parameters; the objection to LayerNorm is never parameter count, it is whether it
distorts the value scale, and pre-activation LayerNorm inside the body (not on the output) leaves the
final atom logits free to span whatever range the `[−250, 250]` support needs. The cost is
proportionate and the benefit — a critic that does not amplify its own scale drift while I push
200,000 bootstrapped updates through it — is exactly the deadly-triad insurance this off-policy
regime needs. That the actor updates every step rather than on a delay is safe for the same reason it
is necessary: the entropy term is a built-in damper on the policy, so a stochastic actor updated
every step does not run away from a still-settling critic the way an undamped deterministic actor
would, and updating it every step is what keeps the exploration distribution current with the value
it is chasing.

Let me trace why this should beat PPO and where it might still fall short, because that is the bar.
Against PPO the advantage is structural and should be decisive: FastSAC reuses every transition out
of the replay buffer many times instead of discarding it after ten epochs, so on the same 12.8M
frames it extracts far more learning, and its exploration is the entropy-driven kind that suits
long-horizon coordination rather than PPO's undirected Gaussian wandering. So I expect FastSAC to
clear PPO on all three tasks — `h1hand-stand-v0`, `h1hand-walk-v0`, `h1hand-run-v0` — and to open
the largest margin on walk and run, exactly the tasks where PPO's sample-inefficiency hurt most,
closing the shortfall the previous feedback diagnosed. Where it might fall short is against a
deterministic off-policy method on this particular substrate: maximizing entropy over a high-
dimensional humanoid action is genuinely hard, and a stochastic actor that must hold a tanh-Gaussian
over tens of dimensions can be noisier to optimize than a deterministic actor whose exploration is
supplied for free by the fleet of 128 parallel environments. The entropy objective also adds the
`−α log π` term to the distributional bootstrap, which is one more moving part in a critic that is
already chasing a projected target. So my falsifiable expectation is two-sided: FastSAC should beat
PPO clearly and across the board, but it may not be the strongest off-policy fill — if a deterministic
actor with the same distributional critic and the same fast-and-stable substrate explores adequately
through parallelism alone, it could exploit the value function more aggressively and edge FastSAC out,
especially on stand where exploration is cheap and exploitation is what matters. That is precisely the
deterministic branch I held open at the fork above, and that comparison is the next rung. The
distilled FastSAC fill of the editable contract is in the answer.
