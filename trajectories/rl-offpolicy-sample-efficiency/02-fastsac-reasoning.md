PPO landed where the structure said it would: weakest of the ladder, because it spends the
budget in the one currency this task cannot afford. On-policy means every one of the 12.8M frames
feeds at most a few gradient steps and is then discarded, and the consensus from the
HumanoidBench source is unambiguous — off-policy actor-critics beat PPO under exactly this budget,
and PPO falls furthest behind on the harder-to-explore behaviors (walk, run) where long-horizon
gait coordination needs the sample efficiency that experience reuse provides and on-policy
learning lacks. So the diagnosis is precise and it is not "PPO is unstable" — the clip makes PPO
reliable — it is "PPO's learning throughput per frame is too low." The fix is not a better
policy-gradient trick; it is to start reusing transitions, which means moving onto the editable
off-policy surface and putting the replay buffer to work. That is the entire content of this step:
keep everything the fixed loop gives me (128 parallel envs, the GPU replay buffer, observation
normalization, the distributional critic substrate) and fill the algorithm contract with an
off-policy actor-critic that revisits each transition many times.

The first design question is which off-policy actor-critic. The deterministic line — DDPG and its
fix TD3 — reuses experience well, but it inherits a specific weakness that I should worry about
here: a deterministic actor explores by a thin band of additive noise, and on a humanoid with
tens of contact-rich action dimensions, "the current policy plus a little Gaussian noise" is a
narrow data distribution. PPO's failure was throughput, and I do not want to trade it for a
different exploration failure. So I reach for the maximum-entropy line instead, where exploration
is not a noise process bolted on but a term the policy *optimizes*: maximize reward plus the
entropy of the policy, `J(π) = Σ_t E[r(s_t,a_t) + α H(π(·|s_t))]`. A policy paid for entropy keeps
probability mass on every action that looks comparably good rather than collapsing onto one
prematurely, and that is directed-by-the-objective exploration that does not need me to hand-tune a
noise schedule per task. Crucially the entropy must live *inside* the value, not be sprinkled on
the actor loss: the soft state value is `V(s) = E_{a~π}[Q(s,a) − α log π(a|s)]`, and the soft
Bellman target bootstraps on that `V`, so the future entropy I will collect shows up in the value
of acting now and shapes long-horizon behavior — which is exactly the regime (walk, run) where PPO
struggled.

Now I have to be careful, because the task's SAC fill is not the textbook scalar-critic SAC from
the standard derivation — it is SAC re-expressed on *this* substrate, and the substrate's critic
is **categorical distributional**. So I cannot use the usual soft-Bellman MSE target
`y = r + γ(min Q' − α log π')`. The critic here emits 101 atom logits over a fixed support
`[−250, 250]`, and the target is a *projected distribution* trained against by cross-entropy. I
need to fold the max-entropy idea into that distributional machinery, and the clean way to do it is
to push the entropy bonus into the reward that the projection operates on. The projection computes,
for each support atom, the shifted location `Tz = reward + bootstrap·discount·z`, clamps it to the
support, and splits its probability mass onto the two neighboring grid atoms by linear
interpolation; cross-entropy then trains the current critic toward that projected target. If I
replace `reward` in that projection with `reward − α·log π(a'|s')` — the entropy-augmented reward
at the *next* action sampled from the current stochastic actor — then the soft value's entropy
term enters the distributional bootstrap exactly as the theory demands, and everything downstream
(clamp, floor/ceil split, cross-entropy) is unchanged. That is the precise edit: sample the next
action and its log-prob from the actor, subtract `α·next_log_prob` from the reward, and project as
usual. The clipped double-Q logic survives the move to distributions the same way FastTD3 does it —
read each target critic's scalar mean `Σ_i p_i z_i`, and keep the *whole distribution* belonging to
whichever critic has the smaller mean; the min selects a distribution by its expectation and that
selected distribution is the cross-entropy target for both critics.

The actor side mirrors this. The stochastic actor is a tanh-squashed Gaussian: the net outputs a
mean and a log-std (clamped to a sane band, here `[−5, 2]`), I draw a reparameterized
pre-activation `u ~ N(μ, σ)`, squash with tanh to get the bounded action, and correct the log-prob
for the squash by subtracting `Σ log(1 − tanh²(u) + ε)` so the entropy term is exact. The actor
objective is the reparameterized `E[α log π(a|s) − min(Q1, Q2)]`, where the two `Q` values are the
support-weighted means of the two critics' predicted distributions at the actor's sampled action —
ascend the value, pay the entropy price. And because exploration now comes from the stochastic
policy itself, the actor updates *every* gradient step rather than on the delayed `policy_frequency`
schedule TD3 uses; the substrate's loop runs the actor update inside the same inner loop as the
critic for SAC. At evaluation the loop calls the deterministic readout `tanh(μ)`, the mean action,
which is the right deterministic projection of an entropy-trained policy.

The temperature `α` is the knob that decides whether this is usable across stand, walk, and run
without per-task tuning, so I do not hand-set it. Recast it as a dual variable that servos the
expected entropy onto a target: constrain `E[−log π] ≥ H̄` and solve the Lagrangian, which gives
the update `α ← argmin_α E[−α(log π + H̄)]`. Parameterize `log α` for positivity, take one
gradient step per update; the gradient sign makes it a thermostat — if the policy is too
deterministic (entropy below `H̄`), `α` rises and forces more exploration; if too random, `α`
falls and lets the policy commit. The target entropy is the scale-aware heuristic `H̄ = −dim(A)`,
one nat per action dimension, so it scales with the humanoid's action count and needs no sweep. The
fastsac fill initializes the entropy coefficient at `0.2` and adapts `log α` from there with its own
Adam optimizer.

Two substrate-specific choices remain, and they are where this differs from a vanilla SAC and lean
toward the FastTD3 design philosophy. First, the architecture: the fill does *not* use a bare MLP —
it uses LayerNorm and SiLU in both the actor and the critic (a descending stack
`512→256→128` for the actor, `1024→512→256` for the critic, each Linear followed by LayerNorm and
SiLU). This is a deliberate stabilizer for off-policy value learning: LayerNorm controls feature
magnitudes so the bootstrapped critic does not amplify its own scale drift, and SiLU is a smoother
nonlinearity than ReLU. I keep it because the whole point of moving off-policy is to push more
gradient signal through the critic, and that is exactly the regime where the deadly triad bites
without normalization. Second, the optimizer and reuse rate stay matched to the substrate: AdamW
with weight decay `0.1`, cosine LR annealing, `num_updates=2` gradient steps per env step, a fast
target update `tau=0.1`. These are not SAC's classic small-tau, single-update settings — they are
the FastTD3-style fast-and-stable settings, and SAC inherits them here because the substrate is
built for them.

Before I trace the expectations, one more substrate detail deserves a sentence because it is where
this fill diverges most from the textbook and it bears on stability. The classic SAC evaluates the
soft value by sampling the next action from the current policy and subtracting its entropy; here
that sampling happens *inside* the distributional projection, and the entropy-augmented reward
`reward − α·next_log_prob` is what gets clamped to the `[−250, 250]` support and split across atoms.
That means a poorly-scaled `α` can push the augmented reward outside the support and get clamped,
silently distorting the target — which is one more reason the temperature must self-tune rather than
be hand-set, and why initializing it modestly at `0.2` and letting the dual update pull it down as
the policy sharpens is the safe regime. The entropy bonus also enters only at the next action, not
the current one, so the critic's cross-entropy target carries the future-entropy credit exactly as
the soft-value theory requires, and the actor's own entropy term is separate, in its loss. Keeping
those two entropy channels straight — one in the bootstrap, one in the policy objective — is what
makes this a faithful soft actor-critic rather than a reward-shaping hack.

Let me trace why this should beat PPO and where it might still fall short, because that is the bar.
Against PPO the advantage is structural and should be decisive: FastSAC reuses every transition out
of the replay buffer many times instead of discarding it after a few epochs, so on the same 12.8M
frames it extracts far more learning, and its exploration is the entropy-driven kind that suits
long-horizon coordination rather than PPO's undirected Gaussian wandering. So I expect FastSAC to
clear PPO on all three tasks, and to open the largest margin on walk and run — exactly the tasks
where PPO's sample-inefficiency hurt most — closing the shortfall the previous feedback diagnosed.
Where it might fall short is against a deterministic off-policy method on this particular substrate:
maximizing entropy over a high-dimensional humanoid action is genuinely hard, and a stochastic
actor that must hold a tanh-Gaussian over tens of dimensions can be noisier to optimize than a
deterministic actor whose exploration is supplied for free by the fleet of 128 parallel
environments. The entropy objective also adds the `−α log π` term to the distributional bootstrap,
which is one more moving part in a critic that is already chasing a projected target. So my
falsifiable expectation is two-sided: FastSAC should beat PPO clearly and across the board, but it
may not be the strongest off-policy fill — if a deterministic actor with the same distributional
critic and the same fast-and-stable substrate explores adequately through parallelism alone, it
could exploit the value function more aggressively and edge FastSAC out, especially on stand where
exploration is cheap and exploitation is what matters. That comparison is the next rung. The
distilled FastSAC fill of the editable contract is in the answer.
