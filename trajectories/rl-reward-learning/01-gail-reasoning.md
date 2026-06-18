The module slot is the whole game, and the scaffold default does nothing in it: an MLP over $[s,a,s']$
trained against a literal zero loss, so the reward net never moves and the policy ascends pure noise.
That is not even a floor I can interpret — it is a non-run. So the first real rung has to be the
simplest *learning* fill of this contract that turns demonstrations into a usable reward, and I want it
to be one whose failure mode, when it comes, tells me what the next rung must fix. Let me reason from the
imitation lineage to that fill.

The cheapest route is behavioral cloning: pool the expert $(s,a)$ pairs, fit $\pi(a\mid s)$, never touch
the environment. But cloning learns *no reward*, and the task scores a reward I then optimize with the
fixed PPO loop — and worse, the structural defect is covariate shift: I fit single-timestep decisions
under the expert's state distribution, the learner's own small errors push it into states it never
trained on, and the errors compound to $\varepsilon T^2$ regret. The lineage's answer to compounding
error is to stop scoring per-step decisions and score *whole trajectories* — recover a cost under which
the full expert trajectory is optimal, so wandering off the expert path accumulates cost and there is no
single-step trap. That is maximum-causal-entropy IRL. But its maximum-likelihood gradient has a positive
phase on demonstrated transitions and a negative phase that is an expectation under the *model's own*
trajectory distribution — the intractable partition function $Z$ over all trajectories — and the classic
way to estimate that negative phase is to solve a full RL problem in an inner loop for every cost update.
On a continuous-control budget that inner RL loop is exactly the expense I cannot pay. And the cheaper
escape, apprenticeship learning, matches expected features over a small linear cost class, which only
pins the expert down if a cost that truly explains the expert lives in that class — for a low-dimensional
linear class on MuJoCo it does not, so the recovered policy need not match the expert.

So I want the thing that sits between these: trajectory-level scoring (to kill BC's compounding error)
with no inner RL loop and no restrictive linear cost class (to kill IRL's and apprenticeship's costs).
The reframing that delivers it is occupancy-measure duality. Write a policy not as $\pi(a\mid s)$ but as
its occupancy measure $\rho_\pi(s,a)=\sum_t\gamma^t P(s_t=s,a_t=a\mid\pi)$ — the discounted distribution
of state-action pairs it actually visits. Expected cost becomes linear in $\rho$, the set of valid
occupancies is a convex polytope (Bellman flow constraints), and policies and occupancies are in
bijection. In these variables, $\psi$-regularized IRL followed by RL collapses, with no cost function
ever materialized, into one optimization: find the policy whose occupancy $\rho_\pi$ is closest to the
expert's $\rho_E$, where "closest" is the convex conjugate $\psi^*$ of whatever cost regularizer I chose.
Choosing the regularizer *is* choosing the distance. A constant regularizer forces exact matching
everywhere — intractable and degenerate on finitely many samples. A cost-class indicator gives the
linear apprenticeship learners that cannot imitate exactly. What I want is a regularizer whose induced
distance is a smooth, finite *divergence* between the two occupancy distributions, minimized only when
they are equal.

A divergence between two distributions is exactly what a *binary classifier* reads off: train a
discriminator to tell expert $(s,a,s')$ transitions from policy transitions, and the better it can
separate them, the further apart the occupancies are; when it is reduced to chance, they match. With the
logistic surrogate, the optimal-classifier objective is, up to a constant, the Jensen-Shannon divergence
between $\rho_\pi$ and $\rho_E$ — zero iff the occupancies are equal. So the imitation objective becomes
adversarial: a discriminator $D$ tries to separate expert from policy transitions; the policy (the
"generator") adjusts to fool it; the discriminator *is* the adaptive cost, supplied fresh each round, so
there is no inner RL loop and no linear cost class. When $D$ can no longer separate them, the policy has
matched the expert's occupancy — which, unlike BC, matches the full visited distribution rather than
per-step decisions, so it is not vulnerable to compounding covariate shift. This is the rung I will fill.

Now I have to land it in *this* scaffold, and the scaffold dictates several departures from the textbook
adversarial-imitation story that I have to respect, because the contract is rigid. First, the textbook
algorithm alternates a discriminator step with a *trust-region* (TRPO/natural-gradient) policy step,
because a raw, over-aggressive policy gradient on a noisy adversarial reward lets the policy lurch into
garbage. But the policy learner here is *fixed* — it is the scaffold's clipped-PPO loop, which I cannot
touch. PPO's clipped surrogate is itself a trust region (a flat spot in the loss past $\pm\epsilon$), so
the trust-region role is already filled by the substrate; my module only supplies the reward. Good — that
is one less thing to build, and one constraint I must not fight.

Second, the reward I hand PPO. The discriminator outputs a logit; the adversarial policy reward is the
log-odds $\log D - \log(1-D)$, but there are sign conventions and shaping variants. In this scaffold I
label expert transitions and policy transitions and train $D$ with binary cross-entropy on the logit
$f(s,a,s')$, and the reward I serve PPO is the imitation-library standard transform $-\log(1-D)$. With
$D=\sigma(f)$, $1-D=\sigma(-f)$, so $-\log(1-D)=-\log\sigma(-f)=-\,\texttt{logsigmoid}(-f)$ — that is the
exact form the task's fill uses in `compute_reward`. This branch gives expert-like transitions *positive*
reward (rather than merely avoiding a penalty), which is the right choice when the policy is being scored
on accumulated return: I want the policy paid for visiting expert-like states, not just spared a cost.

Third — and this is the part the scaffold forces that the generic method never mentions — the
discriminator can cheat. Early in training the policy visits states whose *raw observation scale* is wildly
different from the expert's (an untrained MuJoCo policy flails into extreme joint angles and velocities),
so the discriminator can classify expert-vs-policy on raw magnitude alone, learn nothing about behavior,
and hand the policy a reward that just says "be small." The fix is a running mean/std normalization on the
reward net's observation inputs — a Welford `_RunningMeanStd` updated each round on the freshest policy
rollout, so the discriminator sees whitened observations and must separate expert from policy on
*structure*, not scale. The task's fill builds exactly this: `RewardNetwork` holds a lazily-initialized
`_obs_rms`, normalizes both $s$ and $s'$ before the MLP, and `update_obs_norm` refreshes the stats from
the policy batch each `update()`. The action is left unnormalized (it is already bounded). This matters
because the imitation library's *tuned* GAIL configs require input normalization to work at all — without
it the discriminator saturates and the policy reward carries no usable gradient.

Fourth, the budget knobs. The scaffold's `irl_batch_size` and `n_irl_updates_per_round` are fixed outside
my editable region, and a single small discriminator step per round against a fast-moving PPO policy is
far too few — the discriminator falls behind the generator and the game destabilizes. Since I cannot edit
the args, I bump the effective amounts *inside* `update()`: an inner loop of a few discriminator gradient
steps (`_inner_updates`) on a several-times-larger effective batch (`_batch_mult`), resampling fresh
expert and policy minibatches each inner step. That is the only lever I have to keep the discriminator
roughly co-trained with the policy. The architecture itself is the default-shaped MLP — $[s,a,s']\to
256\to256\to1$ with ReLU — kept inside the parameter budget the scaffold enforces.

Let me put the pieces together as the literal fill. `RewardNetwork` is the discriminator over normalized
$[s,a,s']$, outputting a scalar logit. `IRLAlgorithm.compute_reward` returns $-\texttt{logsigmoid}(-f)$
under no-grad, the per-transition reward the rollout loop stores in place of the environment reward.
`update()` refreshes the obs-normalization stats from the policy batch, then runs the inner loop: sample
expert $(s,a,s')$ and policy $(s,a,s')$, compute logits for both, label expert $1$ / policy $0$, take a
BCE step, and log the discriminator accuracy. The fixed loop then normalizes the resulting buffer rewards
with its own running mean/std before the PPO update — a second normalization layered on top of mine that
I have to be aware of but cannot change.

Now reason about what this rung must do, because that is the point of running it first. Adversarial
imitation is the most *principled* method on the ladder — exact occupancy matching, the right divergence
— but it is also the most *fragile*, and the fragility is structural to the min-max game, not a bug I can
tune away from inside this contract. The discriminator and the policy chase each other: if the
discriminator gets too strong too fast, its logits saturate, $-\log(1-D)$ flattens, and the policy reward
loses its gradient — the policy stops improving and may drift. If the discriminator lags, the reward is
near-random and the policy ascends noise. The only damping I have is the inner-loop balance and the input
normalization; I have *no* trust-region control over the policy (PPO's clip is generic, not tuned to the
adversarial reward), and the reward is recomputed every step from a moving discriminator, so the PPO
value function is chasing a non-stationary target. On a clean, dense MuJoCo demo set this fragility can
dominate: the expert demonstrations are tight and easy to separate, so the discriminator tends to *win*
the game — high accuracy, saturated logits, a vanishing or adversarially-shaped reward — and the policy
can collapse to a degenerate gait that the saturated reward no longer penalizes.

So my falsifiable expectation, against which the next rung's numbers will be read: GAIL should be the
*weakest* learner on this ladder, and it should fail unevenly across environments by exactly the
saturation mechanism. On HalfCheetah — which has no terminal state, so an episode is a fixed-length run
and even a mediocre, partly-collapsed gait accumulates a nonzero return — I expect it to limp to a modest
score, the discriminator's instability hurting but not zeroing it. On Hopper and Walker2d — which
*terminate* the moment the body falls — I expect near-collapse: once the adversarial reward saturates or
shapes pathologically, the policy has no stable signal to keep the body upright, episodes end almost
immediately, and the return falls to near zero. If that split appears — HalfCheetah surviving in the
low-thousands while Hopper and Walker2d crater toward zero — then the diagnosis is confirmed: the
adversarial game is too unstable on clean demos, the reward saturates, and the next rung must make the
recovered reward *structured* enough to stop saturating and start carrying a usable, shaped signal across
terminal transitions. That is the gap I will hand to step 2.
