The module slot is the whole game, and the scaffold default does nothing in it: an MLP over $[s,a,s']$
trained against a literal zero loss, so the reward net never moves and the policy ascends pure noise.
That is not even a floor I can interpret — it is a non-run. So the first real rung has to be the
simplest *learning* fill of this contract that turns demonstrations into a usable reward, and I want it
to be one whose failure mode, when it comes, tells me what the next rung must fix. Let me reason from the
imitation lineage to that fill.

The cheapest route is behavioral cloning: pool the expert $(s,a)$ pairs, fit $\pi(a\mid s)$, never touch
the environment. But cloning learns *no reward*, and the task scores a policy under a reward I then
optimize with the fixed PPO loop — and worse, its structural defect is covariate shift. Suppose the cloned
policy makes a per-step decision error with probability $\varepsilon$. Each mistake nudges the learner off
the demonstrated manifold, where it was never supervised and the next-step error is no longer bounded by
$\varepsilon$; the standard reduction turns this into regret growing like $\varepsilon T^2$ rather than the
$\varepsilon T$ of a policy that stays in-distribution. On these MuJoCo episodes $T\approx1000$, so the
$T^2=10^6$ multiplier compounds even a small $\varepsilon\approx10^{-2}$ into an $O(10^4)$-scale drift
budget — and once a Hopper or Walker body drifts far enough it falls and the episode ends. That quadratic
blow-up is what the whole imitation lineage was built to escape: stop scoring per-step decisions and score
*whole trajectories* — recover a cost under which the full expert trajectory is optimal, so wandering off
the expert path accumulates cost and there is no single-step trap. That is maximum-causal-entropy IRL. But its maximum-likelihood gradient
has a positive phase on demonstrated transitions and a negative phase that is an expectation under the
*model's own* trajectory distribution — the intractable partition function $Z$ over all trajectories — and
the classic way to estimate that negative phase is to solve a full RL problem in an inner loop for every
cost update. On a continuous-control budget that inner RL loop is exactly the expense I cannot pay: PPO
here already needs a full rollout-and-update cycle to move the policy an increment, and nesting that
inside every cost-gradient step multiplies the total cost by the number of cost updates. And the cheaper
escape, apprenticeship learning, matches expected features over a small linear cost class, which only pins
the expert down if a cost that truly explains the expert lives in that class — for a low-dimensional
linear class on MuJoCo it does not, so the recovered policy need not match the expert.

So I want the thing that sits between these: trajectory-level scoring (to kill BC's compounding error)
with no inner RL loop and no restrictive linear cost class (to kill IRL's and apprenticeship's costs).
The reframing that delivers it is occupancy-measure duality. Write a policy not as $\pi(a\mid s)$ but as
its occupancy measure $\rho_\pi(s,a)=\sum_t\gamma^t P(s_t=s,a_t=a\mid\pi)$ — the discounted distribution
of state-action pairs it actually visits (a proper rescalable distribution, since $\sum_{s,a}\rho_\pi=
1/(1-\gamma)$ is finite). Expected cost becomes linear in $\rho$, the set of valid occupancies is a convex
polytope (Bellman flow constraints),
and policies and occupancies are in bijection. In these variables, $\psi$-regularized IRL followed by RL
collapses, with no cost function ever materialized, into one optimization: find the policy whose occupancy
$\rho_\pi$ is closest to the expert's $\rho_E$, where "closest" is the convex conjugate $\psi^*$ of
whatever cost regularizer I chose. Choosing the regularizer *is* choosing the distance. A constant
regularizer forces exact matching everywhere — intractable and degenerate on finitely many samples. A
cost-class indicator gives the linear apprenticeship learners that cannot imitate exactly. What I want is
a regularizer whose induced distance is a smooth, finite *divergence* between the two occupancy
distributions, minimized only when they are equal.

A divergence between two distributions is exactly what a *binary classifier* reads off: train a
discriminator to tell expert $(s,a,s')$ transitions from policy transitions, and the better it can
separate them, the further apart the occupancies are; when it is reduced to chance, they match. With the
logistic (cross-entropy) surrogate the loss-minimizing discriminator is $D^*(x)=p_E(x)/(p_E(x)+p_\pi(x))$,
and substituting it back gives, up to the additive constant $-\log 4$, exactly $2\cdot\mathrm{JS}(\rho_E,
\rho_\pi)$ — twice the Jensen-Shannon divergence, zero iff $\rho_E=\rho_\pi$ and strictly positive
otherwise. So the classifier's best achievable loss *is* a proper divergence that bottoms out only when
the distributions match, and the imitation objective becomes adversarial: a discriminator $D$ tries to separate
expert from policy transitions; the policy (the "generator") adjusts to fool it; the discriminator *is*
the adaptive cost, supplied fresh each round, so there is no inner RL loop and no linear cost class. When
$D$ can no longer separate them, the policy has matched the expert's occupancy — which, unlike BC, matches
the full visited distribution rather than per-step decisions, so it is not vulnerable to compounding
covariate shift. This is the rung I will fill.

Now I have to land it in *this* scaffold, and the scaffold dictates several departures from the textbook
adversarial-imitation story that I have to respect, because the contract is rigid. First, the textbook
algorithm alternates a discriminator step with a *trust-region* (TRPO/natural-gradient) policy step,
because a raw, over-aggressive policy gradient on a noisy adversarial reward lets the policy lurch into
garbage. But the policy learner here is *fixed* — it is the scaffold's clipped-PPO loop, which I cannot
touch. PPO's clipped surrogate is itself a trust region: past the $\pm\epsilon=\pm0.2$ clip the surrogate
objective goes flat, so a single update cannot move the action probability ratio outside $[0.8,1.2]$ no
matter how large the advantage the adversarial reward produces. The trust-region role is already filled by
the substrate; my module only supplies the reward. Good — that is one less thing to build, and one
constraint I must not fight.

Second, the reward I hand PPO. The discriminator outputs a logit $f$ ($D=\sigma(f)$); the adversarial
reward is some transform of the log-odds, and the three conventions live here are not cosmetic on a
terminating body. (a) The symmetric log-odds $\log D-\log(1-D)=f$ is centered at zero — expert-like
positive, policy-like negative; (b) the "survival-biased" $\log D=-\texttt{softplus}(-f)$ is always $\le
0$, a pure penalty largest on policy-like transitions; (c) the imitation-standard $-\log(1-D)$ is always
$\ge 0$, largest on expert-like transitions. Termination decides it. On a body that ends when it falls, a
mostly-negative reward (form b) makes the per-step return of *staying alive* negative, so the shortest path
to the least-bad return can be to end the episode early — a perverse incentive to fall. Form (c) is always
positive, so more steps are always weakly better and the policy is paid to keep the body up long enough to
visit expert-like states — the right choice when the score is accumulated true-reward return. So I take
$-\log(1-D)=-\log\sigma(-f)=-\,\texttt{logsigmoid}(-f)$, the form my `compute_reward` returns. Its range
already carries the seed of the failure mode: at $f=0$ the reward is $-\log\tfrac12=0.69$, at $f=+2$
(expert-like) it is $2.13$, at $f=-2$ (policy-like) only $0.13$ — so as the discriminator wins and
$f\to-\infty$ on policy transitions, that $0.13$ heads to $0$ and the policy's reward flattens to nothing.

Third — and this is the part the scaffold forces that the generic method never mentions — the
discriminator can cheat. Early in training the policy visits states whose *raw observation scale* is
wildly different from the expert's (an untrained MuJoCo policy flails into extreme joint angles and
velocities, so a coordinate that sits near $\pm1$ for the expert can be an order of magnitude larger for
the flailing learner), so the discriminator can classify expert-vs-policy on raw magnitude alone, drive
its BCE loss down without learning anything about *behavior*, and hand the policy a reward that just says
"be small." The fix is a running mean/std normalization on the reward net's observation inputs — a Welford
`_RunningMeanStd` updated each round on the freshest policy rollout, so the discriminator sees whitened
observations and must separate expert from policy on *structure*, not scale. So `RewardNetwork` holds a
lazily-initialized `_obs_rms`, normalizes both $s$ and $s'$ before the MLP, and `update_obs_norm` refreshes
the stats from the policy batch each `update()`. The action is left unnormalized (already bounded into a
fixed range, so there is no scale to whiten). This matters because the tuned adversarial-imitation configs
require input normalization to work at all — without it the discriminator saturates on scale in the first
few rounds and the policy reward carries no usable gradient.

Fourth, the budget knobs. The scaffold's `irl_batch_size` and `n_irl_updates_per_round` are fixed outside
my editable region, and a single small discriminator step per round against a fast-moving PPO policy is
far too few — the discriminator falls behind the generator and the game destabilizes. Since I cannot edit
the args, I bump the effective amounts *inside* `update()`: an inner loop of `_inner_updates=4`
discriminator gradient steps on a `_batch_mult=4`-times-larger effective batch, resampling fresh expert
and policy minibatches each inner step. That is $4\times4=16$ times the discriminator gradient signal per
round, the only lever I have to keep the discriminator roughly co-trained with the policy from inside a
contract that fixes the args. The architecture is the default-shaped MLP — $[s,a,s']\to256\to256\to1$ with
ReLU, about $76$k parameters — comfortably inside the $\approx1.05\times$-largest-baseline cap, with room
for a later step that needs a bigger reward net.

One more scaffold-forced choice hides in the input signature. The classical adversarial-imitation
discriminator reads $(s,a)$ and matches the state-action occupancy $\rho_\pi(s,a)$. But the contract here
hands `forward(state, action, next_state)` and stores $(s,a,s')$ transitions, so my discriminator reads
the full triple and matches the *transition* occupancy $\rho_\pi(s,a,s')$. Is that a problem? Under a
(near-)deterministic MuJoCo transition function $s'=T(s,a)$, $s'$ is an almost-deterministic function of
$(s,a)$, so $\rho_\pi(s,a,s')$ carries essentially the same information as $\rho_\pi(s,a)$ and matching one
matches the other — I am not changing what the method targets. What I *am* doing is handing the
discriminator a redundant, easy shortcut: if the policy's dynamics ever differed from the expert's it
could separate on $s'$ alone. That is a second reason the input normalization matters — with $s'$ whitened
the same way as $s$, the discriminator cannot lean on next-state magnitude as a free tell, and is pushed to
read the behavioral content in the triple rather than the redundant part. So I keep the $(s,a,s')$
signature the contract dictates, normalize both state slots, and accept that I am matching transition
occupancy, which on deterministic dynamics is the occupancy I wanted.

The discriminator dynamics carry a structural bind I should name now. Nothing in my objective *rewards* the
discriminator for staying balanced — BCE always prefers sharper separation — so `disc_acc` drifts from its
balanced $0.5$ toward $1.0$, the equilibrium the loss pulls toward on clean, separable demos. And the bind
is that my $16\times$ inner-loop bump, the only lever I have to keep the discriminator from lagging, keeps
it co-trained by making it *reach high accuracy promptly* — exactly the saturated regime where the policy
reward vanishes (recall the $0.13$-and-falling values as the policy logits go strongly negative). The one
knob that stabilizes the game is the same one that accelerates the saturation that kills it.

So the module is complete: the discriminator over normalized $[s,a,s']$, `compute_reward` returning
$-\texttt{logsigmoid}(-f)$ under no-grad, and `update()` refreshing the obs stats then running the BCE
inner loop. One layering to stay aware of: the fixed loop normalizes the buffer rewards with its own
running mean/std before the PPO update — a second normalization I cannot change, but benign here, since my
reward is bounded in $(0,\infty)$ and does not drift wildly in scale, so it just recenters.

Now, what this method must teach, since it runs first. Adversarial imitation is the most *principled* rung
— exact occupancy matching, the right divergence — but also the most *fragile*, and the fragility is
structural to the min-max game, not something I can tune away inside this contract. Suppose the demos are
clean and tight enough that the discriminator reaches near-perfect separation: $D\to1$ on expert
transitions, $D\to0$ on policy. Then on every policy transition the reward $-\log(1-D)\to0$, so the
*entire rollout* is paid a flat zero — no gradient telling the policy which transitions were closer to
expert. The policy ascends noise (post-normalization, an arbitrary recentering of near-zero), exactly the
placeholder's non-run reached dynamically. My only damping is the inner-loop balance and the input
normalization; I have *no* trust-region control beyond PPO's generic clip, and the reward is recomputed
every step from a moving discriminator, so the PPO value function chases a non-stationary target. On a
clean, dense MuJoCo demo set this should dominate: tight demos are easy to separate, the discriminator
tends to *win*, and the policy collapses to a degenerate gait the flattened reward no longer distinguishes.

So my falsifiable expectation, against which step 2's numbers will be read: GAIL should be the *weakest*
learner here, failing unevenly and split along terminal structure. HalfCheetah has no terminal state — an
episode always runs the full budget, so even a mediocre, partly-collapsed gait integrates a nonzero reward
over all $\sim1000$ steps and the return floors at a modest positive number. Hopper and Walker2d
*terminate the moment the body falls*: once the adversarial reward saturates flat, the policy has nothing
telling it to keep the torso upright, the body topples within a handful of steps, and the return over
those few steps is near zero. So I expect a sharp split — HalfCheetah surviving in the low-thousands while
Hopper and Walker2d crater toward zero, with ragged seed-to-seed jitter from bodies falling at slightly
different first-stumble times. If that split appears, the diagnosis holds and step 2 must make the
recovered reward *structured* enough to stop saturating and carry a usable, shaped signal across terminal
transitions.
