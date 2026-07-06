The module slot is the whole game, and the scaffold default does nothing in it: an MLP over $[s,a,s']$
trained against a literal zero loss, so the reward net never moves and the policy ascends pure noise.
That is not even a floor I can interpret — it is a non-run. So the first real rung has to be the
simplest *learning* fill of this contract that turns demonstrations into a usable reward, and I want it
to be one whose failure mode, when it comes, tells me what the next rung must fix. Let me reason from the
imitation lineage to that fill.

The cheapest route is behavioral cloning: pool the expert $(s,a)$ pairs, fit $\pi(a\mid s)$, never touch
the environment. But cloning learns *no reward*, and the task scores a reward I then optimize with the
fixed PPO loop — and worse, the structural defect is covariate shift. Let me make the size of that defect
concrete rather than wave at it. Suppose the cloned policy makes a per-step decision error with
probability $\varepsilon$. On the first step it is on the expert distribution it trained on. But each
mistake nudges the learner into a state slightly off the demonstrated manifold, where it was never
supervised, so the next-step error rate there is not bounded by $\varepsilon$; the standard reduction
argument turns this into a regret that grows like $\varepsilon T^2$ rather than the $\varepsilon T$ you
would get if every state stayed in-distribution. On these MuJoCo episodes $T$ is on the order of $1000$
steps, so $T^2$ is $10^6$: even a small per-step disagreement of $\varepsilon\approx10^{-2}$ compounds
into an $O(10^4)$-scale drift budget, and once a Hopper or Walker body drifts far enough it falls and the
episode simply ends. That quadratic blow-up is the thing the whole imitation lineage was built to escape.
The lineage's answer is to stop scoring per-step decisions and score *whole trajectories* — recover a
cost under which the full expert trajectory is optimal, so wandering off the expert path accumulates cost
and there is no single-step trap. That is maximum-causal-entropy IRL. But its maximum-likelihood gradient
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
of state-action pairs it actually visits. It is worth checking the object is well-behaved before I build
on it: summing over all state-action pairs gives $\sum_{s,a}\rho_\pi=\sum_t\gamma^t=1/(1-\gamma)$, which
with the substrate's $\gamma=0.99$ is exactly $100$ — finite, so $\rho_\pi$ is a proper (rescalable)
distribution and "distance between two occupancies" is a meaningful thing to minimize. Expected cost
becomes linear in $\rho$, the set of valid occupancies is a convex polytope (Bellman flow constraints),
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
separate them, the further apart the occupancies are; when it is reduced to chance, they match. Let me
make the correspondence exact so I know what I am optimizing. With the logistic (cross-entropy) surrogate,
the discriminator that minimizes the binary loss against a fixed pair of distributions is
$D^*(x)=p_E(x)/(p_E(x)+p_\pi(x))$, and substituting that optimum back into the objective gives, up to the
additive constant $-\log 4$, exactly $2\cdot\mathrm{JS}(\rho_E,\rho_\pi)$ — twice the Jensen-Shannon
divergence between the two occupancies. JS is zero iff $\rho_E=\rho_\pi$ and strictly positive otherwise,
so the classifier's best achievable loss *is* a proper divergence that bottoms out only when the
distributions match. So the imitation objective becomes adversarial: a discriminator $D$ tries to separate
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

Second, the reward I hand PPO. The discriminator outputs a logit; the adversarial policy reward is some
transform of the log-odds $\log D-\log(1-D)$, but there are three sign conventions live here and the
choice is not cosmetic on a terminating body, so let me actually walk them. Writing $D=\sigma(f)$ with $f$
the logit: (a) the symmetric log-odds reward $\log D-\log(1-D)=f$ is centered at zero — expert-like
transitions positive, policy-like negative; (b) the "survival-biased" form $\log D=-\texttt{softplus}(-f)$
is always $\le 0$, a pure penalty that is largest (most negative) on policy-like transitions and pushes
the policy to *avoid* being caught rather than to *seek* expert states; (c) the imitation-standard
$-\log(1-D)$ is always $\ge 0$, largest on expert-like transitions. The choice interacts sharply with
termination. On a body that terminates when it falls, a reward that is mostly negative (form b) makes the
per-step return of *staying alive* negative, so the shortest path to the least-bad return can be to end
the episode early — a perverse incentive to fall. Form (c) is always positive, so accumulating more steps
is always weakly better and the policy is paid to keep the body up long enough to visit expert-like
states; that is the right choice when the score is accumulated true-reward return. So I take
$-\log(1-D)$. With $D=\sigma(f)$, $1-D=\sigma(-f)$, hence $-\log(1-D)=-\log\sigma(-f)=-\,\texttt{logsigmoid}
(-f)$ — the exact form the fill uses in `compute_reward`. Let me sanity-check its range on three logits.
At $f=0$ (indistinguishable) $D=\tfrac12$ and the reward is $-\log\tfrac12=0.693$; at $f=+2$ (expert-like)
$D=0.881$ and the reward is $-\log(0.119)=2.13$; at $f=-2$ (policy-like) $D=0.119$ and the reward is
$-\log(0.881)=0.127$. Monotone, strictly positive, and larger for expert-like transitions exactly as
wanted — and note already the seed of the failure mode: as the discriminator wins and $f\to-\infty$ on
policy transitions, that $0.127$ heads to $0$, and the policy's reward flattens to nothing.

Third — and this is the part the scaffold forces that the generic method never mentions — the
discriminator can cheat. Early in training the policy visits states whose *raw observation scale* is
wildly different from the expert's (an untrained MuJoCo policy flails into extreme joint angles and
velocities, so a coordinate that sits near $\pm1$ for the expert can be an order of magnitude larger for
the flailing learner), so the discriminator can classify expert-vs-policy on raw magnitude alone, drive
its BCE loss down without learning anything about *behavior*, and hand the policy a reward that just says
"be small." The fix is a running mean/std normalization on the reward net's observation inputs — a Welford
`_RunningMeanStd` updated each round on the freshest policy rollout, so the discriminator sees whitened
observations and must separate expert from policy on *structure*, not scale. The task's fill builds
exactly this: `RewardNetwork` holds a lazily-initialized `_obs_rms`, normalizes both $s$ and $s'$ before
the MLP, and `update_obs_norm` refreshes the stats from the policy batch each `update()`. The action is
left unnormalized (it is already bounded into a fixed range, so there is no scale to whiten). This matters
because the tuned adversarial-imitation configs require input normalization to work at all — without it
the discriminator saturates on scale in the first few rounds and the policy reward carries no usable
gradient.

Fourth, the budget knobs. The scaffold's `irl_batch_size` and `n_irl_updates_per_round` are fixed outside
my editable region, and a single small discriminator step per round against a fast-moving PPO policy is
far too few — the discriminator falls behind the generator and the game destabilizes. Since I cannot edit
the args, I bump the effective amounts *inside* `update()`: an inner loop of `_inner_updates=4`
discriminator gradient steps on a `_batch_mult=4`-times-larger effective batch, resampling fresh expert
and policy minibatches each inner step. That is $4\times4=16$ times the discriminator gradient signal per
round relative to the naive single-step fill, the only lever I have to keep the discriminator roughly
co-trained with the policy from inside a contract that fixes the args. The architecture itself is the
default-shaped MLP — $[s,a,s']\to256\to256\to1$ with ReLU. Let me confirm it fits the parameter cap the
scaffold enforces. For HalfCheetah/Walker2d the input is $2\cdot17+6=40$ wide, so the layers are
$40\cdot256+256=10496$, $256\cdot256+256=65792$, and $256+1=257$, totalling $76{,}545$ parameters; for
Hopper the input is $2\cdot11+3=25$ and the total is $72{,}705$. A single default-shaped MLP well under a
hundred-thousand parameters — comfortably inside the $\approx1.05\times$-largest-baseline cap, with room
to spare for a later rung that needs a bigger reward net.

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

It is worth tracing the early discriminator dynamics numerically, because it tells me what "the
discriminator wins" looks like in the logged accuracy and why the inner loop is my only brake. At
initialization the net outputs near-zero logits, so $D\approx\tfrac12$ on both classes, the BCE loss is
$-\log\tfrac12=0.693$, and the logged `disc_acc` sits near $0.5$ — the game is balanced and the policy
reward is a near-uniform $0.693$, uninformative but harmless. As training proceeds on clean demos the
discriminator's accuracy climbs; by the time it reads, say, $0.95$, the expert logits have gone strongly
positive and the policy logits strongly negative, the BCE has dropped toward $-\log(0.95)\approx0.05$, and
by the range trace above the policy transitions are being paid something like the $f=-2$ value, $0.127$,
already a quarter of the initial signal and still falling. There is no term in my objective that *rewards*
the discriminator for staying balanced — BCE always prefers sharper separation — so `disc_acc` drifting
toward $1.0$ is not a bug I can fix by training the discriminator differently; it is the equilibrium the
loss pulls toward on separable data. My $16\times$ inner-loop bump keeps the discriminator co-trained
rather than lagging, but co-trained here means "reaches high accuracy promptly," which is exactly the
saturated regime. That is the structural bind of GAIL on clean demos in one observation: the only knob I
have to stabilize the game (more discriminator training) is the same knob that accelerates the saturation
that kills the reward.

Let me put the pieces together as the literal fill. `RewardNetwork` is the discriminator over normalized
$[s,a,s']$, outputting a scalar logit. `IRLAlgorithm.compute_reward` returns $-\texttt{logsigmoid}(-f)$
under no-grad, the per-transition reward the rollout loop stores in place of the environment reward.
`update()` refreshes the obs-normalization stats from the policy batch, then runs the inner loop: sample
expert $(s,a,s')$ and policy $(s,a,s')$, compute logits for both, label expert $1$ / policy $0$, take a
BCE step, and log the discriminator accuracy. The fixed loop then normalizes the resulting buffer rewards
with its own running mean/std before the PPO update — a second normalization layered on top of mine that
I have to be aware of but cannot change. For now that layering is benign: my reward is bounded in
$(0,\infty)$ and does not drift wildly in scale, so the fixed buffer-norm just recenters it.

Now reason about what this rung must do, because that is the point of running it first. Adversarial
imitation is the most *principled* method on the ladder — exact occupancy matching, the right divergence
— but it is also the most *fragile*, and the fragility is structural to the min-max game, not a bug I can
tune away from inside this contract. The discriminator and the policy chase each other, and I can trace
the collapse quantitatively. Suppose the demos are clean and tight enough that the discriminator reaches
near-perfect separation: $D\to1$ on expert transitions, $D\to0$ on policy transitions. Then on every
policy transition the reward $-\log(1-D)\to-\log(1-0)=0$, so the *entire policy rollout* is paid a reward
that has collapsed to zero — a flat signal with no gradient telling the policy which of its transitions
were closer to expert. The policy is then ascending noise (post-normalization, an arbitrary recentering
of near-zero), exactly the non-run failure of the placeholder, only reached dynamically. The only damping
I have is the inner-loop balance and the input normalization; I have *no* trust-region control over the
policy beyond PPO's generic clip, and the reward is recomputed every step from a moving discriminator, so
the PPO value function is chasing a non-stationary target. On a clean, dense MuJoCo demo set this
fragility should dominate: the expert demonstrations are tight and easy to separate, so the discriminator
tends to *win* the game — high accuracy, saturated logits, a vanishing reward — and the policy can
collapse to a degenerate gait that the flattened reward no longer distinguishes.

So my falsifiable expectation, against which the next rung's numbers will be read: GAIL should be the
*weakest* learner on this ladder, and it should fail unevenly across environments by exactly the
saturation mechanism, split along terminal structure. Here the two body types diverge for a concrete
reason. HalfCheetah has no terminal state, so an episode is always a fixed-length run of the full budget
— even a mediocre, partly-collapsed gait keeps integrating a nonzero (if small) reward over all
$\sim1000$ steps, so the return floors out at a modest positive number rather than zero. Hopper and
Walker2d *terminate the moment the body falls*: once the adversarial reward saturates to a flat signal,
the policy has nothing telling it to keep the torso upright, the body topples within a handful of steps,
the episode ends almost immediately, and the accumulated return over those few steps is near zero. So I
expect a sharp split — HalfCheetah surviving in the low-thousands while Hopper and Walker2d crater toward
zero, with the ragged seed-to-seed jitter of a body that falls at slightly different first-stumble times.
If that split appears, the diagnosis is confirmed: the adversarial game is too unstable on clean demos,
the reward saturates, and the next rung must make the recovered reward *structured* enough to stop
saturating and start carrying a usable, shaped signal across terminal transitions. That is the gap I will
hand to step 2.
