## Research question

Offline reinforcement learning asks for a good policy from a *fixed* dataset
`D = {(s, a, r, s')}` collected by some unknown behavior policy `mu(a|s)`, with no further
interaction with the environment. The central pathology is well documented: any algorithm that
bootstraps — that fits a value function with a Bellman backup `Q(s,a) <- r + gamma * Q(s', a')`
— has to evaluate `Q` at next-state actions `a'`. If those actions come from the *learned*
policy rather than from the data, they drift out of the support of `mu`, and the function
approximator, never having seen them, returns arbitrary and usually *over*-estimated values.
The backup propagates those phantom values, the policy chases them, and training diverges. The
whole difficulty of offline RL is value estimation under this distribution shift.

A second, quieter problem rides along with the first. Even once a value function is trained
that genuinely captures the value of good in-support actions, that value function does not, by
itself, hand you a policy you can run. You still need a *policy extraction* step that turns the
value function into something that maps a state to an action — and that extracted policy must
faithfully represent whatever policy the value function is implicitly evaluating. If the
extraction step distorts that policy, the careful value learning is wasted.

The precise goal here is a single model-free offline policy algorithm that (1) learns values
without ever querying actions outside the data support, so it is stable; (2) actually performs
multi-step dynamic programming (stitches good sub-trajectories together) rather than just
cloning the behavior policy's own value; (3) is robust to hyperparameters, because in offline
RL one cannot tune online against the real return; and (4) extracts a policy that genuinely
matches the value function's implied behavior, even when that behavior is complicated. Each
existing method below achieves some of this; none achieves all of it at once.

## Background

By this time offline RL is dominated by value-based methods adapted from Q-learning and
off-policy actor-critic. The accepted framing of the distribution-shift problem
(Kumar et al. 2019; Levine et al. 2020) is that one must somehow *avoid trusting* the value of
out-of-distribution actions. Three broad families of fixes are on the table:

- **Policy constraints / regularization** — keep the learned policy close to `mu` so the
  actions it proposes stay in support. Implemented via explicit density models of `mu`
  (Wu et al. 2019; Fujimoto et al. 2019; Kumar et al. 2019), via implicit divergence
  constraints from advantage-weighted regression (Peters et al. 2007; Peng et al. 2019;
  Nair et al. 2020), or by adding a behavior-cloning term to the policy objective
  (Fujimoto & Gu 2021, TD3+BC).
- **Critic regularization** — directly push down the `Q`-values of OOD actions
  (Kumar et al. 2020, CQL), so even if they are queried they cannot win.
- **In-sample / implicit backups** — the most recent line, which sidesteps the OOD query
  *entirely* by replacing the explicit `max_{a'} Q(s', a')` with a statistic computable from
  dataset actions alone, using asymmetric loss functions in a SARSA-style backup.

The load-bearing tool for that third family is **expectile regression**, borrowed from
econometrics. The `tau`-expectile of a random variable `X` is the solution of an asymmetric
least-squares problem,

```
m_tau = argmin_m E_{x~X}[ L2^tau(x - m) ],   L2^tau(u) = |tau - 1(u < 0)| * u^2.
```

For `tau = 0.5` this is ordinary mean regression; for `tau > 0.5` it downweights residuals
where `x < m` and upweights residuals where `x > m`, so `m_tau` sits above the mean, and as
`tau -> 1` it approaches the *supremum* of `X`. Applied state-conditionally to the distribution
of `Q(s, a)` over dataset actions `a ~ mu(.|s)`, a high expectile estimates "the value of the
best in-support action at `s`" without ever naming that action — the maximization is performed
*implicitly* by the asymmetry of the loss, leveraging the function approximator's
generalization rather than an explicit `argmax`. A useful monotonicity holds: higher `tau`
gives a higher (and provably no-larger-than the constrained optimum) value estimate, so `tau`
trades off between cloning the behavior policy's mean value (`tau = 0.5`, SARSA) and
approximating constrained Q-learning (`tau -> 1`).

A second background ingredient is the **advantage-weighted regression** view of policy
extraction. Given a value function and advantage `A(s,a) = Q(s,a) - V(s)`, AWR fits a policy by
weighted maximum likelihood,

```
L_pi(phi) = E_{(s,a)~D}[ exp(beta * A(s,a)) * log pi_phi(a|s) ],
```

which is the closed-form solution of a reward-maximization problem subject to a KL constraint
to `mu`: the optimal constrained policy is `pi(a|s) ∝ mu(a|s) * exp(beta * A(s,a))`, and
weighting the behavior-cloning likelihood by `exp(beta*A)` regresses onto exactly that
reweighted behavior distribution. Crucially this only ever uses dataset actions, so it too
avoids OOD queries. `beta` interpolates between behavior cloning (`beta -> 0`) and greedy
maximization (`beta -> infinity`).

A third ingredient is the rise of **expressive generative models for the action
distribution**. Denoising diffusion probabilistic models (Sohl-Dickstein et al. 2015;
Ho et al. 2020; Song et al. 2020) model a distribution through a fixed forward noising chain
`q(a_t|a_{t-1}) = N(sqrt(1-beta_t) a_{t-1}, beta_t I)` and a learned reverse denoising chain.
With epsilon-prediction parameterization the training loss is a simple denoising regression,

```
L_mu(phi) = E_{t, eps, (s,a)~D}[ || eps - mu_phi(sqrt(abar_t) a + sqrt(1-abar_t) eps, s, t) ||^2 ],
```

and sampling runs the reverse chain from Gaussian noise. Diffusion models can represent sharply
multimodal continuous distributions that a single Gaussian cannot, and have begun to be used as
behavior-cloning and offline-RL action models. There is a known diagnostic caveat: a naive MLP
score network on low-dimensional continuous data fits the modes poorly and emits *outlier*
samples far from the data; increasing batch size and capacity helps but outliers persist, and
those outliers are exactly the OOD actions a `Q`-function may erroneously score high.

There is also a recurring empirical observation about expressive models trained with
importance-weighted objectives: such high-capacity models tend to raise the likelihood of *all*
training points regardless of their weight (Byrd et al. 2019; Xu et al. 2021), so an
importance-weighting signal that is supposed to skew the model toward high-advantage actions
gets washed out.

## Baselines

**Implicit Q-learning (IQL; Kostrikov, Nair & Levine, ICLR 2022).** The reference in-sample
method. Three networks: a twin `Q_theta`, its target `Q_targ`, and a value net `V_psi`. The
value net is fit to a `tau`-expectile of the target `Q` over dataset actions,

```
L_V(psi) = E_{(s,a)~D}[ L2^tau(Q_targ(s,a) - V_psi(s)) ],
```

so `V(s)` estimates a high in-support quantile-like statistic of the action-values without an
explicit max. The `Q`-net is then trained by a SARSA-style TD backup whose target uses `V(s')`
rather than `max_{a'} Q(s',a')`,

```
L_Q(theta) = E_{(s,a,s')~D}[ (r + gamma * (1-done) * V_psi(s') - Q_theta(s,a))^2 ].
```

Using `V(s')` (an action-expectile) instead of a per-sample `Q(s',a')` is deliberate: it keeps
environment stochasticity from contaminating the max, since a single "lucky" transition into a
good state should not be mistaken for an action that reliably achieves a high value. IQL proves
that the value defined recursively by these objectives is monotone in `tau`, is bounded above
by the support-constrained optimal `Q*`, and converges to it as `tau -> 1`; so the spectrum
runs from SARSA (`tau = 0.5`) to constrained Q-learning (`tau -> 1`). For the *policy*, IQL
extracts a unimodal conditional Gaussian via AWR (the `exp(beta*A) log pi` objective above).
IQL is fast, simple, and a small modification to a SARSA TD loop. **Gap:** the critic is trained with no
reference to any explicit policy, so it is unclear *which* policy the implicitly trained value
function actually corresponds to for an intermediate `0.5 < tau < 1`; and the policy is then
extracted with a unimodal Gaussian, a representation that may simply be unable to reproduce
whatever the critic implicitly evaluated. Because IQL deliberately decouples critic training
from the actor, the critic never adapts to the actor's representational limits — a strength for
stability, but it means a mismatch between the extracted policy and the implicit one is never
corrected.

**Conservative Q-learning (CQL; Kumar et al. 2020).** Adds a regularizer that pushes down
`Q`-values of actions sampled from the current policy while pushing up dataset-action values,
yielding a conservative lower bound on the value. Strong on locomotion. **Gap:** it does query
and penalize OOD actions, so it reintroduces a sensitivity to the regularizer weight, and in
practice needs substantially different tuning across domains (e.g. locomotion vs AntMaze) to
work well.

**TD3+BC (Fujimoto & Gu, 2021).** A minimalist recipe: standard TD3 off-policy actor-critic
plus a behavior-cloning term on the deterministic actor, with normalized states. **Gap:** the
deterministic, unimodal actor is constrained toward `mu` but cannot represent multimodal optima,
and the actor-critic coupling reintroduces the OOD-action query the actor's `max` performs.

**Diffusion Q-learning (DQL; Wang, Hunt & Zhou, ICLR 2023).** Parameterizes the actor as a
diffusion model and trains it with a TD3+BC-style objective: a behavior-cloning diffusion loss
plus a `Q`-maximization term that backpropagates the critic through the *entire sampling chain*.
A twin `Q` critic is trained jointly; at inference it samples `K` candidate actions from the
diffusion actor and reranks by `Q`. This is the closest diffusion-actor actor-critic baseline.
**Gap:** the diffusion actor and the critic are
tightly coupled — the `Q`-term flows gradients through the sampling chain into the actor — which
is computationally heavy (training time several times that of IQL) and reintroduces the
instability and per-task hyperparameter sensitivity that fully decoupled methods avoided; the
balance between the BC term and the `Q`-term is a sensitive knob.

**Select-from-Behavior-Candidates (SfBC; Chen et al. 2022).** Trains a diffusion behavior
model, then defines the policy by importance-reweighting samples from it with critic weights,
and trains the critic by value iteration using samples from that policy. A close relative in
spirit. **Gap:** the critic is trained on samples from the diffusion policy, so the behavior
model and critic learning are again entangled, which is expensive (very long training time) and
couples the two learning processes.

**One-step / behavior-value methods (Brandfonbrener et al. 2021; Peng et al. 2019).** Fit the
value of the behavior policy with a SARSA objective (equivalently `tau = 0.5`) and extract
greedily via AWR. Simple and stable on easy MuJoCo locomotion. **Gap:** with no multi-step
dynamic programming they cannot stitch sub-optimal trajectories, and collapse on tasks like
AntMaze that need it.

## Evaluation settings

The standard yardstick is the **D4RL** offline RL benchmark (Fu et al. 2020). The relevant
slices:

- **MuJoCo locomotion** — `halfcheetah`, `hopper`, `walker2d`, each in `medium`,
  `medium-replay`, and `medium-expert` dataset variants (`-v2`). Continuous state/action
  control; the `medium` datasets come from a partially trained policy, so good performance
  requires improving on the behavior policy, not just cloning it.
- **AntMaze** — `umaze`, `medium`, `large` in `play`/`diverse` variants; sparse-reward
  navigation that demands stitching sub-trajectories, the canonical multi-step-dynamic-
  programming stress test.
- **Maze2D** — a simpler navigation setting.

Protocol facts in use at the time: D4RL reports a **normalized score** (0 = random policy,
100 = expert) per task, averaged over evaluation episodes and several random seeds; rewards on
locomotion are standardized by the dataset's return range, and AntMaze rewards have `1`
subtracted, per the D4RL recommendation. The value-learning side uses twin critics,
target networks, Adam-style optimizers around learning rate `3e-4`, and soft target updates;
the implementation scaffold uses two-layer LayerNorm+Mish utilities for `Q` and `V`.
Training-time / walltime is itself reported and compared, because a practical offline method is
judged partly on how cheaply and how-tunably it reaches its score.

## Code framework

The available harness is the standard offline-RL training and evaluation loop already used for
the diffusion-actor baselines. The pre-existing pieces are: a D4RL transition dataset and
dataloader; a diffusion model abstraction that owns an epsilon-prediction score network and
exposes `update(action, state_condition)` (one mean-squared denoising-BC step) and
`sample(prior, condition, ...) -> (actions, log)` (run the reverse chain to draw actions);
the standard
twin-`Q` and value MLP modules with Adam optimizers and a Polyak-updated target; and the outer
loop that draws minibatches, takes gradient steps, periodically checkpoints, and at evaluation
rolls out a vectorized environment, mapping each current observation to a single action. What is
*not* settled is the learning rule for the critic, the training objective for the action model,
and how a single action is selected at inference — those are the empty slots.

```python
import torch

# --- existing primitives -----------------------------------------------------
# DiffusionActor: an epsilon-prediction DDPM over actions, conditioned on state.
#   .update(act, obs) -> {"loss": ...}    # one denoising regression step (behavior model)
#   .sample(prior, condition_cfg=obs, n_samples=..., ...) -> (actions, log)
# TwinQ(obs_dim, act_dim): two Q-heads; .both(obs, act) -> (q1, q2); __call__ -> min(q1, q2)
# ValueNet(obs_dim): V(obs) -> scalar value
# d4rl dataset/dataloader yielding batch = {obs, act, rew, next_obs, tml}


def critic_update(q_net, q_target, v_net, q_optim, v_optim, batch, args):
    """One gradient step on the value/Q networks from a minibatch of dataset
    transitions only (no actions outside the data support may be queried)."""
    obs, act = batch["obs"], batch["act"]
    next_obs, rew, tml = batch["next_obs"], batch["rew"], batch["tml"]
    # TODO: the in-sample value/critic learning rule we will design --
    #       fit the value and Q networks using dataset actions only,
    #       then Polyak-update the Q target.
    pass


def behavior_update(actor, batch):
    """One training step on the state-conditioned action model."""
    obs, act = batch["obs"], batch["act"]
    # TODO: the objective we train the action model with.
    return actor.update(act, obs)["loss"]


def train(actor, q_net, q_target, v_net, q_optim, v_optim, dataloader, args):
    for batch in dataloader:
        critic_update(q_net, q_target, v_net, q_optim, v_optim, batch, args)
        behavior_update(actor, batch)


@torch.no_grad()
def select_action(actor, q_net, v_net, obs, args):
    """Map a current observation to ONE action at evaluation time."""
    # TODO: the inference-time action-selection rule we will design,
    #       using the action model and the trained value networks.
    pass
```

The two learning rules and the inference-time selection rule are still empty.
