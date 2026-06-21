# Context: data-driven trajectory optimization for offline control (circa 2021-2022)

## Research question

I am given a fixed batch of logged trajectories from a continuous-control system — a MuJoCo
locomotion agent, say — and no ability to interact further. The dynamics `s' = f(s, a)` are
unknown, the data is whatever some unknown behavior policy left behind, and I want, at each
state I find myself in, a good action to execute. The classical decomposition says: learn a
dynamics model from the data, then hand that model to a trajectory optimizer that searches for
an action sequence `a_{0:T}` maximizing `sum_t r(s_t, a_t)`. How can logged trajectory data
be turned into a planner for receding-horizon control over continuous-control tasks?

## Background

By this time deep generative modeling is moving fast, and a particular family has started to
dominate continuous-data generation.

**Diffusion / denoising generative models (Sohl-Dickstein et al. 2015; Ho, Jain & Abbeel
2020).** The idea is to define a fixed *forward* process that gradually corrupts a data point
`x_0` into noise over `N` steps, and learn the *reverse* process that denoises it back. The
forward step adds a little Gaussian noise and shrinks the signal,
`q(x^i | x^{i-1}) = N(x^i; sqrt(1 - beta_i) x^{i-1}, beta_i I)`, with a prespecified variance
schedule `beta_i`. Writing `alpha_i = 1 - beta_i` and `alpha_bar_i = prod_{j<=i} alpha_j`, the
composition of all forward steps has a closed form,
`q(x^i | x^0) = N(x^i; sqrt(alpha_bar_i) x^0, (1 - alpha_bar_i) I)`, so any noise level can be
sampled in one shot: `x^i = sqrt(alpha_bar_i) x^0 + sqrt(1 - alpha_bar_i) eps`,
`eps ~ N(0, I)`. The reverse process is learned as a Markov chain of Gaussians
`p_theta(x^{i-1} | x^i) = N(x^{i-1}; mu_theta(x^i, i), Sigma^i)` starting from a standard
normal prior `p(x^N) = N(0, I)`, and `theta` is fit by maximizing a variational bound on the
data log-likelihood. Ho et al. (2020) showed this bound decomposes into a sum of per-step
denoising terms, each a KL between Gaussians, and that with the right reparameterization the
whole thing collapses to a strikingly simple regression loss — train one shared network to
undo the noise at a randomly chosen level. Two properties of this family matter here. First,
sampling is *iterative*: you start from pure noise and refine, so the procedure is a sequence
of gradient-like denoising steps rather than a single forward pass. Second, the reverse
transitions are Gaussian and *modifiable* — the sampling distribution can be tilted by
external information injected at each step, and constraints can be imposed by clamping known
coordinates of `x` at every step (the way missing pixels are filled around observed ones).

**Score-based view (Song & Ermon 2019).** The denoising network can be read as an estimate of
the gradient of the log data density (the *score*) at each noise scale, and sampling resembles
Langevin dynamics that walks up the density.

**Noise schedules (Nichol & Dhariwal 2021).** The original linear `beta` schedule was tuned
for high-resolution images and destroys signal quickly. A cosine schedule,
`alpha_bar(i) = f(i) / f(0)` with `f(i) = cos^2( (i/N + s)/(1 + s) * pi/2 )` and a small
offset `s = 0.008`, keeps the signal-to-noise ratio better behaved, which matters when the
number of denoising steps is small.

**Trajectory optimization and control-as-inference (Witkin & Kass 1988; Tassa et al. 2012;
Levine 2018).** Behavioral synthesis by trajectory optimization defines a planning horizon `T`
and searches for the action sequence maximizing the summed reward; with a known model this is a
mature toolbox (DDP/iLQR, collocation, shooting). A useful reframing casts planning as
probabilistic inference: introduce optimality variables and condition on them, which turns
"find the best plan" into "sample from a distribution over plans." A structural fact from this
view is that decision-making is *anti-causal* — the action you take now depends on a future
goal, e.g. `p(s_1 | s_0, s_T)` makes `s_1` depend on a later state — so a strictly forward,
left-to-right factorization of a plan is at odds with how conditioning actually flows.

**The compounding-error and model-exploitation phenomena (Talvitie 2014; Chua et al. 2018;
Wang et al. 2019).** A learned single-step model `s_{t+1} = f_hat(s_t, a_t)`, rolled out
autoregressively to form a plan, accumulates error: each prediction feeds the next, so small
per-step errors compound and the `T`-step rollout drifts off the data manifold. And because the
model is differentiable and imperfect, a strong optimizer finds action sequences that score well
*under the model* by exploiting exactly those off-manifold regions — adversarial plans. This
is why much of contemporary offline/model-based control either leans on model-free value
functions instead of the trajectory-optimization toolbox, or restricts itself to weak,
gradient-free planners (random shooting, cross-entropy method) to avoid handing a powerful
optimizer a flawed model.

## Baselines

These are the prior approaches a new method would be measured against and would react to.

**Autoregressive single-step dynamics models for planning (e.g. PETS, Chua et al. 2018; world
models; ensemble MB-RL, Wang et al. 2019).** Learn `f_hat(s_t, a_t)` to predict the next state
(often with an ensemble for uncertainty), then plan by rolling it forward under candidate
action sequences and scoring them, typically with the cross-entropy method or random shooting.
Core idea: the model is a drop-in proxy for true dynamics, and planning is a separate search on
top.

**Model-free offline value methods (BC; CQL, Kumar et al. 2020; IQL, Kostrikov et al. 2021).**
Sidestep dynamics entirely: learn a policy and/or value function from the batch with a
conservatism penalty that keeps the policy near the data. Core idea: avoid planning, avoid
model exploitation, regress a controller directly.

**Sequence-modeling approaches (Decision Transformer, Chen et al. 2021; Trajectory Transformer,
Janner et al. 2021).** Treat an offline trajectory as a token sequence and fit an
autoregressive Transformer over interleaved states, actions, (and returns), then generate
actions by sampling/beam-search conditioned on a target return. Core idea: cast control as
sequence prediction; let a high-capacity sequence model capture the data distribution over
trajectories.

**Generative dynamics models more broadly (VAEs, normalizing flows, GANs, EBMs as world
models).** Various lines parameterize richer transition or trajectory distributions, keeping
the generative model as a proxy for dynamics that is then handed to a separate planner.

## Evaluation settings

The natural yardsticks already in use for offline continuous control:

- **D4RL offline MuJoCo locomotion (Fu et al. 2020).** Fixed datasets for `hopper-medium-v2`,
  `walker2d-medium-v2`, `halfcheetah-medium-v2` (the "medium" datasets are rollouts of a
  partially trained policy). Observations and actions are continuous; actions are bounded to
  `[-1, 1]`. The dataset is loaded once via `env.get_dataset()`; observations are normalized
  using dataset statistics.
- **Metric: D4RL normalized score** per environment (0 ~ random policy, 100 ~ expert),
  evaluated by rolling out the learned controller in the real simulator in a receding-horizon
  loop — plan from the current state, execute the first action, replan. A standard protocol is
  10 parallel environments times 10 episodes per environment. An aggregate across the three
  environments is taken as the geometric mean. Per-environment training wall-clock is also
  recorded.
- A horizon `T` (e.g. 32 for locomotion) defines how far ahead a plan reaches; the start of
  every plan must be the current observation.

## Code framework

The model plugs into a fixed offline-RL harness: a D4RL dataset loader that yields
length-`horizon` windows of `(obs, act)` from logged trajectories with normalized observations;
a standard optimizer and cosine-annealed learning-rate schedule; a training loop that draws
minibatches and calls an `update`; and an evaluation loop that, for each environment step,
builds a planning problem anchored at the current observation, asks the model for a plan, and
executes the plan's first action in the simulator (receding-horizon control). What is *not*
settled is the trainable planning model itself — what object it models, what network and
training objective it uses, and how it produces a plan. That is the single empty slot.

```python
import torch

from offline_harness import D4RLTrajectoryDataset, loop_dataloader, make_vector_env, set_seed


class PlannerModel:
    """Trainable model used by the offline planning harness.
    Owns the network, the training objective, and the procedure that produces a plan
    anchored at a given current state. The model internals are the empty slot."""

    def __init__(self, obs_dim, act_dim, horizon, device):
        self.obs_dim, self.act_dim, self.horizon = obs_dim, act_dim, horizon
        self.device = device
        # TODO: the model architecture and any state it requires.
        self.net = None
        self.optimizer = None  # standard optimizer over self.net once it exists

    def update(self, obs, act):
        # obs: (batch, horizon, obs_dim); act: (batch, horizon, act_dim)
        # TODO: the training objective; backprop and step.
        raise NotImplementedError

    @torch.no_grad()
    def plan(self, current_obs):
        # current_obs: (num_envs, obs_dim) -- every plan must start here
        # TODO: produce a plan anchored at current_obs and return the first action
        #       to execute, clipped to the valid action range.
        raise NotImplementedError


def train(model, dataloader, n_steps):
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, n_steps)
    step = 0
    for batch in loop_dataloader(dataloader):          # logged (obs, act) windows
        obs = batch["obs"]["state"].to(model.device)
        act = batch["act"].to(model.device)
        model.update(obs, act)
        lr_sched.step()
        step += 1
        if step >= n_steps:
            break


def evaluate(model, env):                              # receding-horizon control
    obs = env.reset()
    while not env.done():
        action = model.plan(obs)                       # plan from current obs
        obs = env.step(action)                         # execute first action, replan
```

The training loop supplies logged state and action windows; `update` decides how to model
them, and `plan` is where producing a plan anchored at the current state will live.
