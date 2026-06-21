# Context: sampling-based stochastic optimal control for fast model-predictive loops

## Research question

A robot has a model of its own dynamics and a cost it wants to minimize over a short horizon,
and it has to act *now* — re-deciding its controls every few milliseconds as the world moves.
Concretely: the system is a continuous-state, control-affine stochastic process

```
dx = ( f(x,t) + G(x,t) u ) dt + B(x,t) dw,
```

with `f` the passive drift, `G` mapping controls into state changes, and `dw` Brownian noise;
the objective over a finite horizon is

```
min_u  E[ phi(x_T) + integral_t^T ( q(x,t) + (1/2) u^T R u ) dt ],
```

an arbitrary state cost `q` plus a quadratic control cost. The dynamics `f`, `G` may be
strongly nonlinear (a tire sliding through a corner, a quadrotor near obstacles) and the state
cost `q` may be non-convex or even discontinuous (an impulse the instant the body touches an
obstacle). The controller must run in a *receding-horizon* (model-predictive) loop: optimize
an open-loop control sequence in the background, execute one step of the current best guess,
shift, and re-optimize from the un-executed tail — fast enough to close a 40–50 Hz control
loop on real hardware.

How can derivative-free, forward-sampling methods be used to solve this stochastic optimal
control problem in real time?

## Background

**Stochastic optimal control and the HJB wall.** For the control-affine system above with the
quadratic-control / arbitrary-state cost, the value function `V(x,t)` satisfies the stochastic
Hamilton-Jacobi-Bellman PDE, and the optimal control is `u* = -R^{-1} G^T V_x`. `V` solves a
backward nonlinear PDE in the full state, and discretizing space and time makes both memory and
compute grow with dimension.

**The path-integral / free-energy reformulation (Kappen 2005; Theodorou & Todorov 2012).**
A line of work showed that, *under a specific relationship between the control cost and the
noise*, the nonlinear stochastic HJB can be turned into a *linear* PDE by an exponential
(logarithmic) change of variables `V = -lambda log(psi)`. Kappen's "linear theory for control
of non-linear stochastic systems" makes this concrete: assuming the noise covariance is
proportional to `R^{-1}` (so a single scalar `lambda` ties the two together, and the noise only
acts on actuated channels), the HJB collapses to a backward Chapman-Kolmogorov equation
`-partial_t psi = H psi`. A linear backward PDE with a terminal condition is exactly what the
**Feynman-Kac lemma** evaluates as an *expectation over trajectories of a forward diffusion*:

```
psi(x_0, t_0) = E_P[ exp( -(1/lambda) S(tau) ) ],     S(tau) = phi(x_T) + integral q dt,
```

where `P` denotes the *uncontrolled* dynamics (the system with `u = 0`). This is the "path
integral": the value function is recast as an integral over all trajectories, which can be
approximated by *forward sampling* — turn the machine on, let the noise generate trajectories,
average a function of their cost. Kappen also noted that `lambda` is a temperature, the
construction is a free energy, and Monte-Carlo or Laplace approximation can evaluate it in
high dimensions. Taking the gradient of `psi` with respect to the initial state turns this into
a path-integral expression for the optimal control itself,

```
u* dt = M · E_P[ exp(-S/lambda) B dw ] / E_P[ exp(-S/lambda) ],
```

where `M = R^{-1} G_c^T (G_c R^{-1} G_c^T)^{-1}`, which becomes `G_c^{-1}` when the
actuated block is square. This ratio of trajectory expectations is under the uncontrolled
measure `P`. The optimal control is an expectation evaluated by forward simulation.

**Importance sampling and changing the sampling distribution.** The standard technique for
estimating an expectation under one distribution using samples from another is importance
sampling: draw from a distribution `q` and reweight by the likelihood ratio `p/q`. Prior
path-integral work used **Girsanov's theorem** to change the *mean* (the drift) of the sampling
distribution — shift the controls you sample around, so that the rollouts cluster near a
promising trajectory. This gives an iterative scheme: sample around the current control guess,
reweight, update the guess. Policy Improvement with Path Integrals (PI^2; Theodorou, Buchli &
Schaal 2010) is the canonical instance — a generalized path-integral reinforcement-learning
method that updates control parameters by a *reward-weighted average* of explored variations.

## Baselines

**Differential dynamic programming / iLQG (Jacobson & Mayne 1970; Todorov & Li 2005;
Theodorou et al. 2010, stochastic DDP).** The dominant fast trajectory optimizer. Along a
nominal trajectory it takes a first- or second-order Taylor expansion of the dynamics and a
quadratic expansion of the cost, computes a second-order local model of the value function by a
backward Riccati-like sweep, and reads off a locally optimal control with a feedback gain; it
iterates this, re-linearizing each pass. It is powerful and converges fast near a good nominal.
It requires derivatives of the dynamics and a quadratic approximation of the cost.

**Open-loop path-integral control with mean-only importance sampling (PI^2; Theodorou et al.
2010, and the Girsanov-based iterative schemes).** Sample control perturbations around the
current open-loop sequence, roll the (uncontrolled-plus-perturbation) dynamics forward, weight
each rollout by `exp(-S/lambda)`, and set the new control to the weighted average of the
perturbations — a derivative-free, principled update straight out of the path-integral form.
The reweighting changes the *mean* of the sampling distribution; the exploration variance
is tied to the system's natural noise.

**Cross-entropy method (CEM) as a derivative-free planner.** A general black-box optimizer
widely used to plan over a learned dynamics model: maintain a Gaussian over action sequences,
sample a population, roll each out and score it, keep the top-`k` lowest-cost "elites", and
refit the Gaussian to the *empirical mean and standard deviation of the elites alone*; iterate.
It is derivative-free, trivially parallel, and adapts both its mean and its spread. The elite
step is a hard selection — every elite counts equally and every non-elite is discarded outright.

## Evaluation settings

The natural yardsticks for a real-time stochastic controller:

- **Simulated control platforms with nonlinear dynamics and challenging costs** — a cart-pole
  swing-up (state cost penalizing cart position/velocity and pole angle/angular velocity, with
  the control mapped through a first-order actuator), a miniature race car on an elliptical
  track using a nonlinear tire-interaction dynamics model (cost keeping the car on the track at
  a target forward speed), and a quadrotor flying through a field of cylindrical obstacles using
  a full nonlinear model (position, velocity, Euler angles, angular acceleration, rotor
  dynamics), with an impulse-style crash cost. These specifically stress non-quadratic and
  discontinuous costs.
- **Receding-horizon (MPC) protocol** — the controller runs at a fixed rate (e.g. 50 Hz, a
  20 ms control period), re-optimizing an open-loop sequence over a short horizon each step,
  executing one control, and warm-starting from the previous solution's tail. Knobs swept
  include the number of sampled rollouts and the exploration-variance level.
- **Derivative-free latent-model planning (goal-conditioned navigation).** A fixed, pre-trained
  latent world model that forward-simulates action sequences in representation space, plus a
  cost (objective) measuring distance of the predicted final latent to a goal latent. A planner
  proposes action sequences, the model rolls them out, the objective scores them. The navigation
  task is reaching a sampled goal in a two-room grid (wall + doorway) under a step budget, with
  several planning horizons (short, medium, long range); the metric is the fraction of episodes
  reaching the goal within a distance threshold. Provided interfaces: `unroll(obs_init,
  actions)` returns predicted latent states; `objective(encodings)` returns a per-sample cost
  (lower is better); `cost_function(actions, obs_init)` composes the two.
- **Baseline of record:** an MPC version of DDP on the simulation platforms; CEM on the
  latent-model planning task.

## Code framework

The substrate is a generic derivative-free trajectory optimizer plugged into a fixed,
pre-trained rollout model. What already exists: a model that forward-simulates a *batch* of
action sequences through the (learned) dynamics, and a scalar cost over the predicted states.
The planner samples candidate action sequences from a distribution over the horizon, scores
them with the rollout cost, and refines the distribution — and the *refinement rule* is the one
thing not yet settled. Below, the rollout model, the cost, the Gaussian sampling, and the
iterate-and-return skeleton are given; the single empty slot is how to turn a batch of scored
samples into the next distribution.

```python
import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, NamedTuple
from einops import rearrange


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None


class Planner(ABC):
    """Derivative-free planner over a fixed rollout model.

    `unroll(obs_init, actions)` forward-simulates a batch of action sequences
    through the (pre-trained) dynamics model and returns predicted states.
    `objective(encodings)` scores predicted states (lower cost is better).
    """

    def __init__(self, unroll: Callable, **kwargs):
        self.unroll = unroll
        self.objective = None

    def set_objective(self, objective: Callable):
        self.objective = objective

    def cost_function(self, actions: torch.Tensor, obs_init: torch.Tensor) -> torch.Tensor:
        # actions: [B, A, T] -> predicted states -> per-sample cost [B]
        predicted = self.unroll(obs_init, actions)
        return self.objective(predicted)

    @abstractmethod
    def plan(self, obs_init, t0=False, eval_mode=False, steps_left=None) -> PlanningResult:
        ...


class SamplingPlanner(Planner):
    """Sample action sequences from a per-timestep Gaussian, score them by the
    model rollout cost, and iteratively refine the distribution."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=500, n_iters=15, max_std=2.0, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.max_std = max_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, t0=False, eval_mode=False, steps_left=None):
        T = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # distribution over action sequences: per-timestep Gaussian
        mean = torch.zeros(T, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(T, self.action_dim, device=self.device)
        actions = torch.empty(T, self.num_samples, self.action_dim, device=self.device)

        losses = []
        for _ in range(self.n_iters):
            # sample a population of action sequences
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                T, self.num_samples, self.action_dim, device=self.device)

            # score every sampled sequence with the model rollout cost
            cost = self.cost_function(rearrange(actions, "t b a -> b a t"), obs_init)  # [B]
            losses.append(cost.min().item())

            # TODO: turn the scored population (actions, cost) into the next
            #       (mean, std). This refinement rule is what we will design.
            pass

        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
        )
```

The outer environment loop calls `plan()` each control step, executes the returned action(s),
and re-plans from the new observation. The empty slot is the distribution-refinement rule.
