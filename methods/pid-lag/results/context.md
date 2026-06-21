## Research question

In safe reinforcement learning the agent must respect a safety budget not only at convergence
but throughout exploration and learning. The problem is formalized as a constrained Markov
decision process (CMDP): alongside the reward `R` there is a cost signal `C` (e.g. a count of
hazard contacts), and the goal is

```
max_pi  J(pi) = E_tau[ sum_t gamma^t R ]   s.t.   J_C(pi) = E_tau[ sum_t gamma^t C ] <= d
```

with a fixed cost limit `d`. A deep-RL agent is only partially optimized at each iteration,
the policy parameters `theta` drift as reward-learning proceeds, and the upward pressure that
reward-learning puts on cost is itself changing. The question is how to design a mechanism
that updates the penalty coefficient `lambda` from the stream of measured episodic costs
so that constraint satisfaction is maintained during training.

## Background

**The constrained-MDP / Lagrangian frame.** The dominant way to handle the CMDP is to dualize it.
Introduce a nonnegative multiplier `lambda` and form the Lagrangian `L(theta, lambda) = J(pi) -
lambda (J_C(pi) - d)`; the constrained problem is handled as a saddle problem over the policy
parameters and the multiplier by alternating policy-gradient steps with projected multiplier
updates. `lambda` acts as a *learned penalty coefficient* that trades reward against cost, and is
updated from the running constraint violation. This reduction is attractive because it is simple and
turns safe RL into ordinary policy optimization on a reweighted objective; it is the popular
baseline in deep safe RL.

**The continuous-time dynamics behind it (Platt & Barr 1988).** Platt and Barr studied the
differential version of the multiplier method for an equality-constrained problem `min f(x) s.t.
g(x)=0`. Plain gradient descent on the Lagrangian energy `L = f + lambda g` fails, because a
constrained extremum is a *saddle point* of `L`, not a minimum: freeze `x` and the energy can be
driven down by sending `lambda` to plus or minus infinity, so descent does not converge. Their
fix — the basic differential multiplier method (BDMM) — is to do gradient *ascent* on `lambda`
(a sign flip), which turns the constrained extrema into attractors:

```
x_i_dot    = - df/dx_i - lambda dg/dx_i
lambda_dot = + alpha g(x)
```

Differentiating the first equation and substituting the second collapses the coupled first-order
system into a single second-order equation per coordinate — a damped mass-spring oscillator:

```
x_ddot + A x_dot + alpha g(x) grad g = 0,     A = grad^2 f + lambda grad^2 g
```

with `A` the matrix multiplying velocity (the weighted sum of Hessians of objective and constraint)
and `alpha g(x) grad g` the restoring term toward the constraint surface. Defining the Lyapunov
energy `E = (1/2) ||x_dot||^2 + (alpha/2) g(x)^2`, its time derivative is
`E_dot = - x_dot^T A x_dot`, so when `A` is positive definite the energy dissipates; under the
usual nondegeneracy assumptions, the limiting state satisfies the constraint. The multiplier
integrates the constraint: `lambda_dot = alpha g(x)` means `lambda` accumulates the running
violation over time. The system is an oscillator that oscillates about the constraint subspace as
it converges, with the frequency rising and the settling time lengthening as the gain `alpha` is
increased.

**Augmenting with a penalty (MDMM).** Platt & Barr also showed BDMM is compatible with a quadratic
penalty: add `E_penalty = (c/2) g(x)^2`, i.e. a force `-c g grad g`, to the `x` dynamics. The
damping matrix becomes

```
A = grad^2 f + lambda grad^2 g + c grad g grad^T g + c g grad^2 g
```

The `c grad g grad^T g` term is the outer product of a vector with itself — positive semidefinite,
so it can only increase, never decrease, the ordered damping eigenvalues and enlarges the region of
positive damping around the minimum, improving convergence. The augmentation also (a) modifies
the primal dynamics `x_dot` (the penalty force must be implemented there), and (b) brings in the
extra `c g grad^2 g` term, which carries a Hessian of the constraint that need not be positive
semidefinite yet shares the same coefficient `c`.

**Oscillatory behavior in practice.** In deep safe-RL runs with the integral-only multiplier, the
cost and the multiplier can oscillate throughout training, with the cost curve and the `lambda`
curve sitting roughly a quarter-cycle out of step.

**Feedback control, generically.** A discrete-time system with feedback control is written
`x_{k+1} = F(x_k, u_k)`, measurement `y_k = Z(x_k)`, control `u_k = h(y_0, ..., y_k)`; the control
rule `h` reads past and present measurements and drives the output to a setpoint. A system is
"control-affine" when `F(x, u) = f(x) + g(x) u`, a form especially amenable to analysis. These are
standard objects; the question of which control rule `h` to use is open.

## Baselines

**Traditional Lagrangian safe RL (Altman; RCPO, Tessler et al. 2019; Chow et al. 2019).** The
multiplier for an inequality constraint is updated by a projected step on the measured violation:

```
lambda_{k+1} = ( lambda_k + K_I (J_C - d) )_+
```

and the policy is optimized on the reweighted objective `J - lambda J_C`. Convergence to an
optimal feasible policy has been shown under a two-timescale assumption: `lambda` must be updated
*more slowly* than `theta`. The update accumulates the running constraint error and is controlled
by a single learning rate `K_I`.

**Constrained Policy Optimization (CPO, Achiam et al. 2017).** A trust-region policy-search method
with a per-iteration near-constraint-satisfaction guarantee, built on a bound relating the returns
of two nearby policies; it adds a projection of the policy parameters back into the feasible set.

**Lyapunov-based methods (Chow et al. 2018, 2019) and interior-point methods (Liu et al. 2019).**
Lyapunov approaches combine a projection step with action-layer interventions (a safety layer);
interior-point methods add logarithmic barrier terms to the objective.

**The quadratic penalty method (Hestenes 1969; Powell 1969).** Replace the constraint by adding
`(c/2) g(x)^2` to the objective. As shown above (MDMM), this improves the damping of the learning
dynamics by adding the positive-semidefinite term `c grad g grad^T g` to the damping matrix.

## Evaluation settings

- **Safety-Gym (Ray, Achiam, Amodei 2019)** — robot navigation on the MuJoCo simulator. Robots
  range from a simple point mass and a non-holonomic car to a 12-jointed quadruped, moving in an
  open arena. Reward has a small dense component for moving toward a goal and a large sparse
  component for reaching it; reaching a goal spawns a new one and the episode continues to a 1000-
  step horizon. Hazards placed randomly each episode induce cost on contact and often lie on the
  path to the goal, so high reward and low cost are in tension. The agent senses goal and hazard
  positions through a coarse LIDAR-like sensor plus internal joint readings. Representative tasks:
  `PointGoal1`, `CarGoal1`/`CarButton1`, `PointButton1`, `DoggoGoal2`, `DoggoButton1`.
- **Metrics.** Episode return (reward), higher is better; episode cost (the non-discounted
  episodic sum of the cost signal), lower is better, with a cost limit that is the constraint
  threshold (e.g. 25 for the point-robot navigation tasks). The finite horizon allows using the
  non-discounted episodic cost directly as the constraint quantity and as the controller input.
- **Protocol.** A leading on-policy deep-RL optimizer (PPO, Schulman et al. 2017) as the policy-
  learning loop, with reward and cost returns/advantages available to the constraint mechanism;
  the mechanism receives an estimate of the average episodic cost over the batch once per iteration.
  Because the trade-off between reward and constraint violation matters, natural diagnostics include
  training curves of return, episodic cost, per-iteration constraint violations, final return, and
  sweeps over the baseline multiplier learning rate.

## Code framework

The mechanism plugs into a fixed PPO-style on-policy harness. The rollout loop, the reward and
cost critics, the optimizer, the environment interface, and the registration plumbing already
exist. Two slots are left open, and exactly what fills them is what is to be designed: how the
penalty coefficient `lambda` is updated from the measured cost each iteration, and how the reward
and cost advantages are combined into the single surrogate the policy gradient is taken through.

```python
import numpy as np
import torch


class ConstrainedPPO:
    """PPO with a safety constraint. The rollout, the reward/cost critics (V_R, V_C),
    the optimizer, and the env interface are fixed. Only the constraint-handling
    mechanism is open: a per-iteration multiplier update, and an advantage combination."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cost_limit = cfg.cost_limit          # the constraint threshold d
        # any state the multiplier rule needs is initialized here
        self._lambda = 0.0
        self._init_controller()

    def _init_controller(self):
        # TODO: initialize whatever state the multiplier-update rule keeps.
        pass

    def _update_multiplier(self, ep_cost):
        # ep_cost: estimated average episodic cost J_C this iteration.
        # TODO: from the stream of measured costs (and any state kept above),
        #       set self._lambda >= 0.
        pass

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        # adv_r, adv_c: reward and cost advantages for the sampled batch.
        # TODO: combine them into one surrogate advantage using self._lambda.
        pass

    def _update(self):
        # fixed PPO outer step
        ep_cost = self.logger.get_stats('EpCost')        # mean episodic cost this iteration
        assert not np.isnan(ep_cost)
        self._update_multiplier(ep_cost)                 # set the penalty coefficient
        # ... existing PPO epochs over minibatches, taking the policy gradient
        #     through the surrogate advantage from _compute_adv_surrogate(adv_r, adv_c) ...
        super()._update()
```

The outer step hands the mechanism one cost measurement per iteration and asks for a coefficient
and an advantage blend; everything else around it is settled.
