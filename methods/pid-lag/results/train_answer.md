In safe reinforcement learning the agent has to respect a safety budget while it is still learning, not merely once it has converged. The problem is a constrained Markov decision process: alongside the reward there is a cost signal, and we want $\max_\pi J(\pi)$ subject to $J_C(\pi) = \mathbb{E}_\tau\big[\sum_t \gamma^t C\big] \le d$ for a fixed cost limit $d$. The textbook move is to dualize — introduce a nonnegative multiplier $\lambda$, optimize the reweighted objective $J - \lambda(J_C - d)$, and grow $\lambda$ from the running violation. That solver is correct in the limit: it converges to a feasible policy. But it is useless for the requirement that actually matters here, which is constraint satisfaction at *every* iterate. When I watch a real run, the cost does not sit politely at $d$ — it swings above and below the limit for the whole training run. The reason is that a deep-RL agent is only partially optimized at each iteration; the parameters $\theta$ drift as reward-learning pushes upward on cost, and the dual update is always chasing a moving target. The existing alternatives do not fix this cleanly. Constrained Policy Optimization adds a trust-region projection of the policy parameters with a per-iteration guarantee, but in practice it needs a costly backtracking line search and simple Lagrangian methods matched or beat it. Lyapunov and interior-point methods bolt on a projection plus a safety layer, or logarithmic barrier terms; they give mixed gains at nontrivial implementation cost, are provably suboptimal, need their barrier strength tuned, and typically assume already-feasible iterates — fragile under random initialization and noisy cost estimates. The plain Lagrangian method, $\lambda \leftarrow (\lambda + K_I(J_C - d))_+$, is the appealing baseline because it keeps safe RL as ordinary policy optimization, but it has exactly one knob $K_I$, and that single gain forces a bad trade: larger $K_I$ damps the oscillation but degrades reward.

So I want to understand *why* it oscillates rather than just damping it by fiddling with the learning rate. Stare at the update: $\lambda$ is a running accumulation of the violation — each iteration adds the current error $J_C - d$ to a stored quantity. In continuous time this is Platt and Barr's $\dot\lambda = \alpha\, g(x)$, with $g$ the constraint; the multiplier is literally the time-integral of the constraint error. And the empirical signature in my runs is telling: the cost curve and the $\lambda$ curve oscillate but offset, with $\lambda$ peaking about a quarter cycle after cost does. A pure integrator turns a sinusoid at its input into a sinusoid shifted by exactly ninety degrees — integrating $\sin$ gives $-\cos$ — so that quarter-cycle lag is not a quirk of deep RL, it is the textbook fingerprint of an integral controller seen in the wild. That reframes everything: the safe-RL training loop *is* a feedback control system. The cost limit $d$ is the setpoint, the measured episodic cost $J_C$ is the output I want to hold there, the multiplier $\lambda$ is the control input I get to set, and the policy-optimization step — PPO grinding $\theta$ forward to chase reward — is the plant, a nonlinear and frankly unknown map $\theta_{k+1} = F(\theta_k, \lambda_k)$ from $\lambda$ to next iteration's cost. The traditional method is just one impoverished choice of control rule: integral-only. The question stops being "how do I tune $K_I$" and becomes "what control rule should I use," and the obvious answer is the most-used controller there is.

I propose PID Lagrangian: update the multiplier with a proportional-integral-derivative controller on the cost-constraint error, rather than the integral-only subgradient step. The mechanism rests on tracing each of the three terms through Platt and Barr's collapse of the coupled first-order multiplier dynamics into one second-order equation per coordinate, $\ddot x + A\,\dot x + \alpha\, g(x)\,\nabla g = 0$, a damped mass-spring oscillator with damping matrix $A = \nabla^2 f + \lambda \nabla^2 g$ and restoring force $\alpha\, g\,\nabla g$ toward the constraint surface, whose Lyapunov energy $E = \tfrac12\|\dot x\|^2 + \tfrac{\alpha}{2}g(x)^2$ dissipates as $\dot E = -\dot x^\top A\, \dot x$ exactly when $A$ is positive definite. The diagnosis of why integral-only rings comes straight from this: integral control has memory but no reflexes. It reacts only to *accumulated* error, so $\lambda$ rises only after violation has been piling up — late by construction. By the time $\lambda$ has integrated up enough to push cost back down, cost has shot well past $d$; cost then falls but is still above $d$, so $\lambda$ keeps climbing, overcorrects, drives cost below $d$, the error flips sign, $\lambda$ comes down, cost rebounds — a limit cycle. And cranking the only knob $\alpha$ (their gain, my $K_I$) raises the oscillation frequency and *lengthens* the settling time. One knob, bad trade.

The proportional term, $K_P(J_C - d)$, is an instantaneous reaction to the current violation, no waiting for accumulation. To see its effect on the dynamics it must enter $\dot\lambda$ as the term that lets $\lambda$ track $g$ instantaneously — its rate carries $g$'s rate — so $\dot\lambda = \alpha\, g + \beta\, \dot g$. Redoing the collapse with this piece leaves the restoring force $\alpha\, g\,\nabla g$ untouched (the equilibrium is still $\dot x = 0,\, g = 0$) and changes only the damping matrix, which gains $\beta\, \nabla g\, \nabla^\top g$:
$$\ddot x + \big(A + \beta\, \nabla g\, \nabla^\top g\big)\,\dot x + \alpha\, g(x)\,\nabla g = 0.$$
That added term is the outer product of $\nabla g$ with itself, positive semidefinite by construction since $v^\top(\nabla g\,\nabla^\top g)\,v = (\nabla g \cdot v)^2 \ge 0$ for any $v$ — it can only add nonnegative damping in the $\nabla g$ direction, and the energy derivative now subtracts $\beta(\nabla g\cdot \dot x)^2$. So proportional control supplies damping, falling straight out of the same energy argument. This is satisfying because it is the *same* $\beta\,\nabla g\,\nabla^\top g$ term the quadratic penalty method produces — but the penalty buys it twice over. The penalty force lives in the primal dynamics $\dot x$, modifying the objective the policy optimizes; my proportional term touches only the $\lambda$ update and leaves the Lagrangian alone. And the penalty's damping comes bundled with an extra $c\, g\, \nabla^2 g$ term carrying a constraint Hessian that need not be positive semidefinite, sharing the single coefficient $c$ so the help cannot be dialed in without the possible harm. Proportional control gives the clean positive-semidefinite damping with none of that baggage.

The derivative term lets me anticipate: if cost is rising fast toward the limit, raise $\lambda$ now, before cost crosses $d$. A term acting on the *trend* of the constraint enters $\dot\lambda$ one derivative higher than the proportional term, as $\dot\lambda = \alpha\, g + \gamma\, \ddot g$. Grinding out $\ddot g$ produces a complication — it contains $\ddot x$, so the acceleration appears on both sides. After substitution the $\ddot x$ terms couple through $B = I + \gamma\, \nabla g\,\nabla^\top g$, identity plus a positive-semidefinite outer product, hence positive definite and invertible; left-multiplying by $B^{-1}$ decouples them:
$$\ddot x + B^{-1}A\,\dot x + \big(\alpha\, g(x) + \gamma\,\dot x^\top \nabla^2 g\,\dot x\big)\,B^{-1}\nabla g = 0.$$
Two effects, genuinely different from proportional. First, $B^{-1}$ has eigenvalues at most one (strictly smaller in the $\nabla g$ direction when $\gamma > 0$), rescaling the restoring direction and reshaping the damping operator to $B^{-1}A$. Second — the predictive part — the scalar $\gamma\,\dot x^\top \nabla^2 g\,\dot x$ now multiplies $B^{-1}\nabla g$, a force quadratic in velocity and modulated by the curvature of the constraint along the direction of motion, carried with a negative sign. Moving where $g$ curves upward, the term pushes along $-B^{-1}\nabla g$ to decrease $g$; if I am already in violation and the curvature says it is about to worsen, this anticipatory force adds to the ordinary restoring force and I brake harder before overshooting. It reads the trend and acts ahead of the violation. It is also the most delicate term — second derivatives of a noisy signal are wild — which I flag for the estimation stage. Because the constraint is one-sided, I rectify it: use the positive part so it brakes cost *increases* but stays silent when cost is falling, which is the good direction.

I keep the integral term because only it eliminates steady-state violation. When the system has settled, the error $J_C - d \approx 0$, so the proportional term vanishes and the derivative term (cost not changing) vanishes too; if those were the only terms, $\lambda$ would collapse to zero and cost would drift back up. The integral has *remembered* the accumulated history and holds a nonzero $\lambda$ that pins cost exactly at $d$. Proportional and derivative shape the transient — the very thing the baseline fails at — while integral guarantees the asymptote. And setting $K_P = K_D = 0$ recovers the traditional Lagrangian method exactly, so this is a strict generalization, widening a one-parameter family of update rules to three.

Discretizing for the per-iteration rule, each iteration $k$ receives an estimate of episodic cost $J_C$, the error is $\Delta = J_C - d$, the integral is the running sum, the proportional reaction is $\Delta$, and the derivative is the rectified change in cost, with two projections. The outer $(\cdot)_+$ on $\lambda$ enforces multiplier nonnegativity ($\lambda$ is a KKT multiplier for an inequality). The inner $(\cdot)_+$ on the integral is anti-windup: I must not let the integrator bank a big *negative* reservoir during a long feasible stretch, because then when a violation finally arrives the controller would first have to climb out of that hole before responding — the exact delayed reaction I am curing. Two further realities shape the shipped form. First, step size: the policy gradient under the Lagrangian is $\nabla J - \lambda\nabla J_C$, and a large $\lambda$ blows up its magnitude so a fixed learning rate takes a destabilizing step. Since $\arg\max_\theta(J - \lambda J_C) = \arg\max_\theta \tfrac{1}{1+\lambda}(J - \lambda J_C)$ — dividing by the positive constant $1+\lambda$ does not move the argmax but *does* normalize the gradient scale, tending to the pure cost-reduction direction $-\nabla J_C$ as $\lambda \to \infty$ rather than a divergent multiple — the inner PPO loop should blend the reward and cost advantages as
$$\text{adv} = \frac{\text{adv}_r - \lambda\,\text{adv}_c}{1 + \lambda}.$$
I apply this rescaling even to the plain Lagrangian baseline so the comparison is about the controller, not a luckier step size, and I keep *separate* reward and cost critics $V_R, V_C$ rather than folding everything into one critic on $r + \lambda c$, because $\lambda$ now changes rapidly and a single critic on the moving target would be perpetually stale. Second, estimation: $J_C$ is a sample average over a minibatch of finished trajectories, so the proportional and especially derivative terms would react to noise — a one-step difference of two noisy estimates is mostly noise. I smooth with an exponential moving average of the error for the proportional input and of the cost for the derivative (factors near one), and for the derivative I difference the smoothed cost against a *delayed* copy from $d_{\text{delay}}$ iterations back, a clean finite difference over a window rather than a literal second derivative; the integral is left un-smoothed since it is already a low-pass accumulation. The integral gain is folded into the integral update itself so the output line stays clean, and the optional clamps cap the normalized output to $[0,1]$ under `diff_norm` or, otherwise, cap the raw penalty at `penalty_max`. A separable reward-scale issue remains: scaling all rewards by $\rho$ forces the optimal $\lambda^\star$ to scale by $\rho$, since $\lambda$ trades reward units against cost units; weighting the cost gradient by $\beta = \|\nabla J\| / \|\nabla J_C\|$ makes $\lambda = 1$ always mean "reward and cost contribute equal-magnitude gradients," so one set of gains transfers across reward scales and $\lambda_0 = 1$ is the natural initialization. That is a complementary knob; the core contribution is the PID controller on $\lambda$.

```python
from __future__ import annotations
from collections import deque

import numpy as np
import torch


class PIDLagrangian:
    """PID controller on the Lagrange multiplier for an inequality safety constraint.
    K_P = K_D = 0 recovers the traditional integral-only Lagrangian update."""

    def __init__(
        self,
        pid_kp: float,
        pid_ki: float,
        pid_kd: float,
        pid_d_delay: int,
        pid_delta_p_ema_alpha: float,
        pid_delta_d_ema_alpha: float,
        sum_norm: bool,
        diff_norm: bool,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: float,
    ) -> None:
        self._pid_kp = pid_kp
        self._pid_ki = pid_ki
        self._pid_kd = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self._sum_norm = sum_norm
        self._diff_norm = diff_norm
        self._penalty_max = penalty_max
        self._pid_i = lagrangian_multiplier_init   # integral accumulator (gain folded in)
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p = 0.0                          # EMA of the error (proportional input)
        self._cost_d = 0.0                           # EMA of the cost (for the derivative)
        self._cost_limit = cost_limit
        self._cost_penalty = 0.0                     # the multiplier (controller output)

    @property
    def lagrangian_multiplier(self) -> float:
        return self._cost_penalty

    def pid_update(self, ep_cost_avg: float) -> None:
        delta = float(ep_cost_avg - self._cost_limit)              # Delta = J_C - d
        # integral with gain folded in, anti-windup clamp at 0
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        # smoothed proportional input
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        # smoothed cost for a clean derivative
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        # rectified delayed-difference derivative (brakes cost increases only)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)                       # lambda >= 0
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)                         # advance the delay queue


def compute_adv_surrogate(
    adv_r: torch.Tensor, adv_c: torch.Tensor, lam: float
) -> torch.Tensor:
    """Step-size-consistent reward/cost advantage blend."""
    return (adv_r - lam * adv_c) / (1.0 + lam)


# Wiring into the fixed constrained-PPO outer step:
class ConstrainedPPO:
    def _init_controller(self) -> None:
        self._pid = PIDLagrangian(**self.cfg.lagrange_cfgs)

    def _update_multiplier(self, ep_cost: float) -> None:
        ep_cost = float(ep_cost)
        assert not np.isnan(ep_cost), 'cost is nan'
        self._pid.pid_update(ep_cost)
        self._lambda = self._pid.lagrangian_multiplier

    def _compute_adv_surrogate(
        self, adv_r: torch.Tensor, adv_c: torch.Tensor
    ) -> torch.Tensor:
        return compute_adv_surrogate(adv_r, adv_c, self._lambda)
```
