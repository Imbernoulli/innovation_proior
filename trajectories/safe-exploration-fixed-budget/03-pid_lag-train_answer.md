PPO-Lag did exactly what the slow-integrator hypothesis predicted, and the prediction being right *is* the diagnosis. Cost came down everywhere — PointGoal from naive's $51$ to $46$, CarGoal from $61$ to $47$, Button from a catastrophic $152$ to $56$ — and the reward collapse that came with it (PointGoal $25.5 \to 15$, Button $19.7 \to 4$) confirms the trade-off is real. But the metric the task counts, $\text{budget\_success\_rate}$, is still zero on every seed of every environment: the mechanism lowers cost without ever crossing under 25. The exact update is the whole problem in one line. With $\lambda_{k+1} = (\lambda_k + K_I(J_C - d))_+$, the multiplier is *literally the time-integral* of the constraint error — it reacts only to *accumulated* error, so $\lambda$ rises only after violation has been piling up for a while, and the response is intrinsically late. Button is the loudest evidence: $\lambda$ had to climb enormously to drag 152 down to 56, and even that enormous $\lambda$ left it $2\times$ over budget, because by the time the integral had accumulated enough to push hard, the policy had already baked the unsafe path in. Turning $K_I$ up does not fix this — a stiffer pure integrator rings faster and longer, trading a late response for an oscillating one and degrading reward further. One knob, bad trade.

The fix is to stop thinking of this as "tune the Lagrangian update" and see what the loop actually is: a feedback control system. The cost limit $d=25$ is a *setpoint*, the measured episodic cost $J_C$ is the *output* I want held there, the multiplier $\lambda$ is the *control input* I set, and the PPO step grinding $\theta$ forward to chase reward is the *plant* — a nonlinear unknown map from $\lambda$ to next epoch's cost. The traditional Lagrangian method is one impoverished choice of controller: integral-only. So I propose *PID-Lag*, driving $\lambda$ with the full three-term controller. Each epoch I read $J_C$, form the error $\Delta = J_C - d$, and combine a proportional, an integral, and a derivative term:
$$\lambda = \text{clip}\big(K_P\,\delta_p + I + K_D\,\text{pid}_d,\ 0,\ 100\big).$$
Each term earns its place against a piece of the failure. *Proportional* adds $K_P \cdot \Delta$, an instantaneous reaction to the *current* error with no waiting for accumulation. In the continuous-time picture the integral term was $\dot\lambda = \alpha g$; the proportional contribution enters $\dot\lambda$ as a term tracking $\dot g$, and redoing the second-order collapse of the coupled primal-dual dynamics with this piece, the restoring force $\alpha g \nabla g$ is untouched — so proportional does not move the equilibrium, the feasible-and-stationary solution set is unchanged — but the damping matrix gains $\beta\,\nabla g\,\nabla g^\top$, an outer product that is positive semidefinite by construction ($v^\top(\nabla g\,\nabla g^\top)v = (\nabla g \cdot v)^2 \ge 0$). More damping is exactly what kills an oscillation, and it touches only the $\lambda$ update, not the policy objective — the quadratic-penalty method injects the *same* damping but bundled with an indefinite constraint-Hessian term and a modified primal step, so proportional gives the penalty method's benefit with none of the baggage. For my failure it is the missing reflex: when cost is over the line *right now*, it reacts this epoch instead of waiting for the integral to build.

*Derivative* adds anticipation. Proportional reacts to the present; I want to react to the *trend* — if cost is rising fast toward 25, raise $\lambda$ now, before it crosses. A term on the rate of cost change enters $\lambda$'s dynamics one derivative higher; grinding through the collapse it scales the restoring direction by $B^{-1} = (I + \gamma\,\nabla g\,\nabla g^\top)^{-1}$ (eigenvalues $\le 1$) and adds a force $\gamma(\dot x^\top \nabla^2 g\,\dot x)\,B^{-1}\nabla g$ quadratic in velocity and modulated by the constraint curvature along the motion. When I am moving where cost curves upward and am already in violation, this anticipatory force adds to the restoring force — I brake harder before overshooting, precisely the overshoot Button suffered. One asymmetry I bake in for the cost case: derivative control should *brake increases* in cost (the dangerous direction) but not fight *decreases* — a falling cost is good — so I rectify it to the positive part, a one-sided derivative for a one-sided constraint. *Integral* I keep because at convergence I need *zero* steady-state violation, and only the integral can supply the standing $\lambda$ that holds cost exactly at 25: when the system settles the error $\approx 0$, so proportional vanishes and derivative (cost not changing) vanishes too, and if those were my only terms $\lambda$ would collapse and cost drift back up. The integral has *remembered* the accumulated history and holds a nonzero $\lambda$ even at zero instantaneous error. The division of labor is clean — proportional and derivative shape the *transient* (the very thing PPO-Lag failed at, getting under the line in time) while integral guarantees the *asymptote* (staying under it). And $K_P = K_D = 0$ recovers PPO-Lag's integral-only rule exactly, so this is a strict generalization: a one-parameter family widened to three with the old method at the origin of the two new axes.

The discrete rule matches what the harness exposes. To tame the noise the derivative would amplify, I smooth two signals with an exponential moving average at $0.95$: a smoothed proportional error $\delta_p \leftarrow 0.95\,\delta_p + 0.05\,\Delta$ and a smoothed cost level $c_d \leftarrow 0.95\,c_d + 0.05\,J_C$. The integral accumulates with anti-windup, $I \leftarrow (I + K_I \Delta)_+$ — the inner positive part stops the integrator from banking a *negative* reservoir during a long feasible stretch, which would otherwise force $\lambda$ to climb out of a hole before its next response, the exact lateness disease I am curing, so I will not reintroduce it. The derivative is the rectified change of the smoothed cost over a short delay window, $\text{pid}_d = (c_d - c_d[t-w])_+$, with $w = 10$ epochs of delay held in a deque; delay-and-rectify gives a clean one-sided trend estimate instead of a noisy single-step difference. Gains are $K_P = 0.1$, $K_I = 0.01$, $K_D = 0.01$ — integral small because it accumulates, proportional an order of magnitude larger to supply the fast reflex, derivative matched to the integral — and the upper clamp at $100$ guards against a runaway controller in the densest-hazard arena. Crucially the policy hook is *unchanged*: I reuse the same scale-normalized blend $A = (\text{adv}_r - \lambda\,\text{adv}_c)/(1+\lambda)$, because the $(1+\lambda)$ normalization that kept PPO's step size sane under a large integral $\lambda$ is needed even more here, where an aggressive PID controller can drive $\lambda$ large fast. The only change from PPO-Lag is the rule that *produces* $\lambda$.

Against the PPO-Lag numbers the claim is that proportional plus derivative get cost under the line *in time* where the pure integrator could not, so I expect this to be the first rung to post nonzero $\text{budget\_success\_rate}$ — cost falling from the 40s–50s to at or below 25 on most runs, with PointGoal (gentlest geometry) crossing under most reliably and Button and CarGoal closer to the line and seed-dependent. The cost of buying that safety is reward: reacting hard and early suppresses reward-seeking much more than the sluggish integral did, so I expect reward to fall further still toward zero on the goal tasks — a feasible-but-timid policy. That is the explicit trade. If it crosses under 25 on most runs it is the strongest baseline on this task even with the lowest reward, because a policy that violates the budget on 100% of runs cannot be called safer than one that satisfies it, whatever its return. The clean test: still-zero $\text{budget\_success\_rate}$ everywhere would falsify the PID hypothesis (lateness was not the cause); one on the gentle geometry and fractional on the dense ones would confirm the transient-shaping story, with the residual misses then an aggressiveness-versus-reward problem rather than a timing one.

```python
# EDITABLE region of custom_lag.py — step 3: PID-Lag (PID-controlled multiplier)
import numpy as np
import torch

from collections import deque

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class CustomLag(PPO):

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._pid_kp: float = 0.1
        self._pid_ki: float = 0.01
        self._pid_kd: float = 0.01
        self._pid_i: float = 0.0
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_ds: deque = deque(maxlen=10)
        self._cost_ds.append(0.0)
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        delta = float(Jc - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        self._delta_p = 0.95 * self._delta_p + 0.05 * delta
        self._cost_d = 0.95 * self._cost_d + 0.05 * float(Jc)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._lagrangian_multiplier = max(0.0, min(pid_o, 100.0))
        self._cost_ds.append(self._cost_d)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'PID-Lag: combine advantages using PID-controlled multiplier.'
        penalty = self._lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
