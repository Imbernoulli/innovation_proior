Naive PPO did exactly what the floor should: it learned the task and posted the highest reward I will see on this ladder — mean $25.5$ on PointGoal, $32.8$ on CarGoal, $19.7$ on Button — while $\text{budget\_success\_rate}$ came back zero on every seed of every environment. The shape of the violation is the tell. PointGoal cost landed around $51$ against a budget of $25$, a clean $2\times$ overshoot; CarGoal $\sim 61$; Button a catastrophic $140$–$164$, more than six times the limit, the densest-hazard task overshooting worst exactly as its geometry predicts. So the failure is a *specification* failure: the policy felt no cost pressure, took the shortest reward-seeking path through whatever hazards lay on it, and the cost is simply whatever that path collects. To get any of these under 25 I have to put the constraint into the thing the policy optimizes — and crucially, with a weight that adapts to the measured violation rather than a constant I would have to guess. A single fixed penalty $\beta$ cannot work: there is no formula mapping my budget $d=25$ to the $\beta$ that achieves it, the naive numbers show the trade-off *moves across environments* (PointGoal needs cost halved, Button cut sixfold), and even within one run $\beta$ would have to stay small early so the incompetent policy can learn to navigate, then push hard once it finds the rewarding-but-risky shortcuts. No single number does both jobs.

So I keep safety as what it is — a constraint — and propose *PPO-Lag*, primal-dual ascent on the Lagrangian of the constrained MDP $\max_\theta J_r(\theta)$ subject to $J_c(\theta) \le d$. Writing the constraint in standard form $g(\theta) = J_c(\theta) - d \le 0$, introducing a multiplier $\lambda \ge 0$, and forming
$$L(\theta, \lambda) = J_r(\theta) - \lambda\,(J_c(\theta) - d),$$
the constrained problem equals $\max_\theta \min_{\lambda \ge 0} L$. Checking the inner min by behavior, with $\theta$ fixed: if the constraint is violated ($g > 0$), $\min_{\lambda \ge 0} -\lambda \cdot(\text{positive})$ drives $\lambda \to \infty$ and $L \to -\infty$, so the outer max never picks such a $\theta$; if it is satisfied with slack ($g < 0$), the min takes $\lambda = 0$ and the penalty switches off, giving pure reward; on the boundary the $\lambda$ term vanishes. The max-min reconstructs exactly "reward if feasible, $-\infty$ if not," and at the saddle complementary slackness $\lambda \cdot g = 0$ holds — $\lambda$ is positive only when the budget binds. This is the convex textbook story and $J_r, J_c$ are nonconvex in $\theta$, but the nonconvexity is in the parameterization $\theta \mapsto \pi_\theta$, not in the problem: in occupancy-measure space both returns are *linear* over the convex set cut out by the Bellman flow equations, a linear program with zero duality gap (Altman's LP view; Paternain et al. 2019). The honest reading is that dual descent aims at the correct saddle of the true problem and my neural updates approximate it — license enough to build the algorithm. I prefer the constraint framing over a penalty not for aesthetics but because a threshold *saturates*: "get cost under $d$, then stop caring," which is the exact shape of the binary $\text{budget\_success\_rate}$, whereas a penalty keeps trading reward against cost forever even deep inside the safe region.

The solver alternates: ascend $\theta$ to push $L$ up, descend $\lambda$ to push it down, chasing the saddle. The $\lambda$ update is where the adaptivity the naive run lacked lives. Since $\partial L/\partial \lambda = -g(\theta)$, descent gives the projected ascent
$$\lambda \leftarrow \big[\lambda + \eta\,(J_c - d)\big]_+ .$$
Checked by behavior: over budget the bracket increases, $\lambda$ climbs, the next policy update weighs cost more heavily; under budget the bracket decreases, $\lambda$ falls toward zero, the policy is freed to chase reward. Structurally $\lambda$ is the running *integral* of the violation signal, so it settles to whatever weight drives the average violation to zero. The constraint is on the episodic cost return, so I update $\lambda$ once per epoch from the logged mean episode cost $J_c = \text{self.\_logger.get\_stats('Metrics/EpCost')[0]}$ — not from any per-step cost — and assert it is not NaN first, because a bad cost statistic would corrupt $\lambda$ silently and kill the safety mechanism quietly. For the $\theta$ side I ascend $L = J_r - \lambda J_c$ (the $\lambda d$ term is constant in $\theta$), whose gradient $\nabla_\theta J_r - \lambda \nabla_\theta J_c$ corresponds to a single combined advantage $\text{adv}_r - \lambda \cdot \text{adv}_c$ fed straight into the PPO clip I already trust. This is the whole appeal: a safe-RL rule that is PPO plus one scalar and one line mixing two advantages, landing precisely in the two editable hooks — $\lambda$ in $\text{\_update}$, the mix in $\text{\_compute\_adv\_surrogate}$.

But shipping $A = \text{adv}_r - \lambda \cdot \text{adv}_c$ raw breaks, and the Button numbers say why it breaks *here specifically*. A $6\times$ violation forces dual ascent to drive $\lambda$ to $5$, $10$, $50$; at $\lambda = 50$ the *scale* of the combined advantage is $\sim 50\times$ a normal advantage. PPO's clip range and learning rate were calibrated assuming the advantage has roughly its usual magnitude, and an advantage whose scale grows linearly in $\lambda$ silently inflates the effective step size exactly when $\lambda$ is large — exactly when I most need the update careful. The clip will not save me, because it clips the probability *ratio*, not the advantage. So $\lambda$ has leaked into two places: the relative weight of reward versus cost, which I want, and the numerical scale of the gradient, which I do not. I peel them apart by normalizing the scale while keeping the direction: divide by $(1+\lambda)$,
$$A = \frac{\text{adv}_r - \lambda \cdot \text{adv}_c}{1 + \lambda}.$$
Setting $u = \lambda/(1+\lambda) \in [0,1)$ gives $A = (1-u)\,\text{adv}_r - u\,\text{adv}_c$, a convex blend. At $\lambda = 0$, $u=0$, pure reward — the unconstrained update, correct because slack means no safety tax; as $\lambda \to \infty$, $u \to 1$, pure cost-avoidance, correct because a huge violation should sacrifice reward. In between the weights sum to 1, so $|A|$ stays on the order of an ordinary advantage no matter how large $\lambda$ grows and PPO's tuned step size behaves identically throughout. The *ratio* of the weights is still $(1-u):u = 1:\lambda$, so the gradient direction and the saddle point are unchanged — the $(1+\lambda)$ factor is a step-size normalization, not a new optimum. For the $\lambda$ step itself I use OmniSafe's $\text{Lagrange}$ helper, which stores $\lambda$ as a learnable scalar and realizes the projected ascent through an Adam step on the loss $-\lambda(J_c - d)$, giving $\lambda \leftarrow \lambda + \eta(J_c - d)$ followed by a nonnegative clamp; wrapping the dual scalar in Adam smooths it against the epoch-to-epoch Monte-Carlo noise in the estimate of $J_c$, with $\text{lambda\_lr} = 0.035$ the dual learning rate.

Against the naive numbers I expect cost to come *down* everywhere — PointGoal from $\sim 51$ toward the 40s, CarGoal from $\sim 61$ toward the 40s, Button most dramatically from $\sim 152$ toward the 50s, since the bigger the violation the harder $\lambda$ pushes — with reward falling in lockstep because every unit of cost pressure is reward traded away. But the mechanism is dual ascent on a single scalar driven by the running-average violation, so it is *slow*: $\lambda$ integrates the error and only settles after many epochs, and on a 1M-step budget it may simply not have climbed high enough by the end. The clean test of that slow-integrator hypothesis is sharp — if $\text{budget\_success\_rate}$ is still zero everywhere while cost has roughly halved, the direction is right but the controller is too sluggish, lowering cost without ever crossing the budget, which is the only thing the metric counts. That failure is exactly what would force a controller reacting to the *rate* of violation, not just its accumulated integral.

```python
# EDITABLE region of custom_lag.py — step 2: PPO-Lag (adaptive Lagrangian multiplier)
import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange


@registry.register
class CustomLag(PPO):

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        self._lagrange.update_lagrange_multiplier(Jc)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'PPO-Lag: penalize cost advantage using the learned multiplier.'
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
