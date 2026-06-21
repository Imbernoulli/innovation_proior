PID-Lag finally crossed the line, and in doing so it told me precisely what is still wrong. It is the first rung to post nonzero $\text{budget\_success\_rate}$ — PointGoal $1.000$ across all three seeds, CarGoal and Button both $0.667$ — with cost sitting right where it should, $19.3$ on the goal tasks and $24.97$ on Button, a whisker under 25. As a *constraint solver* it works: the transient shaping got cost under budget in time. But the reward column is brutal — PointGoal collapsed to $0.20$, CarGoal to $1.996$, Button to $0.46$. The policies are feasible and essentially *do nothing*; PID-Lag bought the budget by making the agent so afraid of cost that it abandoned the task. The two missed seeds (CarGoal 42 at $25.22$, Button 456 at $28.51$) are the same story from the other side — barely over, reward near zero, the controller riding the limit so tightly that ordinary estimation noise tipped them past it. Both failures share one root: the continuous penalty *never switches off*. Even when safely under budget, the blend $A = (\text{adv}_r - \lambda\,\text{adv}_c)/(1+\lambda)$ keeps a positive $\lambda$ (the integral holds a standing value precisely so cost does not drift back up), so the policy is *always* paying a safety tax, always splitting its gradient between reward and cost, even deep in the feasible region where it has slack to spend on reward. That is structural to every Lagrangian method, integral or PID: the multiplier interpolates a trade-off it applies at *all* times.

So I question the premise all three previous rungs share. Ask what the agent should actually do at a given moment. If it is feasible — cost under 25 with slack — I want the whole update spent on reward; there is no reason to sacrifice return while I have budget to spare. If it is infeasible — cost over 25 — the only thing that matters is getting back under; reward is irrelevant until it is safe. So the "trade-off" the multiplier so carefully interpolates is, at the level of what-to-do-this-iteration, not a trade-off at all — it is a *switch*. Feasible $\to$ improve reward; infeasible $\to$ reduce cost. The Lagrange multiplier, whether driven by an integral or a full PID, is a smooth, lagged approximation to a decision that wants to be made sharply, and PID-Lag's numbers are exactly what the smooth approximation costs: a residual penalty left on in the feasible region (killing reward) and a tightly-ridden boundary (so noise tips seeds over). I propose *CRPO — constraint-rectified policy optimization*: drop the dual variable entirely and replace the dual-weighted blend with a direct rectification of the optimization target. At each update I read the mean episode cost $J_c = \text{self.\_logger.get\_stats('Metrics/EpCost')[0]}$ and use it not to drive a multiplier but to *choose which objective the policy step optimizes*:
$$\text{adv}_{\text{surrogate}} = \begin{cases} \text{adv}_r, & J_c \le \text{cost\_limit} + \eta \quad (\text{feasible: maximize reward}) \\ -\,\text{adv}_c, & J_c > \text{cost\_limit} + \eta \quad (\text{infeasible: minimize cost}) \end{cases}$$
There is no $\lambda$ anywhere — no dual learning rate, no integral, no PID gains, no $(1+\lambda)$ normalization (unneeded now, because the advantage is always a single unmodified stream of normal scale). The constraint enters only through the `if`. When feasible, the surrogate ascends $\text{adv}_r$ unmodified, exactly the naive update; when infeasible, it ascends $-\text{adv}_c$, which lowers the cost objective. The policy is pushed back toward feasibility exactly when, and only when, it strays out, and left entirely free otherwise.

I have to defend this against the two ways a naive switch breaks, because PID-Lag's near-miss seeds already show how fragile the boundary is. The first is *chattering*. If I switch the instant $J_c$ crosses 25, then near the boundary the *estimate* of $J_c$ — a noisy Monte-Carlo average, not the true value — fluctuates just above and below 25 from epoch to epoch, and the algorithm flips between "improve reward" (which pushes cost back up) and "reduce cost" (which pushes it down), oscillating at the edge without making reward progress. That is exactly the regime CarGoal 42 and Button 456 were stuck in. The cure is a *deadband*: switch to cost-reduction only when the estimate crosses $25 + \eta$ for a positive tolerance $\eta$. The tolerance absorbs the estimation noise so the switch stops chattering, and it gives the reward step a band of width $\eta$ above the limit in which to operate, so the policy can make real reward progress instead of being yanked back to the boundary every epoch. The price is that the converged policy can sit up to $\eta$ above the true limit, so $\eta$ must be small relative to the budget. I set $\eta = 2.0$ against a limit of 25 — an 8% slack, large enough to swallow the seed-to-seed cost noise I saw (the missed seeds were within $\sim 3.5$ of the line) and small enough that "feasible within tolerance" still means genuinely safe. The second worry is whether alternating between *different* objectives converges at all. Each epoch optimizes a different function, so the iterates descend no single fixed objective — but under accurate cost estimates the switch makes the *right* decision each epoch (improve reward only when genuinely feasible, reduce cost only when genuinely violated), and the per-epoch progress guarantees of the underlying PPO step then bound both the reward optimality gap and the constraint violation, giving an $O(1/\sqrt{T})$ rate to a globally optimal feasible policy with $O(1/\sqrt{T})$ violation. The tolerance $\eta$ is exactly the deadband that the analysis shrinks like $1/\sqrt{T}$; I hold it fixed at $2.0$ here because the 1M-step budget is finite and the noise floor, not the asymptotic rate, sets the right slack.

There is a third thing the switch fixes that the blend could never reach, and PID-Lag's reward column is the proof. Under any Lagrangian rule the feasible-region policy gradient is $(1-u)\,\text{adv}_r - u\,\text{adv}_c$ with $u = \lambda/(1+\lambda) > 0$, so even when safe the reward direction is *contaminated* by a cost-avoidance component pulling orthogonally to it — the policy is not climbing reward, it is climbing a compromise between reward and not-cost, and on these navigation tasks "not-cost" and "more reward" genuinely conflict because the rewarding path runs near hazards. That contamination is why PID-Lag's feasible policies froze at $0.20$–$2.0$ return: a permanently-on cost component, however small, biases every feasible-region step away from the goal. The switch removes the component entirely — in the feasible region the gradient is $\text{adv}_r$ and nothing else, the *same* update the naive rung used to earn the highest reward on the whole ladder — so the policy should reclaim a large fraction of naive's return while staying inside the band. I am not trading reward for safety in the feasible region at all; I spend the whole step on whichever objective is currently binding, and feasibility means reward is the only binding objective. The blend can never do this: a multiplier exactly zero in the feasible region would let cost drift back up (the reason the integral holds a standing value), so the Lagrangian family is structurally forced to keep paying the tax. Only the switch, which re-checks feasibility every epoch and re-engages cost-reduction the instant it is violated, can carry a true zero penalty while feasible without losing the budget — the feasibility check *is* the standing guarantee the integral was faking with a nonzero $\lambda$. In the edit surface this is small: $\text{\_update}$ reads $J_c$, sets $\text{\_in\_violation} = (J_c > \text{cost\_limit} + \text{distance})$, then calls $\text{super().\_update()}$ so the PPO epochs run with the flag fixed for this update ($J_c$ does not change inside the epochs); $\text{\_compute\_adv\_surrogate}$ is the switch, returning $-\text{adv}_c$ in violation else $\text{adv}_r$; I report the multiplier slot as $1.0$ in violation and $0.0$ otherwise, a faithful readout of the binary switch rather than a learned dual. This matches the canonical primal CRPO update line for line.

The bar this must clear is PID-Lag's real numbers, and the prediction is sharp. The whole point of the switch is to *not* penalize reward in the feasible region, so I expect a large reward recovery *at no safety cost*: $\text{budget\_success\_rate}$ at least as good as PID-Lag's (PointGoal $1.0$; CarGoal and Button at or above $0.667$) while reward climbs well above the near-zero values, because once feasible the policy runs the naive reward update at full strength and its return should approach what an unconstrained-but-feasible policy can earn, far above $0.20$ / $2.0$ / $0.46$. The two PID-Lag near-miss seeds are the direct test of the deadband: with an 8% tolerance band and no boundary-riding, they should land comfortably inside it, so the CarGoal and Button success rates should *rise* toward $1.0$, not just hold. The clean falsification: reward pinned near zero would mean the switch hypothesis is wrong and the collapse was not caused by the always-on penalty; $\text{budget\_success\_rate}$ *dropping* below PID-Lag's would mean the deadband is too small and the switch is chattering back into violation. What I am betting on is that PID-Lag solved the *timing* of safety but at the cost of *permanent* timidity, and that a method which spends nothing on safety while feasible and everything on it while violated recovers the reward PID-Lag threw away without giving back the budget it won.

```python
# EDITABLE region of custom_lag.py — finale: CRPO (constraint-rectified switch)
import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class CustomLag(PPO):

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._distance: float = 2.0
        self._in_violation: bool = False
        self._rew_update: int = 0
        self._cost_update: int = 0
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        self._in_violation = bool(Jc > self._cost_limit + self._distance)
        if self._in_violation:
            self._cost_update += 1
            self._lagrangian_multiplier = 1.0
        else:
            self._rew_update += 1
            self._lagrangian_multiplier = 0.0
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'CRPO: within budget+tolerance, maximize reward; otherwise descend cost only.'
        if self._in_violation:
            return -adv_c
        return adv_r
```
