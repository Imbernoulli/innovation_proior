The task is a constrained MDP on three Safety-Gymnasium navigation problems: maximize the discounted reward return $J_r$ while holding the discounted cost return $J_c$ at or below a hard budget $d = 25$, where cost is a separate hazard signal the substrate tracks with its own value head. The only transferable piece I get to design is the *constraint-handling update rule* — how the cost-penalty controller updates and how the reward advantage $\text{adv}_r$ and cost advantage $\text{adv}_c$ get combined into the single advantage the PPO surrogate ascends. Before I can read any controller's contribution honestly, I need the floor: a run with no controller at all, so I can see how far cost drifts when nothing pulls it back and how much reward the backbone extracts when it is free to ignore safety. That is the deliberate purpose of this first rung.

I propose the *naive* baseline: run the PPO backbone unchanged and let the cost stream sit completely unused. PPO ascends the clipped surrogate $L^{\text{CLIP}} = \hat{\mathbb{E}}_t[\min(r_t \hat A_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat A_t)]$, where $r_t = \pi_\theta/\pi_{\text{old}}$ is the importance ratio and $\hat A_t$ the advantage; the clip is a first-order trust region, a flat spot in the loss once the ratio leaves the band, so one batch of on-policy data can be reused for several gradient epochs without the policy walking out of the region where the surrogate is honest. None of that machinery is mine to touch — the rollout, the two critics (one fit to reward, one to cost), the GAE that produces $\text{adv}_r$ and $\text{adv}_c$, the optimizers, the training loop are all fixed. What is mine is the single decision of *which* advantage the surrogate ascends, and here I make the null decision: $\text{\_compute\_adv\_surrogate}$ returns $\text{adv}_r$ and nothing else. The cost advantage is computed by the substrate and then discarded; the Lagrange multiplier is pinned at $0.0$ and never updated. The $\text{\_update}$ method reads the mean episode cost only to assert it is not NaN — a substrate sanity check, not a controller signal — then runs the ordinary PPO epochs. There is no $\text{Lagrange}$ object, no PID memory, no moving window. The edit is the default fill verbatim, which is exactly why it is the floor by construction.

What this configuration *does* is the whole reason to run it. With the cost advantage thrown away, the policy is a pure unconstrained reward maximizer in a hazard field. Whatever earns reward in these navigation tasks — driving straight to the goal, pressing buttons, taking the shortest path through the arena — it will learn, and it will learn it with no pressure whatsoever to route around hazards, because entering a hazard costs nothing in the objective it sees. So two things should hold at once. Reward should be as high as the backbone can make it, plausibly the highest of any rung on this ladder, because the policy spends its entire capacity on return with no safety tax. And cost should run far over budget — not marginally, but multiples of 25, because the shortest reward-seeking path through a hazard-dense arena collects contacts at whatever rate the geometry imposes and nothing in the update opposes that. The binary $\text{budget\_success\_rate}$ should therefore be zero across the board: a reward-greedy policy in a hazard field is the least likely configuration to land under 25 by accident. The three environments should separate by how hazard-dense the reward-seeking path is — gentlest overshoot on $\text{SafetyPointGoal1}$, more on $\text{SafetyCarGoal1}$ with its harder car dynamics, and the largest by a wide margin on $\text{SafetyPointButton1}$, the densest-cost geometry.

The diagnosis this rung is built to produce already points at the next one. If naive PPO posts high reward and a $\text{budget\_success\_rate}$ of zero with cost at multiples of the limit, the problem it exposes is not a learning problem — the policy learns the task fine — it is a *specification* problem: the objective the policy ascends contains no term for the constraint, so of course the constraint is violated. The fix is to put the constraint into the objective, turning the inert multiplier into a learned one and combining the two advantage streams. The substrate exposes $\text{adv}_c$ and a multiplier slot precisely so a future rung can do that. This rung does not.

```python
# EDITABLE region of custom_lag.py — step 1: naive (pure PPO, ignores cost)
import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class CustomLag(PPO):

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Naive: keep multiplier at zero and ignore budget pressure.
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        'Naive: ignore cost advantage entirely, optimize reward only.'
        return adv_r
```
