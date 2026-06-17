# PPO-Lagrangian

For a cost upper bound `J_c(pi) <= d`, use the Lagrangian

```text
L(theta, lambda) = J_r(theta) - lambda (J_c(theta) - d),   lambda >= 0.
```

The multiplier is nonnegative by the KKT conditions for the inequality `J_c - d <= 0`; at the
saddle it also satisfies complementary slackness, `lambda (J_c - d) = 0`. In occupancy-measure
variables, CMDP reward and cost returns are linear and the Bellman-flow constraints define a convex
feasible set, so the underlying constrained RL problem has zero duality gap under the usual
feasibility assumptions. The dual update is therefore aimed at the constrained optimum, even though
the neural policy update is only an approximate primal step.

Update the multiplier from the measured mean episodic cost:

```text
lambda <- [lambda + eta (J_c - d)]_+.
```

Update the policy with PPO on the combined advantage

```text
A_lag = (A_r - lambda A_c) / (1 + lambda).
```

The numerator is the policy-gradient direction for `J_r - lambda J_c`. The denominator preserves the reward-to-cost ratio `1:lambda` while keeping the PPO advantage scale bounded as the multiplier grows.

Equivalently, with `u = lambda / (1 + lambda)`,

```text
A_lag = (1 - u) A_r - u A_c,
```

a convex blend of reward advantage and negated cost advantage.

Faithful OmniSafe-style implementation:

```python
import numpy as np
import torch


class Lagrange:
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        self.cost_limit = cost_limit
        self.lagrangian_upper_bound = lagrangian_upper_bound
        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt([self.lagrangian_multiplier], lr=lambda_lr)

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0.0, self.lagrangian_upper_bound)


class PPOLag(PPO):
    def _init(self) -> None:
        super()._init()
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key("Metrics/LagrangeMultiplier", min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats("Metrics/EpCost")[0]
        assert not np.isnan(Jc), "cost for updating lagrange multiplier is nan"
        self._lagrange.update_lagrange_multiplier(Jc)
        super()._update()
        self._logger.store(
            {"Metrics/LagrangeMultiplier": self._lagrange.lagrangian_multiplier},
        )

    def _compute_adv_surrogate(
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
```

This is PPO with one additional dual variable and one advantage-combination hook. It adapts the penalty from observed constraint violation, but it is not a CPO-style per-update feasibility method; lag and oscillation around the budget are expected practical limitations.
