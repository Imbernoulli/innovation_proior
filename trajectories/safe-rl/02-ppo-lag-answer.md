**Problem.** The naive agent posts strong return but a cost far over budget on every environment and
seed (PointGoal 51, CarGoal 61, PointButton 152, against `d = 25`), because dropping `adv_c` leaves the
reward gradient free to plow through hazards. A fixed reward-minus-penalty scalarization cannot fix it:
the penalty weight that lands cost at 25 cannot be set from the budget, cannot be the same across
environments that violate by different amounts, and cannot be right both early (policy still learning)
and late (policy exploiting risky shortcuts).

**Key idea.** Keep safety as a *constraint*: maximize `J_r` subject to `J_c <= d`. Dualize with a
nonnegative multiplier `lambda`, form `L = J_r - lambda*(J_c - d)`, and solve the saddle point by
primal-dual alternation. The inner min over `lambda >= 0` reconstructs "reward if feasible, minus
infinity if not"; the occupancy-measure LP view gives zero duality gap, so dual descent aims at the
constrained optimum even though the neural step only approximates it.

**Why it works.** The dual update `lambda <- [lambda + eta*(J_c - d)]_+` is an integral controller on
the violation: lambda rises exactly when the agent is over budget and relaxes when there is slack, so it
performs the trade-off tuning the fixed penalty could not. The policy step is ordinary PPO on the
combined advantage `adv_r - lambda*adv_c` (the per-state form of `grad(J_r - lambda*J_c)`). Shipping
that blend raw lets a large lambda inflate PPO's effective step size, so it is divided by `(1 + lambda)`,
turning it into the convex blend `(1-u)*adv_r - u*adv_c` with `u = lambda/(1+lambda)`: unit-scale for any
lambda, pure reward at `lambda = 0`, pure cost-avoidance as `lambda -> inf`, with the `1:lambda`
reward-to-cost ratio (hence the gradient direction and dual stationary point) preserved.

**Hyperparameters.** Dual step size `lambda_lr = 0.035` (substrate default), `cost_limit = 25.0`,
`lagrangian_multiplier_init` floored at 0; lambda is a `torch.nn.Parameter` optimized by Adam via the
loss `-lambda*(Jc - cost_limit)`, then clamped to `[0, inf)`; lambda updated before the PPO step.

**What to watch.** Cost should fall from 51/61/152 toward 25 on all three environments, with return
dropping below the naive 25.5/32.8/19.7 as the price. But because lambda only reacts to *accumulated*
violation it lags, so cost should stay *over* budget (especially on PointButton) and oscillate rather
than settle — the integral-only weakness that motivates a richer controller next.

```python
# EDITABLE region of custom_lag.py -- step 2: PPOLag (Adam dual multiplier)

# (no extra imports)

@registry.register
class CustomLag(PPO):
    """PPOLag: Lagrangian PPO with an Adam-updated dual multiplier."""

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        init_value = max(self._cfgs.lagrange_cfgs.lagrangian_multiplier_init, 0.0)
        self._lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value), requires_grad=True,
        )
        self._lambda_optimizer = torch.optim.Adam(
            [self._lagrangian_multiplier],
            lr=self._cfgs.lagrange_cfgs.lambda_lr,
        )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Lagrange multiplier update via Adam: loss = -lambda * (Jc - d)
        self._lambda_optimizer.zero_grad()
        lambda_loss = -self._lagrangian_multiplier * (Jc - self._cost_limit)
        lambda_loss.backward()
        self._lambda_optimizer.step()
        self._lagrangian_multiplier.data.clamp_(0.0)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier.item()})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """PPOLag: penalize cost advantage using the Lagrange multiplier."""
        penalty = self._lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
