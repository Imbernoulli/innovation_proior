The naive run told me what is broken, and it told me in numbers. Return is strong everywhere — $25.5$ on SafetyPointGoal1, $32.8$ on SafetyCarGoal1, $19.7$ on SafetyPointButton1 — so the base learner optimizes reward just fine; that was never the problem. The problem is the cost column. Against the budget $d = 25$ the naive agent sits at $51$ on PointGoal, $61$ on CarGoal, and a staggering $152$ on PointButton — six times over budget on the hazard-dense one, two-to-three times over on the others, on *every single seed*. There is no lucky safe seed; the violation is systematic, the signature of dropping $\text{adv}_c$. So the diagnosis is sharp: this is not a learning problem, it is a *specification* problem. The cost is a real, separate signal the loop is already measuring, and I am simply not using it.

The reflex fix is to mash the two signals into one scalar and hand it to the optimizer I already trust — ascend $\text{adv}_r - \beta\,\text{adv}_c$ for some penalty $\beta$ — and every time I reach for it I hit the same wall. There is no formula that takes my budget $d = 25$ and returns the $\beta$ that lands cost at $25$; the map from penalty weight to resulting cost return depends on the whole environment and policy class in a way I cannot invert ahead of time. The naive numbers prove it, because the *same* "penalty" (zero) produced wildly different over-budget amounts across the three environments — $51$ versus $61$ versus $152$ — so one constant cannot be the right tax for all three. Worse, the one $\beta$ would have to do two incompatible jobs across a single run: stay small early so the still-incompetent policy can learn to reach goals at all, then push hard once the policy has found rewarding-but-risky shortcuts. A fixed penalty bakes a trade-off I do not know how to set into a number I must commit to before seeing any data.

So I stop folding safety into the objective and keep it as what it is — a constraint. I propose **PPOLag**: the Lagrangian PPO, where I solve

$$\max_\theta J_r(\theta) \quad \text{subject to} \quad J_c(\theta) \le d,$$

with $J_r$, $J_c$ the expected episodic returns of reward and of cost, defined identically except for which signal they sum. I prefer the constraint framing over the penalty framing not for aesthetics but because a threshold has a *saturation* the penalty lacks: it encodes "get cost under $25$, then stop caring; spend everything else on reward," which is the actual shape of the requirement. A penalty keeps trading reward against cost forever, even far inside the safe region.

To optimize this when $J_r$ and $J_c$ are nonconvex in a few hundred thousand neural weights, I dualize. Writing the constraint as $g(\theta) = J_c(\theta) - d \le 0$, introducing a multiplier $\lambda \ge 0$, and forming

$$L(\theta, \lambda) = J_r(\theta) - \lambda\big(J_c(\theta) - d\big),$$

the constrained problem equals $\max_\theta \min_{\lambda \ge 0} L$. The inner min is worth verifying, because it is what makes the Lagrangian the real problem and not a heuristic stand-in. Fix $\theta$. If the constraint is *violated*, $g > 0$, then minimizing $-\lambda \cdot (\text{positive})$ over $\lambda \ge 0$ drives $\lambda \to \infty$ and $L \to -\infty$, so the outer max never picks such a $\theta$: infeasible policies (exactly the naive agent's situation) are punished to minus infinity. If the constraint is slack, $g < 0$, the min picks $\lambda = 0$ and $L = J_r$ — the penalty switches off and I recover pure reward, which is the *correct* naive behavior once safely under budget. At the saddle, complementary slackness $\lambda\, g = 0$ holds: $\lambda$ is positive only when the constraint binds. One worry is that the nonconvexity could open a duality gap and send dual descent chasing the wrong target. But the nonconvexity lives in the parameterization $\theta \mapsto \pi_\theta$, not in the problem: in occupancy-measure space both returns are *linear* and the feasible set is convex, so the underlying problem is a linear program with zero duality gap. Dual descent on the neural parameters is then an approximate primal step aimed at the correct saddle structure — enough license to build the algorithm.

The algorithm is primal-dual alternation. For the dual variable I move $\lambda$ to *decrease* $L$, i.e. gradient descent; since $\partial L / \partial \lambda = -(J_c - d)$, the projected step is

$$\lambda \leftarrow \big[\,\lambda + \eta\,(J_c - d)\,\big]_+.$$

Checking by behavior rather than algebra, because a flipped sign would silently invert the whole mechanism: when the policy is over budget, $J_c > d$, the bracket grows, $\lambda$ climbs, and the next policy update weighs cost more heavily and is pushed toward safety; when under budget, $\lambda$ falls toward zero and the policy is freed to chase reward with its slack. So $\lambda$ does the trade-off tuning *for* me, reading the actual measured violation each epoch — the adaptive coefficient the fixed penalty could never produce. Structurally $\lambda$ is the *integral* of the violation: it accumulates $\sum (J_c - d)$ and settles to whatever penalty weight drives the average violation to zero. The projection $[\cdot]_+$ is not cosmetic — an inequality multiplier must be nonnegative (the KKT condition; $\lambda$ measures the reward gained by loosening a binding budget, which cannot be negative), and a negative $\lambda$ would mean *paying* the agent to incur cost. The constraint is on the *expected episodic* cost, so I do not update from a per-timestep cost; once per epoch I read the mean episodic cost across workers, $J_c = $ `self._logger.get_stats('Metrics/EpCost')[0]` — the same $51/61/152$ the naive feedback reported — assert it is not NaN (a bad statistic would corrupt $\lambda$ silently), and take one dual step.

For the primal variable I ascend $\theta$ on $L = J_r - \lambda J_c$, whose gradient is $\nabla J_r - \lambda \nabla J_c$. PPO already ascends an expected return given an advantage, with gradient $\mathbb{E}[\nabla \log \pi \cdot A]$, so $\nabla J_r$ corresponds to $\text{adv}_r$ and $\nabla J_c$ to $\text{adv}_c$ — which is exactly why the loop keeps two value functions — and the combined gradient corresponds to a single combined advantage $A = \text{adv}_r - \lambda\,\text{adv}_c$. I feed that into the PPO step through `_compute_adv_surrogate`. But shipping it raw breaks. On PointButton, where the naive cost was $152$, the dual update keeps incrementing $\lambda$ epoch after epoch; it does not stay small, it can climb to $50$. Then the *scale* of $A = \text{adv}_r - \lambda\,\text{adv}_c$ is roughly $50\times$ a normal advantage. PPO's clip range and learning rate were calibrated assuming the advantage has its usual magnitude, and the clip will not save me — it clips the probability ratio $r_t$, not the advantage — so a giant $A$ silently inflates the effective step size exactly when $\lambda$ is large and I most need the update careful. $\lambda$ has leaked into two places: the relative weight of reward versus cost, which I want, and the numerical scale of the gradient, which I do not.

I do not want to change $\lambda$'s meaning to fix this — the ratio $1:\lambda$ is what the dual is carefully tuning. So I peel the two effects apart: keep the *direction* of the combined advantage but normalize its *scale* by dividing by $1 + \lambda$, the cleanest normalization that is monotone in $\lambda$ and equals $1$ at $\lambda = 0$:

$$A = \frac{\text{adv}_r - \lambda\,\text{adv}_c}{1 + \lambda}.$$

Setting $u = \lambda/(1+\lambda) \in [0,1)$, this reads $A = (1-u)\,\text{adv}_r - u\,\text{adv}_c$ — a convex blend of the reward advantage and the negated cost advantage. At $\lambda = 0$, $u = 0$ and $A = \text{adv}_r$, the naive update, right because a slack constraint should pay no tax. As $\lambda \to \infty$, $u \to 1$ and $A \to -\text{adv}_c$, pure cost-avoidance, right because a huge $\lambda$ means the budget is badly violated and reward should be sacrificed. In between, the two weights sum to $1$, so $|A|$ stays on the order of the individual advantages no matter how large $\lambda$ grows and PPO's tuned step size behaves throughout. Critically the ratio of reward weight to cost weight is $(1-u):u = 1:\lambda$, identical to the un-normalized version, so the gradient *direction* and the dual stationary point are unchanged — I rescaled the step, not the optimum.

For the dual step I keep $\lambda$ as a learnable `torch.nn.Parameter` and define the loss $\text{loss}_\lambda = -\lambda\,(J_c - d)$, whose gradient with respect to $\lambda$ is $-(J_c - d)$, so an Adam step gives exactly the dual ascent $\lambda \leftarrow \lambda + \eta\,(J_c - d)$. Wrapping the bare step in Adam smooths the single dual scalar against the epoch-to-epoch noise in the Monte-Carlo estimate of $J_c$ — and there is noise, the naive PointButton cost ranged $140$–$164$ across seeds. After the optimizer step I clamp `data` to the nonnegative range, the projection. The order within an epoch matters: read $J_c$, update $\lambda$ from it *first*, then run `super()._update()` with the freshly updated $\lambda$ inside the combined advantage, then log $\lambda$. The dual step size $\eta$ is the substrate's `lambda_lr`, default $0.035$.

I name the one weakness up front because it is intrinsic to $\lambda$'s integral character and predicts where the next step must go. As a pure integrator $\lambda$ responds only to *accumulated* over-budget, so it lags: deep over budget (the $152$ on PointButton) it has to integrate up over several epochs and by the time it is large enough it has overshot; once cost drops below budget it stays large and over-corrects toward timidity, cost undershoots, $\lambda$ falls, cost rises again. The result is oscillation of cost around the budget rather than a clean settle — the generic behavior of a pure-integral controller on a delayed system. The integral term is exactly what makes $\lambda$ converge to the right steady-state penalty with zero steady-state violation, which is why I keep it, but the oscillation is the obvious thing the next mechanism will attack with faster-reacting terms. Concretely I expect cost to come down from $51/61/152$ toward $25$ on all three environments, return to drop below the naive $25.5/32.8/19.7$ as the price, but cost to remain *over* budget — reduced, not controlled, especially on PointButton — which is exactly the integral-only lag showing.

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
