**Problem.** iCEM bought short-horizon reach (horizon 30: 0.70 → 0.80) but *regressed* at the long
horizon (horizon 90: 0.95 → 0.85): aggressive elite reuse plus colored over-smoothing over-commits to
one route, and the hard top-k elite cut throws away how much better the best rollout was and is brittle
near the two-route decision boundary. The fix is a *soft*, information-preserving update.

**Key idea (MPPI — exponential cost-weighted update).** Keep the CEM skeleton — sample white
perturbations around the current mean, roll through the model, take the top-k elites — but replace the
hard elite mean with a Boltzmann-weighted average: weight each elite by exp(temperature·(min_cost −
cost)), normalize, and set the new mean/std to the weighted mean/std. A dramatically-better rollout
dominates; a cluster of mediocre wrong-route elites is exponentially suppressed rather than averaged in
at full weight. Execute the action by *sampling* one elite with probability proportional to its
weight — committing to a real rolled-out route rather than an in-the-wall mean of two incompatible
routes.

**Why.** The soft weighting is the principled analogue of CEM's hard cut: λ → 0 recovers the argmax, λ
→ ∞ recovers plain averaging, and a small temperature (0.005) sits in between — sharp enough that a
clearly better route dominates, soft enough that near-best sequences keep the variance from collapsing
onto one route the way iCEM's hard reuse did. White sampling (not colored) is deliberate — the soft
update does the routing work, so no low-frequency prior bakes in a committed direction before the cost
is read. Stochastic elite action-selection avoids executing the mean, which in a bimodal maze cost can
land in the wall. Like every rung, MPPI reads only function values — the differentiable model's gradient
stays unused.

**Hyperparameters.** `num_samples = 200`, `n_iters = 20`, `num_elites = max(20, num_samples//10) = 20`,
`max_std = 2.0` (wide init), `temperature = 0.005`, `max_norm` not re-clipped here (matches upstream
MPPIPlanner), zero init mean. Returns a weighted-sampled elite; no persisted state; `no_grad`.

```python
# EDITABLE region of eb_jepa/custom_planner.py — step 4: MPPI
class CustomPlanner(Planner):
    """MPPI (Model Predictive Path Integral) planner for JEPA world models."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        # Match upstream MPPIPlanner defaults — planning_mppi.yaml sets
        # var_scale=1.5 but MPPIPlanner doesn't accept that kwarg, so the
        # effective config is max_std=2 (class default), temperature=0.005,
        # num_elites=20 (yaml). Mirror that here.
        self.num_elites = max(20, num_samples // 10)
        self.max_std = 2.0
        self.temperature = 0.005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)

        losses = []
        elite_means = []
        elite_stds = []

        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=self.device,
            )

            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # MPPI weighted update
            min_cost = cost.min(0)[0]
            score = torch.exp(
                self.temperature * (min_cost - elite_loss[:, 0])
            )
            score /= score.sum(0) + 1e-9
            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
            )

        # Select action via weighted sampling
        score_np = score.cpu().numpy()
        selected = elite_actions[
            :, np.random.choice(np.arange(score_np.shape[0]), p=score_np)
        ]

        return PlanningResult(
            actions=selected,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
