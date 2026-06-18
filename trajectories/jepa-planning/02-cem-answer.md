**Problem.** Random shooting throws away every score — it re-draws the same white-noise distribution
every pass and learns nothing within a step, so it wanders (random's `mean_steps_to_success` ran
64/82/112). The fix is to let the sampling distribution *move toward* the low-cost sequences,
iteration after iteration — and to do it from a principle, not an ad-hoc nudge.

**Key idea (Cross-Entropy Method).** Reframe "minimize cost" as "make the rare event `{S ≥ γ}` likely";
the variance-optimal importance density is the baseline restricted to the elites, and the closest
tractable fit to it in KL collapses, on a finite batch, to a *maximum-likelihood fit on the
threshold-beating samples*. For a diagonal Gaussian over (timestep, coordinate) the elite MLE is the
per-entry **mean and variance of the elites**. So: sample a batch from the current Gaussian, clip each
action into the feasible ball, roll through the model, keep the lowest-cost top-10% elites, refit the
Gaussian to their mean and std, repeat — and return the final mean as the plan.

**Why.** The mean re-centers the search where good plans clustered; the variance auto-sizes exploration
(tighter where elites agreed) — the adaptivity the fixed-spread floor lacked. A self-tracking elite
quantile keeps the elite set non-empty so the threshold ratchets toward the optimum on its own. A wide
initial spread (`var_scale = 1.5`) guards against the Gaussian collapsing onto the wrong route through
the door before both routes are sampled. Like random search, CEM only ever reads function values — it
never uses the differentiable model's gradient.

**Hyperparameters.** `num_samples = 200`, `n_iters = 20`, `num_elites = max(10, num_samples//10) = 20`
(top 10%), `var_scale = 1.5` (wide init std), zero init mean, `max_norm = 2.45` clip. Returns the final
mean; no persisted state; `no_grad`.

```python
# EDITABLE region of eb_jepa/custom_planner.py — step 2: CEM
class CustomPlanner(Planner):
    """CEM (Cross-Entropy Method) planner for JEPA world models."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.num_elites = max(10, num_samples // 10)
        self.var_scale = 1.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(plan_length, self.num_samples, self.action_dim, device=self.device)

        losses = []
        elite_means = []
        elite_stds = []

        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=self.device,
            )

            # Clip action norms
            max_norm = 2.45
            eps = 1e-6
            norms = actions.norm(dim=-1, keepdim=True)
            max_norms = torch.ones_like(norms) * max_norm
            min_norms = torch.ones_like(norms) * 0
            coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
            actions = actions * coeff

            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            mean = torch.mean(elite_actions, dim=1)
            std = torch.std(elite_actions, dim=1)

        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
