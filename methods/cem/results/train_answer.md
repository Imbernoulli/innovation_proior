The problem is to optimize a black-box objective over action sequences when we only get function values back, with no reliable gradient, possibly noisy evaluations, many local optima, and a tight query budget. Each query is expensive because it rolls the sequence through a fixed world model and scores the resulting trajectory. Pure random shooting satisfies the gradient-free requirement and explores globally, but it never learns: every batch is drawn from the same fixed distribution, so most evaluations are wasted on regions already known to be poor. Simulated annealing and genetic algorithms add adaptation, yet their updates are heuristic analogies rather than derivable rules, and they do not cleanly answer what distribution the search is converging to. What is missing is a principled, statistically grounded way to turn past scores into the next sampling distribution.

The right bridge comes from rare-event simulation. Suppose we ask for the probability that a random draw exceeds a threshold gamma. As gamma approaches the true optimum, that event becomes rare, and the variance-optimal importance-sampling density for it is the baseline density restricted to the threshold-beating set and renormalized. That restricted density is exactly the "concentrate on the good region" distribution we want, but it is not directly usable because it contains the unknown probability and may be hard to sample from. The Cross-Entropy Method resolves this by choosing the closest member of a tractable parametric family to that ideal restricted density, measured by Kullback-Leibler divergence. This turns the search for a good sampling law into a sequence of familiar parameter-estimation problems.

Minimizing the KL divergence from the restricted target to a parametric family f(·; v) reduces, after dropping terms that do not depend on v, to maximizing the expected log-likelihood of the threshold-beating samples under f(·; v). On a finite batch drawn from the current law, this is simply a maximum-likelihood fit of f(·; v) to the elite samples. A fixed high threshold would often yield an empty elite set, so instead the threshold is set adaptively to the worst score among the top rho fraction of the batch. This self-tuning level ratchets upward as the distribution concentrates, keeping the update well-defined at every iteration while still climbing toward the optimum.

For action sequences the natural tractable family is a diagonal Gaussian. It samples cheaply and its maximum-likelihood fit has a closed form: the new mean is the per-coordinate mean of the elites, and the new variance is the per-coordinate mean squared deviation of the elites. The mean recenters the search on the cluster of good sequences, while the variance auto-sizes exploration, shrinking where elites agree and staying wider where they disagree. Iterating sample, score, select elites, refit mean and variance concentrates the Gaussian onto the high-performing region. Optional smoothing between iterations can slow premature collapse; in the basic form below the raw update is used, and sampled actions are projected into the feasible norm ball before scoring. The loop returns the final mean as the plan, since in the concentrated limit the mean is the point estimate of the optimum.

```python
import torch
from einops import rearrange


class CEMPlanner(Planner):
    """CEM planner for a fixed world model: refit a diagonal Gaussian to low-cost samples."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=300, n_iters=30, var_scale=1,
                 num_elites=10, max_norms=None, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.max_norms = max_norms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        T = min(self.plan_length, steps_left) if steps_left else self.plan_length

        mean = torch.zeros(T, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(T, self.action_dim, device=self.device)
        actions = torch.empty(T, self.num_samples, self.action_dim, device=self.device)
        losses, elite_means, elite_stds = [], [], []

        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                T, self.num_samples, self.action_dim, device=std.device)

            if self.max_norms is not None:
                assert len(self.max_norms) == 1
                max_norm, eps = self.max_norms[0], 1e-6
                norms = actions.norm(dim=-1, keepdim=True)
                max_norms = torch.ones_like(norms) * max_norm
                min_norms = torch.zeros_like(norms)
                coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
                actions = actions * coeff

            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss = cost[elite_idxs]
            elite_actions = actions[:, elite_idxs]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            mean = elite_actions.mean(dim=1)
            std = elite_actions.std(dim=1)

        return PlanningResult(
            actions=mean,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
