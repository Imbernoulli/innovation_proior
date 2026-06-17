# The Cross-Entropy Method (CEM), distilled

The Cross-Entropy Method is a gradient-free, sampling-based optimizer for a black-box objective
`S(x)`. It maintains a parametric sampling distribution `f(·; v)` over candidate solutions and, each
iteration, draws a batch, keeps the top fraction of them (the **elite set**), and **refits the
distribution to the elites by maximum likelihood**. Iterating concentrates the distribution onto the
high-performing region; in the limit it collapses to a point mass at the optimum.

## Problem it solves

Maximize `S(x)` (or minimize a cost `C = −S`) when `S` is a black box queried by value only — no
gradient, possibly noisy, multi-extremal, moderate dimension. The canonical deployment here is a
derivative-free **planner**: `x` is an action sequence, `S(x)` is minus the cost of rolling it
through a fixed world model, and the optimizer searches for a good plan within a fixed query budget.

## Key idea

Associate the optimization with a rare-event probability `ℓ(γ) = P_u(S(X) ≥ γ)`; as `γ → γ* =
S(x*)` this is a rare event, and the **variance-optimal importance-sampling density** for it is the
baseline density restricted to the elite event and renormalized,

```
g*(x) = I{S(x) ≥ γ} · f(x; u) / ℓ ,
```

which is exactly the "all mass on the good region" distribution we want. We cannot use `g*` directly,
so at each iteration we build the same restricted target from the current sampling law,
`g*_t(x) = I{S(x) ≥ γ} f(x; v_{t-1}) / ℓ_t`, and pick the member of a tractable family `f(·; v)`
closest to it in **Kullback-Leibler (cross-entropy) divergence**. Since
`D(g*_t, f(·;v)) = const - ∫ g*_t ln f(·;v)`, minimizing KL equals

```
max_v  E_{v_{t-1}}[ I{S(X) ≥ γ} · ln f(X; v) ]       # current law used as reference
   ≈   max_v  Σ_{k : S(X_k) ≥ γ} ln f(X_k; v)     (X_1..X_N ~ f(·; v_{t-1}))
```

— i.e. **maximum-likelihood fit of the sampling distribution to the elite samples**. A fixed high
`γ` makes the elite set empty, so use an **adaptive level**: sort the current batch's scores and set
`γ̂_t` to the worst score among the top `N^e = ceil(ρN)` samples. The level self-tunes upward toward
`γ*` as the distribution concentrates.

## Generic algorithm (CE for optimization)

```
Choose family f(·; v), initial v_0, sample size N, rarity rho (N^e = ceil(rho*N)).
for t = 1, 2, ... :
    draw X_1, ..., X_N  ~  f(·; v_{t-1})
    score S(X_1), ..., S(X_N); sort; gamma_hat_t = S_(N-N^e+1)           # top N^e are elites
    elites  =  { X_k : S(X_k) >= gamma_hat_t }
    v_tilde_t  =  argmax_v  sum_{k in elites} ln f(X_k; v)               # MLE on the elites
    v_t  =  alpha * v_tilde_t  +  (1 - alpha) * v_{t-1}                  # optional smoothing
stop when the distribution has degenerated (spread < eps) or the best score stalls.
```

For a discrete family (independent Bernoulli/categorical) the MLE is the elite frequency of each
value; for a continuous family the natural choice is the Gaussian.

## Continuous case: Gaussian normal-updating

Sample each coordinate independently `x_j ~ N(μ_j, σ_j²)`. The elite-restricted MLE has a closed
form — set the derivatives of the elite log-likelihood `L = Σ_{k∈I} Σ_j [−ln σ_j − (x_{kj}−μ_j)²/
(2σ_j²) − ½ln2π]` to zero:

```
dL/dmu_j = 0   =>   mu_j   = (1/N^e) sum_{k in I} x_{kj}                 # elite sample mean
dL/dsig_j = 0  =>   sig_j^2 = (1/N^e) sum_{k in I} (x_{kj} - mu_j)^2     # elite MLE variance
```

So the mathematical update is the per-coordinate **mean and MLE variance of the elite samples**,
with the variance divided by `N^e`. The mean re-centers the search; the variance auto-sizes the
exploration spread -- tight where elites agree, wide where they don't. The continuous normal-updating
form stores the spread as `σ`, the standard deviation. The planner implementation below follows the
same mean/std elite refit loop and uses `torch.std` over the elite dimension for the stored spread.
Optional smoothing (`α ≈ 0.4–0.9`) damps the spread so it does not collapse prematurely; the planner
code below uses the raw update (`α = 1`) and clips sampled actions into the feasible norm ball before
scoring them.

## Relation to prior methods

- **Random shooting / pure random search**: fixed-proposal sampling keeps drawing from the same law;
  CEM adds the elite-MLE refit so the proposal learns from past evaluations.
- **Estimation of distribution / evolutionary**: CEM is the principled member of this family — the
  "selection + refit" update is derived as KL minimization to the optimal IS density, not by
  biological analogy; the elite frequencies/moments are the maximum-likelihood fit.
- **Importance sampling for rare events**: CEM is the same adaptive-IS machinery turned toward
  optimization by sending the level `γ` up toward `γ*`.

## Working code (CEM planner over action sequences)

Fills the `plan` slot of the model-predictive-control harness; the cost is a black box queried via
`cost_function` (lower is better, `S = −cost`).

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

        mean = torch.zeros(T, self.action_dim, device=self.device)          # v_0: mu
        std = self.var_scale * torch.ones(T, self.action_dim, device=self.device)  # v_0: sigma
        actions = torch.empty(T, self.num_samples, self.action_dim, device=self.device)
        losses, elite_means, elite_stds = [], [], []

        for _ in range(self.n_iters):
            # draw X_1..X_N ~ N(mean, std^2)
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                T, self.num_samples, self.action_dim, device=std.device)

            # enforce feasible action set when the harness supplies a norm limit
            if self.max_norms is not None:
                assert len(self.max_norms) == 1
                max_norm, eps = self.max_norms[0], 1e-6
                norms = actions.norm(dim=-1, keepdim=True)
                max_norms = torch.ones_like(norms) * max_norm
                min_norms = torch.zeros_like(norms)
                coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)
                actions = actions * coeff

            # query black-box cost (lower is better)
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # elites = lowest cost = largest S = -cost
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss = cost[elite_idxs]
            elite_actions = actions[:, elite_idxs]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # update center and stored spread over the elite dimension
            mean = elite_actions.mean(dim=1)
            std = elite_actions.std(dim=1)

        return PlanningResult(
            actions=mean,   # the Gaussian's center = point estimate of the optimum
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
