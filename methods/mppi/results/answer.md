# MPPI (Model Predictive Path Integral Control), distilled

MPPI is a derivative-free, sampling-based stochastic optimal controller for receding-horizon
control of nonlinear systems with non-convex or discontinuous costs. Each control step it
samples a population of action sequences from a Gaussian, rolls them through the (possibly
learned) dynamics model, weights each by the exponential of its negative cost — the
Gibbs/optimal trajectory distribution that comes out of the path-integral form of optimal
control — and refits the Gaussian's mean *and* variance to those weighted samples, iterating a
few times before executing. Its defining contribution over earlier path-integral control is a
generalized importance-sampling scheme that lets the *variance* of the sampling distribution be
changed, not only the mean.

## Problem it solves

`min_u E[ phi(x_T) + integral (q(x,t) + (1/2) u^T R u) dt ]` for a control-affine stochastic
system `dx = (f + Gu) dt + B dw`, where the dynamics are strongly nonlinear and the state cost
may be discontinuous (e.g. an impulse on collision). Solving the stochastic HJB PDE is
intractable (curse of dimensionality); derivative-based optimizers (DDP/iLQG) need dynamics
derivatives and a quadratic cost model, which a discontinuous cost denies. MPPI needs neither.

## Key idea

1. **Linearize via log-transform + Feynman-Kac.** Assume the noise enters only actuated
   channels with covariance tied to the control cost by a scalar `lambda`:
   `B_c B_c^T = lambda G_c R^{-1} G_c^T`. Then `V = -lambda log(psi)` turns the nonlinear
   stochastic HJB into a *linear* backward PDE for `psi`, whose Feynman-Kac solution is a path
   integral over forward trajectories of the **uncontrolled** dynamics `P`:

   ```
   psi(x_0, t_0) = E_P[ exp( -(1/lambda) S(tau) ) ],   S(tau) = phi(x_T) + integral q dt.
   ```

   The optimal control becomes an expectation evaluated by forward sampling, not a backward
   sweep. `lambda` plays the role of temperature (free-energy view).

2. **Generalized importance sampling (the contribution).** Sampling from `P` is hopeless (the
   natural noise rarely produces low-cost trajectories). Importance-sample from a distribution
   `q` with a shifted mean **and a scaled covariance** `B_E = (0; A_t B_c)`. The likelihood
   ratio between the two discrete diffusion processes is

   ```
   p(tau)/q(tau) = (prod_i |A_{t_i}|) exp( -(Delta t / 2) sum_i Q_i ),
   Q_i = (z_i - mu_i)^T Gamma_i^{-1} (z_i - mu_i) + 2 mu_i^T Sigma_i^{-1} (z_i - mu_i)
         + mu_i^T Sigma_i^{-1} mu_i,
   Gamma_i^{-1} = Sigma_i^{-1} - Lambda_i^{-1},   Lambda_i = A_{t_i}^T Sigma_i A_{t_i}.
   ```

   The last two `Q_i` terms are exactly Girsanov's mean-change terms; the **first term** is new
   and carries the variance change (it vanishes when `A_t = I`), interpreted as a penalty on
   over-aggressive exploration. Proof: divide the two Gaussian-product densities, define
   `Lambda_i` as the sampling covariance in the Gaussian quadratic, set
   `Gamma_i^{-1} = Sigma_i^{-1} - Lambda_i^{-1}`, and complete the square on the difference of
   quadratics. The determinant prefactor reduces to `prod_i |A_{t_i}|`; after substituting
   `z_i = (z_i - mu_i) + mu_i`, the linear and constant `Lambda_i^{-1}` terms cancel, leaving
   `Q_i`.

3. **Fold the ratio into the cost; weighted update.** The state-independent `prod_i |A_{t_i}|`
   cancels between numerator and denominator; the rest becomes an extra running cost. In the
   scalar-`A` special case (`A = sqrt(nu) I`, with the path-integral prefactor reducing to
   `G_c^{-1}`), writing the noise as a random control perturbation `delta u`, the update is a
   reward-weighted average:

   ```
   u_{t_i}* ≈ u_{t_i} + ( sum_k exp(-tilde S(tau_{i,k})/lambda) delta u_{i,k} )
                        / ( sum_k exp(-tilde S(tau_{i,k})/lambda) ),
   ```

   with `H^{-1} = G^{-T} R G^{-1}`, the augmented running cost reduces to
   `tilde q = q + (1 - nu^{-1})/2 delta u^T R delta u + u^T R delta u + (1/2) u^T R u` —
   the importance ratio simply re-injects the original quadratic control cost.

## Free-energy / KL equivalent (the same weight, derived independently)

Free energy `F = log E_P[exp(-S/lambda)]`. Jensen gives the lower bound
`-lambda F <= E_Q[S] + lambda D_KL(Q || P)`, tight at the optimal (Gibbs) distribution
`q*(V) = (1/eta) exp(-S(V)/lambda) p(V)`. Minimizing `D_KL(Q* || Q_U)` and importance-sampling
yields the weight `w(V) ∝ exp(-S(V)/lambda)`. `lambda` is the temperature: `lambda -> 0` puts
all mass on the best trajectory (greedy); `lambda -> infinity` is a uniform average. For
numerical safety, shift costs by the minimum sampled cost `rho` (multiplying weights by
`exp(rho/lambda)/exp(rho/lambda) = 1`), so the best sample's weight is `exp(0) = 1`:
`w ∝ exp(-(S - min S)/lambda)`.

## Defaults and why (latent-planning / sampling form)

- **`max_std` (initial exploration std, ~2):** start wide so the first iteration escapes local
  basins of the cost surface — directly addresses the failure mode of mean-only path-integral
  control, whose fixed low natural variance leaves rollouts too timid.
- **`temperature = 1/lambda` (class default `0.005`):** the cost-weight sharpness. Small temperature (large
  `lambda`) is a soft, broad weighting that averages over the elites — robust when model
  rollouts are noisy, instead of collapsing onto a single sample.
- **`num_elites` (top-k):** keep only the best-`k` samples before applying the soft exp-weight,
  so far-out high-cost samples don't contaminate the weighted mean/variance fit (the variance
  estimate is outlier-sensitive); soft weighting preserves graded quality within the elites.
- **refit both mean and std:** the mean update is the path-integral reward-weighted average; the
  std update is the empirical weighted spread of the good samples — the *variance change* that
  is the method's whole point, adapted online instead of frozen.

## Relation to prior methods

- **PI^2 / mean-only path-integral control:** same exponentiated-cost weighted average of
  control variations, but only changes the **mean** of the sampling distribution; variance is
  fixed at the natural noise. MPPI = this plus the generalized likelihood ratio that also
  changes the variance.
- **CEM:** also samples a Gaussian, scores, keeps top-`k` elites, and refits — but with a
  **hard, unweighted** elite average (every elite equal, the rest discarded). MPPI applies the
  **soft** Gibbs weight `exp((min cost - cost)/lambda)` over the elites, using the cost
  magnitudes the objective reports; CEM is the hard-selection cousin with uniform weights inside
  its elite set.
- **DDP/iLQG:** derivative-based, needs a quadratic cost model; cannot represent a discontinuous
  cost or non-smooth dynamics. MPPI is derivative-free and handles both.

## Final algorithm (sampling-MPPI, receding horizon)

```
each control step, given obs_init and a horizon T:
  mean <- 0  (T, A);   std <- max_std  (T, A)
  repeat n_iters:
    actions  <- mean + std * randn(T, N, A)              # sample population
    cost[k]  <- objective( unroll(obs_init, actions_k) ) # roll out + score (lower better)
    elites   <- top-k samples by lowest cost
    score[e] <- exp( (1/lambda) * (min(cost) - cost[e]) ); score /= sum(score)   # Gibbs weight
    mean     <- sum_e score[e] * elite_actions[e]                                # weighted mean
    std      <- sqrt( sum_e score[e] * (elite_actions[e] - mean)^2 )             # weighted std
  sample one final elite by `score`; execute; re-plan next step
```

## Working code

Concrete planner over a fixed rollout model, filling the refinement slot of the sampling
planner. `unroll` forward-simulates a batch of action sequences; `objective` scores predicted
states (lower is better); `cost_function` composes them.

```python
import numpy as np
import torch
from einops import rearrange


class MPPIPlanner(Planner):
    """Model Predictive Path Integral planner. Maintains a per-timestep Gaussian
    over action sequences; each iteration samples a population, scores it by the
    model rollout cost, and refits the Gaussian to the cost-exponentiated (Gibbs)
    weighted elite samples -- refining BOTH mean and std."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=500, n_iters=15,
                 max_std=2.0, num_elites=64, temperature=0.005, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.max_std = max_std                 # large initial exploration width
        self.num_elites = num_elites           # top-k kept before soft weighting
        self.temperature = temperature         # = 1 / lambda (small => soft, broad weighting)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None):
        T = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # action-sequence Gaussian: per-timestep mean and std
        mean = torch.zeros(T, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(T, self.action_dim, device=self.device)
        actions = torch.empty(T, self.num_samples, self.action_dim, device=self.device)

        losses, elite_means, elite_stds = [], [], []

        for _ in range(self.n_iters):
            # sample a population of action sequences from the current Gaussian
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                T, self.num_samples, self.action_dim, device=std.device)            # T B A

            # score every sequence by rolling it through the model (lower is better)
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init).unsqueeze(1)        # B 1
            losses.append(cost.min().item())

            # keep the top-k elites, then soft-weight within that set
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Gibbs / path-integral weights over the cost gap from the best sample
            # (cost-shift = numerical conditioning; it multiplies the weights by 1)
            min_cost = cost.min(0)[0]
            score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))     # B'
            score /= score.sum(0)

            # refit mean (reward-weighted average) and std (weighted spread = variance change)
            mean = torch.sum(score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9)                                                  # T A
            std = torch.sqrt(torch.sum(
                score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2,
                dim=1) / (score.sum(0) + 1e-9))

        # commit a plan: draw one final elite according to its path-integral weight
        score_np = score.cpu().numpy()
        selected = elite_actions[:, np.random.choice(np.arange(score_np.shape[0]), p=score_np)]

        return PlanningResult(
            actions=selected,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
```
