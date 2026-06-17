# Unconditional trajectory-level diffusion planner, distilled

This is the generative core of a diffusion planner for offline control: a denoising diffusion
probabilistic model over whole trajectories, where **sampling from the model is planning**. A
trajectory is represented as a 2D array (one column per timestep, state features stacked on
action features) and the entire plan is denoised at once. The only conditioning is a hard
start-state constraint imposed by inpainting; there is **no reward, no classifier guidance, no
return conditioning, and no candidate re-ranking**. It is the reward-agnostic, dynamically-
plausible plan distribution.

## Problem it solves

Given a fixed offline dataset of logged trajectories and unknown dynamics, produce a controller
that plans over long horizons without (a) the compounding rollout error of single-step learned
models, (b) the model-exploitation of a strong planner searching against an imperfect model, or
(c) the anti-causal-ordering problem of left-to-right autoregressive generation. The fix is to
collapse "model" and "planner" into one object: a generative model of trajectories whose
sampling procedure is the plan.

## Key ideas

- **Trajectory as a 2D array, states and actions jointly.** `tau` has one column per planning
  timestep and rows = `obs_dim` state features stacked on `act_dim` action features. Actions are
  "just more dimensions" of each column, so states and actions are generated jointly and the
  model is trained for the quality of the controller it yields, not only state prediction.
- **Non-autoregressive, whole-plan denoising.** All timesteps are produced simultaneously by the
  reverse diffusion chain, sidestepping compounding rollout error and the anti-causal ordering
  problem (every timestep conditions on every other at each denoising step).
- **1D temporal U-Net.** Convolve along the horizon axis only (translation-equivariant, local in
  time); treat the heterogeneous, non-interchangeable state/action features as channels. A 2D
  convolution would wrongly impose equivariance across the feature axis. Multi-scale
  down/up-sampling plus many composed local denoising steps turns local consistency into global
  coherence. The implementation is fully convolutional in the horizon but its U-Net skip layout
  requires a power-of-two horizon such as 32. GroupNorm + Mish + a sinusoidal diffusion-step
  embedding added in each block; no attention needed for low-dimensional locomotion trajectories;
  kernel size 5.
- **Start-state conditioning = inpainting.** Every plan must begin at the current observation;
  clamp the column-0 state coordinates to it at every step of the chain (a `fix_mask`) and zero
  their training loss. This is a hard constraint, not reward.
- **No reward input.** `p(tau)` only models which state-action sequences are dynamically
  plausible under the data.
- **First-action loss weighting.** Only `tau[0, obs:]` is executed in receding-horizon control,
  so its reconstruction is up-weighted (e.g. 10x).
- **Short cosine-scheduled chain.** Locomotion trajectories are low-dimensional and smooth, so
  ~20 denoising steps suffice; a normalized cosine schedule keeps SNR well behaved over a short
  chain. EMA (0.9999) weights stabilize sampling. `x0`-prediction (`predict_noise=False`) is the
  CleanDiffuser configuration; the reverse step recovers `eps` in closed form.

## The diffusion engine (DDPM)

DDPM notation: `beta_i` is the forward variance, `a_i = 1 - beta_i`, and
`abar_i = prod_{j<=i} a_j`. The forward process has the closed form

```
q(x^i | x^0) = N(sqrt(abar_i) x^0, (1 - abar_i) I)
x^i = sqrt(abar_i) x^0 + sqrt(1 - abar_i) eps,    eps ~ N(0, I)
```

The tractable forward posterior is

```
q(x^{i-1} | x^i, x^0) = N(mu_tilde_i, beta_tilde_i I)
beta_tilde_i = ((1 - abar_{i-1}) / (1 - abar_i)) beta_i
mu_tilde_i = [sqrt(abar_{i-1}) beta_i / (1 - abar_i)] x^0
             + [sqrt(a_i)(1 - abar_{i-1}) / (1 - abar_i)] x^i
```

With reverse variance `v_i I`, the middle ELBO term is
`(1/(2 v_i)) ||mu_tilde_i - mu_theta||^2 + C`. The epsilon parameterization is

```
mu_theta(x^i, i) = (1 / sqrt(a_i)) (x^i - beta_i / sqrt(1 - abar_i) * eps_theta(x^i, i))
```

which turns the weighted variational term into

```
E [ beta_i^2 / (2 v_i a_i (1 - abar_i))
    * ||eps - eps_theta(sqrt(abar_i) x^0 + sqrt(1 - abar_i) eps, i)||^2 ].
```

The simplified training objective drops that coefficient and samples the noise level uniformly.
CleanDiffuser supports both targets; for this planner `predict_noise=False`, so the network
predicts `x0` and the sampler computes
`eps_theta = (x^i - alpha_i x0_theta) / sigma_i`.

CleanDiffuser stores the cumulative signal/noise scales directly:

```
alpha_i = cos(((clip(t_i, 0, 0.9946) + s)/(1+s)) * pi/2) / cos((s/(1+s)) * pi/2),  s = 0.008
sigma_i = sqrt(1 - alpha_i^2)
x^i = alpha_i x^0 + sigma_i eps
```

The ancestral DDPM reverse step in that `(alpha, sigma)` form is

```
std_i  = (sigma_{i-1}/sigma_i) * sqrt(1 - (alpha_i/alpha_{i-1})^2)          # ancestral posterior std
x^{i-1} = (alpha_{i-1}/alpha_i)(x^i - sigma_i eps_theta)
          + sqrt(sigma_{i-1}^2 - std_i^2) * eps_theta
          + std_i * z,   z ~ N(0, I)   (z = 0 at i = 1)
```

The first term is `alpha_{i-1}` times the implied clean trajectory; the second adds back the
correlated noise that matches the forward marginal at level `i-1`. Re-clamp the start-state
coordinates after every step. With no guidance there is no classifier gradient and no
re-ranking: draw one plan, execute its first action.

## Algorithm

```
Train:
  repeat:
    obs, act ~ logged windows
    x^0 = concat(obs, act)                                  # (B, H, obs+act)
    i ~ Uniform diffusion index;  eps ~ N(0, I)
    x^i = alpha_i x^0 + sigma_i eps
    x^i = (1-fix_mask) x^i + fix_mask x^0                   # inpaint start
    target = eps if predict_noise else x^0
    loss = mean((net(x^i, i) - target)^2 * loss_weight * (1-fix_mask))
    update EMA weights

Plan(current_obs):                                          # receding horizon
  prior = zeros(n, H, obs+act);  prior[:,0,:obs] = current_obs
  x = randn * temperature;  x = (1-fix_mask)*x + fix_mask*prior
  for i = highest sampled diffusion level ... 1:
    pred = ema_net(x, i)
    if predict_noise: eps_theta = pred
    else:             eps_theta = (x - alpha_i pred) / sigma_i
    x = (alpha_{i-1}/alpha_i)(x - sigma_i eps_theta) + sqrt(sigma_{i-1}^2 - std_i^2) eps_theta
    if i > 1: x += std_i * randn
    x = (1-fix_mask)*x + fix_mask*prior                     # hold start-state constraint
  return clip(x[:,0,obs:], -1, 1)                           # first action only
```

## Working code

```python
import torch
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d


class PlannerModel:
    """Trajectory-level diffusion planner with start-state inpainting and no reward guidance."""
    def __init__(self, obs_dim, act_dim, horizon, device,
                 model_dim=32, dim_mult=(1, 2, 2, 2), diffusion_steps=20,
                 predict_noise=False, action_loss_weight=10.0,
                 ema_rate=0.9999, lr=2e-4):
        self.obs_dim, self.act_dim, self.horizon, self.device = obs_dim, act_dim, horizon, device
        self.diffusion_steps = diffusion_steps
        if horizon & (horizon - 1) != 0:
            raise ValueError("JannerUNet1d requires a power-of-two horizon.")

        self.net = JannerUNet1d(
            obs_dim + act_dim,
            model_dim=model_dim,
            emb_dim=model_dim,
            dim_mult=list(dim_mult),
            timestep_emb_type="positional",
            attention=False,
            kernel_size=5,
        )

        self.fix_mask = torch.zeros((horizon, obs_dim + act_dim))
        self.fix_mask[0, :obs_dim] = 1.0
        self.loss_weight = torch.ones((horizon, obs_dim + act_dim))
        self.loss_weight[0, obs_dim:] = action_loss_weight

        self.agent = DiscreteDiffusionSDE(
            nn_diffusion=self.net,
            nn_condition=None,
            fix_mask=self.fix_mask,
            loss_weight=self.loss_weight,
            classifier=None,
            ema_rate=ema_rate,
            optim_params={"lr": lr, "weight_decay": 1e-5},
            diffusion_steps=diffusion_steps,
            noise_schedule="cosine",
            predict_noise=predict_noise,
            device=device,
        )
        self.optimizer = self.agent.optimizer

    def update(self, obs, act):
        x = torch.cat([obs, act], dim=-1).to(self.device)
        return self.agent.update(x)

    @torch.no_grad()
    def plan(self, current_obs, temperature=0.5, sample_steps=None):
        current_obs = current_obs.to(self.device)
        n = current_obs.shape[0]
        prior = torch.zeros((n, self.horizon, self.obs_dim + self.act_dim), device=self.device)
        prior[:, 0, :self.obs_dim] = current_obs
        traj, _ = self.agent.sample(
            prior=prior,
            solver="ddpm",
            n_samples=n,
            sample_steps=sample_steps or self.diffusion_steps,
            use_ema=True,
            temperature=temperature,
            w_cfg=0.0,
            w_cg=0.0,
        )
        return traj[:, 0, self.obs_dim:].clip(-1.0, 1.0)
```
