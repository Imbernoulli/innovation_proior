# RED-diff, distilled

RED-diff is a variational sampler for inverse problems with a pretrained diffusion prior.
Instead of approximating the intractable posterior score along the diffusion trajectory
(as DPS / ΠGDM do, by differentiating through the denoiser), it fits a Gaussian
`q(x_0|y) = N(mu, sigma^2 I)` to the true posterior `p(x_0|y)` by minimizing
`KL(q || p(x_0|y))`. The KL-to-prior term expands, via the maximum-likelihood
diffusion identity, into a time-integrated weighted score-matching loss; its gradient
reduces to a denoising *residual* `lambda_t (epsilon_theta(x_t;t) - epsilon)` with **no
backprop through the frozen denoiser**. Sampling becomes stochastic optimization of `mu`
with an off-the-shelf optimizer (Adam), regularized by denoising — hence the name, in
the spirit of Regularization by Denoising (RED) but generative and using the entire
diffusion trajectory.

## Problem it solves

Sample the posterior `p(x_0|y) ∝ p(y|x_0) p(x_0)` for an inverse problem
`y = f(x_0) + v`, `v ~ N(0, sigma_v^2 I)`, with `f` known (linear or nonlinear), using a
*pretrained* diffusion model as the prior `p(x_0)` in a plug-and-play fashion — universal
across tasks, GPU-efficient, and easy to tune.

## Key idea

1. **Variational, not score-modification.** Posit `q = N(mu, sigma^2 I)` and minimize
   `KL(q(x_0|y) || p(x_0|y))`. Bayes-expand:

   ```
   KL(q || p(x_0|y)) = -E_q[log p(y|x_0)] + KL(q || p(x_0)) + log p(y).
   ```

   `log p(y)` is constant in `q`. The first term is the reconstruction loss
   `(1/2 sigma_v^2) E_q[||y - f(x_0)||^2]`.

2. **Prior-KL → score matching.** Both `q` and `p(x_0)` are diffused by the same forward
   SDE, so (Song et al. 2021; Vahdat et al. 2021)

   ```
   KL(q(x_0|y) || p(x_0)) = ∫_0^T (beta(t)/2) E_{q(x_t|y)}[ ||∇log q(x_t|y) - ∇log p(x_t)||^2 ] dt,
   ```

   with `q(x_t|y) = N(alpha_t mu, (alpha_t^2 sigma^2 + sigma_t^2) I)` (closed-form score)
   and `∇log p(x_t) = -epsilon_theta(x_t;t)/sigma_t` from the pretrained net.

3. **σ → 0** gives a Dirac `q = delta(x_0 - mu)`, and the objective simplifies to

   ```
   min_mu  ||y - f(mu)||^2  +  E_{t,eps}[ 2 omega(t) (sigma_v/sigma_t)^2 ||epsilon_theta(x_t;t) - eps||^2 ],
   x_t = alpha_t mu + sigma_t eps.
   ```

4. **Cheap gradient (the crux).** Using the time-derivative identity
   `d KL_t/dt = -(beta(t)/2) E[||∇log q - ∇log p||^2]` and integration by parts, **if the
   weighting satisfies `omega(0) = 0`** the boundary term vanishes (`KL_T = 0` since
   `x_T` is pure noise; `omega(0)=0` kills the `t=0` end), and the regularizer gradient
   collapses to

   ```
   ∇_mu reg = E_{t~U[0,T], eps~N(0,I)}[ lambda_t (epsilon_theta(x_t;t) - eps) ],
   lambda_t := 2 T sigma_v^2 (alpha_t/sigma_t) omega'(t).
   ```

   `epsilon_theta` enters as a stop-gradient *value* — no score Jacobian, unlike DPS/ΠGDM.

5. **SNR weighting.** The noise residual blows up as `t → 0`. Recast in the signal domain
   using Tweedie `mu_hat_t = E[x_0|x_t] = (x_t - sigma_t epsilon_theta)/alpha_t` and the
   exact identity

   ```
   mu - mu_hat_t = (sigma_t/alpha_t)(epsilon_theta(x_t;t) - eps),
   ```

   so choosing `lambda_t = lambda / SNR_t` with `SNR_t = alpha_t/sigma_t` converts the
   noise-domain residual to the signal-domain one, gives a single interpretable
   bias-variance knob `lambda`, upweights coarse (high-noise) steps and downweights
   detail (low-noise) steps. Step `t` in **descending** order (T → 0).

6. **RED connection.** The per-step regularizer `(sg[epsilon_theta - eps])^T mu` mirrors
   RED's residual penalty (gradient = `x - f_den(x)`, no denoiser derivative). Unlike RED
   it injects noise into every denoiser across the whole trajectory, rather than relying
   on one deterministic denoiser at one fixed point.

## Final algorithm

```
Input: y, f, sigma_v, L steps, schedule {alpha_t, sigma_t, lambda_t}
Initialize mu_0 = 0                              # InverseBench; a pseudo-inverse warm start is also possible
for l = 1..L:                                   # descending t: T -> 0
    eps ~ N(0, I)
    x_t = alpha_t * mu + sigma_t * eps
    loss = ||y - f(mu)||^2 + lambda_t * (sg[epsilon_theta(x_t;t) - eps])^T mu
    mu <- OptimizerStep(loss)                   # Adam / SGD+momentum
return mu
```

The gradient applied is `observation_weight * ∇_mu ||f(mu) - y||^2 + lambda_t (epsilon_theta - eps)`.

## Defaults and why

- `lambda` (e.g. 0.25): the single prior/likelihood trade-off knob; larger leans on the
  prior, smaller on the data.
- `lambda_t = lambda / SNR_t` ("linear" `1/SNR` weighting): the exact rescaling that maps
  the noise residual to the signal residual.
- Adam with `betas = (0.9, 0.99)` and no weight decay: off-the-shelf optimization of the
  clean estimate `mu`; InverseBench exposes `base_lr=0.5`, while the original image code
  commonly uses `lr=0.1`.
- 1000 steps by default; descending time stepping; `sigma = 0` (no Gaussian dispersion,
  because isotropic Gaussian perturbations leave the image manifold).

## Working code

Faithful to the InverseBench implementation: maintain `mu` as the optimized clean-signal
estimate; each step noise it to `x_t`, take one forward pass of the frozen denoiser to
get the predicted noise (detached), combine the data-fitting gradient with the weighted
residual, and let Adam move `mu`. The scheduler's `sigma` is inverse SNR, so the
`linear` lambda schedule implements `lambda/SNR_t`; the benchmark default is constant.

```python
import torch
import tqdm
from .base import Algo
import wandb
from utils.scheduler import Scheduler


class REDDiff(Algo):
    def __init__(self, net, forward_op, num_steps=1000, observation_weight=1.0,
                 base_lambda=0.25, base_lr=0.5, lambda_scheduling_type='constant'):
        super(REDDiff, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.scheduler = Scheduler(num_steps=num_steps, schedule='vp',
                                   timestep='vp', scaling='vp')
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        if lambda_scheduling_type == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_scheduling_type == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_scheduling_type == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        mu = torch.zeros(num_samples, self.net.img_channels,
                         self.net.img_resolution, self.net.img_resolution,
                         device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))

        for step in pbar:
            with torch.no_grad():
                sigma, scaling = self.scheduler.sigma_steps[step], self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            lam = self.lambda_fn(sigma)  # sigma equals 1/SNR; the linear option gives lambda/SNR_t.
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(mu, observation,
                                                            return_loss=True)
            gradient = gradient * self.observation_weight + lam * (pred_epsilon - epsilon)
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(
                f'Iteration {step + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)}, step=step)
        return mu
```

## Dispersion (optional, derived in full)

Keeping `q = N(mu, sigma^2 I)` with `eta_t := (1 + sigma^2 (alpha_t/sigma_t)^2)^{1/2}`,
`x_t = alpha_t mu + eta_t sigma_t eps`: the mean gradient is unchanged in expectation
(`∇_mu reg = E[lambda_t (epsilon_theta - eta_t^{-1} eps)] = E[lambda_t epsilon_theta]`,
since `E[eps] = 0`), and the dispersion gradient
is closed-form,
`∇_sigma reg = sigma E[lambda_t 2 eta_t^{-1} (alpha_t/sigma_t) eps^T (epsilon_theta - eta_t^{-1} eps)]`.
Default `sigma = 0`: Gaussian dispersion perturbs an image off the natural-image
manifold, so it is not a good diversity model; diversity comes from the random `eps` draws.
