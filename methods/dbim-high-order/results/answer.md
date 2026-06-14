# DBIM high-order solver, distilled

The high-order DBIM (Diffusion Bridge Implicit Models) solver is a training-free fast
sampler for a pretrained diffusion bridge. It generalizes a denoising diffusion bridge model
(DDBM) to a family of non-Markovian bridges sharing the same marginals (so the trained data
predictor is reused as-is), whose deterministic limit is the bridge analogue of DDIM. That
deterministic sampler is an Euler discretization of a clean ODE for `x_t/c_t`; recognizing the
ODE as semi-linear and applying an exponential integrator in the bridge log-SNR variable
reduces sampling to an exponentially-weighted integral of the predictor, which is approximated
to second/third order by a multistep (Adams-Bashforth) Taylor expansion — one denoiser call
per step, so a tiny call budget buys the most steps.

## Problem it solves

Sample a pretrained diffusion bridge for image-to-image translation/restoration under a strict
per-sample denoiser-call budget, without retraining and without changing the learned predictor
— by replacing only the per-step transition rule.

## Setup

Base schedule `alpha_t, sigma_t`, `SNR_t = alpha_t^2/sigma_t^2`. Bridge forward kernel given
both endpoints:

```
q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t^2 I),
a_t = (alpha_t/alpha_T)(SNR_T/SNR_t),  b_t = alpha_t (1 - SNR_T/SNR_t),  c_t^2 = sigma_t^2 (1 - SNR_T/SNR_t).
```

Data predictor `x_theta(x_t, t, x_T)` recovers `x_0` (trained by denoising bridge score
matching, reused unchanged).

## Key ideas

1. **Non-Markovian, marginal-preserving family (DBIM).** For variance vector `rho` (with
   `rho_{N-1} = c_{t_{N-1}}`):
   ```
   q^(rho)(x_{t_n}|x_0,x_{t_{n+1}},x_T) = N(a_{t_n} x_T + b_{t_n} x_0
       + sqrt(c_{t_n}^2 - rho_n^2) (x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x_0)/c_{t_{n+1}}, rho_n^2 I).
   ```
   Marginal preservation `q^(rho)(x_{t_n}|x_T) = q(x_{t_n}|x_T)` (downward induction; base case
   forces `rho_{N-1} = c_{t_{N-1}}`), and the variational objective is a positive reweighting of
   the data-prediction denoising terms. After converting data prediction to bridge-score matching,
   the corresponding score-loss weight is
   `gamma(t_n) = d_{n-1}^2 c_{t_n}^4/(2 rho_{n-1}^2 b_{t_n}^2)`,
   `d_n = b_{t_n} - sqrt(c_{t_n}^2 - rho_n^2) b_{t_{n+1}}/c_{t_{n+1}}` — same minimizer, so no retraining.

2. **Sampler / stochasticity knob.** Update `x_{t_n} = a_{t_n} x_T + b_{t_n} x0_hat +
   sqrt(c_{t_n}^2 - rho_n^2)(x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x0_hat)/c_{t_{n+1}} + rho_n eps`.
   `rho_n = 0` is deterministic (DBIM, bridge DDIM); `rho_n = sigma_{t_n} sqrt(1 - SNR_{t_{n+1}}/SNR_{t_n})`
   is Markovian/DDPM-like; intermediate `eta in [0,1]` interpolates (stochasticity aids
   Langevin-style error self-correction).

3. **Booting noise.** The initial deterministic transition is singular (`c_T = 0`). Use the
   Markovian boundary there, which cancels the `1/c_T` and injects one Gaussian `eps`
   (`x = bridge_sample(x0_hat, x_T, ts[0], eps)`).
   This `eps` is the diversity latent (one masked image has many completions).

4. **Clean ODE (rho=0).** `d(x_t/c_t) = x_T d(a_t/c_t) + x_theta d(b_t/c_t)`, which equals the
   DDBM probability-flow ODE (verified by matching `c'/c = f + g^2/sigma^2 - g^2/2c^2`,
   `a' - a c'/c = g^2 a/2c^2`, `b' - b c'/c = -g^2 b/2c^2`).

5. **Exponential integrator + high order.** The ODE is semi-linear; variation-of-constants
   cancels the linear term (the factor from `t` to `s < t` is `c_s/c_t`). With `lambda_t = log(b_t/c_t) =
   (1/2) log(SNR_t - SNR_T)`, the exact solution from `t` to `s < t` is
   ```
   x_s = (c_s/c_t) x_t + (a_s - (c_s/c_t) a_t) x_T + c_s int_{lambda_t}^{lambda_s} e^lambda x_theta dlambda.
   ```
   Taylor-expand `x_theta` in `lambda` about the current node, integrate by parts (`h = lambda_s - lambda_t > 0`):
   ```
   int e^lambda dlambda                       = e^{lambda_s}(1 - e^{-h})
   int (lambda-lambda_t)   e^lambda dlambda    = e^{lambda_s}(h - 1 + e^{-h})
   int (lambda-lambda_t)^2/2 e^lambda dlambda  = e^{lambda_s}(h^2/2 - h + 1 - e^{-h})
   ```
   (the φ-functions of exponential integrators). Estimate the `lambda`-derivatives by **multistep**
   (Adams-Bashforth) finite differences of **past** predictor outputs (one new call per step, so a
   budget of `N` calls buys `M = N` steps, vs. `N/k` for single-step). With one previous node `u`
   (`h_1 = lambda_t - lambda_u`): `x_t^{(1)} = (x_hat_t - x_hat_u)/h_1`. With two previous nodes
   (`h_1 = lambda_t - lambda_{u_1}`, `h_2 = lambda_{u_1} - lambda_{u_2}`, `D_1 = (x_hat_t - x_hat_{u_1})/h_1`,
   `D_2 = (x_hat_{u_1} - x_hat_{u_2})/h_2`):
   ```
   x_t^{(1)} = (D_1 (2 h_1 + h_2) - D_2 h_1)/(h_1 + h_2),   x_t^{(2)} = 2(D_1 - D_2)/(h_1 + h_2).
   ```

## Algorithm

```
Input: condition x_T, decreasing runtime schedule ts = [near T, ..., near 0], predictor x_theta, order o in {2,3}, booting noise eps.
x0_hat <- x_theta(x_T, T, x_T)
x <- a_{ts[0]} x_T + b_{ts[0]} x0_hat + c_{ts[0]} eps      # booting sample, injects diversity latent
for i = (current node s, target node t) marching downward:
    x0_hat <- x_theta(x, s, x_T)
    if first loop step or (lower_order_final and last step):  # order 1: DBIM rho=0 Euler
        x <- (c_t/c_s) x + (b_t - (c_t/c_s) b_s) x0_hat + (a_t - (c_t/c_s) a_s) x_T
    else:                                                  # order 2 or 3 multistep
        h = lambda_t - lambda_s;  lambda_* = log(b_*/c_*)
        I = e^{lambda_t}[(1-e^{-h}) x0_hat + (h-1+e^{-h}) x_t^{(1)} (+ (h^2/2-h+1-e^{-h}) x_t^{(2)})]
        x <- (c_t/c_s) x + (a_t - (c_t/c_s) a_s) x_T + c_t I
    push x0_hat into the past-output buffer; drop the oldest
return x
```

## Working code

Mirrors the canonical `sample_dbim_high_order`, filling the bridge sampler slot. `ts` is
decreasing (`t_max -> 0`); the per-step current node is `s = ts[i]`, target `t = ts[i+1]`;
`denoiser` is the metered data predictor; `get_abc` and `bridge_sample` are the existing bridge
primitives.

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from ddbm.random_util import BatchedSeedGenerator


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


@torch.no_grad()
def sample_dbim_high_order(
    denoiser,
    diffusion,
    x,
    ts,                       # decreasing schedule t_max -> 0
    mask=None,
    order=2,                  # 2 or 3
    lower_order_final=True,
    seed=None,
    **kwargs,
):
    if order not in [2, 3]:
        raise NotImplementedError("Only order 2 or 3 supported")
    x_T = x
    path, pred_x0, nfe = [], [], 0
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    # Booting step: deterministic step 0 is singular (c_T = 0); the Markovian boundary
    # rho_{N-1}=c_{t_{N-1}} cancels it and injects a Gaussian (the diversity latent).
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)   # a x_T + b x0 + c eps
    path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    # Multistep buffers: last (order-1) predictor outputs and their times.
    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5                              # avoid lambda(T) = -inf at the pinned end
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]                              # current node (larger time)
        t = ts[i + 1]                          # target node  (smaller time)

        # ---- First order: first loop transition after boot, or final step ----
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            tmp = c_t / c_s
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            x = tmp * x + (b_t - tmp * b_s) * x0_hat + (a_t - tmp * a_s) * x_T   # DBIM (rho=0) Euler

        # ---- Second order multistep ----
        elif order == 2 or i == 1:
            a_u, b_u, c_u = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = torch.log(b_u / c_u), torch.log(b_s / c_s), torch.log(b_t / c_t)
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h = lambda_t - lambda_s            # step in lambda toward target, > 0
            h2 = lambda_s - lambda_u           # spacing to previous node
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        # ---- Third order multistep ----
        elif order == 3:
            a_u1, b_u1, c_u1 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_u2, b_u2, c_u2 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-2] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2), torch.log(b_u1 / c_u1), torch.log(b_s / c_s), torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1          # near spacing
            h2 = lambda_u1 - lambda_u2         # far spacing
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h ** 2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s); u.pop(0)                  # roll buffers forward
        xu_hat.append(x0_hat); xu_hat.pop(0)
        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise
```

## Relation to prior methods

- **DDBM** (Zhou et al. 2023): same trained bridge and PF-ODE; this replaces the generic hybrid
  Heun sampler with a bridge-tailored exponential-integrator solver.
- **DDIM** (Song et al. 2021): the non-Markovian marginal-preserving construction, transplanted
  to a two-endpoint bridge; DBIM (`rho=0`) is the bridge DDIM, and in the small-`t` regime
  (`a_t ~ 0`, `b_t ~ alpha_t`, `c_t ~ sigma_t`) the DBIM step approximately recovers DDIM.
- **DPM-Solver / DPM-Solver++** (Lu et al. 2022a,b): the exponential integrator with log-SNR
  change of variable and Taylor/φ-function high-order expansion; this uses the data-prediction,
  multistep form on the *bridge* log-SNR `lambda = log(b/c)` with the bridge linear factor `c_t/c_s`.
