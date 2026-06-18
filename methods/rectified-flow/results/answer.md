# Rectified Flow

## Problem

Given empirical samples from two distributions $X_0\sim\pi_0$ and $X_1\sim\pi_1$ on $\mathbb{R}^d$ (e.g. $\pi_0$ Gaussian noise, $\pi_1$ data; or two image domains), learn a transport that turns $\pi_0$ into $\pi_1$. We want it (i) trained by plain regression — no minimax, no likelihood, no SDE machinery; (ii) cheap at inference — ideally **one** network call.

## Key idea

Represent the transport as an ODE $\mathrm{d}Z_t = v(Z_t,t)\,\mathrm{d}t$, $t\in[0,1]$, and make its trajectories follow the **straight line** between paired endpoints as much as possible. The straight line $X_t=(1-t)X_0+tX_1$ has constant velocity $X_1-X_0$ and, if a flow could follow it exactly, one Euler step would be exact. The line itself is non-causal (it needs the endpoint $X_1$) and lines from different pairs cross, so we *causalize* it by least-squares regression of a velocity field onto the line direction.

## Objective

For an arbitrary coupling $(X_0,X_1)$ of $\pi_0,\pi_1$ (typically independent, $(X_0,X_1)\sim\pi_0\times\pi_1$):

$$
\min_v \int_0^1 \mathbb{E}\Big[\big\|(X_1-X_0)-v(X_t,t)\big\|^2\Big]\,\mathrm{d}t,
\qquad X_t=(1-t)X_0+tX_1 .
$$

In practice draw $t\sim\mathrm{Unif}[0,1]$ and minimize $\mathbb{E}\|(X_1-X_0)-v_\theta(tX_1+(1-t)X_0,\,t)\|^2$ by SGD. The pointwise minimizer is

$$
v^X(x,t)=\mathbb{E}[\,X_1-X_0\mid X_t=x\,],
$$

the average line direction passing through $(x,t)$ — which is single-valued and so defines an honest ODE even though the underlying lines cross.

## Properties (the coupling $(Z_0,Z_1)=\mathrm{Rectify}((X_0,X_1))$ from solving the ODE)

- **Marginal preserving.** $\mathrm{Law}(Z_t)=\mathrm{Law}(X_t)$ for all $t$; hence $Z_1\sim\pi_1$ when $Z_0\sim\pi_0$ — $(Z_0,Z_1)$ is a valid coupling. (Both laws solve the same continuity equation $\partial_t\rho_t+\mathrm{div}(v^X_t\rho_t)=0$ with the same initial condition.)
- **Convex transport cost does not increase.** $\mathbb{E}[c(Z_1-Z_0)]\le\mathbb{E}[c(X_1-X_0)]$ for **every** convex $c$ — a Pareto descent over all convex costs, by two applications of Jensen.
- **Straightening by Reflow.** Recoupling on the flow's own output and refitting, $\vec Z^{k+1}=\mathrm{RectFlow}((Z_0^k,Z_1^k))$, drives the straightness $S(\vec Z)=\int_0^1\mathbb{E}\|(Z_1-Z_0)-\dot Z_t\|^2\mathrm{d}t$ to zero. More precisely, $\sum_{k=0}^K(S(\vec Z^{k+1})+V((Z_0^k,Z_1^k)))\le\mathbb E\|X_1-X_0\|^2$, so the best rectification gap among those $K+1$ rounds is at most $\mathbb E\|X_1-X_0\|^2/(K+1)$. A near-straight flow runs accurately with very few (even one) Euler steps.
- **Distillation.** Once near-straight, fit $\hat T(z_0)=z_0+v(z_0,0)$ so $Z_1\approx\hat T(Z_0)$ in a single call; this is exactly the $t=0$ term of the training objective.

## Algorithm

```
Procedure  Z = RectFlow((X0, X1)):
  Train:    minimize  E_t,(X0,X1) || (X1 - X0) - v_theta(t*X1 + (1-t)*X0, t) ||^2 ,  t ~ Unif[0,1]
  Sample:   solve  dZ_t = v_theta(Z_t, t) dt  from Z0 ~ pi_0   (Euler N steps or RK45)
  Return:   the flow Z = {Z_t : t in [0,1]}

Reflow (optional):  Z^{k+1} = RectFlow((Z0^k, Z1^k)),  starting from (X0, X1)
Distill (optional): learn  T_hat  with  Z1^k ~= T_hat(Z0^k) = Z0^k + v(Z0^k, 0)
```

## Code

Velocity-field training: linear interpolation, target $x_1-x_0$, plain L2.

```python
import torch

def rectified_flow_loss(model, x0, x1, eps=1e-3):
    """One step of the rectified-flow regression.
    x0 ~ pi_0 (e.g. Gaussian noise), x1 ~ pi_1 (data); for reflow, (x0,x1) are
    (z0, ODE(z0)) pairs produced by the previous flow."""
    b = x1.shape[0]
    t = torch.rand(b, device=x1.device) * (1.0 - eps) + eps      # t ~ Unif(0,1)
    t_ = t.view(-1, *([1] * (x1.dim() - 1)))                      # broadcast over data dims
    x_t   = t_ * x1 + (1.0 - t_) * x0                            # linear interpolation X_t
    target = x1 - x0                                            # constant line velocity
    v = model(x_t, t * 999)                                     # velocity field v_theta(x_t, t)
    return ((v - target) ** 2).mean()                          # || (X1-X0) - v(X_t,t) ||^2
```

Sampling by forward Euler (a near-straight flow needs only a handful of steps; a perfectly straight one needs `N=1`):

```python
@torch.no_grad()
def euler_sample(model, z0, N=1, eps=1e-3):
    """Integrate dZ_t = v_theta(Z_t, t) dt from Z_0 ~ pi_0 to Z_1."""
    x, dt = z0.clone(), 1.0 / N
    for i in range(N):
        t = torch.ones(z0.shape[0], device=z0.device) * (i / N * (1.0 - eps) + eps)
        x = x + model(x, t * 999) * dt
    return x

@torch.no_grad()
def rk45_sample(model, z0, eps=1e-3):
    """Adaptive black-box ODE solve, for an accurate reference / reflow targets."""
    import numpy as np
    from scipy import integrate
    shape = z0.shape
    def ode_func(t, x_flat):
        x = torch.tensor(x_flat, device=z0.device, dtype=torch.float32).reshape(shape)
        vt = torch.ones(shape[0], device=z0.device) * t
        return model(x, vt * 999).reshape(-1).cpu().numpy()
    sol = integrate.solve_ivp(ode_func, (eps, 1.0), z0.reshape(-1).cpu().numpy(),
                              rtol=1e-5, atol=1e-5, method='RK45')
    return torch.tensor(sol.y[:, -1], dtype=torch.float32).reshape(shape).to(z0.device)
```

Reflow — recouple on the flow's own output and refit to straighten:

```python
def reflow(model_ctor, train_one_flow, pi0_sampler, pi1_data, K=1, n_pairs=4_000_000):
    """K rounds of recoupling; each round retrains a fresh flow on (z0, ODE(z0)) pairs."""
    model = train_one_flow(rectified_flow_loss, pi0_sampler, pi1_data)   # 1-rectified flow
    for k in range(K):
        z0 = pi0_sampler(n_pairs)
        z1 = rk45_sample(model, z0)                                      # deterministic pairing
        model = train_one_flow(rectified_flow_loss, given_pairs=(z0, z1))  # refit on coupling
    return model
```

Distillation to a one-step generator (train only at $t=0$; LPIPS works better than L2 on images):

```python
def distill_one_step_loss(model, z0, z1):
    """One-step map T_hat(z0) = z0 + v(z0, 0); implemented at a small endpoint eps."""
    t = torch.full((z0.shape[0],), 1e-3, device=z0.device)
    v = model(z0, t * 999)
    return ((v - (z1 - z0)) ** 2).mean()        # or LPIPS(z0 + v, z1)
```

The velocity model is a standard time-conditioned U-Net (DDPM++/NCSN++), trained with Adam and EMA; for $\pi_0=\mathcal N(0,I)$ this is image generation, and setting $\pi_0,\pi_1$ to two image domains gives image-to-image translation with the same code.
