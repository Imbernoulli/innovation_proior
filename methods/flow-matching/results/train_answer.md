We want a continuous generative model that carries a simple Gaussian prior to a complex data distribution through a learned time-dependent vector field. The natural formulation is a continuous normalizing flow: a neural network v_t(x; θ) defines an ODE whose solution pushes samples forward from noise at t = 0 to data at t = 1. This is appealing because it is deterministic, invertible, and gives exact likelihoods via the instantaneous change-of-variables formula. The obstacle has always been training. Maximum likelihood requires solving the ODE inside every gradient step, and computing the divergence is expensive in high dimensions unless one uses stochastic trace estimators. That simulation-in-the-loop cost is why continuous flows have not scaled to large images, even though the model class is very expressive. Prescribed-density alternatives avoid the ODE solve but still need to estimate an intractable integral or the divergence of the learned field, and they often bias the gradient. Diffusion and score-based models escape the simulation cost by corrupting data with a fixed simple SDE and regressing against the closed-form conditional score, but that locks the path family to whatever the SDE produces—typically curved trajectories that overshoot and require many solver steps at sampling time.

The right answer is Flow Matching, specifically Conditional Flow Matching (CFM). Instead of trying to maximize likelihood, we regress the network field directly onto a target vector field that generates a chosen probability path from noise to data. The ideal marginal objective would compare v_t(x) to the true generating field u_t(x) of the marginal path p_t(x), but u_t is an intractable integral over the unknown data distribution, so that objective is unusable. CFM solves this by working with per-example conditional paths. We build the marginal path as a mixture of conditional Gaussian paths p_t(x|x_1), each starting at the shared prior N(0, I) and ending at a small Gaussian around a data point x_1. The key insight is that the conditional objective, which regresses v_t on the closed-form conditional field u_t(x|x_1) while sampling x from p_t(x|x_1), has exactly the same gradient as the intractable marginal objective. This is not just a similar minimizer: expanding the squared losses shows that the θ-dependent terms have identical expectations because the marginal density cancels in the cross term. Therefore we can train with a single network evaluation per example, no ODE simulation, no density approximation, and no loss weighting.

The remaining design choice is the conditional path family. For Gaussian conditionals p_t(x|x_1) = N(μ_t(x_1), σ_t(x_1)^2 I), the canonical affine flow ψ_t(x_0) = σ_t(x_1) x_0 + μ_t(x_1) gives a closed-form conditional field u_t(x|x_1) = (σ_t'(x_1)/σ_t(x_1))(x − μ_t(x_1)) + μ_t'(x_1). Different choices of μ_t and σ_t recover different methods. For example, the diffusion schedules fall out as special cases: variance-exploding and variance-preserving paths produce the usual diffusion probability-flow fields, but derived without any SDE or time-reversal reasoning. However, those paths only approach pure noise asymptotically and their trajectories curve in time, which makes the network harder to fit and the sampler slower.

The preferred instantiation uses the optimal-transport linear path μ_t(x_1) = t x_1 and σ_t(x_1) = 1 − (1 − σ_min)t. This satisfies the boundary conditions exactly: at t = 0 we have N(0, I), and at t = 1 we have N(x_1, σ_min^2 I). Plugging into the Gaussian field formula gives u_t(x|x_1) = (x_1 − (1 − σ_min)x) / (1 − (1 − σ_min)t), and along the sampled straight-line trajectory x_t = (1 − (1 − σ_min)t)x_0 + t x_1 the target collapses to the constant x_1 − (1 − σ_min)x_0. These straight constant-speed paths are the Wasserstein-2 displacement interpolant between the endpoint Gaussians, so they are geometrically optimal. The regression target no longer rotates with time, and the ODE solver can integrate them accurately with very few steps, giving both easier training and low-NFE sampling.

```python
import torch


def pad_t_like_x(t, x):
    """Reshape a batch of scalar times so it broadcasts over x."""
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def sample_prior_like(x):
    return torch.randn_like(x)


@torch.no_grad()
def odeint(field, x0, t0, t1, steps=100):
    """Simple Euler integrator; straight OT paths tolerate very few steps."""
    x = x0
    dt = (t1 - t0) / steps
    n = x0.shape[0]
    for i in range(steps):
        t = torch.full((n,), t0 + i * dt, device=x.device, dtype=x.dtype)
        x = x + field(x, t) * dt
    return x


class TrainingPairBuilder:
    """OT Gaussian conditional path for Conditional Flow Matching.

    p_t(x|x1) = N(t*x1, (1 - (1 - sigma_min) t)^2 I)
    psi_t(x0) = (1 - (1 - sigma_min) t) x0 + t x1
    u_t(x_t|x1) = (x1 - (1 - sigma_min) x_t) / (1 - (1 - sigma_min) t)
                = x1 - (1 - sigma_min) x0       (constant in t along the path)
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def compute_mu_t(self, x0, x1, t):
        del x0
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t):
        return 1 - (1 - self.sigma_min) * t

    def sample_xt(self, x0, x1, t):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = pad_t_like_x(self.compute_sigma_t(t), x1)
        return mu_t + sigma_t * x0

    def compute_conditional_flow(self, x0, x1, t, xt):
        del x0
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma_min) * xt) / (1 - (1 - self.sigma_min) * t)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut


def train_step(pair_builder, v_theta, opt, x1):
    """One CFM gradient step: regress v_theta(x_t, t) onto the conditional field."""
    x0 = sample_prior_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    t, xt, ut = pair_builder.sample_location_and_conditional_flow(x0, x1, t)
    loss = ((v_theta(xt, t) - ut) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss


def sample(v_theta, n, shape, steps=100, device="cpu"):
    """Integrate dx/dt = v_theta(x, t) from noise (t=0) to data (t=1)."""
    x = torch.randn(n, *shape, device=device)
    return odeint(lambda x, t: v_theta(x, t), x, 0.0, 1.0, steps=steps)
```
