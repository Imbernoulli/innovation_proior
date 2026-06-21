Generating an image with a diffusion model is the numerical solution of an ODE: starting from pure Gaussian noise we follow a trajectory through decreasing noise levels $\sigma_0 = \sigma_{\max} > \sigma_1 > \dots > \sigma_{N-1} = \sigma_{\min}$ down to clean data, and every solver step calls the learned denoiser $D(x;\sigma)$ once (twice for a two-stage corrector). That network evaluation is the entire cost — everything else is arithmetic on tensors — so the only figure of merit that matters is image quality at a small step budget $N$. What stops me from simply taking few steps is that a numerical solver is an approximation: at each step it drifts off the true trajectory by a local truncation error, those errors accumulate into a global error, and with few steps the accumulated error is large enough to visibly corrupt the output. So $N$ is really set by how much error I can tolerate, and if I can shrink the per-step error I can shrink $N$. The component that decides where the few steps land is the time-step discretization $\{\sigma_i\}$, and that is a free choice the sampler makes — it treats $D$ as a black box and never cares how $D$ was trained. Yet every schedule in use is inherited from the model's training-time forward noising process: a DDPM-style $\beta$ schedule turned into cumulative $\bar\alpha$, a geometric sequence of $\sigma$ from the variance-exploding forward SDE, a cosine on $\bar\alpha$ from improved DDPM. None of these was chosen to make the reverse integration accurate; each is a single fixed shape with no interpretable control for how to distribute the few sampling steps across the noise range. That is the gap: equal spacing in $\sigma$ ($\rho=1$ below) pours a fixed step size into the low-noise region that demands small steps while wasting fine steps where the field is gentle; the geometric and cosine and variance-preserving schedules are likewise handed down from the forward process, not derived from the per-step error of the solve.

I propose the Karras noise schedule: place the noise levels by warping a uniform grid through a power law, choosing the warp purely to minimize the damaging part of the integration error and decoupled from how the denoiser was trained. The first move is to insist that the discretization $\{\sigma_i\}$, the noise schedule $\sigma(t)$, the scaling $s(t)$, and the trained $D$ are independent dials, which frees me to pick everything for sampling accuracy. Before choosing any spacing I straighten the trajectory, because the local error scales with the curvature of $dx/dt$ and a straighter trajectory is free error reduction. Setting $\sigma(t) = t$ and $s(t) = 1$ collapses the ODE to $dx/dt = (x - D(x;t))/t$, and a single Euler step straight to $t=0$ gives $x + (0-t)(x-D)/t = D(x;t)$ — the denoiser output itself. So the tangent of the solution trajectory everywhere points at the current denoised estimate, and that estimate changes only slowly with noise level (near the dataset mean at high $\sigma$, near the input at low $\sigma$), so the tangent barely rotates and the trajectory is nearly straight except in a narrow middle band. This also makes $\sigma$ and $t$ interchangeable, which simplifies everything downstream. Where the error lives I read off a direct measurement on a pretrained variance-exploding model: estimate the true step endpoint by two hundred tiny Euler sub-steps and compare to one big step. The single-step RMSE is large at low noise (around $0.56$ for $\sigma \le 0.5$ with even $\sigma$-spacing of $1.25$) and small at high noise, and the spread across random samples is tiny — the error is essentially a function of the noise level, not of the particular image. That second fact means one deterministic schedule suffices; the first means the steps must shrink as $\sigma$ decreases.

To get a controllable family I warp a uniform grid: pick a monotone unbounded $w$, sample its argument uniformly, and set $\sigma_{i<N} = w(Ai + B)$ with $A,B$ fixing the endpoints. The simplest one-parameter monotone warp is the power law $w(z) = z^\rho$. Working in the warped coordinate $z = \sigma^{1/\rho}$, the interpolation is linear there and the endpoints fall out automatically, giving the closed form

$$\sigma_{i<N} = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right) \right)^{\rho}, \qquad \sigma_N = 0.$$

At $i=0$ this is $(\sigma_{\max}^{1/\rho})^\rho = \sigma_{\max}$ and at $i=N-1$ it is $\sigma_{\min}$, so the constants are pinned by phrasing the interpolation in $z$. The single knob $\rho$ redistributes the steps: at $\rho=1$ it is uniform spacing in $\sigma$ (the baseline I already argued is wrong); as $\rho$ grows the map $z\mapsto z^\rho$ stretches the high end and compresses the low end, so equal steps in $z$ become long steps near $\sigma_{\max}$ and short steps near $\sigma_{\min}$, exactly the reallocation toward low noise I need. Taking $\rho\to\infty$, $x^{1/\rho}\approx 1 + (1/\rho)\log x$ gives $\rho\log z_i \to \log\sigma_{\max} + \frac{i}{N-1}(\log\sigma_{\min} - \log\sigma_{\max})$, which is uniform spacing in $\log\sigma$ — the geometric variance-exploding schedule. So this family is a parametric bridge containing the uniform-in-$\sigma$ baseline at one end and the geometric baseline at the other, with a continuous knob between; the right generalization should contain the things it beats as special cases, and this one does.

What makes the method work is choosing $\rho$ for perception rather than for numerical balance. If I optimize the objective I set out with — equalize the local truncation error across noise levels so no single step is the bottleneck, minimizing $\max_i \|\tau_i\|$ — then with Euler the crossover where the RMSE curve flattens is around $\rho=2$, but the error there is still a few hundredths, the signal that Euler alone is not enough. Bringing in the solver knob, Heun's method (the explicit trapezoidal rule, a two-stage second-order Runge–Kutta scheme) takes the Euler step as a predictor, re-evaluates the right-hand side at the predicted endpoint, and averages the two slopes; this drops the local error from $O(h^2)$ to $O(h^3)$ at one extra network evaluation per step, a good trade for an expensive denoiser. With Heun the error curve is almost flat across the range at $\rho\approx 3$, with RMSE pinned in a tight band of roughly $0.003$ to $0.0045$ — so $\rho=3$ is very nearly the truncation-error-optimal schedule, and if I cared about raw RGB-space accuracy (running the ODE forward to encode and back to reconstruct), I would stop there. But generation is judged by FID, not by fidelity to one trajectory, and different noise levels do not contribute equally to FID. At very high $\sigma$ the sample is indistinguishable from pure Gaussian noise — the exact value of $\sigma_{\max}$ is somewhat arbitrary — so error introduced there is absorbed into which noise sample I started from. At low $\sigma$ the denoiser is committing to actual content and an error shows up as a visible artifact. Accuracy is therefore perceptually cheap up high and decisive down low, so equalizing the numerical error leaves FID on the table; I should deliberately unbalance it, spending coarser steps up high to afford finer steps down low. That means pushing $\rho$ past $3$, and indeed FID keeps improving beyond it until the high-$\sigma$ steps get so coarse the early trajectory falls apart. Across all the models and step budgets there is a broad, robust sweet spot, and $\rho=7$ sits comfortably in it, so I fix $\rho=7$ rather than treating it as a per-model knob: $\rho=3$ answers "balance the integration error," $\rho=7$ answers "make the best images," and they differ exactly because high-noise accuracy is perceptually cheap.

A few details close the construction. Among the RK2 family $x_{i+1} = x_i + h[(1 - \tfrac{1}{2a})d_i + \tfrac{1}{2a} f(x_i + a h d_i; t_i + a h)]$, I take $a=1$ (plain Heun, second slope at the full endpoint $t_{i+1}$): it is essentially optimal experimentally, and uniquely among RK2 variants it lands the extra evaluation exactly on a schedule node, so models trained at a discrete set of noise levels stay usable, which the midpoint ($a=\tfrac12$) and Ralston ($a=\tfrac23$) variants cannot promise. Stepping all the way to $\sigma=0$ would divide by zero in the corrector (the right-hand side has $\sigma$ in the denominator), so on the final hop to clean data I fall back to a plain Euler step. The endpoints are not arbitrary: $\sigma_{\min}=0.002$ is the lowest resolved level — below it discerning the noise is hard and pointless — with the single final Euler step taking $\sigma_{\min}\to 0$; $\sigma_{\max}=80$ is enormous compared to the data scale $\sigma_\mathrm{data}\approx 0.5$, so the start is effectively pure noise, the very same fact that makes high-$\sigma$ error cheap. The count is $N$ scheduled nonzero levels plus the appended zero, giving $N+1$ nodes and $N$ solver intervals; the ramp runs $0$ to $1$ over the nonzero levels with `linspace(0, 1, N)`, and using $N+1$ there would be an off-by-one that quietly folds the terminal node into the warped interpolation instead of appending it as the explicit clean-data zero. The schedule is strictly decreasing by construction: $z_i$ is linear-decreasing in $i$ and $z\mapsto z^\rho$ is increasing for $z>0$, and appending $0$ keeps it decreasing.

```python
import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Noise-level schedule: power-law warp of a uniform grid.

    n solver steps -> n+1 nodes (sigma_0 = sigma_max ... sigma_{n-1} = sigma_min, sigma_n = 0).
    """
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)                       # sigma_min^{1/rho}
    max_inv_rho = sigma_max ** (1 / rho)                       # sigma_max^{1/rho}
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)                      # append terminal sigma_n = 0
```

```python
@torch.no_grad()
def sample_heun(denoiser, x, sigmas):
    """x: initial noise scaled to sigma_max (= randn(...) * sigmas[0])."""
    for i in range(len(sigmas) - 1):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        d = (x - denoiser(x, sigma_cur)) / sigma_cur                 # dx/dt at sigma_cur
        x_pred = x + (sigma_next - sigma_cur) * d                    # Euler predictor
        if sigma_next == 0:
            x = x_pred                                               # final hop: Euler only
        else:
            d_prime = (x_pred - denoiser(x_pred, sigma_next)) / sigma_next
            x = x + (sigma_next - sigma_cur) * (0.5 * d + 0.5 * d_prime)   # trapezoidal correction
    return x
```

```python
import numpy as np
import torch


@torch.no_grad()
def edm_sampler(net, latents, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, randn_like=torch.randn_like):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time-step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho)
               + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # Optional churn (stochastic sampling); S_churn=0 gives deterministic Heun.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd-order correction (skip on the last step into sigma=0).
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next
```
