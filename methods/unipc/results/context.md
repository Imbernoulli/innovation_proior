# Context: A Unified Predictor-Corrector for Fast Guided Diffusion Sampling

## Research question

Diffusion models generate images by reversing a noising process, running a trained network once per
step down a sequence of noise levels. For interactive text-to-image use the budget is brutally tight —
on the order of 5 to 20 network evaluations — and the images must stay sharp and prompt-faithful, which
in practice means *guided* sampling at a large guidance scale. Fast high-order ODE solvers
(DPM-Solver, DPM-Solver++, DEIS) already cut unconditional sampling to ~10–20 steps, and DPM-Solver++
restored stability under guidance by moving to the data-prediction parameterization. But all of these
are *predictor-only* solvers: each step extrapolates from past (or intermediate) network outputs and
commits, with no within-step refinement. The precise problem here: given an already-trained model (no
retraining), build a training-free sampler that reaches higher *effective* accuracy at the *same*
number of network evaluations as the best existing multistep solver — especially in the extreme
few-step regime (5–10 steps) where every existing method's per-step truncation error dominates — and
that supports *arbitrary order* through one unified update rather than a different hand-derived formula
per order.

## Background

**Forward process, schedule, half-log-SNR.** A diffusion model defines `q(x_t|x_0) = N(alpha_t x_0,
sigma_t^2 I)`, i.e. `x_t = alpha_t x_0 + sigma_t eps`, with the signal-to-noise ratio
`alpha_t^2/sigma_t^2` strictly decreasing in `t` (Kingma et al. 2021). The half-log-SNR
`lambda_t = log(alpha_t/sigma_t)` is therefore strictly monotone and invertible; one step of any
exponential-integrator solver is parameterized by `h = lambda_t - lambda_s` (the change in `lambda`
across the step).

**Two parameterizations.** The same network reads as a noise predictor `eps_theta(x_t,t)` or a
data predictor `x_theta(x_t,t) = (x_t - sigma_t eps_theta)/alpha_t` (Kingma et al. 2021). The
data-prediction form is the one that stays stable under large classifier-free guidance and exposes a
clean-image estimate; the construction here is built on it (`predict_x0=True`).

**Exponential-integrator solution.** The diffusion ODE is semi-linear; variation of constants plus the
change of variable to `lambda` gives, in the data-prediction form, the exact step
`x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta(lambda) d
lambda` (Lu et al. 2022). The exponential-integrator `phi` functions
`phi_1(z) = (e^z - 1)/z`, `phi_{k+1}(z) = (phi_k(z) - phi_k(0))/z`, `phi_k(0) = 1/k!`, arise from
integrating the Taylor monomials of `x_theta` against the exponential weight (Hochbruck & Ostermann
2010). Existing multistep solvers (DPM-Solver++(kM), DEIS) approximate the derivatives of `x_theta`
from past network outputs and weight them by these `phi`s.

**The predictor-only limitation.** Every solver above is a *predictor*: at each step it forms an
estimate `x_t` from already-available information and proceeds, so its order is capped by how many past
points it can extrapolate through, and the leading truncation error is never corrected within the step.
A classical remedy in ODE numerics is a *predictor-corrector* pair (Adams–Moulton on top of
Adams–Bashforth): take a predictor step, evaluate the right-hand side at the predicted point, then
*correct* using that new evaluation — gaining an order without extra structure. The opening that
matters for diffusion: in a multistep loop, the network evaluation taken at the start of step `i+1`
(the predictor's base for step `i+1`) is *exactly* an evaluation at the point step `i` predicted. So
that evaluation can serve double duty — correct step `i` and seed step `i+1` — and a corrector can be
added at **zero extra network cost**.

**Order through one analytical form.** Hand-deriving a separate corrector formula per order is what has
kept correctors out of fast diffusion solvers. The unification here writes both predictor and corrector
as the same update — a base data-prediction step plus a linear combination of finite differences of
past model outputs, with the combination coefficients obtained by *solving a small linear system*
`R rho = b` whose rows are powers of the `lambda`-ratios `r_k = (lambda_{s_k}-lambda_{s_0})/h` and
whose right-hand side is the `phi`-derived sequence divided by a chosen scalar `B(h)`. Choosing the
system to match the Taylor expansion of the exact integral to order `p` makes the method order-`p` for
*any* `p`, with the predictor solving the `(p-1)`-dimensional truncation and the corrector solving the
full `p`-dimensional system (because it has one more usable evaluation).

**The `B(h)` degree of freedom.** The scalar `B(h)` multiplying the high-order correction is free; two
standard choices are `B(h) = h` ("bh1") and `B(h) = e^h - 1` ("bh2"), the latter matching the
exponential weight more closely and preferred at larger step sizes / tighter budgets.

**Guidance stability and schedule.** The same observations that shaped the data-prediction solvers
apply: large guidance scales amplify the model's derivatives (favoring smaller per-step error
constants, which the corrector provides), latent-space models have no `[-1,1]` bound to threshold, and
a power-law (Karras/EDM, `rho=7`) or uniform-`lambda` time grid concentrates steps where truncation
error lives.

## Baselines

**DDIM (Song et al. 2020).** First-order discretization of the diffusion ODE; reliable and stable
under guidance. **Gap:** first order only, no within-step correction.

**DPM-Solver / DPM-Solver++ (Lu et al. 2022).** High-order exponential-integrator solvers, the `++`
family on the data prediction for guidance stability; singlestep (kS) and multistep (kM) variants.
**Gap:** predictor-only — order capped by the extrapolation, leading truncation error uncorrected; each
order is a separately-derived formula.

**DEIS (Zhang & Chen 2022).** Multistep exponential-integrator solver on the noise prediction. **Gap:**
predictor-only; same per-order derivation; noise-face instability under large guidance.

**Karras/EDM schedule (Karras et al. 2022).** The time grid, not a solver: `sigma_i = (sigma_max^{1/rho}
+ (i/(N-1))(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho`, `rho=7`, then `sigma=0`.

## Evaluation settings

Guided image-generation benchmarks at fixed small NFE. Class-conditional ImageNet 256x256 with
classifier guidance; latent text-to-image (Stable Diffusion / SDXL) with classifier-free guidance at
the scales practitioners use (e.g. 7.5). Quality by FID (lower better) against a reference set and CLIP
score (higher better); cost by NFE, judged especially in the 5–20 range. Weights, prompts, guidance
scale, NFE budget, and metric computation are held fixed across solvers; only the per-step update and
its grid vary.

## Code framework

A generic multistep sampling harness: a schedule object knowing `alpha_t, sigma_t, lambda_t`; a wrapper
turning the network into a `(x, sigma) -> x_theta` prediction (combining conditional and unconditional
passes for guidance); a routine laying out the decreasing noise levels; and a loop marching the latent
down, calling the network once per step and keeping a short history of past model outputs and step
sizes. The contribution lives in the per-step update — a predictor that extrapolates the history and a
corrector that refines the previous step using the current evaluation — and in the linear-system
machinery that yields both at arbitrary order.

```python
import torch


class Schedule:
    """Known: the noise schedule and quantities derived from it."""
    def alpha(self, t): ...
    def sigma(self, t): ...
    def lam(self, t): ...           # half log-SNR, strictly decreasing in t
    def inverse_lam(self, lam): ...


def get_noise_levels(n, sigma_min, sigma_max, device):
    """Known: a decreasing sequence of n+1 noise levels down to 0 (spacing TBD)."""
    raise NotImplementedError  # TODO


def wrap_model(net, schedule):
    """Known: (x, sigma) -> data prediction x_theta, combining cond/uncond for guidance."""
    def predict(x, sigma):
        ...
    return predict


class Sampler:
    def __init__(self, predict, schedule):
        self.predict = predict
        self.ns = schedule

    def predictor_step(self, x, sigma, sigma_next, history, order):
        # extrapolate the history to advance the latent (the slot to fill)
        raise NotImplementedError  # TODO

    def corrector_step(self, x, sigma, model_t, history, order):
        # refine the previous step using the current model output (zero extra NFE)
        raise NotImplementedError  # TODO

    @torch.no_grad()
    def sample(self, x, sigmas):
        history = {}
        for i in range(len(sigmas) - 1):
            x = self.predictor_step(x, sigmas[i], sigmas[i + 1], history, order=...)
        return x
```
