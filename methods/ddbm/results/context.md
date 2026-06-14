# Context: principled translation between two arbitrary image distributions with score-based processes (circa 2022-2023)

## Research question

Diffusion models generate images by learning to reverse a stochastic process that turns data into
Gaussian noise: train on samples, then start from noise and integrate a reverse-time SDE (or its
deterministic probability-flow ODE) back to data. This is now the dominant recipe for high-fidelity
image synthesis. But an enormous class of practically important problems is not "noise to data" — it is
"data to data": image-to-image translation (sketch to photo, edges to handbags, day to night), super-
resolution, deblurring, inpainting/restoration. In all of these the *input* is itself a structured image
drawn from some source distribution, not a sample of white noise. The precise goal is a generative
process that directly transports between two arbitrary, *paired* distributions — given a training set of
pairs `(x_0, x_T) ~ p_data(x, y)` (clean target, observed source), learn to sample `x_0 ~ p(x_0 | x_T)`
by running a learned process from the source endpoint to the target endpoint.

A solution has to clear several bars at once. (1) It must connect two *arbitrary* endpoints, not force one
of them to be Gaussian noise. (2) It must remain *tractable to train*: there should be a closed-form way
to sample intermediate states `x_t` from the endpoints, and a closed-form regression target, so training is
one cheap forward pass per step rather than path simulation. (3) It should *reuse the machinery that made
diffusion models work* — the input/output preconditioning, the noise schedules, the high-order and fast
samplers — rather than starting from scratch. (4) It should generate *sharp, diverse* targets, not a
single averaged reconstruction. The existing tools below each meet some of these and miss others.

## Background

By this time the score-based / diffusion framework is mature. A diffusion is a forward SDE
`dx_t = f(x_t, t) dt + g(t) dw_t` that carries data `x_0 ~ p_data` to a prior `x_T ~ p_prior`; the
remarkable Anderson (1982) result is that its time reversal is again a diffusion,
`dx_t = [f - g^2 ∇ log p_t(x_t)] dt + g dw_bar`, with an equivalent deterministic *probability-flow ODE*
`dx_t = [f - ½ g^2 ∇ log p_t(x_t)] dt` that shares the same marginals (Song et al. 2020). The only unknown
is the score `∇ log p_t(x_t)`, learned by denoising score matching: because the transition kernel
`p(x_t | x_0)` is chosen Gaussian (`x_t = α_t x_0 + σ_t ε`), the conditional score is closed form, and
regressing a network onto it recovers the marginal score. Two standard instantiations: VE (variance
exploding, `f = 0`, `σ_t` grows) and VP (variance preserving). The signal-to-noise ratio
`SNR_t = α_t^2 / σ_t^2` is the natural clock. Sampling integrates the reverse SDE or ODE numerically.

The structural problem is that this whole framework is welded to a *Gaussian* prior. The forward SDE has
no mechanism to drive the process to a *specific data point*; it only knows how to add noise until the
distribution forgets where it started. So for translation, the field has resorted to workarounds, each
unprincipled in a different way. One conditions the denoiser on the source image and hopes the network
learns the mapping; another (SDEdit) partially noises the source image and denoises with an
*unconditional* model, trading off how much structure to keep against how much to regenerate; another
(DDIB) trains two separate unconditional models and reverses one ODE into a shared latent then runs the
other forward. These map in only one direction, lose cycle consistency, and have no clean account of what
distribution they actually sample from.

A classical tool from probability theory says how to bend a diffusion to a *fixed* endpoint:
Doob's h-transform. Given the base diffusion, the process
`dx_t = [f(x_t,t) + g^2(t) h(x_t,t,y,T)] dt + g(t) dw_t`, with
`h(x_t,t,y,T) = ∇_{x_t} log p(x_T = y | x_t)` (the gradient of the *backward* transition kernel of the
base diffusion), is guaranteed to arrive at `x_T = y` almost surely. With a Gaussian base kernel, `h` is
closed form. A process pinned at both ends — a fixed `x_0` and a fixed `x_T` — is a *diffusion bridge*,
and it has long been studied in probability (Särkkä & Solin 2019; Doob 1984). The added drift `g^2 h`
"pulls" trajectories toward `y` — structurally the same shape as classifier guidance
(Dhariwal & Nichol 2021), where a gradient term steers the diffusion toward a class. The pieces — a base
diffusion with a tractable Gaussian kernel, the reverse-SDE/ODE duality, denoising score matching, and a
classical drift that pins an endpoint — all exist; what is missing is how to assemble them into a
*learnable, trainable, generative* process between two arbitrary distributions.

Two empirical facts about existing systems frame the design. First, the *prediction-target* observation
behind EDM (Karras et al. 2022): if a network is trained to predict noise `ε` and the clean signal is
reconstructed as `x_0 = x_t − σ F(x_t)`, then at large noise the network's errors are amplified by `σ`,
which destabilizes training; predicting the clean signal directly with a `σ`-dependent skip connection is
far more robust. EDM turns this into a preconditioning `D_θ(x;σ) = c_skip x + c_out F(c_in x; c_noise)`
whose scalings are fixed *from first principles* by requiring the network input and the regression target
to have unit variance, with `c_skip` chosen to minimize `c_out` (and hence the amplification of network
error). Second, the *sampler* observations: EDM shows Heun's second-order (trapezoidal) integrator hits a
given quality with far fewer function evaluations than Euler, that a power-law time discretization
`t_i = (t_max^{1/ρ} + i/(N−1)(t_min^{1/ρ} − t_max^{1/ρ}))^ρ` with `ρ ≈ 7` allocates steps well for
images, and that injecting and then removing a controlled amount of noise each step ("churn") — a
Langevin-like correction in the spirit of Song's predictor-corrector sampler — repairs the errors a
deterministic solver accumulates. Deterministic transport methods also expose a modeling risk for
one-to-many conditional tasks: one fixed source should be allowed to lead to multiple plausible targets,
so a solver that always follows a single path can collapse toward an averaged reconstruction.

## Baselines

These are the prior approaches a method for arbitrary-distribution translation would be measured against.

**Score-based diffusion models (Song et al. 2020; Ho et al. 2020; Sohl-Dickstein et al. 2015).** Forward
SDE `dx = f dt + g dw` to a Gaussian prior; reverse SDE `dx = [f − g^2 ∇log p_t] dt + g dw_bar` and
probability-flow ODE `dx = [f − ½ g^2 ∇log p_t] dt`; score learned by denoising score matching against
the Gaussian kernel `p(x_t|x_0) = N(α_t x_0, σ_t^2 I)`. Superb unconditional and class-conditional
generation. **Gap:** the prior is hard-wired to be Gaussian. There is no native way to make the process
start from, or be conditioned on, a structured source image; conditioning, guidance, or partial-noising
are bolted on after the fact and map only one direction.

**EDM (Karras et al. 2022).** A unified design study of diffusion: pred-signal preconditioning with
unit-variance scalings derived from first principles; loss reweighting `λ(σ) c_out(σ)^2`; the `ρ ≈ 7`
time schedule; Heun's 2nd-order ODE integrator; and a stochastic sampler that adds noise to reach
`t̂_i = t_i + γ_i t_i` (`γ_i = S_churn / N`, clamped) before each deterministic step. State of the art for
generation quality and sampling efficiency. **Gap:** every part of it assumes the *Gaussian-noise* prior
of unconditional generation — the preconditioning is derived for `x_t = x_0 + σε`, and the sampler
integrates from noise to data. None of it is defined for a process pinned to a data endpoint `x_T` with a
nontrivial correlation `cov(x_0, x_T)`.

**Flow matching / Rectified Flow / stochastic interpolants (Lipman et al. 2023; Liu et al. 2022;
Albergo & Vanden-Eijnden 2023; Tong et al. 2023).** Learn a deterministic ODE whose velocity field
matches a prescribed interpolation between two distributions, classically the straight line `x_T − x_0`;
the network regresses the velocity. These *do* connect two arbitrary distributions and avoid Doob's
h-functions. **Gap:** they are deterministic-only and have mostly been applied to generation, not
translation; being outside the diffusion-SDE formalism, they cannot directly reuse diffusion's
stochasticity, preconditioning, or fast SDE samplers, and a deterministic map to a conditional target
tends to the blurry conditional mean.

**Schrödinger-bridge and bridge-matching models (De Bortoli et al. 2021; Liu et al. 2023, I²SB; Shi et
al. 2023; Peluchetti 2023).** Solve an entropic optimal-transport bridge between two distributions, via
iterative proportional fitting (IPF) or iterative Markovian fitting (IMF); I²SB exploits a tractable SB
class for a simulation-free algorithm and shows strong restoration results. **Gap:** the general SB
solvers are expensive iterative approximations that have found limited empirical scale; the tractable
special cases (e.g. Brownian bridge) cover only a narrow slice of the design space and do not inherit the
full diffusion toolbox.

**Discrete-time Brownian-bridge translation (Li et al. 2023, BBDM).** Directly reverses a Brownian bridge
in discrete time for translation. **Gap:** tied to one specific (VE-type) bridge and a discrete-time
formulation, so it cannot freely borrow continuous-time diffusion schedules, preconditioning, or
high-order ODE/SDE samplers.

## Evaluation settings

The natural yardsticks for image-to-image translation and restoration, all pre-existing:

- **Edges→Handbags** (Isola et al. 2017) and **DIODE** depth/normal datasets — standard paired
  translation benchmarks at 64×64 (and higher) resolution.
- **ImageNet** center-region **inpainting / restoration** — a paired source (masked/corrupted) to target
  (clean) workload; mask semantics must be preserved.
- **Day→Night** (Isola et al. 2017), evaluated in a pretrained autoencoder's latent space (factor-8
  downsampling, e.g. 256×256 → 32×32 latent) to test latent-space translation.
- **Unconditional generation** sanity checks on CIFAR-10 (32×32) and FFHQ (64×64), where the source
  distribution is set to Gaussian noise, as a yardstick for any approach that tries to reuse diffusion
  machinery while moving beyond pure noise-to-data generation.
- Metrics: **FID** (Fréchet Inception Distance; lower better) for perceptual quality, plus **LPIPS** and
  **MSE** for translation faithfulness, and **IS** for generation. Protocol: identical network
  architecture across methods per experiment; samplers compared at matched numbers of sampling steps /
  denoiser calls (NFE), since under a tight NFE budget the sampler's per-step accuracy is what is being
  measured. The number of denoiser calls per sample is counted and enforced.

## Code framework

The process plugs into an existing diffusion-style training/sampling harness. The data pipeline yields
*paired* tensors `(x_0, x_T)`; a U-Net-style denoiser network already exists; the EDM-style
preconditioning wrapper, the Gaussian-kernel noise schedule (with its signal `α_t` and noise `σ_t`
functions, drift `f`, diffusion `g^2`), and a numerical sampling loop with an enforced per-sample budget
on denoiser calls are all in place. What is *not* settled is the actual mathematics of the process: how to
sample intermediate states from a pair, what closed-form quantity to regress, what update step to follow
from the source endpoint, and how to allocate deterministic and stochastic moves under a tight call
budget. Those are the empty slots below.

```python
import torch


class NoiseSchedule:
    """Pre-existing Gaussian-kernel schedule: signal alpha_t, noise sigma_t,
    base-diffusion drift f and diffusion g^2 as functions of continuous time t.
    These come from the underlying VP/VE diffusion and are NOT to be altered."""

    def get_alpha_sigma(self, t):        # alpha_t, sigma_t for the base diffusion kernel
        raise NotImplementedError

    def get_f_g2(self, t):               # base-diffusion drift f(t) and diffusion g^2(t)
        raise NotImplementedError

    def intermediate_coeffs(self, t):
        # TODO: the coefficients for the intermediate distribution.
        raise NotImplementedError


class Denoiser:
    """Pre-existing EDM-style preconditioned network wrapper. Given a noisy state and a
    time, returns a prediction of the clean target. The scalings for the paired setting
    are part of what must be worked out."""

    def __init__(self, net, noise_schedule):
        self.net, self.ns = net, noise_schedule

    def predict_x0(self, x_t, t, x_T):
        # TODO: the scaling functions for the paired process.
        raise NotImplementedError


def reverse_step(denoiser, noise_schedule, x, x_T, t_cur, t_next, **opts):
    # The single reverse-time transition: given the current state x at time t_cur and the
    # fixed source endpoint x_T, move to time t_next.
    # TODO: the update rule, including drift and optional noise injection.
    raise NotImplementedError


@torch.no_grad()
def sample(denoiser, noise_schedule, x_T, ts):
    # x_T: the source image (the fixed starting endpoint). ts: decreasing time schedule.
    x = x_T
    path, pred_x0, nfe = [], [], 0
    for i in range(len(ts) - 1):
        # TODO: how to allocate the limited denoiser calls across the trajectory.
        x = reverse_step(denoiser, noise_schedule, x, x_T, ts[i], ts[i + 1])
        path.append(x.detach().cpu())
    return x, path, nfe, pred_x0, ts, None
```

The empty slots are `intermediate_coeffs`, `predict_x0`, `reverse_step`, and the call schedule inside
`sample`.
