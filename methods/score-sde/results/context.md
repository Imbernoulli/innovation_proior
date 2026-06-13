## Research question

Two of the strongest generative models of the moment share one recipe: corrupt training data with
slowly increasing noise, then learn to undo the corruption. Both, however, use a hand-chosen
*finite ladder* of discrete noise scales, each with its own bespoke training loss and its own
bespoke sampling rule, and there is no shared theory connecting them. The precise problem is:

- Is there a *principled, parameter-free* way to bridge a complex data distribution to a simple
  known prior — one that does not require choosing the number and spacing of noise scales by hand,
  and that exposes whatever common structure the existing recipes are hiding?
- The two existing samplers (annealed Langevin dynamics; an ancestral reverse Markov chain) look
  unrelated. Are they secretly the same thing, and can a better sampler subsume both?
- Score-based models do not give an exact likelihood, unlike normalizing flows. Can the same model
  be made to compute one?
- Can a *single* trained model perform conditional tasks — class-conditional generation, inpainting,
  colorization — without retraining a new conditional model for each task?

A solution would have to estimate something learnable from data alone, drive a generation process
that provably inverts the corruption, and do all of the above within one framework rather than four.

## Background

**Modeling the gradient of log-density (the score), not the density.** An energy-based model writes
p_θ(x) = e^{-f_θ(x)} / Z_θ, and the normalizing constant Z_θ = ∫ e^{-f_θ} is intractable, which is
why such models are hard to train by maximum likelihood and force restrictive architectures. The
*score* s(x) := ∇_x log p(x) sidesteps this entirely: ∇_x log p_θ = -∇_x f_θ(x) - ∇_x log Z_θ, and
the second term is zero because Z_θ does not depend on x. So a network that outputs a score is free
of the normalizability constraint. Given a score model one samples by Langevin dynamics,
x ← x + ε s(x) + sqrt(2ε) z, which converges to p as ε→0.

**Why a single noise scale fails, and why many scales help.** Fitting s_θ to ∇log p_data by the
Fisher divergence E_{p_data(x)} ||s_θ(x) - ∇log p_data(x)||² weights the error by p_data(x). In
high dimensions real data concentrates near a low-dimensional manifold, so almost everywhere p_data
is tiny and the learned score there is unconstrained and wrong. Langevin chains initialized in those
empty regions get inaccurate gradients and wander; they also mix between separated modes extremely
slowly. Perturbing the data with Gaussian noise of standard deviation σ fixes both problems at once:
it spreads mass into the empty regions (so the score is well-defined and learnable there) and smooths
the landscape (so chains mix). Large σ helps coverage and mixing; small σ keeps the perturbed
distribution close to the true data. Using *many* scales σ_1 < ... < σ_N and a single
noise-conditioned network s_θ(x, σ) gives the best of both, and one samples by walking down the ladder
from the largest scale to the smallest.

**Denoising score matching (Vincent, 2011) is what makes the score target tractable.** The score of
a *noisy marginal* p_σ(x̃) = ∫ p_data(x) N(x̃; x, σ²I) dx is still intractable. Vincent's identity
removes the obstruction: minimizing E_{p_σ(x̃)} ||s_θ(x̃) - ∇_x̃ log p_σ(x̃)||² is equivalent, up to a
constant independent of θ, to
E_{p_data(x)} E_{p_σ(x̃|x)} ||s_θ(x̃) - ∇_x̃ log p_σ(x̃|x)||². The conditional kernel is Gaussian,
N(x̃; x, σ²I), so its score is simply ∇_x̃ log p_σ(x̃|x) = -(x̃ - x)/σ² = -z/σ for x̃ = x + σz. The
regression target becomes a known, trivial quantity. The original score matching of Hyvärinen (2005)
instead uses integration by parts to turn the objective into one needing the trace of the Jacobian
∇_x s_θ(x), which costs a number of backprops proportional to dimension and is impractical for images;
denoising and sliced (Song et al., 2019) variants are the scalable replacements.

**Time reversal of diffusions (Anderson, 1982).** A diffusion described by the Itô SDE
dx = f(x,t) dt + g(t) dw (with w a standard Wiener process) is, when run backward in time, *also* a
diffusion, governed by
dx = [f(x,t) - g(t)² ∇_x log p_t(x)] dt + g(t) dw̄,
where w̄ is a reverse-time Wiener process and dt is a negative timestep. The reverse drift differs
from the forward drift by a term involving the marginal score ∇_x log p_t(x). The result can be
obtained from the Fokker–Planck equation together with a Bayes/Girsanov argument.

**The Fokker–Planck (Kolmogorov forward) equation.** The marginal density of
an SDE dx = f dt + G dw evolves by
∂_t p_t(x) = -Σ_i ∂_{x_i}[f_i p_t] + ½ Σ_{i,j} ∂²_{x_i x_j}[Σ_k G_{ik} G_{jk} p_t].
The second-order (diffusion) term is what distinguishes this evolution from the continuity (Liouville)
equation ∂_t p_t = -∇·(v p_t) of a deterministic transport. Relating stochastic and deterministic
processes through their shared marginals is studied e.g. in the interacting-particle work of Maoutsa
et al. (2020).

**Closed-form transition kernels for affine drift (Särkkä & Solin, 2019).** When the drift f(x,t) is
affine in x, the transition kernel p_{0t}(x(t)|x(0)) is Gaussian and its mean and covariance satisfy
linear ODEs with closed-form solutions (their Eqs. 5.50/5.51), available without simulating the
process.

**Neural ODEs and exact likelihood (Chen et al., 2018; Grathwohl et al., 2018).** For a deterministic
flow dx = f̃(x,t) dt, the instantaneous change-of-variables formula gives
d log p_t(x(t))/dt = -∇·f̃(x(t),t), so integrating the divergence along a trajectory yields the exact
log-density. The divergence is expensive (O(d) backprops), but the Skilling–Hutchinson estimator
∇·f̃ = E_v[vᵀ (∇f̃) v] with E[v]=0, Cov[v]=I turns it into a single vector-Jacobian product, unbiased
and arbitrarily accurate by averaging.

**A measured discrepancy in sample quality.** Samples from these models carry residual high-frequency
noise imperceptible to the eye but damaging to FID, and the additive-noise (NCSN) family had been
measuring worse on FID than the DDPM family despite comparable models.

## Baselines

**Score matching with Langevin dynamics (SMLD / NCSN), Song & Ermon (2019, 2020).** A geometric
ladder of noise scales σ_min = σ_1 < ... < σ_N = σ_max, with σ_min small enough that
p_{σ_min} ≈ p_data and σ_max large enough that p_{σ_max} ≈ N(0, σ_max²I). A single Noise Conditional
Score Network s_θ(x, σ) is trained with a weighted sum of denoising score matching objectives,

  θ* = argmin_θ Σ_i σ_i² E_{p_data(x)} E_{p_{σ_i}(x̃|x)} ||s_θ(x̃, σ_i) - ∇_x̃ log p_{σ_i}(x̃|x)||²,

with σ_i² chosen because it is proportional to 1/E||∇log p_{σ_i}(x̃|x)||², equalizing the per-scale
loss. Sampling runs M steps of Langevin dynamics
x ← x + ε_i s_θ(x, σ_i) + sqrt(2ε_i) z at each scale, annealing from i=N down to i=1, carrying the
final state of one scale into the next. *Gaps it leaves:* the ladder is a discrete hyperparameter;
sampling needs many Langevin steps per scale; and (as it was usually run) no final denoising step,
hurting FID.

**Denoising diffusion probabilistic models (DDPM), Sohl-Dickstein et al. (2015); Ho et al. (2020).** A
discrete forward Markov chain with scales 0 < β_1, ..., β_N < 1,
p(x_i | x_{i-1}) = N(x_i; sqrt(1-β_i) x_{i-1}, β_i I), so that
p(x_i | x_0) = N(x_i; sqrt(α_i) x_0, (1-α_i) I) with α_i = Π_{j≤i}(1-β_j), and the schedule is set so
x_N ≈ N(0, I). A variational reverse chain
p_θ(x_{i-1}|x_i) = N(x_{i-1}; (x_i + β_i s_θ(x_i,i))/sqrt(1-β_i), β_i I) is trained with a reweighted
ELBO that, rewritten, is again a weighted sum of denoising score matching terms,

  θ* = argmin_θ Σ_i (1-α_i) E_{p_data(x)} E_{p_{α_i}(x̃|x)} ||s_θ(x̃, i) - ∇_x̃ log p_{α_i}(x̃|x)||²,

so the optimum matches the score of the perturbed data. Sampling is ancestral:
x_{i-1} = (x_i + β_i s_θ(x_i, i))/sqrt(1-β_i) + sqrt(β_i) z, from i=N down to 1. *Gaps it leaves:* the
forward chain rescales the signal (sqrt(1-β_i)) rather than just adding noise, so it is not obviously
the same object as NCSN's additive ladder; the sampler is tied to this particular discretization; and
there is no exact likelihood (only an ELBO).

**Where these meet.** Both objectives are weighted sums of denoising score matching, and their
weights, σ_i² and 1-α_i, are each proportional to the inverse expected squared conditional score.
Beyond this, the two recipes present as unrelated: two forward processes, two losses, two samplers.

**Yardstick generative models (not built on the same recipe).** GANs (BigGAN; StyleGAN2-ADA) held the
best FID/Inception scores on CIFAR-10 at the time; normalizing flows (RealNVP, Glow, Residual Flow,
FFJORD, Flow++) and autoregressive/flow hybrids (MintNet) held the best exact likelihoods. These set
the bars a new method would be measured against.

## Evaluation settings

- **Datasets.** CIFAR-10 (32×32) for sample quality and likelihood; CelebA (64×64) and LSUN
  bedroom/church_outdoor (256×256) for higher-resolution sampling; CelebA-HQ (1024×1024) as a
  high-resolution stress test.
- **Metrics.** Fréchet Inception Distance (FID, lower better) and Inception Score (IS, higher better)
  for sample quality, computed on 50k samples; negative log-likelihood in bits/dim on uniformly
  dequantized images for density estimation, compared only against models evaluated the same way.
- **Protocol.** Score networks built on the existing DDPM/NCSN U-Net architecture; Adam optimizer
  with learning-rate warmup and gradient clipping; exponential-moving-average of weights for
  evaluation; 1000 training levels for the discrete baselines; sample quality measured from 50k
  generated images; likelihood reported only when a model supplies a tractable density estimate under
  the same dequantization protocol.

## Code framework

The pieces that already exist: a data pipeline, an Adam optimizer with warmup/clipping, a
noise-conditioned U-Net score network from prior work, denoising score matching, and Langevin
dynamics. What does *not* yet exist is a common process abstraction: the forward corruption is a
discrete ladder, the loss is a per-scale sum, and the sampler is a hand-derived chain.

```python
import torch

# Score network supplied by prior noise-conditioned models.
class ScoreNet(torch.nn.Module):
    """Takes a (perturbed) image and a noise-level / time conditioning input; returns a
    field with the same shape as the image. Architecture is inherited from prior work."""
    def forward(self, x, cond):
        raise NotImplementedError  # provided by existing codebase

# Data and optimizer.
def get_dataloader(): ...
def get_optimizer(params):
    return torch.optim.Adam(params, lr=2e-4, betas=(0.9, 0.999))

# --- TODO: the process abstraction the current code lacks ---
# Today the corruption is a fixed discrete ladder, the loss is a per-scale sum, and the
# sampler is a hand-derived chain. What replaces them goes here.
...

# Training loop skeleton (loss to be supplied above).
def train(model):
    opt = get_optimizer(model.parameters())
    for batch in get_dataloader():
        opt.zero_grad()
        loss = ...  # TODO
        loss.backward()
        opt.step()
```
