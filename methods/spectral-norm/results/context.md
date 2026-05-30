# Context

## Research question

Generative adversarial networks pit a generator G, which maps a latent noise vector to a
sample, against a discriminator D, which tries to tell generated samples apart from real
data. They are trained by the alternating min–max game

    min_G max_D V(G, D),  with  V(G, D) = E_{x~q_data}[log D(x)] + E_{x'~p_G}[log(1 - D(x'))].

The generator never sees the data directly; its only learning signal is the gradient that
flows back through D. So the quality and the behaviour of D dictate whether G learns anything
at all. The central difficulty is that D is, in practice, badly behaved during training:

- In high-dimensional spaces the density-ratio estimate that D is implicitly learning is
  inaccurate and unstable, and the generator fails to capture the multimodal structure of the
  target distribution.
- Worse, when the support of the model distribution p_G and the support of the data
  distribution q_data are (almost) disjoint — which is the generic situation when both lie on
  low-dimensional manifolds in a high-dimensional ambient space — there exists a discriminator
  that separates the two perfectly. Once the optimiser finds such a D, its derivative with
  respect to the input is zero almost everywhere, so the gradient passed back to G vanishes and
  training comes to a complete halt.

The goal, then, is a way to **restrict the function class from which D is chosen** so that the
adversarial game stays well-conditioned — D cannot become arbitrarily sharp, its input
gradient cannot blow up or collapse to zero — while the restriction should be cheap, should
add essentially no hyper-parameters, and should not throw away the discriminator's capacity to
use many features at once. A solution must stabilise training across a range of learning rates
and architectures without per-problem tuning.

## Background

**The optimal discriminator and why its gradient is dangerous.** For a fixed generator, the
discriminator that maximises the standard objective is

    D*_G(x) = q_data(x) / (q_data(x) + p_G(x)) = sigmoid(f*(x)),  f*(x) = log q_data(x) − log p_G(x).

Its input gradient is

    ∇_x f*(x) = ∇_x q_data(x) / q_data(x) − ∇_x p_G(x) / p_G(x),

which can be unbounded or even ill-defined. This is the concrete reason the field began to
suspect that the *regularity* of D — specifically its sensitivity to the input — is what needs
to be controlled, rather than D's accuracy per se.

**Lipschitz continuity as the controlled quantity.** A growing line of work argued that the
discriminator should be searched for not over all functions but over the set of K-Lipschitz
functions,

    argmax_{‖f‖_Lip ≤ K} V(G, D),

where ‖f‖_Lip is the smallest M with ‖f(x) − f(x')‖ / ‖x − x'‖ ≤ M for all x, x' (ℓ2 norm).
Bounding the Lipschitz constant bounds how fast D can change, which keeps the density-ratio
statistics finite and keeps a usable gradient flowing to G. Two facts about Lipschitz
constants are load-bearing for everything that follows.
First, for a differentiable map the Lipschitz constant equals the supremum over inputs of the
spectral norm of its Jacobian: ‖g‖_Lip = sup_h σ(∇g(h)), where the **spectral norm** of a
matrix A is

    σ(A) = max_{h≠0} ‖A h‖_2 / ‖h‖_2 = max_{‖h‖_2≤1} ‖A h‖_2,

i.e. the largest singular value of A. Second, the Lipschitz constant is sub-multiplicative
under composition: ‖g_1 ∘ g_2‖_Lip ≤ ‖g_1‖_Lip · ‖g_2‖_Lip.

**Activations.** Common activations — ReLU, leaky ReLU — are exactly 1-Lipschitz, and most
others satisfy a K-Lipschitz bound for a fixed K. So a feed-forward discriminator's Lipschitz
constant is governed by its linear/convolutional layers.

**Power iteration.** The largest singular value of a matrix W, together with its leading left
and right singular vectors u_1, v_1, can be obtained without a full singular-value
decomposition: starting from a random u, the repeated update v ← W^T u / ‖W^T u‖,
u ← W v / ‖W v‖ converges to (u_1, v_1), and u^T W v converges to σ(W). This is standard
numerical linear algebra and is cheap — each step is two matrix–vector products.

**Diagnostic observation about the alternatives.** A recurring empirical finding about the
existing weight-constraint methods (weight clipping, row-wise weight normalization, Frobenius
normalization) is that the trained discriminators end up with weight matrices whose singular
values are concentrated on a few components — the matrices are close to low rank. Inspecting
the squared singular values of the layers of a trained discriminator shows them piling up on
one or two directions for these methods. A near-rank-one weight matrix means the discriminator
is using essentially one feature per layer to separate the distributions, which is exactly the
kind of impoverished discriminator that produces a sloppy generator. This is a fact about the
*existing* methods, observable before any new method is proposed, and it is the clue that the
restriction we want should bound the top of the spectrum without crushing the rest of it.

## Baselines

**Weight clipping (Wasserstein GAN; Arjovsky, Chintala & Bottou 2017).** WGAN replaces the
Jensen–Shannon objective with the Wasserstein-1 distance, whose Kantorovich–Rubinstein dual
requires the critic D to be 1-Lipschitz. The proposed enforcement is brutally simple: after
each gradient step, clip every weight entry into a box, w ← clip(w, −c, c) for a small constant
c (e.g. c = 0.01). This does bound the Lipschitz constant, but it constrains the matrix far
more than intended: to make ‖W x‖ large under a box constraint on entries the optimiser drives
W toward rank one, so the critic collapses onto a few features ("capacity underuse"), and
training is reported to be slow.

**Gradient penalty (WGAN-GP; Gulrajani et al. 2017).** To fix clipping, WGAN-GP drops the box
and instead augments the loss with a soft penalty that pushes the input-gradient norm of D
toward 1 at sampled points:

    λ · E_{x̂~p_x̂}[ (‖∇_x̂ D(x̂)‖_2 − 1)^2 ],   x̂ = ε x + (1 − ε) x̃,  ε ~ U[0,1],

interpolating a real sample x and a generated sample x̃ = G(z); typically λ = 10. This avoids
the rank-collapse pathology and trains strong critics. But the penalty is only evaluated where
samples currently live: it regularises D on the support of the *current* generator
distribution and says nothing about the rest of input space. As G changes, that support drifts,
so the regularisation target moves and the effect is fragile — high learning rates are observed
to destabilise it. It is also expensive: computing ‖∇_x̂ D‖_2 inside the loss requires an extra
forward–backward round (a gradient of a gradient) per update.

**Weight normalization (Salimans & Kingma 2016) and Frobenius normalization.** Weight
normalization rewrites each row of a weight matrix as w̄_i = w_i / ‖w_i‖_2 (the original
formulation also multiplies by a learned scalar γ_i, dropped here to study the pure Lipschitz
constraint). Frobenius normalization instead divides the whole matrix by its Frobenius norm,
W / ‖W‖_F. Both originated as generalisation aids in supervised training and were repurposed
as discriminator regularisers. The hidden cost: row normalization forces
Σ_t σ_t(W̄)^2 = d_o (and Frobenius forces Σ_t σ_t^2 = 1), because
Σ_t σ_t(W̄)^2 = tr(W̄ W̄^T) = Σ_i (w_i/‖w_i‖)(w_i^T/‖w_i‖) = d_o. Under a fixed sum of squared
singular values, ‖W̄ h‖ for a unit h is maximised when one singular value takes the whole
budget (σ_1 = √d_o, the rest 0) — i.e. W̄ is rank one. So the very pressure to keep the
discriminator sensitive drives these methods to low rank, the same capacity-underuse pitfall
as weight clipping.

**Orthonormal regularization (Brock et al. 2016).** Adds ‖W^T W − I‖_F^2 to the objective,
driving the weight toward an orthonormal matrix. This stabilises training but sets *all*
singular values to 1, which erases the spectral information entirely and forces the
discriminator to spend capacity on every dimension, including ones that ought to be discarded.

**Spectral norm regularization (Yoshida & Miyato 2017).** Adds a penalty on σ(W) to the loss
(estimating σ by power iteration). It is data-independent, like L2 regularization — it nudges
the spectral norm down but never sets it to a designated value, and it lives as an extra loss
term rather than as a constraint baked into the weight.

## Evaluation settings

Unsupervised image generation on CIFAR-10 (32×32) and STL-10 (48×48), and conditional
generation on ILSVRC2012 (ImageNet) downsampled to 128×128. Discriminator and generator are
convolutional networks (a standard DCGAN-style CNN and a ResNet-based architecture); leaky
ReLU slopes are 0.1; the generator uses batch normalization, conditional batch normalization
for the class-conditional case. Latent dimension d_z = 128, p(z) = N(0, I). Optimisation by
Adam, swept over several settings of learning rate α, momentum (β_1, β_2), and n_dis (the
number of discriminator updates per generator update), spanning conservative settings used in
prior work and deliberately aggressive high-learning-rate settings. Quality is measured by
inception score and by Fréchet inception distance (FID), the 2-Wasserstein distance between
Gaussian fits to inception-feature statistics of real and generated images; mode collapse is
gauged by intra-class MS-SSIM. The natural points of comparison are the discriminator
constraints above (weight clipping, WGAN-GP, gradient penalty on the standard loss, batch
norm, layer norm, weight norm, orthonormal regularization).

## Code framework

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ---- existing primitives -------------------------------------------------
# Linear / Conv2d layers, leaky ReLU (1-Lipschitz), Adam, the GAN loss --
# all already available. The generator is a standard transposed-conv stack.

class Generator(nn.Module):
    def __init__(self, dim_z=128, ch=512):
        super().__init__()
        # standard DCGAN-style generator: dense -> (deconv -> BN -> ReLU)* -> tanh
        ...
    def forward(self, z):
        ...

# ---- the one open slot ---------------------------------------------------
# We need some transform applied to each linear/conv weight of D that
# constrains how much the layer can stretch its input -- i.e. controls the
# layer's Lipschitz constant. Its exact form is what we have to design.

def constrain_weight(module, name="weight"):
    # TODO: register a transform on `module`'s weight that bounds the
    # TODO: layer's sensitivity (its largest input-stretching factor),
    # TODO: cheaply, recomputed each forward pass.
    raise NotImplementedError

class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        # conv stack + leaky ReLU(0.1), final dense -> scalar.
        # every weight-bearing layer is wrapped by constrain_weight(...)
        ...
    def forward(self, x):
        ...

# ---- training loop (already standard) ------------------------------------
def d_loss(D, G, x_real, z):      # standard adversarial / hinge loss
    ...
def g_loss(D, G, z):
    ...

def train_step(D, G, opt_d, opt_g, x_real, z, n_dis=1):
    for _ in range(n_dis):
        opt_d.zero_grad(); d_loss(D, G, x_real, z).backward(); opt_d.step()
    opt_g.zero_grad(); g_loss(D, G, z).backward(); opt_g.step()
```
