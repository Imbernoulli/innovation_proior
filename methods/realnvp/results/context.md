# Research question

We want to learn a probabilistic generative model of high-dimensional, highly
structured continuous data — natural images being the canonical case. The goal is
unsupervised: leverage large pools of unlabeled data to learn the underlying density
p(x), so that we can both *create* novel content and support reconstruction-style
applications (inpainting, denoising, colorization, super-resolution).

The data lives in a very high-dimensional space and is far from factorial: pixels are
heavily correlated in intricate, multi-scale ways. The question is how to build a
model expressive enough to capture that structure, yet still trainable at scale.

# Background

**The maximum-likelihood ideal and tractability.** The cleanest way to fit a
generative model is to maximize the likelihood of the data under it. For latent-variable
models p(x) = ∫ p(x|z) p(z) dz, this requires the marginal, which is generally
intractable; for undirected models it requires the partition function, also intractable.
Almost all of the machinery in the field exists to work around one of these two
intractabilities.

**The change-of-variables route.** There is one setting where maximum likelihood is
*directly* tractable in principle. Suppose the generator g: z → x is a smooth bijection,
with inverse f = g^{-1}: x → z, and z has a simple prior p_Z. Then the density on x
follows from the change-of-variables formula,

  p_X(x) = p_Z(f(x)) · |det( ∂f(x)/∂xᵀ )|,

so that

  log p_X(x) = log p_Z(f(x)) + log |det( ∂f(x)/∂xᵀ )|,

where ∂f/∂xᵀ is the Jacobian of f at x. Sampling is inverse-transform sampling: draw
z ~ p_Z and return x = g(z) = f^{-1}(z). Inference is just z = f(x). No discriminator,
no approximate posterior, no partition function — provided f is bijective and the
Jacobian determinant is computable. This identity is old: it underlies the
maximum-likelihood view of independent components analysis (Bell & Sejnowski 1995;
Hyvärinen et al. 2004), Gaussianization (Chen & Gopinath 2000), and earlier deep density
models (Bengio 1991; Rippel & Adams 2013).

For an arbitrary differentiable map on R^D, computing the Jacobian and its determinant
costs on the order of D^3. This cost is the central computational challenge of the
change-of-variables approach.

**Observations about existing image generators.** Latent-variable models trained against
a fixed-form reconstruction cost (a Gaussian decoder, i.e., an L2 term) systematically
produce blurry samples — the L2 objective rewards capturing low-frequency content more
than high frequency. Autoregressive models, which factor p(x) = ∏_i p(x_i | x_{<i}) and
give exact, flexible likelihoods, sample *sequentially*: generating a D-dimensional image
requires D conditional draws in order.

**Building blocks that already exist.** Deep convolutional residual networks (He et al.
2015, 2016) make very deep image models trainable via skip connections. Batch
normalization (Ioffe & Szegedy 2015) and weight normalization (Salimans & Kingma 2016)
stabilize and speed up training of deep nets. Convolutions encode the 2-D local
correlation prior of images. Multi-scale processing and the factoring of computation
across resolutions are standard (e.g. the VGG design, Simonyan & Zisserman 2014). Deep
supervision — attaching loss signal to intermediate layers (Lee et al. 2014) — is known to
help train deep stacks. The Adam optimizer (Kingma & Ba 2014) is the default for such
models. All of these are available to plug into whatever new map we design.

# Baselines

**Undirected graphical models — RBM / DBM.** Restricted Boltzmann Machines (Smolensky
1986) and Deep Boltzmann Machines (Salakhutdinov & Hinton 2009) define p(x) through an
energy over a bipartite (or layered) structure. Their conditional-independence structure
permits efficient block conditional inference, but the marginal over latents involves an
intractable partition function. Training (contrastive divergence / persistent CD),
evaluation (annealed importance sampling), and sampling (MCMC) rest on approximations.

**Variational autoencoders.** The VAE (Kingma & Welling 2013; Rezende et al. 2014) is a
directed latent-variable model with a generator network mapping a Gaussian z to x, trained
jointly with an amortized approximate inference network q(z|x) by maximizing a variational
lower bound (ELBO) on log p(x), using the reparameterization trick (Williams 1992) to
backpropagate through the sampling. *Math:* maximize E_{q(z|x)}[log p(x|z)] − KL(q(z|x) ‖
p(z)) ≤ log p(x). Ancestral sampling is exact and parallel, and inference is amortized and
fast.

**Autoregressive models.** Fully-visible models — from logistic-autoregressive Bayes nets
and NADE/MADE (Frey 1998; Bengio & Bengio 1999; Larochelle & Murray 2011; Germain et al.
2015) to PixelRNN/PixelCNN (van den Oord et al. 2016) — write p(x) = ∏_i p(x_i | x_{<i})
under a fixed ordering. *Math:* the joint is the product of learned conditionals; the
log-likelihood is exact and a sum of per-dimension terms. They are extremely flexible and
give state-of-the-art likelihoods on images.

**Generative adversarial networks.** GANs (Goodfellow et al. 2014; Denton et al. 2015;
Radford et al. 2015) train an arbitrary differentiable generator g: z → x by pitting it
against a discriminator that distinguishes samples from data, bypassing likelihood
entirely. *Math:* a minimax game; the discriminator supplies the generator's training
signal. They produce sharp, realistic samples.

**The bijective-generator family — NICE.** The most direct prior attempt to make the
change-of-variables route practical is the additive-coupling construction (Dinh et al.
2014). It trains a bijection f: x → z by exactly maximizing log p_X(x) = log p_H(f(x)) +
log|det ∂f/∂x|. Its core trick is the **additive coupling layer**: split the input into two
parts (I_1, I_2) and set

  y_{I_1} = x_{I_1},   y_{I_2} = x_{I_2} + m(x_{I_1}),

where m is an arbitrary neural net. The inverse is immediate, x_{I_2} = y_{I_2} − m(y_{I_1}),
and requires no inverse of m. The Jacobian is lower-triangular with an *identity* diagonal,

  ∂y/∂x = [[ I, 0 ], [ ∂y_{I_2}/∂x_{I_1}, I ]],

so its determinant is exactly 1. Stacking these with alternating partitions yields a deep,
freely-parameterized, exactly-invertible map with a trivially tractable Jacobian
determinant, and a factorial prior p_H (logistic or Gaussian). The whole map is
**volume-preserving** because every coupling has unit-determinant Jacobian; NICE
includes a single final diagonal scaling layer S contributing Σ_i log|S_ii| to the
log-likelihood. It is fully-connected and operates on flat vectors, without
convolutional structure or multi-scale hierarchy.

# Evaluation settings

The natural yardstick is density estimation on natural-image datasets, reported as
**bits per dimension** — the negative log-likelihood per pixel-channel under the model,
in base 2, after accounting for the data preprocessing. Because pixels are discrete
(k = 256 levels per channel), a continuous-density model is compared fairly only after
**dequantization** (adding uniform noise so the discrete data becomes continuous). The
log-likelihood used for comparison subtracts the constant D·log k for those discrete
levels; equivalently, the negative log-likelihood adds D·log k before division by
D·log 2. Standard image corpora of the time are the benchmarks:
CIFAR-10 (Krizhevsky 2009); downsampled ImageNet at 32×32 and 64×64 (Russakovsky et al.
2015; van den Oord et al. 2016); LSUN bedroom/tower/church-outdoor (Yu et al. 2015), with
the smallest side downsampled to 96 pixels and 64×64 random crops; and CelebA (Liu et al.
2015), with an approximately central 148×148 crop resized to 64×64. Data augmentation with
horizontal flips is standard. Beyond the likelihood number, the qualitative protocol is to
inspect samples drawn from the model, to perform inference followed by reconstruction, and
to traverse the latent space (e.g. interpolating between encoded examples) to probe whether
the latent representation is semantically organized.

# Code framework

The ordinary PyTorch primitives are enough for the harness: tensor modules,
convolutional residual stacks, normalization layers, tensor reshaping, and Adam. The open
slot is the concrete data-to-latent map and the exact objective that scores it.

```python
import torch
import torch.nn as nn

class ConvResidualStack(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_blocks):
        super().__init__()
        # standard convolutions, residual blocks, ReLU, normalization
        pass

    def forward(self, x):
        pass

def make_optimizer(params):
    return torch.optim.Adam(params, lr=1e-3)

class InvertibleMap(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x, log_det, reverse=False):
        # TODO: implement a bijection whose Jacobian log-determinant is tractable.
        pass

class DensityModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = InvertibleMap(*args, **kwargs)

    def forward(self, x, log_det=None, reverse=False):
        # TODO: map observations to latent variables and back.
        return self.transform(x, log_det, reverse=reverse)

def prepare_observation(x):
    # TODO: turn bounded discrete observations into continuous tensors suitable
    # for a density model, while tracking the transform's log-determinant.
    pass

class ExactDensityLoss(nn.Module):
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, z, total_log_det):
        # TODO: combine a simple latent prior with the accumulated log-determinant
        # and the discrete-observation correction.
        pass
```
