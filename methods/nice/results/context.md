# Research question

We want to learn a probabilistic model of complex, high-dimensional continuous data —
natural images and faces being the motivating case — in a fully unsupervised way. The
guiding view is representational: *a good representation is one in which the distribution
of the data is easy to model*. Concretely, can we find a transformation `h = f(x)` of the
data into a new space such that the transformed distribution **factorizes**,

  p_H(h) = ∏_d p_{H_d}(h_d),

i.e. the latent components h_d become independent? If we can, then modeling p_X reduces to
fitting a transformation into a simple, fixed, factorial prior.

What would make such a model genuinely useful is a single family that delivers all of the
following at once:

1. a **tractable, exact** training objective — the actual log-likelihood, not a variational
   lower bound and not an adversarial surrogate;
2. **easy, unbiased ancestral sampling**;
3. **exact inference** of the latent code for any data point;
4. a **meaningful latent representation** of dimension equal to the data.

The hard part: the data is high-dimensional and far from factorial (pixels are heavily,
intricately correlated), so f must be very expressive; yet the families that give an exact
likelihood are either intractable to train (partition functions) or buy tractability with a
sequential bottleneck, and the families that sample easily give up the exact likelihood or
the exact encoder. The question is whether one transformation family can carry all four
properties simultaneously.

# Background

**The maximum-likelihood ideal and where tractability breaks.** The cleanest way to fit a
generative model is to maximize the likelihood of the data. For latent-variable models
p(x) = ∫ p(x|h) p(h) dh this needs the marginal, generally intractable; for undirected
energy-based models it needs the partition function, also intractable. Most of the field's
machinery exists to work around one of those two intractabilities.

**The change-of-variables route.** There is one setting where exact maximum likelihood costs
nothing extra. Let f: x → h be a smooth bijection with the same dimension on both sides, and
let h carry a fixed prior p_H. Conservation of probability mass under the map gives the
change-of-variables identity,

  p_X(x) = p_H(f(x)) · |det( ∂f(x)/∂x )|,

so that

  log p_X(x) = log p_H(f(x)) + log |det( ∂f(x)/∂x )|,

where ∂f/∂x is the Jacobian of f at x. If the prior factorizes, the first term becomes a sum
Σ_d log p_{H_d}(f_d(x)). With f a bijection, inference is exact and direct (h = f(x)),
sampling is inverse-transform / ancestral (draw h ~ p_H, return x = f^{-1}(h)), and the
objective is the exact log-likelihood — no discriminator, no approximate posterior, no
partition function. The log-determinant term is what stops the model from cheating: a
bijective preprocessing can inflate likelihood arbitrarily just by contracting the data, and
the determinant term exactly counteracts that, rewarding local volume *expansion* at the data
points (the regions worth representing) and penalizing contraction. This identity is old —
it underlies the maximum-likelihood view of independent components analysis (Hyvärinen et al.
2000), Gaussianization (Chen & Gopinath 2000), and earlier attempts to learn deep
density-shaping transforms (Bengio 1991; Rippel & Adams 2013).

**Why the change-of-variables route had not scaled.** The obstruction is the determinant. For
an arbitrary differentiable f on R^D, forming the D×D Jacobian and its determinant is an
O(D^3) operation, numerically ill-conditioned, and must be redone every gradient step;
requiring f to also be invertible on top of that makes a naive implementation both expensive
and fragile. A clean, exact objective is strangled by one term, which is why large-scale
density models built directly on the change-of-variables formula had not entered use.

**Some matrix structures have cheap determinants.** The determinant of a triangular matrix is
just the product of its diagonal entries — O(D), no factorization. Autoregressive density models
exploit this: conditioning each coordinate on the earlier ones makes the implied map's Jacobian
strictly triangular, which is what makes their exact likelihood tractable (at the cost of
sequential sampling). It is also a standard fact that a composition f = f_L ∘ … ∘ f_1 has a
Jacobian determinant equal to the product of the layers' determinants, and inverts by composing
the layer inverses in reverse.

**Dequantization.** Image pixels are discrete (256 levels per channel). A continuous density
fit directly to discrete points can place arbitrarily tall spikes on them and drive likelihood
to infinity. Adding uniform noise of one quantization bin and rescaling — *dequantization*
(Uria et al. 2013) — turns the discrete data into continuous data and imposes a proper upper
bound on the expected log-likelihood, making continuous-density numbers meaningful and
comparable.

**Building blocks that already exist.** Deep rectified (ReLU) multilayer perceptrons are
standard flexible function approximators. The Adam optimizer (Kingma & Ba 2014) is the default
first-order method. Whitening / ZCA and PCA are standard linear preprocessing tools, and the
eigenspectrum of PCA is the usual lens for "how much variation lives in each direction."
Theano/Pylearn2-style autodiff frameworks make it cheap to differentiate composed maps. All of
these are available to plug into whatever transformation we design.

# Baselines

**Undirected graphical models — RBM / DBM.** Deep Boltzmann Machines (Salakhutdinov & Hinton
2009) define p(x) through an energy over layered binary units. Their conditional-independence
structure permits efficient approximate inference and learning, which made them a major
research subject. *Math/algorithm:* an energy model whose normalizer is the partition function.
*Gaps:* training and sampling both rely on MCMC, which mixes slowly when the target has sharp
modes and yields correlated samples; the log-likelihood is intractable, and the best estimator,
annealed importance sampling (Salakhutdinov & Murray 2008), can be overly optimistic (Grosse et
al. 2013). No exact likelihood, no exact sampling, no fast exact inference.

**Variational autoencoders / SGVB.** The VAE (Kingma & Welling 2014; Rezende et al. 2014;
related: Mnih & Gregor 2014; Gregor et al. 2014) is a directed latent-variable model with a
generator p(x|h) over a Gaussian prior, trained jointly with an amortized *stochastic* encoder
q(h|x) by maximizing a variational lower bound on log p(x), using the reparameterization trick
to backpropagate through sampling. *Math:* maximize E_{q(h|x)}[log p(x|h)] − KL(q(h|x)‖p(h)) ≤
log p(x). Ancestral sampling is exact and fast and inference is amortized. *Gaps:* the encoder
q(h|x) is *stochastic* and only a variational *approximation* to the true posterior, so noise
is injected into the autoencoder loop; the objective is a *bound*, not the likelihood, and a
suboptimal bound can leave unstructured noise in the generative process; and an imperfect
decoder p(x|h) forces a reconstruction term and the modeling of low-level noise at the visible
layer.

**Autoregressive density models — NADE / neural autoregressive nets.** Fully-visible models
(Bengio & Bengio 1999; NADE, Larochelle & Murray 2011) write p(x) = ∏_i p(x_i | x_{<i}) under a
fixed ordering. *Math:* the joint is a product of learned conditionals; the implied map's
adjacency / Jacobian is strictly triangular, so the exact log-likelihood is a sum of
per-dimension terms. They are flexible and exact. *Gaps:* sampling is inherently sequential — D
ordered conditional draws — hence non-parallelizable and slow on high-dimensional data like
images; there is no natural latent representation; and the chosen ordering matters.

**Generative adversarial networks.** GANs (Goodfellow et al. 2014) train an arbitrary
differentiable generator turning a simple factorial noise distribution into the data
distribution, supervised by a discriminator that tries to separate samples from data. *Math:* a
minimax game; the classifier supplies the generator's training signal. They sidestep inference
and the likelihood entirely and produce sharp samples. *Gaps:* no tractable likelihood (so no
density evaluation and no diversity measurement), and no encoder mapping x back to the latent.

**Earlier learned-transform density models.** Maximum-likelihood ICA (Hyvärinen et al. 2000)
learns an *orthogonal* transform, requiring a costly orthogonalization between updates and
limited to linear maps. Bengio (1991) proposed learning a richer (neural-network) transform but
the general network class lacks the structure to make inference and optimization practical.
Gaussianization (Chen & Gopinath 2000) learns a layered transform toward a Gaussian but greedily
and without a tractable sampling procedure. Rippel & Adams (2013) revive the learned-transform
idea but, lacking a bijectivity constraint, fall back to a regularized-autoencoder proxy for
log-likelihood rather than the likelihood itself. Nonlinear ICA via ensemble learning (Hyvärinen
1999; Roberts; Lappalainen) uses the variational bound as a more principled proxy. *Gap across
all of these:* none combines an expressive nonlinear transform, an exact tractable likelihood, a
trivial inverse for sampling, and a trivial Jacobian determinant.

# Evaluation settings

The natural yardstick is density estimation on image corpora of the time, reported as
**log-likelihood** in nats (and comparably as bits-per-dimension after dividing by D·log 2).
Because pixels are discrete with k = 256 levels per channel, continuous-density numbers are made
fair by **dequantization** before training (add uniform noise of one bin, rescale to a bounded
range); the bits-per-dimension comparison accounts for the discrete levels by adding the constant
D·log k to the negative log-likelihood. Standard datasets are MNIST (LeCun et al. 1998; D = 784),
the Toronto Face Dataset (Susskind 2010; D = 2304, trained on the unlabeled split), Street View
House Numbers (Netzer et al. 2011; D = 3072), and CIFAR-10 (Krizhevsky 2010; D = 3072). Linear
preprocessing is part of the protocol: none for MNIST, approximate whitening for TFD, exact ZCA
for SVHN and CIFAR-10. Comparison points reported in nats include deep mixtures of factor
analysers (Tang et al. 2012) and Gaussian RBMs; continuous-MNIST generative models are otherwise
usually scored by Parzen-window estimation, which makes likelihood comparisons there unreliable.
Beyond the number, the qualitative protocol is to draw unbiased samples (h ~ p_H, x = f^{-1}(h)),
to perform inference followed by reconstruction, to inpaint by clamping observed pixels and doing
gradient ascent on the likelihood over the missing ones, and to inspect the learned latent
spectrum and manifold (traversing a sphere in latent space, mapping it back through f^{-1}).

# Code framework

Ordinary deep-learning primitives suffice for the harness: a ReLU MLP module, tensor reshaping,
a factorial prior with a `log_prob`, and Adam. The open slot is the concrete data-to-latent
bijection and the exact objective that scores it.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    """A plain deep rectified network: the flexible function we are allowed to use freely."""
    def __init__(self, in_dim, mid_dim, hidden, out_dim):
        super().__init__()
        # Linear -> ReLU stack with a linear output head.
        pass

    def forward(self, x):
        pass

class Bijection(nn.Module):
    """The data-to-latent map. Must run forward and backward, and expose how it
    changes volume (log|det Jacobian|)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x, reverse=False):
        # TODO: an invertible map whose Jacobian determinant is tractable.
        pass

class DensityModel(nn.Module):
    def __init__(self, prior, *args, **kwargs):
        super().__init__()
        self.prior = prior
        # TODO: assemble the bijection.

    def f(self, x):
        # TODO: data -> latent, accumulating log|det Jacobian|.
        pass

    def g(self, z):
        # TODO: latent -> data (the inverse of f).
        pass

    def log_prob(self, x):
        # TODO: exact log-likelihood = prior log-density at f(x) + accumulated log-determinant.
        pass

    def sample(self, n):
        # TODO: draw from the prior and map back through g.
        pass

def dequantize(x, num_levels=256):
    # TODO: turn bounded discrete pixels into continuous data so a continuous density is well posed.
    pass

def make_optimizer(params):
    return torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.01), eps=1e-4)
```
