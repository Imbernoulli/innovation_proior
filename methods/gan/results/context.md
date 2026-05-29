# Context

## Research question

Supervised deep learning has just become spectacularly effective at *discrimination* — mapping a
rich, high-dimensional input (an image, a spectrogram, a sentence) to a label — by riding three
well-behaved ingredients: backpropagation, dropout, and piecewise-linear units (ReLU, maxout),
whose gradients neither vanish nor explode through many layers. Deep *generative* modeling has not
had the same moment, and the gap is structural rather than accidental. The trouble is that the
dominant way to *learn* a generative model is maximum likelihood, and maximum likelihood forces you
to write down — and normalize, and differentiate — an explicit probability density over the data.
For the deep models people actually want, that explicit density is intractable: an intractable
normalizing constant in energy-based/undirected models, intractable posterior inference in directed
latent-variable models, or a dependence on a Markov chain that must mix between modes. None of these
composes cleanly with the piecewise-linear-unit + pure-backprop recipe that made deep classifiers
take off — a feedback loop or an MCMC inner loop is exactly where well-behaved gradients stop being
available.

It helps to see the space of options as a tree. To do maximum likelihood you need a tractable handle
on `p(x)`. One branch makes the density *explicit and tractable* (e.g. fully-visible belief networks
that factor `p(x) = ∏ p(x_i | x_{<i})` — exact, but sampling is inherently sequential and slow, and
there is no latent code). A second branch keeps the density explicit but only *tractable up to a
constant or a bound* (Boltzmann machines, which carry an intractable partition function; variational
autoencoders, which optimize a lower bound on the likelihood through an inference network). A third
branch gives up on writing the density at all and only keeps the ability to *draw samples* — an
*implicit* model. The implicit branch is the one nobody has made train cleanly with pure backprop and
no Markov chain.

The precise question, then: **is there a way to estimate a generative model — to fit a model
distribution to a data distribution and draw samples from it — using only backpropagation and forward
propagation, with no intractable partition function, no Markov chain in either training or sampling,
and no approximate-inference network?** A solution would have to supply a learning signal that drives
the model toward the data distribution without ever evaluating, or even analytically specifying, an
explicit density — while staying compatible with the well-behaved-gradient units that make deep nets
trainable. Matching that requirement is the whole difficulty: every existing route pays an
explicit-probability tax somewhere.

## Background

The field state (the early-2010s deep-learning surge): discriminative deep nets are booming on the
back of backprop, dropout, and rectified/maxout units, while generative models lag. The load-bearing
concepts a new method rests on are:

- **The maximum-likelihood gradient for energy-based models.** For `p(x) = exp(-E(x))/Z`, the gradient
  is `∂ log p(x)/∂θ = -E_data[∂E/∂θ] + E_model[∂E/∂θ]`. The negative-phase term, an expectation under
  the *model*, is exactly what the partition function `Z` hides, forcing an MCMC approximation.
  **Mixing** between modes is the chronic pain: local transition operators cannot cross
  low-probability "deserts," so negative-phase samples are biased and learning is slow and fragile
  (Bengio et al., ICML 2013/2014). This is the canonical example of the explicit-probability tax — the
  price of having written `p(x)` down at all.

- **Discriminative criteria for generative fitting.** A counter-tradition fits a generative model
  *without* maximizing likelihood, by setting up a classification-style objective — score matching
  (Hyvärinen, 2005) and especially noise-contrastive estimation (Gutmann & Hyvärinen, 2010), which
  turns density estimation into logistic regression. The central, reusable idea: *a classifier's
  success can serve as the learning signal for a generative model.* The hidden weakness is that the
  contrast is against a *fixed* distribution, so the classification problem becomes easy — and the
  gradient slack — the moment the model is even approximately right.

- **The density ratio.** Telling apart two distributions is governed by their ratio
  `p_data(x) / p_g(x)`; the Bayes-optimal classifier of "real vs. generated" with equal class priors
  outputs `p_data / (p_data + p_g)`, a monotone transform of that ratio. This quantity is well-defined
  even when neither density can be written in closed form — which is why a *classifier* can stand in
  for two intractable densities.

- **The reparameterized / differentiable generator.** Express a continuous latent (or a whole sample)
  as a deterministic differentiable function of injected noise, `x = G(z; θ)` with `z ~ p_z`, so
  gradients can pass through the sampling step. Then sampling is a single forward pass and `θ` can be
  trained by backprop through any differentiable downstream signal. This change-of-variable idea is old
  in statistics (its derivative identities trace to Price 1958, Bonnet 1964) and is in active use — it
  underlies stochastic backprop and the variational-autoencoder line. **The open hole is the downstream
  signal:** what loss should train `θ`, if not a likelihood or a variational bound?

- **Persistent negative chains.** SML/PCD (Younes, 1999; Tieleman, 2008) trains energy-based models by
  carrying a persistent set of Markov-chain samples across learning steps instead of burning in a fresh
  chain inside the inner loop — the template for keeping an inner quantity *near* its optimum while the
  outer parameters move slowly.

- **Information-theoretic divergences.** KL divergence `KL(p‖q) = ∫ p log(p/q)` and the symmetric
  Jensen–Shannon divergence `JSD(p‖q) = ½KL(p‖m) + ½KL(q‖m)` with `m = (p+q)/2`; JSD is non-negative and
  zero iff `p = q`, and is finite even when supports do not overlap. KL is asymmetric (it has a
  mode-covering and a mode-seeking direction and can be infinite on disjoint supports), whereas JSD is
  symmetric and bounded — the difference will matter for which divergence a *symmetric* real-vs-fake
  comparison produces.

## Baselines

The prior methods a new generative procedure would be measured against and reacts to:

- **Restricted Boltzmann Machines (Smolensky, 1986; Hinton, 2006) and Deep Boltzmann Machines
  (Salakhutdinov & Hinton, 2009).** Undirected, energy-based: `p(x) = exp(-E(x))/Z` with `Z`
  summing/integrating over all states. Trained by approximating the negative-phase expectation with
  MCMC (contrastive divergence, persistent CD). *Gap:* the intractable `Z` never goes away and learning
  is at the mercy of Markov-chain mixing between modes; sampling itself needs a chain.

- **Deep Belief Networks (Hinton, 2006).** Hybrids: one undirected (RBM) top layer over several
  directed layers, trained greedily layer-by-layer with a fast approximate criterion. *Gap:* being a
  hybrid, they inherit the computational difficulties of *both* the undirected and the directed worlds.

- **Score matching (Hyvärinen, 2005).** Fits a model specified up to normalization by matching the
  gradient of the log-density `∇_x log p` between model and data, which cancels `Z`. Denoising
  autoencoders (Vincent et al., 2008) and contractive autoencoders end up with learning rules close to
  score matching on an RBM. *Gap:* it still requires the unnormalized density to be written down
  analytically; for models with several layers of latent variables you cannot even derive a tractable
  unnormalized density, so it does not apply to the deep models in question.

- **Noise-Contrastive Estimation (Gutmann & Hyvärinen, 2010).** The closest ancestor in spirit. Turns
  estimation into logistic regression: train a classifier to distinguish observed data from samples of
  a *fixed* auxiliary noise distribution, using the model's own unnormalized log-density inside the
  logistic nonlinearity and learning `Z` as a parameter. *Gaps:* (1) it still needs the model density
  specified analytically up to `Z`; (2) the contrast distribution is fixed, so once the model is even
  approximately right on a small subset of the variables, telling data from the fixed noise becomes
  trivial, the classifier saturates, and learning slows dramatically. The hanging question it leaves:
  *what if the contrast were not fixed, but a model that keeps improving so the classification task
  never goes slack?*

- **Generative Stochastic Networks (Bengio et al., ICML 2014), extending generalized denoising
  autoencoders (Bengio et al., NIPS 2013).** Give up an explicit density and train a generative
  *machine* that emits samples, parameterizing one step of a generative Markov chain so it is trainable
  by backprop. The closest precedent for an implicit, sample-only model. *Gap:* sampling still requires
  running a Markov chain (the mixing problem returns), and the feedback loop makes piecewise-linear
  units awkward — they can blow up with unbounded activations inside the recurrence, so the very units
  that make backprop nice become liabilities.

- **Auto-encoding variational Bayes / the variational autoencoder (Kingma & Welling, 2014) and
  stochastic backpropagation (Rezende et al., 2014).** Same-era backprop-into-a-generator methods: use
  the reparameterization trick so gradients flow through sampling, and maximize a variational lower
  bound (the ELBO) on the log-likelihood with a learned approximate-inference (encoder) network
  regularized to match the prior. Genuinely close — a differentiable generator trained by backprop, no
  Markov chain. *Gap:* still likelihood-based (a variational *bound*), still needs an inference network
  during training, and the explicit reconstruction term in the ELBO tends toward blurry samples.

- **SML/PCD (Younes, 1999; Tieleman, 2008)** as a *training-schedule* baseline: keep a persistent inner
  state near its optimum across outer steps rather than re-solving it from scratch each step.

- **Wake-sleep (Hinton et al., 1995):** trains a separate recognition network to invert a generator —
  the template for bolting learned approximate inference onto a model after the fact.

## Evaluation settings

The benchmarks, datasets, and protocol that form the natural yardstick:

- **Datasets.** MNIST (LeCun et al., 1998) handwritten digits; the Toronto Face Database (TFD;
  Susskind et al., 2010); and CIFAR-10 (Krizhevsky & Hinton, 2009) natural images. Both fully-connected
  and convolutional treatments of these are standard.

- **Metric for implicit models.** When a model can *sample* but cannot evaluate its likelihood, the
  established quantitative protocol is a **Gaussian Parzen-window log-likelihood estimate**: fit a
  Parzen (kernel-density) estimator with isotropic Gaussian kernels to a set of generated samples,
  choose the kernel bandwidth `σ` by cross-validation on a validation set, and report the mean
  log-likelihood of the held-out test set under that density. The procedure was introduced by
  Breuleux & Bengio (2011) and used for several generative models whose exact likelihood is intractable
  (Rifai et al., 2012; Bengio et al., ICML 2013/2014). It is known to have high variance and to behave
  poorly in high dimensions, but is the accepted yardstick for sample-only models.

- **Qualitative protocol.** Display fair random draws from the model (not cherry-picked, not
  conditional means over hidden units), and show, for each sample, its nearest training example to
  demonstrate the model is not memorizing the training set. For latent-variable generators, interpolate
  linearly between two points in `z`-space and decode along the path to inspect whether the learned
  manifold is smooth.

## Code framework

The available code substrate is a generic implicit-generative-model harness: noise sampling, a
differentiable network that maps noise to samples, automatic differentiation, an optimizer, and a
loop over minibatches. The missing part is the scalar training signal.

- **Theano (Bergstra et al., 2010; Bastien et al., 2012).** A symbolic-expression compiler with GPU
  support and automatic differentiation. The load-bearing capability: given one scalar expression and a
  named list of parameters, `T.grad(cost, params)` returns the gradient of the expression *with respect
  to exactly those parameters*. It also supplies the noise sources (`MRG_RandomStreams.normal/uniform`)
  needed to feed a noise-driven generator.

- **Pylearn2 (Goodfellow et al., 2013).** A research library on top of Theano supplying the
  model/training scaffolding: `MLP` and `Layer` (including maxout layers), dropout via `dropout_fprop`,
  a `Cost` abstraction with `expr`/`get_gradients` hooks, and SGD-with-momentum training driven by
  `TrainExtension` callbacks.

- **Piecewise-linear units and dropout.** ReLU (Jarrett et al., 2009; Glorot et al., 2011) and maxout
  (Goodfellow et al., 2013), plus dropout (Hinton et al., 2012). These give the well-behaved gradients
  that make deep feedforward nets train — provided there is no feedback loop forcing unbounded
  activations back through themselves.

```python
# Bare harness for an implicit generative model.
# Theano supplies T.grad; Pylearn2 supplies MLP layers and SGD with momentum.

import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

rng = MRG_RandomStreams(seed)

def sample_noise(m, n_z):
    # the only stochasticity: a noise prior z ~ p(z)
    return rng.normal(size=(m, n_z))          # prior is arbitrary (gaussian/uniform); sampler exists

class Generator:
    # a differentiable net mapping noise -> a sample; one forward pass, no Markov chain.
    # Built from existing piecewise-linear-unit MLP layers.
    def __init__(self, mlp): self.mlp = mlp
    def sample(self, m, n_z):
        z = sample_noise(m, n_z)
        return self.mlp.fprop(z)              # x = G(z); implicitly defines a model distribution
    params = property(lambda self: self.mlp.get_params())

def train_step(generator, real_batch):
    # TODO: define a scalar objective using generated samples and real data.
    objective = None                          # TODO: define the scalar to optimize
    grads = T.grad(objective, generator.params)   # autodiff will fill this once the objective exists
    return grads

# An SGD-with-momentum optimizer + a loop over real minibatches drive train_step (Pylearn2 SGD).
```

This harness can backpropagate any scalar objective through a noise-driven generator, but by itself
it supplies only sampling and optimization; the likelihood-free learning signal remains the open
technical problem.
