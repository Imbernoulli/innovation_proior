# Context

## Research question

Unsupervised representation learning aims to extract value from the vast supply of unlabeled data by
learning a representation that exposes the salient semantic factors of a data instance as easily
decodable, independent coordinates — a *disentangled* representation. For a face dataset, the ideal
would allocate one coordinate each to expression, eye color, hairstyle, presence of glasses, identity;
for handwritten digits, one coordinate to digit identity and others to stroke angle and thickness.
Such a representation is useful precisely because the downstream tasks are unknown at training time:
if the salient factors are laid out as separate, interpretable dimensions, whatever supervised task
arrives later can read off the dimensions it needs.

Generative modeling is an attractive route — the belief is that a model good enough to *synthesize*
the data must have captured its factors of variation. But there is no free lunch: it is trivial to
build a perfect generative model whose internal representation is arbitrarily entangled and useless.
So generation alone does not produce disentanglement. The two prominent deep generative models, the
variational autoencoder and the adversarial network, both take an unstructured latent vector and are
free to use it however minimizes their training loss — which generally means entangling all the
factors together, so no single latent coordinate corresponds to a semantic feature.

The precise question: **is there a way to make a deep generative model learn a disentangled,
interpretable representation — one latent coordinate per semantic factor — entirely without
supervision, on complex image datasets, and without paying a large computational or training-stability
cost?** Every prior method that recovers disentangled factors leans on supervision of some kind:
labels for the factors, or knowledge of which pairs of examples share a factor, or temporal grouping.
A solution would have to *induce* the structure from the data alone, forcing the model to spend
specific latent coordinates on specific factors without ever being told what those factors are.

## Background

The field state (mid-2010s deep generative modeling): adversarial and variational generators produce
increasingly convincing images, but their latent spaces are entangled, and disentangling them has so
far required supervision. The load-bearing concepts a solution rests on:

- **The adversarial framework (Goodfellow et al., 2014).** A generator `G` maps a noise vector
  `z ~ p_noise` to a sample, implicitly defining a distribution `P_G`; a discriminator `D(x)`
  estimates the probability that `x` is real. They play the minimax game
  `min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z}[log(1 - D(G(z)))]`, whose optimal discriminator
  for a fixed `G` is `D*(x) = p_data(x) / (p_data(x) + P_G(x))`. The input noise `z` is a single
  unstructured vector with no imposed meaning, so `G` is free to use it in a highly entangled way.

- **A stable convolutional adversarial architecture (Radford et al., 2015).** The set of architectural
  constraints — strided/fractionally-strided convolutions, batch normalization in both networks
  (except the boundary layers), ReLU in the generator with a tanh output, leaky ReLU in the
  discriminator, Adam with a low momentum term — that makes a deep convolutional adversarial pair
  train stably across datasets, and whose generator's noise space already supports some linear
  algebra. This is the stable substrate any new adversarial method would build on rather than
  reinventing.

- **Mutual information.** For random variables `X, Y`,
  `I(X; Y) = H(X) − H(X|Y) = H(Y) − H(Y|X)`, the reduction in uncertainty about one from observing the
  other. `I(X; Y) = 0` iff `X` and `Y` are independent; it is maximal when one is a deterministic,
  invertible function of the other. Read as a design target: if a latent code and a generated image
  have high mutual information, the information in the code is *preserved* in the image and cannot have
  been thrown away.

- **Variational Information Maximization (Barber & Agakov, 2003).** Mutual information involving a
  posterior `P(c|x)` that cannot be evaluated can still be *lower-bounded* by introducing an auxiliary
  distribution `Q(c|x)` that approximates the true posterior. The bound becomes tight as `Q` approaches
  `P(·|x)`. Similar mutual-information objectives have been used to drive clustering (Bridle et al.,
  1992; Krause et al., 2010). This is the tool that converts an intractable information target into
  something differentiable.

- **The Helmholtz machine and Wake-Sleep (Dayan et al., 1995; Hinton et al., 1995).** A generative
  model `P(x|c)` paired with a learned recognition model `Q(c|x)`, trained by alternating a "wake"
  phase (update the generator on data passed through the recognition net) and a "sleep" phase (update
  the recognition net on the generator's own dreamed samples). The template for bolting a recognition
  network onto a generator.

## Baselines

The prior methods a new procedure would be measured against and reacts to:

- **The adversarial framework with an unstructured noise input (Goodfellow et al., 2014); the stable
  convolutional variant (Radford et al., 2015).** Powerful generators, and the convolutional variant's
  noise space supports basic vector arithmetic. *Gap:* the latent input is unstructured, so the model
  has no incentive to map any single latent coordinate to a single semantic factor — the
  representation is entangled by default.

- **DC-IGN — Deep Convolutional Inverse Graphics Network (Kulkarni et al., 2015).** Learns graphics
  codes (pose, lighting, elevation) for 3D-rendered images by *clamping*: during training it presents
  minibatches in which only one factor varies and forces the rest of the code to stay fixed.
  *Gap:* requires supervised knowledge of which factor varies in each minibatch — the salient factors
  must be known and grouped in advance, so factors that are not anticipated cannot be discovered.

- **disBM — disentangling Boltzmann machine (Reed et al., 2014).** A higher-order Boltzmann machine
  that disentangles by clamping a part of the hidden units across a pair of data points known to match
  in all-but-one factor. *Gap:* weak supervision (knowing which examples share which factor).

- **hossRBM — higher-order spike-and-slab RBM (Desjardins et al., 2012).** To the extent any prior
  method disentangles fully unsupervised, this one does, separating emotion from identity on a face
  dataset. *Gap:* it can only disentangle *discrete* latent factors, and its computational cost grows
  *exponentially* in the number of factors, so it does not scale.

- **Supervised / weakly-supervised disentangling: bilinear models (Tenenbaum & Freeman, 2000),
  multi-view perceptron (Zhu et al., 2014), recurrent latent-transform models (Yang et al., 2015),
  adversarial autoencoders (Makhzani et al., 2015), semi-supervised deep generative models (Kingma et
  al., 2014).** All separate a labeled factor (often the class) from the rest of the variation by
  *matching part of the representation to a supplied label*. *Gap:* they all consume supervision.

## Evaluation settings

The benchmarks, datasets, and protocol that form the natural yardstick:

- **Datasets.** MNIST handwritten digits; the StreetView House Numbers dataset (SVHN; Netzer et al.,
  2011), which is noisy with distracting digits and variable resolution; CelebA (Liu et al., 2015),
  ~200k celebrity faces with large pose variation and background clutter; and the 3D-rendered Faces
  (Paysan et al., 2009) and 3D Chairs (Aubry et al., 2014) datasets, which are exactly the datasets on
  which a supervised graphics-code learner demonstrated its codes, enabling direct comparison.

- **Protocol for judging disentanglement.** The accepted way to test whether a single latent
  coordinate captures a single factor: hold all latent variables and the noise fixed except one, vary
  that one coordinate across its range (e.g. from −2 to 2 for a continuous code, or sweep a categorical
  code through its categories), decode each setting, and inspect whether the generated images change in
  exactly one interpretable way (only rotation, only width, only digit identity). A *generalization*
  variant: vary the code beyond the range seen in training (e.g. −2 to 2 when trained on −1 to 1) to
  test whether the learned factor extrapolates. For a discrete code, one can also test how well its
  recovered category matches the true class as an unsupervised classifier.

- **Comparison setting.** Against a supervised graphics-code learner on the 3D datasets, present the
  factor representation that most resembles the supervised result out of several random runs, since the
  unsupervised method does not control which coordinate captures which factor.

## Code framework

The available substrate is a stable convolutional adversarial-training harness (the
strided-conv / batchnorm / leaky-ReLU recipe with Adam) into which a structured latent input and a
recognition mechanism must be added. What exists: a generator that takes a single latent vector, a
discriminator/classifier body, a binary cross-entropy adversarial loss, and the alternating update
loop. What is missing: how to *structure* the latent input so that part of it carries semantics, an
auxiliary mechanism to *recover* that structured part from a generated image, and the loss that ties
the two together.

```python
import torch
import torch.nn as nn

latent_dim = 62          # unstructured noise dimension

class Generator(nn.Module):
    # maps a latent vector -> an image (stable up-convolutional architecture)
    def __init__(self):
        super().__init__()
        # TODO: how should the latent INPUT be structured so that part of it
        #       carries recoverable semantic factors?
        self.net = None
    def forward(self, latent):
        pass

class Discriminator(nn.Module):
    # shared convolutional body + a real/fake head
    def __init__(self):
        super().__init__()
        self.body = None                  # conv stack (leaky-ReLU, batchnorm)
        self.validity_head = None         # -> scalar P(real)
        # TODO: an auxiliary head that RECOVERS the structured part of the latent
        #       from the generated image, and its output parameterization
    def forward(self, x):
        pass

adversarial_loss = nn.BCELoss()           # the GAN game's loss
# TODO: the loss that forces the generator to actually USE the structured latent,
#       and how it is optimized jointly with the recovery mechanism

def train_step(real, opt_g, opt_d):
    # 1) update D on real vs. fake (adversarial)
    # 2) update G to fool D (adversarial)
    # 3) TODO: an update that maximizes how recoverable the structured latent is
    pass
```

This harness can train a stable convolutional adversarial pair, but it has no notion of a structured
latent code, no mechanism to recover one, and no objective tying a latent coordinate to a semantic
factor — those are the open problems.
