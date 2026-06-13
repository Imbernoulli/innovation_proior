# Context

## Research question

Supervised convolutional networks have, over the last few years, become the dominant tool in
computer vision: with backpropagation, ReLU units, and large labeled datasets, deep CNNs map images
to labels astonishingly well. But labels are the bottleneck. There is a practically unlimited supply
of *unlabeled* images and video, and the open question is how to turn that supply into reusable
feature representations — intermediate features that, once learned without labels, transfer to
downstream supervised tasks (classification, detection) and cut the amount of labeled data needed.

Adversarial networks are an attractive route to such features. Unlike maximum-likelihood generative
models they need no explicit, normalizable density, and unlike autoencoders they are not tied to a
pixel-wise reconstruction loss (which is known to produce blurry, over-smoothed outputs because
averaging over plausible reconstructions is itself low-loss). The hope is that a generator forced to
synthesize convincing images, and a discriminator forced to tell real from fake, will each build a
hierarchy of useful features as a *byproduct* of the adversarial game — features one could then peel
off and reuse.

Two things block this. First, adversarial training of convolutional generators is **unstable**:
attempts to scale the adversarial game up with the CNN architectures that work in the supervised
literature have repeatedly failed — generators produce nonsense, or collapse so that every latent
input yields the same output. Second, even when something trains, the resulting networks are
**black boxes**: there is almost no published understanding of what a multi-layer convolutional
generator or discriminator actually learns internally. The precise question, then: **is there a
family of convolutional architectures and training settings that makes the adversarial game train
stably across a range of image datasets — stably enough to scale to deeper models and higher
resolution — and do the networks it produces learn feature representations that are reusable for
supervised tasks and inspectable enough to understand?** A solution has to keep the well-behaved
gradients that make deep CNNs trainable while surviving the moving-target dynamics of a two-player
game, and it has to do so without a per-image reconstruction loss.

## Background

The field state (the deep-CNN surge of the early-to-mid 2010s): discriminative CNNs dominate vision;
unsupervised feature learning lags behind. The load-bearing concepts a new method rests on:

- **The adversarial framework (Goodfellow et al., 2014).** Two networks play a minimax game on one
  value function: a generator `G(z)` maps a noise vector `z ~ p_z` to a sample, implicitly defining a
  model distribution `p_g` (no density is ever written down); a discriminator `D(x) ∈ (0,1)`
  estimates the probability that `x` is real. They optimize
  `min_G max_D E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]`. In practice `D` is trained with
  binary cross-entropy (real→1, fake→0) and `G` is trained, non-saturatingly, to *maximize*
  `log D(G(z))` so its gradient does not vanish when `D` is confident. The framework needs only
  forward and backward passes — no partition function, no Markov chain — which is exactly why it is an
  attractive substrate for CNNs. Its documented weaknesses are training instability and mode collapse,
  and it has so far resisted being scaled up with deep convolutional networks.

- **Replacing pooling with strided convolutions (Springenberg et al., 2014, "Striving for
  Simplicity").** Deterministic spatial pooling (max-pooling) downsamples a feature map with a fixed,
  hand-chosen operator that discards spatial information in a way the network cannot adapt. A
  *strided* convolution — a convolution that steps by more than one pixel — downsamples too, but the
  downsampling operator is *learned*. The all-convolutional net showed that removing pooling entirely
  in favor of strided convolutions matches or beats pooling-based classifiers. A *fractionally-strided*
  (transposed) convolution is the corresponding operator that *increases* spatial size, with a learned
  rather than fixed kernel; the conventional alternative for enlarging a feature map is a fixed
  unpooling or nearest-neighbor resize. (These transposed convolutions are sometimes loosely called
  "deconvolutions.")

- **Removing fully-connected layers; global average pooling.** State-of-the-art image classifiers of
  the period moved away from large fully-connected heads on top of convolutional features, the
  strongest example being global average pooling, which collapses each final feature map to a single
  number and feeds those directly to the output. This removes the bulk of a network's parameters and
  acts as a structural regularizer.

- **Batch Normalization (Ioffe & Szegedy, 2015).** Normalizes the input to each unit over the
  minibatch to zero mean and unit variance (with learned scale and shift), reducing internal
  covariate shift. It stabilizes learning, makes deep networks robust to poor initialization, and
  improves gradient flow through many layers — the difference between a deep net that starts learning
  and one that stalls.

- **Activation functions.** ReLU (Nair & Hinton, 2010), `max(0, x)`, gives clean non-vanishing
  gradients on its active side but a hard zero (and zero gradient) on the negative side, so units can
  "die." Leaky ReLU (Maas et al., 2013; Xu et al., 2015), `max(αx, x)` with small `α`, keeps a small
  negative-side slope so gradient always flows. Tanh is a *bounded* activation saturating to `[-1,1]`.
  The original adversarial-network work used maxout units (Goodfellow et al., 2013) in the
  discriminator.

- **Linear structure in learned representation spaces (Mikolov et al., 2013).** Word embeddings
  trained for one task exhibit linear vector arithmetic —
  `vec("King") − vec("Man") + vec("Woman") ≈ vec("Queen")` — evidence that a learned latent space can
  encode semantic attributes as linear directions. The same question can be asked of a generator's
  noise-input space.

- **CNN-internals visualization.** Deconvolutional visualization (Zeiler & Fergus, 2014) and guided
  backpropagation (Springenberg et al., 2014) trace which input patterns maximally activate a given
  convolutional filter, giving a way to ask *what a filter has learned to detect* — a tool for opening
  the black box.

## Baselines

The prior methods a new procedure would be measured against and reacts to:

- **The adversarial framework with naive CNN generators (Goodfellow et al., 2014).** The base game,
  instantiated with the convolutional architectures borrowed from supervised vision. *Gap:* unstable —
  it frequently produces nonsensical generators or collapses all latents to one output, and has not
  been made to scale to deep convolutional models or higher resolution.

- **LAPGAN — Laplacian Pyramid of GANs (Denton et al., 2015).** Sidesteps the instability of one large
  convolutional generator by training a *pyramid* of conditional adversarial models, each one adding a
  band of high-frequency detail to upscale a lower-resolution image one level at a time. Produces
  visibly higher-quality images than a single naive GAN. *Gap:* chaining multiple independent models
  injects noise at each stage, so objects come out "wobbly"; and the pyramid is built for image
  synthesis, not as a single feature extractor one could reuse for supervised tasks.

- **Variational autoencoder (Kingma & Welling, 2013).** A directed latent-variable generator trained
  by maximizing a variational lower bound, with a learned inference network. Trains stably and admits
  a latent code. *Gap:* the explicit per-pixel reconstruction term pushes samples toward blurriness.

- **Autoencoder-family unsupervised feature learners.** Stacked / convolutional autoencoders (Vincent
  et al., 2010), what-where autoencoders (Zhao et al., 2015), ladder networks (Rasmus et al., 2015),
  and deep belief networks (Lee et al., 2009) all learn features by reconstructing or modeling the
  input. *Gap:* tied to reconstruction objectives and, in the deep-belief case, to layerwise
  generative training; the question of whether an adversarial discriminator yields competitive
  reusable features was open.

- **Hand-engineered / clustering feature pipelines.** A strong tradition fits features without deep
  end-to-end training: K-means feature learning on image patches (Coates et al.), including
  multi-layer and view-invariant variants, achieving the top unsupervised CIFAR-10 numbers of the
  time; and Exemplar CNNs (Dosovitskiy et al., 2014), which train a discriminative CNN to tell
  aggressively-augmented exemplar patches apart. *Gap:* either shallow (K-means pipelines need very
  many feature maps) or reliant on a hand-designed surrogate task.

## Evaluation settings

The benchmarks, datasets, metrics, and protocols that form the natural yardstick:

- **Datasets for training the generator.** Large-scale Scene Understanding (LSUN; Yu et al., 2015) —
  the bedrooms subset, ~3M images — for scaling to high resolution and large data; Imagenet-1k (Deng
  et al., 2009), center-cropped and min-resized to 32×32, as a source of natural images; and a scraped
  Faces dataset. MNIST is the standard small testbed for a conditional variant.

- **Feature-reuse protocol.** The accepted way to judge an unsupervised representation: treat the
  trained network as a fixed feature extractor and fit a simple linear model on top of its features,
  then measure supervised accuracy. For a convolutional feature extractor this means taking the
  convolutional feature maps (optionally spatially pooled to a small grid), flattening and
  concatenating them, and training a regularized linear classifier (e.g. an L2-SVM). The supervised
  yardsticks are CIFAR-10 (Krizhevsky & Hinton, 2009) classification accuracy and StreetView House
  Numbers (SVHN; Netzer et al., 2011) test error in the label-scarce regime (e.g. 1000 labels).

- **What deliberately is *not* used to judge sample quality.** Log-likelihood is a poor metric for
  these models, and nearest-neighbor search in pixel or feature space is trivially fooled by small
  image transforms (Theis et al., 2015); neither is used.

- **Qualitative protocols.** Walk along a path between two latent vectors and decode each point to
  inspect whether transitions are smooth (sharp jumps suggest memorization). Perform vector arithmetic
  on latent vectors and decode the result. Visualize discriminator filters via guided
  backpropagation. Ablate ("drop") chosen feature maps to test whether the generator's representation
  of an object is localized and disentangled.

## Code framework

The available substrate is a generic adversarial-training harness over an automatic-differentiation
deep-learning library: a noise sampler, a generator network, a discriminator network, a binary
cross-entropy loss, an optimizer, and a loop alternating discriminator and generator updates. What is
missing is the *internal architecture* of the two convolutional networks and the training settings
that make the game stable.

```python
import torch
import torch.nn as nn
import torch.optim as optim

nz = 100            # latent dimension
nc = 3              # image channels
image_range = (-1, 1)   # images pre-scaled to this range

class Generator(nn.Module):
    # maps a noise vector z -> an image; differentiable, one forward pass
    def __init__(self):
        super().__init__()
        # TODO: the convolutional architecture that maps nz-dim noise to an image,
        #       and the per-layer normalization / activation choices that make it stable
        self.net = None
    def forward(self, z):
        pass

class Discriminator(nn.Module):
    # maps an image -> scalar probability "real"
    def __init__(self):
        super().__init__()
        # TODO: the convolutional architecture that maps an image to one scalar,
        #       and the per-layer normalization / activation choices
        self.net = None
    def forward(self, x):
        pass

def init_weights(m):
    pass            # TODO: how to initialize weights for a stable start

netG, netD = Generator(), Discriminator()
criterion = nn.BCELoss()                       # binary cross-entropy: the adversarial game's loss
# TODO: which optimizer and which hyperparameters keep a two-player game stable?
optG = None
optD = None

def train_step(real, opt_g, opt_d):
    # standard alternating adversarial update
    z = torch.randn(real.size(0), nz, 1, 1)    # sample noise
    # 1) update D on real (label 1) and fake (label 0)
    # 2) update G so D scores the fake as real (label 1)
    pass
```

This harness can run the adversarial game on any pair of networks, but the stable convolutional
architecture, the placement of normalization, the activation choices, and the optimizer settings are
the open problems.
