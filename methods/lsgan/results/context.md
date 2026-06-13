# Context

## Research question

The adversarial framework trains a generative model end-to-end through differentiable networks, with
no intractable partition function and no approximate inference — a clean advantage over the
energy-based and variational generative models. But the quality of the images it produces is still
limited for realistic tasks, and training it is notoriously unstable. In the standard formulation the
discriminator is a *classifier* trained with the sigmoid cross-entropy loss, and there is a specific,
diagnosable failure in how that loss feeds gradient back to the generator.

Picture the discriminator's decision boundary in data space. When the generator is updated, it is
pushed to make the discriminator classify its fake samples as real. Consider fake samples that already
lie *on the correct (real) side* of the decision boundary but are still *far* from the real data. Under
sigmoid cross-entropy these samples are already classified confidently as real, so the loss they incur
is essentially zero and the gradient they pass back to the generator vanishes — the generator receives
almost no signal to pull those far-but-correctly-classified samples closer to the real data. The
sigmoid saturates exactly where you still need to move. The precise question: **is there a
discriminator loss that keeps supplying a useful gradient to the generator for samples that are
correctly classified but still far from the real data — pulling them toward the real-data manifold —
and that thereby also makes adversarial training more stable?**

## Background

The field state (mid-2010s adversarial image generation): the adversarial framework produces the most
convincing unsupervised image samples, but image quality and training stability are the two open
fronts, and several lines of work trace both back to the *objective function*. The load-bearing
concepts and the key diagnostic observation:

- **The adversarial game and its sigmoid cross-entropy loss (Goodfellow et al., 2014).** A generator
  `G(z)` maps noise `z ~ p_z` to data space, implicitly defining `p_g`; a discriminator `D(x)`, a
  classifier with a sigmoid output, is trained with binary cross-entropy. The game is
  `min_G max_D E_{x~p_data}[log D(x)] + E_z[log(1 − D(G(z)))]`. It was shown that, at the optimal
  discriminator, this minimizes the Jensen–Shannon divergence between data and model:
  `C(G) = KL(p_data ‖ (p_data+p_g)/2) + KL(p_g ‖ (p_data+p_g)/2) − log 4`.

- **The saturation of sigmoid cross-entropy (the diagnostic insight).** The sigmoid cross-entropy loss,
  as a function of the classifier's pre-activation, is flat (saturated) once a sample is confidently on
  the correct side. So a fake that is already classified as real contributes a vanishing gradient,
  regardless of how far it is from the real data — which is exactly the regime where the generator
  still needs a push. This is the mechanism behind the vanishing-gradient difficulty when updating the
  generator.

- **The decision boundary must cross the real-data manifold.** For successful adversarial learning the
  discriminator's boundary has to pass through the region where the real data lives; otherwise learning
  saturates. Consequently, moving generated samples *toward the boundary* is, geometrically, moving
  them *toward the real-data manifold*.

- **f-divergences and the GAN objective (Nowozin et al., 2016; Nguyen et al., 2010).** The original
  game's link to Jensen–Shannon divergence is a special case of a general principle: an adversarial
  objective can be made to estimate and minimize an arbitrary f-divergence between `p_data` and `p_g`.

- **Stability is partly an objective problem.** Multiple analyses (Arjovsky et al., 2017; Metz et al.,
  2016; Qi, 2016; Che et al., 2016) attribute the instability of adversarial training in part to the
  objective. Arjovsky et al. argued the Wasserstein distance behaves better than Jensen–Shannon and
  introduced a stress test for stability: remove batch normalization (Ioffe & Szegedy, 2015) from the
  networks and see whether learning still converges. Qi's Loss-Sensitive GAN built a loss with
  non-vanishing gradient almost everywhere from the assumption that real samples should have smaller
  loss than fakes.

- **Conditioning on labels (Mirza & Osindero, 2014; Hornik, 1989).** Feeding label information to both
  networks makes the input→output relation deterministic, which a feed-forward network can represent;
  this is the standard way to make an adversarial generator class-conditional.

## Baselines

The prior methods a new procedure would be measured against and reacts to:

- **The adversarial game with a sigmoid cross-entropy discriminator (Goodfellow et al., 2014).** The
  base method; minimizes Jensen–Shannon divergence at optimum. *Gap:* the sigmoid loss saturates, so
  fakes that are correctly classified yet far from the data give a vanishing gradient — limiting image
  quality and destabilizing training.

- **The stable convolutional recipe (Radford et al., 2015).** Strided/fractionally-strided
  convolutions, batch normalization, ReLU in the generator, leaky ReLU in the discriminator — the
  architecture template that made convolutional adversarial training work, and the backbone any new
  variant would build on. *Gap:* it addresses architecture, not the objective's saturation.

- **Laplacian-pyramid GANs (Denton et al., 2015) and feature matching (Salimans et al., 2016).**
  Quality- and convergence-oriented improvements: a coarse-to-fine pyramid of conditional models, and
  matching the statistics of an intermediate discriminator layer. *Gap:* leave the core sigmoid
  objective in place.

- **Wasserstein GAN (Arjovsky et al., 2017).** Replaces the Jensen–Shannon objective with a
  Wasserstein-distance critic, improving stability and admitting training without batch normalization.
  *Gap:* requires multiple discriminator updates per generator update (a Lipschitz-constrained critic
  trained closer to optimality each step), which is comparatively slow.

- **Energy-based GAN (Zhao et al., 2016).** Views the discriminator as an energy function realized by
  an autoencoder, improving stability. *Gap:* a different discriminator design, still not addressing
  the classifier-loss saturation directly.

## Evaluation settings

The benchmarks, datasets, and protocol that form the natural yardstick:

- **Datasets.** LSUN scene categories (Yu et al., 2015) — bedroom, church-outdoor, dining room,
  kitchen, conference room — for scene image generation at `112×112`; and HWDB1.0 (Liu et al., 2011), a
  handwritten Chinese character dataset with thousands of classes (≈3740), for the many-class
  conditional setting. Noise `z` is drawn from a uniform or Gaussian prior.

- **Stability protocol (Arjovsky et al., 2017).** Compare training stability by *excluding* batch
  normalization: build architectures that drop BN from the generator only (BN_G), or from both
  generator and discriminator (BN_GD), and run each with multiple optimizers (Adam, RMSProp), since the
  optimizer choice strongly affects whether learning converges. Observe whether the method still
  reaches a good state and whether it avoids mode collapse.

- **A controlled toy distribution (Metz et al., 2016).** A 2D mixture of 8 Gaussians, modeled by small
  fully-connected generator and discriminator, with the model distribution visualized over training via
  Gaussian kernel density estimation — a setting that makes mode collapse directly visible.

## Code framework

The available substrate is the stable convolutional adversarial harness — a fractionally-strided
generator, a strided convolutional discriminator, batch normalization, ReLU/leaky-ReLU activations,
Adam — together with its standard discriminator loss. What is variable, and what the method will
change, is the *discriminator's loss function* (and through it the generator's), plus an optional
label-conditioning path for the many-class case.

```python
import torch
import torch.nn as nn

latent_dim = 100

class Generator(nn.Module):
    # noise -> image (fractionally-strided convolutional stack, ReLU, BN, tanh output)
    def __init__(self):
        super().__init__()
        self.net = None        # standard convolutional generator
    def forward(self, z):
        pass

class Discriminator(nn.Module):
    # image -> scalar score
    def __init__(self):
        super().__init__()
        self.features = None   # strided-conv / leaky-ReLU / BN stack
        # TODO: the discriminator's output form and its loss
        self.out = None
    def forward(self, x):
        pass

# TODO: the adversarial loss for D and for G. The base game uses a sigmoid + cross-entropy
#       classifier loss, which saturates; the open question is what loss to use instead.
adversarial_loss = None

def train_step(real, opt_g, opt_d):
    z = torch.randn(real.size(0), latent_dim)
    fake = G(z)
    # 1) update D to score real high and fake low (under the chosen loss)
    # 2) update G so D scores its fakes as real (under the chosen loss)
    pass
```

This harness can run the adversarial game on a stable convolutional pair, but the discriminator's
output form and the loss that drives both networks — the piece responsible for the vanishing-gradient
behavior — is the open problem.
