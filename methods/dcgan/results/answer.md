# DCGAN — a stable convolutional architecture for the adversarial game

## The problem it solves

Train a *single* deep convolutional generator and discriminator inside the adversarial game stably
enough to scale to deeper models and higher resolution, and harvest the resulting networks as
reusable, inspectable feature representations learned from unlabeled images — without a per-pixel
reconstruction loss (which blurs) and without the multi-model pyramid workaround that prior
convolutional-GAN results needed. The adversarial *objective* is kept unchanged; the contribution is
the *architecture and training recipe* that makes the convolutional version trainable.

## The key idea

Walk through what destabilizes a naive CNN GAN and fix each piece, yielding a small set of
architectural constraints:

- **No pooling.** Replace deterministic pooling with *strided* convolutions in the discriminator
  (learned downsampling) and *fractionally-strided* (transposed) convolutions in the generator
  (learned upsampling).
- **No fully-connected hidden layers.** The generator's input is a matrix-multiply of the noise that
  is immediately *reshaped* into a small-spatial-extent, many-channel tensor; the discriminator's last
  convolutional feature map is *flattened* into a single sigmoid output. Nothing fully-connected in
  between. (Global average pooling was more stable but converged slower; this is the middle ground.)
- **Batch normalization in both networks, except two layers.** It lets a deep generator begin learning
  and prevents the generator collapsing all latents to one output. But not on the **generator output
  layer** (it would fight the pixel statistics and the bounded range) and not on the **discriminator
  input layer** (it would normalize away the raw-pixel real/fake signal).
- **Activations.** ReLU throughout the generator except a **tanh** output (bounded, saturates to the
  `[-1,1]` data range and learns to cover the color space). **Leaky ReLU** (slope 0.2) throughout the
  discriminator, so no units die and the generator keeps receiving a healthy gradient.
- **Training.** Images pre-scaled to `[-1,1]`; weights initialized from `N(0, 0.02)`; minibatch size
  128; latent `z` a 100-dim uniform vector. **Adam** with `lr = 0.0002` (the 0.001 default was too
  high) and `β1 = 0.5` (the 0.9 default caused oscillation — the game is non-stationary, so a shorter
  momentum memory tracks the moving opponent), `β2 = 0.999`.

## The objective (unchanged adversarial game)

    min_G max_D  E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

trained with binary cross-entropy: `D` pushes `D(real) → 1`, `D(fake) → 0`; `G` is trained
non-saturatingly to push `D(G(z)) → 1`.

## Architecture (64×64 RGB)

Generator: project-reshape `z` to `4×4×(8f)`, then transposed convs (kernel 4, stride 2, pad 1)
`4×4 → 8×8 → 16×16 → 32×32 → 64×64` with channels `8f → 4f → 2f → f → 3`; BatchNorm+ReLU after each
except the tanh output (no BatchNorm). Discriminator: convs (kernel 4, stride 2, pad 1)
`64 → 32 → 16 → 8 → 4` with channels `3 → f → 2f → 4f → 8f`, LeakyReLU(0.2) after each, BatchNorm
after each except the first; a final kernel-4 stride-1 conv collapses `4×4 → 1×1`, then sigmoid.
`f = 64`.

## Working code

```python
import torch
import torch.nn as nn
import torch.optim as optim

nz, ngf, ndf, nc = 100, 64, 64, 3

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),                                  # output: bounded, no batchnorm
        )
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),    # input: no batchnorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.main(x).view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

netG, netD = Generator(), Discriminator()
netG.apply(weights_init); netD.apply(weights_init)

criterion = nn.BCELoss()
optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
real_label, fake_label = 1.0, 0.0

def train_step(real):
    b = real.size(0)
    netD.zero_grad()
    loss_real = criterion(netD(real), torch.full((b,), real_label))
    loss_real.backward()
    z = torch.randn(b, nz, 1, 1)
    fake = netG(z)
    loss_fake = criterion(netD(fake.detach()), torch.full((b,), fake_label))
    loss_fake.backward()
    optD.step()
    netG.zero_grad()
    loss_g = criterion(netD(fake), torch.full((b,), real_label))
    loss_g.backward()
    optG.step()
    return loss_real + loss_fake, loss_g
```

## Why it works

Each constraint removes one source of the moving-target instability that wrecks naive convolutional
GANs: learned (not fixed) spatial resampling top-to-bottom; no fully-connected hidden layers to
regularize and stabilize; batchnorm to start deep training and resist collapse, but kept off the two
boundary layers where it would corrupt the pixel-level signal; leaky-ReLU so the discriminator never
starves the generator of gradient; tanh to bound and color-saturate the output; and Adam with a short
momentum memory to track an opponent that keeps moving. The trained discriminator's convolutional
features then serve as a reusable representation, and the generator's latent space is smooth enough
to interpolate and do vector arithmetic in.
