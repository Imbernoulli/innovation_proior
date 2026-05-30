# LSGAN — Least Squares Generative Adversarial Networks

## The problem it solves

The standard adversarial game uses a sigmoid cross-entropy discriminator loss, which saturates: a fake
sample that is already on the correct (real) side of the decision boundary but still far from the real
data incurs almost no loss, so it passes a vanishing gradient back to the generator — precisely where
the generator still needs a push. The result is limited image quality and unstable training. LSGAN
replaces the classifier loss with a least-squares (regression) loss that penalizes by *distance*, not
by *side*.

## The key idea

A quadratic loss is flat at exactly one point, whereas sigmoid cross-entropy is flat along a whole
half-line. So regressing the discriminator's output toward a target with a squared loss keeps
penalizing far-but-correctly-classified fakes, pulling them toward the decision boundary. Since the
boundary must cross the real-data manifold for learning to proceed, pulling fakes toward the boundary
pulls them toward real data — better samples — and the loss supplies gradient almost everywhere — more
stable training (it converges even without batch normalization).

## The objective (a-b-c coding)

`a` = label for fake, `b` = label for real (discriminator targets); `c` = value the generator wants the
discriminator to output on fakes. The discriminator has **no sigmoid** — a plain scalar output
regressed by least squares:

    min_D V(D) = ½ E_{x~p_data}[(D(x) − b)²] + ½ E_{z~p_z}[(D(G(z)) − a)²]
    min_G V(G) = ½ E_{z~p_z}[(D(G(z)) − c)²]

## Relation to the Pearson χ² divergence

Add a generator-parameter-free term `½ E_{x~p_data}[(D(x) − c)²]` to `V(G)` (does not change the
optimum). The optimal discriminator for fixed `G` (minimize `p_d(D−b)² + p_g(D−a)²` pointwise) is

    D*(x) = (b·p_d(x) + a·p_g(x)) / (p_d(x) + p_g(x)).

Substituting and combining the two expectations:

    2C(G) = ∫ ((b−c)p_d + (a−c)p_g)² / (p_d + p_g) dx,    with numerator = (b−c)(p_d+p_g) − (b−a)p_g.

Setting `b − c = 1` and `b − a = 2` makes the numerator `(p_d+p_g) − 2p_g`, so

    2C(G) = ∫ (2p_g − (p_d + p_g))² / (p_d + p_g) dx = χ²_Pearson(p_d + p_g ‖ 2p_g).

So minimizing the least-squares objective minimizes a Pearson χ² divergence between `p_d + p_g` and
`2p_g`.

## Parameter choices

- **Pearson scheme** (`b−c=1, b−a=2`): `a = −1, b = 1, c = 0` →
  `min_D ½E_{p_d}[(D−1)²] + ½E_z[(D(G)+1)²]`, `min_G ½E_z[(D(G))²]`.
- **0-1 scheme** (`c = b`, "make fakes as real as possible"): `a = 0, b = 1, c = 1` →
  `min_D ½E_{p_d}[(D−1)²] + ½E_z[(D(G))²]`, `min_G ½E_z[(D(G)−1)²]`. Used in practice; similar
  performance to the Pearson scheme.

## Architecture

Built on the stable convolutional recipe: fractionally-strided convolutional generator (ReLU, batch
norm, tanh output); strided convolutional discriminator (leaky-ReLU), but its **final layer is a plain
linear scalar, not a sigmoid**. For high-resolution scenes the generator is deepened (two extra
stride-1 deconvolutional layers after the top two upsampling layers, VGG-motivated). For many-class
data (thousands of classes), condition both networks on a label compressed by a **linear mapping**
`Φ(y)` (one-hot over thousands of classes is infeasible to concatenate directly):

    min_D ½E_{p_d}[(D(x|Φ(y))−1)²] + ½E_z[(D(G(z)|Φ(y)))²],   min_G ½E_z[(D(G(z)|Φ(y))−1)²].

Adam, `β1 = 0.5`.

## Working code (0-1 scheme)

```python
import torch
import torch.nn as nn

latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, z):
        out = self.l1(z).view(z.size(0), 128, self.init_size, self.init_size)
        return self.conv(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o, bn=True):
            b = [nn.Conv2d(i, o, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn: b.append(nn.BatchNorm2d(o, 0.8))
            return b
        self.features = nn.Sequential(*block(1, 16, bn=False), *block(16, 32),
                                      *block(32, 64), *block(64, 128))
        ds = 32 // (2 ** 4)
        self.out = nn.Linear(128 * ds ** 2, 1)        # no sigmoid: least squares on raw scalar
    def forward(self, x):
        return self.out(self.features(x).view(x.size(0), -1))

adversarial_loss = nn.MSELoss()                       # the least-squares loss

G, D = Generator(), Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

def train_step(real):
    b = real.size(0)
    valid, fake = torch.ones(b, 1), torch.zeros(b, 1)   # b=1 real, a=0 fake
    z = torch.randn(b, latent_dim)
    gen = G(z)
    opt_G.zero_grad()
    g_loss = adversarial_loss(D(gen), valid)            # c=1: regress fake score toward 1
    g_loss.backward(); opt_G.step()
    opt_D.zero_grad()
    d_loss = 0.5 * (adversarial_loss(D(real), valid) + adversarial_loss(D(gen.detach()), fake))
    d_loss.backward(); opt_D.step()
    return g_loss, d_loss
```

## Why it works

The single change — least squares instead of sigmoid cross-entropy — removes the half-line of dead
gradient. Far-but-correctly-classified fakes are now penalized and pulled toward the boundary (hence
toward the data manifold), improving quality; and gradient is available almost everywhere, improving
stability. The choice has a clean theoretical reading: with `b−c=1, b−a=2` the objective is the Pearson
χ² divergence between `p_d + p_g` and `2p_g`.
