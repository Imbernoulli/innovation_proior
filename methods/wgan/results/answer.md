# Wasserstein GAN (WGAN)

## Problem

Learn a distribution `P_theta` (the law of `g_theta(z)`, `z ~ p(z)`) to match a real data
distribution `P_r`, where both live on thin, low-dimensional manifolds in pixel space that
generically overlap only on a measure-zero set. In that regime density-ratio divergences
such as KL, Jensen–Shannon, and other f-divergences become flat, infinite, or singular, so
their gradients vanish or become unusable. This is the root cause of GAN training instability:
vanishing generator gradients when the discriminator is good, the need to balance the
discriminator against the generator, mode collapse, and the absence of any loss curve that
tracks sample quality. For the ordinary GAN value function, the optimal discriminator is
`D*(x)=P_r(x)/(P_r(x)+P_g(x))`, and substituting it back gives the generator objective
`2 JS(P_r,P_g) - 2 log 2`; on disjoint supports `JS=log 2` and `TV=1`, so this objective is
flat and supplies no descent direction.

## Key idea

Replace the JS objective with the **Earth-Mover / Wasserstein-1 distance**

```
W(P_r, P_g) = inf_{γ ∈ Π(P_r,P_g)}  E_{(x,y)~γ} ||x − y||,
```

the minimal cost of transporting the mass of `P_g` into `P_r`. Because it measures transport
cost using the metric on `X` rather than a pointwise density ratio, `W` stays continuous and
almost-everywhere differentiable in `theta` even when the supports do not overlap (on the
parallel-segments example `W = |theta|` with gradient `sign(theta)`, while `JS = log 2`,
`KL = +∞`, and `TV = 1` for `theta ≠ 0`). `W` is the weakest of the standard distances —
`KL ⇒ {JS, TV} ⇒ W` strictly — so it
yields a usable loss in strictly more situations.

The inf over couplings is intractable, so use **Kantorovich–Rubinstein duality**:

```
W(P_r, P_g) = sup_{||f||_L ≤ 1}  E_{x~P_r}[f(x)] − E_{x~P_g}[f(x)].
```

If the supremum ranged over all K-Lipschitz functions, it would equal `K·W`. In code,
parameterize `f` by a neural network `f_w` constrained to be K-Lipschitz, so maximizing
`E[f_w(real)] − E[f_w(fake)]` gives a lower estimate of `K·W` that improves with critic
capacity and optimization. For comparison, the IPM over all functions bounded in `[-1,1]`
is `2 TV`, not `TV`, under `TV(P,Q)=sup_A |P(A)-Q(A)|`. This network is a **critic**, not a
classifier: no sigmoid, no log, it outputs a raw real score, so its gradient to the
generator never saturates. By the envelope theorem the generator gradient is

```
∇_theta W(P_r, P_theta) = − E_z[ ∇_theta f_w(g_theta(z)) ],
```

i.e. minimize `-E_z[f_w(g_theta(z))]`, so the update raises the critic's score on fakes.

Enforce the Lipschitz constraint by **weight clipping**: after each critic update, clamp
every weight into a small box `[−c, c]` (`c = 0.01`). A compact weight box makes the whole
family uniformly K-Lipschitz. This is crude — too large a `c` makes the critic slow to
reach optimality, too small a `c` makes gradients vanish through depth — but it is simple and
exposes everything else.

Two consequences flip the GAN intuition. First, since `W` is differentiable a.e., training
the critic toward optimality *helps* the gradient instead of killing it, so use several
critic steps per generator step (and many during warmup) — no D/G balancing tightrope, and
no fixed discriminator for the generator to chase for long. Second, `E[f(real)] − E[f(fake)]`
estimates `W` up to the constant `K`, giving a meaningful training diagnostic instead of a
saturated classifier loss. Use RMSProp (no momentum): the critic loss is highly
non-stationary and momentum optimizers diverged on it.

## Algorithm

```
Require: α=5e-5 (learning rate), c=0.01 (clip), m=64 (batch), n_critic=5
while not converged:
    for t = 1 .. n_critic:                       # 100 for first 25 gen iters / every 500th
        sample real batch {x} ~ P_r, noise {z} ~ p(z)
        g_w  ← ∇_w [ (1/m) Σ f_w(x) − (1/m) Σ f_w(g_theta(z)) ]
        w    ← w + α · RMSProp(w, g_w)            # ascend the critic objective
        w    ← clip(w, −c, c)                     # Lipschitz constraint
    sample noise {z} ~ p(z)
    g_theta ← −∇_theta (1/m) Σ f_w(g_theta(z))
    theta   ← theta − α · RMSProp(theta, g_theta) # descend W
```

## Code

DCGAN convolutional bodies are reused for both generator and critic; the contribution is the
objective, Lipschitz constraint, and training loop.

```python
import torch
import torch.nn as nn
import torch.optim as optim

nz, ngf, ndf, nc = 100, 64, 64, 3      # latent dim, gen/critic widths, image channels
c           = 0.01                     # weight-clip box -> keeps the critic K-Lipschitz
n_critic    = 5                        # critic steps per generator step
lr          = 5e-5                     # small step; RMSProp, no momentum
batch_size  = 64

# Generator g_theta: DCGAN transpose-conv stack, z -> 64x64x3 image.
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co, k=4, s=2, p=1):
            return [nn.ConvTranspose2d(ci, co, k, s, p, bias=False),
                    nn.BatchNorm2d(co), nn.ReLU(True)]
        self.net = nn.Sequential(
            *block(nz, ngf*8, 4, 1, 0), *block(ngf*8, ngf*4),
            *block(ngf*4, ngf*2), *block(ngf*2, ngf),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())
    def forward(self, z):
        return self.net(z)

# Critic f_w: DCGAN conv body, NO final sigmoid -- it scores, it does not classify.
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co, bn=True):
            layers = [nn.Conv2d(ci, co, 4, 2, 1, bias=False)]
            if bn: layers.append(nn.BatchNorm2d(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.net = nn.Sequential(
            *block(nc, ndf, bn=False), *block(ndf, ndf*2),
            *block(ndf*2, ndf*4), *block(ndf*4, ndf*8),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False))     # no sigmoid
    def forward(self, x):
        return self.net(x).mean(0).view(1)                # scalar critic value

G, D = Generator(), Critic()
optD = optim.RMSprop(D.parameters(), lr=lr)               # RMSProp, not Adam
optG = optim.RMSprop(G.parameters(), lr=lr)

def enforce_constraint(net):
    for p in net.parameters():
        p.data.clamp_(-c, c)

def scoring_network_loss(net, real, fake):
    return net(fake) - net(real)             # minimize this == maximize E[f(real)] - E[f(fake)]

def generator_loss(net, fake):
    return -net(fake)                        # raise critic score on generated samples

def set_requires_grad(net, flag):
    for p in net.parameters():
        p.requires_grad = flag

def train_step(real, gen_iter):
    # Critic phase: estimate W by maximizing E[f(real)] - E[f(fake)].
    real = real[0] if isinstance(real, (list, tuple)) else real
    device = real.device
    set_requires_grad(D, True)
    iters = 100 if (gen_iter < 25 or gen_iter % 500 == 0) else n_critic
    lossD = None
    enforce_constraint(D)
    for _ in range(iters):
        z = torch.randn(real.size(0), nz, 1, 1, device=device)
        fake = G(z).detach()                     # critic phase: freeze generator
        lossD = scoring_network_loss(D, real, fake)
        optD.zero_grad(); lossD.backward(); optD.step()
        enforce_constraint(D)                    # Lipschitz constraint via weight clipping

    # Generator phase: descend W via grad_theta W = -E[grad_theta f(g(z))].
    set_requires_grad(D, False)
    z = torch.randn(real.size(0), nz, 1, 1, device=device)
    fake = G(z)
    lossG = generator_loss(D, fake)
    optG.zero_grad(); lossG.backward(); optG.step()
    # -lossD = E[f(real)] - E[f(fake)] estimates W up to scale K.
    return (-lossD).item()
```

## Why it works

- **Stable gradients:** the linear critic objective never saturates, so a near-optimal critic
  gives a stronger (not weaker) generator gradient — the opposite of the JS/log-D regime.
- **Less D/G balancing pressure:** the critic is re-optimized each step, so there is no stale
  fixed classifier target for the generator to chase for long.
- **A meaningful loss:** `E[f(real)] − E[f(fake)]` tracks the Wasserstein distance up to a
  constant, so the training curve is interpretable as a distance estimate.
- **Architecture-isolating:** the change is purely the objective and loop, with the DCGAN body
  left intact.
