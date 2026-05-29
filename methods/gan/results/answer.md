# GAN — the adversarial generative framework

## The problem it solves

Estimate a deep generative model and sample from it using only backpropagation and forward
propagation — with no intractable partition function, no Markov chain in training or sampling, and
no approximate-inference network. The generative-model landscape is organized around how you get a
tractable handle on `p(x)`: make the density explicit-and-tractable (autoregressive models — exact
but sequential, slow sampling), explicit-up-to-something (energy-based models with an intractable
normalizer `Z`; variational autoencoders with a likelihood *bound* and an inference network), or
give up the density and only *sample* (implicit models). Every explicit route pays a tax — `Z` and
MCMC mixing, an analytic unnormalized density, a variational bound plus encoder. GAN takes the
implicit branch on purpose and supplies the one missing piece an implicit model lacked: a learning
signal that isn't a likelihood. That signal is a *learned adversary*.

## The key idea

Pit two networks against each other.

- A **generator** `G(z; θ_g)` maps noise `z ~ p_z` to a sample. It implicitly defines a model
  distribution `p_g` (the pushforward of `p_z` through `G`); its density is never written down.
  Because `G` is a *differentiable deterministic function of injected noise*, gradients flow through
  it by backprop — the only stochasticity is `z` at the input.
- A **discriminator** `D(x; θ_d) ∈ (0,1)` estimates the probability that `x` is real data rather than
  a `G`-sample.

`D` is trained to classify real vs. generated correctly; `G` is trained to make `D` wrong. Two design
choices carry the whole idea. The contrast is a *learned generator* rather than fixed noise (as in
NCE) or a fixed statistic (as in MMD), so the classification task keeps getting *harder* as `G`
improves and the learning signal never goes slack. And the opponent is a *classifier* rather than a
direct density-ratio estimator, so its sigmoid output `p_data/(p_data + p_g)` is the ratio squashed
into `(0,1)` — bounded and stable, where the raw ratio `p_data/p_g ∈ (0, ∞)` would have to be clipped.

## The objective

A two-player minimax game on one value function — exactly the (negative) binary cross-entropy of the
optimal real-vs-fake classifier:

    min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))].

**Optimal discriminator (G fixed).** Rewrite the second term as an expectation under `p_g` (pushforward),
so `V = ∫ [p_data log D + p_g log(1 - D)] dx`. Pointwise this maximizes `a log y + b log(1-y)` with
`a = p_data`, `b = p_g` (concave; derivative `a/y - b/(1-y) = 0`), giving

    D*_G(x) = p_data(x) / (p_data(x) + p_g(x)),

the Bayes-optimal classifier with equal priors — literally the squashed density ratio.

**What G then minimizes.** Substitute `D*_G` and subtract `-log 4 = E_{p_data}[-log 2] + E_{p_g}[-log 2]`,
folding `log 2` into each log to form the two KLs to the mixture `m = (p_data + p_g)/2`:

    C(G) = -log 4 + KL(p_data ‖ m) + KL(p_g ‖ m)
         = -log 4 + 2·JSD(p_data ‖ p_g).

Since `JSD ≥ 0` with equality iff the distributions are equal, the **unique global optimum is
`p_g = p_data`**, at value `-log 4`, where `D ≡ 1/2`. The divergence is symmetric (because real-vs-fake
is symmetric) and finite even on disjoint supports — properties that came for free from the game, not
from a design choice. Viewing `V` as `U(p_g, D)`: it is linear hence convex in `p_g`, so `sup_D U` is
convex with that unique optimum, and the subgradient of the sup at the maximizing `D*` is a valid
descent direction — so small alternating steps with `D` kept near optimal converge `p_g → p_data`
(in distribution space; optimizing `θ_g` through an MLP is nonconvex, justified empirically).

## Two practical fixes

1. **k-step schedule.** Optimizing `D` to completion each step is prohibitive and overfits, so take
   `k` discriminator steps per generator step (`k = 1` works and is cheapest), keeping `D` *near*
   optimal as `G` moves slowly — the persistent-chain (SML/PCD) trick, with `D`'s parameters as the
   carried-over state. This also stops `G` from outrunning a stale `D` into a **collapse** (the
   "Helvetica" scenario: many `z` mapped to a few outputs that fool the current `D`, losing diversity).
2. **Non-saturating generator loss.** Early on `D(G(z)) ≈ 0`. With discriminator logit `a` and
   `D = σ(a)`, the minimax generator term has logit derivative
   `d/da log(1 - σ(a)) = -σ(a) = -D`, so the signal to `G` vanishes when `D` confidently rejects
   fakes. Train `G` to **maximize `log D(G(z))`** instead, equivalently minimize `-log D(G(z))`:
   `d/da[-log σ(a)] = -(1 - D) ≈ -1` when `D ≈ 0`. Same game equilibrium, much stronger early
   gradient. (The minimax analysis and Algorithm-1 box use the `log(1 - D(G(z)))` generator term; the
   implementation uses the non-saturating form to actually train.)

## Algorithm

```
for number of training iterations:
    for k steps:                                   # k = 1 in practice
        sample minibatch z^(1..m) ~ p_z,  x^(1..m) ~ p_data
        ascend  ∇_{θ_d}  (1/m) Σ [ log D(x^i) + log(1 - D(G(z^i))) ]
    sample minibatch z^(1..m) ~ p_z
    ascend  ∇_{θ_g}  (1/m) Σ  log D(G(z^i))        # non-saturating (impl.); minimax box: descend log(1 - D)
```

Updates use any gradient rule (the original used SGD with momentum). The generator uses rectifier +
sigmoid units; the discriminator uses maxout with dropout — maxout for clean piecewise-linear
gradients, dropout to keep a powerful `D` from overfitting the moving target. Quantitative evaluation
of these implicit models uses a Gaussian Parzen-window log-likelihood estimate on samples (bandwidth
`σ` cross-validated).

## Code

The objective reuses one binary-cross-entropy loss against hard targets: `BCE(D, target=1) = -log D`
and `BCE(D, target=0) = -log(1 - D)`. The two players' parameter sets are disjoint, so autodiff
produces the two gradients independently. This is an original-style MLP implementation: ReLU +
sigmoid generator, maxout + dropout discriminator, and SGD with momentum. `BCEWithLogitsLoss` is the
numerically stable version of the sigmoid-output cross-entropy used by the Pylearn2 cost; conceptually
`D(x) = sigmoid(a(x))`.

```python
import math
import torch
import torch.nn as nn

latent_dim, img_dim = 100, 28 * 28
k = 1


class Maxout(nn.Module):
    def __init__(self, in_features, out_features, pieces):
        super().__init__()
        self.out_features = out_features
        self.pieces = pieces
        self.linear = nn.Linear(in_features, out_features * pieces)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), self.out_features, self.pieces)
        return x.max(dim=2).values


class Generator(nn.Module):          # x = G(z): noise -> data; one forward pass, no Markov chain
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, img_dim),
            nn.Sigmoid(),               # MNIST-style [0,1] pixels; implicitly defines p_g
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):      # returns logit a(x); D(x) = sigmoid(a(x))
    def __init__(self):
        super().__init__()
        self.input_drop = nn.Dropout(p=0.5)  # Pylearn2 keep prob .5 on discriminator inputs
        self.h0 = Maxout(img_dim, 240, pieces=5)
        self.h0_drop = nn.Dropout(p=0.2)     # Pylearn2 keep prob .8 after h0
        self.h1 = Maxout(240, 240, pieces=5)
        self.out = nn.Linear(240, 1)

    def forward(self, x):
        x = self.input_drop(x)
        x = self.h0(x)
        x = self.h0_drop(x)
        x = self.h1(x)
        return self.out(x)


def sample_noise(m, device):
    # Uniform with variance 1, matching the original MNIST config's uniform noise.
    bound = math.sqrt(3.0)
    return torch.empty(m, latent_dim, device=device).uniform_(-bound, bound)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G, D = Generator().to(device), Discriminator().to(device)
bce = nn.BCEWithLogitsLoss()         # target 1: -log sigmoid(a); target 0: -log(1-sigmoid(a))
opt_G = torch.optim.SGD(G.parameters(), lr=0.1, momentum=0.5)
opt_D = torch.optim.SGD(D.parameters(), lr=0.1, momentum=0.5)

for real, _ in dataloader:           # real: minibatch x ~ p_data, assumed scaled to [0,1]
    real = real.to(device).view(real.size(0), -1)
    valid = torch.ones(real.size(0), 1, device=real.device)  # label "real"
    fake = torch.zeros(real.size(0), 1, device=real.device)  # label "fake"

    # ---- update D: ascend log D(x) + log(1 - D(G(z))) by minimizing BCE ----
    for _ in range(k):
        opt_D.zero_grad()
        z = sample_noise(real.size(0), real.device)
        gen = G(z).detach()
        real_loss = bce(D(real), valid)              # -log D(x)
        fake_loss = bce(D(gen), fake)                # -log(1 - D(G(z)))
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        opt_D.step()

    # ---- update G: non-saturating, maximize log D(G(z)) ----
    opt_G.zero_grad()
    z = sample_noise(real.size(0), real.device)
    gen = G(z)
    g_loss = bce(D(gen), valid)        # target 1 on fakes -> minimize -log D(G(z))
    g_loss.backward()
    opt_G.step()
```

The Pylearn2 `AdversaryCost2` mapping is the same: it builds
`d_obj = 0.5 * (D.last_layer.cost(ones, D(X)) + D.last_layer.cost(zeros, D(G(z))))` and the
non-saturating `g_obj = D.last_layer.cost(ones, D(G(z)))`, then returns `T.grad(d_obj, θ_d)` and
`T.grad(g_obj, θ_g)` separately, gating the generator update once per `k` discriminator updates. Note
the discrepancy worth flagging: the Algorithm-1 box descends the minimax generator term
`log(1 - D(G(z)))`, but the released code maximizes `log D(G(z))` (the non-saturating form) — same
fixed point, healthier early gradient, and it is the non-saturating form that actually trains.

## What it buys and costs

- **Buys:** no Markov chain (a sample is one forward pass through `G`); no inference network in
  training; gradient is pure backprop through `D` into `G`; piecewise-linear units usable freely
  (no generation-time feedback loop); `G` never copies data directly (it only sees data through `D`'s
  gradient, so it can overfit only if `D` does, which is easy to control); can represent sharp /
  degenerate distributions that MCMC methods cannot.
- **Costs:** no explicit `p_g(x)` (likelihood must be estimated indirectly, e.g. Parzen window), and
  `D` must be kept synchronized with `G` to avoid the Helvetica-style collapse.
