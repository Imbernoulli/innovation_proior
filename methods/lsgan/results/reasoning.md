Let me look hard at one specific failure in adversarial training, because I think it points at the objective rather than the architecture, and if I'm right the fix is a one-line change with a clean theory behind it. The setup: a generator `G` maps noise to images, and a discriminator `D` is a classifier — a network with a sigmoid output — trained with binary cross-entropy to call real images real and fakes fake. The generator is then updated to make `D` classify its fakes as real. Everyone uses this. And it's unstable, and the images plateau in quality. Why?

Draw the discriminator's decision boundary in data space and watch what happens to a generator update. Some of the generator's current fake samples land *on the correct side* of the boundary — `D` already thinks they're real — but they sit *far* from where the real data actually is. Intuitively these are exactly the samples I most want to fix: they fool the current discriminator but they're not close to real data, so I want to drag them toward the real data. Now ask what gradient the sigmoid cross-entropy loss gives me for those samples. The loss, as a function of the discriminator's pre-activation, saturates: once a sample is confidently on the correct side, the loss is essentially flat, so its gradient is essentially zero. So the generator gets *almost no signal* from precisely the samples it most needs to move. The sigmoid is flat exactly where I still need to push. That's the vanishing-gradient problem, and it's baked into the choice of loss, not the choice of network.

So I want a discriminator loss that *keeps* penalizing a fake sample according to *how far* it is from where the discriminator wants it, even when that sample is already on the correct side. "Penalize by distance, not by side." What's the simplest loss with that property? A least-squares loss. Think about the shape of the two losses as functions of the discriminator's output value. Sigmoid cross-entropy is flat for all large correct-side values — a whole half-line of zero gradient. A quadratic `(D − target)²` is flat at *exactly one point*, its minimum; everywhere else it has a nonzero slope that grows with distance. So if I make the discriminator regress its output toward a target with a squared loss, then a fake whose output is far from the target — even on the "correct" side — still incurs a large loss and hands back a large gradient. The least-squares loss penalizes a sample that lies a long way on the correct side of the boundary, which sigmoid cross-entropy refuses to do.

What does that buy the generator geometrically? When I update `G`, the discriminator is fixed, so the boundary is fixed. The least-squares penalty pulls the generator's fakes toward the value the discriminator assigns *on the boundary* — i.e. it pulls the fake samples *toward the decision boundary*. And here's the connecting fact: for the adversarial game to be learning at all, the decision boundary has to cross the manifold of real data (if it didn't, the discriminator would have a trivial job and learning would saturate). So pulling fakes toward the boundary is, geometrically, pulling them toward the real-data manifold. That's the quality benefit. And because the loss is flat at only one point, it generates gradient almost everywhere, which is the stability benefit — there's no half-line of dead gradient to fall into.

Let me write the objective generally, because I don't want to hardcode the label values yet — I want to see what freedom I have. Use an `a`-`b`-`c` coding scheme: let `a` be the label the discriminator regresses fakes toward, `b` the label it regresses real data toward, and `c` the value the *generator* wants the discriminator to output on its fakes (the value `G` pretends is "real"). Then

    min_D V(D) = ½ E_{x~p_data}[(D(x) − b)²] + ½ E_{z~p_z}[(D(G(z)) − a)²],
    min_G V(G) = ½ E_{z~p_z}[(D(G(z)) − c)²].

Notice the discriminator no longer has a sigmoid — it outputs a real scalar, and least squares regresses that scalar to the labels. That's the whole method: swap the classifier loss for a regression loss. But I have three free constants `a, b, c`, and "it gives more gradient" is a hand-wave. I want to know what divergence this objective actually minimizes, the way the original game provably minimizes Jensen–Shannon. So let me derive it.

To make the algebra clean I'll work with a slightly extended generator objective. Add a term that depends only on the real data to `V(G)`:

    min_G V(G) = ½ E_{x~p_data}[(D(x) − c)²] + ½ E_{z~p_z}[(D(G(z)) − c)²].

This is legitimate: the added term `E_{x~p_data}[(D(x)−c)²]` contains no parameters of `G`, so it doesn't change the generator's optimum — it just symmetrizes the expression so I can collect both expectations under a common form.

First, the optimal discriminator for a fixed `G`. The discriminator objective, written as an integral over `x`, has the integrand `p_data(x)(D(x)−b)² + p_g(x)(D(x)−a)²`. Minimize pointwise: for each `x`, differentiate with respect to the scalar `D(x)`,

    2 p_data(x)(D(x) − b) + 2 p_g(x)(D(x) − a) = 0
    ⇒ D(x)(p_data(x) + p_g(x)) = b p_data(x) + a p_g(x)
    ⇒ D*(x) = (b p_data(x) + a p_g(x)) / (p_data(x) + p_g(x)).

(It's a minimum: the second derivative `2(p_data + p_g) > 0`.) Write `p_d` for `p_data` from here. Now substitute `D*` into the (symmetrized) generator objective. Call the value `C(G)`; then

    2C(G) = E_{x~p_d}[(D*(x) − c)²] + E_{x~p_g}[(D*(x) − c)²].

Compute `D*(x) − c`:

    D*(x) − c = (b p_d + a p_g)/(p_d + p_g) − c = ((b − c) p_d + (a − c) p_g) / (p_d + p_g).

Let `N(x) = (b − c) p_d(x) + (a − c) p_g(x)` be the numerator. Then the two expectations combine:

    2C(G) = ∫ p_d · N²/(p_d + p_g)² dx + ∫ p_g · N²/(p_d + p_g)² dx
          = ∫ (p_d + p_g) · N²/(p_d + p_g)² dx
          = ∫ N² / (p_d + p_g) dx
          = ∫ ((b − c) p_d(x) + (a − c) p_g(x))² / (p_d(x) + p_g(x)) dx.

Now I want this to be a recognizable divergence, and a divergence between two distributions should compare something to `p_d + p_g` or similar. Rewrite the numerator by pulling out `(b − c)(p_d + p_g)`:

    N = (b − c) p_d + (a − c) p_g = (b − c)(p_d + p_g) − (b − a) p_g,

which checks out: `(b−c)(p_d + p_g) − (b−a)p_g = (b−c)p_d + (b−c)p_g − (b−a)p_g = (b−c)p_d + (a−c)p_g`. Good. So I have two knobs in `N`: the coefficient `(b − c)` on `(p_d + p_g)` and the coefficient `(b − a)` on `p_g`. Choose them to make `N` a *difference* of two distribution-like quantities. Set `b − c = 1` and `b − a = 2`. Then

    N = (p_d + p_g) − 2 p_g = p_d − p_g,    i.e. N = −(2 p_g − (p_d + p_g)),

and therefore

    2C(G) = ∫ (2 p_g(x) − (p_d(x) + p_g(x)))² / (p_d(x) + p_g(x)) dx.

That is the Pearson `χ²` divergence between `p_d + p_g` and `2 p_g` — the quantity `∫ (P − Q)²/Q` form with the two distribution-like arguments `(p_d + p_g)` and `2 p_g`. So minimizing the least-squares objective, with labels chosen so that `b − c = 1` and `b − a = 2`, *provably minimizes a Pearson `χ²` divergence* between `p_d + p_g` and `2 p_g`. The hand-wave is gone: the least-squares game is f-divergence minimization, with a different divergence than the original game's Jensen–Shannon.

Now pick concrete constants. One family satisfies the `χ²` conditions `b − c = 1`, `b − a = 2`. The natural choice is `a = −1`, `b = 1`, `c = 0` (real labelled `+1`, fake labelled `−1`, generator pretending `0` — the boundary value). That gives

    min_D V(D) = ½ E_{x~p_d}[(D(x) − 1)²] + ½ E_{z}[(D(G(z)) + 1)²],
    min_G V(G) = ½ E_{z}[(D(G(z)))²].

There's a second, more pragmatic choice. Forget about matching the `χ²` conditions and instead set `c = b`: make the generator try to produce fakes that the discriminator scores *exactly like real data*. With the simple `0`-`1` coding `a = 0`, `b = 1`, `c = 1`,

    min_D V(D) = ½ E_{x~p_d}[(D(x) − 1)²] + ½ E_{z}[(D(G(z)))²],
    min_G V(G) = ½ E_{z}[(D(G(z)) − 1)²].

In practice these two schemes behave about the same, so I'll use the `0`-`1` one for training — it's the cleanest to read: discriminator regresses real to `1` and fake to `0`, generator regresses its fake's score to `1`.

A consequence worth checking is the stability claim under stress. Adversarial training is fragile, and one stress test removes batch normalization and sees whether learning still converges. Because the least-squares loss supplies gradient almost everywhere (flat at one point only), the generator keeps getting a useful signal even in regimes where the sigmoid would have died, so I'd expect this objective to converge in conditions — like no batch normalization — that break the sigmoid game. That's the second benefit, and it falls straight out of the same "flat at one point" property that gave the quality benefit.

For the networks I'll build on the stable convolutional recipe rather than reinvent it: a fractionally-strided convolutional generator with ReLU and batch norm and a tanh output, and a strided convolutional discriminator with leaky-ReLU — except the discriminator's final layer is *not* a sigmoid; it's a plain linear output that the least-squares loss regresses to the labels. For higher-resolution scenes I deepen the generator slightly, motivated by the very-deep-small-kernel design, by adding two stride-1 deconvolutional layers after the top two upsampling layers, so the generator has more capacity to refine before output. Adam with the low momentum `β1 = 0.5` that keeps a moving adversarial game tracking its opponent.

There's one more case I should handle, because it surfaces a real failure mode: generating images for a dataset with a huge number of classes — thousands, say handwritten characters. Training the unconditional game on many classes at once doesn't produce readable outputs, and the reason is structural: the input noise carries no class information, so the network is asked to map one input region to many different class-outputs, but a feed-forward network represents a *deterministic* input→output map, and "one input, many possible class outputs" has no deterministic functional form to learn. The fix is to condition both networks on the class label, which restores a deterministic relation: given the label, the output class is fixed. But a one-hot label over thousands of classes is enormous and concatenating it everywhere is infeasible in memory and compute. So pass the label through a small *linear mapping* `Φ(y)` that compresses the huge one-hot vector into a small dense vector first, and concatenate *that* into the network layers. The conditional objective is just the least-squares game conditioned on `Φ(y)`:

    min_D V(D) = ½ E_{x~p_d}[(D(x | Φ(y)) − 1)²] + ½ E_{z}[(D(G(z) | Φ(y)))²],
    min_G V(G) = ½ E_{z}[(D(G(z) | Φ(y)) − 1)²].

Let me write the core least-squares game as real code, using the `0`-`1` scheme.

```python
import torch
import torch.nn as nn

latent_dim = 100

class Generator(nn.Module):
    # fractionally-strided conv stack, BN, ReLU/LeakyReLU, tanh output (DCGAN-style backbone)
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
        self.out = nn.Linear(128 * ds ** 2, 1)     # NO sigmoid: a plain scalar regressed by least squares
    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        return self.out(f)

# the one-line change: least-squares (regression) loss instead of sigmoid cross-entropy
adversarial_loss = nn.MSELoss()

G, D = Generator(), Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

def train_step(real):
    b = real.size(0)
    valid = torch.ones(b, 1)     # label b = 1 for real
    fake  = torch.zeros(b, 1)    # label a = 0 for fake
    z = torch.randn(b, latent_dim)
    gen = G(z)
    # --- G: regress D's score on fakes toward c = 1 (0-1 scheme) ---
    opt_G.zero_grad()
    g_loss = adversarial_loss(D(gen), valid)
    g_loss.backward(); opt_G.step()
    # --- D: regress real toward 1, fake toward 0; the 1/2 is folded into the 0.5 averaging ---
    opt_D.zero_grad()
    real_loss = adversarial_loss(D(real), valid)
    fake_loss = adversarial_loss(D(gen.detach()), fake)
    d_loss = 0.5 * (real_loss + fake_loss)
    d_loss.backward(); opt_D.step()
    return g_loss, d_loss
```

The causal chain: the sigmoid cross-entropy discriminator loss saturates for fakes that are correctly classified yet far from the data, so the generator gets a vanishing gradient exactly where it most needs to move; replacing it with a least-squares (regression) loss — flat at one point instead of along a half-line — penalizes those far-but-correct fakes and pulls them toward the decision boundary, which (since the boundary crosses the data manifold) pulls them toward real data, giving both better samples and a non-vanishing gradient for stability; writing the loss with general labels `a, b, c` and substituting the optimal discriminator shows the objective is `∫ ((b−c)p_d + (a−c)p_g)²/(p_d+p_g)`, which under `b−c=1, b−a=2` is exactly the Pearson `χ²` divergence between `p_d + p_g` and `2p_g`; concretely I use the `0`-`1` scheme with a sigmoid-free discriminator regressed by mean-squared error, and for many-class data I condition on a linearly-compressed label. What I'd want to validate next: that against the sigmoid game this produces higher-quality scene images and that it still converges under the batch-normalization-removal stress test where the sigmoid game collapses.
