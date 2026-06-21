The adversarial framework gives us something rare among generative models: an end-to-end differentiable generator trained against a discriminator, with no intractable partition function and no approximate inference. Yet in practice it produces images of limited quality and it trains unstably, and I am convinced the root of both problems is the objective rather than the architecture. In the standard game the discriminator is a *classifier* with a sigmoid output trained by binary cross-entropy, and the generator is updated to make that classifier call its fakes real. The trouble is geometric. Picture the discriminator's decision boundary in data space and consider the generator's fake samples that already lie on the correct, real side of the boundary but still sit *far* from where the real data actually lives. These are exactly the samples I most want to drag toward the real-data manifold — they fool the current discriminator without being any good. But ask what gradient sigmoid cross-entropy returns for them: as a function of the discriminator's pre-activation the loss saturates, so once a sample is confidently on the correct side the loss is flat and its gradient vanishes. The sigmoid is dead precisely in the regime where the generator still needs to move. Architectural recipes like the stable convolutional template, coarse-to-fine Laplacian pyramids, or feature matching leave this saturating objective untouched; the Wasserstein critic fixes stability but pays for it with many discriminator updates per generator step; energy-based discriminators change the discriminator's form without confronting the classifier-loss saturation directly. What we need is a discriminator loss that keeps handing the generator a useful gradient for samples that are correctly classified but far from the data — penalizing by *distance*, not by *side*.

I propose LSGAN, the Least Squares Generative Adversarial Network: replace the discriminator's sigmoid cross-entropy with a least-squares regression loss. The reasoning is about the shape of the loss as a function of the discriminator's output. Sigmoid cross-entropy is flat along an entire half-line — every confidently-correct value has zero slope — whereas a quadratic $(D-\text{target})^2$ is flat at exactly one point, its minimum, and everywhere else has a slope that grows with distance. So if the discriminator outputs a plain scalar and we regress that scalar toward a target with a squared loss, a fake whose output is far from the target still incurs a large loss and returns a large gradient even when it is on the correct side. Geometrically this is what fixes quality: when the generator is updated the discriminator is fixed, so the least-squares penalty pulls the fakes toward the value the discriminator assigns on the boundary, i.e. *toward the boundary itself*. And for the game to be learning at all the boundary must cross the manifold of real data — otherwise the discriminator's task is trivial and learning saturates — so pulling fakes toward the boundary is pulling them toward real data. The same "flat at one point" property gives the stability benefit: there is no half-line of dead gradient to fall into, so the generator keeps getting signal even in stressed conditions, such as networks with batch normalization removed, where the sigmoid game collapses.

To make this precise and to avoid hardcoding the target values, I write the objective with an $a$-$b$-$c$ coding scheme: $a$ is the label the discriminator regresses fakes toward, $b$ the label it regresses real data toward, and $c$ the value the generator wants the discriminator to produce on its fakes. The discriminator has no sigmoid; it emits a raw scalar that least squares regresses to these labels:
$$\min_D V(D) = \tfrac{1}{2}\,\mathbb{E}_{x\sim p_{\text{data}}}\big[(D(x) - b)^2\big] + \tfrac{1}{2}\,\mathbb{E}_{z\sim p_z}\big[(D(G(z)) - a)^2\big],$$
$$\min_G V(G) = \tfrac{1}{2}\,\mathbb{E}_{z\sim p_z}\big[(D(G(z)) - c)^2\big].$$
"It gives more gradient" is only a hand-wave until I know what divergence this actually minimizes, the way the original game provably minimizes Jensen–Shannon, so I derive it. First add to $V(G)$ a term $\tfrac{1}{2}\,\mathbb{E}_{x\sim p_{\text{data}}}[(D(x)-c)^2]$ that contains no parameters of $G$; it leaves the generator's optimum unchanged but symmetrizes the two expectations under a common form. For fixed $G$ the discriminator objective, as an integral over $x$, has integrand $p_d(x)(D(x)-b)^2 + p_g(x)(D(x)-a)^2$, which I minimize pointwise by differentiating with respect to the scalar $D(x)$:
$$2\,p_d(x)(D(x)-b) + 2\,p_g(x)(D(x)-a) = 0 \;\Rightarrow\; D^\*(x) = \frac{b\,p_d(x) + a\,p_g(x)}{p_d(x) + p_g(x)},$$
a genuine minimum since the second derivative $2(p_d+p_g)>0$. Substituting $D^\*$ into the symmetrized generator value $C(G)$ and using $D^\*(x)-c = \big((b-c)p_d + (a-c)p_g\big)/(p_d+p_g)$, the two expectations combine because the common denominator $(p_d+p_g)$ cancels against the weighting:
$$2C(G) = \int \frac{\big((b-c)\,p_d(x) + (a-c)\,p_g(x)\big)^2}{p_d(x) + p_g(x)}\,dx.$$
Now I want this to be a recognizable divergence. Rewrite the numerator $N = (b-c)p_d + (a-c)p_g = (b-c)(p_d+p_g) - (b-a)p_g$, which exposes two free knobs: the coefficient $(b-c)$ on $(p_d+p_g)$ and the coefficient $(b-a)$ on $p_g$. Choosing $b-c = 1$ and $b-a = 2$ makes $N = (p_d+p_g) - 2p_g = p_d - p_g$, so
$$2C(G) = \int \frac{\big(2p_g(x) - (p_d(x)+p_g(x))\big)^2}{p_d(x)+p_g(x)}\,dx,$$
which is exactly the Pearson $\chi^2$ divergence between $p_d + p_g$ and $2p_g$. The least-squares game is therefore f-divergence minimization, with a different divergence than the original game's Jensen–Shannon, and the hand-wave is gone.

Two concrete codings satisfy the requirements. The Pearson scheme takes $a=-1,\,b=1,\,c=0$ — real labelled $+1$, fake $-1$, generator pretending $0$ at the boundary — giving $\min_D \tfrac{1}{2}\mathbb{E}_{p_d}[(D-1)^2] + \tfrac{1}{2}\mathbb{E}_z[(D(G)+1)^2]$ and $\min_G \tfrac{1}{2}\mathbb{E}_z[D(G)^2]$. The more pragmatic $0$-$1$ scheme instead sets $c=b$, making the generator try to produce fakes the discriminator scores *exactly like real data*: $a=0,\,b=1,\,c=1$, so the discriminator regresses real toward $1$ and fake toward $0$ while the generator regresses its fake's score toward $1$. The two behave about the same in practice, and I use the $0$-$1$ scheme for training because it reads cleanest. For the networks I build on the stable convolutional recipe — a fractionally-strided convolutional generator with batch norm, ReLU, and a tanh output, and a strided convolutional discriminator with leaky-ReLU — with the one decisive change that the discriminator's final layer is a plain linear scalar, not a sigmoid, so the least-squares loss regresses it to the labels. For higher-resolution scenes I deepen the generator with two stride-1 deconvolutional layers after the top two upsampling layers, in the spirit of very-deep small-kernel designs, to give it more capacity to refine before output, and I train with Adam at the low momentum $\beta_1 = 0.5$ that keeps a moving adversarial game tracking its opponent. One further failure mode deserves a fix: a dataset with thousands of classes (handwritten characters, say) cannot be learned by the unconditional game, because the input noise carries no class information and a feed-forward network can only represent a *deterministic* input-to-output map — "one input region, many class outputs" has no deterministic functional form. Conditioning both networks on the class label restores determinism, but a one-hot vector over thousands of classes is far too large to concatenate everywhere, so I first pass the label through a small linear mapping $\Phi(y)$ that compresses it into a dense vector and concatenate that, giving the conditional least-squares game $\min_D \tfrac{1}{2}\mathbb{E}_{p_d}[(D(x\mid\Phi(y))-1)^2] + \tfrac{1}{2}\mathbb{E}_z[D(G(z)\mid\Phi(y))^2]$ and $\min_G \tfrac{1}{2}\mathbb{E}_z[(D(G(z)\mid\Phi(y))-1)^2]$.

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
