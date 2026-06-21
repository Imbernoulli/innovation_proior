The goal is to learn a distribution over high-dimensional data — concretely, to generate natural images — by training an implicit sampler $g_\theta: Z \to X$ that pushes a fixed noise prior $z \sim p(z)$ to a model law $P_\theta$, and adjusting $\theta$ so $P_\theta$ matches the real data law $P_r$. The honest difficulty lies in the geometry. Natural images do not fill the $64\times64\times3$ cube; they sit on a thin, curved, low-dimensional sheet inside it. A generator that maps a low-dimensional $z$ through a smooth net also concentrates its mass on a low-dimensional sheet. Two such sheets in general position either miss each other or cross on a set of measure zero, so $P_\theta$ puts no mass where $P_r$ lives and vice versa: the Radon–Nikodym derivative does not exist, $\mathrm{KL}(P_r\|P_\theta)=+\infty$, and the log-likelihood is $-\infty$. Maximum likelihood is literally undefined for the object I actually have. The standard escape — convolving the model with Gaussian noise so a density exists everywhere — needs noise on the order of $\sigma \approx 0.1$ per pixel on $[0,1]$ images, large enough to visibly blur every sample, and practitioners quietly drop it when displaying results. That tells me the noise is not a modeling choice but a crutch to rescue the wrong distance. The right move is to keep the implicit sampler, which represents manifold-supported distributions natively, and fix the distance instead.

So the only real question is which distance $\rho(P_\theta, P_r)$ to descend. The choice is not cosmetic: it decides whether the loss even has a gradient. A distance defines which sequences converge, and a weaker distance calls more sequences convergent; what I need is for $\theta \mapsto \rho(P_\theta, P_r)$ to be continuous and almost-everywhere differentiable, so I should be hunting for the *weakest* useful distance, not the strongest. The standard ones fail exactly in the non-overlapping regime the manifold geometry forces. On the simplest toy — $Z\sim U[0,1]$, $P_0$ the law of $(0,Z)$ and $P_\theta$ the law of $(\theta,Z)$, two parallel unit segments — total variation is $1$ for every $\theta\neq0$ and jumps to $0$ at $\theta=0$ (discontinuous); both directions of $\mathrm{KL}$ are $+\infty$ for $\theta\neq0$; and Jensen–Shannon equals the constant $\log 2$ on a punctured neighborhood of $0$, dropping to $0$ only at $\theta=0$, so it is flat with zero gradient everywhere. For the ordinary GAN value function the inner-optimal discriminator is $D^*(x)=P_r(x)/(P_r(x)+P_g(x))$ and substituting it back leaves the generator minimizing $2\,\mathrm{JS}(P_r,P_g)-2\log 2$, which inherits exactly this flatness: a perfect discriminator gives a vanishing generator gradient, forcing a miserable tightrope where $D$ must be kept deliberately weak. The $-\log D$ variant only swaps the disease for another — its inner-optimal form is $\nabla_\theta[\mathrm{KL}(P_g\|P_r)-2\,\mathrm{JS}(P_g\|P_r)]$, where the JS term has the wrong sign and the reverse KL charges almost nothing for dropping modes, matching observed mode collapse, while the singular density ratio gives exploding gradient variance. Switching to any other $f$-divergence cannot help, because every $f$-divergence is a functional of the density ratio $dP_r/dP_g$, which is ill-defined precisely when the supports do not overlap. The fix has to change the *geometry* of the comparison, not the divergence label.

I propose the Wasserstein GAN (WGAN): minimize the Earth-Mover / Wasserstein-1 distance
$$W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r,P_g)} \mathbb{E}_{(x,y)\sim\gamma}\,\|x-y\|,$$
the minimal cost of transporting the mass of $P_g$ into the shape of $P_r$, where $\Pi$ is the set of couplings with the right marginals. Because $W$ measures transport cost using the metric on $X$ rather than a pointwise density ratio, it stays continuous and a.e. differentiable even across non-overlapping supports — on the parallel-segments toy the optimal plan slides each $(0,z)$ straight to $(\theta,z)$, so $W=|\theta|$ with gradient $\mathrm{sign}(\theta)$, a clean nonzero signal exactly where JS, KL, and TV all died. This is not a 1-D artifact. For any generator continuous in $\theta$ I can bound $W$ by exhibiting one coupling rather than optimizing over all of them: use the same noise for both, $\gamma = \mathrm{law}\,(g_\theta(Z), g_{\theta'}(Z))$, whose marginals are correct, giving $W(P_\theta,P_{\theta'}) \le \mathbb{E}_z\|g_\theta(z)-g_{\theta'}(z)\|$. On a compact $X$ the integrand is bounded and tends to $0$ pointwise as $\theta'\to\theta$, so by bounded convergence the bound vanishes; the reverse triangle inequality $|W(P_r,P_\theta)-W(P_r,P_{\theta'})|\le W(P_\theta,P_{\theta'})$ then makes the loss continuous. Strengthening "continuous" to "locally Lipschitz" — assuming $\mathbb{E}_z[L(\theta,z)]<\infty$, which holds for any feedforward net with Lipschitz nonlinearities whenever $\mathbb{E}\|z\|<\infty$, since the gradient norm is bounded by $C_1(\theta)+C_2(\theta)\|z\|$ — and invoking Rademacher's theorem gives differentiability almost everywhere, exactly the regularity gradient descent needs. And $W$ is the weakest of the standard distances: $\mathrm{KL}\Rightarrow\{\mathrm{JS},\mathrm{TV}\}\Rightarrow W$ strictly (via Pinsker, $\delta\le\sqrt{2\,\mathrm{JS}}$, and $W\le B\,\delta$ on a space of diameter $B$), so it yields a usable loss in strictly more situations.

The primal inf over couplings is intractable, but Wasserstein-1 is a linear transport program, and its Kantorovich–Rubinstein dual collapses the search over joint distributions into a search over a single scalar function:
$$W(P_r, P_g) = \sup_{\|f\|_L \le 1} \mathbb{E}_{x\sim P_r}[f(x)] - \mathbb{E}_{x\sim P_g}[f(x)].$$
Dualizing the marginal constraints with a potential $f$ turns the per-unit transport cost $\|x-y\|$ into the requirement $f(x)-f(y)\le\|x-y\|$, i.e. $f$ is $1$-Lipschitz; relaxing to $K$-Lipschitz functions merely rescales the value to $K\cdot W$. This is an integral probability metric $d_F(P,Q)=\sup_{f\in F}\mathbb{E}_P[f]-\mathbb{E}_Q[f]$ with $F$ the Lipschitz ball — the choice of $F$ is the whole story, and the Lipschitz one is the sweet spot whose IPM is the weak distance $W$ (bounded-in-$[-1,1]$ functions give $2\,\mathrm{TV}$, the strong topology; an RKHS ball gives MMD, closed-form but $O(\text{samples}^2)$ and unreliable in high dimensions). I realize the sup with a neural network $f_w$ that maximizes $\mathbb{E}[f_w(\text{real})]-\mathbb{E}[f_w(\text{fake})]$ by gradient ascent, a lower estimate of $K\cdot W$ that improves with capacity and optimization. This network is a *critic*, not a discriminator: no sigmoid, no log, no probability — it outputs a raw real score and I read off the difference of its means, the dual potential. By the envelope theorem, holding the optimal critic fixed while differentiating the value through $\theta$ (which enters only the fake term),
$$\nabla_\theta W(P_r, P_\theta) = -\,\mathbb{E}_z\big[\nabla_\theta f_w(g_\theta(z))\big],$$
so the generator update simply backprops the critic's score through $g_\theta$ and negates it — minimize $-\mathbb{E}_z[f_w(g_\theta(z))]$, which raises the critic's score on its own fakes.

What makes this work is that the failure mode of GANs inverts. The dual sup is only meaningful if $f_w$ is constrained, since an unconstrained net can scale its output without bound and send the sup to $+\infty$; I must keep $f_w$ inside the $K$-Lipschitz ball. A net's Lipschitz constant is controlled by the operator norms of its weight matrices, so the crudest sufficient enforcement is weight clipping: after each critic update, clamp every weight into a small box $[-c,c]$ with $c=0.01$. The parameter space $[-c,c]^l$ is compact, so all $f_w$ share one Lipschitz constant $K$. Clipping is a blunt instrument — too large a $c$ and the weights take forever to reach the box boundary, leaving the constraint loose so the critic cannot be driven to optimality; too small and the products of small weights make gradients vanish through depth — so $c$ lives in a narrow band, but it is simple and exposes everything else. On two separated Gaussians the constraint bites usefully: a GAN discriminator snaps to near $0/1$ and saturates flat, while the clipped critic, capped to grow at most linearly, settles into a smooth roughly-linear ramp between the modes, manufacturing exactly the gentle nonzero slope the generator can descend. Because $W$ is differentiable a.e., training the critic *toward* optimality strengthens the generator gradient rather than killing it, so I lean into many critic steps per generator step instead of balancing a weak discriminator — and re-optimizing the critic every step removes the stale fixed target whose argmax a generator would collapse onto, attacking the mode-collapse mechanism directly. The quantity $\mathbb{E}[f(\text{real})]-\mathbb{E}[f(\text{fake})]$ now estimates $W$ up to the constant $K$, so the training curve finally tracks sample quality. The optimizer choice follows the same logic: the critic loss is highly non-stationary, momentum optimizers like Adam diverged on it (the Adam step turned anti-parallel to the gradient), so I use RMSProp — running-magnitude rescaling, no momentum — with a small learning rate $5\times10^{-5}$. The DCGAN convolutional body is reused for both nets so the only change isolated is the objective and training discipline: a transpose-conv generator mapping a $100$-D Gaussian to a $64\times64\times3$ image, and the DCGAN discriminator body with its final sigmoid removed and output averaged to a scalar as the critic.

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
