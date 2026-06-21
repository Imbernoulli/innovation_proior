We can train a progressive-growing GAN today that produces gorgeous 1024×1024 faces with excellent FID, and yet the generator is a black box we cannot steer. Every visible property of the image is controlled jointly by the input latent $z$ through a single input layer: nudge one coordinate and pose, identity, hair, and lighting all drift together. There is no knob for "same person, different hair," no knob for "same pose, different person." A second pain compounds this. Real faces carry an enormous amount of stochastic micro-detail — the placement of individual hairs, freckles, the speckle in an iris, skin pores — and that detail should be genuinely random, yet the only input is a deterministic $z$, so the network is forced to *manufacture* spatially varying pseudo-randomness internally out of $z$ through the conv stack. That wastes capacity and it shows: generated images carry faint repeating textures, the tell-tale signature of faked noise. And underneath both, the input latent space is structurally entangled, but we have no number for how entangled it is — the existing disentanglement metrics all require an encoder from images back to latents, or known ground-truth factors, and a plain GAN has neither. So the problem is to redesign the generator alone — leaving the discriminator and the loss untouched, since those are orthogonal to this — so that different scales of structure become independently controllable, stochastic detail gets a dedicated cheap source, and the latent representation becomes more disentangled, and to invent metrics that can actually quantify that last claim without an encoder.

The entanglement is not a training accident; it is forced. The input $z$ is sampled from a fixed round Gaussian, and the generator must reproduce the data density, so the probability of each combination of factors in latent space has to match its frequency in the data. If the data manifold has a hole — some combination of attributes that never occurs — then mapping a hole-free round blob onto a manifold with a hole, while preserving densities, leaves the mapping $z \to \text{features}$ no choice but to *curve* so the forbidden region receives no preimage. A curved mapping is by definition an entangled one. Any fixed input distribution pinned to the data density inherits this curvature, so the cause is precisely that the latent the user touches is the same space that must match the data density.

I propose StyleGAN, a style-based generator. The central move comes from the normalization story in fast style transfer. Instance normalization, $\mathrm{IN}(x) = \gamma\,(x-\mu(x))/\sigma(x) + \beta$ with $\mu,\sigma$ taken over the spatial extent of each channel of each sample, outperforms batch norm for style transfer because it is itself performing *style normalization*: the per-channel mean and variance of a feature map *are* the style (matching them is equivalent to matching the Gram matrix), so instance norm strips an instance's style and leaves content. Conditional instance norm then showed that swapping only a per-style learned affine $(\gamma^s,\beta^s)$ — same convolution weights — produces a completely different output style, which means the per-channel affine of a normalization layer is by itself a *complete* style controller. Adaptive instance normalization, $\mathrm{AdaIN}(x,y)=\sigma(y)\,(x-\mu(x))/\sigma(x)+\mu(y)$, removes the learned affine and instead computes the scale and bias on the fly from a style input. The key realization is that AdaIN does not care where its two numbers per channel come from — so instead of a style *image*, I let the *latent* emit them. The latent becomes a style.

Because a progressive-growing generator is already a stack of conv layers from coarse to fine, injecting a fresh style at *every* layer makes the latent control the image scale by scale: coarse-layer styles steer pose and face shape, fine-layer styles steer skin and hair texture — exactly the scale-specific control that was impossible when $z$ entered only at the bottom. The ordering is load-bearing: at each layer I normalize first and *then* write the style. Instance normalization resets the per-channel mean and variance written by the previous layer's style before the new style is applied, so the direct channel-statistics handle is renewed at each layer rather than accumulating; if I added the style without normalizing, each layer's style would ride on the previous one's residual statistics and the scales would smear together. A further consequence falls out for free: a style is spatially invariant — one scale and one bias hit the whole feature map — so it can only express things global to the image at that scale (pose, identity, lighting, color). It physically cannot place "this hair here, that freckle there," because that would need spatially varying values.

That gap is exactly the second pain, so I fill it with a dedicated stochastic source. After each convolution I add a single-channel image of uncorrelated Gaussian noise, broadcast across channels, with a *learned per-channel scale* $B$ so each channel decides how much it wants, $x \leftarrow x + B \odot \text{noise}$, the scales initialized to zero so noise fades in. A fresh noise image is drawn at every layer. The network is pushed to route stochastic content here and global content through the styles with no explicit supervision: if it tried to encode pose through per-pixel noise the pose would flicker spatially and the discriminator would punish it, since real faces have no per-pixel pose; and because free fresh noise is available at every layer, there is no incentive to spend capacity synthesizing randomness from earlier activations, which also localizes the effect to each layer's scale and frees the capacity that used to make those repeating artifacts.

Since the styles now inject the latent everywhere, the bottom input layer is suspect — the latent is already everywhere — so I drop it and start synthesis from a *learned constant*, a fixed $4\times4\times512$ tensor trained like any other weight, serving as a learned coordinate frame; the only per-image signal is then the styles and the noise. And rather than derive styles from the entangled $z$, I interpose an *intermediate* latent space whose density I do not constrain: a learned mapping $f: z \to w$ where $w$, not $z$, drives the styles. Because $W$'s density need not be Gaussian and need not match the data density, $f$ can absorb the curvature and warp the round $z$ into a $w$-space where factors lie along flatter, more linear directions. There is pressure for $f$ to do this rather than leave entanglement in place, because it is easier for the synthesis network to generate from a disentangled representation, so gradient descent pushes the unwarping into $f$ where it is allowed to live. I make $f$ an 8-layer MLP with $z,w \in \mathbb{R}^{512}$ and leaky-ReLU ($\alpha=0.2$) — real depth, since unwarping a curved manifold is not a one-layer affine job — and I pixel-normalize $z$ to unit length first (which also makes interpolation in $z$ well-defined as spherical). A deep mapping network is easy to make too aggressive, and since its output is re-scaled per channel before injection its gradient scale is off, so I slow it with a learning-rate multiplier $\lambda' = 0.01\,\lambda$.

The per-layer affine turns $w$ into a style by emitting $2C$ numbers for a $C$-channel layer. Rather than use the scale directly — which, with small init weights and zero bias, would start near $0$ and kill the signal — the affine emits a scale *deviation* $d_s$ and the layer computes

$$y_i \;=\; (d_{s,i}+1)\,\frac{x_i-\mu(x_i)}{\sigma(x_i)} \;+\; y_{b,i},$$

so $d_s\approx 0$ at init makes the actual scale near $1$; the scale-side bias stays at $0$ (biasing it to $1$ would start the scale near $2$). The full per-layer epilogue is therefore: convolution, add noise with its learned per-channel scale, add the layer bias, leaky-ReLU, instance-normalize, then apply the style. Noise enters before the normalization so the stochastic perturbation is subject to the same statistical reset and styling and lands as genuine spatial variation.

Training does not automatically separate adjacent layers' styles — they both come from the same $w$, so the network could lean on that correlation and blur the per-scale control. To break that assumption I use mixing regularization: with high probability ($\sim 0.9$) I generate an image from *two* latents, mapping $z_1,z_2$ to $w_1,w_2$, picking a random crossover layer, and using $w_1$ before it and $w_2$ after. The network never knows where the switch lands, so it cannot assume adjacent styles share a $w$ and must make each layer's style stand on its own. The payoff is twofold: stronger localization, and a free test-time operation — take the coarse styles of face A and the fine styles of face B to synthesize A's pose-and-shape with B's coloring-and-microstructure. Finally, since low-density regions of any prior yield poor samples, I trade variation for quality with truncation *in $w$*: compute the center of mass $\bar w = \mathbb{E}_z[f(z)]$ (tracked as a moving average of batch means during training) and use $w' = \bar w + \psi\,(w-\bar w)$ with $\psi<1$, applied selectively to only the coarse layers via a cutoff so global structure is cleaned while fine detail stays at full variation. Working in $w$ rather than $z$ makes this reliable without touching the loss. The discriminator and loss are unchanged — non-saturating logistic with R1 ($\gamma=10$) on faces, WGAN-GP for CelebA-HQ, Adam, progressive growing, equalized learning rate, generator weight EMA — though since R1 lets FID keep falling far longer, I train substantially longer (on the order of 25M images).

To actually check the disentanglement claim without an encoder, I derive two metrics from first principles. The first measures the *curviness* of a latent space: a flat, disentangled space changes the image smoothly and at a steady rate under interpolation, while a curved one produces lurches and pop-in. I use LPIPS, a learned weighted L2 between VGG-16 embeddings calibrated to human similarity, as a perceptual ruler, and sum perceptual distance over tiny interpolation segments — the perceptual arc length. There is a normalization subtlety: LPIPS is quadratic for small steps, so a single segment costs $\approx \text{(local speed)}^2\,\varepsilon^2$, and dividing each segment's distance by $\varepsilon^2$ recovers the squared local speed whose expectation is the path-length functional. For the input space ($z$ on the sphere, hence spherical interpolation) and the intermediate space ($w$ unnormalized, hence linear interpolation, driving only the synthesis $g$):

$$l_Z = \mathbb{E}\!\left[\tfrac{1}{\varepsilon^2}\,d\big(G(\mathrm{slerp}(z_1,z_2;t)),\,G(\mathrm{slerp}(z_1,z_2;t+\varepsilon))\big)\right],$$
$$l_W = \mathbb{E}\!\left[\tfrac{1}{\varepsilon^2}\,d\big(g(\mathrm{lerp}(f(z_1),f(z_2);t)),\,g(\mathrm{lerp}(f(z_1),f(z_2);t+\varepsilon))\big)\right],$$

with $\varepsilon=10^{-4}$, $d$ = LPIPS; shorter is flatter and less entangled. The second metric attacks from the linear angle: in a disentangled space each factor is a linear direction, so a binary attribute should be linearly separable by a hyperplane. I train auxiliary attribute classifiers (the discriminator's architecture minus minibatch-stddev), generate 200k images, classify them, sort by confidence and keep the 100k most-confident labels (the ambiguous ones carry noisy labels that would drown the signal), then per attribute fit a linear SVM on the latent ($z$ or $w$) and score with the conditional entropy $H(Y\mid X)$, where $Y$ is the attribute label and $X$ is which side of the hyperplane the point falls on. A low $H(Y\mid X)$ means the plane already determines the attribute, i.e. the attribute is a linear direction; I report $\exp\!\big(\sum_i H(Y_i\mid X_i)\big)$ over the 40 attributes, the exponentiation mapping from the logarithmic to the linear domain in the spirit of the inception score. Lower means more separable, more disentangled.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Equalized learning rate: store N(0,1) weights, apply He scaling at runtime.
# ---------------------------------------------------------------------------
class EqLinear(nn.Module):
    def __init__(self, fin, fout, gain=2**0.5, lrmul=1.0, bias_init=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fout, fin) / lrmul)
        self.bias   = nn.Parameter(torch.full((fout,), float(bias_init) / lrmul))
        self.w_coef = gain / np.sqrt(fin) * lrmul
        self.b_coef = lrmul
    def forward(self, x):
        return F.linear(x, self.weight * self.w_coef, self.bias * self.b_coef)

class EqConv2d(nn.Module):
    def __init__(self, fin, fout, k, gain=2**0.5, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fout, fin, k, k))
        self.bias   = nn.Parameter(torch.zeros(fout)) if bias else None
        self.w_coef = gain / np.sqrt(fin * k * k)
        self.pad = k // 2
    def forward(self, x):
        return F.conv2d(x, self.weight * self.w_coef, self.bias, padding=self.pad)

def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + eps)

def blur2d(x, f=(1, 2, 1)):
    k = torch.tensor(f, dtype=x.dtype, device=x.device)
    k = (k[:, None] * k[None, :]); k = k / k.sum()
    k = k.expand(x.size(1), 1, 3, 3)
    return F.conv2d(x, k, padding=1, groups=x.size(1))

# ---------------------------------------------------------------------------
# Mapping network: z -> w (the unconstrained intermediate latent space).
# ---------------------------------------------------------------------------
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, depth=8, lrmul=0.01):
        super().__init__()
        self.fc = nn.ModuleList(
            [EqLinear(z_dim if i == 0 else w_dim, w_dim, lrmul=lrmul) for i in range(depth)])
    def forward(self, z):
        w = pixel_norm(z)
        for fc in self.fc:
            w = F.leaky_relu(fc(w), 0.2)
        return w

# ---------------------------------------------------------------------------
# Noise injection (per-layer per-pixel) and style modulation (AdaIN).
# ---------------------------------------------------------------------------
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3),
                                device=x.device, dtype=x.dtype)
        return x + self.scale.view(1, -1, 1, 1) * noise

class StyleMod(nn.Module):
    def __init__(self, w_dim, channels):
        super().__init__()
        self.affine = EqLinear(w_dim, channels * 2, gain=1.0)
    def forward(self, x, w):
        x = F.instance_norm(x, eps=1e-8)
        ds, yb = self.affine(w).unsqueeze(2).unsqueeze(3).chunk(2, dim=1)
        return x * (ds + 1) + yb

# ---------------------------------------------------------------------------
# Synthesis layer epilogue: [upsample] conv -> noise -> bias+act -> AdaIN.
# ---------------------------------------------------------------------------
class StyleLayer(nn.Module):
    def __init__(self, fin, fout, w_dim, upsample):
        super().__init__()
        self.upsample = upsample
        self.conv  = EqConv2d(fin, fout, 3, bias=False)
        self.noise = NoiseInjection(fout)
        self.bias  = nn.Parameter(torch.zeros(fout))
        self.style = StyleMod(w_dim, fout)
    def forward(self, x, w):
        if self.upsample:
            x = blur2d(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.conv(x)
        x = self.noise(x)
        x = x + self.bias.view(1, -1, 1, 1)
        x = F.leaky_relu(x, 0.2)
        return self.style(x, w)

class ConstStyleLayer(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.noise = NoiseInjection(channels)
        self.bias  = nn.Parameter(torch.zeros(channels))
        self.style = StyleMod(w_dim, channels)
    def forward(self, x, w):
        x = self.noise(x)
        x = x + self.bias.view(1, -1, 1, 1)
        x = F.leaky_relu(x, 0.2)
        return self.style(x, w)

class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim=512, resolution=1024):
        super().__init__()
        log2res = int(np.log2(resolution))
        self.num_layers = log2res * 2 - 2
        ch = lambda s: min(8192 // (2 ** s), 512)
        self.const  = nn.Parameter(torch.ones(1, ch(1), 4, 4))
        self.input  = ConstStyleLayer(ch(1), w_dim)
        self.layer0 = StyleLayer(ch(1), ch(1), w_dim, upsample=False)
        self.blocks = nn.ModuleList()
        for res in range(3, log2res + 1):
            self.blocks.append(StyleLayer(ch(res-2), ch(res-1), w_dim, upsample=True))
            self.blocks.append(StyleLayer(ch(res-1), ch(res-1), w_dim, upsample=False))
        self.to_rgb = EqConv2d(ch(log2res - 1), 3, 1, gain=1.0)
    def forward(self, ws):                       # ws: [N, num_layers, w_dim]
        x = self.const.expand(ws.size(0), -1, -1, -1)
        x = self.input(x, ws[:, 0])
        x = self.layer0(x, ws[:, 1])
        for i, blk in enumerate(self.blocks, start=2):
            x = blk(x, ws[:, i])
        return self.to_rgb(x)

# ---------------------------------------------------------------------------
# Generator: mapping + style mixing + truncation + synthesis.
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, resolution=1024,
                 mixing_prob=0.9, w_avg_beta=0.995):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.synthesis = SynthesisNetwork(w_dim, resolution)
        self.num_layers = self.synthesis.num_layers
        self.mixing_prob = mixing_prob
        self.w_avg_beta = w_avg_beta
        self.register_buffer('w_avg', torch.zeros(w_dim))

    def _broadcast(self, w):
        return w.unsqueeze(1).repeat(1, self.num_layers, 1)

    def forward(self, z, truncation_psi=1.0, truncation_cutoff=None):
        w = self.mapping(z)
        if self.training:
            self.w_avg.lerp_(w.detach().mean(0), 1 - self.w_avg_beta)
        ws = self._broadcast(w)

        if self.training and self.mixing_prob > 0 and torch.rand((), device=z.device).item() < self.mixing_prob:
            ws2 = self._broadcast(self.mapping(torch.randn_like(z)))
            cutoff = int(torch.randint(1, self.num_layers, (), device=z.device).item())
            idx = torch.arange(self.num_layers, device=z.device).view(1, -1, 1)
            ws = torch.where(idx < cutoff, ws, ws2)

        if truncation_psi != 1.0:
            wavg = self.w_avg.view(1, 1, -1)
            coefs = torch.ones(self.num_layers, device=z.device)
            if truncation_cutoff is None:
                coefs[:] = truncation_psi
            else:
                coefs[:truncation_cutoff] = truncation_psi
            ws = wavg + (ws - wavg) * coefs.view(1, -1, 1)

        return self.synthesis(ws)

# ---------------------------------------------------------------------------
# Loss (unchanged from prior art): non-saturating logistic + R1.
# ---------------------------------------------------------------------------
def d_logistic_r1(D, reals, fakes, gamma=10.0):
    reals = reals.requires_grad_(True)
    real_scores = D(reals)
    fake_scores = D(fakes.detach())
    loss = F.softplus(fake_scores).mean() + F.softplus(-real_scores).mean()
    grad = torch.autograd.grad(real_scores.sum(), reals, create_graph=True)[0]
    r1 = grad.pow(2).flatten(1).sum(1).mean()
    return loss + 0.5 * gamma * r1

def g_nonsaturating(D, fakes):
    return F.softplus(-D(fakes)).mean()

# ---------------------------------------------------------------------------
# Encoder-free disentanglement metrics.
# ---------------------------------------------------------------------------
def lerp(a, b, t):
    return a + (b - a) * t

def slerp(a, b, t):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    omega = torch.acos((a * b).sum(-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    out = (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b
    return torch.where(so.abs() > 1e-7, out, lerp(a, b, t))

def perceptual_path_length(G, lpips, space='w', n=100000, eps=1e-4,
                           batch=16, z_dim=512):
    device = next(G.parameters()).device
    total = 0.0
    seen = 0
    G.eval()
    with torch.no_grad():
        while seen < n:
            bsz = min(batch, n - seen)
            z1 = torch.randn(bsz, z_dim, device=device)
            z2 = torch.randn(bsz, z_dim, device=device)
            t = torch.rand(bsz, 1, device=device)
            if space == 'z':
                img_a = G(slerp(z1, z2, t))
                img_b = G(slerp(z1, z2, t + eps))
            else:
                w1, w2 = G.mapping(z1), G.mapping(z2)
                img_a = G.synthesis(G._broadcast(lerp(w1, w2, t)))
                img_b = G.synthesis(G._broadcast(lerp(w1, w2, t + eps)))
            dist = lpips(img_a, img_b).reshape(bsz, -1).mean(1)
            total += dist.sum().item() / (eps ** 2)
            seen += bsz
    return total / n

def linear_separability(G, classifiers, space='w', n=200000,
                        batch=32, z_dim=512):
    from sklearn.svm import LinearSVC
    import numpy as np
    device = next(G.parameters()).device
    G.eval()
    score = 0.0
    for attr, clf in enumerate(classifiers):
        clf.eval()
        xs, ys, confs = [], [], []
        with torch.no_grad():
            for start in range(0, n, batch):
                bsz = min(batch, n - start)
                z = torch.randn(bsz, z_dim, device=device)
                w = G.mapping(z)
                imgs = G.synthesis(G._broadcast(w))
                logits = clf(imgs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits[:, attr] if logits.ndim == 2 and logits.size(1) > 1 else logits.reshape(-1)
                xs.append((w if space == 'w' else z).detach().cpu())
                ys.append((logits > 0).long().cpu())
                confs.append(logits.abs().cpu())
        X_all = torch.cat(xs, 0).numpy()
        Y_all = torch.cat(ys, 0).numpy()
        conf_all = torch.cat(confs, 0)
        keep = conf_all.argsort(descending=True)[: n // 2].numpy()
        X = X_all[keep]
        Y = Y_all[keep]
        if np.unique(Y).size < 2:
            continue
        svm = LinearSVC(max_iter=10000).fit(X, Y)
        side = svm.predict(X)

        h = 0.0
        for s in (0, 1):
            m = side == s
            if m.sum() == 0:
                continue
            p_y_given_s = Y[m].mean()
            p_s = m.mean()
            for q in (p_y_given_s, 1 - p_y_given_s):
                if q > 0:
                    h -= p_s * q * np.log(q)
        score += h
    return float(np.exp(score))
```
