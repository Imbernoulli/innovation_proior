# StyleGAN — A Style-Based Generator

## Problem

A conventional GAN generator feeds the latent `z` through an input layer and a conv stack; the result is high-quality but uncontrollable. There is no scale-specific control, the latent space is entangled (a fixed Gaussian prior forced to match a non-uniform data density must curve), and stochastic micro-detail has to be faked from `z`. StyleGAN redesigns the **generator only** — the discriminator and loss are untouched — to give scale-specific control, a dedicated stochastic source, and a more disentangled latent representation. It also introduces two encoder-free disentanglement metrics.

## Key idea

Stop feeding the latent in at the bottom. Instead:

1. **Mapping network** `f: Z → W`. An 8-layer MLP maps the (pixel-normalized) input latent `z ∈ R^512` to an intermediate latent `w ∈ R^512`. Because `W`'s density is not constrained to match the data, `f` can "unwarp" the entanglement that a fixed prior would otherwise force. Run at 1/100 the learning rate for stability.
2. **Style injection via AdaIN at every layer.** A per-layer learned affine maps `w` to `y = (d_s, y_b)` of size `2·C`, where `d_s` is a deviation from scale 1. Each synthesis layer applies `(d_{s,i}+1)·(x_i − μ(x_i))/σ(x_i) + y_{b,i}`. Normalizing first resets the incoming per-channel statistics before the next style is written, so direct channel-statistics control is localized by layer even though downstream spatial content can still carry earlier effects. Styles are spatially invariant -> they carry only global structure (pose, identity, lighting).
3. **Per-layer noise inputs.** A fresh single-channel Gaussian image is added after each convolution, broadcast with a learned per-channel scale (init 0). This is the dedicated, cheap source of spatial stochasticity (hair, freckles, pores). Spatial inconsistency would be penalized by the discriminator, so the network routes only stochastic detail here; fresh noise at every layer keeps the effect localized.
4. **Learned constant input.** Since styles inject the latent everywhere, the input layer is removed; synthesis starts from a learned `4×4×512` constant.
5. **Mixing regularization.** With high probability during training, two latents `w_1, w_2` drive the styles, switching at a random crossover layer. This decorrelates adjacent styles (better localization) and enables test-time style mixing.
6. **Truncation in W.** At inference, `w' = w̄ + ψ(w − w̄)` with `ψ < 1`, optionally only on coarse layers, trading variation for quality without changing the loss.

## Final objective and metrics

Loss is unchanged: non-saturating logistic `softplus(−D(G(z)))` with R1 penalty `(γ/2)·E[‖∇_x D(x)‖²]`, `γ = 10` (WGAN-GP for CelebA-HQ). Adam, equalized learning rate, progressive growing, generator weight EMA.

**Perceptual path length** (latent curvature): for `z` (spherical interp, normalized) and `w` (linear interp),
`l_Z = E[(1/ε²) d(G(slerp(z_1,z_2;t)), G(slerp(z_1,z_2;t+ε)))]`,
`l_W = E[(1/ε²) d(g(lerp(f(z_1),f(z_2);t)), g(lerp(f(z_1),f(z_2);t+ε)))]`,
with `ε = 10^{-4}`, `d` = LPIPS. Divide by `ε²` because LPIPS is quadratic for small steps. Shorter = flatter = less entangled.

**Linear separability**: train auxiliary attribute classifiers; generate 200k images, keep the 100k most-confident labels; per attribute fit a linear SVM on the latent (`z` or `w`) and compute `H(Y|X)` (Y = classifier label, X = SVM side); report `exp(Σ_i H(Y_i|X_i))` over 40 attributes. Lower = more separable.

## Working code

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

## Notes on faithfulness

- Style modulation uses `x * (d_s + 1) + y_b`; the scale-side affine bias stays at 0, so the actual scale starts near 1. Biasing `d_s` to 1 would incorrectly start the scale near 2.
- Layer epilogue order is conv → noise → bias+activation → instance-norm → style. Instance normalization carries no learnable affine (the style provides it).
- Equalized-learning-rate weights are stored as `N(0,1)/lrmul` and multiplied by `gain/sqrt(fan_in)*lrmul` at runtime; the initial effective weight scale stays He-scaled, while Adam's effective runtime-weight update is multiplied by `lrmul`.
- The mapping network uses `lrmul=0.01`; `z` is pixel-normalized before mapping; `w` is not normalized.
- Progressive-growing fade-in (`lod` cross-fades and fused up/down convolutions) is omitted here for clarity.
