# Progressive Growing of GANs

## Problem

Train an adversarial image generator at high resolution (up to 1024×1024) that is sharp, varied, and stable to train. The core obstacle: at high resolution the discriminator becomes near-perfect almost immediately (millions of pixels expose every flaw of a generated image), and a near-perfect discriminator on two barely-overlapping distributions returns a vanishing or near-random gradient to the generator. Memory limits also force small minibatches at high resolution, further hurting stability; and generators tend to drop variation and to enter a magnitude-escalation spiral with the discriminator.

## Key idea

Do not learn the latent→megapixel map all at once. **Grow** mirror-image generator G and discriminator D in synchrony, starting at 4×4 and adding one resolution stage at a time (8×8, 16×16, …, 1024×1024). Each stage is a small, local refinement (add the next octave of detail), the discriminator always compares distributions at the current — still-overlapping — resolution, and most iterations run cheaply at low resolution. Bring each new stage in with a smooth **fade-in** so the trained lower layers are never shocked. Add three parameter-free ingredients: a **minibatch-standard-deviation** feature in D for variation, **pixelwise feature normalization** in G to cap signal magnitudes, and **equalized learning rate** so all layers learn at the same effective rate. Train with the WGAN-GP loss.

## The pieces

**Progressive growing.** G and D are mirror stacks of per-resolution blocks; all layers stay trainable. A `lod` (level-of-detail) scalar tracks training progress: its integer part selects the current top resolution, its fractional part is the fade weight α ∈ [0,1).

**Fade-in.** Each resolution has a 1×1 `toRGB` (features→RGB) in G and `fromRGB` (RGB→features) in D. When introducing resolution 2R, treat the new block as a residual branch:

  G:  image = (1−α)·upsample(toRGB_R(features)) + α·toRGB_{2R}(block(features))
  D:  feat  = (1−α)·fromRGB_R(downsample(x))    + α·downblock(fromRGB_{2R}(x))

with α ramped 0→1 over a phase. Real images are blended between the two resolutions by the same α so D never gets a spurious high-frequency tell.

**Minibatch standard deviation.** Per feature and spatial location, compute the std across the (group of the) minibatch; average to one scalar; tile to a constant [N,1,H,W] feature map and concatenate near the end of D (at 4×4). Parameter-free; lets D detect and punish low-variation batches.

**Pixelwise feature normalization (G).** After each conv in G, normalize each pixel's feature vector to unit RMS:

  b_{x,y} = a_{x,y} / sqrt( (1/N) Σ_{j} (a^j_{x,y})² + ε ),  ε = 1e-8, N = #feature maps.

Parameter-free; prevents the magnitudes in G and D from spiraling during competition. Not used in D.

**Equalized learning rate.** Initialize all weights ∼ N(0,1); at runtime use ŵ = w·c with c = gain/√(fan_in) the He constant (gain = √2 for leaky ReLU). The forward pass matches a He-initialized net, but the *trained* weights all share unit scale, so Adam/RMSProp's scale-invariant update moves every layer at the same effective rate.

**Loss & training.** WGAN-GP: L_D = E[D(fake)] − E[D(real)] + λ·E_{x̂}[(‖∇_{x̂}D(x̂)‖₂ − 1)²] + ε_drift·E[D(real)²], with λ=10, ε_drift=1e-3, x̂ on lines between real/fake; L_G = −E[D(fake)]. Alternate one D and one G step per minibatch (n_critic=1). Adam α=1e-3, β1=0, β2=0.99. Leaky-ReLU 0.2 everywhere except linear outputs; no batch/layer/weight norm. 512-d latent on the hypersphere. Nearest-neighbor 2× upsample, average-pool downsample. EMA of G weights (decay 0.999) for evaluation sampling. ~800k images per fade-in and per stabilize phase; minibatch shrinks with resolution (16 → 3 at 1024²).

## Code

```python
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def leaky_relu(x, a=0.2): return torch.maximum(x * a, x)
def upscale2d(x, f=2):
    N, C, H, W = x.shape
    return x.view(N, C, H, 1, W, 1).expand(N, C, H, f, W, f).reshape(N, C, H*f, W*f)
def downscale2d(x, f=2): return F.avg_pool2d(x, f)

# ---- equalized learning rate: N(0,1) weights scaled by the He constant at runtime ----
class EqConv2d(nn.Module):
    def __init__(self, cin, cout, k, gain=np.sqrt(2)):
        super().__init__()
        self.w = nn.Parameter(torch.randn(cout, cin, k, k))
        self.b = nn.Parameter(torch.zeros(cout))
        self.c = gain / np.sqrt(cin * k * k); self.pad = k // 2
    def forward(self, x): return F.conv2d(x, self.w * self.c, self.b, padding=self.pad)

class EqLinear(nn.Module):
    def __init__(self, cin, cout, gain=np.sqrt(2)):
        super().__init__()
        self.w = nn.Parameter(torch.randn(cout, cin)); self.b = nn.Parameter(torch.zeros(cout))
        self.c = gain / np.sqrt(cin)
    def forward(self, x): return F.linear(x, self.w * self.c, self.b)

# ---- pixelwise feature normalization (generator) ----
def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)

# ---- minibatch standard deviation (discriminator) ----
def minibatch_stddev(x, group_size=4):
    N, C, H, W = x.shape; G = min(group_size, N)
    y = x.view(G, -1, C, H, W)
    y = y - y.mean(0, keepdim=True)
    y = (y.pow(2).mean(0) + 1e-8).sqrt()           # [M,C,H,W] per-group stddev
    y = y.mean([1, 2, 3], keepdim=True)            # [M,1,1,1] avg over features+pixels
    y = y.view(-1, 1, 1, 1).expand(N, 1, H, W)
    return torch.cat([x, y], dim=1)

def toRGB(cin):   return EqConv2d(cin, 3, 1, gain=1)
def fromRGB(cout): return EqConv2d(3, cout, 1)

class GBlock(nn.Module):
    def __init__(self, cin, cout, first=False):
        super().__init__(); self.first = first; self.cout = cout
        if first:
            self.dense = EqLinear(cin, cout * 16, gain=np.sqrt(2) / 4)
            self.conv  = EqConv2d(cout, cout, 3)
        else:
            self.conv0 = EqConv2d(cin, cout, 3); self.conv1 = EqConv2d(cout, cout, 3)
    def forward(self, x):
        if self.first:
            x = pixel_norm(x)
            x = pixel_norm(leaky_relu(self.dense(x).view(-1, self.cout, 4, 4)))
            x = pixel_norm(leaky_relu(self.conv(x)))
        else:
            x = upscale2d(x)
            x = pixel_norm(leaky_relu(self.conv0(x)))
            x = pixel_norm(leaky_relu(self.conv1(x)))
        return x

class DBlock(nn.Module):
    def __init__(self, cin, cout, last=False, label_size=0):
        super().__init__(); self.last = last
        if last:
            self.conv   = EqConv2d(cin + 1, cin, 3)
            self.dense0 = EqLinear(cin * 16, cout)
            self.dense1 = EqLinear(cout, 1 + label_size, gain=1)
        else:
            self.conv0 = EqConv2d(cin, cin, 3); self.conv1 = EqConv2d(cin, cout, 3)
    def forward(self, x):
        if self.last:
            x = minibatch_stddev(x)
            x = leaky_relu(self.conv(x))
            x = leaky_relu(self.dense0(x.flatten(1)))
            return self.dense1(x)
        x = leaky_relu(self.conv0(x)); x = leaky_relu(self.conv1(x))
        return downscale2d(x)

# ---- progressive generator with fade-in ----
class Generator(nn.Module):
    def __init__(self, nf):                         # nf[i] = #feature maps at stage i (4x4 -> top)
        super().__init__()
        self.blocks = nn.ModuleList(
            [GBlock(512, nf[0], first=True)] + [GBlock(nf[i-1], nf[i]) for i in range(1, len(nf))])
        self.torgb = nn.ModuleList([toRGB(c) for c in nf])
    def forward(self, z, lod):
        top = (len(self.blocks) - 1) - int(np.floor(lod))   # current top stage index
        alpha = lod - np.floor(lod)
        x = self.blocks[0](z); prev = x
        for i in range(1, top + 1):
            prev = x; x = self.blocks[i](x)
        img = self.torgb[top](x)
        if alpha > 0 and top >= 1:
            old = upscale2d(self.torgb[top - 1](prev))       # faded-out lower-res path
            img = (1 - alpha) * old + alpha * img
        return img

# ---- progressive discriminator with fade-in ----
class Discriminator(nn.Module):
    def __init__(self, nf, label_size=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DBlock(nf[i], nf[i-1]) for i in range(len(nf) - 1, 0, -1)] +
            [DBlock(nf[0], nf[0], last=True, label_size=label_size)])
        self.fromrgb = nn.ModuleList([fromRGB(c) for c in nf])   # indexed by stage
    def forward(self, x, lod):
        top = (len(self.fromrgb) - 1) - int(np.floor(lod))
        alpha = lod - np.floor(lod)
        h = leaky_relu(self.fromrgb[top](x))
        first = len(self.blocks) - 1 - top                       # index into self.blocks
        h = self.blocks[first](h)
        if alpha > 0 and top >= 1:
            old = leaky_relu(self.fromrgb[top - 1](downscale2d(x)))
            h = (1 - alpha) * old + alpha * h
        for i in range(first + 1, len(self.blocks)):
            h = self.blocks[i](h)
        return h

# ---- training schedule, real-image processing, and one step ----
def training_schedule(images_seen, max_lod, train_kimg=800, transition_kimg=800):
    # within each phase: stabilize for train_kimg, then fade the next stage in over transition_kimg.
    phase = (train_kimg + transition_kimg) * 1000
    phase_idx = images_seen // phase
    phase_pos = (images_seen % phase) / 1000.0          # kimg into the current phase
    lod = max_lod - phase_idx
    lod -= max(phase_pos - train_kimg, 0.0) / transition_kimg   # fractional part = fade-in
    return max(lod, 0.0)

def process_reals(x, lod):
    alpha = lod - np.floor(lod)
    if alpha > 0:
        blur = upscale2d(downscale2d(x))            # lower-octave version
        x = (1 - alpha) * blur + alpha * x
    f = int(2 ** np.floor(lod))                     # upscale to network working size
    return upscale2d(x, f) if f > 1 else x

def train_step(G, D, G_opt, D_opt, reals, lod, lam=10.0, drift=1e-3):
    z = torch.randn(reals.size(0), 512, device=reals.device)
    z = z / z.norm(dim=1, keepdim=True)             # latent on the hypersphere
    reals = process_reals(reals, lod)

    fake = G(z, lod).detach()
    d_real, d_fake = D(reals, lod), D(fake, lod)
    eps = torch.rand(reals.size(0), 1, 1, 1, device=reals.device)
    xhat = (eps * reals + (1 - eps) * fake).requires_grad_(True)
    g = torch.autograd.grad(D(xhat, lod).sum(), xhat, create_graph=True)[0]
    gp = (g.flatten(1).norm(dim=1) - 1).pow(2).mean()
    d_loss = d_fake.mean() - d_real.mean() + lam * gp + drift * d_real.pow(2).mean()
    D_opt.zero_grad(); d_loss.backward(); D_opt.step()

    g_loss = -D(G(z, lod), lod).mean()
    G_opt.zero_grad(); g_loss.backward(); G_opt.step()
    # maintain an EMA copy of G (decay 0.999) for evaluation sampling.
```
