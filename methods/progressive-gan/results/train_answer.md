The problem I set out to solve is high-resolution adversarial image synthesis, up to $1024\times1024$, where the usual recipes break down in a specific way: samples can be sharp at modest resolution, but as the target side length grows the training becomes brittle, slow, and low in variety. The reason is structural. The generator $G$ learns only through whatever gradient the discriminator (critic) $D$ hands back, and at high resolution the adversarial game is dangerously asymmetric. A real photograph carries a dense set of high-frequency cues, and a random or half-trained generator gets many of them wrong; with millions of pixel coordinates to inspect, $D$ can become a near-perfect separator while $G$ is still struggling to learn global layout. That is exactly the regime where the classic Jensen-Shannon diagnosis says the gradient stops pointing anywhere useful. The standard options each fix one face of this but not the whole. WGAN-GP replaces the saturating classifier with a critic that estimates the Wasserstein-1 distance and penalizes the input-gradient norm on interpolates, which keeps the distance side well-behaved under support mismatch — but a randomly initialized full-resolution generator still has to learn the entire latent-to-megapixel map at once. DCGAN-style BatchNorm stabilizes ordinary deep nets but targets covariate shift, whereas the failure here is better described as a magnitude race between the two players, and high-resolution minibatches are too small for comfortable batch statistics anyway. Minibatch discrimination supplies a real anti-collapse signal, but with a learned projection tensor and extra placement choices it adds fragility to an already sensitive recipe. And memory grows quadratically with side length, so the hardest stages are forced into the smallest minibatches.

What I propose is to stop presenting the final problem at the start; I call the method Progressive Growing of GANs. I train one generator and one discriminator from $4\times4$ upward, adding a single resolution-doubling block at a time, with all existing layers kept trainable so the lower scales can still adjust as new detail arrives. The work at each step is local — turn an already-meaningful $R\times R$ representation into a $2R\times2R$ one — which converts one brutal optimization into a sequence of smaller, mostly cheap ones, since the early stages are shallow and tiny. The central danger is insertion shock: if I append a random high-resolution block and immediately route output through it, the trained generator is suddenly wired to random filters, $D$ sees garbage, and the lower-resolution solution gets destroyed. So each new block is faded in. I track a single scalar $\mathrm{lod}$ (level-of-detail) that starts high and decreases as resolution increases — for a $1024$ model, stable $4\times4$ is $\mathrm{lod}=8$, stable $8\times8$ is $\mathrm{lod}=7$, and a transition runs $\mathrm{lod}$ from $8$ down to $7$. The active stage is $\lfloor\mathrm{lod}\rfloor$ and the fractional part is the weight on the old, lower-resolution path:
$$\text{old\_weight} = \mathrm{lod} - \lfloor \mathrm{lod}\rfloor, \qquad \text{new\_weight} = 1 - \text{old\_weight}.$$
The sign is easy to get backwards, so I anchor it on the two endpoints: at $\mathrm{lod}=7.999$ the old weight must be almost $1$ so the random new block contributes almost nothing, and at $\mathrm{lod}=7.000$ the old weight is $0$ and the new block fully owns the output. The generator's blend at the new resolution $2R$ is therefore
$$\text{image} = \text{new\_weight}\cdot \text{toRGB}_{2R}\!\big(\text{block}(\text{features}_R)\big) + \text{old\_weight}\cdot \text{upscale}\!\big(\text{toRGB}_{R}(\text{features}_R)\big),$$
which at the instant of growth equals the old network's output (the $R$ image projected to RGB and upsampled), then hands control to the new block gradually. The discriminator uses the same convention in reverse, mapping the sharp $2R$ input through the new fromRGB and down-block to features at $R$, and the downsampled image through the old fromRGB, and blending the two $R$-resolution tensors with the same weights:
$$\text{features} = \text{new\_weight}\cdot \text{downblock}\!\big(\text{fromRGB}_{2R}(x_{2R})\big) + \text{old\_weight}\cdot \text{fromRGB}_{R}\!\big(\text{downscale}(x_{2R})\big).$$
Crucially the real images must be faded too. If real images are fully sharp during a transition while fakes are a mixture of sharp and blocky paths, $D$ can simply seize on high-frequency sharpness as a free tell, so I blur the reals by one octave by the matching amount,
$$x_\text{faded} = (1 - \text{old\_weight})\,x + \text{old\_weight}\cdot \text{upscale}\!\big(\text{downscale}(x)\big),$$
which makes them effectively low-resolution at the start of a fade and sharp at the end. To keep tensor shapes uniform across stages, $G$ and the real pipeline both emit at the fixed final size by upsampling with $2^{\lfloor\mathrm{lod}\rfloor}$ after the blend, and $D$ downsamples by the same factor before its active fromRGB.

Three more parameter-free mechanisms make it work. For variation, a discriminator that scores each image independently can be fooled by a generator that repeats a few convincing samples, so I give $D$ cross-sample information with the simplest statistic that says "this batch lacks variety" — the standard deviation across the minibatch. I split the activations $[N,C,H,W]$ into groups, subtract each group's mean, take the standard deviation over the group axis per channel and location, average it to one scalar per group, tile it as a constant $[N,1,H,W]$ map, and concatenate it near $D$'s final $4\times4$ layers; there are no learned parameters, and if $G$ collapses this scalar shrinks and $D$ has a clean handle to punish it. For the magnitude race, I want a brake in $G$ rather than BatchNorm, so I normalize each pixel's feature vector to unit RMS across channels,
$$b_{x,y} = \frac{a_{x,y}}{\sqrt{\tfrac{1}{C}\sum_j (a^j_{x,y})^2 + \epsilon}}, \qquad \epsilon = 10^{-8},$$
which preserves direction and removes only the scale degree of freedom; it goes after each $3\times3$ convolution in $G$ and on the latent input, and is never used in $D$. Finally there is a subtle scale issue in the optimizer itself: He initialization sets each layer's effective weight scale to $\text{gain}/\sqrt{\text{fan\_in}}$, but Adam normalizes updates by a running gradient scale, so if the stored weights live at different dynamic ranges the same normalized step is a different fraction of each layer's natural range. I fix this with an equalized learning rate — store every weight at unit normal scale and multiply by the He constant at runtime,
$$w \sim \mathcal{N}(0,1), \qquad w_\text{eff} = w\cdot c, \qquad c = \frac{\text{gain}}{\sqrt{\text{fan\_in}}},$$
so the forward-pass variance matches He initialization while all learned parameters share one dynamic range. The loss is WGAN-GP with the signs kept straight and a small drift term: $D$ minimizes $\mathbb{E}[D(\text{fake})] - \mathbb{E}[D(\text{real})] + 10\,\mathbb{E}[(\lVert\nabla_{\hat x}D(\hat x)\rVert_2 - 1)^2] + 0.001\,\mathbb{E}[D(\text{real})^2]$, with $\hat x$ on straight-line interpolates between real and fake, and $G$ minimizes $-\mathbb{E}[D(\text{fake})]$; the tiny $0.001\,\mathbb{E}[D(\text{real})^2]$ drift pulls real scores toward zero without disturbing the ordering. Adam runs with $\text{lr}=0.001$, $\beta_1 = 0$, $\beta_2 = 0.99$, $\epsilon = 10^{-8}$, one discriminator update per generator update, with the latent drawn from a $512$-dimensional Gaussian normalized to the hypersphere. The CelebA-HQ schedule budgets $800\text{k}$ real images at the initial $4\times4$ stage, then $800\text{k}$ to fade in and $800\text{k}$ to stabilize each added block, exposed as schedule parameters rather than hidden constants so they can be retuned.

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def leaky_relu(x, alpha=0.2):
    return F.leaky_relu(x, negative_slope=alpha)

def upscale2d(x, factor=2):
    return x.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)

def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor)

def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + eps)

def minibatch_stddev(x, group_size=4, eps=1e-8):
    n, c, h, w = x.shape
    g = min(group_size, n)
    while n % g != 0:
        g -= 1
    y = x.view(g, -1, c, h, w).float()
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt(y.square().mean(dim=0) + eps)
    y = y.mean(dim=(1, 2, 3), keepdim=True).to(x.dtype)
    y = y.repeat(g, 1, h, w)
    return torch.cat([x, y], dim=1)

class EqConv2d(nn.Module):
    def __init__(self, cin, cout, kernel, gain=math.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.scale = gain / math.sqrt(cin * kernel * kernel)
        self.pad = kernel // 2

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.pad)

class EqLinear(nn.Module):
    def __init__(self, cin, cout, gain=math.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.scale = gain / math.sqrt(cin)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class GBlock(nn.Module):
    def __init__(self, cin, cout, first=False):
        super().__init__()
        self.first = first
        self.cout = cout
        if first:
            self.dense = EqLinear(cin, cout * 4 * 4, gain=math.sqrt(2) / 4)
            self.conv = EqConv2d(cout, cout, 3)
        else:
            self.conv0 = EqConv2d(cin, cout, 3)
            self.conv1 = EqConv2d(cout, cout, 3)

    def forward(self, x):
        if self.first:
            x = pixel_norm(x)
            x = self.dense(x).view(-1, self.cout, 4, 4)
            x = pixel_norm(leaky_relu(x))
            x = pixel_norm(leaky_relu(self.conv(x)))
            return x
        x = upscale2d(x)
        x = pixel_norm(leaky_relu(self.conv0(x)))
        x = pixel_norm(leaky_relu(self.conv1(x)))
        return x

class DStage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv0 = EqConv2d(cin, cin, 3)
        self.conv1 = EqConv2d(cin, cout, 3)

    def forward(self, x):
        x = leaky_relu(self.conv0(x))
        x = leaky_relu(self.conv1(x))
        return downscale2d(x)

class DHead(nn.Module):
    def __init__(self, channels, label_size=0):
        super().__init__()
        self.conv = EqConv2d(channels + 1, channels, 3)
        self.dense0 = EqLinear(channels * 4 * 4, channels)
        self.dense1 = EqLinear(channels, 1 + label_size, gain=1)

    def forward(self, x):
        x = minibatch_stddev(x)
        x = leaky_relu(self.conv(x))
        x = leaky_relu(self.dense0(x.flatten(1)))
        return self.dense1(x)

def lod_parts(lod, max_stage):
    lod_value = float(lod)
    lod_floor = int(np.floor(lod_value))
    top = max_stage - lod_floor
    old_weight = lod_value - lod_floor
    new_weight = 1.0 - old_weight
    return top, old_weight, new_weight

def lod_downscale_factor(lod):
    return int(2 ** np.floor(float(lod)))

class Generator(nn.Module):
    def __init__(self, nf, latent_size=512):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GBlock(latent_size, nf[0], first=True)] +
            [GBlock(nf[i - 1], nf[i]) for i in range(1, len(nf))]
        )
        self.torgb = nn.ModuleList([EqConv2d(c, 3, 1, gain=1) for c in nf])

    def forward(self, z, lod):
        top, old_weight, new_weight = lod_parts(lod, len(self.blocks) - 1)
        x = self.blocks[0](z)
        if top == 0:
            img = self.torgb[0](x)
            factor = lod_downscale_factor(lod)
            return upscale2d(img, factor) if factor > 1 else img

        prev = x
        for stage in range(1, top + 1):
            prev = x
            x = self.blocks[stage](x)

        img = self.torgb[top](x)
        if old_weight > 0.0:
            old = upscale2d(self.torgb[top - 1](prev))
            img = new_weight * img + old_weight * old
        factor = lod_downscale_factor(lod)
        return upscale2d(img, factor) if factor > 1 else img

class Discriminator(nn.Module):
    def __init__(self, nf, label_size=0):
        super().__init__()
        self.fromrgb = nn.ModuleList([EqConv2d(3, c, 1) for c in nf])
        self.stages = nn.ModuleList([DStage(nf[i], nf[i - 1]) for i in range(1, len(nf))])
        self.head = DHead(nf[0], label_size=label_size)

    def forward(self, x, lod):
        top, old_weight, new_weight = lod_parts(lod, len(self.fromrgb) - 1)
        factor = lod_downscale_factor(lod)
        x = downscale2d(x, factor) if factor > 1 else x
        if top == 0:
            return self.head(leaky_relu(self.fromrgb[0](x)))

        h_new = self.stages[top - 1](leaky_relu(self.fromrgb[top](x)))
        if old_weight > 0.0:
            h_old = leaky_relu(self.fromrgb[top - 1](downscale2d(x)))
            h = new_weight * h_new + old_weight * h_old
        else:
            h = h_new

        for stage in range(top - 1, 0, -1):
            h = self.stages[stage - 1](h)
        return self.head(h)

def process_reals(x, lod):
    # Matches the official pipeline: the dataset loader has already selected
    # the current resolution implied by lod.
    old_weight = float(lod - np.floor(lod))
    if old_weight > 0.0:
        blurred = upscale2d(downscale2d(x))
        x = (1.0 - old_weight) * x + old_weight * blurred
    factor = int(2 ** np.floor(lod))
    return upscale2d(x, factor) if factor > 1 else x

def training_schedule(images_seen, max_lod, train_kimg=800, transition_kimg=800):
    phase = (train_kimg + transition_kimg) * 1000
    phase_idx = images_seen // phase
    phase_pos = (images_seen % phase) / 1000.0
    lod = max_lod - phase_idx
    if transition_kimg > 0:
        lod -= max(phase_pos - train_kimg, 0.0) / transition_kimg
    return max(lod, 0.0)

def train_step(G, D, G_opt, D_opt, reals, lod, lam=10.0, drift=0.001):
    batch = reals.shape[0]
    z = torch.randn(batch, 512, device=reals.device)
    z = z / z.norm(dim=1, keepdim=True)
    reals = process_reals(reals, lod)

    fake = G(z, lod).detach()
    d_real = D(reals, lod)[:, :1]
    d_fake = D(fake, lod)[:, :1]

    eps = torch.rand(batch, 1, 1, 1, device=reals.device)
    xhat = (eps * reals + (1.0 - eps) * fake).requires_grad_(True)
    d_hat = D(xhat, lod)[:, :1].sum()
    grad = torch.autograd.grad(d_hat, xhat, create_graph=True)[0]
    gp = (grad.flatten(1).norm(dim=1) - 1.0).square().mean()

    d_loss = d_fake.mean() - d_real.mean() + lam * gp + drift * d_real.square().mean()
    D_opt.zero_grad()
    d_loss.backward()
    D_opt.step()

    z = torch.randn(batch, 512, device=reals.device)
    z = z / z.norm(dim=1, keepdim=True)
    g_loss = -D(G(z, lod), lod)[:, :1].mean()
    G_opt.zero_grad()
    g_loss.backward()
    G_opt.step()
```

For the $1024\times1024$ CelebA-HQ feature-map ladder, use:

```python
nf = [512, 512, 512, 512, 256, 128, 64, 32, 16]  # 4, 8, ..., 1024
```
