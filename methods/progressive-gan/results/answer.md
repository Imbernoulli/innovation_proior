# Progressive Growing Of GANs

Train one generator and one discriminator from 4x4 upward, adding one resolution block at a time. All existing layers remain trainable. Each new block is faded in smoothly, so a just-added random block cannot abruptly replace a lower-resolution solution.

The key convention is the reference implementation's `lod`:

```text
top_stage  = max_stage - floor(lod)
old_weight = lod - floor(lod)
new_weight = 1 - old_weight
```

For a 1024x1024 model, stable 4x4 has `lod=8`; during the fade to 8x8, `lod` decreases from 8 to 7. Therefore `old_weight` is almost 1 at the start of a transition and 0 at the end.

## Fade-In

Generator, when introducing resolution `2R`:

```text
image = new_weight * toRGB_2R(block(features_R))
      + old_weight * upscale(toRGB_R(features_R)).
```

In the fixed-output implementation, this current-resolution RGB image is then upscaled by `2^floor(lod)` so G always returns the final tensor size.

Discriminator, at the matching transition:

```text
active_input = downscale(full_resolution_input, 2^floor(lod))
feat = new_weight * downblock(fromRGB_2R(active_input))
     + old_weight * fromRGB_R(downscale(active_input)).
```

Real images are faded by the same old-path weight:

```text
x_faded = (1 - old_weight) * x + old_weight * upscale(downscale(x)).
```

## Normalization And Variation

Pixelwise feature normalization in the generator:

```text
b_{x,y} = a_{x,y} / sqrt(mean_j (a^j_{x,y})^2 + epsilon),  epsilon = 1e-8.
```

Minibatch standard deviation in the discriminator: split the minibatch into groups, compute standard deviation over the group axis for each feature and spatial location, average to one scalar per group, tile it as a constant `[N,1,H,W]` feature map, and concatenate it at 4x4 before the final discriminator layers.

Equalized learning rate:

```text
w ~ N(0, 1)
w_eff = w * c
c = gain / sqrt(fan_in)
```

The paper text describes this as applying the He scale dynamically; the official TensorFlow code implements it as runtime multiplication by `std = gain/sqrt(fan_in)`.

## Loss And Schedule

WGAN-GP with drift:

```text
L_D = E[D(fake)] - E[D(real)]
    + 10 * E[(||grad_xhat D(xhat)||_2 - 1)^2]
    + 0.001 * E[D(real)^2]

L_G = -E[D(fake)]
```

`xhat` lies on straight-line interpolates between real and fake images. The ordinary unsupervised setup uses one discriminator update per generator update. Adam uses `lr=0.001`, `beta1=0`, `beta2=0.99`, `epsilon=1e-8`. The paper's CelebA-HQ schedule uses 800k real images at the initial 4x4 stage, then 800k images to fade in each new block and 800k to stabilize it; the public code exposes these as `lod_training_kimg` and `lod_transition_kimg` schedule parameters.

## Faithful PyTorch Skeleton

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

For the 1024x1024 CelebA-HQ feature-map ladder, use:

```python
nf = [512, 512, 512, 512, 256, 128, 64, 32, 16]  # 4, 8, ..., 1024
```

The paper also introduces a patch-distribution metric, sliced Wasserstein distance on Laplacian-pyramid patches: sample 16384 images, 128 descriptors per pyramid level, descriptors of size `7x7x3`, normalize per color channel, and compute a randomized sliced Wasserstein estimate using 512 projections. Lower is better because generated local-patch distributions are closer to training local-patch distributions at each scale.
