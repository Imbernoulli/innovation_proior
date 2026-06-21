The goal is a generative model of natural images that gives an exact log-likelihood, an exact latent code for any image, and parallel training and sampling, while still producing sharp realistic pictures. None of the main existing families satisfy all of these. Autoregressive models evaluate exact likelihoods but generate one subpixel at a time, so sampling is serial and slow for high-resolution images. Variational autoencoders sample in parallel but only optimize a lower bound on the likelihood and infer the latent approximately. Adversarial models synthesize striking images but have no encoder, no tractable likelihood, and no direct way to measure fit. Normalizing flows built from exactly invertible transformations can in principle deliver all four properties, but earlier instances such as NICE and RealNVP were weaker on likelihood and had not scaled to realistic high-resolution synthesis.

The way forward is to keep the change-of-variables framework. The density is defined by pushing a spherical Gaussian prior through a bijection x = g(z) with inverse z = f(x), giving log p(x) = log p(z) + log|det(dz/dx)|. Because the full transformation is a composition of layers, the log-determinant decomposes into a sum of per-layer terms. Glow makes each invertible step more expressive by replacing the fixed inter-layer permutation with a learned invertible 1×1 convolution, replacing batch normalization with data-initialized but batch-independent activation normalization, and using channel-wise affine coupling with zero-initialized output heads.

The method is Glow, short for Generative Flow with Invertible 1×1 Convolutions. One flow step consists of three sublayers applied in sequence: activation normalization, invertible 1×1 convolution, and affine coupling. Actnorm applies a per-channel scale and bias, initialized on the first minibatch so each channel starts with zero mean and unit variance, and then treated as fixed trainable parameters afterward. This avoids the activation noise of batch normalization, which becomes harmful when per-GPU batch size is one, as it often is for high-resolution images. Its contribution to the log-determinant is h·w·Σ log|s|, where s is the per-channel scale and h, w are spatial dimensions.

The invertible 1×1 convolution applies the same learned c×c matrix W to the channel vector at every spatial location. This generalizes the fixed channel reversals or random permutations of earlier flows, since a permutation matrix is just a special case of W, and lets the model learn how to route information between coupling layers. W is parameterized in PLU form as W = P·L·(U + diag(s)), with P fixed at initialization, L unit-diagonal lower triangular, U strictly upper triangular, and s stored as sign·exp(log|s|). Then log|det W| = Σ log|s|, so the layer contributes h·w·Σ log|s| to the total log-determinant. The inverse is a 1×1 convolution with W^{-1}, computed when needed for sampling.

The affine coupling layer splits the input along channels into z1 and z2, leaves z1 unchanged, and transforms z2 by a shift and positive scale predicted by a neural network taking only z1 as input. Concretely, the network outputs shift and scale_logits, scale = sigmoid(scale_logits + 2), and z2 becomes (z2 + shift) ⊙ scale. The layer is inverted without inverting the network by subtracting the shift and dividing by the scale. Its log-determinant is Σ log(scale). Because the learned 1×1 convolution already mixes channels flexibly, Glow drops RealNVP's checkerboard masking and splits purely on channels. The coupling network uses three convolutions, 3×3→512, 1×1→512, and 3×3→output, with the last layer initialized to zero so each coupling starts near identity. Flow steps are stacked K times per level over L levels, with squeeze reshaping and a multi-scale factoring-out of half the channels at each level, scored against Gaussian priors predicted from the kept half.

Training dequantizes integer pixels with uniform noise, runs the image through the flow while accumulating every sublayer's log-determinant, scores the final and factored-out latents under Gaussian priors, and minimizes the negative dequantized log-likelihood in bits per dimension. The objective is loss = -(log p(z) + logdet - log(n_bins)·D) / (log 2 · D). Sampling uses the inverse map, optionally with temperature T < 1 by drawing z ~ N(0, T^2 I) for cleaner outputs.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess(x, n_bits=8):
    n_bins = 2 ** n_bits
    x = x.float()
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    x = x + torch.rand_like(x) / n_bins
    return x, n_bins


def squeeze2d(x, factor=2):
    b, c, h, w = x.shape
    x = x.view(b, c, h // factor, factor, w // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * factor * factor, h // factor, w // factor)


def unsqueeze2d(x, factor=2):
    b, c, h, w = x.shape
    x = x.view(b, c // factor ** 2, factor, factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, c // factor ** 2, h * factor, w * factor)


class FlowModule(nn.Module):
    def forward(self, x, logdet):
        raise NotImplementedError

    def reverse(self, y):
        raise NotImplementedError


class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.logs = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def _init(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = (x - mean).pow(2).mean(dim=[0, 2, 3], keepdim=True).sqrt()
            self.bias.copy_(-mean)
            self.logs.copy_(torch.log(1.0 / (std + 1e-6)))
            self.initialized.fill_(1)

    def forward(self, x, logdet):
        if self.initialized.item() == 0:
            self._init(x)
        h, w = x.shape[2], x.shape[3]
        y = (x + self.bias) * torch.exp(self.logs)
        return y, logdet + h * w * self.logs.sum()

    def reverse(self, y):
        return y * torch.exp(-self.logs) - self.bias


class InvConv1x1(nn.Module):
    def __init__(self, channels):
        super().__init__()
        w0 = torch.linalg.qr(torch.randn(channels, channels))[0]
        P, L, U = torch.linalg.lu(w0)
        self.register_buffer("P", P)
        s = torch.diag(U)
        self.register_buffer("sign_s", torch.sign(s))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(torch.triu(U, 1))
        self.register_buffer("l_mask", torch.tril(torch.ones_like(L), -1))
        self.register_buffer("eye", torch.eye(channels))

    def _w(self):
        L = self.L * self.l_mask + self.eye
        U = self.U * self.l_mask.t() + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x, logdet):
        h, w = x.shape[2], x.shape[3]
        W = self._w()
        y = F.conv2d(x, W.view(*W.shape, 1, 1))
        return y, logdet + h * w * self.log_s.sum()

    def reverse(self, y):
        W_inv = torch.inverse(self._w())
        return F.conv2d(y, W_inv.view(*W_inv.shape, 1, 1))


class OutputConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, logscale_factor=3.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.logs = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.logscale_factor = logscale_factor

    def forward(self, x):
        return self.conv(x) * torch.exp(self.logs * self.logscale_factor)


class CouplingNN(nn.Module):
    def __init__(self, in_ch, out_ch, width=512):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 1)
        self.c3 = OutputConv2d(width, out_ch)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return self.c3(x)


class AffineCoupling(nn.Module):
    def __init__(self, channels, width=512):
        super().__init__()
        self.net = CouplingNN(channels // 2, channels, width)

    def forward(self, x, logdet):
        z1, z2 = x.chunk(2, dim=1)
        h = self.net(z1)
        shift = h[:, 0::2]
        scale_logits = h[:, 1::2]
        scale = torch.sigmoid(scale_logits + 2.0)
        z2 = (z2 + shift) * scale
        return torch.cat([z1, z2], dim=1), logdet + torch.log(scale).flatten(1).sum(1)

    def reverse(self, y):
        z1, z2 = y.chunk(2, dim=1)
        h = self.net(z1)
        shift = h[:, 0::2]
        scale_logits = h[:, 1::2]
        scale = torch.sigmoid(scale_logits + 2.0)
        z2 = z2 / scale - shift
        return torch.cat([z1, z2], dim=1)


class FlowStep(FlowModule):
    def __init__(self, channels, width=512):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = InvConv1x1(channels)
        self.coupling = AffineCoupling(channels, width)

    def forward(self, x, logdet):
        x, logdet = self.actnorm(x, logdet)
        x, logdet = self.invconv(x, logdet)
        x, logdet = self.coupling(x, logdet)
        return x, logdet

    def reverse(self, y):
        y = self.coupling.reverse(y)
        y = self.invconv.reverse(y)
        return self.actnorm.reverse(y)


def gaussian_logp(z, mean, log_sd):
    return -0.5 * (math.log(2 * math.pi) + 2 * log_sd
                   + (z - mean) ** 2 / torch.exp(2 * log_sd))


class ImageFlow(nn.Module):
    def __init__(self, in_ch=3, depth=32, levels=3, width=512):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        self.split_priors = nn.ModuleList()
        c = in_ch * 4
        for i in range(levels):
            self.blocks.append(nn.ModuleList(
                [FlowStep(c, width) for _ in range(depth)]))
            if i < levels - 1:
                self.split_priors.append(OutputConv2d(c // 2, c))
                c = c * 2
        self.top_prior = OutputConv2d(c, 2 * c)

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        log_p = torch.zeros_like(logdet)
        z = squeeze2d(x)
        for i in range(self.levels):
            for step in self.blocks[i]:
                z, logdet = step(z, logdet)
            if i < self.levels - 1:
                z1, z2 = z.chunk(2, dim=1)
                mean, log_sd = self.split_priors[i](z1).chunk(2, dim=1)
                log_p = log_p + gaussian_logp(z2, mean, log_sd).flatten(1).sum(1)
                z = squeeze2d(z1)
        mean, log_sd = self.top_prior(torch.zeros_like(z)).chunk(2, dim=1)
        log_p = log_p + gaussian_logp(z, mean, log_sd).flatten(1).sum(1)
        return z, logdet, log_p

    def reverse(self, z, eps=None, eps_std=1.0):
        eps = [None] * (self.levels - 1) if eps is None else eps
        for i in reversed(range(self.levels)):
            if i < self.levels - 1:
                z1 = unsqueeze2d(z)
                mean, log_sd = self.split_priors[i](z1).chunk(2, dim=1)
                noise = torch.randn_like(mean) * eps_std if eps[i] is None else eps[i]
                z2 = mean + torch.exp(log_sd) * noise
                z = torch.cat([z1, z2], dim=1)
            for step in reversed(self.blocks[i]):
                z = step.reverse(z)
        return unsqueeze2d(z)


def loss_bits_per_dim(logdet, log_p, n_bins, n_pixels):
    ll = log_p + logdet - math.log(n_bins) * n_pixels
    return (-ll / (math.log(2) * n_pixels)).mean()


def train_step(model, batch, opt):
    x, n_bins = preprocess(batch)
    _, logdet, log_p = model(x)
    loss = loss_bits_per_dim(logdet, log_p, n_bins, x[0].numel())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss
```
