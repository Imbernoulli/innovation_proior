# HiFi-GAN, distilled

HiFi-GAN is a GAN-based neural vocoder that inverts a mel-spectrogram to a 22.05 kHz raw waveform in a single parallel pass, matching autoregressive fidelity while running faster than real time. Its central idea: because speech is a sum of sinusoids at many periods, the discriminator should audit the waveform's **periodic** structure directly, not only progressively averaged (low-passed) views of it.

## The problem

A stage-two vocoder must turn a mel-spectrogram into a waveform that is both perceptually indistinguishable from a recording and far faster than real time. Autoregressive models (WaveNet) sound great but emit one sample per forward pass (far too slow); flow models (WaveGlow) parallelize but are very large; existing GAN vocoders (MelGAN, Parallel WaveGAN) are fast but trail on quality. The unaddressed weakness: discriminators that view the signal only at raw and average-pooled scales are blind to fine periodic high-frequency detail, because average pooling is a low-pass filter — exactly the content the ear is most sensitive to.

## Key ideas

**Multi-Period Discriminator (MPD).** For each period `p ∈ {2, 3, 5, 7, 11}` (primes, to minimize redundant coverage), reshape the 1D waveform of length `T` losslessly into a 2D array of shape `(T/p, p)` and apply 2D convolutions with kernel **width 1** along the period axis. This processes each periodic phase independently — capturing periodicity without any low-pass averaging — and because it reshapes (rather than subsamples), gradients reach every timestep.

**Multi-Scale Discriminator (MSD).** Three sub-discriminators on raw, 2×- and 4×-average-pooled audio (from MelGAN), to capture consecutive long-term structure. The raw-audio sub-discriminator uses spectral norm; the others weight norm.

**Generator with Multi-Receptive Field Fusion (MRF).** Transposed convolutions upsample the mel to waveform rate. After each upsample, several residual blocks with different kernel sizes and dilations run in parallel; their outputs are summed and divided by the number of blocks (averaged), so the module sees many receptive-field patterns at once with a stable output scale. `conv_pre` (kernel 7) maps 80 mel bands to the hidden width; `conv_post` (kernel 7) + `tanh` produce the waveform.

## The objective

LSGAN (squared-error, non-vanishing gradients), anchored by a heavily weighted mel-spectrogram reconstruction and a discriminator feature-matching term. With sub-discriminators `D_k` (the MPD and MSD members):

```
L_adv(D_k; G) = E[(D_k(x) − 1)² + (D_k(G(s)))²]
L_adv(G; D_k) = E[(D_k(G(s)) − 1)²]
L_mel(G)      = E[ ‖ φ(x) − φ(G(s)) ‖₁ ]                     # L1 on mel
L_fm(G; D_k)  = E[ Σ_i (1/N_i) ‖ D_k^i(x) − D_k^i(G(s)) ‖₁ ]  # feature matching

L_G = Σ_k [ L_adv(G; D_k) + λ_fm · L_fm(G; D_k) ] + λ_mel · L_mel(G)
L_D = Σ_k L_adv(D_k; G)
```

with `λ_fm = 2`, `λ_mel = 45`. Here `x` is the real waveform, `s` its mel conditioning, `φ` the waveform→mel transform.

**Config (V1):** hidden width 512, upsample kernels [16, 16, 4, 4], MRF kernel sizes [3, 7, 11], dilations [[1,1],[3,1],[5,1]]×3; leaky-ReLU slope 0.1; weight normalization throughout. Trained on LJSpeech (22.05 kHz, 80-band mel; FFT/win/hop = 1024/1024/256) with AdamW (β₁=0.8, β₂=0.99, wd=0.01), lr 2×10⁻⁴, 0.999/epoch decay.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

LRELU_SLOPE = 0.1
def get_padding(k, d=1): return int((k * d - d) / 2)

class ResBlock1(nn.Module):
    def __init__(self, ch, k=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d)))
            for d in dilation])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=1, padding=get_padding(k, 1)))
            for _ in dilation])
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
            x = xt + x
        return x

class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.num_kernels = len(h['resblock_kernel_sizes'])
        ic = h['upsample_initial_channel']
        self.conv_pre = weight_norm(nn.Conv1d(80, ic, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(ic // 2**i, ic // 2**(i+1), k, u, padding=(k-u)//2))
            for i, (u, k) in enumerate(zip(h['upsample_rates'], h['upsample_kernel_sizes']))])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = ic // 2**(i+1)
            for k, d in zip(h['resblock_kernel_sizes'], h['resblock_dilation_sizes']):
                self.resblocks.append(ResBlock1(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(len(self.ups)):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            xs = None
            for j in range(self.num_kernels):
                b = self.resblocks[i*self.num_kernels + j](x)
                xs = b if xs is None else xs + b
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

class DiscriminatorP(nn.Module):
    def __init__(self, period, use_spectral_norm=False):
        super().__init__()
        self.period = period
        nf = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            nf(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nf(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nf(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nf(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nf(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)))])
        self.conv_post = nf(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    def forward(self, x):
        fmap = []; b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect"); t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU_SLOPE); fmap.append(x)
        x = self.conv_post(x); fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in (2, 3, 5, 7, 11)])
    def forward(self, y, y_hat):
        drs, dgs, frs, fgs = [], [], [], []
        for d in self.discriminators:
            dr, fr = d(y); dg, fg = d(y_hat)
            drs += [dr]; dgs += [dg]; frs += [fr]; fgs += [fg]
        return drs, dgs, frs, fgs

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        nf = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            nf(nn.Conv1d(1, 128, 15, 1, padding=7)),
            nf(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            nf(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            nf(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nf(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            nf(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            nf(nn.Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = nf(nn.Conv1d(1024, 1, 3, 1, padding=1))
    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU_SLOPE); fmap.append(x)
        x = self.conv_post(x); fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])
    def forward(self, y, y_hat):
        drs, dgs, frs, fgs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i:
                y = self.meanpools[i-1](y); y_hat = self.meanpools[i-1](y_hat)
            dr, fr = d(y); dg, fg = d(y_hat)
            drs += [dr]; dgs += [dg]; frs += [fr]; fgs += [fg]
        return drs, dgs, frs, fgs

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(real, gen):
    loss = 0
    for dr, dg in zip(real, gen):
        loss += torch.mean((1 - dr) ** 2) + torch.mean(dg ** 2)
    return loss

def generator_loss(gen):
    return sum(torch.mean((1 - dg) ** 2) for dg in gen)
```
