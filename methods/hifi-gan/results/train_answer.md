The task is to invert a mel-spectrogram into a 22.05 kHz raw waveform with two competing requirements: the output must be perceptually close to a real recording, and it must be generated far faster than real time, ideally on a single GPU or CPU. Autoregressive vocoders such as WaveNet satisfy quality by modeling p(x_t | x_{<t}) with dilated causal convolutions, but they emit one sample per forward pass, so one second of audio requires tens of thousands of sequential evaluations. Flow-based vocoders such as WaveGlow parallelize generation, but they need very deep stacks of coupling layers and a correspondingly large parameter budget. Existing GAN vocoders such as MelGAN and Parallel WaveGAN are small and fast, but they still leave an audible gap next to autoregressive or flow outputs. The underlying reason is that their discriminators inspect the waveform only at raw and progressively average-pooled scales, and average pooling is a low-pass filter that attenuates high frequencies and smears phase. A generator can therefore match the smoothed views while corrupting the fine periodic structure that the human ear is most sensitive to, producing buzzy or noisy speech.

I propose HiFi-GAN. It is a GAN-based neural vocoder that generates the full waveform in one parallel pass. The core change is in the discriminator: instead of relying solely on average-pooled scales, HiFi-GAN adds a multi-period discriminator that audits the signal's periodic structure directly. For each chosen period p, the 1D waveform is reflect-padded so its length divides p and then reshaped losslessly into a 2D tensor with dimensions (T/p, p). Each column now contains samples separated by exactly one period, i.e. one phase of the periodic structure. A stack of 2D convolutions with kernel width 1 along the period axis processes each phase independently, preserving high-frequency content without low-pass filtering. Because the reshape keeps every sample in place, gradients reach all timesteps. The periods are chosen to be coprime, typically {2, 3, 5, 7, 11}, so the sub-discriminators cover diverse periodicities without redundant overlap. To preserve the ability to judge consecutive long-term structure, HiFi-GAN also retains a multi-scale discriminator operating on raw, 2x-average-pooled, and 4x-average-pooled audio, with spectral normalization on the raw scale and weight normalization on the pooled scales.

The generator must upsample the 80-band mel-spectrogram to waveform resolution in a single forward pass. It begins with a 1D convolution that maps the mel channels to a hidden width, then applies a sequence of transposed-convolution upsampling stages. After each upsampling stage, a multi-receptive-field fusion module runs several residual blocks in parallel, each with a different kernel size and dilation pattern, so that the model can capture patterns at many temporal scales simultaneously. The parallel outputs are summed and divided by the number of blocks to keep the activation scale stable. Each residual block contains two convolution layers with leaky ReLU activations and a residual add, using dilations such as (1, 3, 5) in the first layer and dilation 1 in the second. A final 1D convolution maps to a single channel, followed by tanh to bound the waveform to [-1, 1].

Training uses a least-squares adversarial loss, which avoids the vanishing gradients of binary cross-entropy once the discriminator becomes confident. The discriminator regresses real audio toward 1 and generated audio toward 0, while the generator pushes generated audio toward 1. To stabilize early training and keep the output aligned with the conditioning, a heavily weighted L1 mel-spectrogram reconstruction loss is added, comparing the mel of the generated waveform with the input mel. Finally, a feature-matching loss matches intermediate feature maps of the discriminator on real and generated audio, giving the generator a dense layer-wise perceptual signal. The generator objective is the sum of the adversarial and feature-matching terms over all sub-discriminators, plus the weighted mel loss. A standard V1 configuration uses hidden width 512, upsample rates [8, 8, 2, 2], MRF kernel sizes [3, 7, 11], and is trained with AdamW.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

LRELU_SLOPE = 0.1

def get_padding(k, d=1):
    return int((k * d - d) / 2)

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
            weight_norm(nn.ConvTranspose1d(ic // 2**i, ic // 2**(i + 1), k, u, padding=(k - u) // 2))
            for i, (u, k) in enumerate(zip(h['upsample_rates'], h['upsample_kernel_sizes']))])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = ic // 2**(i + 1)
            for k, d in zip(h['resblock_kernel_sizes'], h['resblock_dilation_sizes']):
                self.resblocks.append(ResBlock1(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(len(self.ups)):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            xs = None
            for j in range(self.num_kernels):
                b = self.resblocks[i * self.num_kernels + j](x)
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
        fmap = []
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in (2, 3, 5, 7, 11)])

    def forward(self, y, y_hat):
        drs, dgs, frs, fgs = [], [], [], []
        for d in self.discriminators:
            dr, fr = d(y)
            dg, fg = d(y_hat)
            drs.append(dr); dgs.append(dg); frs.append(fr); fgs.append(fg)
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
            x = F.leaky_relu(l(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        drs, dgs, frs, fgs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            dr, fr = d(y)
            dg, fg = d(y_hat)
            drs.append(dr); dgs.append(dg); frs.append(fr); fgs.append(fg)
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

def mel_l1(mel_real, mel_fake):
    return torch.mean(torch.abs(mel_real - mel_fake))

# Generator update:
# generator_loss(mpd) + generator_loss(msd)
# + feature_loss(mpd) + feature_loss(msd)
# + 45 * mel_l1(mel_real, mel_fake)
```
