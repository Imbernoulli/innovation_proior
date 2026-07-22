Let me start from the thing that actually bothers me about stage-two vocoding. I have a mel-spectrogram and I want the raw waveform back, and I want two things at once that the field treats as a trade-off: fidelity good enough to pass for a real recording, and speed good enough to beat real time on a single GPU, maybe even on a CPU. WaveNet gets the fidelity — it models `p(x_t | x_{<t})` with dilated causal convs and it sounds great — but it emits one sample per forward pass, so a single second of 22 kHz audio is twenty-two thousand sequential network calls. That is a non-starter for anything interactive. The flow people fix the speed: WaveGlow transforms a Gaussian noise vector of the same length into the whole waveform in one parallel pass, conditioned on the mel. But it needs ninety-plus coupling layers and a huge parameter budget to do it, and even then it isn't clearly better than WaveNet on quality. And the GAN vocoders — MelGAN, Parallel WaveGAN — are small and fast, genuinely real-time on CPU, but there's an audible gap. They sound a little buzzy, a little rough. So the question I want to answer is: why do the fast adversarial models still lose on quality, and is there something structural I can fix rather than just throwing more parameters at it?

So let me think about what a waveform actually *is*, not as a generic sequence but physically. Voiced speech is quasi-periodic: it's a sum of sinusoids at a fundamental frequency and its harmonics, plus formant structure. A vowel held for 100 ms at, say, a 200 Hz pitch is a signal that repeats roughly every 110 samples and stays correlated across thousands of samples. Two facts fall out of this. One, long-range coherence matters — a phoneme spans thousands of samples and they all have to be consistent, so whatever discriminator I use needs a big receptive field. The field already knows this and addresses it by widening receptive fields. Two — and this is the one I think is under-served — the *periodic* structure has to be reproduced exactly, because the ear is extremely sensitive to it. Get the period slightly wrong, smear the phase of the harmonics, and it sounds synthetic even if the average spectrum is fine.

Now look at how MelGAN's discriminator sees the signal. It has three sub-discriminators: one on the raw waveform, one on a 2×-average-pooled version, one on a 4×-average-pooled version. Multi-scale, evaluating the audio at progressively coarser resolutions. That's good for catching consecutive long-term patterns. But stare at what average pooling does. Averaging adjacent samples is a low-pass filter — it attenuates high frequencies and smears phase. So the 2× and 4× sub-discriminators are looking at smoothed, low-passed versions of the audio. That makes me suspect a generator could produce a waveform whose *average-pooled* views match the real audio's while the fine, high-frequency periodic detail is wrong. But "suspect" isn't enough — I should actually try to fool the pool and see how badly.

Let me build the smallest example that has the failure mode I'm worried about: a slow component the generator gets right, plus a fast harmonic it gets wrong. Take a common low-frequency part `low = sin(2π n / 40)` (a slow, formant-like oscillation), and add a period-4 harmonic that I deliberately get the *phase* of wrong — `A = low + 0.5·sin(2π n / 4)` versus `B = low + 0.5·sin(2π n / 4 + π)`. A and B share their slow envelope but the fast detail is phase-flipped, which is the kind of error the ear hears as buzzy/synthetic. Raw L1 distance between them: mean|A−B| = 0.5. Clearly different signals. Now run them through the pools. The 2×-average-pool L1 comes out 0.50 as well — fine, 2× isn't enough to wash out a period-4 component. But the 4×-average-pool: L1 = 2.9e-15. That is machine zero. A 4× pool averages over exactly one full period of the harmonic, so the harmonic integrates to nothing and *both* signals collapse onto the same low-passed sequence. The pooled discriminator's input for the real and the faked audio is literally bit-for-bit identical to floating-point precision — it has no gradient to give, no way to object. So the failure is real, not hypothetical: average-pooling-based discrimination is blind to exactly the periodic high-frequency content the ear cares most about, and the blindness is total whenever the pooling window is commensurate with the offending period. That's the crack.

So what I want is a discriminator that looks at periodic structure *without* low-pass filtering it away. The naive idea: to inspect a signal at period `p`, take every `p`-th sample — the subsequence at one phase of the period. That keeps the high frequencies (no averaging). But if I just *subsample*, I throw away all the other samples, and then gradients from the discriminator only flow to the samples I kept; the generator gets no learning signal on the rest of the timesteps. That's a problem. I need to keep every sample in play.

Take the 1D waveform of length `T` and *reshape* it into a 2D array of height `T/p` and width `p`. The intent is that each column is one phase of the period and each row is one period's worth of samples. Let me check that the reshape actually lays the samples out the way I claim, on the same example with `p = 4`. Reshaping `arange(T)` into shape `(T/4, 4)` and reading off column 0, the original indices it collects are `[0, 4, 8, 12, 16, ...]` — exactly the "every 4th sample" subsequence, one fixed phase of the period. And the reshape is a pure view: total elements `(T/4)·4 = T`, nothing dropped or duplicated. So the layout is right and lossless.

The reason this matters over plain subsampling is the gradient reach, and I can see it in the same numbers. On the A/B pair above, reshape both by period 4 and look at what a width-1 column-wise op would see: the per-phase-column L1 distances are `[0.0, 1.0, 0.0, 1.0]`. The two columns sitting on the harmonic's peak/trough phases show an L1 of 1.0 — the full phase-flip error, completely undamped — while the two columns on its zero-crossings show 0. So the same error that a 4× average pool annihilated to 1e-15 is preserved at magnitude 1.0 by the period-4 reshape view. That is the whole point: the reshape keeps the high-frequency error visible, and because it touches all `T` samples (not just one subsampled phase), gradients reach every timestep rather than only the kept ones. So: run 2D convolutions over this array with the kernel width restricted to 1 along the period (width) axis. A width-1 kernel processes each phase column independently and never mixes across phases — convolving along the "every `p`-th sample" subsequence, exactly the periodic view I just confirmed keeps the error. That's the multi-period discriminator: a sub-discriminator for each period `p`, a stack of strided 2D convs with leaky ReLU, each one auditing the signal at its own periodicity.

Which periods? I want to cover as many distinct periodicities as possible while minimizing the case where two sub-discriminators end up looking at overlapping/redundant phase structure. If I picked `[2, 4, 8]` they'd share factors and overlap heavily — period-4 and period-8 views are commensurate with period-2. Coprime periods don't share that structure, so prime numbers are the natural choice. Pick `[2, 3, 5, 7, 11]` — small primes, no shared factors, diverse coverage. Apply weight normalization to these. I'll keep MelGAN's multi-scale discriminator too — operating on raw, 2× and 4× average-pooled audio — because it's still good at the consecutive/long-term patterns that the disjoint periodic views don't directly capture. For the multi-scale one, the first sub-discriminator works on raw audio so I'll give it spectral normalization to stabilize it, and weight-norm the rest. So the discriminator is two families: one auditing periodicity (multi-period, the new piece), one auditing consecutive multi-resolution structure (multi-scale, inherited). They're complementary.

The discriminator only tells me what to police; I still need a generator that can produce the waveform in one parallel pass. I'll keep it fully convolutional, no noise input — the mel already conditions everything. Upsample the mel up to waveform resolution with transposed convolutions, in stages. The question is what to put after each upsampling step. I want the generator to be able to model patterns of *various lengths* at each resolution, because the periodic structure lives at many scales simultaneously. A single residual block with one kernel size and one dilation pattern only sees one receptive-field shape. So instead of one block, put several residual blocks in parallel, each with a different kernel size and different dilation rates — so together they observe diverse receptive-field patterns at the same resolution — and combine them. How to combine? Concatenating would blow up the channel count downstream. Summing keeps the channel dimension fixed and lets the blocks contribute additively. If I just sum `n` blocks the magnitude grows with `n`, which interacts badly with the next layer's scale; averaging — sum then divide by the number of kernels — keeps the scale stable regardless of how many parallel blocks I use. So the fusion is: run the parallel residual blocks, sum their outputs, divide by the count. That's the multi-receptive-field fusion module, one after every upsample.

Inside a residual block I want a wide effective receptive field cheaply, so I stack a couple of conv layers with increasing dilation — dilations like (1, 3, 5) in the first conv of each of three sub-layers, with the second conv in each pair at dilation 1 — each preceded by leaky ReLU, with a residual add `x = conv(act(x)) + x`. Weight-normalize the convs. The whole generator is conv_pre (a 1D conv mapping the 80 mel bands up to the hidden width with kernel 7), then the upsample-then-fuse stages halving the channel count each upsample, then conv_post (kernel-7 1D conv down to a single channel) and a `tanh` to bound the waveform to [-1, 1]. For the V1 configuration: hidden width 512, upsample rates [8, 8, 2, 2] with kernels [16, 16, 4, 4], fusion kernel sizes [3, 7, 11], and the table's residual dilation pairs [[1,1],[3,1],[5,1]] repeated for each kernel size, which the code stores as first-conv dilation lists [(1,3,5), (1,3,5), (1,3,5)] because every second conv uses dilation 1.

The architecture still needs training signals. The original GAN uses binary cross-entropy, and the trouble with BCE here is that once the discriminator is confident, its gradient to the generator vanishes — and I'm training a high-capacity generator that will spend a long time on the wrong side of perfect. I want gradients that stay alive for samples that are classified correctly but still far from the real distribution. Least-squares GAN gives me exactly that: replace the BCE terms with squared errors. The discriminator regresses real audio to 1 and generated audio to 0; the generator pushes its fakes toward 1. So, treating the whole bank of sub-discriminators as one `D` for now,

  L_adv(D; G) = E_{(x,s)}[ (D(x) − 1)² + (D(G(s)))² ]
  L_adv(G; D) = E_{s}[ (D(G(s)) − 1)² ]

where `x` is the real waveform and `s` is the conditioning mel. The squared error keeps pushing even when a fake is already on the "fake" side but far from realistic.

Adversarial loss alone is unstable early and can drift off the actual content. The lesson from image-to-image translation and from the STFT-loss vocoders is to add an explicit reconstruction term to the ground truth, which sharpens output and stabilizes the early phase. What should the reconstruction be in? Not raw-waveform L1 — phase makes that ill-behaved. A spectral loss is right, and since the human auditory system is the judge, weight it perceptually: use the *mel*-spectrogram. So compute the mel of the generated waveform and the mel of the real waveform and take their L1 distance:

  L_mel(G) = E_{(x,s)}[ ‖ φ(x) − φ(G(s)) ‖₁ ]

with `φ` the waveform→mel transform. This directly tells the generator to match the time-frequency energy distribution it's being conditioned on, and it stabilizes the adversarial game from the first steps.

There's a third term worth taking from MelGAN: feature matching. The discriminator, as it learns, builds up a hierarchy of features that distinguish real from fake; I can use those features as a learned perceptual metric. Extract every intermediate feature map of the discriminator on the real and on the generated sample and match them in L1, normalized by the number of features per layer:

  L_fm(G; D) = E_{(x,s)}[ Σ_i (1/N_i) ‖ D^i(x) − D^i(G(s)) ‖₁ ]

summed over the `T` discriminator layers, where `D^i` is the `i`-th layer's features and `N_i` their count. This gives the generator a dense, layer-wise target instead of only the scalar real/fake verdict.

Putting it together, and now remembering that `D` is actually a bank of `K` sub-discriminators (the period ones and the scale ones), I sum the adversarial and feature-matching terms over the sub-discriminators and keep one mel term:

  L_G = Σ_{k=1}^{K} [ L_adv(G; D_k) + λ_fm · L_fm(G; D_k) ] + λ_mel · L_mel(G)
  L_D = Σ_{k=1}^{K} L_adv(D_k; G)

The coefficients matter because the mel term and the GAN/feature terms live on very different numerical scales. Let me estimate the scales to pick a weight rather than guess one. The LSGAN terms are means of squared errors of discriminator outputs, each on the order of 0.1 to 1 per sub-discriminator; summed over the roughly eight sub-discriminators, the generator's adversarial contribution is order a few — call it ~3. The mel term is a mean absolute difference of log-mel values; once the generator is partly trained, a typical per-bin L1 sits around 0.1–0.5, and it's already averaged over the 80 bands and the frames, so its raw magnitude is order 0.1–0.5 — an order of magnitude smaller than the adversarial sum, and unweighted it would simply be ignored. I want the mel reconstruction to be the strong anchor that keeps the output faithful while the adversarial terms add the fine realism, so it needs a large multiplier to even compete. Try λ_mel = 45: with a mid-training mel-L1 of ~0.2 that's 45·0.2 ≈ 9, comfortably above the ~3 adversarial sum, and early in training when mel-L1 is ~0.5 it's 45·0.5 ≈ 22, dominating outright — so the mel loss carries the generator into the right neighborhood first and the adversarial terms sharpen it later, which is the ordering I want. The feature-matching term is a helper on top of the adversarial signal, so a small multiplier: λ_fm = 2.

The code can be the same object I just derived: a pre-conv, then for each upsample a transposed conv followed by the fused parallel residual blocks (sum then average), then a post-conv and tanh. The multi-period discriminator reshapes 1D→2D by period and uses 2D convs with width-1 kernels. The multi-scale discriminator pools and uses 1D convs. The three losses are exactly the LSGAN, mel-L1, and feature-matching forms above.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

LRELU_SLOPE = 0.1

def get_padding(k, d=1):
    return int((k * d - d) / 2)

class ResBlock1(nn.Module):
    # parallel-receptive-field building block: dilated convs with residual add
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
            xt = c1(F.leaky_relu(x, LRELU_SLOPE))
            xt = c2(F.leaky_relu(xt, LRELU_SLOPE))
            x = xt + x
        return x

class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.num_kernels = len(h['resblock_kernel_sizes'])
        self.conv_pre = weight_norm(nn.Conv1d(80, h['upsample_initial_channel'], 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h['upsample_rates'], h['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                h['upsample_initial_channel'] // (2 ** i),
                h['upsample_initial_channel'] // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h['upsample_initial_channel'] // (2 ** (i + 1))
            for k, d in zip(h['resblock_kernel_sizes'], h['resblock_dilation_sizes']):
                self.resblocks.append(ResBlock1(ch, k, d))   # one MRF = several of these in parallel
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(len(self.ups)):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))     # transposed-conv upsample
            xs = None                                         # multi-receptive-field fusion:
            for j in range(self.num_kernels):
                b = self.resblocks[i * self.num_kernels + j](x)
                xs = b if xs is None else xs + b              # sum parallel blocks
            x = xs / self.num_kernels                         # then average -> stable scale
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)                                  # waveform in [-1, 1]

class DiscriminatorP(nn.Module):
    # one period sub-discriminator: reshape 1D->2D by period, width-1 kernels keep phases separate
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
        if t % self.period:                                  # pad so length divides the period
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect"); t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)       # lossless reshape: columns = phases
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU_SLOPE); fmap.append(x)
        x = self.conv_post(x); fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in (2, 3, 5, 7, 11)])  # primes
    def forward(self, y, y_hat):
        drs, dgs, frs, fgs = [], [], [], []
        for d in self.discriminators:
            dr, fr = d(y); dg, fg = d(y_hat)
            drs.append(dr); dgs.append(dg); frs.append(fr); fgs.append(fg)
        return drs, dgs, frs, fgs

class DiscriminatorS(nn.Module):
    # one scale sub-discriminator (operates on raw or average-pooled audio)
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
                y = self.meanpools[i - 1](y); y_hat = self.meanpools[i - 1](y_hat)
            dr, fr = d(y); dg, fg = d(y_hat)
            drs.append(dr); dgs.append(dg); frs.append(fr); fgs.append(fg)
        return drs, dgs, frs, fgs

def feature_loss(fmap_r, fmap_g):                # learned perceptual metric, x2
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(real, gen):               # LSGAN: real->1, fake->0
    loss = 0
    for dr, dg in zip(real, gen):
        loss += torch.mean((1 - dr) ** 2) + torch.mean(dg ** 2)
    return loss

def generator_loss(gen):                          # LSGAN: push fakes toward 1
    loss = 0
    for dg in gen:
        loss += torch.mean((1 - dg) ** 2)
    return loss

def mel_l1(mel_real, mel_fake):
    return torch.mean(torch.abs(mel_real - mel_fake))   # L_mel

# feature_loss already returns the lambda_fm-weighted term.
# generator step:  L_G = generator_loss(mpd) + generator_loss(msd)
#                       + feature_loss(mpd) + feature_loss(msd)
#                       + 45 * mel_l1(mel(wav), mel(G(mel(wav))))
```
