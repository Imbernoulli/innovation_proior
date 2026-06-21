I want to generate sound honestly — by producing the raw waveform itself, the literal sequence of amplitude samples a speaker would push into a DAC, not vocoder parameters and not a spectrogram I would then have to invert. Every system that goes through an intermediate representation pays for it. Statistical parametric synthesis extracts vocoder parameters (cepstra or line spectral pairs for the vocal tract, fundamental frequency and aperiodicity for the source) every few milliseconds, trains an HMM or an LSTM over those trajectories, and reconstructs a waveform with a vocoder; it is small and flexible but sounds muffled and buzzy, and the muffle traces back to three things — the vocoder caps quality, the acoustic model is imperfect, and the predicted parameter trajectories get oversmoothed. Underneath all of that sit assumptions that are simply false about speech: linear predictive analysis models the waveform as a linear autoregressive zero-mean Gaussian process, $x_t = \sum_p a_p x_{t-p} + \varepsilon_t$ with $\varepsilon_t$ Gaussian, fit inside a fixed window — but speech is non-stationary on sub-20-ms scales, the sample-to-sample map is wildly nonlinear, and the amplitude distribution is nothing like a Gaussian. Concatenative synthesis sidesteps modeling entirely by gluing recorded units together; it sounds good segmentally but has no generative model at all, so it cannot bend to a new voice or new prosody and it carries a huge database. I want to throw all of this out and learn $p(\text{waveform})$ directly from data with as little prior as possible. The obstacle is one of scale and memory at once: a single audio sample needs thousands of samples of past context to be predicted well — the periodicity of a vowel is sub-millisecond, a phoneme is tens of milliseconds, intonation spans hundreds of milliseconds to seconds — yet at 16 kHz one second is already 16,000 samples, so the model has to reach a very large receptive field while staying cheap to train.

I propose WaveNet. The honest, fully general way to model a distribution over the waveform $x = \{x_1,\dots,x_T\}$ is the chain rule, $$p(x) = \prod_{t=1}^{T} p(x_t \mid x_1,\dots,x_{t-1}),$$ which is exact, with no independence assumption and no window. This is the recipe that succeeded on images and text, and it trains by maximizing $\sum_t \log p(x_t\mid x_{<t})$, a tractable log-likelihood that gives a clean validation signal. The first question is what each conditional outputs. The instinct is a continuous density since amplitude is "really" continuous, but a plain categorical softmax over discretized values beats a Gaussian or mixture-density output even on continuous data, for a clean reason: a categorical makes no assumption about the *shape* of the distribution, so it can be arbitrarily multimodal, skewed, or spiky, while a Gaussian mixture is straitjacketed into a sum of bumps. A 16-bit sample has 65,536 values, which is an absurd softmax; uniformly quantizing to 8 bits would throw away resolution everywhere equally, including the small amplitudes the ear is most sensitive to. So I borrow $\mu$-law companding from telephony: warp the signal through $$f(x) = \operatorname{sign}(x)\,\frac{\ln(1+\mu|x|)}{\ln(1+\mu)}, \qquad -1 < x < 1, \quad \mu = 255,$$ which is steep near zero and flat near the extremes, *then* quantize to 256 uniform levels. Because the warp is steep near zero, the 256 levels bunch up at small amplitudes — fine resolution where hearing cares, coarse where it does not — and on speech the 256-level reconstruction is essentially indistinguishable from the original. The output is therefore a 256-way categorical trained by cross-entropy.

The core question is what network computes $p(x_t\mid x_{<t})$, where two constraints fight: causality (the prediction at $t$ must not see $x_t$ or anything after, or the factorization is a lie) and memory (it must see thousands of past samples). The obvious tool for long memory is an RNN, but training one on raw audio is ruled out twice over — backpropagation through time is sequential, unrolling step by step over $10^5$–$10^6$ steps with no parallelism across $t$, and in practice the gradient cannot carry dependencies past a few hundred steps, so the RNN is slow to train *and* short-memoried on exactly the regime I need. I flip to convolutions, which have the property the RNN lacks: since the whole ground-truth waveform is known at training time, all $t$ can be computed in a single parallel forward pass. I enforce causality by feeding the prefix and shifting the target by one — the causal output at position $t$ may read $x_1,\dots,x_t$ and is trained against $x_{t+1}$, which is exactly $p(x_{t+1}\mid x_1,\dots,x_t)$ with no leakage — implemented by left-padding the input so the filter at $t$ never reaches past $t$.

But a plain stack of causal convolutions grows its receptive field only linearly in depth: $L$ layers of filter width $k$ reach back $L(k-1)+1$, because every layer's taps are adjacent and so re-examine the immediate neighborhood, buying only a constant $(k-1)$ of new reach. With width-2 filters, a receptive field of 1024 samples — a mere 64 ms — would need about a thousand layers. That is hopeless. The fix is to make the taps *non-adjacent*: a dilated convolution skips $d-1$ inputs between taps, equivalent to inflating the filter with zeros but without paying for them, keeping full output resolution (no pooling) while reaching $d$ times farther per tap. The decisive move is to *double* the dilation each layer, $d = 1,2,4,\dots$. For a stack with dilations $d_1,\dots,d_L$ and filter width $k$, each layer extends the reach by $(k-1)d_\ell$, so $$\mathrm{RF} = (k-1)\sum_{\ell} d_\ell + 1.$$ With $k=2$ and a doubling block $1,2,4,\dots,512$ (ten layers), $\sum d_\ell = 2^{10}-1 = 1023$ and $\mathrm{RF} = 1024$ — the same reach that needed a thousand plain layers, now from ten. The sum of a doubling sequence is geometric, so the receptive field grows *exponentially* in depth while compute grows only linearly: each block is a cheap nonlinear stand-in for a width-1024 convolution. If 1024 is not enough I do not keep doubling to gigantic dilations that leave huge holes; I *repeat* the block, $1,2,\dots,512$ over again, which by the same formula gives $\mathrm{RF} = (k-1)\,M(2^N-1)+1$ for $M$ repeats of an $N$-layer block, and restarting at dilation 1 each block keeps the fine adjacent-sample structure modeled at every stage rather than only at the bottom.

Two choices inside each layer are load-bearing. The first is the nonlinearity: instead of a ReLU I use a gated unit, $$z = \tanh(W_f * x) \odot \sigma(W_g * x),$$ two convolutions of the same input — a tanh "content" path and a sigmoid "gate" path multiplied elementwise. The sigmoid lives in $(0,1)$ and so acts as a learned, data-dependent, per-unit multiplicative mask on the content: for each feature at each timestep the network decides, conditioned on the input, how much to let through. A ReLU can only gate by its own sign; a separate sigmoid is a far more flexible, input-conditioned switch, and that flexibility matters for the sharp conditional structure of audio (silence vs. voiced vs. fricative). The second is that the stack is deep and hard to optimize, so each layer emits two paths. A residual connection passes the gated output through a $1\times1$ convolution back to the trunk width and adds it to the layer's input, so each layer learns a correction and gradients flow straight through the additive path, making the depth trainable. A skip connection runs a separate $1\times1$ convolution from each layer's gated output to the top, and all skip contributions are summed there. Both are needed: the residual path makes depth trainable but information must climb through many transforms to reach the output, while the skip path gives the read-out direct access to the features at every depth — i.e. every timescale, the dilation-1 layer (fine periodicity) and the dilation-512 layer (long-range structure) both speaking directly to the output. The summed skips are post-processed by $\mathrm{ReLU} \to 1\times1 \to \mathrm{ReLU} \to 1\times1 \to \mathrm{softmax}$ into per-timestep logits over the 256 classes. Training is the one parallel masked pass; generation is sequential by necessity (sample $x_1$, feed it back, predict $p(x_2\mid x_1)$, and so on), but the cost I cared about was training, and the convolutions made that cheap.

A model of $p(x)$ alone only babbles, so to control it I model $p(x\mid h) = \prod_t p(x_t\mid x_{<t}, h)$ and inject $h$ exactly where the network decides what to emit — inside the gated unit. For a single global vector such as a speaker code, I add a broadcast linear projection into both gates, $z = \tanh(W_f * x + V_f^\top h) \odot \sigma(W_g * x + V_g^\top h)$, so the same bias tilts content and gate at every step and one trunk can voice many speakers. For a conditioning time-series at a lower rate, such as linguistic features for TTS, I first upsample $h$ to the audio rate with a learned transposed convolution to get $y$ (cleaner than repeat-upsampling, which cannot shape smooth frame transitions), then inject it as a $1\times1$ convolution, $z = \tanh(W_f * x + V_f * y) \odot \sigma(W_g * x + V_g * y)$. As an optional refinement for very long context, a separate smaller context stack — fewer units, even pooling for the slowest timescales — can run over a long span and locally-condition a larger main stack over a short recent span.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANT = 256  # 8-bit after mu-law companding


def mu_law_encode(audio, mu=QUANT - 1):
    audio = np.clip(audio, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int64)


def mu_law_decode(quantized, mu=QUANT - 1):
    signal = 2 * (quantized.astype(np.float32) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return np.sign(signal) * magnitude


class CausalConv1d(nn.Module):
    """Same-length 1-D conv whose output[t] depends only on input[:t+1]."""
    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))  # left-pad only -> causal


class ResidualBlock(nn.Module):
    """Dilated causal layer: gated activation -> 1x1 residual + 1x1 skip."""
    def __init__(self, res_ch, dil_ch, skip_ch, dilation, cond_ch=None):
        super().__init__()
        self.filter_conv = CausalConv1d(res_ch, dil_ch, 2, dilation)
        self.gate_conv   = CausalConv1d(res_ch, dil_ch, 2, dilation)
        if cond_ch is not None:
            self.cond_filter = nn.Conv1d(cond_ch, dil_ch, 1)
            self.cond_gate   = nn.Conv1d(cond_ch, dil_ch, 1)
        self.res_conv  = nn.Conv1d(dil_ch, res_ch, 1)
        self.skip_conv = nn.Conv1d(dil_ch, skip_ch, 1)

    def forward(self, x, cond=None):
        f, g = self.filter_conv(x), self.gate_conv(x)
        if cond is not None:
            f = f + self.cond_filter(cond)
            g = g + self.cond_gate(cond)
        z = torch.tanh(f) * torch.sigmoid(g)         # gated activation
        return x + self.res_conv(z), self.skip_conv(z)


class WaveNet(nn.Module):
    def __init__(self, n_blocks=5, n_layers=10,
                 res_ch=32, dil_ch=32, skip_ch=512, cond_ch=None):
        super().__init__()
        self.input_conv = CausalConv1d(QUANT, res_ch, 2, 1)
        self.dilations = [2 ** i for i in range(n_layers)] * n_blocks  # 1..512 repeated
        self.blocks = nn.ModuleList(
            ResidualBlock(res_ch, dil_ch, skip_ch, d, cond_ch)
            for d in self.dilations)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(skip_ch, skip_ch, 1),
            nn.ReLU(), nn.Conv1d(skip_ch, QUANT, 1))

    @property
    def receptive_field(self):
        # Reference-code convention: dilated stack plus the initial width-2 causal conv.
        return (2 - 1) * sum(self.dilations) + 1 + (2 - 1)

    def forward(self, x_onehot, cond=None):
        x = self.input_conv(x_onehot)
        skips = 0
        for block in self.blocks:
            x, skip = block(x, cond)
            skips = skips + skip
        return self.head(skips)  # logits aligned with input positions


def loss_fn(model, waveform, cond=None):
    # output position t reads waveform[:t+1] and predicts waveform[t+1].
    x = F.one_hot(waveform[:, :-1], QUANT).float().transpose(1, 2)
    target = waveform[:, 1:]
    logits = model(x, cond)
    # Match the reference implementation: drop positions without a full receptive field.
    warmup = model.receptive_field - 1
    if target.size(1) <= warmup:
        raise ValueError("waveform must be longer than the model receptive field")
    return F.cross_entropy(logits[:, :, warmup:], target[:, warmup:])
```
