OK, let me think this through from scratch. I want to generate audio — actual sound — and I want to do it honestly, by producing the raw waveform itself: the sequence of amplitude samples that a speaker would push into a DAC. Not vocoder parameters, not a spectrogram I then have to invert, the samples. The reason to want this is that every system I know of that goes through an intermediate representation pays for it. Statistical parametric synthesis extracts vocoder parameters — cepstra or line spectral pairs for the vocal tract, fundamental frequency and aperiodicity for the source — every five milliseconds, trains a model (an HMM, or lately an LSTM) over those trajectories, and then reconstructs a waveform with a vocoder. It's small and flexible but it sounds muffled and buzzy, and when you trace the muffle back, it's three things: the vocoder caps the quality, the acoustic model is imperfect, and the predicted parameter trajectories get oversmoothed. And under all of that sit assumptions that are just false about speech: linear predictive analysis literally models the waveform as a linear autoregressive zero-mean Gaussian process, x_t = Σ_p a_p x_{t-p} + ε_t with ε_t Gaussian, fit inside a fixed-length window — but speech is non-stationary on sub-20-ms scales, the sample-to-sample map is wildly nonlinear, and the amplitude distribution is nothing like a Gaussian. Concatenative synthesis sidesteps modeling by gluing recorded units together; it sounds good segmentally but there's no generative model at all, so it can't bend to a new voice or new prosody, and it carries a huge database. I want to throw all of that out and just learn p(waveform) directly from data, with as little prior as possible.

So how do I model a distribution over a waveform? The waveform is x = {x_1, …, x_T}, a long sequence of samples. The honest, fully general thing is the chain rule:

  p(x) = Π_{t=1}^{T} p(x_t | x_1, …, x_{t-1}).

Each sample is conditioned on every sample before it, and that's exact — no independence assumption, no window. This is exactly what's been working spectacularly on images and text: PixelCNN factorizes an image into a product over pixels, p(x) = Π_i p(x_i | x_{<i}), and neural language models do the same over tokens. These models train by maximizing log-likelihood, which is tractable, so I get a clean validation signal — I can watch the held-out likelihood and know whether I'm over- or underfitting. And they routinely model thousands of strongly-dependent variables. So the frame is settled: I'll build a network for the conditional p(x_t | x_{<t}) and maximize Σ_t log p(x_t | x_{<t}).

What does each conditional output? The sample is, on disk, a 16-bit integer — 65,536 possible values. The instinct is to predict a continuous density, a Gaussian or a mixture of Gaussians, since amplitude is "really" continuous. But the people who pushed the image models found the opposite: a plain categorical softmax over discretized values beats a mixture density model even on implicitly-continuous data, and the reason is clean — a categorical makes no assumption about the *shape* of the distribution, so it can be arbitrarily multimodal, skewed, spiky, whatever the data demands, while a Gaussian-mixture is straitjacketed into a sum of bumps. So I'll output a categorical. But 65,536 classes per sample is absurd — a giant softmax, most of whose mass I'll never need at that resolution.

Let me cut it down. If I just uniformly quantize 16 bits to 8 bits, I get 256 levels but I throw away resolution everywhere equally, including the small amplitudes — and the ear is most sensitive to *small* amplitude differences; linear 8-bit quantization sounds gritty. There's an old telephony trick exactly for this: μ-law companding. Pass the signal through a logarithmic amplitude warp first,

  f(x) = sign(x) · ln(1 + μ|x|) / ln(1 + μ),   −1 < x < 1,   μ = 255,

which is steep near zero and flat near the extremes, *then* quantize the warped signal to 256 uniform levels. Because the warp is steep near zero, the 256 levels bunch up at small amplitudes — fine resolution where the ear cares, coarse where it doesn't. So companding-then-quantize to 256 buys me a tractable 256-way softmax and, on speech, the reconstruction sounds essentially identical to the original. Good — output settled: μ-law to 8 bits, 256-way categorical, cross-entropy loss.

Now the real problem. What network computes p(x_t | x_{<t})? Two hard constraints fight each other. First, causality: the prediction at t must not look at x_t or anything after it, or the chain-rule factorization is a lie and the model cheats. Second, *memory*: to predict one audio sample well I need to see a lot of past, because audio's structure spans many scales — the periodicity of a vowel is sub-millisecond, a phoneme is tens of milliseconds, intonation and rhythm are hundreds of milliseconds to seconds. And the time resolution is brutal: at 16 kHz, one second is 16,000 samples. To have even a fraction of a second of context I need a receptive field of thousands of samples.

The obvious tool for "sequence with long memory" is an RNN — an LSTM carries state forward indefinitely, in principle. But think about what training one on raw audio costs. The recurrence is sequential: backpropagation through time has to unroll step by step, t = 1, 2, 3, …, and you can't parallelize across t because step t needs the hidden state from t−1. On sequences of 10^5 or 10^6 steps that's punishing. And the memory I was promised doesn't really materialize — gradients through an LSTM struggle to carry dependencies past a few hundred steps in practice. So the RNN is slow to train *and* short-memoried on exactly the regime I need. That's the wall: I want long memory and parallel training, and the RNN gives me neither.

Let me flip to convolutions. A convolutional model of p(x_t | x_{<t}) has a lovely property the RNN lacks: at training time, since the whole ground-truth waveform is known, I can compute the conditionals for *all* t in a single parallel forward pass — no unrolling. (Generation is still sequential, sample fed back in, but training is the bottleneck and training parallelizes.) To respect causality I make the convolution *causal*: each output at t depends only on inputs at t and earlier. For images that's a masked 2-D kernel; for a 1-D signal it's even simpler — take a normal convolution and shift its output forward by a few steps, equivalently left-pad the input and drop the tail, so the filter at position t never reaches past t. No recurrence, so no sequential training. 

But now the memory question comes back in a different form: what's the receptive field of a stack of causal convolutions? Stack L layers, each with filter width k. Layer 1 sees k inputs; each subsequent layer extends the reach by (k−1). So the receptive field is

  RF = L·(k−1) + 1.

It grows *linearly* in depth. With width-2 filters, to get a receptive field of 1,024 samples — that's only 64 milliseconds at 16 kHz, barely a phoneme — I'd need about 1,023 layers. A thousand-layer network to see a sixteenth of a second. That's hopeless. So plain deep causal convolutions don't solve the memory problem either; they just trade the RNN's sequential cost for an absurd depth.

I need the receptive field to grow much faster than linearly in depth. Let me stare at *why* it only grows linearly. Each layer reaches back (k−1) input positions beyond what the layer below already covers, because the filter taps are adjacent: it looks at positions t, t−1, …, t−(k−1). Those adjacent taps are the problem — every layer re-examines the immediate neighborhood at the finest resolution, so each layer only buys a constant (k−1) of new reach. What if a layer's taps weren't adjacent? What if, instead of looking at t and t−1, a filter looked at t and t−d, skipping the d−1 samples in between?

That's a dilated convolution — à trous, "with holes." It applies the filter over a region larger than its length by skipping inputs with a fixed step d, the dilation. It's equivalent to inflating the filter with zeros between taps, but you don't pay for the zeros. Dilation 1 is the ordinary convolution. The output keeps the same resolution as the input — there's no pooling, no downsampling, every timestep still gets a prediction — but each layer now reaches *d* times farther per tap. These have been used in signal processing forever and recently for image segmentation, precisely to aggregate context at multiple scales cheaply.

So don't use a fixed dilation; *double* it every layer: d = 1, 2, 4, 8, …. Let me compute the receptive field of such a stack. With filter width k, a single dilated layer with dilation d extends the reach by (k−1)·d beyond the layer below. So for a stack of layers with dilations d_1, d_2, …, d_L,

  RF = (k−1)·Σ_ℓ d_ℓ + 1.

Plug in the doubling pattern d = 1, 2, 4, …, 512 (ten layers) and k = 2. Then Σ d_ℓ = 1 + 2 + 4 + … + 512 = 2^10 − 1 = 1023, so

  RF = (2−1)·1023 + 1 = 1024.

Ten layers, receptive field 1024. The same 1024 that needed a thousand plain layers. The sum of a doubling sequence is geometric, ≈ 2^(number of layers), so the receptive field grows *exponentially* with depth while the cost — number of layers, number of multiply-adds — grows only linearly. That's the whole game: exponential reach at linear cost. Each 1,2,4,…,512 block is, in effect, a nonlinear, far cheaper stand-in for a single width-1024 convolution. And if 1024 samples isn't enough context, I don't need to keep doubling to gigantic dilations (which would leave huge holes and miss fine structure); I just *repeat the block*: 1,2,4,…,512, 1,2,4,…,512, 1,2,4,…,512. Stacking M of these blocks adds capacity and, by the same formula, multiplies the receptive field — RF ≈ (k−1)·M·(2^N − 1) + 1 for M blocks of an N-deep doubling. Repeating restarts at dilation 1 each block so the fine, adjacent-sample structure keeps getting modeled at every block, not just at the bottom.

So the skeleton is: μ-law to 256 levels, one-hot, an initial causal conv, then a stack of dilated causal conv layers with doubling-and-repeating dilations, then read out a 256-way softmax per timestep. Let me now get the *inside* of each layer right, because there are two more choices that matter.

First, the nonlinearity. The default would be a ReLU after each conv. But in early experiments on audio that worked noticeably worse than a *gated* unit, and there's a reason to expect that. Take the layer's output to be

  z = tanh(W_f * x) ⊙ σ(W_g * x),

two convolutions of the same input: one through a tanh ("content"), one through a sigmoid ("gate"), multiplied elementwise. The sigmoid lives in (0,1), so it acts as a learned, data-dependent, per-unit multiplicative mask on the content — for each feature at each timestep the network decides, conditioned on the input, how much of that feature to let through. A ReLU can only gate by its own sign; a separate sigmoid gate is a much more flexible, input-conditioned switch, and that flexibility seems to matter a lot for modeling the sharp, conditional structure of audio (silence vs. voiced vs. fricative). The gated PixelCNN found the same thing on images. So: gated activation per layer, not ReLU.

Second, the network is going to be deep — many dilated layers, several repeated blocks — and deep stacks are hard to optimize and slow to converge. Two connection types fix this. A *residual* connection: take the gated output, pass it through a 1×1 convolution to get back to the residual channel width, and add it to the layer's input, so each block learns a correction to its input rather than a fresh transform — gradients flow straight through the additive path and I can train a much deeper stack. And a *skip* connection: from each layer's gated output, also run a separate 1×1 convolution to produce a skip contribution, and *sum the skip contributions from all layers* at the top. Why both? The residual path is what makes depth trainable, but it passes information *upward* through many transforms before it reaches the output; the skip path gives the final read-out direct access to the features computed at every depth, i.e. at every timescale — the layer with dilation 1 (fine periodicity) and the layer with dilation 512 (long-range structure) both speak directly to the output, instead of the fine information having to survive a long climb. So each layer emits two things: a residual it adds back into the trunk, and a skip it sends to the top.

Then the read-out: sum all the skip outputs, and post-process with ReLU → 1×1 conv → ReLU → 1×1 conv → softmax over the 256 μ-law classes. Two 1×1 convs with nonlinearities to mix the aggregated multi-scale features into per-class logits. Train by cross-entropy of the predicted categorical against the next sample, which is exactly maximizing Σ_t log p(x_t | x_{<t}).

Let me make sure I can actually generate. Training is one parallel pass over the ground-truth waveform with the causal masking — all timesteps at once. Generation is sequential by necessity: predict p(x_1), sample x_1, feed it back, predict p(x_2 | x_1), sample, and so on. Slow per sample, but that's intrinsic to any autoregressive model; the thing I cared about — *training* cost — is what the convolutions made cheap and parallel, and that's what the RNN couldn't give me.

Now, a model of p(x) alone just babbles — free-form audio, language-like but meaningless. To do anything useful, like text-to-speech, I need to condition. Add an input h and model p(x | h) = Π_t p(x_t | x_{<t}, h). How does h enter? It should enter right where the network decides what to emit — inside the gated unit. Two cases, depending on whether h is one thing for the whole utterance or a time-series.

If h is a single global vector — say a speaker identity — I want it to bias every timestep's distribution the same way. So inject it as an added bias inside both gates:

  z = tanh(W_f * x + V_f^T h) ⊙ σ(W_g * x + V_g^T h),

where V_f^T h and V_g^T h are learned linear projections of h, broadcast across time. The same speaker bias tilts the content and the gate at every step; one model can then voice any of many speakers by changing h, and I'd expect (and want to validate) that sharing the trunk across speakers actually *helps* each, since the mechanics of speech are shared.

If h is itself a time-series at a lower rate than the audio — linguistic features for TTS, derived from text, say one vector per 5 ms — I can't just broadcast it; it has to align in time with the 16 kHz samples. So first upsample it to the audio rate. The clean way is a learned upsampling: a transposed (fractionally-strided) convolutional network that maps the low-rate h to a series y = f(h) at the audio resolution. Then inject y the same way, but now the projection is a convolution rather than a per-step linear map:

  z = tanh(W_f * x + V_f * y) ⊙ σ(W_g * x + V_g * y),

with V_f * y a 1×1 convolution over the upsampled conditioning. (I could instead just repeat each low-rate vector across its window to upsample — cruder; it works slightly worse than learning the upsampling, because the transposed conv can shape smooth transitions between frames.) For TTS the receptive field still has to be long enough to cover the relevant context; even a few hundred milliseconds may not capture the slow F0 (intonation) contour, so it helps to also condition on an externally-predicted F0 — but the conditioning *mechanism* is exactly this injection into the gates.

One more lever if I want very long context without paying for it everywhere. Instead of making the main stack's receptive field cover seconds, I can run a separate, smaller *context* stack over a long span of audio and use *its* output to locally-condition a larger main stack that only processes a short recent span. The context stack can be cheaper — fewer units per layer for the longest spans, even pooling to run at a lower rate — which fits the intuition that you need less capacity to model correlations at the longest, slowest timescales. It's an optional refinement on top of the same machinery.

Let me write the model, with the receptive-field arithmetic that justified the dilation schedule kept right there in the code.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANT = 256  # 8-bit after mu-law companding


def mu_law_encode(audio, mu=QUANT - 1):
    # logarithmic warp: steep near 0 (fine resolution where the ear cares),
    # flat near the extremes -> then 256 uniform levels are bunched at small
    # amplitudes. f(x) = sign(x) * ln(1+mu|x|) / ln(1+mu).
    audio = np.clip(audio, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int64)


class CausalConv1d(nn.Module):
    # depend only on current + PAST inputs: left-pad by (k-1)*dilation, drop tail
    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))  # left pad only -> causal
        return self.conv(x)


class ResidualBlock(nn.Module):
    # one dilated causal layer: gated activation -> 1x1 residual + 1x1 skip
    def __init__(self, res_ch, dil_ch, skip_ch, dilation, cond_ch=None):
        super().__init__()
        self.filter_conv = CausalConv1d(res_ch, dil_ch, 2, dilation)  # tanh path
        self.gate_conv   = CausalConv1d(res_ch, dil_ch, 2, dilation)  # sigmoid path
        if cond_ch is not None:  # local conditioning injected into both gates
            self.cond_filter = nn.Conv1d(cond_ch, dil_ch, 1)
            self.cond_gate   = nn.Conv1d(cond_ch, dil_ch, 1)
        self.res_conv  = nn.Conv1d(dil_ch, res_ch, 1)   # back to trunk width
        self.skip_conv = nn.Conv1d(dil_ch, skip_ch, 1)  # to the top read-out

    def forward(self, x, cond=None):
        f = self.filter_conv(x)
        g = self.gate_conv(x)
        if cond is not None:                    # z = tanh(Wf*x + Vf*y) (.) sig(Wg*x + Vg*y)
            f = f + self.cond_filter(cond)
            g = g + self.cond_gate(cond)
        z = torch.tanh(f) * torch.sigmoid(g)    # gated activation: learned mask on content
        skip = self.skip_conv(z)
        res  = self.res_conv(z)
        return (x + res), skip                  # residual add keeps the deep stack trainable


class WaveNet(nn.Module):
    def __init__(self, n_blocks=3, n_layers=10,
                 res_ch=32, dil_ch=32, skip_ch=256, cond_ch=None):
        super().__init__()
        self.input_conv = CausalConv1d(QUANT, res_ch, 2, 1)
        # doubling-and-repeating dilations: 1,2,4,...,512 repeated n_blocks times.
        # restarting at 1 each block re-models fine adjacent-sample structure.
        self.dilations = [2 ** i for i in range(n_layers)] * n_blocks
        self.blocks = nn.ModuleList(
            ResidualBlock(res_ch, dil_ch, skip_ch, d, cond_ch)
            for d in self.dilations)
        # post-process the summed skips: ReLU -> 1x1 -> ReLU -> 1x1 -> 256 logits
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(skip_ch, skip_ch, 1),
            nn.ReLU(), nn.Conv1d(skip_ch, QUANT, 1))

    @property
    def receptive_field(self):
        # dilated stack: RF = (k-1) * sum(dilations) + 1, with k=2.
        # one 1..512 block: 1*(1+2+...+512)+1 = 1024; n_blocks of them stack up.
        # + (k-1) for the initial causal conv (width 2).
        return (2 - 1) * sum(self.dilations) + 1 + (2 - 1)

    def forward(self, x_onehot, cond=None):
        x = self.input_conv(x_onehot)
        skips = 0
        for block in self.blocks:
            x, skip = block(x, cond)
            skips = skips + skip            # sum skip contributions from every depth
        return self.head(skips)             # (B, QUANT, T) logits


def loss_fn(model, waveform, cond=None):
    # predict each sample from its past; cross-entropy == -Σ log p(x_t | x_<t)
    x = F.one_hot(waveform[:, :-1], QUANT).float().transpose(1, 2)
    logits = model(x, cond)
    return F.cross_entropy(logits, waveform[:, 1:])
```

So the chain is: I wanted to model raw audio directly as p(x) = Π_t p(x_t | x_{<t}), and a categorical softmax over μ-law-companded 8-bit samples gives a tractable, shape-free output. The RNN's sequential training and short memory ruled it out for million-sample sequences, so I went to causal convolutions for parallel training — but their receptive field grows only linearly in depth, an absurd thousand layers for a sixteenth of a second. Dilating the convolutions and doubling the dilation made the receptive field grow exponentially in depth at linear cost — RF = (k−1)·Σ d_ℓ + 1, a single 1,2,…,512 block reaching 1024 samples in ten layers — and repeating the block extends it further while keeping fine structure modeled at every stage. A gated tanh/sigmoid activation gave each layer a learned, input-conditioned mask that beat ReLU on audio; residual connections made the deep stack trainable and parameterized skip connections fed every timescale directly to a small softmax head. Finally, injecting a projected conditioning signal into both gates — broadcast for a global speaker code, upsampled by a transposed conv for a local linguistic time-series — turns the babbling p(x) into a controllable p(x | h).
