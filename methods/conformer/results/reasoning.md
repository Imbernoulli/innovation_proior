Let me start from the tension that actually limits an ASR encoder, because the architecture should fall out of it. I need to model two kinds of dependency in an utterance. One is global: a sound's identity can hinge on context far away — coarticulation, the word the speaker is mid-way through, the long-range rhythm. The other is local: the sharp, position-local feature patterns — an onset, a formant transition, a burst — that separate one nearby phone from the next. These are genuinely different computations, and the two leading building blocks each nail exactly one of them.

Self-attention relates every position to every other in a single layer, with weights computed from content. That is exactly what I want for global, content-based interactions — distance is no obstacle, one hop reaches anywhere. But stare at what attention *is* at a fine scale: each output is a softmax-weighted average over all positions. A weighted average is a smoothing operation; it's not built to extract a crisp, position-specific local pattern. So attention is globally strong and locally blunt.

A convolution is the mirror image. A kernel slides over a local window, so it picks up edges, onsets, short patterns cheaply, with translation equivariance — perfect for local acoustic detail. But its receptive field only grows by one window per layer. To see across a whole utterance I'd need a deep stack or fat kernels, i.e. I pay for global reach in depth or parameters. So convolution is locally strong and globally expensive.

I could try to patch one side. ContextNet takes the convolutional route and bolts on a squeeze-and-excitation module per block to inject global context. But think about what squeeze-and-excitation actually does: it global-average-pools the whole sequence into one vector and uses it to rescale channels. That's a *single*, static, averaged global summary — it can't say "position 40 should attend to position 200 but not position 30." It cannot model dynamic, position-dependent global interactions, which is precisely attention's gift. So patching convolution with a global average is a weak substitute for attention.

So neither block alone is right, and faking the missing half is weak. The move I keep coming back to is to *use both* — let attention do the global, content-based interactions and let convolution do the local, position-based ones, each doing what it is genuinely good at. The question is only how to arrange them in a block.

The simplest arrangement is parallel: split the input into an attention branch and a convolution branch, run both, concatenate. That's a reasonable instinct — it lets both computations happen — but concatenation just sets the two representations side by side and leaves it to later layers to fuse them; the convolution never gets to act on attention's globally-mixed output, nor vice versa. I'd rather they *compose*: have one refine the other within the block, so local detail is extracted from a representation that already carries global context (or the reverse). Composition gives a deeper interaction than concatenation for the same budget.

If they compose, in which order? Two options: convolution then attention, or attention then convolution. Let me reason about it. If attention runs first, it produces a representation where every position is already a content-weighted mixture of the whole sequence — globally informed. Then the convolution operates on that, sharpening local patterns *within the already-globalized features*. The reverse — local first, then global — is an equally coherent story: extract clean local features, then let attention relate them at range. I can't actually settle this from first principles; both compose, both are cheap, and which wins is the kind of thing only an ablation decides. I'll go with self-attention-then-convolution as the working choice — establish long-range content, then carve local detail on top — and flag the order as something I'd want to A/B, not something I've shown.

Now I have a block whose spine is: self-attention sublayer, then convolution sublayer, each in a residual unit. I want to nail down each sublayer's internals, with a reason for every choice.

Start with attention. Speech utterances vary enormously in length, and what matters acoustically is the *offset* between two frames, not their absolute indices. Absolute positional encodings tie the representation to position number, which generalizes poorly when test utterances are longer or shorter than training ones. So I'll use *relative* positional encoding — the attention score depends on the gap between query and key positions. That makes the encoder robust to utterance length and is the natural fit for a translation-equivariant signal. I'll also wrap the attention in a *pre-norm* residual unit — layer-norm inside the residual branch, before the sublayer — because pre-norm keeps the residual path clean and is what lets deep stacks train stably; with dropout for regularization.

The relative scheme needs care in the implementation, so let me think it through concretely before I trust it. I'll build one sinusoidal embedding row per *possible offset*. For a sequence of length t, offsets run from −(t−1) (query at the end looking back to the start) to +(t−1) (query at the start looking forward to the end), which is 2t−1 rows. The attention score for query i attending to key j then needs to read the row for offset (j − i). To index a table whose first row is offset −(t−1), I add (t−1): the row for query i, key j is index (j − i) + (t − 1). Let me actually tabulate that for t = 4 to be sure I haven't flipped a sign. The index matrix (query rows i, key columns j) comes out as

```text
[[3, 4, 5, 6],
 [2, 3, 4, 5],
 [1, 2, 3, 4],
 [0, 1, 2, 3]]
```

The diagonal — query attending to itself, offset 0 — is all 3 = t − 1, which is the center row of a 2t−1 = 7-row table, exactly where offset 0 should sit. Moving right along a row (key ahead of query) increases the index; moving down (query ahead of key) decreases it. Min index 0, max 6, every entry inside [0, 2t−2], so the gather never reads out of bounds. That matches what I wanted, so the `+ t - 1` shift and the gather are right.

Now the convolution sublayer, and here I want to be deliberate. First normalize (LayerNorm) inside the residual branch. Then I want a *gate*, because not every channel/position should pass through equally — a learned multiplicative gate lets the module suppress irrelevant activations. The GLU does this: project to twice the channels with a pointwise (1×1) convolution, split in half, and gate one half by the sigmoid of the other, x = a ⊙ σ(b). So: pointwise conv with expansion factor 2, then GLU, which brings the channel count back down. Then the actual local operator: a single 1-D *depthwise* convolution — depthwise because it's per-channel and cheap, and it's the part that captures the local temporal pattern; I'll use a fairly wide kernel (32) so the local window is long enough to be useful for speech. After the depthwise conv I need to stabilize a deep conv stack, so BatchNorm right after the convolution. Then a Swish activation — smooth, and it tends to beat ReLU in deep nets. Then a second pointwise conv to project back to the model dimension, and dropout.

Let me trace the tensor shapes through that module on a toy case to make sure the channel arithmetic and the padding actually close. Take d = 8, kernel = 5, a (B, T) = (2, 7) input. After LayerNorm I transpose to (B, d, T) = (2, 8, 7) so the conv1d sees channels-first. The expansion pointwise conv takes 8 → 16 channels: (2, 16, 7). GLU on the channel dim halves it back: (2, 8, 7) — good, the gate restores d, so the depthwise conv sees the model dimension as I assumed. For the depthwise conv to preserve T with kernel 5 I pad by left = ⌊(k−1)/2⌋ = 2 and right = ⌊k/2⌋ = 2; that sums to 4 = k − 1, which is exactly the length a kernel-k conv eats, so the output stays length 7. The trace confirms (2, 8, 7) out. So the module is length-preserving and lands back at d channels, which is what the residual add requires.

And is the depthwise choice actually buying the parameter saving I claimed? A depthwise conv has k·d weights (each channel gets its own length-k kernel); a full 1-D conv mixing channels has k·d² (plus bias). For my toy d = 8, k = 5 that's 48 vs 328 params — about 7× fewer. The gap is exactly the factor d: at the real d = 512 a full conv of kernel 32 would be ~512× heavier than the depthwise one. That's a real saving, not a rounding one, and it's why a *wide* kernel of 32 is affordable here — depthwise keeps the cost linear in d instead of quadratic. So the convolution module is: LayerNorm → pointwise conv (×2) → GLU → depthwise conv (k=32) → BatchNorm → Swish → pointwise conv → dropout, all as a residual.

Now the feed-forward. The Transformer block has a single position-wise feed-forward layer after attention: two linears with a nonlinearity between, inner width 4× the model dim, in a residual. I'll keep that primitive (with Swish and pre-norm), but I want to reconsider its *placement*, because there's an argument that a single post-attention FFN is not the best arrangement. The Macaron view treats a Transformer block like a step of an ODE solver and argues that the lone feed-forward should be split into *two half-step* feed-forward layers — one before the attention and one after — each contributing a *half-weighted* residual, sandwiching the attention. The intuition: two symmetric half-steps around the mixing operation approximate the underlying dynamics better than one full step on one side. So instead of one FFN, I'll place two FFN modules, each with a 1/2 residual weight, bracketing the attention-and-convolution core.

Let me assemble the block and write its forward, then check the residual weighting is wired the way I mean. For input x_i to block i:

  x̃_i = x_i + (1/2) FFN(x_i)              # first half-step feed-forward
  x'_i = x̃_i + MHSA(x̃_i)                 # self-attention: global, content-based
  x''_i = x'_i + Conv(x'_i)               # convolution: local, on globalized features
  y_i = LayerNorm( x''_i + (1/2) FFN(x''_i) )   # second half-step FFN, then final norm

So the block is two Macaron half-step FFNs sandwiching the MHSA-then-Conv core, with a closing LayerNorm. I want to be sure the 1/2 only lands on the two FFNs and the attention/convolution carry full unit-weight residuals. A clean way to check: freeze the two FFNs to output a constant 1, zero out MHSA and Conv, drop the final LayerNorm to identity, and feed a zero input. Then the only thing that should move the activation is two adds of 0.5·1. Running that, the output is uniformly 1.0 — the two half-steps contribute 0.5 + 0.5 and nothing else, confirming the FFNs are halved while attention and convolution (here zeroed) would have entered at full weight. Good: the residual weights are wired as intended.

A quick end-to-end sanity pass too: a real block at d = 144, 4 heads, kernel 32 on a (2, 50, 144) input returns (2, 50, 144) with all-finite values. So the block is shape-preserving across the stack and nothing in the relative-attention gather, the GLU, or the padded depthwise conv blows up on a realistic shape.

In front of the stack of blocks I put a convolutional subsampling front end — a couple of stride-2 convolutions over the 80-channel filterbank features — to cut the time resolution before the expensive attention, then a linear projection to the model dimension and dropout. After that, N conformer blocks. Sizes: I'll instantiate small/medium/large budgeted encoders as 16 layers / dim 144 / 4 heads, 16 / 256 / 4, and 17 / 512 / 8, with the convolution kernel fixed at 32 throughout. As a rough budget check I counted the parameters of my from-scratch small encoder (16 layers, d = 144): ~8.4M. The reported small Conformer is 10.3M — same ballpark, and the gap is about what I'd expect from the pieces my skeleton omits (the LSTM-transducer decoder and joiner, plus the exact subsampling channel widths). So the budget is in the right neighborhood; I'd reconcile the last ~2M once the decoder is attached, not treat 8.4M as the final figure. The encoder feeds a single-LSTM-layer transducer decoder.

For training: dropout 0.1 in each residual unit (on each module's output before the add), a small L2 weight penalty, variational noise, and the Transformer learning-rate schedule — Adam with β₁=0.9, β₂=0.98, ε=1e-9, 10k warmup steps, and a peak learning rate of 0.05/√d where d is the encoder model dimension (so wider models get a proportionally smaller peak rate). SpecAugment on the filterbank inputs.

Let me write it, mirroring how I'd build it. First the two sublayer primitives — relative-position MHSA and the gated depthwise convolution module — then the macaron block.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, d, expansion=4, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, expansion * d), Swish(), nn.Dropout(p),
            nn.Linear(expansion * d, d), nn.Dropout(p))

    def forward(self, x):
        return self.net(self.ln(x))


class RelativeSinusoidalEncoding(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head

    def forward(self, length, device):
        positions = torch.arange(-(length - 1), length, device=device, dtype=torch.float32)
        freq = torch.arange(0, self.d_head, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (freq / self.d_head))
        angles = positions[:, None] * inv_freq[None, :]
        emb = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return emb[:, :self.d_head]


class RelMultiHeadSelfAttention(nn.Module):
    def __init__(self, d, heads, p=0.1):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.heads = heads
        self.d_head = d // heads
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.rel = RelativeSinusoidalEncoding(self.d_head)
        self.content_bias = nn.Parameter(torch.zeros(heads, self.d_head))
        self.pos_bias = nn.Parameter(torch.zeros(heads, self.d_head))
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        h = self.ln(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.heads, self.d_head).transpose(1, 2)

        content_scores = torch.einsum(
            "bhtd,bhsd->bhts", q + self.content_bias[None, :, None, :], k)
        rel = self.rel(t, x.device)
        rel_scores = torch.einsum(
            "bhtd,md->bhtm", q + self.pos_bias[None, :, None, :], rel)
        offsets = (
            torch.arange(t, device=x.device)[None, :]
            - torch.arange(t, device=x.device)[:, None]
            + t - 1
        )
        offsets = offsets.view(1, 1, t, t).expand(b, self.heads, t, t)
        rel_scores = torch.gather(rel_scores, dim=-1, index=offsets)

        scores = (content_scores + rel_scores) / math.sqrt(self.d_head)
        weights = self.drop(torch.softmax(scores, dim=-1))
        y = torch.einsum("bhts,bhsd->bhtd", weights, v)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.drop(self.out(y))


class ConvolutionModule(nn.Module):
    def __init__(self, d, kernel=32, p=0.1):
        super().__init__()
        self.kernel = kernel
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.act = Swish()
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.ln(x).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        left = (self.kernel - 1) // 2
        right = self.kernel // 2
        h = self.dw(F.pad(h, (left, right)))
        h = self.act(self.bn(h))
        h = self.drop(self.pw2(h))
        return h.transpose(1, 2)
```

Now the conformer block — two half-step FFNs sandwiching MHSA-then-Conv, closing with LayerNorm — exactly as derived:

```python
class ConformerBlock(nn.Module):
    def __init__(self, d, heads, kernel=32, p=0.1):
        super().__init__()
        self.ff1 = FeedForward(d, p=p)
        self.mhsa = RelMultiHeadSelfAttention(d, heads, p)
        self.conv = ConvolutionModule(d, kernel, p)
        self.ff2 = FeedForward(d, p=p)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)
```

And the encoder: convolutional subsampling, projection, a stack of conformer blocks, feeding a transducer decoder:

```python
class ConvSubsampling(nn.Module):                   # two stride-2 convs over filterbanks
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(1, channels, 3, 2)
        self.c2 = nn.Conv2d(channels, channels, 3, 2)

    def forward(self, x):                           # (B, T, n_mels)
        x = F.relu(self.c1(x.unsqueeze(1)))
        x = F.relu(self.c2(x))
        b, c, t, f = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, c * f)

class ConformerEncoder(nn.Module):
    def __init__(self, d=512, layers=17, heads=8, kernel=32):
        super().__init__()
        self.subsample = ConvSubsampling(d)
        self.proj = nn.LazyLinear(d)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(ConformerBlock(d, heads, kernel) for _ in range(layers))

    def forward(self, x):
        x = self.drop(self.proj(self.subsample(x)))
        for blk in self.blocks:
            x = blk(x)
        return x
```

Tracing the whole thing back: an ASR encoder must model both global, content-based dependencies and local, position-based feature patterns; self-attention is globally strong but a blunt local extractor, convolution is locally sharp but globally expensive, and patching a conv stack with a single averaged global summary can't model dynamic position-dependent global interactions; so I compose the two — self-attention to establish global context and a gated depthwise-convolution module to carve local detail on the globalized features (the order being my working choice, not a proven one) — each inside a pre-norm residual unit with relative positional attention I checked indexes correctly for length robustness; and following the ODE/Macaron view I bracket that core with two half-step feed-forward modules, whose 1/2 weighting I confirmed in isolation, and a closing layernorm, giving the conformer block.
