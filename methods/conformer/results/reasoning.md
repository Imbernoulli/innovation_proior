Let me start from the tension that actually limits an ASR encoder, because the architecture should fall out of it. I need to model two kinds of dependency in an utterance. One is global: a sound's identity can hinge on context far away — coarticulation, the word the speaker is mid-way through, the long-range rhythm. The other is local: the sharp, position-local feature patterns — an onset, a formant transition, a burst — that separate one nearby phone from the next. These are genuinely different computations, and the two leading building blocks each nail exactly one of them.

Self-attention relates every position to every other in a single layer, with weights computed from content. That is exactly what I want for global, content-based interactions — distance is no obstacle, one hop reaches anywhere. But stare at what attention *is* at a fine scale: each output is a softmax-weighted average over all positions. A weighted average is a smoothing operation; it's not built to extract a crisp, position-specific local pattern. So attention is globally strong and locally blunt.

A convolution is the mirror image. A kernel slides over a local window, so it picks up edges, onsets, short patterns cheaply, with translation equivariance — perfect for local acoustic detail. But its receptive field only grows by one window per layer. To see across a whole utterance I'd need a deep stack or fat kernels, i.e. I pay for global reach in depth or parameters. So convolution is locally strong and globally expensive.

I could try to patch one side. ContextNet takes the convolutional route and bolts on a squeeze-and-excitation module per block to inject global context. But think about what squeeze-and-excitation actually does: it global-average-pools the whole sequence into one vector and uses it to rescale channels. That's a *single*, static, averaged global summary — it can't say "position 40 should attend to position 200 but not position 30." It cannot model dynamic, position-dependent global interactions, which is precisely attention's gift. So patching convolution with a global average is a weak substitute for attention.

So neither block alone is right, and faking the missing half is weak. The honest move is to *use both* — let attention do the global, content-based interactions and let convolution do the local, position-based ones, each doing what it is genuinely good at. The question is only how to arrange them in a block.

The simplest arrangement is parallel: split the input into an attention branch and a convolution branch, run both, concatenate. That's a reasonable instinct — it lets both computations happen — but concatenation just sets the two representations side by side and leaves it to later layers to fuse them; the convolution never gets to act on attention's globally-mixed output, nor vice versa. I'd rather they *compose*: have one refine the other within the block, so local detail is extracted from a representation that already carries global context (or the reverse). Composition gives a deeper interaction than concatenation for the same budget.

If they compose, in which order? Two options: convolution then attention, or attention then convolution. Let me reason about it. If attention runs first, it produces a representation where every position is already a content-weighted mixture of the whole sequence — globally informed. Then the convolution operates on that, sharpening local patterns *within the already-globalized features*. That feels right for speech: establish the long-range content interactions, then let the local operator carve fine detail on top. The reverse — local first, then global — is a real alternative, but the speech-recognition setting favors the convolution module stacked after the self-attention module.

Now I have a block whose spine is: self-attention sublayer, then convolution sublayer, each in a residual unit. I want to nail down each sublayer's internals, with a reason for every choice.

Start with attention. Speech utterances vary enormously in length, and what matters acoustically is the *offset* between two frames, not their absolute indices. Absolute positional encodings tie the representation to position number, which generalizes poorly when test utterances are longer or shorter than training ones. So I'll use *relative* positional encoding — the attention score depends on the gap between query and key positions. That makes the encoder robust to utterance length and is the natural fit for a translation-equivariant signal. I'll also wrap the attention in a *pre-norm* residual unit — layer-norm inside the residual branch, before the sublayer — because pre-norm keeps the residual path clean and is what lets deep stacks train stably; with dropout for regularization.

Now the convolution sublayer, and here I want to be deliberate. First normalize (LayerNorm) inside the residual branch. Then I want a *gate*, because not every channel/position should pass through equally — a learned multiplicative gate lets the module suppress irrelevant activations. The GLU does this: project to twice the channels with a pointwise (1×1) convolution, split in half, and gate one half by the sigmoid of the other, x = a ⊙ σ(b). So: pointwise conv with expansion factor 2, then GLU, which brings the channel count back down. Then the actual local operator: a single 1-D *depthwise* convolution — depthwise because it's per-channel and cheap, and it's the part that captures the local temporal pattern; I'll use a fairly wide kernel (32) so the local window is long enough to be useful for speech. After the depthwise conv I need to stabilize a deep conv stack, so BatchNorm right after the convolution. Then a Swish activation — smooth, and it tends to beat ReLU in deep nets. Then a second pointwise conv to project back to the model dimension, and dropout. So the convolution module is: LayerNorm → pointwise conv (×2) → GLU → depthwise conv (k=32) → BatchNorm → Swish → pointwise conv → dropout, all as a residual.

Now the feed-forward. The Transformer block has a single position-wise feed-forward layer after attention: two linears with a nonlinearity between, inner width 4× the model dim, in a residual. I'll keep that primitive (with Swish and pre-norm), but I want to reconsider its *placement*, because there's an argument that a single post-attention FFN is not the best arrangement. The Macaron view treats a Transformer block like a step of an ODE solver and argues that the lone feed-forward should be split into *two half-step* feed-forward layers — one before the attention and one after — each contributing a *half-weighted* residual, sandwiching the attention. The intuition: two symmetric half-steps around the mixing operation approximate the underlying dynamics better than one full step on one side. So instead of one FFN, I'll place two FFN modules, each with a 1/2 residual weight, bracketing the attention-and-convolution core.

Let me assemble the block and write its forward exactly, because the half-step weights and the order are the whole point. For input x_i to block i:

  x̃_i = x_i + (1/2) FFN(x_i)              # first half-step feed-forward
  x'_i = x̃_i + MHSA(x̃_i)                 # self-attention: global, content-based
  x''_i = x'_i + Conv(x'_i)               # convolution: local, on globalized features
  y_i = LayerNorm( x''_i + (1/2) FFN(x''_i) )   # second half-step FFN, then final norm

So the block is two Macaron half-step FFNs sandwiching the MHSA-then-Conv core, with a closing LayerNorm. Note the residual weights: the two FFNs are half-step (the 1/2), while attention and convolution carry full unit-weight residuals — they are the main mixing operations, the FFNs are the symmetric ODE half-steps around them.

In front of the stack of blocks I put a convolutional subsampling front end — a couple of stride-2 convolutions over the 80-channel filterbank features — to cut the time resolution before the expensive attention, then a linear projection to the model dimension and dropout. After that, N conformer blocks. Sizes: I'll instantiate small/medium/large budgeted encoders as 16 layers / dim 144 / 4 heads at 10.3M parameters, 16 / 256 / 4 at 30.7M, and 17 / 512 / 8 at 118.8M, with the convolution kernel fixed at 32 throughout. The encoder feeds a single-LSTM-layer transducer decoder.

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

The causal chain, start to end: an ASR encoder must model both global, content-based dependencies and local, position-based feature patterns; self-attention is globally strong but a blunt local extractor, convolution is locally sharp but globally expensive, and patching a conv stack with a single averaged global summary can't model dynamic position-dependent global interactions; so I compose the two — self-attention first to establish global context, then a gated depthwise-convolution module to carve local detail on the globalized features — each inside a pre-norm residual unit with relative positional attention for length robustness; and following the ODE/Macaron view I bracket that core with two *half-step* feed-forward modules and a closing layernorm, giving the conformer block.
