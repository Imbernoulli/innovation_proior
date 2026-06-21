An end-to-end ASR encoder has to handle two different kinds of temporal structure at once. Global structure is the long-range, content-based context that resolves a phoneme from what happens many frames away, such as coarticulation or word-level rhythm. Local structure is the sharp, position-specific acoustic detail that distinguishes one sound from the next: onsets, formant transitions, and short spectral patterns. Self-attention captures the first one naturally because every position can attend to every other position in a single layer, but its output is a softmax-weighted average, which smooths away fine local detail. Convolution captures the second one naturally through a sliding kernel with translation equivariance, but its receptive field grows only one window per layer, so covering an entire utterance costs depth or parameters. Earlier attempts to patch one side, such as adding a squeeze-and-excitation module to a convolutional stack, only provide a single averaged global vector and cannot model dynamic, position-dependent global interactions.

The right move is to let each primitive do what it is actually good at and compose them inside a single block. Attention should establish global, content-based dependencies first. Then a convolution module should refine the representation by extracting local acoustic patterns from features that already carry long-range context. The block also needs a careful residual and feed-forward arrangement so the stack trains cleanly and parameter-efficiently. The resulting architecture is called Conformer.

Conformer is an end-to-end ASR encoder built from repeated Conformer blocks. Each block sandwiches a self-attention sublayer and a convolution sublayer between two half-step feed-forward layers, following the Macaron view of a Transformer block as an ODE step. The block starts with a feed-forward sublayer whose residual contribution is weighted by one half, then applies multi-head self-attention with a full residual, then applies a convolution module with a full residual, then applies a second half-step feed-forward sublayer, and finally applies layer normalization. The half-step feed-forward layers play the role of symmetric ODE half-steps around the main mixing operations rather than a single full step after attention.

The self-attention sublayer uses relative positional encoding instead of absolute positions. In speech, what matters is the offset between two frames, not their absolute indices, because utterance lengths vary widely at test time. Relative encoding makes the attention score depend on that offset, giving length robustness. The attention is wrapped in a pre-norm residual unit: layer normalization is applied inside the residual branch before the sublayer, and dropout is applied to the output before the residual add.

The convolution sublayer is a gated depthwise convolution module. It begins with layer normalization, followed by a pointwise convolution that expands the channel count by two, followed by a GLU that gates half the channels by the sigmoid of the other half. The gated output passes through a 1-D depthwise convolution with a wide kernel, then batch normalization, then a Swish activation, then a final pointwise convolution that projects back to the model dimension. Depthwise convolution keeps the parameter count low while the wide kernel captures a long enough local window for speech. The GLU gives the module a learned multiplicative gate so irrelevant channels can be suppressed.

The encoder is preceded by a convolutional subsampling front end that runs two stride-2 convolutions over the 80-channel log-mel filterbank features, cutting the time resolution before the expensive attention stack. A linear projection maps the subsampled features to the model dimension, dropout is applied, and then a stack of Conformer blocks processes the sequence. Typical configurations are 16 layers of dimension 144 with 4 attention heads at about 10.3 million parameters, 16 layers of dimension 256 with 4 heads at about 30.7 million parameters, or 17 layers of dimension 512 with 8 heads at about 118.8 million parameters. The depthwise convolution kernel is kept at size 32 across these variants. Training uses dropout of 0.1 on each module output, L2 regularization, variational noise, Adam with beta1 = 0.9 and beta2 = 0.98, and a Transformer learning-rate schedule with 10,000 warmup steps and a peak learning rate proportional to 0.05 divided by the square root of the model dimension, together with SpecAugment on the filterbank inputs.

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
            nn.Linear(d, expansion * d),
            Swish(),
            nn.Dropout(p),
            nn.Linear(expansion * d, d),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.net(self.ln(x))


class RelativeSinusoidalEncoding(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head

    def forward(self, length, device):
        positions = torch.arange(
            -(length - 1), length, device=device, dtype=torch.float32)
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

        q_content = q + self.content_bias[None, :, None, :]
        q_pos = q + self.pos_bias[None, :, None, :]
        content_scores = torch.einsum("bhtd,bhsd->bhts", q_content, k)

        rel = self.rel(t, x.device)
        rel_scores = torch.einsum("bhtd,md->bhtm", q_pos, rel)
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


class ConvSubsampling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(1, channels, 3, stride=2)
        self.c2 = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x):
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
        self.blocks = nn.ModuleList(
            ConformerBlock(d, heads, kernel) for _ in range(layers))

    def forward(self, x):
        x = self.drop(self.proj(self.subsample(x)))
        for block in self.blocks:
            x = block(x)
        return x
```
