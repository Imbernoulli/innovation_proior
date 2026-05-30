# PixelRNN / PixelCNN

## Problem

Build an exact, tractable, expressive density model `p(x)` over natural images — one that gives the exact likelihood of an image and can sample new ones, while capturing the full nonlinear, long-range, multimodal dependencies between pixels (including the dependencies between the R, G, B channels within a pixel) with no independence shortcuts.

## Key idea

Factorize the joint over pixels in raster order with the chain rule and model each conditional with a deep neural network:

`p(x) = Π_{i=1}^{n²} p(x_i | x_1, …, x_{i-1})`,

extended within each pixel over color channels,
`p(x_i | x_{<i}) = p(x_{i,R} | x_{<i}) · p(x_{i,G} | x_{<i}, x_{i,R}) · p(x_{i,B} | x_{<i}, x_{i,R}, x_{i,G})`.

Each conditional is a **discrete 256-way softmax** (multinomial), not a continuous density — arbitrarily multimodal, no shape assumption, easier to train. Likelihoods are computed in parallel over all positions during training (true pixels known); generation is sequential.

Causality — position `i` may see only `x_{<i}` — is enforced by **masked convolutions**: zero the kernel weights connecting a position to itself or future positions. **Mask A** (first layer only) excludes a color's self-connection at the current pixel to block the raw-value leak; **Mask B** (all later layers) restores the self-connection, safe once the leak is structurally impossible.

Three model families build the conditional:
- **PixelCNN** — a stack of masked convolutions preserving spatial resolution (no pooling); fastest (all positions in parallel at train time) but a *bounded* receptive field.
- **Row LSTM** — a recurrent layer that advances an LSTM state down the rows, computing a whole row at once with a `k × 1` convolution; *triangular* receptive field (misses the sides).
- **Diagonal BiLSTM** — skews the image so diagonals become columns, sweeps them with a `2 × 1` recurrent kernel in two directions (the second shifted down one row to avoid the future), reaching the *entire* available context.

Residual connections enable depth (up to twelve layers); a multi-scale variant conditions a full-resolution PixelRNN on a cheaply generated low-resolution image to inject global structure.

## LSTM recurrence (Row / Diagonal)

`[o_i, f_i, i_i, g_i] = σ(K^{ss} ⊛ h_{i-1} + K^{is} ⊛ x_i)`, `c_i = f_i ⊙ c_{i-1} + i_i ⊙ g_i`, `h_i = o_i ⊙ tanh(c_i)`, with σ logistic for the `o, f, i` gates and tanh for the content gate `g`; the input-to-state contribution `K^{is}` is precomputed for the whole map by a masked convolution.

## Architecture and hyperparameters

First layer `7×7` mask A; body of residual blocks (PixelCNN: `3×3` mask B; Row LSTM: i-s `3×1` mask B, s-s `3×1` unmasked; Diagonal BiLSTM: i-s `1×1` mask B, s-s `1×2` unmasked); two `ReLU + 1×1` mask-B layers (1024 feature maps for CIFAR/ImageNet, 32 for MNIST); 256-way softmax per channel (sigmoid for MNIST). MNIST: 7-layer Diagonal BiLSTM, `h = 16`. CIFAR-10: 12-layer Row/Diagonal BiLSTM, `h = 128`; PixelCNN 15 layers, `h = 128`. ImageNet 32×32: 12-layer Row LSTM, `h = 384`. ImageNet 64×64: 4-layer Row LSTM, `h = 512` (no residuals).

## Code

```python
import torch
from torch import nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0   # center row: self (A) / future
        self.mask[:, :, kH // 2 + 1:] = 0                            # rows below center: future
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# PixelCNN: stack of masked convs, resolution preserved, 256-way head
fm = 128
pixelcnn = nn.Sequential(
    MaskedConv2d('A', 1, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    *[m for _ in range(7) for m in (
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True))],
    nn.Conv2d(fm, 256, 1))


class RowLSTM(nn.Module):
    def __init__(self, h, k=3):
        super().__init__()
        self.input_to_state = MaskedConv2d('B', h, 4 * h, (k, 1), 1, (k // 2, 0))
        self.state_to_state = nn.Conv1d(h, 4 * h, k, padding=k // 2)
        self.h = h
    def forward(self, x):                      # x: [B, h, H, W]
        gates_is = self.input_to_state(x)
        B, _, H, W = x.shape
        h_prev = x.new_zeros(B, self.h, W); c_prev = x.new_zeros(B, self.h, W)
        outs = []
        for i in range(H):
            g = gates_is[:, :, i] + self.state_to_state(h_prev)
            o, f, ii, gg = g.chunk(4, dim=1)
            c_prev = torch.sigmoid(f) * c_prev + torch.sigmoid(ii) * torch.tanh(gg)
            h_prev = torch.sigmoid(o) * torch.tanh(c_prev)
            outs.append(h_prev)
        return torch.stack(outs, dim=2)        # [B, h, H, W]


# training: exact likelihood as 256-way cross-entropy, conditionals in parallel
optimizer = torch.optim.Adam(pixelcnn.parameters())
for img in loader:                             # img in [0,1], shape [B,1,H,W]
    target = (img[:, 0] * 255).long()          # discrete target in {0..255}
    logits = pixelcnn(img)                     # [B, 256, H, W]
    loss = F.cross_entropy(logits, target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()


# generation: sequential, feed each sampled pixel back in
@torch.no_grad()
def sample(net, n, H, W):
    x = torch.zeros(n, 1, H, W)
    for r in range(H):
        for c in range(W):
            probs = F.softmax(net(x)[:, :, r, c], dim=1)
            x[:, 0, r, c] = torch.multinomial(probs, 1).float() / 255.
    return x
```

For color images the feature channels are split into R/G/B groups and the mask additionally enforces the R→G→B ordering at the current pixel, with a 256-way softmax per channel.
