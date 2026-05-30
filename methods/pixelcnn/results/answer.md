# PixelRNN and PixelCNN

## Problem

Model the distribution `p(x)` of natural images with a single model that is at once **expressive**,
**tractable** (exact, evaluable likelihood — not a bound), and **scalable**. Latent-variable models
(VAEs) and undirected models give only bounds or estimates of the likelihood; continuous-density
autoregressive models impose a parametric shape on inherently discrete pixel values.

## Key idea

Factorize the image likelihood with the chain rule over a fixed raster-scan ordering, and within each
pixel over its color channels, then model every conditional with a neural network constrained to read
only earlier pixels:

```
p(x) = Π_{i=1}^{n^2} p(x_i | x_1, ..., x_{i-1}),
p(x_i | x_{<i}) = p(x_{i,R} | x_{<i}) · p(x_{i,G} | x_{<i}, x_{i,R}) · p(x_{i,B} | x_{<i}, x_{i,R}, x_{i,G}).
```

Each conditional is a **256-way softmax** over discrete intensities `{0, ..., 255}` (a single sigmoid
for binary images): arbitrarily multimodal, no shape prior, no probability mass outside the valid
range. Training minimizes the exact negative log-likelihood — the sum of per-pixel categorical
cross-entropies — and runs fully in parallel (all conditionals at once, since the "previous pixels"
are the known input); generation is sequential (sample a value, feed it back, repeat).

## Enforcing the ordering: masked convolutions

A spatial convolution is made causal by **zeroing kernel weights** that touch the current pixel or any
later one: all rows below center → 0; in the center row, columns at/after center → 0.

- **Mask A** (first layer only) additionally zeroes the center weight, because at the input the center
  is the raw value `x_i` being predicted.
- **Mask B** (all later layers) keeps the center weight, because there it is a feature that — by
  induction — already depends only on `x_{<i}`.

The mask extends into the channel dimension to enforce the R→G→B order: with feature maps split into
three color groups, mask A lets group `g` read only earlier groups (G←R, B←R,G; no self), and mask B
lets `g` read groups up to and including itself.

## Architectures

- **Row LSTM.** Processes the image row by row; a `k×1` masked convolution computes the four LSTM gate
  inputs for a whole row in parallel, and a `k×1` state-to-state convolution carries the recurrence
  from the row above:
  ```
  [o, f, i, g] = activation(K^{ss} ⊛ h_{i-1} + K^{is} ⊛ x_i)   # σ for o,f,i ; tanh for g
  c_i = f ⊙ c_{i-1} + i ⊙ g
  h_i = o ⊙ tanh(c_i)
  ```
  Receptive field is a **triangle** above the pixel (it misses far-side pixels on earlier rows).

- **Diagonal BiLSTM.** **Skews** each row right by its index so diagonals become columns; a `1×1`
  input-to-state conv plus a `2×1` column-wise state-to-state conv sweep each diagonal in parallel.
  Two mirrored scans (the right one shifted down one row before adding, to preserve causality) give an
  **unbounded, complete** receptive field for any image size.

- **PixelCNN.** A fully-convolutional stack (first layer `7×7` mask A, then `3×3` mask B), no pooling,
  preserving resolution; fastest to train but with a **bounded** receptive field.

- **Residual connections** around each recurrent layer (input `2h` features → `h` per gate →
  recurrence → `1×1` upsample to `2h` → add input) let the networks train to ~12 layers; depth keeps
  improving likelihood.

## The blind spot (and how to close it)

Stacking the single left-biased mask never reaches a triangular wedge of pixels just to the
upper-right of the current position — context the model is *allowed* to use but architecturally
cannot. The fix (gated, two-stream PixelCNN) splits the field into a **vertical stack** over all rows
strictly above and a **horizontal stack** over the current row strictly to the left, with the vertical
stack feeding the horizontal stack one-way; together they cover the full valid context with no gap.
ReLUs are replaced by a **gated activation** `tanh(a) ⊙ σ(b)` (LSTM-style), and the residual is placed
on the horizontal stack only — never on the causal input layer, which would leak `x_i`.

## Code

A minimal masked-convolution PixelCNN (grounded in jzbontar's clean implementation):

```python
import torch
import torch.nn.functional as F
from torch import nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0   # center row, from center on
        self.mask[:, :, kH // 2 + 1:] = 0                              # all rows below center

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


def build_pixelcnn(fm=64, n_values=256):
    return nn.Sequential(
        MaskedConv2d("A", 1, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
        *[m for _ in range(7) for m in (
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True))],
        nn.Conv2d(fm, n_values, 1),
    )


def train_step(net, x, opt):
    target = (x[:, 0] * 255).long()
    loss = F.cross_entropy(net(x), target)          # exact NLL = summed per-pixel cross-entropy
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def sample(net, n, H, W):
    img = torch.zeros(n, 1, H, W)
    for i in range(H):
        for j in range(W):
            probs = F.softmax(net(img)[:, :, i, j], dim=1)
            img[:, 0, i, j] = torch.multinomial(probs, 1).float() / 255.
    return img
```

The gated two-stream PixelCNN that removes the blind spot (grounded in EugenHotaj's
pytorch-generative implementation):

```python
class GatedActivation(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class GatedPixelCNNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, mask_center=False):
        super().__init__()
        assert kernel_size % 2 == 1
        self.mask_center = mask_center
        p = (kernel_size - 1) // 2
        self.act = GatedActivation()
        # vertical stack: all rows strictly above (top-padded, cropped/shifted down)
        self.v_1xN = nn.Conv2d(in_ch, out_ch, (1, kernel_size), padding=(0, p))
        self.v_Nx1 = nn.Conv2d(out_ch, 2 * out_ch, (kernel_size // 2 + 1, 1), padding=(p + 1, 0))
        self.v_1x1 = nn.Conv2d(in_ch, 2 * out_ch, 1)
        self.link = nn.Conv2d(2 * out_ch, 2 * out_ch, 1)            # vertical -> horizontal (one-way)
        # horizontal stack: current row strictly to the left
        self.h_1xN = nn.Conv2d(in_ch, 2 * out_ch, (1, kernel_size // 2 + 1),
                               padding=(0, p + int(mask_center)))
        self.h_res = nn.Conv2d(out_ch, out_ch, 1)
        self.h_skip = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, v_in, h_in):
        _, _, H, W = v_in.shape
        v = self.v_Nx1(self.v_1xN(v_in))[:, :, :H, :]
        link = self.link(v)
        v = self.act(v + self.v_1x1(v_in))
        h = self.act(link + self.h_1xN(h_in)[:, :, :, :W])
        skip = self.h_skip(h)
        h = self.h_res(h)
        if not self.mask_center:                                    # no residual on the causal layer
            h = h + h_in
        return v, h, skip


class GatedPixelCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_gated=10, ch=128, head_ch=32):
        super().__init__()
        self.input = GatedPixelCNNLayer(in_ch, ch, kernel_size=7, mask_center=True)   # mask A
        self.gated = nn.ModuleList(
            GatedPixelCNNLayer(ch, ch, kernel_size=3, mask_center=False) for _ in range(n_gated)
        )
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv2d(ch, head_ch, 1), nn.ReLU(), nn.Conv2d(head_ch, out_ch, 1)
        )

    def forward(self, x):
        v, h, skip = self.input(x, x)
        for layer in self.gated:
            v, h, s = layer(v, h)
            skip = skip + s
        return self.head(skip)
```
