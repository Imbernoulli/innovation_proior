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

- **Row LSTM.** Processes the image row by row; a masked one-dimensional convolution of length `k`
  along the row computes the four LSTM gate inputs for a whole row in parallel, and a matching
  row-wise state-to-state convolution carries the recurrence from the row above:
  ```
  [o, f, i, g] = activation(K^{ss} ⊛ h_{i-1} + K^{is} ⊛ x_i)   # σ for o,f,i ; tanh for g
  c_i = f ⊙ c_{i-1} + i ⊙ g
  h_i = o ⊙ tanh(c_i)
  ```
  Receptive field is a **triangle** above the pixel (it misses far-side pixels on earlier rows).

- **Diagonal BiLSTM.** **Skews** each row right by its index so one diagonal family becomes columns; a
  `1×1` input-to-state conv plus a `2×1` column-wise state-to-state conv sweep each diagonal in
  parallel.
  Two mirrored scans (the right one shifted down one row before adding, to preserve causality) give an
  **unbounded, complete** receptive field for any image size.

- **PixelCNN.** A fully-convolutional stack (first layer `7×7` mask A, then `3×3` mask B), no pooling,
  preserving resolution; fastest to train but with a **bounded** receptive field.

- **Residual connections** around each recurrent layer (input `2h` features → `h` per gate →
  recurrence → `1×1` upsample to `2h` → add input) give signal and gradients a direct path through
  deep recurrent stacks.

## The blind spot (and how to close it)

Stacking the single left-biased mask never reaches a triangular wedge of pixels just to the
upper-right of the current position — context the model is *allowed* to use but architecturally
cannot. The gated two-stream fix splits the finite convolutional field into a **vertical stack** over
above-row information and a **horizontal stack** over the current row strictly to the left, with the
vertical stack feeding the horizontal stack one-way; within the finite range reached by the stack,
the upper-right blind wedge is gone. ReLUs are replaced by a **gated activation** `tanh(a) ⊙ σ(b)`
(LSTM-style), and the horizontal residual is omitted on the causal input layer, where it would leak
`x_i`.

## Code

A compact implementation of the masked-convolution and two-stream variants:

```python
import torch
import torch.nn.functional as F
from torch import nn


class OrderedConv2d(nn.Conv2d):
    """Mask the kernel so position i never reads positions >= i.
    type 'A' excludes the center weight; type 'B' keeps it."""
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class GatedActivation(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class OrderedImageLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, first_layer=False):
        super().__init__()
        assert kernel_size % 2 == 1
        self.first_layer = first_layer
        p = (kernel_size - 1) // 2
        self.act = GatedActivation()

        self.v_1xN = nn.Conv2d(in_ch, out_ch, (1, kernel_size), padding=(0, p))
        self.v_Nx1 = nn.Conv2d(out_ch, 2 * out_ch, (kernel_size // 2 + 1, 1),
                               padding=(p + 1, 0))
        self.v_1x1 = nn.Conv2d(in_ch, 2 * out_ch, 1)
        self.link = nn.Conv2d(2 * out_ch, 2 * out_ch, 1)

        self.h_1xN = nn.Conv2d(in_ch, 2 * out_ch, (1, kernel_size // 2 + 1),
                               padding=(0, p + int(first_layer)))
        self.h_res = nn.Conv2d(out_ch, out_ch, 1)
        self.h_skip = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, above_state, row_state):
        v_in, h_in = above_state, row_state
        _, _, H, W = v_in.shape
        v = self.v_Nx1(self.v_1xN(v_in))[:, :, :H, :]
        link = self.link(v)
        v = self.act(v + self.v_1x1(v_in))

        h = link + self.h_1xN(h_in)[:, :, :, :W]
        h = self.act(h)
        skip = self.h_skip(h)
        h = self.h_res(h)
        if not self.first_layer:
            h = h + h_in
        return v, h, skip


class ImageDensityModel(nn.Module):
    def __init__(self, in_ch=1, n_values=256, n_layers=10, channels=64,
                 split_state=True, head_channels=64):
        super().__init__()
        self.split_state = split_state
        if split_state:
            self.input_layer = OrderedImageLayer(in_ch, channels, kernel_size=7, first_layer=True)
            self.layers = nn.ModuleList(
                OrderedImageLayer(channels, channels, kernel_size=3, first_layer=False)
                for _ in range(n_layers)
            )
            head_in = channels
        else:
            self.input_layer = OrderedConv2d("A", in_ch, channels, 7, padding=3, bias=False)
            self.layers = nn.ModuleList(
                OrderedConv2d("B", channels, channels, 3, padding=1, bias=False)
                for _ in range(n_layers)
            )
            head_in = channels
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv2d(head_in, head_channels, 1),
            nn.ReLU(), nn.Conv2d(head_channels, n_values, 1)
        )

    def forward(self, x):
        if self.split_state:
            v, h, skip = self.input_layer(x, x)
            for layer in self.layers:
                v, h, s = layer(v, h)
                skip = skip + s
            return self.head(skip)

        h = F.relu(self.input_layer(x))
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.head(h)


def nll_loss(logits, target_pixels):
    per_pixel = F.cross_entropy(logits, target_pixels, reduction="none")
    return per_pixel.flatten(1).sum(1).mean()


def train_step(model, x, opt):
    target = (x[:, 0] * 255).long()
    loss = nll_loss(model(x), target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def sample(model, n, H, W, n_values=256):
    device = next(model.parameters()).device
    img = torch.zeros(n, 1, H, W, device=device)
    for i in range(H):
        for j in range(W):
            logits = model(img)[:, :, i, j]
            probs = F.softmax(logits, dim=1)
            img[:, 0, i, j] = torch.multinomial(probs, 1).squeeze(1).float() / (n_values - 1)
    return img
```
