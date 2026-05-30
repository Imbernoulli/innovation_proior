OK, let me think about what I'm really after. I want a probability model over natural images — a genuine `p(x)` — that I can do two things with: compute the exact likelihood of any image, and sample new images. If I have that, I get image compression, inpainting, deblurring, and generation almost for free, and conditioned on side information I could even do text-to-image or next-frame prediction. The trouble is that images are high-dimensional and ferociously structured: pixels have nonlinear, long-range, multimodal correlations. So I'm caught in the usual vise — expressive models tend to be intractable, tractable models tend to be too weak. I want both.

Let me survey what I'd reach for and find exactly where each one breaks, because the cracks will point me somewhere.

The popular thing is a latent-variable model — a VAE. Encode to a latent, decode back, train on a bound. But two problems for *my* goal. The inference is intractable: I only ever get a *lower bound* on the likelihood, the ELBO, never the exact `log p(x)` I said I wanted. And the latent structure carries independence assumptions that I'd rather not impose, because the whole point is to capture the full dependency web between pixels. So latent variables cost me the exact likelihood. Set aside.

What gives me an *exact* likelihood with no approximate inference? The autoregressive trick: pick an ordering of the pixels and use the chain rule to factor the joint into a product of conditionals,

  p(x) = Π_{i=1}^{n²} p(x_i | x_1, …, x_{i-1}).

Take the pixels row by row, left to right, so `x_1 … x_{n²}` is just the raster scan of the image. Now there's no latent, no bound — the likelihood is exactly this product, and I can sample ancestrally: draw `x_1`, then `x_2 | x_1`, and so on. This is the fully-visible / NADE family. The catch is that I've converted density estimation into a *sequence* problem: predict the next pixel given all the previous ones. The model is exactly as good as how well I can model each conditional `p(x_i | x_{<i})`. The old fully-visible nets used shallow conditionals and couldn't capture the nonlinear long-range structure of real images. So tractability is solved; expressiveness of the conditional is the open problem.

What models conditionals-given-a-long-history well? Recurrent nets. An RNN gives a compact, *shared* parametrization of a whole series of conditionals — the same weights predict every step — and they're proven on hard sequence tasks (handwriting, character prediction, translation). And for images specifically, a two-dimensional LSTM that sweeps from the top-left pixel to the bottom-right has shown promise on grayscale: its 2D recurrent state propagates signals both left-to-right and top-to-bottom, so it can in principle carry the long-range information that object and scene structure need. That's the seed. The issue is it's slow and was demonstrated on grayscale; I want to make 2D recurrence fast and scale it to large color images. So the plan crystallizes: keep the exact autoregressive factorization, and make the conditional model a powerful, fast 2D recurrent network.

Before architecture, one modeling decision that turns out to matter a lot: what *form* does each conditional `p(x_i | x_{<i})` take? Prior pixel models used a continuous density over intensity. But a continuous density forces me to commit to a *shape* — Gaussian, logistic, a mixture — and pixel conditionals in real images are weirdly multimodal (an edge pixel might be very dark or very light but rarely in between). Why fight that? A pixel intensity is one of 256 discrete values. So let me model each conditional as a plain *multinomial* over those 256 values, produced by a softmax. No assumption about the density's shape at all — it can be arbitrarily multimodal, spiky, bimodal, whatever the data wants. It's representationally simple and, as I'll see when I train it, easier to learn and gives better likelihood than a continuous parametrization. For binary MNIST a single sigmoid per pixel suffices.

There's a subtlety I almost glossed: a pixel isn't one value, it's three — R, G, B. Are they independent given the context? Definitely not; color channels within a pixel are correlated. So I shouldn't predict them jointly-but-independently. Apply the chain rule *within* the pixel too: order the channels R → G → B and condition each on the earlier ones,

  p(x_i | x_{<i}) = p(x_{i,R} | x_{<i}) · p(x_{i,G} | x_{<i}, x_{i,R}) · p(x_{i,B} | x_{<i}, x_{i,R}, x_{i,G}).

So G sees R of the same pixel, B sees R and G. This will force a constraint on how I wire the network, and I'll come back to it.

Now the architecture. I want to compute, at every pixel position, the conditional distribution given the context — and I want to do it efficiently. Here's the tension that shapes everything: at *training* time I already know all the true pixels, so I'd love to compute all the conditionals *in parallel* in one forward pass (this is teacher forcing — feed the true image, read off every position's prediction at once). At *generation* time it's unavoidably sequential, since each sampled pixel must be fed back before the next. So the architecture should let me compute every position's prediction simultaneously during training while strictly respecting the causal ordering: position `i` must depend only on `x_{<i}`, never on itself or any future pixel. If even a sliver of `x_i` leaks into its own prediction, the likelihood is cheating and generation will be inconsistent.

How do I make a layer that sees every position at once but only looks *backward*? A convolution naturally sees a neighborhood around each position — including the center and the future. So zero out the parts of the convolution kernel that reach the current and future pixels. A *mask* on the weights. Concretely, for a kernel centered on the pixel being predicted, I keep the weights for pixels above and to the left, and zero the weights for the center and everything below/right. Multiply the kernel by a 0/1 mask before convolving and the receptive field becomes strictly causal, while I still compute all positions in one shot.

But the RGB chain rule complicates the mask. Split the feature channels at every layer into three groups, one per color. When predicting R of the current pixel, only earlier pixels (left/above) may be used. When predicting G, I may additionally use the *current pixel's R*. For B, the current R and G. So the mask isn't just "spatially causal," it also has to respect this color ordering at the center pixel. And there's a one-time wrinkle at the very first layer. In that first layer the input *is* the raw image, so I must absolutely forbid a color from seeing *itself* at the current pixel — otherwise R's prediction would peek at the true R. So define a strict mask — call it **mask A** — used only on the first layer, which excludes the connection from a color to itself at the current pixel (the center of its own channel is zeroed). After the first layer, the value at a position is a *feature*, not the raw pixel, and the first layer already guaranteed no direct self-access — so I can safely *relax* and allow a color to connect to itself. Call that **mask B**, used on all subsequent layers; it keeps the center connection (within the allowed color ordering) and only kills the future. The reason for two masks is exactly this: A prevents the raw-value leak at the input; B restores capacity once the leak is structurally impossible.

So that's one architecture already: stack masked convolutions, preserve spatial resolution throughout (no pooling — I need a prediction at every pixel), and end with the 256-way softmax head. Call it the **PixelCNN**. The first layer is a large masked conv (7×7, mask A) to grab a wide initial context; then a stack of masked 3×3 (mask B) convolutional layers; then a couple of ReLU + 1×1 conv layers; then the per-channel 256-way softmax. It computes every position in parallel during training, which makes it the *fast* member of the family. But notice its receptive field is *bounded* — fixed by the number and size of the conv layers. A pixel far enough away from `x_i` simply never enters its conditional. For long-range structure that's a real limitation; it's the price of using plain convolutions.

Can I get an *unbounded* receptive field — full context — and keep the parallelism? This is where the recurrence comes back. Let me design a recurrent layer that uses convolution to compute a whole spatial slice at once.

First attempt: the **Row LSTM**. Process the image row by row, top to bottom, and at each step compute the LSTM state for an *entire row* at once. The input-to-state part — the contribution of the current input to the four gates — I can precompute for the whole 2D image map with a single masked convolution of size `k × 1` (a vertical strip), producing a `4h × n × n` tensor: the four gate pre-activations at every position. Then the recurrent state-to-state part sweeps down the rows. Given the previous row's hidden and cell states `h_{i-1}, c_{i-1}`, the new states follow the standard LSTM equations, but with the gate computations done by convolution along the row:

  [o_i, f_i, i_i, g_i] = σ( K^{ss} ⊛ h_{i-1} + K^{is} ⊛ x_i ),
  c_i = f_i ⊙ c_{i-1} + i_i ⊙ g_i,
  h_i = o_i ⊙ tanh(c_i),

where ⊛ is convolution, ⊙ is elementwise product, σ is the logistic sigmoid for the output/forget/input gates `o, f, i`, and tanh for the content gate `g`. `K^{ss}` is the state-to-state kernel (size `3 × 1`), `K^{is}` the precomputed input-to-state contribution. Each step advances one whole row. The convolution's weight sharing gives translation invariance along the row, which is exactly what I want.

But what context does a pixel actually see under the Row LSTM? Trace it: the state flows downward row by row, and within the input-to-state strip it reaches a few columns left and right. Following the recurrence upward, the dependency region above a pixel is *triangular* — it fans out as you go up but never reaches pixels far out to the *sides* in nearby rows. So the Row LSTM has a triangular receptive field that misses part of the available context. Better than bounded-and-small, but still not the *entire* context I claimed I wanted.

So, second attempt: the **Diagonal BiLSTM**, built specifically to reach the *entire* context for any image size while still parallelizing. The idea: instead of sweeping rows, sweep *diagonals*. If I process the image along diagonals from a top corner toward the opposite bottom corner, then by the time I compute a given pixel, the diagonal sweep has already visited everything above-and-left of it — the full causal context. The problem is that a diagonal isn't a nice axis-aligned line for a convolution. So *skew* the image: offset each row by one position to the right relative to the row above it, turning the `n × n` map into an `n × (2n−1)` map in which the original diagonals become straight *columns*. Now a column-wise convolution runs along what used to be diagonals.

In skewed space the layer is cheap: the input-to-state part is just a `1 × 1` convolution `K^{is}` producing the `4h × n × n` gate tensor, and the state-to-state part is a *column-wise* convolution `K^{ss}` with a tiny `2 × 1` kernel, plugged into the same LSTM equations above. A `2 × 1` kernel is deliberate — it processes a minimal amount of information per step, which keeps each step highly nonlinear, and making the kernel bigger wouldn't help because the diagonal sweep already gives a *global* receptive field; there's nothing further to reach. After the recurrence I skew the output map back to `n × n` by removing the offsets.

One direction of diagonal sweep covers everything above-and-left. To also cover above-and-right I run a *second* diagonal sweep in the mirror direction, giving a bidirectional layer — hence Bi-LSTM. But naively adding the two would let the right-going sweep peek at *future* pixels (below-and-right). So before combining, shift the right (second-direction) output map *down by one row*, which removes any access to the current and future pixels, then add it to the left output map. Now every pixel sees its complete legal context — above and to both sides — and never the future. Full dependency field, exactly what the Row LSTM couldn't give.

These recurrent layers are powerful but deep stacks of them are hard to train, and I want up to twelve. So add residual connections around each LSTM layer (add the layer's input to its output). The bookkeeping: the input map carries `2h` features; the input-to-state step reduces to `h` features per gate; after the recurrent layer I upsample back to `2h` per position with a `1 × 1` convolution and add the input map. This speeds convergence and propagates signal directly through depth, and unlike gating-along-depth schemes it needs no extra gates. I can also add learnable skip connections straight from each layer to the output.

There's one more thing I'd like that none of these gives cheaply: *global* structure on large images without a huge sequential cost. The Diagonal BiLSTM has global context but every state is sequential; PixelCNN is parallel but its field is bounded. Compromise with a **multi-scale** scheme: first run an *unconditional* PixelRNN to generate a small `s × s` subsampled version of the image — cheap, because it's tiny — and then run a *conditional* PixelRNN that takes that small image as extra input and generates the full `n × n` image. The conditioning is by biasing: upsample the `s × s` image to a `c × n × n` feature map with a small deconvolutional network, and for each layer of the conditional network map that to a `4h × n × n` tensor (a `1 × 1` *unmasked* convolution — unmasked is fine because the conditioning image is fully known and global) and add it into that layer's input-to-state gates. The small image injects global layout; the conditional PixelRNN fills in the detail.

Let me pin down the concrete network and then write the masked layer, since the mask is the load-bearing piece. The first layer is a `7 × 7` mask-A convolution. Then the body: for the PixelCNN, repeated `3 × 3` mask-B convolutions; for the Row LSTM, input-to-state `3 × 1` mask B and state-to-state `3 × 1` unmasked; for the Diagonal BiLSTM, input-to-state `1 × 1` mask B and state-to-state `1 × 2` unmasked (the state-to-state convs are unmasked because the masking is already enforced by the sweep geometry and the first layer). Then two layers of ReLU + `1 × 1` mask-B conv (1024 feature maps for CIFAR/ImageNet, 32 for MNIST), and finally the 256-way softmax per color channel (sigmoid for binary MNIST). Sizes: MNIST a 7-layer Diagonal BiLSTM with `h = 16`; CIFAR-10 a 12-layer Row or Diagonal BiLSTM with `h = 128`, or a 15-layer PixelCNN with `h = 128`; ImageNet 32×32 a 12-layer Row LSTM with `h = 384`; ImageNet 64×64 a 4-layer Row LSTM with `h = 512` (no residuals at that size).

Here is the masked convolution, the heart of the causal wiring — written for the single-channel case where the mask just enforces spatial causality, with the A/B distinction being exactly "exclude the center pixel" vs "include it":

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
        # in the center row, zero the center pixel (mask A) or keep it (mask B), and everything to its right
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        # zero every row below the center: those are future pixels
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask          # blank out future/self connections before convolving
        return super().forward(x)
```

`mask_type == 'A'` evaluates to 0, so `kW//2 + 0` zeros the center column too — no self-connection, used on the first layer. `mask_type == 'B'` evaluates to 1, so `kW//2 + 1` starts zeroing *after* the center — the self-connection is kept, used on every later layer.

The PixelCNN is then just a stack of these, preserving resolution, ending in a 256-way head:

```python
fm = 128
net = nn.Sequential(
    MaskedConv2d('A', 1, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),   # first layer: strict mask
    *[layer for _ in range(7) for layer in (
        MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True))],  # body: relaxed mask
    nn.Conv2d(fm, 256, 1))                                                              # head: 256 logits per pixel
```

Training maximizes the exact log-likelihood, computed in parallel over all positions because the true image is known — which is precisely a 256-way cross-entropy between the predicted distribution and the true discrete pixel value:

```python
target = (img[:, 0] * 255).long()            # true intensity in {0..255}
logits = net(img)                            # [B, 256, H, W]
loss = F.cross_entropy(logits, target)       # negative log-likelihood per pixel
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

And the Row LSTM, sketched, to show how the recurrent layer reuses convolution for the gates and runs the standard LSTM recurrence down the rows:

```python
class RowLSTM(nn.Module):
    def __init__(self, h, k=3):
        super().__init__()
        self.input_to_state = MaskedConv2d('B', h, 4 * h, (k, 1), 1, (k // 2, 0))  # k x 1, masked, all rows at once
        self.state_to_state = nn.Conv1d(h, 4 * h, k, padding=k // 2)               # 3x1 recurrent kernel, unmasked
        self.h = h
    def forward(self, x):                          # x: [B, h, H, W]
        gates_is = self.input_to_state(x)          # [B, 4h, H, W]: input-to-state for every position, precomputed
        B, _, H, W = x.shape
        h_prev = x.new_zeros(B, self.h, W); c_prev = x.new_zeros(B, self.h, W)
        outs = []
        for i in range(H):                         # sweep rows top -> bottom (sequential in training only over rows)
            g = gates_is[:, :, i] + self.state_to_state(h_prev)
            o, f, ii, gg = g.chunk(4, dim=1)
            c_prev = torch.sigmoid(f) * c_prev + torch.sigmoid(ii) * torch.tanh(gg)
            h_prev = torch.sigmoid(o) * torch.tanh(c_prev)
            outs.append(h_prev)
        return torch.stack(outs, dim=2)            # [B, h, H, W]
```

So the causal chain, end to end: I want an exact, expressive density over images, which rules out latent-variable models (only a bound, plus latent independence assumptions) and sends me to the autoregressive factorization `p(x) = Π_i p(x_i | x_{<i})` — exact likelihood, ancestral sampling, the conditional model now carrying all the weight. I model each conditional as a discrete 256-way softmax rather than a continuous density, because that captures arbitrary multimodality with no shape assumption and trains better, and I extend the chain rule to the R→G→B channels within each pixel. To compute every conditional in parallel at training time while strictly forbidding a pixel from seeing itself or the future, I mask the convolution weights — a strict mask A on the first layer to block the raw-value leak, a relaxed mask B afterward to restore capacity once the leak is structurally impossible. Stacking masked convolutions gives the fast, bounded-receptive-field PixelCNN; to reach long-range and then the *entire* context I build recurrent layers that use convolution to advance a whole spatial slice at once — the Row LSTM (triangular field, missing the sides) and then the Diagonal BiLSTM, which skews the image so diagonals become columns, sweeps them with a tiny `2×1` recurrent kernel in both directions, and shifts the second direction down a row to avoid the future, reaching full context. Residual connections make stacks of up to twelve such layers trainable, and a multi-scale scheme conditions a high-resolution PixelRNN on a cheaply generated low-resolution image to inject global structure. Everything is trained as exact-likelihood cross-entropy in parallel and sampled pixel by pixel.
