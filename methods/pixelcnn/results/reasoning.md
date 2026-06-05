OK, let me think this through from scratch. I want a generative model of natural images, and the property I care about most is an *exact* likelihood — given an image, I want to compute how probable it is, as a real number, with no bound and no estimator standing in the way. That rules out the families I'd otherwise reach for. A variational autoencoder gives me `p(x) = ∫ p(x|z)p(z)dz`, and that integral is intractable, so I'd be training an evidence lower bound and reporting a bound, not the thing itself; worse, the latent bottleneck tends to make the pixels conditionally independent given `z`, which throws away exactly the fine structure of images I want to capture. An RBM or DBM has a partition function I can't compute, so again no exact likelihood. I keep coming back to the same wish: I want a number I can actually evaluate and compare across models.

So forget latents for a second and ask: what is the most honest factorization of a joint distribution I can write down? The chain rule. For *any* joint over variables `x_1, ..., x_N`, repeated conditioning gives

```
p(x) = Π_i p(x_i | x_1, ..., x_{i-1})
```

with no approximation whatsoever. The only freedom is the ordering, and for an image the obvious choice is the raster scan: top row left to right, then the next row, and so on. Flatten the `n×n` image into `x_1, ..., x_{n^2}` in that order and I have

```
p(x) = Π_{i=1}^{n^2} p(x_i | x_1, ..., x_{i-1}).
```

The likelihood is now a *product* of conditionals, and its log is a *sum* of per-pixel log-probabilities. That's the whole appeal: if I can model each conditional well, the joint likelihood is exact and trivially decomposed. This isn't a new idea in spirit — it's the fully-visible directed model of Neal and of Bengio & Bengio, and it's what NADE does with a single shared hidden layer tied across positions. But those were built for small binary vectors. The question is whether I can carry this factorization to natural images, where the conditionals are wildly nonlinear and long-range, and whether I can do it efficiently.

At *training* time I'm handed a real image, so all the "previous pixels" `x_{<i}` are known — they're just the input. That means I can compute *every* conditional `p(x_i | x_{<i})` for all `i` simultaneously in one forward pass and sum their log-probs. Training is fully parallel. It's only *generation* that's sequential: to sample a fresh image I have to draw `x_1`, feed it back, draw `x_2`, and so on, `n^2` steps. I'll accept that asymmetry — parallel training, serial sampling — because it buys me the exact likelihood.

Now, what functional form should each conditional take? The default in the prior recurrent image models — RNADE, the spatial-LSTM RIDE model — is a *continuous* density: the net emits, per pixel, the mean and variance of a Gaussian, or the parameters of a mixture of Gaussians. Let me actually think about whether that's a good idea for pixels, because it bugs me. A pixel intensity is an integer in `{0, ..., 255}`. If I put a continuous density on it I've committed to a parametric *shape*: a single Gaussian is unimodal, a mixture is only as multimodal as its number of components, and either way I've assumed the value lives on a smooth continuum where 51 and 52 are "close." And a Gaussian has tails — it puts probability mass below 0 and above 255, which is mass wasted on impossible values, and I'd have to fuss about truncation. For the very first pixel of an image, the true distribution of, say, the red channel is whatever the dataset says — it could be bimodal with spikes at 0 and 255 (sky and shadow), and a low-component mixture would smear right through that.

So let me drop the continuity assumption entirely and just treat each pixel value as one of 256 discrete categories. The conditional becomes a 256-way **softmax** — a plain multinomial over the intensities. What do I gain? It's arbitrarily multimodal with no shape prior: the net can put a spike on 0, a spike on 255, a bump in the middle, whatever the data demands, because each of the 256 logits is free. No mass leaks outside `[0, 255]` because there's nothing outside the 256 categories. And the loss is just categorical cross-entropy, which is *exactly* the negative log-likelihood I'm after — `−log p(x)` is the sum over pixels of the cross-entropy between the one-hot true intensity and the predicted softmax. Clean.

The obvious worry: by going discrete I've thrown away the *ordinal* structure — the softmax doesn't know that 51 neighbors 52. Doesn't it have to relearn from scratch that nearby intensities tend to co-occur? Yes, but that's cheap: with enough data the net just *learns* smooth, peaked, skewed, or long-tailed shapes wherever they're appropriate, and it learns the empirical fact that 0 and 255 are over-represented. I'd rather pay that small cost than be locked into a parametric family, and I would test the softmax head directly against a continuous mixture head on the same backbone.

But wait — a pixel isn't one value, it's three: R, G, B. How do I factorize *within* a pixel? If I emit three independent softmaxes I'm asserting the three channels are independent given the context, which is false — the red, green, and blue of a single pixel are tightly coupled (that's what color *is*). So apply the chain rule again, inside the pixel, with an order R → G → B:

```
p(x_i | x_{<i}) = p(x_{i,R} | x_{<i}) · p(x_{i,G} | x_{<i}, x_{i,R}) · p(x_{i,B} | x_{<i}, x_{i,R}, x_{i,G}).
```

Green is conditioned on red of the same pixel; blue on both red and green. Full color dependence, still exact, still three softmaxes — just chained. Good. Now I have a complete specification of the *distribution*. The hard part is the *architecture*: how do I build a net that, at every position, produces `p(x_i | x_{<i})` and never, ever peeks at `x_i` or any later pixel?

My first instinct is a convolutional net, because images are translation-structured and convs share weights and run in parallel — exactly the scalability I want, unlike a per-pixel RNN. But a convolution is the problem incarnate. A `k×k` conv centered at output position `i` mixes a whole neighborhood, which includes `x_i` itself and the pixels to its right and below. If `x_i` is the answer I'm trying to predict, and the conv reads `x_i`, then the network can get zero loss by copying — it's seen the target. The conditional would be `p(x_i | everything including x_i)`, which is garbage. I need the conv to respect the raster ordering.

How? The connectivity I'm allowed is precisely: all pixels in rows strictly above `i`, plus the pixels in `i`'s own row strictly to its left. Everything else — `i` itself, the rest of its row, all rows below — is forbidden. A convolution's connectivity *is* its kernel: weight `(a, b)` connects the output at `(r, c)` to the input at `(r + a − k//2, c + b − k//2)`. So I can forbid a connection simply by *zeroing that weight*. Mask the kernel. In a centered `k×k` kernel, every weight in the rows below the center must be zero, the center row must be zero from the center column through the right edge, and the rows above plus the left part of the center row can stay on.

That's a masked convolution, and stacking masked convs gives me a net where, by construction, the output at `i` is a function only of `x_{<i}`. Beautiful — it's the MADE masking idea, but adapted to a spatial conv kernel instead of a fully-connected matrix, so I keep weight sharing and the 2-D structure.

Now there's a subtlety about the center pixel that I have to get exactly right, and it splits into two cases. In the *first* layer, the input is the raw image, so the center weight reads the actual value `x_i` — I absolutely must zero the center weight there. Call that **mask A**: it excludes the center. But in *later* layers, the "center" position carries a *feature* computed by the previous layer, and by induction that feature already depended only on `x_{<i}` (never on `x_i`). So reading it is fine — in fact I *want* to, because it lets information about position `i`'s legitimate context flow straight up the stack. Call that **mask B**: it includes the center. If I used mask A everywhere, every layer would shave one pixel off the reachable context and I'd be needlessly crippling the receptive field. So: mask A on the first layer only, mask B on all the rest.

Let me sanity-check the scalar version of this mask, because it's easy to get an off-by-one wrong. For a `kH×kW` kernel I fill the mask with ones, then set the center row from the center column onward to zero — but for type B I want to *keep* the center, so I start the zeroing one column later. In index terms: zero `mask[..., kH//2, kW//2 + (type=='B') : ]`, and zero `mask[..., kH//2 + 1 :, :]` for all rows below. For type A the `kW//2` column (the center) is zeroed; for type B the zeroing starts at `kW//2 + 1`, so the center survives. Rows above (`< kH//2`) are untouched, fully visible. That matches exactly the allowed region. Good.

But there are three channels, and the within-pixel R→G→B order has to be enforced *inside* the channel dimension too, not just spatially. Split the feature maps of each layer into three groups, one per color. The off-center spatial weights follow the rule above for all groups. The interesting part is the *center* weight, which is now a little `3×3` block of group-to-group connections. For mask A (no self-connection, since at the input the center is the raw value): R connects to nothing of itself, G connects to R, B connects to R and G — group `g` reads only groups *before* it. For mask B (self allowed, since now it's a feature): R reads R, G reads R and G, B reads R, G, B — group `g` reads groups up to and including itself. That's precisely the `p(R) p(G|R) p(B|R,G)` chain expressed as a connectivity pattern. (For a single-channel image like binarized MNIST this collapses to the scalar mask, no channel grouping needed.)

So I have a fully-convolutional, parallel-trainable, exact-likelihood image model: stack masked convs, end with a per-position 256-way softmax per channel, train with cross-entropy. Let me call this the convolutional version. It's fast. But there's a limitation staring at me: a stack of convolutions has a *bounded* receptive field. Each masked conv reaches `k` pixels; `L` of them reach roughly `L·k`. For a big image, the conditional `p(x_i | x_{<i})` can only actually depend on a finite window of earlier pixels, not the entire history. Objects and scenes have long-range structure — the left edge of a face constrains the right edge — and a bounded window can't represent that. I want, in principle, an *unbounded* dependency range.

That's what recurrence buys. An RNN's hidden state is a running summary of everything seen so far, so its receptive field is unbounded. The natural fit for a raster-scanned image is a 2-D LSTM — a recurrence that walks the grid carrying state in both spatial directions, as in the spatial-LSTM work. But a naive 2-D LSTM processes one pixel at a time with a full sequential dependency on the pixel above *and* to the left, which is `O(n^2)` sequential steps and slow. I want the unbounded field *and* as much parallelism as I can claw back.

Idea one: process a whole *row* at once. Within a row, the LSTM steps from row to row (top to bottom), and at each step it ingests the entire row in parallel. The input-to-state contribution for the whole image is a single masked one-dimensional convolution with kernel length `k` (`k ≥ 3`) along the row, written as `k×1` in the architecture convention, masked to look only at valid context. It produces, for every position, the four LSTM gate pre-activations stacked as `4h` channels. Then the state-to-state recurrence runs once per row: given the previous row's hidden and cell states `h_{i-1}, c_{i-1}`, another row-wise convolution of length `k` over `h_{i-1}` adds the recurrent contribution, and the standard LSTM update fires:

```
[o_i, f_i, i_i, g_i] = activation( K^{ss} ⊛ h_{i-1} + K^{is} ⊛ x_i )
c_i = f_i ⊙ c_{i-1} + i_i ⊙ g_i
h_i = o_i ⊙ tanh(c_i)
```

with `σ` (logistic) for the output, forget, and input gates `o, f, i`, and `tanh` for the content gate `g`. One step advances an entire row. Call this the **Row LSTM**. Now I have `n` sequential steps (one per row) instead of `n^2`, and everything within a step is convolution-parallel.

What's the receptive field, though? Let me trace it. Each row pulls a `k`-wide window from the row above (via the `k×1` state-to-state conv). Going up two rows, the reachable window widens by `k` again, so the dependency region above pixel `i` is a *triangle* — a cone that fans out as it goes up, of half-width set by `k`. So the Row LSTM *does* reach unbounded distance vertically, but it has a triangular blind region: pixels on earlier rows that are far to the side, outside the cone, never reach `i`. So it captures *most* but not *all* of the valid context. Better than the bounded conv, but not the full thing.

Can I get the *entire* valid context — every pixel up-and-left of `i`, all of it — and still parallelize? I want a recurrence whose state-to-state step touches the pixel directly above and the pixel directly to the left, because that pair, composed over the grid, transitively reaches the whole upper-left region. The trouble is that "above" and "left" are along two different axes, so I can't just run one convolution along one axis to cover both. Unless... I change coordinates so that the two directions I care about line up with a single axis.

Skew the image. Offset each row to the right by its row index: row 0 unchanged, row 1 shifted right by 1, row 2 by 2, and so on. An `n×n` map becomes `n×(2n−1)`. Now a down-left diagonal of the original — `(r, c)`, `(r+1, c−1)`, ... — lands in one column, because the column index decreases by one while the row offset increases by one. So propagating along that diagonal is just propagating down a column of the skewed map, which a column-wise convolution can do for all diagonals in parallel. The input-to-state is a `1×1` convolution, producing the `4h` gates; the state-to-state is a two-position column convolution (`2×1`, equivalently `1×2` under the other width-height convention) that combines the previous diagonal state and runs the same LSTM update as above. Skew back afterward by removing the offsets to recover the `n×n` map. Call this scanning a **Diagonal LSTM**.

One diagonal scan covers the context that flows from one top corner through one diagonal family — but not everything; I also need the contributions coming from the other top corner. So run a *second* scan in the mirror direction on the flipped map. Now I have two output maps. If I just add them I'll break causality, because the mirrored scan, at pixel `i`, can have incorporated pixels that are on `i`'s own row to its *right* or even below-right — future pixels. To keep it honest, before adding, shift the right map *down by one row*. After the shift, every pixel the right scan contributes to position `i` is strictly *previous* in raster order. Add the shifted right map to the left map. Call the pair a **Diagonal BiLSTM**.

What's its receptive field now? Each direction's diagonal recurrence transitively reaches everything along its diagonals back to the corner, and the two directions between them sweep the *entire* region up-and-left of `i`. No triangular gap, no bounded window — the full valid context, for any image size. And the two-position kernel is the *minimal* nonlinear step; making it bigger would not broaden a field that is already global, so a larger kernel mostly adds parameters. This is where I would expect the most expressive likelihood model to come from: global context first, triangular context second, bounded convolution last.

Now depth. I want many of these recurrent layers stacked — say up to twelve — to build expressive conditionals. Deep stacks are hard to train; signal and gradient attenuate. The fix that's been working elsewhere is residual connections: add a layer's input to its output so the layer only has to learn a *residual*, giving the signal a direct path. Let me wire it for the recurrent block. The block's input map carries `2h` features; the input-to-state step reduces to `h` features per gate, the recurrence runs, then a `1×1` convolution upsamples the output back to `2h`, and I add the input map to it. I can also add layer-to-output skip connections that route each layer straight to the final prediction. The point is not to change the probability model; it is to make enough depth trainable that the conditional predictor can use the context it is allowed to see.

Let me step back to the fast convolutional version, the masked-conv stack, because I waved at its bounded receptive field but I glossed over something more insidious, and it bothers me now that I look hard at the mask. I claimed stacking masked convs gives the output at `i` access to "all of `x_{<i}` within the window." Let me actually verify that the reachable set is the full upper-left region (within the window), by composing the masks. One masked conv's allowed region is the rows strictly above plus the left part of the current row — a flat-topped shape that leans left. Stack two: the second conv at position `i` reads positions in *its* allowed region, each of which reads *its* allowed region in the layer below. Compose these and the reachable footprint grows upward and leftward... but trace the upper-*right*. A pixel that sits a couple of rows above `i` and a few columns to its *right* is legitimately in `x_{<i}` (it's on an earlier row). Is it reachable? To carry its information down to `i`, some chain of allowed connections has to route from it to `i`. But every masked conv leans left — within a row it only ever connects leftward, and across rows it fans within the same left-leaning cone. There's no path that goes up-and-to-the-right and then comes back down to `i`. So that whole wedge of pixels up and to the right of `i` is *never reached*, no matter how deep I stack.

That's a blind spot. And it's a nasty kind of failure: those pixels are ones the model is *allowed* to use — they're real context, strictly previous in the ordering — but the architecture *cannot* see them. It's not a correctness bug (causality holds, I never peek ahead), it's a *capacity* bug: a triangular region of legitimate context is silently amputated. It arises precisely because a single masked kernel is left-biased and the only way convolution carries information rightward-and-upward is through paths that the same-row mask blocks. The Row LSTM has its own version of this (its triangular field is the same phenomenon); only the Diagonal BiLSTM, with its two skewed scans, has no blind spot at all.

So if I want to keep the speed of a pure conv net but kill the blind spot, I shouldn't try to patch one cleverly-shaped mask — the geometry fights me. The clean fix is to split the finite convolutional receptive field into two stacks that are each individually easy to reason about, and combine them. A **vertical stack** reads rows above the current one, with no same-row complication, so the upper-right pixels inside its finite range are no longer amputated. A **horizontal stack** reads the current row strictly to the left. Each stack is causal on its own, and the vertical stack can feed the horizontal stack through a `1×1` link, but never the reverse. That one-way coupling keeps causality while letting the horizontal stack use above-row information from both left and right. The model is still a finite-depth convolutional model, but within that finite range the blind wedge is gone.

I have to be careful implementing the vertical-to-horizontal link so it really excludes the current row. I can use a `1×k` convolution to mix horizontally in the vertical stream, then a `(k//2 + 1)×1` convolution with extra top padding and crop the output back to height `H`; the cropped tensor passed through the `1×1` link is shifted so that the horizontal stack receives only rows above. The vertical stream itself may also carry a pointwise transform of the current row forward, but that pointwise part must be added *after* the link is computed, so it can only help lower rows later. The horizontal stack gets its strictly-left field with a `1×(k//2+1)` convolution padded on the left and cropped back to width `W`. And the residual connection: I can put one on later horizontal layers to ease optimization, but I must not put a residual on the very first causal input layer, because that would add the raw `x_i` straight back in.

While I'm redesigning the conv stack, one more thing nags at me. The recurrent layers were expressive partly because LSTM gating gives a multiplicative, highly nonlinear interaction; the plain conv stack with ReLUs is comparatively flat. Why not borrow the gate? Replace the ReLU between conv layers with a **gated activation**: split the `2h` conv output into two halves and compute `tanh(a) ⊙ σ(b)` — a content path squashed by `tanh`, modulated multiplicatively by a sigmoid gate. That mimics the LSTM's content-times-gate structure inside a feed-forward conv layer, and it's a richer nonlinearity than ReLU. So the fast convolutional model becomes a causal input layer, then many gated two-stream layers, residual connections on the later horizontal stack, skip connections feeding a head of `1×1` convs and ReLUs, ending in the per-channel softmax.

Let me also not forget the binary case as a sanity check: on binarized MNIST each pixel is one bit, so the 256-way softmax collapses to a single sigmoid and a binary cross-entropy loss — same model, two-category head.

Let me write the model down as an ordered convolution, an ordered image layer, the full density model, the exact NLL, one update, and raster-order sampling.

```python
import torch
import torch.nn.functional as F
from torch import nn


class OrderedConv2d(nn.Conv2d):
    """Mask the kernel so position i never reads positions >= i.
    type 'A' excludes the center weight (first layer: center == raw x_i);
    type 'B' keeps it (later layers: center is a feature of x_{<i})."""
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # center row: zero from center column onward; type B keeps the center.
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B"):] = 0
        # all rows below the center: fully zeroed (they reach pixels below i).
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

So the chain of reasoning lands here: I wanted an exact, comparable likelihood, so I took the chain rule's product of conditionals over a raster-scan ordering — and, within each pixel, over R then G then B. I made each conditional a 256-way softmax to be freely multimodal and to honor the discreteness of pixel intensities. To predict each pixel from only its predecessors I masked the convolution kernels, with type A excluding the current pixel at the input and type B including its already-known features afterward, and I extended the mask into the channel dimension to enforce R→G→B. To reach unbounded context I built two parallel 2-D recurrences — the Row LSTM with its triangular field, and the Diagonal BiLSTM that skews the image so a column convolution sweeps a diagonal family and, with two mirrored directions, captures the entire valid context — and I added residual connections so deep recurrent stacks remain trainable. Staring at the pure masked-conv stack I found its blind spot, a wedge of legitimate upper-right context the left-biased masks can never reach, and I closed that gap within the finite convolutional range by splitting into a vertical stack over rows above and a horizontal stack over the strict left, coupled one way, with a gated `tanh ⊙ σ` activation borrowed from the LSTM to keep the conditionals expressive. Train by minimizing the summed per-pixel cross-entropy in parallel; sample one pixel at a time.
