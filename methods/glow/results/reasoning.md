OK, let me think about this starting from what I actually need. I want a generative model of images that does four things at once: gives me the exact likelihood of a datapoint so I can actually measure fit and compress; lets me recover the exact latent code of any image with no approximation; trains and samples in parallel so it's fast on a GPU; and produces sharp, realistic, high-resolution pictures. Right now nothing does all four. Autoregressive pixel models give me exact likelihood and gorgeous detail, but sampling is one subpixel at a time — for a 256×256 image that's hundreds of thousands of sequential network calls, hopeless. Variational autoencoders sample in one shot but only optimize a lower bound, and the latent I infer is only an approximation of the posterior. Adversarial models make beautiful images but hand me no encoder, no likelihood, no way to even check whether I've overfit. So I keep coming back to the one family that could in principle give me all four: a model defined by an exactly invertible transformation.

Let me make that precise, because the whole thing lives or dies on one formula. I'll define the model by pushing a simple latent through a bijection. Take `z ~ p(z)` with `p(z)` something trivial like a spherical Gaussian `N(0, I)`, and let `x = g(z)` be invertible, with inverse `z = f(x)`. The density of `x` is then forced on me by conservation of probability mass — if I squeeze a region of `z`-space into a smaller region of `x`-space, the density has to go up by exactly the inverse of the volume ratio, and that ratio is the Jacobian determinant. So

```
p(x) = p(z) · |det(dz/dx)|,   z = f(x),
log p(x) = log p(z) + log|det(dz/dx)|.
```

That second term is the price I pay, and it's also the whole problem. `f` is going to be a big neural net, so `dz/dx` is a `D×D` matrix with `D` in the hundreds of thousands, and a general determinant is `O(D^3)`. If I can't compute that cheaply, the model is dead on arrival.

Two things rescue it. First, I'll never use one monster bijection; I'll compose many simple ones, `f = f_1 ∘ f_2 ∘ ... ∘ f_K`, with intermediate states `h_0 = x, h_1, ..., h_K = z`. The chain rule turns the determinant of the product of Jacobians into the product of determinants, and the log turns that into a sum:

```
log p(x) = log p(z) + Σ_{i=1}^K log|det(dh_i / dh_{i-1})|.
```

So each layer contributes one log-determinant and I can reason about them one at a time. Second — and this is the lever everything hangs on — if I build each layer so its Jacobian is *triangular*, the determinant is just the product of the diagonal entries, and the log-determinant collapses to `sum(log|diag|)`, an `O(D)` quantity. A triangular Jacobian is the design target for every layer I'm about to invent.

Before architecture, a nagging detail: my pixels are 8-bit integers, but I'm modeling a *continuous* density. A continuous density over discrete points is a trap — the model can pile unbounded density onto the exact integer grid and run the likelihood off to infinity without learning anything. The fix is to smear the integers into little boxes: add independent uniform noise `u ~ U([0, a)^M)` and model `x̃ = x + u`, where `a` is the bin width. Then `E_u[log p(x̃)] + M log a` lower-bounds the log-probability the discrete data would get, so optimizing it is honest. In the log-likelihood accumulator this is the fixed term `M log a`; with `a = 1/n_bins`, it is `−M log(n_bins)`. I'll report everything in bits per dimension, `nll / (log 2 · D)`, since the negative log-likelihood literally is a compression cost.

Now, what invertible layer can I actually build with a triangular Jacobian? The trick that makes this whole family work is the coupling layer. Split the variables into two parts, `(x_a, x_b)`. Leave `x_a` completely alone and transform `x_b` using only the part I left alone. The simplest version is

```
y_a = x_a,    y_b = x_b + m(x_a),
```

where `m` is *any* neural net. Why is this invertible no matter how crazy `m` is? Because to invert I just recompute `m(y_a) = m(x_a)` from the untouched half and subtract: `x_a = y_a`, `x_b = y_b − m(y_a)`. I never invert `m` itself — I only ever evaluate it forward, in both directions. And the Jacobian? Order the variables as `(a, b)`. `∂y_a/∂x_a = I`, `∂y_a/∂x_b = 0`, `∂y_b/∂x_a = m'(x_a)`, `∂y_b/∂x_b = I`. That's

```
[     I        0 ]
[ m'(x_a)      I ]
```

lower-triangular with all ones on the diagonal, so `det = 1` and `log|det| = 0`. The layer is volume-preserving. Lovely: invertible, arbitrary network inside, zero-cost log-determinant.

But pure additivity worries me. If every layer has log-determinant exactly zero, the *entire* flow has log-determinant zero, and then `log p(x) = log p(z)` with no volume adjustment at all — the model can only relocate mass, never concentrate or spread it to match the data's density. I need the flow to be able to change volume. So I keep the same asymmetric split, but let the transformed half be shifted and scaled by functions of the untouched half:

```
y_a = x_a,    y_b = (x_b + t(x_a)) ⊙ exp(s(x_a)).
```

Inverse: `x_a = y_a`, `x_b = y_b ⊙ exp(−s(y_a)) − t(y_a)`, again only ever evaluating `s, t` forward. Now `∂y_b/∂x_b = diag(exp(s(x_a)))`, so the Jacobian is

```
[        I                  0       ]
[    ∂y_b/∂x_a     diag(exp s(x_a)) ]
```

still triangular, with diagonal entries `1` for the untouched half and `exp(s(x_a))` for the transformed half, so its log-determinant is the sum of the log-scales, `log|det| = Σ s(x_a)`. The layer can change volume now, and additive coupling falls out as the special case `s ≡ 0`. Good, I'll use affine coupling, and I'll predict the scale parameter and the shift jointly from one network so they share features.

The structural problem with coupling is that each layer freezes half the variables — `y_a = x_a` passes through untouched. If I stack coupling layers without doing anything in between, the same half stays in the conditioning role forever. So between couplings I have to *mix* the variables: shuffle them so that the half that was passive becomes active next time, and over enough layers every dimension can influence every other. The standard move is to permute. The simplest permutation is to reverse the order of the channels before each coupling, so the two halves swap roles. A fixed random permutation is another option. Either way it's a hand-fixed reordering.

Let me sit with that for a second, because it feels like the place where I'm leaving something on the table. Between two coupling layers I'm applying a *fixed permutation* of the channels. A permutation is an invertible linear map — multiplying the channel vector at each spatial position by a permutation matrix `P`. Reversing channels is one such `P`; a random shuffle is another. But the set of invertible linear maps is enormous, and permutation matrices are a tiny, rigid corner of it. Why should the best way to route information between couplings be a hand-picked permutation? Why not *learn* the mixing?

So replace the fixed permutation with a learned invertible linear map of the channels. Concretely, at every spatial location `(i, j)` apply the same `c×c` matrix `W`: `y_{i,j} = W x_{i,j}`. Sharing one matrix across all spatial positions and acting only across channels — that's exactly a convolution with a `1×1` kernel and `c` input, `c` output channels. So the operation I want is an invertible `1×1` convolution, and a channel permutation is literally the special case where `W` is a permutation matrix. This strictly generalizes what the fixed permutation was doing, and it lets the model *learn* how to mix channels into the next coupling's split rather than having me guess.

I have to check the two things that could break it. Invertibility: I need `W` invertible, and the inverse operation is just the `1×1` conv with weight `W^{-1}`, which I compute once per layer (cheap, needed only at sampling time). Initialization: if I start `W` as a random rotation (an orthogonal matrix), it's guaranteed invertible and `|det W| = 1`, so `log|det W| = 0` at the start — the layer begins as a clean volume-preserving mixing, just like a permutation, and only drifts away from that once gradients start flowing. Good.

Now the log-determinant, which is the part I actually need for the objective. The layer acts on a tensor of shape `h × w × c`. At each of the `h·w` spatial positions, the map is the *same* linear map `x_{i,j} ↦ W x_{i,j}`, and these positions don't interact — position `(i,j)`'s output depends only on position `(i,j)`'s input. So if I order all `h·w·c` variables by grouping the `c` channels within each spatial position, the full Jacobian is **block-diagonal** with `h·w` identical `c×c` blocks, each equal to `W`. The determinant of a block-diagonal matrix is the product of the block determinants, so `det(full Jacobian) = (det W)^{h·w}`, and therefore

```
log|det(d conv2d(h; W)/dh)| = h · w · log|det W|.
```

Clean. Each `1×1` conv contributes `h·w·log|det W|` to the running sum. Let me check the cost: computing `det W` (and `W^{-1}` for sampling) is `O(c^3)`, while the convolution itself is `O(h·w·c^2)`. For the channel counts I'll actually run, `c` is small enough that `c^3` is comparable to `h·w·c^2`, so the overhead is minor — a few percent of wallclock. So at these sizes I can just compute `det W` directly each step.

But `O(c^3)` per step nags at me for large `c`, and there's a cleaner way that also removes the determinant computation entirely. The reason a determinant is expensive is that `W` is unstructured. What if I *parameterize* `W` so its determinant is read off for free? Any square matrix has an LU (really PLU) factorization: a permutation, a lower-triangular factor, an upper-triangular factor. The determinant of a triangular matrix is the product of its diagonal. So if I write `W` directly in factored form, I never form a general determinant. Concretely:

```
W = P · L · (U + diag(s)),
```

with `P` a permutation matrix that I **fix at initialization** (not trained), `L` lower-triangular with **ones on the diagonal**, `U` strictly upper-triangular (**zero diagonal**), and `s` a vector that carries all of `W`'s diagonal scaling. Then `det P = ±1`, `det L = 1` (unit diagonal), and `det(U + diag(s)) = Π_k s_k` (triangular, diagonal is exactly `s`), so

```
log|det W| = Σ log|s|,
```

an `O(c)` quantity with no determinant computation at all. To initialize, I still want to start from a random rotation: sample an orthogonal `W_0`, take its PLU decomposition to get `P` (fixed forever), `L`, `U`, and `s` (all trainable from there). One subtlety — if I optimize `s` directly it can wander through zero, and `s_k = 0` makes `W` singular and the log-determinant `−∞`. So I store the *sign* of each `s_k` as a fixed constant and optimize `log|s_k|`; the diagonal is reconstructed as `sign(s) ⊙ exp(log|s|)`, which can never hit zero and never flips sign. At my channel counts the wallclock difference between the direct-`det` version and this LU version isn't large, but the LU form is the principled fix when `c` gets big, so I'll use the LU parameterization in code.

So one step now has: a coupling layer, and before it, an invertible `1×1` conv that mixes channels. There's a third thing I need, and it comes from a different failure. Deep flows are hard to train — many stacked layers, and the activations drift in scale until things blow up or vanish. The usual remedy is batch normalization, and prior flows did use it, even folding its scale into the Jacobian. But I'm targeting high-resolution images, and at 256×256 a single image barely fits in GPU memory, so my minibatch is *size 1 per processing unit*. Batch normalization is exactly wrong here: the "noise" it injects into the activations has variance inversely proportional to the per-PU batch size, so at batch size 1 that noise is maximal and the batch statistics are garbage. I need normalization that doesn't depend on batch statistics at training time.

What does batch norm actually buy me that I care about? Mostly that, at the start of training, each channel's activations are roughly zero-mean and unit-variance, so the deep stack starts in a sane regime. I can get that *once* and then stop depending on the batch. So: a per-channel affine layer, `y_{i,j} = s ⊙ (x_{i,j} + b)`, with `s` and `b` length-`c` vectors shared across all spatial positions — and I initialize `b` to center the first minibatch and `s` to divide by its per-channel standard deviation. After that single data-dependent initialization, `s` and `b` are ordinary trainable parameters with no further dependence on any batch. This is activation normalization, and because it's batch-independent after init it works perfectly fine at batch size 1.

Its log-determinant: at one spatial position the map is `x ↦ s ⊙ (x + b)`, whose Jacobian is `diag(s)`, contributing `Σ log|s|`. And just like the `1×1` conv, the *same* `s` is applied independently at every one of the `h·w` positions, so the total is

```
h · w · Σ log|s|.
```

So my flow step is three sublayers in sequence: activation normalization, then the invertible `1×1` convolution, then the affine coupling. Forward I accumulate `h·w·Σlog|s_actnorm|`, then `h·w·log|det W|` (or `h·w·Σlog|s_W|` in the LU form), then `Σ log|s_coupling|` from the coupling (the coupling scale `s` already varies per spatial position because it's the output of a conv net, so there's no `h·w` factor — I sum over all positions and channels). Reverse, I undo coupling, then conv with `W^{-1}`, then divide out actnorm, subtracting each log-determinant.

A couple of choices inside the coupling network I should pin down. First, the split: I'll split purely along the **channel** dimension and drop the spatial checkerboard masking that earlier flows alternated with channel masking. Why can I drop it? The point of having two mask types was to make sure information mixes across both spatial neighbors and channels. But I've just inserted a learned `1×1` convolution before every coupling, and that already does a flexible cross-channel mixing — far more than a fixed checkerboard ever did. So checkerboard becomes redundant, and dropping it simplifies the architecture to a single split pattern. Second, the coupling network itself: three convolutions, `3×3 → 512` channels, ReLU, then `1×1 → 512`, ReLU, then a final conv back to the output width. The middle layer is `1×1` rather than `3×3` precisely because at that point both its input and output are wide (512 channels), and a `3×3` over 512→512 channels would be wastefully expensive for little gain, whereas the first and last need spatial `3×3` receptive fields. Third — and this matters a lot for stability of a very deep flow — I initialize the *last* convolution of each coupling network to **zero**. In the clean exponential parameterization, that makes `s = exp(0) = 1` and `t = 0`, so each coupling layer starts as identity. In the implementation I use a bounded positive scale, `sigmoid(scale_logits + 2)`, so a zero final layer starts at a mild constant scale rather than an uncontrolled exponential; the log-determinant records that scale exactly, and the bounded parameterization avoids extreme early multipliers.

Two more pieces of plumbing, both about giving the channel-wise operations something to work with and keeping the model efficient. The coupling and the `1×1` conv both act across channels, but a raw image has only 3. So I **squeeze**: reshape `h × w × c → (h/2) × (w/2) × 4c`, folding each `2×2` spatial block into the channel dimension. This quadruples the channels (giving the channel operations room) and halves each spatial dimension (cheaper deeper layers). And I use a **multi-scale** layout: after a block of flow steps at one resolution, I factor out *half* of the channels right there as part of the final latent `z` — modeling them with a Gaussian whose mean and log-scale are predicted (by a zero-initialized conv) from the half I keep — and only the kept half continues into the next, coarser level. This gives a coarse-to-fine latent, and it means the deeper, more expensive levels operate on fewer dimensions. The factored-out variables just contribute their Gaussian log-probability to the objective; they don't add a Jacobian term beyond what the flow already accounted for. At the very top is the final Gaussian prior (which I can make class-conditional by predicting its parameters from a label).

Let me sanity-check the whole objective once, end to end. I dequantize `x̃ = x + u`. I run it through the stack — squeeze, then per level `[K steps of (actnorm → 1×1 conv → affine coupling)]`, with a multi-scale split after each level — accumulating every sublayer's log-determinant into one running scalar `logdet`. At the top I have the latent `z`, and the factored-out latents along the way, all scored under their Gaussians. The continuous model log-density is exact: `log p(z) + logdet`. The discrete-data lower-bound term adds `−log(n_bins)·D`, so the optimized log quantity is `log p(z) + logdet − log(n_bins)·D`, and the loss is its negative divided by `log 2 · D` in bits per dimension. The only bound is the dequantization step; after that, the flow likelihood and every Jacobian term are exact.

One last thing for sampling quality, only relevant at generation time. When I sample `z ~ N(0, I)` and push it through `g`, the rawest samples can be noisy. If instead I sample `z ~ N(0, T^2 I)` with `T < 1`, then in the volume-preserving case the latent Gaussian term is proportional to the unit-temperature density raised to `1/T^2`; that concentrates samples closer to the high-density part of latent space. In practice this is just multiplying the prior standard deviation by `T`. A temperature around `0.7` trades a little diversity for noticeably cleaner samples. That's a knob, not part of the objective.

So the method is: build the bijection out of repeated steps of `actnorm → invertible 1×1 conv → affine coupling`, wrapped in squeeze-and-multi-scale, and train the dequantized likelihood with exact change-of-variables terms. Now I can write each sublayer and its inverse with the log-determinant bookkeeping that the objective needs.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess(x, n_bits=8):
    n_bins = 2 ** n_bits
    x = x.float()
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    x = x + torch.rand_like(x) / n_bins
    return x, n_bins


def squeeze2d(x, factor=2):
    # h x w x c  ->  (h/factor) x (w/factor) x (c*factor^2): give channel ops room
    b, c, h, w = x.shape
    x = x.view(b, c, h // factor, factor, w // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * factor * factor, h // factor, w // factor)


def unsqueeze2d(x, factor=2):
    b, c, h, w = x.shape
    x = x.view(b, c // factor ** 2, factor, factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, c // factor ** 2, h * factor, w * factor)


class FlowModule(nn.Module):
    def forward(self, x, logdet):
        raise NotImplementedError

    def reverse(self, y):
        raise NotImplementedError


class ActNorm(nn.Module):
    # per-channel scale+bias, initialized from the first batch (data-dependent),
    # then batch-independent -> works at batch size 1, unlike batch norm.
    def __init__(self, channels):
        super().__init__()
        self.logs = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def _init(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = (x - mean).pow(2).mean(dim=[0, 2, 3], keepdim=True).sqrt()
            self.bias.copy_(-mean)
            self.logs.copy_(torch.log(1.0 / (std + 1e-6)))
            self.initialized.fill_(1)

    def forward(self, x, logdet):
        if self.initialized.item() == 0:
            self._init(x)
        h, w = x.shape[2], x.shape[3]
        y = (x + self.bias) * torch.exp(self.logs)
        # same scale at every one of h*w positions -> h*w * sum(log|s|)
        logdet = logdet + h * w * self.logs.sum()
        return y, logdet

    def reverse(self, y):
        return y * torch.exp(-self.logs) - self.bias


class InvConv1x1(nn.Module):
    # learned invertible channel mixing = generalization of a fixed permutation.
    # LU parameterization so log|det W| = sum(log|s|), O(c) and never singular.
    def __init__(self, channels):
        super().__init__()
        w0 = torch.linalg.qr(torch.randn(channels, channels))[0]  # random rotation
        P, L, U = torch.linalg.lu(w0)
        self.register_buffer("P", P)                  # fixed permutation
        s = torch.diag(U)
        self.register_buffer("sign_s", torch.sign(s))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(torch.triu(U, 1))       # strictly upper
        self.register_buffer("l_mask", torch.tril(torch.ones_like(L), -1))
        self.register_buffer("eye", torch.eye(channels))

    def _w(self):
        L = self.L * self.l_mask + self.eye          # unit diagonal
        U = self.U * self.l_mask.t() + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x, logdet):
        h, w = x.shape[2], x.shape[3]
        W = self._w()
        y = F.conv2d(x, W.view(*W.shape, 1, 1))
        logdet = logdet + h * w * self.log_s.sum()   # h*w*log|det W| = h*w*sum(log|s|)
        return y, logdet

    def reverse(self, y):
        W_inv = torch.inverse(self._w())
        return F.conv2d(y, W_inv.view(*W_inv.shape, 1, 1))


class OutputConv2d(nn.Module):
    # zero-initialized output head, with a learned output logscale like the canonical utility.
    def __init__(self, in_ch, out_ch, kernel_size=3, logscale_factor=3.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.logs = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.logscale_factor = logscale_factor

    def forward(self, x):
        return self.conv(x) * torch.exp(self.logs * self.logscale_factor)


class CouplingNN(nn.Module):
    # 3x3 -> 1x1 -> zero-initialized 3x3, matching the flow coupling network.
    def __init__(self, in_ch, out_ch, width=512):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 1)          # middle 1x1: in/out are wide
        self.c3 = OutputConv2d(width, out_ch)

    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        return self.c3(x)


class AffineCoupling(nn.Module):
    # split on channels only (the 1x1 conv already mixes channels -> no checkerboard).
    def __init__(self, channels, width=512):
        super().__init__()
        self.net = CouplingNN(channels // 2, channels, width)

    def forward(self, x, logdet):
        z1, z2 = x.chunk(2, dim=1)
        h = self.net(z1)
        shift = h[:, 0::2]
        scale_logits = h[:, 1::2]
        scale = torch.sigmoid(scale_logits + 2.0)
        z2 = (z2 + shift) * scale
        logdet = logdet + torch.log(scale).flatten(1).sum(1)
        return torch.cat([z1, z2], dim=1), logdet

    def reverse(self, y):
        z1, z2 = y.chunk(2, dim=1)
        h = self.net(z1)
        shift = h[:, 0::2]
        scale_logits = h[:, 1::2]
        scale = torch.sigmoid(scale_logits + 2.0)
        z2 = z2 / scale - shift
        return torch.cat([z1, z2], dim=1)


class FlowStep(FlowModule):
    # the contribution: actnorm -> invertible 1x1 conv -> affine coupling.
    def __init__(self, channels, width=512):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = InvConv1x1(channels)
        self.coupling = AffineCoupling(channels, width)

    def forward(self, x, logdet):
        x, logdet = self.actnorm(x, logdet)
        x, logdet = self.invconv(x, logdet)
        x, logdet = self.coupling(x, logdet)
        return x, logdet

    def reverse(self, y):
        y = self.coupling.reverse(y)
        y = self.invconv.reverse(y)
        y = self.actnorm.reverse(y)
        return y


def gaussian_logp(z, mean, log_sd):
    return -0.5 * (math.log(2 * math.pi) + 2 * log_sd
                   + (z - mean) ** 2 / torch.exp(2 * log_sd))


class ImageFlow(nn.Module):
    # squeeze + K steps + multi-scale split, per level.
    def __init__(self, in_ch=3, depth=32, levels=3, width=512):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        self.split_priors = nn.ModuleList()
        c = in_ch * 4                                  # after first squeeze
        for i in range(levels):
            self.blocks.append(nn.ModuleList(
                [FlowStep(c, width) for _ in range(depth)]))
            if i < levels - 1:
                # predict Gaussian for the factored-out half from the kept half
                self.split_priors.append(OutputConv2d(c // 2, c))
                c = c * 2                              # squeeze next level
        self.top_prior = OutputConv2d(c, 2 * c)

    def forward(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        log_p = torch.zeros_like(logdet)
        z = squeeze2d(x)
        for i in range(self.levels):
            for step in self.blocks[i]:
                z, logdet = step(z, logdet)
            if i < self.levels - 1:
                z1, z2 = z.chunk(2, dim=1)            # factor out half as latent
                mean, log_sd = self.split_priors[i](z1).chunk(2, dim=1)
                log_p = log_p + gaussian_logp(z2, mean, log_sd).flatten(1).sum(1)
                z = squeeze2d(z1)
        mean, log_sd = self.top_prior(torch.zeros_like(z)).chunk(2, dim=1)
        log_p = log_p + gaussian_logp(z, mean, log_sd).flatten(1).sum(1)
        return z, logdet, log_p

    def reverse(self, z, eps=None, eps_std=1.0):
        eps = [None] * (self.levels - 1) if eps is None else eps
        for i in reversed(range(self.levels)):
            if i < self.levels - 1:
                z1 = unsqueeze2d(z)
                mean, log_sd = self.split_priors[i](z1).chunk(2, dim=1)
                noise = torch.randn_like(mean) * eps_std if eps[i] is None else eps[i]
                z2 = mean + torch.exp(log_sd) * noise
                z = torch.cat([z1, z2], dim=1)
            for step in reversed(self.blocks[i]):
                z = step.reverse(z)
        return unsqueeze2d(z)


def loss_bits_per_dim(logdet, log_p, n_bins, n_pixels):
    ll = log_p + logdet - math.log(n_bins) * n_pixels
    return (-ll / (math.log(2) * n_pixels)).mean()


def train_step(model, batch, opt):
    x, n_bins = preprocess(batch)
    _, logdet, log_p = model(x)
    loss = loss_bits_per_dim(logdet, log_p, n_bins, x[0].numel())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss
```

The chain now holds together. I wanted exact likelihood, exact inference, and parallel sampling at once, which points at an invertible model and the change-of-variables identity `log p(x) = log p(z) + log|det(dz/dx)|`; to make the log-determinant tractable I compose many layers each with a triangular Jacobian, summing `sum(log|diag|)`; coupling layers give me invertibility with an arbitrary inner net and a triangular Jacobian, and going affine lets the flow change volume (`log|det| = Σ log scale`); coupling freezes half the variables so I must mix between layers, and rather than a hand-fixed permutation I learn the mixing as an invertible `1×1` convolution whose log-determinant is `h·w·log|det W|`, made `O(c)` by an LU parameterization where `log|det W| = Σ log|s|`; batch norm fails at the batch-size-1 regime high-resolution forces on me, so I replace it with data-dependently-initialized but batch-independent activation normalization, log-determinant `h·w·Σ log|s|`; the learned channel mixing makes spatial checkerboard masking redundant, so I split on channels only; zero-initialized coupling and prior heads give the deep stack a quiet start; and squeeze-plus-multi-scale gives the channel operations room and a coarse-to-fine latent. Train it by minimizing the dequantized negative log-likelihood in bits per dimension.
