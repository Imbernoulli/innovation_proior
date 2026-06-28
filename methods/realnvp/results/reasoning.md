OK, let me think this through from scratch. I want to learn a generative model of natural images — really learn the density p(x), not just produce pretty pictures. And I want the thing that nobody currently has all at once: I want to *evaluate* the exact likelihood of a data point, I want to *sample* fast, I want to *infer* the latent code of any image, and I want that latent space to mean something. Every approach I know buys me a subset of these and pays for the rest.

If I go the undirected route — a Boltzmann machine — the energy is easy to write but the partition function is intractable, so training leans on contrastive divergence, evaluation leans on annealed importance sampling, and sampling leans on MCMC whose mixing time I can't bound and whose samples come out correlated. Exact anything is off the table. If I go the directed latent-variable route with a variational autoencoder, ancestral sampling is clean and inference is amortized and fast, but I'm maximizing a *lower bound*, my inference is *approximate*, and worst of all the Gaussian decoder hands me a fixed L2 reconstruction cost. That L2 term rewards getting the low-frequency content right far more than the high-frequency detail, and the visible consequence is blur — VAE samples look soft. If I go fully autoregressive and write p(x) = ∏_i p(x_i | x_{<i}), I get an *exact* likelihood and enormous flexibility, which is wonderful, but to draw one image I have to sample D pixels one after another in a fixed order — sequential, non-parallelizable, and painful at image scale — and I'm left with no latent representation at all and a nagging sensitivity to the ordering I chose. And if I go adversarial, I get sharp samples and I escape the fixed reconstruction cost, but I've thrown away the likelihood entirely, so I can't evaluate density or even measure diversity, training is unstable, and there's no encoder from x back to z.

Stare at that list. The VAE blur comes from a fixed-form reconstruction cost. The autoregressive slowness comes from the sequential factorization. The GAN's missing likelihood and missing encoder come from abandoning maximum likelihood. What I want is the maximum-likelihood principle (for an exact objective and an exact encoder) but *without* a fixed reconstruction cost (so samples stay sharp) and *without* a sequential bottleneck (so sampling parallelizes). Is there any way to train by exact maximum likelihood and still get a generator I can run forward in one shot?

There is one corner of probability where exact maximum likelihood costs me nothing extra. Suppose my generator g that turns a latent z into an image x is not just any neural net but a *bijection* — invertible, with inverse f = g^{-1} carrying x back to z. Then I don't need a discriminator to avoid the likelihood, and I don't need an approximate posterior to do inference, because inference is literally f(x), exact and direct. If z = f(x) and I push an infinitesimal volume dx through f, it lands on a volume |det(∂f/∂xᵀ)| dx in z-space. Probability mass is conserved: p_X(x) dx = p_Z(z) dz, so p_X(x) = p_Z(f(x)) · |det(∂f(x)/∂xᵀ)|. Taking logs,

  log p_X(x) = log p_Z(f(x)) + log |det(∂f(x)/∂xᵀ)|.

That's it. Pick a simple prior p_Z — say a unit Gaussian — and I can compute the exact log-likelihood of any x, train by maximizing it directly, sample by drawing z ~ p_Z and returning x = f^{-1}(z), and infer by z = f(x). All four wishes, in principle, from one identity. This isn't a new idea in spirit — it's the maximum-likelihood view of ICA, it's Gaussianization, it's the old deep-density-model dream. So why isn't everyone doing this already?

Because of that determinant. For an arbitrary f on R^D, computing ∂f/∂x is a D×D Jacobian and its determinant is an O(D^3) operation, and a badly conditioned one. For an image with thousands of dimensions that's hopeless per training step, and it's numerically fragile on top of being slow. The exact, clean objective is strangled by one term. So the entire problem reduces to: can I design a bijection f that is genuinely expressive, trivially invertible, *and* whose Jacobian determinant I can read off cheaply?

Let me think about when a determinant is cheap. The determinant of a triangular matrix is just the product of its diagonal entries — no factorization, O(D). So if I could force the Jacobian of f to be triangular by construction, the determinant collapses to a sum of logs of diagonal terms and the whole obstruction evaporates. That's not a coincidence sitting off to the side, either: it's exactly why autoregressive models are tractable — conditioning each coordinate on the earlier ones makes the implied map's Jacobian triangular. But autoregressive triangularity comes bundled with sequential sampling, which is the very thing I'm trying to escape. I want the triangular Jacobian *without* the sequential inverse.

How do I build a map that is invertible, has a triangular Jacobian, and inverts in one parallel pass? Here's a construction worth trying. Split the D coordinates of x into two blocks. Leave the first block completely untouched, and transform the second block using only a function of the first block. Concretely, with a split at index d,

  y_{1:d}   = x_{1:d}
  y_{d+1:D} = something( x_{d+1:D} , using only x_{1:d} ).

Does this actually buy me triangularity? Look at the Jacobian. The top-left block ∂y_{1:d}/∂x_{1:d} is the identity because that half is copied. The top-right block ∂y_{1:d}/∂x_{d+1:D} is zero because the copied half doesn't depend on the second half at all. So the Jacobian is automatically block-lower-triangular,

  ∂y/∂xᵀ = [[ I_d , 0 ],
            [ ∂y_{d+1:D}/∂x_{1:d}ᵀ , ∂y_{d+1:D}/∂x_{d+1:D}ᵀ ]].

The determinant is the product of the two diagonal blocks' determinants, and the top one is 1. So the whole determinant is just det(∂y_{d+1:D}/∂x_{d+1:D}ᵀ) — it doesn't involve that messy off-diagonal block at all, which means the function that maps x_{1:d} into the transformation can be *arbitrarily* complicated and still never appear in the determinant. And invertibility: given y, the first block is already x_{1:d} = y_{1:d}, so I have in hand everything the second-block transform was conditioned on, and as long as that transform is invertible *in x_{d+1:D} for fixed x_{1:d}*, I can undo it. The conditioner never has to be inverted.

This recovers the additive-coupling idea from the earlier NICE work, where the second block is shifted by a learned function of the first:

  y_{1:d} = x_{1:d},   y_{d+1:D} = x_{d+1:D} + m(x_{1:d}),

with m any neural net, inverse x_{d+1:D} = y_{d+1:D} − m(y_{1:d}). The bottom-right Jacobian block ∂y_{d+1:D}/∂x_{d+1:D}ᵀ is the identity — a pure shift has unit slope — so the determinant of the *whole* coupling should be exactly 1, with the log-det term vanishing, m as deep as I like, and the inverse a subtraction. Let me actually check that on a concrete case before I lean on it. Take D = 4, d = 2, x = (0.3, −1.1, 0.7, 2.0), and a deliberately nonlinear conditioner m(a) = (a_0² − a_1, sin(a_0 a_1)). Building the 4×4 numerical Jacobian by central differences and taking its determinant gives det = 1.0000000006 — exactly 1 up to finite-difference error. So additive coupling is genuinely volume-preserving, which I'll come back to. Stack several with the roles of the two blocks alternating, so that whatever was frozen in one layer gets transformed in the next, and after enough layers every coordinate has been mixed.

But that unit determinant is also the limitation. det = 1 everywhere means the map is *volume-preserving*: it can move and curve probability mass around, but it can never locally compress or expand it. And density estimation is fundamentally about putting a lot of mass in some regions and very little in others — that *requires* the map to contract volume where the data piles up and expand it where the data is sparse. A volume-preserving f simply cannot do that with its coupling layers; the only place it can change volume at all is a single bolted-on diagonal scaling at the very end, contributing Σ_i log|S_ii| to the log-likelihood. So the expressive coupling stack is rigid exactly where I need it to be flexible, and one global diagonal is asked to carry all the volume change. That's the crack to pry open.

What's the minimal change that keeps the triangular-Jacobian gift but lets each coupling change volume? The additive layer's bottom-right block was the identity because I only *added* a function of the first half. The off-diagonal block is harmless to the determinant regardless of what's in it; the only thing the determinant sees is the bottom-right block. So to get a non-unit determinant, I should make the second block's transform *scale* the second block, not just shift it — and scale it by a function of the first block, so the conditioner still stays out of the determinant. Multiply elementwise and then add an offset:

  y_{1:d}   = x_{1:d}
  y_{d+1:D} = x_{d+1:D} ⊙ exp(s(x_{1:d})) + t(x_{1:d}),

with s and t — scale and translation — two functions from R^d to R^{D−d}, and ⊙ the elementwise product. Why exp(s) rather than a raw scale? Exp is always positive, so the per-coordinate scaling never hits zero and the map stays invertible; it also makes the log-det fall out cleanly. The bottom-right Jacobian block ∂y_{d+1:D}/∂x_{d+1:D}ᵀ is now diagonal, because coordinate j of the second block is x_{d+1:D,j} times exp(s(x_{1:d})_j) plus a constant-in-x_{d+1:D} offset, so its derivative is exactly exp(s(x_{1:d})_j) on the diagonal and zero off it. The full Jacobian should be

  ∂y/∂xᵀ = [[ I_d , 0 ],
            [ ∂y_{d+1:D}/∂x_{1:d}ᵀ , diag(exp(s(x_{1:d}))) ]],

triangular as before, with determinant the product of the diagonal of that bottom block, i.e. ∏_j exp(s_j) = exp(Σ_j s_j), so log|det| = Σ_j s(x_{1:d})_j.

That last step is the load-bearing claim of the whole method, so I don't want to take it on faith — let me trace it numerically on the same D = 4, d = 2 point. Pick genuinely nonlinear scale and translation nets, s(a) = (tanh(a_0) + 0.5 a_1, sin(a_0 a_1)) and t(a) = (a_0² − a_1, 0.3 a_0 + cos a_1). With a = x_{1:2} = (0.3, −1.1) that gives s = (−0.2587, −0.3240), so Σ s = −0.5827. Now form the forward map y = (x_{1:2}, x_{3:4}⊙exp(s) + t) and build its 4×4 Jacobian by central differences:

  [[ 1.0000  0       0       0     ]
   [ 0       1.0000  0       0     ]
   [ 1.0946 −0.7298  0.7721  0     ]
   [−1.2052  1.3017  0       0.7232]]

This is exactly the predicted shape: identity top-left, a zero top-right block, a dense and frankly ugly bottom-left block holding the derivatives of s and t through x_{1:2}, and a diagonal bottom-right block. The two diagonal entries of that bottom-right block are 0.7721 and 0.7232, and exp(−0.2587) = 0.7721, exp(−0.3240) = 0.7232 — they match exp(s) coordinatewise. Taking the log-determinant of the full matrix numerically gives −0.58273042, against Σ s = −0.58273042. They agree to eight digits, and crucially the determinant came out completely independent of the messy bottom-left entries (1.0946, −0.7298, −1.2052, 1.3017) — those are the derivatives of s and t with respect to x_{1:d}, sitting strictly below the diagonal, and they never entered the determinant. So s and t can be arbitrarily deep convolutional nets at zero cost to the determinant. That is the property I was after.

And the inverse should be just as cheap, with no inverse of s or t required. From y, read off x_{1:d} = y_{1:d}, feed it through s and t, and undo the affine map on the second block:

  x_{1:d}   = y_{1:d}
  x_{d+1:D} = ( y_{d+1:D} − t(y_{1:d}) ) ⊙ exp( −s(y_{1:d}) ).

Subtract the translation, divide by the scale (i.e. multiply by exp(−s)). Running this on the y I just computed returns (0.3, −1.1, 0.7, 2.0), bit-for-bit the original x — so the closed-form inverse really is exact, and it costs one pass through s and t, the same as the forward direction. That means sampling is exactly as efficient as inference, the autoregressive bottleneck dissolved. And det = exp(Σ s) is generically not 1 — here it's exp(−0.5827) ≈ 0.56, a genuine local contraction of volume — so this map can locally contract or expand mass, exactly the freedom the additive version lacked, and the global diagonal scaling NICE bolted on is no longer needed because the layers themselves reshape volume.

One coupling freezes half the coordinates, so I have to compose. Does composition keep both gifts — tractable determinant and tractable inverse? Take f_b ∘ f_a. By the chain rule the Jacobian factorizes,

  ∂(f_b ∘ f_a)/∂x_aᵀ (x_a) = ∂f_b/∂x_bᵀ(x_b = f_a(x_a)) · ∂f_a/∂x_aᵀ(x_a),

and determinants are multiplicative, det(AB) = det(A)det(B), so the log-dets should simply *add* across layers. Let me make sure I'm not fooling myself, because the second layer is evaluated at the *output* of the first, and it would be easy to forget that. Stack two affine couplings with the frozen half alternating — layer A freezes x_{1:2}, layer B freezes x_{3:4} — each with its own little nonlinear s and t, and compare the numerically-differentiated log-det of the composed map against the sum of the two layers' individual Σ s. Composed log|det| = −0.38114923; ld_A + ld_B = −0.38114923. They match, and the second layer's contribution was correctly its s evaluated at the intermediate state, not at x. So stacking is free on both counts — the inverse composes in the obvious reversed order, (f_b ∘ f_a)^{-1} = f_a^{-1} ∘ f_b^{-1}. I just need to alternate which half is frozen so that every coordinate gets transformed by some layer; a coordinate that's only ever copied would be modeled as a raw Gaussian forever.

Now, how do I split an *image* into two halves? A flat index split into first-d-and-rest ignores everything I know about images — that correlation is local in 2-D and across channels. Let me make the partition a binary mask b and write the coupling in masked form, so the "frozen half" is wherever b = 1:

  y = b ⊙ x + (1 − b) ⊙ ( x ⊙ exp(s(b ⊙ x)) + t(b ⊙ x) ).

Where b = 1, y = x (frozen); where b = 0, y is the affine transform conditioned on the masked input b ⊙ x. Feeding b ⊙ x into s and t means the conditioner literally sees only the frozen pixels, keeping the triangular structure intact. What masks respect image structure? A *spatial checkerboard*: b = 1 where the sum of the spatial coordinates is odd, 0 elsewhere — so each transformed pixel is conditioned on its immediate spatial neighbors, which is where the correlation is. And a *channel-wise* mask: b = 1 on the first half of the channels, 0 on the second half — so whole channels condition on other whole channels. With s and t as rectified convolutional nets, the conditioner exploits the 2-D neighborhood directly.

There's a practical danger lurking in exp(s) that I should head off now. If s ever outputs a large positive number, exp(s) explodes and the forward pass — and the gradient — blow up; NaNs follow. I don't want to cap expressiveness forever, but I do want to keep the raw scale signal bounded and let the model *learn* how much range it needs. So pass the scale net's output through a tanh, which bounds the raw output to (−1, 1), and then multiply by a separate learned per-channel scale factor that can grow as needed. The translation t needs no such guardrail — it's additive, exp doesn't touch it. Wrapping that learned rescale in weight normalization keeps it well-behaved during training. So s in practice is (learned scale) · tanh(raw conv output), a controlled parameterization that prevents immediate scale explosions while preserving room to expand the range.

Even with that, a deep stack of these exp-scaled layers is numerically touchy — the interplay of linear and exponential contributions makes the loss surface nasty, and training a conditional distribution through a scale parameter by gradients is exactly where instability tends to bite. I'd like to normalize activations inside the flow the way batch norm normalizes activations in ordinary nets. But here's a subtlety: batch norm is itself a transformation of the variable, so if I drop it into the flow I'm changing the density and I owe its Jacobian term. Is that a problem? Batch norm rescales each dimension as x ↦ (x − μ̃) / √(σ̃² + ε), which is just a per-dimension affine map — diagonal Jacobian, slope 1/√(σ̃² + ε) on each coordinate. Its determinant is the product of those slopes, ∏_i (σ̃_i² + ε)^{−1/2}, so its forward log-det should be −½ Σ_i log(σ̃_i² + ε). Quick check on a 3-dimensional case with variances (0.5, 2.0, 0.3) and ε = 1e−4: the diagonal Jacobian diag(1/√(σ̃²+ε)) has numerical log|det| = 0.60169477, and −½ Σ log(σ̃²+ε) = 0.60169477. Same number. In the inverse direction, x ↦ x√(σ̃² + ε) + μ̃ multiplies by √(σ̃² + ε), so it contributes +½ Σ_i log(σ̃_i² + ε), the same magnitude with flipped sign. Batch norm isn't an obstacle to the change-of-variables bookkeeping; it's another invertible layer with an easy determinant whose sign follows the direction of the map. Including it lets a much deeper coupling stack train, and it tames the scale-parameter instability directly. One refinement: with the usual per-batch statistics, very small minibatches make μ̂, σ̂² noisy. So use a running average over recent minibatches, μ̃_{t+1} = ρ μ̃_t + (1−ρ) μ̂_t and σ̃²_{t+1} = ρ σ̃²_t + (1−ρ) σ̂²_t, while backpropagating only through the current-batch statistics — a small lag that makes training robust at small batch size.

Now I have to worry about cost. Pushing the full D-dimensional image through every coupling layer at full resolution is wasteful in computation, memory, and parameters, and it also means deep, fine spatial detail and coarse global structure are being modeled at the same resolution throughout. Images are multi-scale; the model should be too. Reshaping an s×s×c tensor into (s/2)×(s/2)×4c by taking each 2×2×c spatial block and stacking it into 1×1×4c trades spatial extent for channels without changing any values. It's a pure deterministic reshuffle — a permutation of the elements — so its Jacobian is a permutation matrix with absolute determinant 1 and it contributes nothing to the log-det. The convolutions now see a larger effective receptive field, and a *channel-wise* mask becomes meaningful because the squeeze has packed spatially-adjacent pixels into the channel axis. So a natural rhythm per scale is: three coupling layers with alternating checkerboard masks at full resolution, then squeeze, then three coupling layers with alternating channel-wise masks — choosing the channel partition so it isn't redundant with the checkerboard partition I just used.

And rather than carry all D dimensions through all scales, *factor out* half of them at each scale. After a scale's couplings, peel off half the coordinates as finished latent variables and send only the other half deeper. Recursively, with h^{(0)} = x,

  (z^{(i+1)}, h^{(i+1)}) = f^{(i+1)}( h^{(i) }),   z^{(L)} = f^{(L)}( h^{(L−1)} ),   z = (z^{(1)}, …, z^{(L)}),

so the final latent is the concatenation of everything peeled off at each scale. This is not just a shortcut. It slashes computation, memory, and parameter count, letting me train a much larger model — the same factoring-of-computation logic behind multi-scale conv architectures like VGG. It distributes the loss throughout the network: because peeled-off units are scored against the prior right where they're factored out, every scale gets a direct training signal, the deep-supervision philosophy of guiding intermediate layers with their own objectives. It also builds a hierarchy: units factored out at a finer (earlier) scale must be Gaussianized before those factored out at a coarser (later) scale, so the model naturally separates local fine-grained features from global coarse ones — coarse-to-fine levels of representation. As the spatial resolution halves at each squeeze, I double the number of hidden features in s and t, keeping capacity matched to the data at each scale. For the very last scale, where there's nothing left to factor into, I just apply a few coupling layers with checkerboard masks.

The objective still has to meet the data domain. The model assumes its variable lives in unbounded continuous space, but pixels are bounded integers with 256 levels per channel. If the tensor is normalized to [0,1], dequantize by adding uniform noise to the underlying integer level, r = (255x + u)/256 with u ~ U[0,1], so the discrete grid becomes continuous. Then keep the logit away from the exact boundaries: choose α = .05, set β = 1 − 2α = .9, form v = α + βr, and model y = logit(v). I need the log-det of this preprocessing map too. The derivative of y = logit(v) with respect to r is β·(1/v + 1/(1−v)) = β/[v(1−v)], so log|dy/dr| = log β − log v − log(1−v). Since v = sigmoid(y), I claim −log v − log(1−v) = softplus(y) + softplus(−y); let me confirm at a representative point r = 0.37: there v = 0.383, y = logit(v) = −0.476, the numerical log|dy/dr| works out to 1.33724603, and softplus(y) + softplus(−y) + log β = 1.33724603 — identical, and the −log v − log(1−v) versus softplus(y)+softplus(−y) halves match on their own. So the log-det of this constraint-and-logit transform is softplus(y) + softplus(−y) + log β per dimension. The dequantization scale is the constant I account for when reporting a discrete-pixel code length. The prior can stay simple: log p_Z(z) = Σ_i −½(z_i² + log 2π) under an isotropic unit Gaussian. Any factorial prior would do — even an autoregressive model or a VAE could sit at the top — but the unit Gaussian is the clean default, trivially differentiable, and it's what the whole flow is Gaussianizing the data into.

So the full objective is exactly the change-of-variables log-likelihood I start from, with the total log-det accumulated across every coupling layer (Σ s terms), every batch-norm layer (−½ Σ log(σ̃²+ε) terms), and the preprocessing transform, and with z = f(x) scored under the unit Gaussian. For k = 256 discrete levels, I subtract D·log k from the log-likelihood, equivalently add D·log k to the negative log-likelihood, and then divide by D·log 2 for bits per dimension. I maximize this directly with Adam — exact likelihood, parallel sampling via x = f^{-1}(z), exact inference via z = f(x), and a latent space organized coarse-to-fine.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_norm = nn.BatchNorm2d(channels)
        self.in_conv = WNConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_conv = WNConv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x
        x = self.in_conv(F.relu(self.in_norm(x)))
        x = self.out_conv(F.relu(self.out_norm(x)))
        return x + skip


class ResNet(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_blocks,
                 kernel_size=3, padding=1, double_after_norm=False):
        super().__init__()
        self.in_norm = nn.BatchNorm2d(in_ch)
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(2 * in_ch, mid_ch, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_ch, mid_ch, kernel_size=1, padding=0, bias=True)
        self.blocks = nn.ModuleList([ResidualBlock(mid_ch) for _ in range(num_blocks)])
        self.skips = nn.ModuleList([
            WNConv2d(mid_ch, mid_ch, kernel_size=1, padding=0, bias=True)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.BatchNorm2d(mid_ch)
        self.out_conv = WNConv2d(mid_ch, out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_norm(x)
        if self.double_after_norm:
            x = 2. * x
        x = self.in_conv(F.relu(torch.cat((x, -x), dim=1)))
        x_skip = self.in_skip(x)
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip = x_skip + skip(x)
        return self.out_conv(F.relu(self.out_norm(x_skip)))


def squeeze_2x2(x, reverse=False, alt_order=False):
    if reverse:                                   # 4c -> c, double H and W
        b, c4, h, w = x.size()
        c = c4 // 4
        if alt_order:
            x = x.view(b, 4, c, h, w).permute(0, 2, 1, 3, 4)
            x = x[:, :, [0, 2, 3, 1]].contiguous().view(b, c4, h, w)
        return x.view(b, c, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).contiguous().view(
            b, c, 2 * h, 2 * w)

    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, 4 * c, h // 2, w // 2)
    if alt_order:
        x = x.view(b, c, 4, h // 2, w // 2)
        x = x[:, :, [0, 3, 1, 2]].permute(0, 2, 1, 3, 4).contiguous().view(
            b, 4 * c, h // 2, w // 2)
    return x


def checkerboard_mask(h, w, reverse=False, device=None):
    cb = [[((i % 2) + j) % 2 for j in range(w)] for i in range(h)]
    mask = torch.tensor(cb, dtype=torch.float32, device=device)
    if reverse:
        mask = 1 - mask
    return mask.view(1, 1, h, w)


class Rescale(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        return self.weight * x


class FlowBatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-4, rho=0.99):
        super().__init__()
        self.eps = eps
        self.rho = rho
        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, num_channels, 1, 1))

    def _stats(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            mean = self.rho * self.running_mean + (1. - self.rho) * batch_mean
            var = self.rho * self.running_var + (1. - self.rho) * batch_var
            with torch.no_grad():
                self.running_mean.copy_(mean.detach())
                self.running_var.copy_(var.detach())
            return mean, var
        return self.running_mean, self.running_var

    def forward(self, x, sldj, reverse=False):
        _, _, h, w = x.size()
        mean, var = (self.running_mean, self.running_var) if reverse else self._stats(x)
        log_var = torch.log(var + self.eps)
        if reverse:
            x = x * torch.exp(0.5 * log_var) + mean
            if sldj is not None:
                sldj = sldj + 0.5 * h * w * log_var.view(-1).sum()
            return x, sldj
        x = (x - mean) * torch.exp(-0.5 * log_var)
        if sldj is not None:
            sldj = sldj - 0.5 * h * w * log_var.view(-1).sum()
        return x, sldj


class CouplingLayer(nn.Module):
    """Affine coupling: y_change = x_change * exp(s) + t."""
    def __init__(self, in_channels, mid_channels, num_blocks, channel_wise, reverse_mask):
        super().__init__()
        self.channel_wise = channel_wise
        self.reverse_mask = reverse_mask
        cond_in = in_channels // 2 if channel_wise else in_channels
        self.st_net = ResNet(cond_in, mid_channels, 2 * cond_in, num_blocks,
                             double_after_norm=not channel_wise)
        self.rescale = nn.utils.weight_norm(Rescale(cond_in))
        self.flow_bn = FlowBatchNorm(in_channels)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            x, sldj = self.flow_bn(x, sldj, reverse=True)

        if not self.channel_wise:                          # spatial checkerboard
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, x.device)
            st = self.st_net(x * b)                         # condition on frozen half
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))                 # tanh + learned scale: keep exp(s) sane
            s, t = s * (1 - b), t * (1 - b)                 # only act where mask is 0
            if reverse:                                     # undo: subtract t, then divide by exp(s)
                x = (x - t) * s.mul(-1).exp()
                if sldj is not None:
                    sldj = sldj - s.view(s.size(0), -1).sum(-1)
            else:                                           # apply: scale by exp(s), then shift by t
                x = x * s.exp() + t
                sldj = sldj + s.view(s.size(0), -1).sum(-1) # accumulate log-det = sum s
        else:                                              # channel-wise split
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)
            st = self.st_net(x_id)                          # condition on the frozen channels
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            if reverse:
                x_change = (x_change - t) * s.mul(-1).exp()
                if sldj is not None:
                    sldj = sldj - s.view(s.size(0), -1).sum(-1)
            else:
                x_change = x_change * s.exp() + t
                sldj = sldj + s.view(s.size(0), -1).sum(-1)
            x = torch.cat((x_id, x_change), dim=1) if self.reverse_mask \
                else torch.cat((x_change, x_id), dim=1)
        if not reverse:
            x, sldj = self.flow_bn(x, sldj, reverse=False)
        return x, sldj


class _RealNVP(nn.Module):
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super().__init__()
        self.is_last = scale_idx == num_scales - 1
        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, False, False),
            CouplingLayer(in_channels, mid_channels, num_blocks, False, True),
            CouplingLayer(in_channels, mid_channels, num_blocks, False, False),
        ])
        if self.is_last:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, False, True))
        else:
            self.out_couplings = nn.ModuleList([                      # after squeeze: 4x channels
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, False),
            ])
            self.next = _RealNVP(scale_idx + 1, num_scales,
                                 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for c in self.in_couplings:
                x, sldj = c(x, sldj, reverse)
            if not self.is_last:
                x = squeeze_2x2(x)                            # space -> channels
                for c in self.out_couplings:
                    x, sldj = c(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
                x = squeeze_2x2(x, alt_order=True)            # factor out half the dims
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)         # only the other half goes deeper
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
        else:
            if not self.is_last:
                x = squeeze_2x2(x, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
                x = squeeze_2x2(x)
                for c in reversed(self.out_couplings):
                    x, sldj = c(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
            for c in reversed(self.in_couplings):
                x, sldj = c(x, sldj, reverse)
        return x, sldj


class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super().__init__()
        self.register_buffer('data_constraint', torch.tensor([0.9]))
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def _preprocess(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.           # dequantize discrete pixels
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()                         # logit -> unbounded support
        ldj = F.softplus(y) + F.softplus(-y) + self.data_constraint.log()
        sldj = ldj.view(ldj.size(0), -1).sum(-1)             # log-det of the preprocessing
        return y, sldj

    def _postprocess(self, y):
        x = y.sigmoid()
        x = (2. * x - 1.) / self.data_constraint
        return ((x + 1.) / 2.).clamp(0., 1.)

    def forward(self, x, reverse=False):
        sldj = None
        if not reverse:
            x, sldj = self._preprocess(x)
        x, sldj = self.flows(x, sldj, reverse)
        return x, sldj

    def sample(self, z):
        y, _ = self.forward(z, reverse=True)
        return self._postprocess(y)

class RealNVPLoss(nn.Module):
    def __init__(self, k=256):
        super().__init__()
        self.k = k

    def log_likelihood(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))       # log N(z;0,I) per dim
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) \
                   - np.log(self.k) * np.prod(z.size()[1:])  # discrete-levels correction
        return prior_ll + sldj                                # + accumulated log-det

    def forward(self, z, sldj):
        return -self.log_likelihood(z, sldj).mean()

    def bits_per_dim(self, z, sldj):
        dims = np.prod(z.size()[1:])
        return (-self.log_likelihood(z, sldj) / (dims * np.log(2))).mean()
```

Looking back at the path: I started wanting exact likelihood, fast sampling, exact inference, and a meaningful latent at the same time, and the change-of-variables identity gives all four through a bijective generator — provided the Jacobian determinant is cheap. Forcing the Jacobian triangular (freeze half the coordinates, transform the other half conditioned on the frozen half) makes the determinant a product of its diagonal, and the small numeric trace confirmed the off-diagonal derivatives of the conditioner never enter it. The additive version of that idea pinned the determinant at exactly 1 — I verified it — which is volume-preserving and too rigid for density estimation; switching the shift to an exp-scale gives det = exp(Σ s), and the same numeric check confirmed the log-det is Σ s and the closed-form inverse is exact. Composition adds log-dets (also checked, with the second layer evaluated at the intermediate state), so I stack alternating masks; checkerboard then channel-wise around a squeeze exploits image locality; tanh plus learned scale controls the scale parameterization; batch norm folds in as one more easy-determinant bijector contributing −½ Σ log(σ̃²+ε) forward and +½ Σ log(σ̃²+ε) reverse (the magnitudes matched a 3-D check); squeezing and factoring out half the dimensions per scale cut cost, distribute the loss, and build a coarse-to-fine hierarchy; and dequantization, the constrained logit transform (whose softplus log-det I checked at a point), the −D·log k discrete correction, and a unit-Gaussian prior close the loop into an exact log-likelihood I maximize with Adam.
