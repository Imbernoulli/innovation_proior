# Non-local Neural Networks

## Problem
Convolution and recurrence both operate on a *local* neighborhood, so they capture long-range
dependencies only by stacking many layers and propagating signals step by step — which is
computationally inefficient, hard to optimize, and poor at multi-hop interactions between distant
positions. Build a single, generic operator that relates *any* two positions directly (in space, time,
or spacetime), preserves variable input size and positional layout, and drops into existing networks.

## Key idea
Generalize the classical **non-local means** filter into a network operation: the response at a
position is a data-weighted sum of a content embedding at *all* positions.

  y_i = (1/C(x)) · Σ_{∀j} f(x_i, x_j) · g(x_j),

with i the output position, j ranging over all positions, f a pairwise affinity, g a unary content
embedding (g(x_j) = W_g x_j, a 1×1(×1) conv), and C(x) a normalizer. The "∀j" is the non-locality.
This differs from convolution/recurrence (which are local) and from a fully-connected layer (whose
weights are fixed parameters, not data-dependent, and which loses variable size and position).

## Instantiations of the affinity f
- **Gaussian:** f = exp(x_iᵀ x_j), C(x) = Σ_j f.
- **Embedded Gaussian:** f = exp(θ(x_i)ᵀ φ(x_j)), θ = W_θ x, φ = W_φ x, C(x) = Σ_j f. Here the
  normalized weight (1/C)·f is a **softmax over j**, so y = softmax(xᵀW_θᵀW_φ x)·g(x) — exactly
  self-attention. Self-attention is the embedded-Gaussian special case of the non-local operation.
- **Dot product:** f = θ(x_i)ᵀ φ(x_j), C(x) = N (number of positions) — no softmax; dividing by the
  constant N keeps the scale size-invariant and simplifies the gradient.
- **Concatenation:** f = ReLU(w_fᵀ[θ(x_i), φ(x_j)]), C(x) = N.

The model is insensitive to which f is used — the softmax/attentional behavior is *not* essential; the
non-locality is what matters.

## Non-local block
Wrap the operation in a residual block:

  z_i = W_z y_i + x_i,

where W_z (a 1×1×1 conv) projects y back to the input channel count. Initializing W_z to zero
(equivalently, a BatchNorm after W_z with its scale initialized to zero) makes the block an **identity
at initialization**, so it can be inserted into any pretrained network without disturbing it.

## Efficiency
- **Bottleneck:** W_g, W_θ, W_φ output half the input channels (≈½ the block compute).
- **Subsampling:** max-pool the keys/values (after φ and g) so the pairwise sum runs over ~¼ as many
  positions; still non-local, just sparser.
- **Placement:** the pairwise cost is O(N²), so blocks go on the mid/high, already-subsampled stages
  (res3/res4) where N is small; the very last stage (res5, 7×7) has too few positions to gather rich
  long-range structure. For several blocks, insert one every other residual block.

## Video instantiation
Host the blocks in an ImageNet-pretrained ResNet used as a per-frame **C2D** backbone (temporal info via
pooling) or inflated to **I3D** (2D k×k kernels → 3D t×k×k, each temporal plane = the pretrained 2D
kernel rescaled by 1/t, so a static clip reproduces the 2D model). Non-local blocks capture *long-range*
spacetime structure, complementary to and cheaper than local 3D convolution. Fine-tune with BatchNorm
enabled (reduces overfitting on the large video model); dropout 0.5 after global pooling.

## Code
```python
import torch
from torch import nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    """z = W_z y + x; y_i = (1/C(x)) sum_j f(x_i,x_j) g(x_j). Identity at init."""
    def __init__(self, channels, mode="embedded_gaussian", subsample=True):
        super().__init__()
        self.mode = mode
        inter = channels // 2                            # bottleneck
        self.g     = nn.Conv3d(channels, inter, 1)
        self.theta = nn.Conv3d(channels, inter, 1)
        self.phi   = nn.Conv3d(channels, inter, 1)
        if mode == "concatenation":
            self.w_f = nn.Conv2d(2 * inter, 1, 1)
        self.W_z = nn.Conv3d(inter, channels, 1)
        self.bn  = nn.BatchNorm3d(channels)
        nn.init.zeros_(self.bn.weight)                   # block starts as identity
        self.pool = nn.MaxPool3d((1, 2, 2)) if subsample else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        g_x   = self.pool(self.g(x)).flatten(2)          # (B, inter, Nj)
        theta = self.theta(x).flatten(2)                 # (B, inter, Ni)
        phi   = self.pool(self.phi(x)).flatten(2)        # (B, inter, Nj)
        Ni, Nj = theta.size(2), phi.size(2)

        if self.mode in ("gaussian", "embedded_gaussian"):
            scores = torch.bmm(theta.transpose(1, 2), phi)
            weights = scores.softmax(dim=-1)             # C = sum_j f
        elif self.mode == "dot_product":
            scores = torch.bmm(theta.transpose(1, 2), phi)
            weights = scores / Nj                        # C = N, no softmax
        elif self.mode == "concatenation":
            ti = theta[:, :, :, None].expand(-1, -1, Ni, Nj)
            pj = phi[:, :, None, :].expand(-1, -1, Ni, Nj)
            f = F.relu(self.w_f(torch.cat([ti, pj], 1))).squeeze(1)
            weights = f / Nj                             # C = N

        y = torch.bmm(weights, g_x.transpose(1, 2)).transpose(1, 2).view(B, -1, T, H, W)
        return self.bn(self.W_z(y)) + x                  # residual; identity at init


def inflate_2d_to_3d(kernel_2d, t):
    """3D t x k x k kernel from a pretrained 2D k x k kernel."""
    return kernel_2d[:, :, None].repeat(1, 1, t, 1, 1) / t


class NonLocalResNetVideo(nn.Module):
    def __init__(self, depth=50, num_classes=400, n_blocks=5, inflate=False):
        super().__init__()
        self.stem, self.res2, self.res3, self.res4, self.res5 = build_resnet_video(depth, inflate)
        self.nl3 = nn.ModuleList([NonLocalBlock(512) for _ in range(n_blocks // 2)])
        self.nl4 = nn.ModuleList([NonLocalBlock(1024) for _ in range(n_blocks - n_blocks // 2)])
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, clip):                             # (B, 3, T, 224, 224)
        x = self.res2(self.stem(clip))
        x = interleave(self.res3, self.nl3, x)           # block every other residual unit
        x = interleave(self.res4, self.nl4, x)
        x = self.res5(x).mean(dim=(2, 3, 4))             # global spacetime pool
        return self.head(self.dropout(x))
```
