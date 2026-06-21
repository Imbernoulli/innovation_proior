The dependencies I most want a network to use are often the long-range ones: a ball in the first frame of a video and the same ball in the last frame are related, but they are far apart in both space and time. The two operators I have for connecting positions — convolution and recurrence — share a property that turns out to be the whole problem: they only ever touch a *local* neighborhood. A size-3 convolution at position $i$ mixes $i-1, i, i+1$; a recurrent step at time $i$ looks only at the current and the latest step. Neither relates two distant positions in a single operation. The only way these operators reach far is by *stacking*: apply the local op again and again until the receptive field grows large enough that distant positions can, indirectly, influence one another. That works, but it is a workaround with three costs. It is computationally wasteful — you pay for a deep stack mostly to slowly grow a receptive field. It is hard to optimize — deep stacks have the signal-propagation problems that residual connections and gating were invented to patch. And it makes *multi-hop* reasoning clumsy, because if a message must travel from position $A$ to $B$ and back, every hop costs more depth and the gradient must survive the round trip. A fully-connected layer does touch all positions, but its weights are fixed parameters rather than functions of the data, it forces a fixed input size, and it destroys the correspondence between an input position and its output position. So none of convolution, recurrence, or the fully-connected layer combines all-position reach with data-dependent, size-flexible, position-preserving behavior.

The idea I want already exists outside neural networks. Non-local means denoises a pixel not by averaging its local neighborhood but by averaging over *all* pixels in the image, weighting each by how similar its surrounding patch is to the patch around the target — a pixel on the far side of the image, if it sits in a similar texture, contributes. That non-locality is the engine behind block-matching denoising, texture synthesis, super-resolution, and inpainting, and it has been essentially ignored in modern vision networks. I propose to turn it into a generic network operator: the **non-local block**. For an input $x$ (an image, a sequence, a video, or their feature maps) and an output $y$ of the same size, I define the response at output position $i$ as

$$y_i = \frac{1}{C(x)} \sum_{\forall j} f(x_i, x_j)\, g(x_j),$$

where $j$ ranges over *all* positions — that $\forall j$ is the non-locality, nothing excluded by distance — $f(x_i, x_j)$ is a scalar saying how related $i$ and $j$ are, $g(x_j) = W_g x_j$ is a learned unary content embedding (a $1\times1$ or $1\times1\times1$ convolution, the cheapest per-position representation), and $C(x)$ normalizes. This is unlike convolution and recurrence because the sum runs over all $j$ rather than a local set, and unlike a fully-connected layer because $f$ is computed *from the data*, the operation handles any number of positions, and output $i$ stays tied to input position $i$ — so, crucially, it can live in the *middle* of a network where I want long-range mixing on feature maps, not only at the end.

What remains is the affinity $f$. The natural first choice, following non-local means and bilateral filters, is a Gaussian of dot-product similarity, $f(x_i,x_j) = e^{x_i^\top x_j}$ with $C(x) = \sum_{\forall j} f$ so the weights over $j$ sum to one. (Classical non-local means used Euclidean distance in the exponent; the dot product is equivalent up to normalization and is just a matrix multiply, which is far friendlier on a deep-learning platform.) But comparing raw features is rigid — the features I am handed were not necessarily produced to make this similarity meaningful — so I compute the similarity in a *learned* embedding space, embedding $x_i$ and $x_j$ through two separate linear maps before comparing:

$$f(x_i,x_j) = e^{\theta(x_i)^\top \phi(x_j)}, \qquad \theta(x_i) = W_\theta x_i, \quad \phi(x_j) = W_\phi x_j,$$

again with $C(x) = \sum_j f$. This embedded Gaussian is strictly more flexible than the plain one (set $W_\theta = W_\phi = I$ to recover it), and now the network can *learn* what "related" means. Writing out the normalized weight from $j$ reveals something: it is $e^{\theta(x_i)^\top \phi(x_j)} / \sum_{j'} e^{\theta(x_i)^\top \phi(x_{j'})}$, which is exactly a *softmax* over $j$, so the whole operation becomes $y = \mathrm{softmax}(x^\top W_\theta^\top W_\phi x)\, g(x)$ — precisely self-attention. Self-attention, the mechanism from machine translation, is therefore a special case of the non-local operation: the embedded-Gaussian instantiation with softmax normalization, applied to a 1-D sequence. What I am doing generalizes it from 1-D language sequences to space and spacetime.

That unification makes me suspicious of the softmax: if attention is just one instantiation, is the softmax doing the essential work, or is the non-locality the real source of power? To probe this I deliberately build versions *without* softmax. First, the raw dot product, $f(x_i,x_j) = \theta(x_i)^\top \phi(x_j)$, linear, with $C(x) = N$, the number of positions. I divide by the constant $N$ rather than by a data-dependent $\sum_j f$ for two reasons: it keeps the output scale roughly invariant to how many positions there are (which I need, since input size varies), and it simplifies the gradient, since the normalizer no longer depends on the features. Second, borrowing the pairwise form from relation networks, $f(x_i,x_j) = \mathrm{ReLU}(w_f^\top[\theta(x_i),\phi(x_j)])$, again with $C(x) = N$. These four instantiations — Gaussian, embedded Gaussian (= self-attention), dot product, concatenation — perform comparably, and that is itself the point: the softmax/attention framing is not the essence; the non-locality, summing over all $j$ with data-dependent weights, is what matters. The choice of $f$ is a detail; the $\forall j$ is the substance.

To make this a reusable component I can insert into a *pretrained* network without retraining from scratch, I wrap the operation in a residual block,

$$z_i = W_z y_i + x_i,$$

where $W_z$ (a $1\times1\times1$ conv) projects the non-local response $y$ back to the channel count of $x$. The key move: initialize $W_z$ to *zero* — equivalently, put a batch-norm after $W_z$ and zero-initialize its scale — so that at initialization $z_i = 0 + x_i = x_i$, an exact identity. The block can then be dropped into any pretrained network where it does nothing at first, leaving pretrained behavior intact, and training gradually learns a useful non-local correction. The pairwise computation, in tensor form, is just two matrix multiplications and a softmax: form the $N\times N$ affinity matrix by multiplying the $\theta$-embedded positions against the $\phi$-embedded ones, normalize (softmax along rows, or scale by $1/N$), and multiply against the $g$-embedded positions to get the weighted sums.

The $N\times N$ matrix is the cost worry, so I shave it two ways. A bottleneck, as in residual networks: set $W_g, W_\theta, W_\phi$ to output *half* the channels of $x$, halving the block's compute, with $W_z$ mapping back up. And a subsampling trick: I do not need every position to be "non-local," so I attend to a subsampled set, replacing the keys/values $x$ by a pooled $\hat x$,

$$y_i = \frac{1}{C(\hat x)} \sum_{\forall j} f(x_i, \hat x_j)\, g(\hat x_j),$$

by max-pooling after $\phi$ and $g$; pooling the keys spatially by $2\times$ cuts the number of $j$'s by roughly four. The output is still computed for every $i$ and still gathers from across the whole map, just from a sparser set. Placement interacts with the $O(N^2)$ scaling: the block is cheap on the high-level, already-subsampled maps deep in the network, where resolution is small enough that the matmul is comparable to one ordinary convolution. So I add blocks in the mid-to-late ResNet stages (res3/res4), spreading several across them one every other residual block; the very last stage (res5, $7\times7$) has too few positions to gather rich long-range structure, so I avoid it.

For video, I host the blocks in an ImageNet-pretrained ResNet used as a per-frame **C2D** backbone — all kernels act within a frame as $1\times k\times k$ and the temporal dimension is handled only by pooling — which isolates what the blocks add, since the backbone itself does almost nothing temporal. As a stronger comparison I inflate the 2D kernels into 3D (**I3D**): a $k\times k$ kernel becomes a $t\times k\times k$ kernel by copying it into each of the $t$ temporal planes and rescaling by $1/t$, so that a static repeated clip reproduces exactly the 2D model's output — a clean initialization. The non-local blocks are *complementary* to 3D convolution rather than redundant: 3D conv captures *local* spacetime structure, non-local blocks capture *long-range* spacetime structure, and one block on a subsampled map costs about as much as a single conv layer while connecting the whole spacetime volume. I fine-tune with batch-norm *enabled* (which, unusually, reduces overfitting on this large video model), initialize the new layers with the standard rectifier scheme, keep the single zero-initialized BN on $W_z$ so the block starts as identity, and use dropout $0.5$ after global pooling. Nothing in the affinity is temporally special: positions are positions, and the same $\forall j$ sum spans the entire spacetime volume, which is exactly how it relates the ball in the first frame to the ball in the last.

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
