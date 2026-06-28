Let me start from a frustration I keep running into: how badly our standard operators handle long-range dependencies. In a video, a ball in the first frame and the same ball in the last frame are related, and that relationship is what I'd want the network to use — but the two are far apart in both space and time. The tools I have for connecting them are convolution and recurrence, and both share a property that, the more I think about it, looks like the core problem: they only ever touch a *local* neighborhood. A convolution with a size-3 kernel at position i mixes positions i−1, i, i+1 — purely local. A recurrent step at time i looks at the current and the latest step, j = i or i−1 — purely local. Neither can relate two distant positions in a single operation.

So how do convolutional and recurrent networks ever capture long-range structure? By stacking. Apply the local operation again and again, and a signal propagates a little further each layer, until after enough layers the receptive field is large enough that distant positions can, indirectly, influence each other. That works, but it's a workaround, and it has three costs that bother me. It's computationally wasteful — I'm paying for a deep stack of layers mostly to slowly grow a receptive field. It's hard to optimize — deep stacks have the signal-propagation problems that residual connections and gating were invented to patch. And it makes *multi-hop* reasoning clumsy: if the answer requires passing a message from position A to position B and back, every hop costs more depth, and the gradient has to survive the round trip. What I really want is an operator that connects any two positions *directly*, in one step, no matter how far apart they are.

Let me look outside neural networks for where this has already been solved, because "let distant things interact directly" is an old idea in image processing. Non-local means is exactly it: to denoise a pixel, don't average its local neighborhood — average over *all* pixels in the image, weighting each by how similar its surrounding patch looks to the patch around the target pixel. A pixel on the far side of the image, if it sits in a similar texture, contributes. The whole point is non-locality: similar locations interact regardless of distance. This idea is the engine behind block-matching denoising, texture synthesis, super-resolution, inpainting — it's been quietly central to image processing for years, and it has essentially been ignored in modern vision networks. Let me try to turn it into a network operator and see how far the analogy carries.

Let me write down the most generic form of "the response at a position is a weighted combination of the contents at all positions, with data-dependent weights." For an input signal x (an image, a sequence, a video, or their feature maps) and an output y of the same size, the response at output position i is

  y_i = (1/C(x)) Σ_{∀j} f(x_i, x_j) g(x_j).

Here i indexes the output position whose response I'm computing, j ranges over *all* positions (this ∀j is the non-locality — nothing is excluded by distance), f(x_i, x_j) is a scalar saying how related positions i and j are, g(x_j) is a representation of the content at j, and C(x) normalizes. The output at i is the affinity-weighted sum of contents everywhere.

Before I pick specific f and g, let me check this is genuinely different from the operators I already have, because if it collapses to one of them I've gained nothing. The thing to compare against is the sum's *support* and where its weights *come from*. Convolution and recurrence restrict the sum to a local set of j (i−1..i+1, or i−1..i); this sums over all j, so it isn't either of those. The harder case is the fully-connected layer, because it also touches all positions — so I should pin down exactly how this differs from an fc layer rather than assume it does. An fc layer computes y_i = Σ_j W_{ij} x_j, where W_{ij} is a learned *parameter*: the weight relating position j to position i is fixed once trained, independent of the data, and the matrix W is sized to a fixed N (and the row i has no built-in tie to input position i — it's just whatever the optimizer put there). My weight is f(x_i, x_j)/C(x), which is a *function of the current input* — two different inputs get two different mixing patterns over the same shared parameters W_θ, W_φ, W_g; the operation accepts any N because there is no per-pair parameter; and output i is, by construction, the response computed *at* input position i, so the i↔i correspondence is preserved. So the differences from an fc layer are concrete (data-dependent weights, variable N, position-preserving), not cosmetic — and crucially it can live in the *middle* of a network on feature maps, where I want the long-range mixing, rather than only at the end like an fc layer.

Now I need concrete choices for f and g. Take g first, it's easy: a linear embedding g(x_j) = W_g x_j, learnable, implemented as a 1×1 convolution (in space) or 1×1×1 convolution (in spacetime) — the cheapest possible learned per-position content representation.

For f — the affinity — let me reason through options, starting from what non-local means itself used and seeing where I can improve. Non-local means and bilateral filters use a Gaussian of feature similarity, so the natural first choice is

  f(x_i, x_j) = e^{x_iᵀ x_j},

a Gaussian in the dot-product similarity x_iᵀ x_j. (Classical non-local means used Euclidean distance in the exponent; dot product is equivalent up to normalization and is much friendlier to implement, since it's just a matrix multiply on a deep-learning platform.) With this f, the natural normalizer is C(x) = Σ_{∀j} f(x_i, x_j), so the weights over j sum to one — a proper weighted average.

But comparing raw features x_i, x_j directly is rigid; the features I'm handed weren't necessarily produced to make this similarity meaningful. So let me compute the similarity in a *learned* embedding space instead — embed x_i and x_j through two separate linear maps before comparing:

  f(x_i, x_j) = e^{θ(x_i)ᵀ φ(x_j)},   θ(x_i) = W_θ x_i,   φ(x_j) = W_φ x_j,

again with C(x) = Σ_j f. Now the network can *learn* what "related" means rather than relying on the raw feature geometry. I claimed this is strictly more flexible than the plain Gaussian — that I recover the plain one by setting W_θ = W_φ = I — so let me actually confirm that reduction rather than wave at it. Take x_i = (1, −2, 0.5), x_j = (0.3, 0.7, −1.0). The plain score is x_iᵀ x_j = 0.3 − 1.4 − 0.5 = −1.6, so the plain f = e^{−1.6} ≈ 0.2019. Pushing both vectors through the identity maps leaves them unchanged, so the embedded score is (I x_i)ᵀ(I x_j) = −1.6 and the embedded f = e^{−1.6} ≈ 0.2019 — identical. The embedded form contains the plain one as the W = I corner, good, so adopting it costs nothing and can only add expressivity.

Now let me write out the normalized weight for this embedded-Gaussian version and look hard at its shape, because the form it takes might tell me what this operation really *is*. The weight from j is

  (1/C(x)) f(x_i, x_j) = e^{θ(x_i)ᵀ φ(x_j)} / Σ_{j'} e^{θ(x_i)ᵀ φ(x_{j'})}.

That right-hand side is, term for term, the softmax of the pairwise scores s_{ij} = θ(x_i)ᵀ φ(x_j) taken over j. Let me verify that's not a coincidence of notation by running a number through both. Fix i and give it two keys with scores s = (0.5, 2.0). The non-local route: f = (e^{0.5}, e^{2.0}) = (1.6487, 7.3891), C = 9.0378, weights = (0.18243, 0.81757), and they sum to 1.0. The softmax route on the same s: subtracting the max for stability, (e^{−1.5}, e^{0}) = (0.22313, 1.0), normalized to (0.18243, 0.81757). The two weight vectors agree to ~1e-16. So the normalized embedded-Gaussian weight *is* a softmax over j; that's an identity, not an approximation. Then the whole operation is y = softmax(xᵀ W_θᵀ W_φ x) g(x): softmax of (a query-embedding times a key-embedding) times a value-embedding. Written that way it is exactly the self-attention layer from machine translation. So self-attention turns out to be a special case of this non-local operation — the embedded-Gaussian instantiation with C(x) = Σ_j f — applied to a 1-D sequence, and the classical vision filter and the sequence-attention layer are the same idea seen from two sides. What I'm doing is generalizing it from 1-D language sequences to space and spacetime feature maps for images and video.

That unification immediately makes me suspicious of the softmax. If self-attention is just *one* instantiation, I shouldn't assume its softmax — the "attentional" part — is where the power comes from; it might be the non-locality (the ∀j with data-dependent weights) doing the real work, with softmax along for the ride. The honest way to find out is to build versions *without* softmax and compare them empirically, so let me make sure I can write down sensible non-softmax instantiations at all.

First, drop the exponential entirely and use the raw dot product as the affinity:

  f(x_i, x_j) = θ(x_i)ᵀ φ(x_j),

linear, no softmax. Here Σ_j f is no longer guaranteed positive and doesn't make a clean averaging normalizer, so I set C(x) = N, the number of positions. Why N specifically? Two reasons: it keeps the output scale roughly invariant to how many positions there are (which I need, since input size varies), and dividing by a constant N rather than by a data-dependent Σ_j f simplifies the gradient (the normalizer no longer depends on the features, so it doesn't contribute its own term to the backward pass). The only real difference from the embedded-Gaussian version is the absence of softmax, which had been acting like an activation/normalization on the affinities.

Second non-softmax option, borrowing the pairwise form from relation networks: concatenate the two embeddings and project to a scalar through a small nonlinearity,

  f(x_i, x_j) = ReLU(w_fᵀ [θ(x_i), φ(x_j)]),

again normalized by C(x) = N. Now the affinity is a learned function of the concatenation rather than a similarity.

These give me four instantiations — Gaussian, embedded Gaussian (= self-attention), dot product, concatenation. I want to be careful about what I'm entitled to conclude here. I've shown the four are *writable* and that two of them dispense with softmax entirely; what I have *not* shown is that they perform comparably — that's the experiment I'd run (swap f, hold everything else fixed, read off Kinetics accuracy), and I can't run it on this page. My hypothesis is that they land close together, because if they do, then the softmax/attention framing isn't the essence and the non-locality is; if instead the embedded-Gaussian clearly wins, the attentional normalization is carrying weight after all and I'd have to revise the whole "non-locality is the point" story. So I'll build the block to make f a swappable mode and treat "which f matters little" as a claim to be tested, not one I've established.

Now I have an operation, but I want a reusable *block* I can stick into existing, pretrained networks without retraining from scratch. The risk is concrete: insert a randomly initialized new operation into a trained ResNet and you derail it — the features downstream suddenly see garbage in place of the activations they were trained on. The residual-learning trick is the obvious lever. Wrap the non-local output in a residual block:

  z_i = W_z y_i + x_i,

where y_i is the non-local response, W_z projects y back to the channel count of x (a position-wise 1×1×1 embedding), and "+ x_i" is a residual connection. The move I want is to make the *whole branch* vanish at initialization so z_i = x_i, leaving the pretrained behavior untouched, and let training grow the non-local correction from zero. The cleanest way to force the branch to start at zero is to put a batch-norm after W_z and zero-initialize its scale (γ). I want to actually trace that this gives an exact identity and not just approximately, including in both BN modes, because "approximately identity" would still nudge a pretrained net. BN computes γ·(u − μ)/√(σ²+ε) + β. With γ = 0 the multiplicative term is 0 regardless of u, μ, σ, so the output is just β; and β is zero-initialized too, so the branch outputs exactly 0 and z = 0 + x = x. I ran the actual tail (a 1×1×1 conv into a BN3d with γ zeroed, feeding random x and random y): in eval mode max|z − x| came out 0.0, and in train mode — where BN subtracts the batch mean and would behave differently — it was also exactly 0.0, since γ = 0 kills the scaled term before the running-stats-vs-batch-stats distinction can matter. So the block is a true identity at init in both modes, and I can drop it anywhere in a pretrained network without disturbing it. (Equivalently I could zero-initialize W_z itself; the zeroed-BN-scale version is what I'll use.)

The pairwise computation, when I write it as tensors, is just matrix multiplications: form the affinity matrix by multiplying the θ-embedded positions against the φ-embedded positions (an N×N matrix of all pairwise scores), apply softmax-along-rows (or scale by 1/N for the non-softmax versions), and multiply that against the g-embedded positions to get the weighted sums. Two matmuls and a softmax.

But N×N is the worry — N is the number of positions, and the affinity matrix is N². Let me get concrete about the cost before deciding it's acceptable, using a res3-sized map: say T = 4 frames at 28×28, so N = 4·28·28 = 3136 positions, and channel width C = 512. The two matmuls (θᵀφ to make the N×N scores, then weights·g) each cost N·N·d multiply-adds where d is the embedding width, so 2·N²·d. At full width d = 512 that's 2·3136²·512 ≈ 1.0×10¹⁰ MACs — comparable to a single 3×3 conv on the same map, but not negligible, and I'd like to shave it. Two levers. First, a bottleneck, exactly as in residual networks: set the embeddings W_g, W_θ, W_φ to output *half* the channels of x, d = C/2. Both matmuls scale linearly in d, so halving d halves the block — the cost ratio is exactly 2.0 by the formula, which I confirmed by plugging in (full 2N²·512 over bottleneck 2N²·256 = 2.0). W_z then maps the half-width result back up to full width. Second, a subsampling trick on the key/value side: I don't actually need to attend to every single position to be "non-local" — I can attend to a subsampled set. Replace x by a pooled version x̂ on the key/value side,

  y_i = (1/C(x̂)) Σ_{∀j} f(x_i, x̂_j) g(x̂_j),

by adding a max-pooling layer after φ and g. A 2×2 spatial pool takes Nj from 4·28·28 = 3136 down to 4·14·14 = 784, exactly a quarter, and since the matmul cost is N_i·N_j·d the per-block cost drops by that same factor of 4 (I checked: bottleneck-only over bottleneck-plus-subsample = 4.0, so the two tricks together cut the original by 8×). The output is still computed for every i, and it still gathers from across the whole map — just from a sparser set of representative positions — so the non-local behavior is preserved; the computation is merely sparser.

The other thing that controls cost is *where* I put the block, and this interacts with the N² scaling in a way that also decides where it's useful. The pairwise cost is quadratic in N, so the block is cheap on the high-level, already-subsampled feature maps deep in the network, where the spatial (and temporal) resolution is small — there N is small enough that the N² matmul stays in the neighborhood of one ordinary convolution, as the res3 estimate above showed. So I add non-local blocks in the mid-to-late stages of a ResNet (the res3 / res4 stages), where the maps are e.g. a few frames by 14×14 or 7×7. There's a tension at the very deepest stage though: go too deep (res5, a 7×7 map) and N becomes so small that there just aren't many positions for non-local gathering to relate — at, say, 4·7·7 ≈ 200 positions the "sum over everything" is summing over very little, and I'd expect long-range structure to be thin there. So I'd bet the sweet spot is the intermediate high-level stages rather than the very last one, though which stage actually helps most is again something to read off the experiments. And when I want several blocks (say 5 or 10), I spread them across these stages, inserting one every other residual block, so each sits on a reasonably-sized map.

For video specifically, I need a backbone to host these blocks. Start simple: a 2D ResNet applied frame by frame (a C2D model — all kernels act within a frame as 1×k×k, and the temporal dimension is handled only by pooling), initialized directly from an ImageNet-pretrained ResNet. This isolates what the non-local blocks add, because the backbone itself does almost nothing temporal. As a stronger, more standard point of comparison I can also inflate the 2D kernels into 3D: a 2D k×k kernel becomes a 3D t×k×k kernel spanning t frames, initialized from the pretrained 2D weights by copying the k×k kernel into each of the t temporal planes and rescaling by 1/t. Let me check that the 1/t is the right scale and not just a plausible-looking constant: feed a *static* clip, the same frame repeated t times, into the inflated kernel. The 3D conv at a pixel sums over the t temporal taps, each tap being (2D kernel /t) convolved with the identical frame, so the result is t·(2D kernel /t)·frame = (2D kernel)·frame — exactly the 2D model's output on that frame. So 1/t is precisely the factor that makes a repeated-frame clip reproduce the pretrained 2D result; any other constant would rescale the activations and break the clean initialization. That's an I3D backbone. The non-local blocks slot into either the C2D or the I3D backbone. And conceptually they're *complementary* to 3D convolution rather than redundant: 3D conv captures *local* spacetime structure, non-local blocks capture *long-range* spacetime structure, so using both should beat either alone — and a non-local block is far cheaper than adding 3D convolutions everywhere, since one block on a subsampled map costs about as much as a single conv layer while connecting the whole spacetime volume.

A couple of training notes for the video setting, where the model is large and the data is clips. Fine-tune from the ImageNet-pretrained backbone with batch-norm *enabled* — the usual practice when fine-tuning ResNets is to freeze BN, but here, on this large video model, I expect leaving BN active to act as a regularizer that reduces overfitting, so I'll keep it on and watch the val curve. Initialize the new weight layers in the non-local blocks with the standard scheme for rectifier networks, and put the single zero-initialized BN on the W_z output so the block starts as the identity I traced above. Dropout after the global pooling layer. And nothing temporally fancy in the affinity itself: positions are positions, whether they're separated in space, in time, or in both — the same ∀j sum spans the entire spacetime volume, which is exactly how it relates the ball in the first frame to the ball in the last.

Let me write it, grounded in how the non-local block actually gets built — the three embeddings, the affinity matmul with the choice of normalization, the weighted-sum matmul, and the identity-initialized residual wrapper, with the bottleneck and subsampling.

```python
import torch
from torch import nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    """z = W_z y + x, with y_i = (1/C(x)) sum_j f(x_i, x_j) g(x_j) over ALL positions.
    Identity at init (BN scale on W_z = 0), so it drops into a pretrained net unchanged."""
    def __init__(self, channels, mode="embedded_gaussian", subsample=True):
        super().__init__()
        self.mode = mode
        inter = channels // 2                       # bottleneck: half channels (cuts compute ~2x)
        self.g     = nn.Conv3d(channels, inter, 1)  # g(x_j) content embedding
        self.theta = nn.Conv3d(channels, inter, 1)  # theta(x_i)
        self.phi   = nn.Conv3d(channels, inter, 1)  # phi(x_j)
        if mode == "concatenation":
            self.w_f = nn.Conv2d(2 * inter, 1, 1)
        self.W_z = nn.Conv3d(inter, channels, 1)    # project back to C channels
        self.bn  = nn.BatchNorm3d(channels)
        nn.init.zeros_(self.bn.weight)              # zero-init BN scale -> block starts as identity
        # subsample keys/values: max-pool after phi and g -> ~1/4 the pairwise cost
        self.pool = nn.MaxPool3d((1, 2, 2)) if subsample else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        g_x = self.pool(self.g(x)).flatten(2)            # (B, inter, Nj)
        theta = self.theta(x).flatten(2)                 # (B, inter, Ni)
        phi   = self.pool(self.phi(x)).flatten(2)        # (B, inter, Nj)
        Ni, Nj = theta.size(2), phi.size(2)

        if self.mode in ("gaussian", "embedded_gaussian"):
            # f = exp(theta^T phi); normalized weight = softmax over j  (= self-attention)
            scores = torch.bmm(theta.transpose(1, 2), phi)        # (B, Ni, Nj)
            weights = scores.softmax(dim=-1)                      # (1/C) f with C = sum_j f
        elif self.mode == "dot_product":
            # f = theta^T phi, NO softmax; C(x) = N -> divide by Nj (eases gradient, size-invariant)
            scores = torch.bmm(theta.transpose(1, 2), phi)
            weights = scores / Nj
        elif self.mode == "concatenation":
            ti = theta[:, :, :, None].expand(-1, -1, Ni, Nj)
            pj = phi[:, :, None, :].expand(-1, -1, Ni, Nj)
            f = F.relu(self.w_f(torch.cat([ti, pj], dim=1))).squeeze(1)   # (B, Ni, Nj)
            weights = f / Nj                                      # C(x) = N

        y = torch.bmm(weights, g_x.transpose(1, 2))              # (B, Ni, inter): weighted sum
        y = y.transpose(1, 2).view(B, -1, T, H, W)
        z = self.bn(self.W_z(y)) + x                             # residual; identity at init
        return z


class NonLocalResNetVideo(nn.Module):
    """ImageNet-pretrained ResNet as a per-frame C2D (or inflated I3D) video backbone,
    with non-local blocks inserted in the mid/high stages (res3, res4)."""
    def __init__(self, depth=50, num_classes=400, n_blocks=5, inflate=False):
        super().__init__()
        self.stem, self.res2, self.res3, self.res4, self.res5 = build_resnet_video(depth, inflate)
        c3, c4 = 512, 1024
        # spread blocks over res3/res4, one every other residual block; res5 (7x7) is too small
        self.nl3 = nn.ModuleList([NonLocalBlock(c3) for _ in range(n_blocks // 2)])
        self.nl4 = nn.ModuleList([NonLocalBlock(c4) for _ in range(n_blocks - n_blocks // 2)])
        self.head = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, clip):                                     # clip: (B, 3, T, 224, 224)
        x = self.res2(self.stem(clip))
        x = interleave(self.res3, self.nl3, x)                   # block after every other res unit
        x = interleave(self.res4, self.nl4, x)
        x = self.res5(x)
        x = x.mean(dim=(2, 3, 4))                                # global spacetime average pool
        return self.head(self.dropout(x))


def inflate_2d_to_3d(kernel_2d, t):
    """t x k x k 3D kernel from a pretrained k x k 2D kernel: copy into each temporal
    plane and rescale by 1/t, so a static repeated clip reproduces the 2D model."""
    return kernel_2d[:, :, None].repeat(1, 1, t, 1, 1) / t
```

The causal chain: long-range dependencies matter, but convolution and recurrence are local and can only reach distant positions by deep stacking, which is inefficient, hard to optimize, and bad at multi-hop interactions; so, generalizing the classical non-local-means filter, I define an operation whose response at each position is a data-weighted sum over *all* positions, y_i = (1/C(x)) Σ_j f(x_i,x_j) g(x_j), which is unlike conv/recurrence (non-local) and unlike fc (data-dependent weights, variable size, position-preserving, checked term by term); I instantiate the affinity f as Gaussian, embedded Gaussian, dot product, or concatenation — finding by a direct numeric check that the embedded-Gaussian-with-softmax case is exactly self-attention, hence a special case, and leaving "which f matters little / non-locality is the source of the gain" as a hypothesis the swap experiment must settle; I wrap it as a residual block z = W_z y + x with the W_z output BN scale zero-initialized, traced to give an exact identity (max|z−x| = 0 in both BN modes) so it can be inserted into any pretrained network; I cut its N² cost with a channel bottleneck (×2) and key/value subsampling (×4), computed on a res3-sized map, and place it on the mid/high stages where the maps are small; and I host the blocks in an ImageNet-pretrained C2D or 1/t-inflated I3D video backbone (the 1/t checked to reproduce the 2D model on a static clip), where they capture long-range spacetime structure complementary to (and cheaper than) local 3D convolution.
