Let me start from what actually goes wrong, because the failure is specific and I can see it in the samples. My generator is a stack of convolutions — a low-resolution feature map gets carried up through residual conv blocks and upsampling until it becomes an image — and it gets the *local* statistics right. Texture is convincing, color is convincing, edges look like edges. What it gets wrong is *global geometry*: I draw an animal whose fur is photorealistic everywhere but whose legs don't agree with each other, or a background that is locally plausible but doesn't join up across the image, or a face that is fine on the left and fine on the right but where the two sides aren't the same face. The thing being botched is always a relationship between two regions that are far apart in space. So the question is sharp: why is my network good at the local and bad at the long-range, and what would fix exactly that.

The answer is baked into convolution. A 3×3 kernel mixes a 3×3 neighborhood and nothing else. One conv layer can only relate a position to its immediate neighbors; for two positions that are, say, thirty pixels apart to influence each other at all, their receptive fields have to overlap, and that only happens after the signal has passed through enough conv (and downsampling) layers for the fields to grow that wide. So a long-range dependency in my network is never expressed in one place — it is smeared across a long chain of local operations, each of which has to be tuned in concert with the others for the global relationship to come out right. That's three separate problems stacked on top of each other. Representationally, a shallow part of the net simply *can't* reach across the image, so the capacity to express the dependency might not be there at all. Optimization-wise, even if the capacity is there, getting many layers to cooperate to carry one coordinate-far-from-coordinate signal is a hard credit-assignment problem. And statistically, a dependency that only exists as a fragile conspiracy of many layers tends to break on inputs unlike the training set. Local statistics don't have this problem because they live inside a single kernel; that's exactly why the local part works and the global part doesn't.

So what's the obvious brute-force patch, and why does it fail. I could make the kernels bigger — a 7×7 or 11×11 kernel reaches further per layer. But this is the wrong trade twice over. It buys reach at the price of the very thing that makes convolution efficient: a k×k kernel costs k² parameters and k² multiply-adds per position per channel pair, and most of that is spent on a fixed grid of nearby positions I didn't need to relate. And it still imposes a *fixed-shape* neighborhood — a square window — when the thing I actually want to relate is "this leg" to "that leg," two regions whose relative position is content-dependent and not a fixed offset. Bigger kernels give me a bigger fixed window; I want a *content-addressed* one. Wall. Enlarging the receptive field by force isn't the move.

Let me think about what I actually want as an operation, ignoring how to implement it for a second. I want: the response at output location `i` should be allowed to depend on the features at *every* location `j`, in a single step, with the *amount* it depends on `j` decided by how relevant `j` is to `i` — not by how close `j` is to `i`. That's a weighted sum over all positions, weighted by a learned, content-dependent relevance. And I've seen this shape before. Classical non-local means in image denoising does exactly this with a fixed similarity: the denoised value at a pixel is a weighted average of all other pixels, weighted by patch similarity, so a repeated texture far away can vote on how to clean up a pixel here. The template is right there — *response at a location equals a similarity-weighted sum over all locations* — it's just that in non-local means the similarity is hand-defined and the operation isn't learned. I want the learned version.

There's a learned version sitting in the sequence-modeling world, and it's worth being precise about it rather than name-dropping it. Wang, Girshick, Gupta and He took the non-local-means template and made it a generic, learnable building block for deep nets: define the non-local operation as

  y_i = (1 / C(x)) · Σ_j  f(x_i, x_j) · g(x_j),

where `f(x_i, x_j)` is a pairwise affinity between positions `i` and `j`, `g(x_j)` is a per-position embedding of the feature at `j`, and `C(x)` is a normalizer. The output at `i` is the affinity-weighted, normalized sum of the embedded features at *all* `j`. This is precisely the operation I just said I wanted, in learnable form: `f` is the learned relevance, `g` carries the content. They give a couple of choices for `f`. One is the embedded-Gaussian, `f(x_i, x_j) = exp(θ(x_i)ᵀ φ(x_j))` with `C(x) = Σ_j f(x_i, x_j)` — and notice what that combination *is*: dividing `exp(θᵀφ)` by `Σ_j exp(θᵀφ)` is a **softmax over `j`** of the learned similarities `θ(x_i)ᵀφ(x_j)`. So `y_i = Σ_j softmax_j(θ(x_i)ᵀφ(x_j)) · g(x_j)`. Another choice is the plain dot product `f = θ(x_i)ᵀφ(x_j)` with `C(x) = N`, the number of positions — a normalized linear average instead of a softmax average. And the same construction is what the sequence people call self-attention: query `θ`, key `φ`, value `g`, the response a softmax-weighted average of values. It's the same object viewed from two fields.

I want the softmax version, and I should be clear *why* softmax and not the plain `1/N` linear normalization. The output `y_i` is supposed to be "the features at `i` should be replaced/augmented by a re-weighted blend of features from the most relevant locations." If I want that blend to be a genuine selection — concentrate on a few relevant positions and ignore the rest — I need nonnegative weights that sum to one, i.e. a convex combination, so the output stays in the same regime as the inputs and "attend strongly to two locations, ignore the other thousand" is expressible. The `1/N` linear version can give negative or unbounded weights and dilutes everything by the count of positions; the softmax gives me exactly the nonnegative, normalized, sharply-or-softly-peaked weighting I want, and it's differentiable so the relevance is learned end to end. Softmax it is.

Now make this concrete on a feature map. My feature map is `x ∈ ℝ^{C×N}` — `C` channels, `N = H·W` spatial locations flattened. I need three learned linear maps of the per-position feature vectors: a query map, a key map, a value map. The key realization is that a *per-position linear map of the channels* is exactly a **1×1 convolution**: a 1×1 conv applies the same linear transform to the channel vector at every spatial location, with no spatial mixing. That's perfect, because I want the spatial mixing to be done by the attention, not by the projections. So: `q_i = W_q x_i`, `k_j = W_k x_j`, `v_j = W_v x_j`, each a 1×1 conv or the equivalent per-token linear layer after flattening space. Then the affinity logit from output/query position `i` to source/key position `j` is `l_{ij} = q_iᵀ k_j / √d`, the attention weight is

  a_{ij} = exp(l_{ij}) / Σ_{j'=1}^{N} exp(l_{ij'}),

the softmax over all keys `j`, so for a fixed output position `i` the weights are nonnegative and sum to one. The attended feature at `i` is

  o_i = Σ_{j=1}^{N} a_{ij} · v_j.

So `o` is, for every output location, a content-weighted average over all input locations of the value embeddings. Because the weights form a simplex, each `o_i` is a convex combination of value vectors before the output projection. One layer, every-to-every, content-addressed. That's the operation I couldn't get from bigger kernels.

Before I bolt this onto the network I have to deal with cost, because this is where attention on images bites in a way it doesn't on short sentences. The affinity is computed between every pair of the `N` positions, so the attention matrix `A = [a_{ij}]` is `N × N`, and `N = H·W`. At a 32×32 feature map `N = 1024` and `N² ≈ 10⁶` entries — per head, per attention layer — and computing it is an `N²·d` matmul. That's not free, but it's manageable at small maps; at a large map it explodes quadratically. This single fact, `O(N²)` in the number of spatial locations, is going to govern every decision about *where* I put this thing, so I file it away as the binding constraint.

It also tells me to economize inside the block. Do the query and key embeddings need the full channel width? Their only job is to compute a scalar similarity `q_iᵀ k_j` between two positions — a comparison, not a representation that has to carry content forward. A low-dimensional embedding is enough to compare positions, and the affinity matmul cost scales with that embedding width. So I bottleneck the query/key channels aggressively to `C/8`. The value path has a different job: it carries the content that will actually be averaged into the output. I keep that path wider, at `C/2`, so the attended message still has enough capacity while remaining cheaper than full width. A final 1×1 conv projects the averaged value back to the full `C` channels so the block's output matches its input width:

  o_i = W_o( Σ_j a_{ij} v_j ),   W_o ∈ ℝ^{C×(C/2)}.

There's a second cost lever I get from the same `O(N²)` observation. The keys and values are summed over, so I don't strictly need one key/value per spatial location — I can spatially subsample the key and value maps with a max-pool by 2 before the affinity, so the attention matrix becomes `N × (N/4)`, a quarter of the pairwise work, while every query still gets to attend over the whole subsampled map. The queries stay at full resolution so every output location still gets its own attended feature. This doesn't change the non-local behavior — it's still all-to-many positions — it just thins the keys and values, which is exactly the kind of trick that makes attention affordable at a feature map that would otherwise be too big.

Now there's a subtlety in the softmax I almost skipped, and it's the difference between this working and silently failing. The logit before scaling is a dot product over the embedding dimension `d`: `q_iᵀ k_j = Σ_{c=1}^{d} q_{ic} k_{jc}`. Think about its scale. If the components of `q_i` and `k_j` are roughly independent with mean 0 and variance 1, then each product has mean 0 and variance 1, and the sum has mean 0 and **variance `d`** — its typical magnitude grows like `√d`. Push logits to magnitude `√d` and feed them to a softmax and the softmax saturates: it puts almost all the mass on the single largest logit and almost none elsewhere, and in that saturated regime its Jacobian is tiny, so gradients to the query/key projections nearly vanish and the attention can't learn which positions to attend to. That's a real wall — the bigger the embedding dimension, the harder the attention is to train, for a reason that has nothing to do with the data. The fix is to cancel the `√d` growth: divide the logits by `√d` before the softmax,

  l_{ij} = q_iᵀ k_j / √d,

so the logit variance is back to `O(1)` regardless of the embedding width, the softmax stays in a responsive regime, and the gradients survive. This scaling isn't cosmetic; without it the very bottleneck-width choice I just made would interact badly with training. (In the form where the query/key embedding is at the full channel width `C`, the same argument says divide by `√C` — same `d^{-1/2}` rule, with `d` whatever the per-head key dimension actually is.)

I should also decide whether to run one attention or several in parallel. One attention over the whole embedding forces every output to use a single relevance pattern — a single "which other positions matter" map. But the relations I care about are plural: there's "the other leg," "the symmetric half of the background," "the same object's far edge," and a single softmax map has to be a compromise across all of them. If instead I split the embedding into `h` groups, run an independent attention in each lower-dimensional subspace, and concatenate, then different heads can specialize to different relations at once. And it's nearly free: each head is `1/h` the width, so the total cost is comparable to one full-width attention. So multi-head is the natural choice when I have channels to spare — though I'll note that even a single full-width head already captures the all-positions averaging that fixes the geometry problem, and the simplest instantiation (one head over the full `C`, scaled by `C^{-1/2}`) is a perfectly good version of the block. The head count is a knob, not the essence.

Now the most important design decision, and it's not about the attention math at all — it's about how to *insert* this block without wrecking a network that already works. My convolutional generator, before I touch it, produces good local statistics. If I drop a freshly-initialized attention block into the middle of it, the block starts out computing garbage (its projections are random), and that garbage gets injected straight into a working feature stream — at best it slows training while the net learns to undo the damage, at worst it destabilizes the whole thing. I don't want the new module to *replace* anything; I want it to *augment*, and I want it to start by contributing nothing and earn its influence. So I wrap it in a residual and I make the residual start at zero. Concretely, the block's output is

  y_i = γ · o_i + x_i,

where `γ` is a learnable scalar **initialized to 0**. At the start of training `γ = 0`, so `y_i = x_i` exactly — the block is the identity, the convolutional network is completely undisturbed, and adding the block literally cannot hurt. Then, only if the non-local evidence actually helps the loss, gradients push `γ` off zero and the attention's contribution grows. This is the right curriculum: learn the easy local task first with the conv backbone intact, and *gradually* turn on the long-range mechanism as the network discovers it's useful. It's also exactly the trick the non-local block uses for the same reason — there, the output projection `W_z` in `z_i = W_z y_i + x_i` is initialized to zero so the block is identity at init and can be inserted into a pretrained network without breaking it. Same idea, two spellings: a zero-initialized scalar `γ`, or a zero-initialized output projection. Either way the invariant is *the block is identity at initialization*, and that invariant is what makes "just add attention" safe.

Let me reconcile the two spellings, because they correspond to two natural ways to write the safe block. In the GAN-style spelling I keep a separate scalar: `y = γ·o + x`, `γ` a single learnable number starting at 0. In the direct denoising-UNet spelling I can fold the zero-init into the last 1×1 conv (call it the output projection) — initialize its weights to zero, so `o` itself is zero at init and `y = o + x = x`. The second is slightly more expressive (a full zero-init projection rather than one shared scalar) and is exactly the same identity-at-initialization invariant. If I use a library attention block whose output projection is not explicitly zero-initialized, I should not pretend that invariant comes for free; then the residual path is present, but exact identity at init requires the separate zero gate or zero output projection.

One more piece for the denoiser spelling: normalization before the projections. The dot-product logits are only well-scaled if the per-position feature vectors that feed q and k are themselves well-scaled; if some channels run hot, they dominate the dot product regardless of the `√d` correction. So I normalize the feature map before computing query/key/value. I use GroupNorm rather than BatchNorm here, because the generative training I care about runs with small or noise-varying batches where per-batch statistics are unreliable, and GroupNorm normalizes within channel groups per sample, independent of batch size. So the block, in the denoiser spelling, is: GroupNorm the input, project to q/k/v with 1×1 convs or per-token linear layers, attend with the `√d`-scaled softmax, project back, and add the residual with the identity-at-init invariant preserved by the output projection.

So I have a self-contained module that takes a feature map `[B, C, H, W]`, lets every spatial location attend to every other in one step with content-dependent weights, costs `O(N²)` in the number of locations, and is identity at initialization so it can be slotted into the convolutional backbone harmlessly. Now back to the binding constraint — where do I put it. The `O(N²)` cost means I cannot sprinkle it everywhere for free: at the highest-resolution feature map, `N` is largest and `N²` is brutal. So the placement of attention across the resolution levels of the UNet is itself the design space. At one extreme I put attention at *every* resolution level — every down-block and every up-block gets a self-attention block interleaved with its convolutional residual blocks — which is the maximally expressive option: every feature map, coarse and fine, gets full global mixing, at the cost of paying the `N²` price at the large maps too. At the other extreme I attend only at the coarsest, cheapest maps where `N` is tiny and the quadratic cost is negligible, leaving the expensive high-resolution maps purely convolutional. Between those is attending at one or a few middle resolutions. The construction itself is agnostic to the placement; the same block drops into any resolution level. The variant that takes the expressive extreme — self-attention at *every* resolution of the denoiser — is the cleanest statement of "let every feature map, at every scale, see globally."

Let me write the feature-map block first in the bottlenecked generator spelling, because that is the most explicit form of the mechanism: 1×1 projections, `C/8` query/key, `C/2` value, optional key/value subsampling, softmax over keys, output projection, and a zero-initialized residual gate. If I want the older embedded-Gaussian spelling exactly, the scale can be set to `1`; in the scaled-dot-product denoiser spelling I keep `d^{-1/2}` for the variance reason I just worked through.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMapBlock(nn.Module):
    """Bottlenecked all-position attention over a [B, C, H, W] feature map."""

    def __init__(self, channels, pool_keys_values=True, scale_logits=True):
        super().__init__()
        qk_channels = max(1, channels // 8)
        value_channels = max(1, channels // 2)
        self.pool_keys_values = pool_keys_values
        self.scale = qk_channels ** -0.5 if scale_logits else 1.0

        self.to_q = nn.Conv2d(channels, qk_channels, 1)
        self.to_k = nn.Conv2d(channels, qk_channels, 1)
        self.to_v = nn.Conv2d(channels, value_channels, 1)
        self.proj_out = nn.Conv2d(value_channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.to_q(x).flatten(2).transpose(1, 2)   # [B, H*W, C/8]

        k_map = self.to_k(x)
        v_map = self.to_v(x)
        if self.pool_keys_values:
            k_map = F.max_pool2d(k_map, kernel_size=2, stride=2)
            v_map = F.max_pool2d(v_map, kernel_size=2, stride=2)

        k = k_map.flatten(2).transpose(1, 2)          # [B, M, C/8]
        v = v_map.flatten(2).transpose(1, 2)          # [B, M, C/2]

        logits = torch.einsum("bid,bjd->bij", q, k) * self.scale
        weights = logits.softmax(dim=-1)              # softmax over key/value positions j
        out = torch.einsum("bij,bjd->bid", weights, v)
        out = out.transpose(1, 2).reshape(B, v.shape[-1], H, W)
        return x + self.gamma * self.proj_out(out)    # gamma=0 makes this identity at init
```

In the diffusion denoiser spelling, I do not need to reimplement the processor by hand. The attention-bearing UNet blocks already do the processor mechanics: after each convolutional residual block, GroupNorm is applied inside `Attention`, `[B, C, H, W]` is flattened to `N = H·W` tokens, linear projections produce `q/k/v`, the heads compute scaled dot-product weights with a softmax over key positions, the weighted values are projected back, reshaped to the feature map, and added to the residual stream. With the usual `UNet2DModel` default `attention_head_dim=8`, a `C`-channel feature map is split into `C/8` heads of width 8; if I override that number, the same `1/√d` rule follows the chosen per-head width `d`. The full-attention variant is the placement choice: choose the attention-bearing down and up blocks at every resolution level while leaving the training target, sampler, optimizer, timestep embedding, residual blocks, and channel widths fixed.

```python
import os
from diffusers import UNet2DModel


def build_model(device):
    channels = (128, 256, 256, 256)
    if os.environ.get("BLOCK_OUT_CHANNELS"):
        channels = tuple(int(x) for x in os.environ["BLOCK_OUT_CHANNELS"].split(","))
    layers = int(os.environ.get("LAYERS_PER_BLOCK", 2))

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=channels,
        down_block_types=("AttnDownBlock2D",) * len(channels),
        up_block_types=("AttnUpBlock2D",) * len(channels),
        layers_per_block=layers,
        attention_head_dim=8,
        norm_num_groups=32,
        norm_eps=1e-6,
        act_fn="silu",
        time_embedding_type="positional",
        flip_sin_to_cos=False,
        freq_shift=1,
        downsample_padding=0,
    ).to(device)
```

The samples fail on long-range geometry because convolution is local, so a dependency between two distant regions has to be carried through a long chain of local layers — expensive to represent, hard to optimize, brittle — while local statistics, living inside a single kernel, come out fine. Bigger kernels don't fix it: they buy reach by sacrificing convolution's efficiency and still impose a fixed-shape window when I need a content-addressed one. The operation that fixes the missing reach is the non-local / self-attention template — response at a location equals a content-weighted sum over all locations — which couples any two positions in one step. I implement it with per-position projections for query/key/value, a softmax over learned dot-product affinities, and a weighted average of value vectors. The `O(N²)` cost in the number of spatial locations is the binding constraint: it motivates `C/8` query/key embeddings, a wider `C/2` value path, optional key/value subsampling, and it makes the choice of attended resolutions the main architectural knob. The `√d` logit scaling keeps the softmax out of its saturated, gradient-starved regime in the scaled-dot-product form. The zero-initialized residual — either a scalar gate starting at zero or a zero-initialized output projection — makes the block exactly identity at initialization, so it can be inserted into the working conv backbone without disturbing it and can grow its influence only as it proves useful. GroupNorm before the projections keeps the dot products well-scaled in the denoiser form. The maximally global instantiation places attention-bearing blocks at every resolution of the denoising UNet, so coarse and fine feature maps alike are mixed globally, paying the quadratic price at the large maps in exchange for unrationed long-range coordination.
