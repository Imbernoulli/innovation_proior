OK, let me think this through from scratch. I have a problem I keep running into: in vision there's one backbone — a deep convnet, ResNet-style — that you train once on classification and then reuse, untouched, as the feature extractor for detection, instance segmentation, semantic segmentation. In language the Transformer plays that universal role. I want the same thing in vision: a single Transformer backbone that serves *every* recognition task. And I have an existence proof that Transformers can see at all — the patchify-then-global-attention classifier that's already out there. It takes the image, cuts it into 16×16 patches, linearly embeds each patch into a token, prepends a class token, adds a learned absolute position embedding, and runs a stack of ordinary Transformer encoder blocks — `x ← x + MSA(LN(x))`, `x ← x + MLP(LN(x))`, pre-norm, GELU, the works — with *global* self-attention, then reads off the class token. On classification with enough pre-training data it's excellent.

So why can't I just take that and bolt a detector on top? Let me actually try to, in my head, and watch it break.

The first thing that breaks is scale. Think about what a detector needs. Objects come at wildly different sizes in the same image — a far-off pedestrian a dozen pixels tall, a bus filling half the frame. The whole apparatus the field built to handle this — feature pyramid networks, U-Net, the FPN inside Mask R-CNN — assumes the backbone hands it a *stack* of feature maps at several resolutions: stride 4, 8, 16, 32. Small objects get localized on the fine high-resolution map, big objects on the coarse one. A convnet gives you exactly that for free, because it's organized into stages, each stage halving the spatial size and doubling the channels. But the patchify-and-global-attention model gives me *one* feature map, at a single stride of 16, all the way through. There's no pyramid. I can fake it by upsampling/deconvolving that one map into several, and people do, but it underperforms — I'm manufacturing scales that the features were never computed at. So requirement number one is concrete: the backbone must natively emit feature maps at strides 4, 8, 16, 32, the same shape a convnet emits, so it slots into the existing heads with zero surgery.

The second thing that breaks is cost, and this one is fatal, not just inconvenient. Let me write down what global self-attention actually costs, because the scaling is the whole story. Say I have a feature map of `h × w` tokens, channel dimension `C`. Standard multi-head self-attention does four big linear-algebra operations. First, project the tokens to queries, keys, values: each projection is an `(hw) × C` matrix times a `C × C` weight, so `hw·C²` multiply-adds each, three of them, `3hw·C²`. Then the score matrix `QKᵀ`: an `(hw) × C` times a `C × (hw)`, which is `(hw)²·C`. Then weighting the values, `(scores) V`: an `(hw) × (hw)` times an `(hw) × C`, again `(hw)²·C`. Then the output projection, another `hw·C²`. Sum it:

  Ω(MSA) = 4hw·C² + 2(hw)²·C.

Stare at the second term. It's quadratic in `hw`, the number of tokens. For a 224² image at stride 16 that's 14×14 = 196 tokens — fine. But a detector runs at, say, a 1333-pixel long side, and dense prediction wants a *fine* stride, not 16. The token count goes into the thousands and then the `(hw)²` term explodes. Global attention is simply unaffordable at the resolutions dense prediction needs. So requirement number two: complexity must be linear in `hw`, not quadratic.

Now — can I get linearity? The quadratic term came from "every token attends to every other token." The obvious lever is: don't. Restrict attention to a local neighborhood. There's prior work that does exactly this for convnet replacement: put each pixel at the center of a small window and let it attend only inside that window — a sliding window of attention, the attention analogue of a conv kernel. Slightly better accuracy/FLOPs than the conv it replaces, even. So that's promising — but I remember the catch those papers ran into, and it's a sharp one. With a *sliding* window, every query pixel has its *own* window, centered on itself, so every query has a *different* key set. That means I can't gather the work into a single big batched matrix multiply the way a conv shares one kernel across the whole map; the memory-access pattern is scattered. The result is that the real wall-clock latency is far worse than the FLOP count promises — much slower than the convnet it was supposed to beat. So "local attention" is the right instinct, but "*sliding* local attention" is hardware-hostile. The lesson is precise: I want locality *and* a regular, shareable computation pattern.

So what if the windows don't slide? Partition the feature map into *non-overlapping* tiles of `M × M` tokens — a clean grid, like cutting a checkerboard — and run full self-attention *inside each tile independently*. Now every window is the same size, `M²` tokens, and crucially every query in a window shares the same key set (the other tokens of that window). That's a single batched matmul over `(number of windows)` independent attention problems — exactly the hardware-friendly pattern I wanted. Let me recompute the cost. With windows of `M²` tokens, there are `hw/M²` of them. The two attention terms inside one window cost `2(M²)²·C = 2M⁴·C`; across all `hw/M²` windows that's `2M⁴·C · hw/M² = 2M²·hw·C`. The projection terms are per-token and don't care how I group tokens, so they stay `4hw·C²`. Total:

  Ω(W-MSA) = 4hw·C² + 2M²·hw·C.

If I hold `M` fixed — say `M = 7` — the second term is `2·49·hw·C`, a *constant* times `hw`. The quadratic `(hw)²` has become linear `hw`. Let me put numbers on it rather than trust the symbols, because the whole argument rests on this term actually being the dominant one at dense-prediction scale. Take an 800-pixel image at stride 4 — that's `h = w = 200`, `hw = 40000` tokens — and `C = 96`. Global attention's quadratic term is `2·(hw)²·C = 2·40000²·96 ≈ 3.07×10¹¹` multiply-adds. The window term is `2·49·hw·C = 2·49·40000·96 ≈ 3.76×10⁸`. The ratio is about **816×** — the window version does roughly three orders of magnitude less attention work at this size. Including the projection terms (which are identical for both), total global cost ≈ `3.09×10¹¹` vs window ≈ `1.85×10⁹`, still ~167× cheaper. And the gap *widens* with resolution, because the quadratic term grows as `(hw)²` while the window term grows as `hw`: at the smaller `56×56` stage-1 map the total ratio is only ~14×, at `200×200` it's ~167×. So this isn't a constant-factor win that a faster GPU erases — it's an asymptotic one that gets more decisive exactly where dense prediction hurts. Requirement two, met, and with the batched-matmul access pattern that the sliding-window people couldn't get.

Why `M = 7` and not 1 or 50? `M = 1` is no context at all. Cost grows like `M²` per the formula — at `M = 50` the attention term is `(50/7)² ≈ 51×` bigger than at 7, eating much of the savings I just computed — and locality is the point, so I don't want it large; 7 is enough for a meaningful local receptive field and lines up cleanly with the stage resolutions I'm about to build (56, 28, 14, 7). Hold that thought.

And requirement one, the pyramid? Now that tokens live on a 2-D grid and I'm processing them stage by stage, I can downsample between stages the way a convnet does. Start fine: instead of 16×16 patches, use small 4×4 patches at the stem, so stage 1 is at stride 4 — a high-resolution map, which dense prediction wants. Then between stages, merge each `2×2` block of neighboring tokens into one: concatenate their four feature vectors into a `4C` vector, and apply a linear layer to bring `4C` back down to `2C`. That halves the spatial resolution in each dimension (4× fewer tokens) and doubles the channel width — precisely the convnet stage transition. Four stages give strides 4, 8, 16, 32 and widths `C, 2C, 4C, 8C`. The feature maps now have exactly the resolutions ResNet emits, so the existing detection/segmentation heads attach with no changes. (I'll normalize before the reduction — LayerNorm on the `4C` concatenation, then the linear `4C → 2C`.) Requirement one, solved.

Let me sketch where I am. Stem: patchify into 4×4 patches, linear-embed to `C`. Then four stages; each stage is some Transformer blocks whose attention is my non-overlapping window attention, and between stages a patch-merging downsample. Pre-norm blocks, GELU MLP at 4× width, residuals — keep all of that from the standard recipe; the only thing I've changed inside the block is the attention sublayer.

But wait. Stare at the window attention for a second, because something's wrong with it. I partitioned the map into fixed non-overlapping tiles and attend *only* within each tile. The tiles never talk to each other. Token at the left edge of one tile and token at the right edge of the neighbor are physically adjacent in the image, but they can *never* attend to each other, in any layer, because the partition is the same in every block. So the receptive field is frozen at `M` forever — I've built a model that can never integrate information across window boundaries. That kills modeling power; it's basically a stack of independent `M×M` attention problems that never communicate. This is a real wall.

How do convnets avoid this? Their kernels overlap and slide, so information leaks across any boundary in one step, and after a few layers the receptive field is global. But overlapping/sliding is exactly the hardware-hostile pattern I rejected. I want the non-overlapping, batched pattern *and* cross-window flow. Those feel contradictory — a fixed partition can't both be a clean grid and leak across its own seams.

Unless the partition isn't fixed. What if I *move the grid* between consecutive blocks? In block `l`, partition starting at the top-left corner, the regular grid. In block `l+1`, shift the whole partition by half a window — `⌊M/2⌋` tokens — down and to the right, so the new windows straddle the seams of the old ones. A token that was at the right edge of one window in block `l` now sits in the *interior* of a window in block `l+1` that also contains tokens from what used to be the neighboring window. So in two consecutive blocks — one regular partition, one shifted — every token has talked to tokens on both sides of a boundary. Cross-window connection, achieved by *alternating* the partition, with no overlapping windows and no sliding: each individual block is still the clean batched non-overlapping computation. Stack pairs of (regular, shifted) blocks and the effective receptive field should grow just like a convnet's — each pair widens the reach by roughly a window, so after `k` pairs a token can in principle integrate over a `~(k·M)`-wide neighborhood. Half a window is the natural shift — it maximally interleaves: a token sits as far as possible from being on a new boundary, every old boundary lands near the middle of a new window. (A shift of 1 would barely move the seams; a shift of `M` would just re-create the original partition, connecting nothing — half is the value that puts every old boundary deepest inside a new window.) This forces depths to come in even numbers — each block pairs a regular partition with a shifted one, so a stage of depth 2, 6, 18, etc., is a whole number of these pairs.

Now the shifted partition creates an annoying bookkeeping problem, and I have to be careful or I'll throw away the efficiency I just earned. Take an `8×8` map with `M=4`: the regular partition is a tidy `2×2` grid of full `4×4` windows. Shift by 2: the grid no longer aligns with the map edges, so now I get a `3×3` arrangement of windows, and the ones along the top and left edges are *partial* — smaller than `4×4`. Nine windows instead of four, several of them ragged. If I handle this naively by padding every partial window up to `M×M` and masking the padding, I'm now computing `3×3 = 9` windows where I used to compute `2×2 = 4` — that's 2.25× the attention work, and it gets worse the fewer windows a stage has. I just made the cheap thing expensive again. Wall.

Here's the way around it. Instead of padding, *cyclically shift* the whole feature map toward the top-left by `⌊M/2⌋` — `torch.roll` with negative shift on the two spatial axes — so the tokens that fell off the top and left wrap around to the bottom and right. After this roll, the regular `M×M` partition (the cheap one!) again tiles the map perfectly: I'm back to exactly the original number of windows, four not nine. The catch: a window near the bottom/right edge now contains a mix of tokens that are *not* actually neighbors in the image — some are the genuine bottom-right tokens, others are the wrapped-around top/left tokens, glued together only by the roll. If I let them attend freely, I'd be letting the bottom of the image attend to the top, which is nonsense — those regions aren't adjacent.

So I need attention to respect the *original* region each token came from, even though the roll has interleaved them inside one window tensor. That's a masking job. Label every token by which of the original sub-regions it belongs to. For a shift `s = ⌊M/2⌋`, the implementation slices each spatial axis as `[0:-M]`, `[-M:-s]`, and `[-s:]`; on an `8×8` map with `M=4, s=2` that becomes bands of widths `4, 2, 2`, but the first band is generally `H-M`, not another fixed-width window strip. The 2-D cross product gives up to nine region labels. Within a batched window, two tokens should attend to each other only if they carry the *same* region label: form the pairwise table of (label of query) vs (label of key); where the labels match, bias 0; where they differ, bias `−∞` (in practice a large negative like `−100`), added to the logits *before* softmax so the off-region entries exponentiate to zero.

I want to actually run this on the `8×8, M=4, s=2` case and confirm two things — that I really get back four windows (not nine) and that the mask blocks exactly the wrapped pairs. Painting region ids into the `H−M, M−s, s = 4, 2, 2` bands gives the id-image

```
0 0 0 0 1 1 2 2
0 0 0 0 1 1 2 2
0 0 0 0 1 1 2 2
0 0 0 0 1 1 2 2
3 3 3 3 4 4 5 5
3 3 3 3 4 4 5 5
6 6 6 6 7 7 8 8
6 6 6 6 7 7 8 8
```

Partition this with the *regular* `4×4` grid (the cheap one) and I get four windows — `(8/4)·(8/4) = 4`, exactly the regular count, padding bought me nothing extra. Now look at where the ids land per window. The top-left window covers rows 0–3, cols 0–3: all id `0`, one region — its pairwise id-difference table is all zeros, so its mask is fully open, every token attends to every other, as it should (this was the genuine interior of the image, no wrapping). The bottom-right window covers rows 4–7, cols 4–7 and straddles ids `4,5,7,8` — four different regions stitched together by the roll. Computing the fraction of cross-region (masked) pairs per window I get `[0.0, 0.5, 0.5, 0.75]`: the clean interior window is 0% masked, the two edge windows 50%, and the four-corner window 75% masked. So the construction does confine each token to its own original region — the bottom-of-image tokens that got wrapped up next to top-of-image tokens are blocked from attending to them, while genuine neighbors stay connected. After the attention, roll the map back (`torch.roll` with the positive shift) to put every token home. The number of windows stayed at the regular count, so I kept the linear, batched efficiency, and the shifted-window cross-talk costs essentially nothing beyond a roll and an additive mask.

Let me make sure I can actually build that mask mechanically, because it's the fiddly part. I want a per-window `M²×M²` matrix of 0/`−∞`. Trick: make a single-channel image the size of the feature map, paint it with integer region ids using the band structure — slice the height into `[0:−M]`, `[−M:−⌊M/2⌋]`, `[−⌊M/2⌋:]` and the width the same way, and assign a running counter `0..8` to each of the nine `(h-band, w-band)` rectangles. Run the *same* window-partition on this id-image. Now for each window I have a length-`M²` vector of region ids; compute the pairwise difference `ids[:,None] - ids[None,:]`; entries equal to 0 mean same region (keep, bias 0), nonzero means different region (mask, bias `−100`). That's the attention mask, precomputed once per shifted block since it depends only on geometry, not on the data.

There's one more thing inside the window attention I haven't pinned down: position. The plain dot-product `QKᵀ/√d` is permutation-invariant — it has no idea token `i` sits to the upper-left of token `j`. the patchify classifier added a learned *absolute* position embedding to the input tokens. But within a small window what actually matters is the *relative* arrangement — that one token is two to the right and one down from another — and that relation is the same wherever the window sits in the image. Absolute coordinates throw away that translation equivariance, which the local statistics of images reward. Several prior works inject geometry as a *relative* bias added straight to the logits. So let me put a learned bias `B` into the attention as

  Attention(Q, K, V) = softmax(QKᵀ/√d + B) V,

where `B` for a pair of tokens depends only on their relative offset. (I'd expect this to clearly beat absolute embeddings, and to actively *help* the dense tasks where translation equivariance matters most — whereas tacking an absolute embedding on top tends to hurt them.)

Now, how big is `B` and how do I store it? Naively `B` is `M²×M²` — a bias for every ordered pair of in-window positions. But most of those entries should be *tied*, because they share the same relative offset. Let me count the distinct offsets. Index window tokens by 2-D coordinates `(y, x)` with `y, x ∈ {0, …, M−1}`. The relative offset between query `i` and key `j` is `(Δy, Δx) = (y_i − y_j, x_i − x_j)`. Each component ranges over `−(M−1) … (M−1)` — that's `2M−1` distinct values per axis. So there are only `(2M−1) × (2M−1)` distinct relative offsets, far fewer than `M⁴` ordered pairs. I'll parameterize a small table `B̂ ∈ ℝ^{(2M−1)×(2M−1)}` (per head), and the full `B` is just a *gather* from `B̂` indexed by the relative offset of each pair.

I need the index arithmetic to be airtight. The raw `(Δy, Δx)` lives in `[−(M−1), M−1]²`, which I can't index with directly (negative). Shift each component by `+(M−1)` so it lands in `[0, 2M−2]` — `2M−1` slots per axis. Now I have a 2-D index `(Δy + M − 1, Δx + M − 1)` into the `(2M−1)×(2M−1)` table. To flatten it to a single integer I multiply the *row* component by the number of columns, `2M−1`, and add the column component:

  idx = (Δy + M − 1)·(2M − 1) + (Δx + M − 1),

which I claim ranges over `0 … (2M−1)² − 1`, one slot per distinct offset. This is exactly the kind of index arithmetic that's easy to get off-by-one, and a silent collision (two different offsets mapping to the same slot) or an out-of-range index would corrupt the bias without crashing — so let me trace it on the smallest non-trivial window, `M = 2`, and check it by hand. The four tokens have coordinates `(0,0), (0,1), (1,0), (1,1)`. Building the full `4×4` table of flat indices over every (query, key) pair, the formula gives

```
4 3 1 0
5 4 2 1
7 6 4 3
8 7 5 4
```

Read off what this is telling me. The distinct values are exactly `{0,1,2,3,4,5,6,7,8}` — nine of them, which is `(2M−1)² = 3² = 9`, the full table with nothing wasted and nothing colliding. The diagonal is all `4`: query attending to itself has offset `(0,0)`, which maps to `(0+1)·3 + (0+1) = 4` — the center of a `3×3` offset grid, correct. And it's *anti-symmetric* about that center: entry `(i,j)` and entry `(j,i)` sum to `8` (e.g. `3` and `5`, `1` and `7`, `0` and `8`), which is right because swapping query and key negates `(Δy, Δx)` and the index of `−offset` is the mirror of the index of `+offset` through slot `4`. So the arithmetic is sound: every distinct relative offset gets its own slot, the count is `(2M−1)²`, no two offsets collide. (Concretely the implementation does this same computation: stack the per-axis coordinate grids, take pairwise differences to get `Δy, Δx` for every pair, add `M−1` to each, multiply the `Δy` channel by `2M−1`, sum the two channels, and register it as a buffer so the gather is just a lookup at every forward.) Then `B = B̂[idx]`, broadcast across heads (one table per head). Learned, tiny, shared across windows, translation-equivariant. That's the position story.

Let me also settle the small knobs so the block is fully specified. Heads: I keep the per-head dimension fixed at `d = 32` and let the number of heads scale with width, so stages with `C, 2C, 4C, 8C` channels use `C/32, 2C/32, …` heads — head count doubles each stage as the channels double. The scaling inside attention is the usual `1/√d` so the logits don't blow up and saturate the softmax. MLP expansion ratio 4×, GELU, pre-norm LayerNorm before each sublayer, residual after each — all inherited unchanged from the standard block; I had no reason to disturb what already trains well. For regularization on the big variants I'll lean on stochastic depth (drop-path) with a rate that grows with depth, following the data-efficient training recipe that made vision Transformers trainable on ImageNet-1K alone — AdamW, cosine schedule with warm-up, RandAugment, Mixup, CutMix, random erasing — and I'll classify with global average pooling over the final tokens rather than a class token; pooling is just as accurate here and is the natural convnet-style head. One guard for the deepest stage: by stride 32 the map is only `7×7`, which is `≤ M`; when the window would be at least as big as the whole map I just don't partition (one window = the whole map, shift 0), since shifting a single full-map window would do nothing.

Let me now write the whole thing out as code and check that every piece traces back to a step above.

```python
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    # standard pre-norm Transformer FFN, unchanged: Linear -> GELU -> Linear
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x, window_size):
    # (B, H, W, C) -> (num_windows*B, M, M, C): cut the map into the non-overlapping MxM grid
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    # inverse of window_partition: stitch the windows back into (B, H, W, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    # full self-attention INSIDE one window, plus the learned relative position bias
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (M, M)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # the 1/sqrt(d) so logits don't saturate softmax

        # the small bias table B-hat: (2M-1)*(2M-1) distinct offsets, one column per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # precompute the flat offset index idx = (dy+M-1)*(2M-1) + (dx+M-1) for every pair
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))      # 2, M, M
        coords_flatten = torch.flatten(coords, 1)                       # 2, M*M
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # 2, M*M, M*M  (dy, dx)
        rel = rel.permute(1, 2, 0).contiguous()                        # M*M, M*M, 2
        rel[:, :, 0] += self.window_size[0] - 1                        # dy -> [0, 2M-2]
        rel[:, :, 1] += self.window_size[1] - 1                        # dx -> [0, 2M-2]
        rel[:, :, 0] *= 2 * self.window_size[1] - 1                    # row component * (2M-1)
        relative_position_index = rel.sum(-1)                          # flat index, M*M, M*M
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: (num_windows*B, M*M, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))                              # QK^T inside the window

        # gather B from B-hat via the precomputed index, add to logits (softmax(QK^T/sqrt d + B))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # shifted block: add the 0/-100 cross-region mask so wrapped tokens don't mix
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class ShiftedWindowBlock(nn.Module):
    # one Transformer block; even blocks use the regular grid (shift 0), odd blocks the shifted grid
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        # deepest stage guard: if the whole map is <= one window, don't partition / don't shift
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # precompute the cross-region attention mask from the band structure of the roll
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            slices = (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for h in slices:
                for w in slices:
                    img_mask[:, h, w, :] = cnt        # 9 region ids
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # pairwise id diff
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # cyclic shift toward top-left so the regular partition tiles the shifted config
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)      # (S)W-MSA

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        # roll back to undo the cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)                             # residual around attention
        x = x + self.drop_path(self.mlp(self.norm2(x)))             # residual around MLP
        return x


class PatchMerging(nn.Module):
    # inter-stage downsample: concat each 2x2 of neighbors (4C), LN, linear 4C -> 2C
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)       # 2x downsample, 4C
        return self.reduction(self.norm(x))                         # -> 2C


class BasicLayer(nn.Module):
    # one stage: `depth` blocks alternating regular/shifted, then an optional downsample
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            ShiftedWindowBlock(dim, input_resolution, num_heads, window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,  # even regular, odd shifted
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop, attn_drop=attn_drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                      norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim, norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    # stem: split into 4x4 patches and linearly embed -> stride-4 token map
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size, patch_size = to_2tuple(img_size), to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class WindowedPyramidTransformer(nn.Module):
    # stem -> 4 stages (C, 2C, 4C, 8C; strides 4,8,16,32) -> LN -> global-average-pool -> linear head
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        res = self.patch_embed.patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # drop-path grows w/ depth
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(res[0] // 2 ** i, res[1] // 2 ** i),
                depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer,
                downsample=PatchMerging if i < self.num_layers - 1 else None))

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)   # global average pool, no class token
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
```
