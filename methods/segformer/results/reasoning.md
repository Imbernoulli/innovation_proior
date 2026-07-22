OK, let me think this through from scratch. I want a semantic segmentation model — a label for every pixel — that's accurate *and* efficient *and* doesn't fall apart when I run it at a resolution different from the one I trained on. That last property keeps getting ignored and it's the one that bites in practice: I train on 512² crops and then deploy on a 2-megapixel street scene, and the accuracy I measured on the benchmark evaporates. So I'm going to keep "robust to test-time resolution" as a first-class constraint the whole way through, not an afterthought.

Start from the tools I have. There's a pure-Transformer image classifier sitting right there: cut the image into 16×16 patches, linearly embed each patch into a token, prepend a class token, add a learned absolute positional embedding, run a stack of standard pre-norm encoder blocks with global self-attention, read off the class token. It classifies beautifully. Can I just bolt a segmentation decoder onto it? Let me try, in my head, and watch where it breaks — that's where the design has to happen.

First break: it gives me *one* feature map, at stride 16, all the way through. But segmentation lives or dies on two things at once — I need fine spatial detail to get boundaries right, and I need coarse, large-receptive-field semantics to know *what* a region is. A convnet hands me both for free because it's organized in stages: stride 4, 8, 16, 32, fine maps early and coarse maps late, and the decoder fuses them. With a single stride-16 map I have neither the fine detail (stride 16 is already too coarse for crisp boundaries) nor an explicit coarse map; I'd have to manufacture scales the features were never computed at. So one thing I'll need: the encoder should natively emit a *hierarchy* — feature maps at strides 4, 8, 16, 32 — like a convnet, so the decoder gets real multi-scale features.

Second break, and this one looks more serious: cost. Global self-attention compares every token to every other token. Let me write down what that costs because the scaling decides everything. A feature map of `N = H·W` tokens, channel dim `C`. The score matrix `QKᵀ` is `(N×C)·(C×N) = N²·C` multiply-adds; weighting the values `(scores)·V` is again `N²·C`. So attention is Θ(N²) in the number of tokens. For a 224² classification image at stride 16 that's 14×14 = 196 tokens, fine. But segmentation wants a *fine* stride and runs on *large* images. Let me actually put numbers on a 512² crop, which is the standard ADE20K crop. At stride 4 that's a 128×128 map, so `N = 16384`, and `N²` = 268,435,456 — a quarter-billion score entries, in the *first* stage, for *one* image. At stride 8 it's 64×64 = 4096 tokens, `N²` ≈ 16.8M; stride 16 gives 1024 tokens, `N²` ≈ 1.05M; stride 32 gives 256 tokens, `N²` ≈ 65K. So the cost is wildly front-loaded: the finest stage alone is ~250× the cost of the coarsest, and on a Cityscapes 1024×2048 image the stride-4 map would have 512×256 ≈ 131K tokens and `N²` in the seventeen-billion range. Global attention at the fine strides dense prediction needs is simply not affordable. Whatever I do, the attention cost at the early, high-resolution stages cannot be allowed to scale like `N²`.

Third break, the resolution one I promised to take seriously. The classifier injects position via a learned absolute positional embedding — a fixed-length table, one vector per token of the *training* grid, added to the tokens. Attention is permutation-invariant, so without *some* position signal the model can't tell that token `i` sits above token `j`; I do need position somehow. But this particular *form* of position is exactly the thing that breaks under resolution change. If I train with a 14×14 grid and test on a 32×32 grid, the table is the wrong length — I have to interpolate it to the new grid, and interpolating a learned embedding throws away precisely the alignment it learned. The reported pattern is consistent with that: the more the test resolution departs from training, the more accuracy falls. For a task whose entire job is to run at varied, large sizes, a fixed-length positional embedding is a liability, not a convenience. So the position signal has to come from somewhere with *no fixed length to interpolate*.

Let me take these one at a time, because they're somewhat separable.

The hierarchy first, because it's the easiest. Don't keep the resolution fixed — shrink it stage by stage, the convnet way. Stem to stride 4, then each subsequent stage halves the spatial size and grows the channel count. Four stages give me strides 4, 8, 16, 32 and a pyramid of feature maps `F₁…F₄`. Each stage starts with a patch-embedding step that downsamples a token map to a coarser one, then runs some Transformer blocks at that resolution. That gives me the pyramid in principle — now the cost of attention at the *fine* early stages (stride 4 has the most tokens, the 268M I just computed) is the thing I have to fix.

So, the quadratic. The `N²` came from "every token attends to every other token." The obvious move from the windowed-Transformer line is to make attention *local* — restrict each query to a window of nearby tokens. That does cut the cost, but it changes the whole bookkeeping (window partitioning, then shifting windows so information crosses window borders, then un-shifting), and worse, it caps the receptive field of a single attention layer at the window size, which fights the "I want global context cheaply" goal I started from. Let me see if there's a less invasive lever before I take on all that machinery.

Here's the asymmetry I want to exploit. In attention, the *query* length is what sets the output resolution — I have `N` queries because I want an answer at every one of the `N` spatial locations, and I can't shrink that without losing resolution. But the *keys and values* are only there to be summarized into context for each query; nothing says I need a key/value for every single token. What if I keep all `N` queries but reduce the keys and values to `N/R` of them, for some reduction ratio `R`?

How do I reduce them in a way that respects the 2-D structure? The tokens live on an `H×W` grid. Take the key/value token map, reshape it back to spatial, and pool each `R'×R'` block (where `R = R'²`) into a single token — concretely, reshape the `N×C` sequence to `(N/R)×(C·R)` by grouping `R` neighboring tokens, then a linear layer `C·R → C` maps each group down to one token. Now I have `N/R` key/value tokens. Re-derive the cost: the score matrix is `(N×C)·(C×(N/R)) = N²·C/R`, and the value aggregation is `(N×(N/R))·((N/R)×C) = N²·C/R`. So the quadratic cost is divided by `R`.

Let me check this actually buys what I need, by putting `R` against the per-stage `N²` I computed. I want the relief to be largest exactly where `N` is largest. Try `R = 64, 16, 4, 1` across the four stages and recompute the divided cost for the 512² crop:

- stage 1: `N²` = 268,435,456, `R=64` → 4,194,304
- stage 2: `N²` = 16,777,216, `R=16` → 1,048,576
- stage 3: `N²` = 1,048,576, `R=4` → 262,144
- stage 4: `N²` = 65,536, `R=1` → 65,536

That's the shape I wanted: stage 1 drops by 64× — from a quarter-billion down to ~4.2M — and after the reduction the four stages are all within a factor of ~64 of each other instead of ~4000×, so no single stage dominates the bill anymore. The last stage keeps `R=1`, i.e. exact full attention, which is free because `N` is already only 256 there. And note what I did *not* touch: the queries stayed at full resolution `N` the whole time, so the output is computed at every spatial location — I never gave up output precision to get the speedup. That's strictly cheaper than going local-window and it keeps a global view (every query still sees a summary of the *whole* map, not just a window), so I'll take this over windowing.

One honest worry before I move on: reducing the keys/values is a *summarization* — I'm throwing away resolution on the context side. Does that cost me the fine detail I went to all this trouble to preserve? Walking through it: the fine detail enters through the *query* stream, which is full-resolution at every stage, and the late stages run exact attention, so detail is never bottlenecked through the reduced keys. The keys carry *context*, and context is exactly the thing that's fine to summarize — neighboring tokens in a 64-token block are largely redundant for "what region am I in." So the trade lands on the right side: full-resolution queries, summarized context. I'm willing to make it.

Now the hard one — position without a fixed-length embedding. Let me reconsider where I even need it. The standard story is: tokens go in, you add the positional embedding once at the input, attention is now position-aware. But I just argued that input embedding is the fragile part. Is there a *different* place in the network that already carries position, that I can exploit instead?

Stare at what a convolution does on a token map. A 3×3 conv slides over the `H×W` grid. In the interior it's translation-equivariant and carries no absolute-position information. But at the *borders* the conv kernel hangs off the edge of the map and the padding (zeros) fills in — and the *amount* of padding a location sees depends on how close it is to the border. So a token near the edge of the map should get a systematically different convolution response than one in the middle, purely because of the zero-padding pattern.

Let me actually check that this leak is real and not just a story I'm telling myself, because the whole position scheme rests on it. The sharpest test: feed a *spatially constant* input — a map that is identical at every location, so it carries literally zero position information — through a zero-padded depthwise 3×3 conv, and see whether the output is still constant (conv is position-blind) or varies by location (conv has manufactured position out of the padding). On a 6×6 uniform input with a random fixed kernel, the output channel comes back as:

```
[[ 0.196 -0.056 -0.056 -0.056 -0.056 -0.116]
 [ 0.100 -0.154 -0.154 -0.154 -0.154  0.060]
 [ 0.100 -0.154 -0.154 -0.154 -0.154  0.060]
 [ 0.100 -0.154 -0.154 -0.154 -0.154  0.060]
 [ 0.100 -0.154 -0.154 -0.154 -0.154  0.060]
 [-0.135 -0.382 -0.382 -0.382 -0.382 -0.197]]
```

The entire interior is the same value, −0.154 — exactly as expected, the conv is translation-equivariant where it sees no border. But every border cell differs: the top-left corner is +0.196, a top edge cell is −0.056, the interior is −0.154, and the four corners, four edges, and interior all take distinct values. The spread from this position-free input is 0.578. So the conv genuinely reads off "how close am I to which edge" and writes it into the feature map. That's a usable absolute-position signal pulled out of nothing but the padding pattern. Good — the mechanism is real, and crucially a convolution has *no fixed length*: it's the same 3×3 kernel whether the map is 14×14 or 128×128, so there's nothing to interpolate when I change resolution. That's the resolution-robustness property I needed, supplied without any positional embedding.

Where do I put the convolution? The cleanest place is inside the feed-forward sublayer, which already runs per token. The standard FFN is `Linear → activation → Linear`. I'll slip a 3×3 depthwise convolution between the two linears, operating on the tokens reshaped to their 2-D grid. Depthwise (one filter per channel, `groups = C`) so it's cheap — it's there to mix spatial neighbors and leak position, not to do heavy channel mixing, which the linears already do. So the sublayer becomes

  x_out = MLP( GELU( Conv₃ₓ₃( MLP(x_in) ) ) ) + x_in,

a "mixed" FFN — part MLP, part conv. The conv gives me locality *and* the implicit position signal, the linears give me the channel mixing, and there is no positional embedding anywhere in the model. Now when I test at a resolution I never trained on, nothing has to be interpolated; the conv just runs over a bigger grid the same way.

Let me revisit the patch-embedding/downsampling step, because there's a subtle failure I want to avoid. The classifier cuts the image into *non-overlapping* 16×16 patches: a clean grid, each pixel belongs to exactly one patch. For classification that's fine. But for segmentation I care intensely about what happens *at patch boundaries* — two pixels straddling a patch border are physically adjacent in the image, yet a non-overlapping embedding puts them in different tokens with no shared support, so the boundary information between them is severed at every downsampling step. That hurts exactly where segmentation is most fragile: object edges. So I'll make the patch embedding *overlap*. Implement it as a strided convolution where the kernel is larger than the stride: stem uses kernel 7, stride 4, padding 3 (so output stride 4, but each output token's receptive field overlaps its neighbors'); the later downsampling stages use kernel 3, stride 2, padding 1. Overlapping patches preserve local continuity across token borders — the boundary detail survives the downsampling. A single strided conv does the whole "merge neighboring patches and downsample" in one operation.

Let me pin the small architectural knobs. Heads scale with width across stages: `1, 2, 5, 8` for the four stages, so head dimension stays in a sensible range as channels grow `32, 64, 160, 256` in the smallest variant and `64, 128, 320, 512` in the larger ones. The spatial reduction ratios are `8, 4, 2, 1`, which means the key/value sequence reductions are `R = sr² = 64, 16, 4, 1` — the exact reductions I checked against the per-stage costs above. The FFN expansion is wider in the early narrow stages for the table configurations — `8×, 8×, 4×, 4×` through B0-B4 — while the largest variant uses `4×` everywhere. Pre-norm LayerNorm before each sublayer, residual after each, GELU — all kept from the standard block; I have no reason to disturb what already trains. Stochastic depth (drop-path) on the residual branches regularizes the bigger variants. The attention scaling is the usual `1/√d` so logits don't saturate the softmax. Depths grow with model size, and the smallest stays `2, 2, 2, 2`.

Now the decoder, and here I want to resist the field's instinct to make it heavy. DeepLab bolts on ASPP, PSPNet a pyramid pooling module, SETR a progressive-upsampling stack — all because their *convolutional or single-scale* encoders have a small effective receptive field, so the decoder has to do the long-range-context work. But stare at my encoder for a second: the attention is global-ish from the very first stage (every query attends to a summary of the whole map), and the late stages have an enormous effective receptive field. The context-gathering job the heavy decoders exist to do — my encoder has largely *already done it*. So I'll hypothesize the decoder doesn't need to enlarge receptive field at all. It only needs to take the four feature maps, which are at different strides and different channel counts, and fuse them into one per-pixel map. That could be almost trivially simple, pure MLP.

Four steps. (1) Each stage feature `Fᵢ` has its own channel count `Cᵢ`; run a linear layer to map them all to a common channel `C`: `F̂ᵢ = Linear(Cᵢ, C)(Fᵢ)`. (2) They're at strides 4, 8, 16, 32 — bilinearly upsample each to the finest, stride 4: `F̂ᵢ = Upsample(¼)(F̂ᵢ)`. (3) Concatenate the four along channels and fuse with one linear: `F = Linear(4C, C)(Concat(F̂₁, F̂₂, F̂₃, F̂₄))`. (4) A final linear to class logits: `M = Linear(C, N_cls)(F)`, then upsample to input size. No convolutions in the decoder beyond what the linears amount to (1×1), no dilation, no pooling pyramid — just unify-channels, upsample, concatenate, fuse, classify. The bet is that a Transformer encoder with a large receptive field lets the decoder be a fuser, not a context-builder; a convnet encoder could not get away with a decoder this thin.

Before I trust the whole thing, let me build it and actually run it, because three of my claims are checkable right now without any training: that the encoder really emits the stride-4/8/16/32 pyramid, that the decoder really is lightweight (the bet I just made), and that the model really runs at a resolution it never saw with nothing to interpolate.

Wire up the B0 config — `embed_dims = (32,64,160,256)`, `depths = (2,2,2,2)`, the overlap patch embeds, efficient attention with `sr = (8,4,2,1)`, Mix-FFN, the all-MLP head — and push a `512×512` batch through. The encoder returns four maps with spatial sizes `(128,128), (64,64), (32,32), (16,16)` and channels `32, 64, 160, 256`. Dividing 512 by each spatial size gives strides `4, 8, 16, 32` — the pyramid I designed, exactly, and the channels grow as the resolution shrinks. The full model returns `(2, 150, 512, 512)`: one class score per pixel at full input resolution. So the pipeline is internally consistent.

Now the lightweight-decoder bet. Count parameters: the encoder has 3,409,760 and the all-MLP decoder has 433,302 — so the decoder is 11.3% of the model. That's a real number behind the word "lightweight": a tenth of the parameters live in the head, and that's *with* the decoder doing all four stages' channel-unification at `C=256`. The heavy-decoder baselines put a far larger fraction of their compute in the head precisely because the encoder didn't gather the context; here the head is an order of magnitude smaller and I'm betting the encoder's receptive field carries the slack. (I can't *measure* the accuracy of that bet without training, but the structural claim — head is cheap — is now verified, not asserted.)

The resolution claim is the one I built the whole position scheme around, so I should actually exercise it rather than reason about it. Take the *same* trained-config model and feed it `768×1024`, a size and aspect ratio it was never built for, with no code change. It returns `(1, 150, 768, 1024)` — a clean per-pixel map at the new size. Nothing in the forward pass had to be reshaped or interpolated *internally*: there's no positional table whose length depends on the grid, so the only thing that "changes" with resolution is that the convs and attention run over more tokens, which they do natively. This is the payoff of dropping the positional embedding: the model is resolution-agnostic by construction, and I just watched it produce a valid output at an unseen resolution.

So, against the three breaks I started from: the single stride-16 map is replaced by a real stride-4/8/16/32 pyramid (verified by the emitted feature sizes); the `N²` attention at the fine stages is divided by `R = 64,16,4,1`, which I checked drops stage 1 from 268M to 4.2M and flattens the per-stage cost profile; and the fragile positional embedding is gone, replaced by a conv whose padding I confirmed leaks position on a constant input and which has no length to interpolate — verified by running the model at an unseen 768×1024. The decoder collapsed to an 11%-of-the-model all-MLP fuser on the bet that the encoder already did the long-range work. Let me write it out.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class OverlapPatchEmbed(nn.Module):
    # Overlapping patch merge + downsample, as a single strided conv with kernel > stride.
    # Stem: k=7,s=4,p=3 (-> stride 4). Later stages: k=3,s=2,p=1 (halve). Overlap preserves
    # local continuity across token borders -- the boundary detail that segmentation needs.
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                       # B, C, H, W
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # B, N, C
        x = self.norm(x)
        return x, H, W


class EfficientAttention(nn.Module):
    # Full-resolution queries; keys/values are spatially reduced by sr_ratio, so the
    # sequence reduction is R = sr_ratio^2 and attention cost is N^2 C / R.
    def __init__(self, dim, num_heads, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5     # 1/sqrt(d) so logits don't saturate softmax
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # reduce K,V token count by R = sr_ratio^2 via a strided conv (the "Reshape+Linear")
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x).reshape(B, N, h, C // h).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)     # back to spatial
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)   # B, N/R, C
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, h, C // h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale     # B,h,N,N/R
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DWConv(nn.Module):
    # 3x3 depthwise conv inside the FFN: mixes spatial neighbors and leaks absolute position
    # through its zero-padded borders -- replaces any fixed-length positional embedding.
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    # The branch computes MLP(GELU(Conv3x3(MLP(x)))); the Block adds the residual.
    # Position comes from the conv, not from an embedding.
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(self.dwconv(x, H, W))
        return self.fc2(x)


class Block(nn.Module):
    # Pre-norm: residual around efficient-attention, residual around Mix-FFN.
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


class MixVisionTransformer(nn.Module):
    # Four stages; each = OverlapPatchEmbed (downsample) + `depth` Blocks. Emit all 4 maps.
    def __init__(self, in_chans=3, embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(8, 8, 4, 4), sr_ratios=(8, 4, 2, 1), depths=(2, 2, 2, 2),
                 drop_path_rate=0.1):
        super().__init__()
        patch = [(7, 4), (3, 2), (3, 2), (3, 2)]
        dims_in = [in_chans, *embed_dims[:-1]]
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            pe = OverlapPatchEmbed(patch[i][0], patch[i][1], dims_in[i], embed_dims[i])
            blocks = nn.ModuleList([
                Block(embed_dims[i], num_heads[i], mlp_ratios[i], sr_ratios[i], dpr[cur + j])
                for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            self.stages.append(nn.ModuleList([pe, blocks, norm]))
            cur += depths[i]

    def forward(self, x):
        B = x.shape[0]
        feats = []
        for pe, blocks, norm in self.stages:
            x, H, W = pe(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   # B,C,H,W
            feats.append(x)
        return feats                       # strides 4, 8, 16, 32


class AllMLPDecoder(nn.Module):
    # unify channels -> upsample to 1/4 -> concat -> fuse -> classify. Lightweight because the
    # encoder's large receptive field already did the long-range context work.
    def __init__(self, in_channels=(32, 64, 160, 256), embed_dim=256, num_classes=150):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(c, embed_dim) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * embed_dim, embed_dim, 1, bias=False),   # the Linear(4C, C)
            nn.BatchNorm2d(embed_dim), nn.ReLU())
        self.dropout = nn.Dropout2d(0.1)
        self.pred = nn.Conv2d(embed_dim, num_classes, 1)          # the Linear(C, N_cls)

    def forward(self, features):
        size = features[0].shape[2:]            # stride-4 spatial size
        outs = []
        for f, lin in zip(features, self.linears):
            B, C, H, W = f.shape
            f = lin(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, -1, H, W)
            f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            outs.append(f)
        f = self.fuse(torch.cat(outs[::-1], dim=1))               # coarse to fine: c4,c3,c2,c1
        return self.pred(self.dropout(f))


class SegFormer(nn.Module):
    def __init__(self, num_classes=150, embed_dims=(32, 64, 160, 256), decoder_dim=256):
        super().__init__()
        self.encoder = MixVisionTransformer(embed_dims=embed_dims)
        self.decoder = AllMLPDecoder(embed_dims, decoder_dim, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.decoder(feats)         # stride-4 logits
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
```
