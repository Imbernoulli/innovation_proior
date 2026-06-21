I want a semantic-segmentation model — a class label for every pixel — that is accurate, efficient, and, crucially, robust to the gap between the resolution it was trained at and the resolution it runs at in deployment. That last property is the one that keeps getting ignored, and it is the one that bites: I train on $512^2$ crops and then deploy on a two-megapixel street scene, and the benchmark accuracy evaporates. So I treat "robust to test-time resolution" as a first-class constraint, not an afterthought.

The obvious starting point is a pure-Transformer classifier: cut the image into non-overlapping $16\times16$ patches, linearly embed each into a token, prepend a class token, add a learned absolute positional embedding, run a stack of pre-norm global-attention blocks, read off the class token. It classifies well, but bolting a segmentation decoder onto it breaks in three places, and each break tells me what the design has to fix. First, it emits a *single* feature map at stride 16 all the way through, whereas segmentation needs both fine spatial detail for boundaries and large-receptive-field semantics to know *what* a region is; a convnet hands me both because it is organized in stages (strides 4, 8, 16, 32), and a single stride-16 map has neither. Second, global self-attention compares every token to every other one, so for $N = H\cdot W$ tokens the score matrix $QK^\top$ and the value aggregation each cost $\Theta(N^2 C)$ — at stride 4 on a $512^2$ crop that is $128\times128 = 16384$ tokens and a quarter-billion multiply-adds, and on a $1024\times2048$ Cityscapes image it is hopeless. Third, the learned absolute positional embedding is a fixed-length table tied to the *training* token grid; at a different test resolution it has to be interpolated, and interpolating a learned embedding throws away exactly the alignment it learned, so accuracy falls the further test resolution departs from training. The convolutional encoder–decoders (DeepLab's ASPP, PSPNet's pyramid pooling, U-Net's skips) sidestep the positional-embedding problem but pay for global context with heavy, compute-dominating modules, because a convolution's receptive field grows only slowly with depth. None of the existing options gives me accuracy, efficiency, and resolution robustness at once.

I propose SegFormer: a positional-encoding-free hierarchical Transformer encoder (the Mix Transformer, MiT-B0 through B5) paired with a lightweight all-MLP decoder. Each of the three breaks gets a targeted fix. For the hierarchy, I stop holding resolution fixed and shrink it stage by stage the convnet way — a stem to stride 4, then each later stage halving the spatial size and growing the channel count — so four stages emit feature maps $F_1\ldots F_4$ at strides 4, 8, 16, 32, with the stem producing a stride-4 grid far finer than ViT's stride-16. The downsampling at each stage is done by *overlapped patch merging*: a single strided convolution whose kernel is larger than its stride (stem $K=7, S=4, P=3$; later stages $K=3, S=2, P=1$). Non-overlapping patches would sever two physically adjacent pixels that straddle a patch border into tokens with no shared support, destroying boundary information at every downsampling step — exactly where segmentation is most fragile — so overlap is what preserves local continuity across token borders.

For the cost, I attack the source of the $N^2$ directly. The query length sets the output resolution, so I cannot shrink it without losing spatial precision; but the keys and values exist only to be summarized, and nothing requires one per token. So I keep all $N$ queries at full length and reduce the key/value sequence by a ratio $R$ before attention: reshape the $N\times C$ key/value map to $(N/R)\times(C\cdot R)$ by grouping $R$ neighbouring tokens, then a linear maps $C\cdot R \to C$, giving $N/R$ keys and values. In code this reduction is a strided convolution with kernel = stride = sr_ratio, so the spatial map shrinks by sr_ratio and the sequence by $R = \text{sr\_ratio}^2$. Re-costing the attention, the score matrix becomes $(N\times C)\cdot(C\times N/R) = N^2 C / R$ and the value aggregation likewise $N^2 C / R$, so

$$\text{cost} = \Theta\!\left(\frac{N^2 C}{R}\right), \qquad \text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_\text{head}}}\right)V.$$

With sr_ratios $[8, 4, 2, 1]$ from stage 1 to stage 4 the effective reductions are $R = [64, 16, 4, 1]$ — heaviest at the fine early stages where $N$ is largest, down to exact full attention in the last stage where $N$ is already tiny. The $1/\sqrt{d_\text{head}}$ scale keeps the logits from saturating the softmax. The queries stay full-resolution throughout, so I never trade away output precision; the keys/values are merely a summarization of context, which is the trade I am willing to make.

For resolution robustness I remove the positional embedding entirely and find position somewhere the network already has it. A $3\times3$ convolution on a token map is translation-equivariant in the interior, but at the *borders* the kernel hangs off the edge and the zero-padding fills in, and the amount of padding a location sees depends on how close it is to the border — so a padded convolution leaks absolute position through its borders, and the network can read off "how close am I to an edge." A convolution also has no fixed length: the same $3\times3$ kernel runs over a $14\times14$ or a $128\times128$ grid with nothing to interpolate. So I place a $3\times3$ *depthwise* convolution (one filter per channel, $\text{groups}=C$ — cheap, since it is there to mix spatial neighbours and leak position, not to do channel mixing the linears already handle) inside the feed-forward sublayer, between its two linears. The sublayer, the Mix-FFN, computes

$$x_\text{out} = \mathrm{MLP}\!\big(\mathrm{GELU}(\mathrm{Conv}_{3\times3}(\mathrm{MLP}(x_\text{in})))\big) + x_\text{in},$$

with the conv supplying locality and the implicit position signal and the linears the channel mixing — and no positional embedding anywhere, so at an unseen test resolution nothing is interpolated and the conv simply runs over a larger grid.

The decoder I deliberately keep trivial, resisting the field's instinct to make it heavy. ASPP, PPM, and progressive-upsampling stacks exist because *convolutional or single-scale* encoders have a small effective receptive field, so the decoder must do the long-range-context work. But this encoder's attention is global-ish from the first stage (every query attends to a summary of the whole map) and its late stages have an enormous effective receptive field, so the context-gathering job is already done. The decoder therefore only fuses, in four steps: (i) a linear maps each $F_i$ from its own channel count $C_i$ to a common channel $C$; (ii) bilinearly upsample each to stride 4; (iii) concatenate the four and fuse with a single linear ($4C\to C$); (iv) a final linear predicts the $H/4\times W/4\times N_\text{cls}$ mask, then upsample to input size. A convnet encoder could not get away with a decoder this thin — it works here precisely because the Transformer encoder already carries the long-range context.

The architectural knobs follow from these choices: num_heads $[1,2,5,8]$ so head dimension stays sensible as channels grow, sr_ratios $[8,4,2,1]$, patch-embed $(K,S,P) = [(7,4,3),(3,2,1),(3,2,1),(3,2,1)]$, embed_dims $(32,64,160,256)$ for the smallest B0 and $(64,128,320,512)$ for the larger variants, depths starting at $(2,2,2,2)$ for B0 with stage 3 carrying most of the depth in larger variants, pre-norm LayerNorm with residuals and GELU kept from the standard block, drop-path on the residual branches for the bigger variants, and decoder channel $C=256$ for B0/B1 and $C=768$ for the rest. The causal chain end to end: shrink resolution stage by stage for the stride-4/8/16/32 pyramid; keep full-resolution queries but reduce the key/value sequence by $R=[64,16,4,1]$ to turn $N^2$ attention into $N^2/R$ where it matters most; drop the positional embedding and let a depthwise conv inside each Mix-FFN leak position through its padded borders with nothing to interpolate; overlap the patch embeddings so boundary detail survives; and collapse the decoder to an all-MLP fuser because the encoder's receptive field is already large.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class OverlapPatchEmbed(nn.Module):
    # Overlapping patch merge + downsample: one strided conv with kernel > stride.
    # Stem k=7,s=4,p=3 (-> stride 4); later stages k=3,s=2,p=1 (halve). Overlap keeps
    # local continuity across token borders -- the boundary detail dense prediction needs.
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
    # Full-length queries; K,V are spatially reduced by sr_ratio, so sequence reduction
    # is R = sr_ratio**2 and attention cost is O(N^2 / R).
    def __init__(self, dim, num_heads, sr_ratio=1, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5     # 1/sqrt(d_head); keeps softmax from saturating
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # kernel=stride=sr_ratio -> spatial shrink by sr_ratio, sequence by sr_ratio**2
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x).reshape(B, N, h, C // h).permute(0, 2, 1, 3)   # B,h,N,d
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)               # back to spatial
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)       # B, N/R, C
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, h, C // h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                           # B,h,N/R,d
        attn = (q @ k.transpose(-2, -1)) * self.scale                 # B,h,N,N/R
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)               # B,N,C
        return self.proj_drop(self.proj(x))


class DWConv(nn.Module):
    # 3x3 depthwise conv inside the FFN: mixes spatial neighbours and leaks absolute
    # position through its zero-padded borders -- replaces any fixed-length positional embedding.
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    # Branch computes MLP(GELU(Conv3x3(MLP(x)))); Block adds the residual.
    # Position comes from the conv, no PE.
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Block(nn.Module):
    # Pre-norm: residual around efficient attention, residual around Mix-FFN.
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x


class MixVisionTransformer(nn.Module):
    # Four stages; each = OverlapPatchEmbed (downsample) + L Blocks. Emit all 4 maps.
    # Defaults below are MiT-B0.
    def __init__(self, in_chans=3, embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(8, 8, 4, 4), sr_ratios=(8, 4, 2, 1), depths=(2, 2, 2, 2),
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        patch = [(7, 4), (3, 2), (3, 2), (3, 2)]
        dims_in = [in_chans, *embed_dims[:-1]]
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            pe = OverlapPatchEmbed(patch[i][0], patch[i][1], dims_in[i], embed_dims[i])
            blocks = nn.ModuleList([
                Block(embed_dims[i], num_heads[i], mlp_ratios[i], sr_ratios[i],
                      qkv_bias, drop_rate, attn_drop_rate, dpr[cur + j])
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
        self.fuse = nn.Sequential(                                # the Linear(4C, C)
            nn.Conv2d(4 * embed_dim, embed_dim, 1, bias=False),
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
        f = self.fuse(torch.cat(outs[::-1], dim=1))   # coarse to fine: c4,c3,c2,c1
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


if __name__ == "__main__":
    model = SegFormer(num_classes=150)       # MiT-B0 + all-MLP head
    y = model(torch.randn(2, 3, 512, 512))
    print(y.shape)                           # torch.Size([2, 150, 512, 512])
    # robustness to test-resolution change: no PE to interpolate, conv just runs on a bigger grid
    print(model(torch.randn(1, 3, 768, 1024)).shape)   # torch.Size([1, 150, 768, 1024])
```
