# SegFormer

SegFormer is a semantic-segmentation framework that pairs a **positional-encoding-free, hierarchical Transformer encoder** (Mix Transformer, MiT-B0 through MiT-B5) with a **lightweight all-MLP decoder**. It targets accuracy, efficiency, and robustness to the train/test resolution gap at the same time, and it does so without the hand-crafted, compute-heavy context modules (ASPP, PPM, progressive-upsampling decoders) that earlier segmentation systems leaned on.

## Problem

Per-pixel labelling needs both fine spatial detail (for boundaries) and large-receptive-field semantic context (to know *what* a region is). A plain-ViT segmentation backbone fails this on three counts: it emits a single low-resolution (stride-16) feature map instead of a multi-scale pyramid; global self-attention costs O(N²) in the token count N = H·W, which is prohibitive at the fine strides dense prediction needs; and its learned absolute positional embedding is a fixed-length table tied to the training token grid, so at a different test resolution it must be interpolated, which drops accuracy. Segmentation is routinely trained on fixed crops and deployed at arbitrary, larger sizes, so that last point is a real liability.

## Key ideas

1. **Hierarchical feature representation.** The encoder downsamples stage by stage like a CNN, emitting four feature maps F_i at strides {4, 8, 16, 32} with channels C_i. The input stem produces stride-4 tokens, much finer than ViT's stride-16 grid, which favours dense prediction.

2. **Overlapped patch merging.** Each stage begins with a strided convolution whose kernel is larger than its stride (stem K=7, S=4, P=3; later stages K=3, S=2, P=1), so adjacent patches overlap. This preserves local continuity across patch borders — the boundary detail a non-overlapping ViT-style merge would sever at every downsampling step.

3. **Efficient self-attention.** Queries stay at full length N, but the key/value sequence is reduced by a ratio R before attention: reshape K from N×C to (N/R)×(C·R), then a linear maps C·R → C, giving N/R keys/values. This cuts attention from O(N²) to O(N²/R). The reduction is realised as a strided conv with kernel = stride = sr_ratio, so the sequence shrinks by R = sr_ratio². With sr_ratios = [8, 4, 2, 1] from stage 1 to stage 4, the effective sequence reductions are R = [64, 16, 4, 1] — heaviest where N is largest (the fine early stages), down to exact full attention in the last stage.

4. **Mix-FFN.** No positional encoding anywhere. Instead a 3×3 **depthwise** convolution is placed inside the feed-forward network, between its two linears: `x_out = MLP(GELU(Conv3x3(MLP(x_in)))) + x_in`. The conv's zero-padding leaks absolute position information through the borders, so the network gets the position signal a PE used to supply — but a conv has no fixed length to interpolate, so accuracy is robust under a train/test resolution mismatch.

5. **Lightweight all-MLP decoder.** Four steps: (i) a linear maps each F_i to a common channel C; (ii) bilinearly upsample each to stride 4; (iii) concatenate the four and fuse with one linear (4C → C); (iv) a final linear predicts the H/4 × W/4 × N_cls mask, then upsample to input size. This works *because* the hierarchical Transformer encoder already has a large effective receptive field (local-looking responses at early stages, highly non-local responses at stage 4), so the decoder only has to fuse multi-scale features — it does not need to build long-range context the way ASPP/PPM do on top of a small-receptive-field CNN.

## MiT-B0..B5 dimensions

All variants share num_heads N = [1, 2, 5, 8], sr_ratios = [8, 4, 2, 1], patch-embed (K, S, P) = [(7,4,3), (3,2,1), (3,2,1), (3,2,1)], and strides {4, 8, 16, 32}.

| | embed_dims C | depths L | FFN expansion E |
|---|---|---|---|
| B0 | (32, 64, 160, 256) | (2, 2, 2, 2) | (8, 8, 4, 4) |
| B1 | (64, 128, 320, 512) | (2, 2, 2, 2) | (8, 8, 4, 4) |
| B2 | (64, 128, 320, 512) | (3, 3, 6, 3) | (8, 8, 4, 4) |
| B3 | (64, 128, 320, 512) | (3, 4, 18, 3) | (8, 8, 4, 4) |
| B4 | (64, 128, 320, 512) | (3, 8, 27, 3) | (8, 8, 4, 4) |
| B5 | (64, 128, 320, 512) | (3, 6, 40, 3) | (4, 4, 4, 4) |

Channels grow as resolution shrinks, and stage 3 carries most of the depth. Decoder channel C = 256 for B0/B1 and C = 768 for B2-B5. (Depths and FFN expansions above follow the MiT architecture table; the released reference code differs slightly — uniform FFN expansion 4 across stages, and B2 stage-2 depth 4 rather than 3.)

## Runnable PyTorch

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

## Training protocol

Encoder pre-trained on ImageNet-1K; decoder randomly initialised; the full model fine-tuned with AdamW, initial learning rate 6e-5, "poly" LR decay (power 1.0), on ADE20K (150 classes), Cityscapes (19 classes), and COCO-Stuff (172 classes). Augmentation: random resize (ratio 0.5–2.0), random horizontal flip, random crop (512² on ADE20K/COCO-Stuff, 1024² on Cityscapes; 640² for B5 on ADE20K). No OHEM, auxiliary, or class-balance losses. Metric: mean IoU.

## Verification (each detail grounded in the retrieved source)

- **Two modules; 4×4 input patches; strides {1/4, 1/8, 1/16, 1/32}; mask at H/4 × W/4 × N_cls** — `src/sec3_method.tex`, Method intro: "divide it into patches of size 4×4", "multi-level features at {1/4, 1/8, 1/16, 1/32}", "predict the segmentation mask at a H/4 × W/4 × N_cls resolution".
- **MiT-B0..B5; hierarchical F_i at H/2^{i+1} × W/2^{i+1} × C_i with C_{i+1} > C_i** — `src/sec3_method.tex`, Hierarchical Feature Representation.
- **Overlapped patch merging; K/S/P = (7,4,3) and (3,2,1)** — `src/sec3_method.tex`, Overlapped Patch Merging: "K=7, S=4, P=3, and K=3, S=2, P=1". Per-stage values in `src/tables/arch.tex`.
- **Efficient self-attention: Softmax(QKᵀ/√d_head)V, O(N²); reduce K via Reshape(N/R, C·R) then Linear(C·R, C) → O(N²/R); R = [64,16,4,1]** — `src/sec3_method.tex`, Efficient Self-Attention, Eqs. (1)–(2): "we set R to [64, 16, 4, 1] from stage-1 to stage-4". The conv realisation (kernel=stride=sr_ratio, sr_ratios=[8,4,2,1], sequence shrinks by sr_ratio²) is the canonical `mix_transformer.py`, captured in `refs/nvlabs_segformer_impl.md`, consistent with R = sr_ratio².
- **Mix-FFN: 3×3 conv leaks position via zero padding, replaces PE; x_out = MLP(GELU(Conv3x3(MLP(x_in)))) + x_in; depthwise** — `src/sec3_method.tex`, Mix-FFN, Eq. (3): "positional encoding is actually not necessary", "a 3×3 Conv in the feed-forward network", "depth-wise convolutions". The PE-vs-Mix-FFN resolution-robustness experiment is `src/sec4_exper.tex`, Mix-FFN vs. PE.
- **All-MLP decoder four steps, Eq. (4): Linear(C_i,C) → Upsample to 1/4 → Linear(4C,C)∘Concat → Linear(C,N_cls)** — `src/sec3_method.tex`, Lightweight All-MLP Decoder. `linear_fuse` as a 1×1 ConvModule + BN + ReLU and `linear_pred`, plus the c4,c3,c2,c1 concat order, confirmed in `segformer_head.py` (`refs/nvlabs_segformer_impl.md`).
- **"Works because larger ERF": encoder local at early stages, non-local at stage 4; a thin decoder fails on CNN backbones (ERF bounded by stage-4)** — `src/sec3_method.tex`, Effective Receptive Field Analysis.
- **num_heads N=[1,2,5,8]; sr_ratios=[8,4,2,1]; per-variant depths/expansions; decoder C=256 (B0/B1) / 768 (rest)** — `src/tables/arch.tex` and `src/sec4_exper.tex`, Influence of C: "we choose C=256 for our real-time models B0, B1 and C=768 for the rest". embed_dims/depths cross-checked against `refs/nvlabs_segformer_impl.md` (note: the architecture table lists FFN expansion 8 for the two early stages of B0–B4 and the depths in the table above; the released code uses a uniform expansion of 4).
- **Training: mmsegmentation; ImageNet-1K encoder pretrain; AdamW; lr 6e-5; poly power 1.0; crops 512²/1024²/512² (640² for B5 on ADE20K); no OHEM/aux/class-balance; mIoU** — `src/sec4_exper.tex`, Implementation details.
- **Datasets: ADE20K (150 classes, 20210 imgs), Cityscapes (19 classes, 5000 hi-res imgs), COCO-Stuff (172 classes, 164k imgs)** — `src/sec4_exper.tex`, Datasets.
