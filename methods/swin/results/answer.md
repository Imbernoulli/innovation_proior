# Swin Transformer — the method, distilled

## Problem

Build a *single* Transformer backbone that serves every vision task — classification, detection, instance segmentation, semantic segmentation — the way a convolutional network does. Two obstacles block a direct use of a patchify-and-global-attention image Transformer:

1. **No pyramid.** It emits one feature map at a single stride (16), but dense-prediction heads (FPN, U-Net, Mask R-CNN) need a multi-scale pyramid at strides 4/8/16/32.
2. **Quadratic cost.** Global self-attention costs `Ω(MSA) = 4hwC² + 2(hw)²C` for an `h×w` token map — the `(hw)²` term makes it intractable at the resolutions dense prediction requires.

## Key idea

Compute self-attention **locally, inside non-overlapping `M×M` windows**, and build a **hierarchy** by merging patches between stages. Window attention costs

```
Ω(W-MSA) = 4hwC² + 2M²·hwC,
```

which is **linear** in the number of tokens `hw` when `M` is fixed (default `M=7`); it also gives every query in a window the same key set, enabling a single batched matmul (unlike per-pixel sliding windows). Non-overlapping windows alone never exchange information across their borders, so the partition is **shifted by `⌊M/2⌋` in alternating blocks**: block `2k` uses the regular grid (W-MSA), block `2k+1` the shifted grid (SW-MSA), so consecutive blocks connect neighboring windows and the receptive field grows. The shifted grid is made as cheap as the regular one by a **cyclic shift + masking** trick. Geometry is injected with a learned **relative position bias** added to the attention logits.

## Final architecture

- **Patch embed (stem):** split the image into non-overlapping `4×4` patches, linearly embed each to dimension `C` → stride-4 token map.
- **Four stages** producing strides 4/8/16/32 with widths `C, 2C, 4C, 8C`. Each stage is `depth` Swin blocks; between stages a **patch-merging** layer concatenates each `2×2` group of neighbors (→ `4C`), applies LayerNorm, then a linear `4C→2C` (2× spatial downsample, 2× channels).
- **Swin block** (pre-norm Transformer block with windowed attention):
  ```
  ẑ = (S)W-MSA(LN(z⁻)) + z⁻
  z = MLP(LN(ẑ)) + ẑ
  ```
  even blocks use W-MSA (regular windows), odd blocks SW-MSA (windows shifted by `⌊M/2⌋`). MLP is 2-layer, GELU, 4× hidden. Depths are even so each (regular, shifted) pair is complete.
- **Windowed attention:** `Attention(Q,K,V) = softmax(QKᵀ/√d + B) V`, with `Q,K,V ∈ ℝ^{M²×d}`, per-head dim `d=32`, heads doubling per stage. `B` is gathered from a learned table `B̂ ∈ ℝ^{(2M−1)×(2M−1)}` (per head): relative offset `(Δy,Δx) ∈ [−(M−1), M−1]²` → index `(Δy+M−1)(2M−1) + (Δx+M−1)`.
- **Shifted-window batching:** cyclically roll the map top-left by `⌊M/2⌋` so the *regular* partition tiles it (same window count, no padding), then add a `0/−100` logit mask so tokens only attend within their original region; roll back afterward.
- **Head:** final LayerNorm → global average pool over tokens (no class token) → linear classifier.

Variants: Swin-T (`C=96`, depths `{2,2,6,2}`), Swin-S (`{2,2,18,2}`), Swin-B (`C=128`), Swin-L (`C=192`); `M=7`, `d=32`, MLP ratio 4 throughout. Trained with the data-efficient recipe: AdamW, cosine schedule + warm-up, RandAugment/Mixup/CutMix/random-erasing, stochastic depth (drop-path) scaled with model depth, weight decay, gradient clipping.

## Code

Reference-faithful V1 classifier path, omitting only checkpointing and the optional fused CUDA window kernels:

```python
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
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
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Self-attention within an M x M window, with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))      # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                       # 2, Wh*Ww
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # 2, N, N
        rel = rel.permute(1, 2, 0).contiguous()                        # N, N, 2
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))   # N, N

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            slices = (slice(0, -self.window_size),
                      slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for h in slices:
                for w in slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
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

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
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
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
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
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size, patch_size = to_2tuple(img_size), to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
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


class SwinTransformer(nn.Module):
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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
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
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
```
