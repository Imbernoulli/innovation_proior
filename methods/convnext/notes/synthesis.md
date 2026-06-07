# ConvNeXt synthesis

## Verified arXiv id
2201.03545 "A ConvNet for the 2020s" (Liu, Mao, Wu, Feichtenhofer, Darrell, Xie; FAIR/Berkeley, CVPR 2022).
Canonical code: github.com/facebookresearch/ConvNeXt → models/convnext.py (fetched to code/).

## Pain point / research question
By 2021, hierarchical vision Transformers (Swin) overtook ConvNets as generic vision backbones. The performance gap was attributed to self-attention's "intrinsic superiority." But Swin reintroduced ConvNet priors (local windows, hierarchy). Question: is the gap really attention, or is it a bundle of confounds — modern training recipe + macro/micro architecture choices — that a pure ConvNet could also adopt? Test the limits of a pure ConvNet by gradually "modernizing" a standard ResNet-50 toward Swin-T, controlling FLOPs (~4.5G; ResNet-50/Swin-T regime), measuring each step on ImageNet-1K.

## The modernization roadmap (ResNet-50 baseline 78.8% → ConvNeXt-T 82.0%)
All numbers from Table tab:modernizing-t (3-seed means). Baseline torchvision ResNet-50 = 76.1%.

1. Training recipe (the big confound first): 300 epochs (was 90), AdamW, Mixup, Cutmix, RandAugment, Random Erasing, Stochastic Depth, Label Smoothing → 76.1 → 78.8 (+2.7). This is the fixed recipe held constant thereafter.
2. Macro — stage compute ratio: blocks (3,4,6,3) → (3,3,9,3), matching Swin-T's 1:1:3:1. → 78.8 → 79.4.
3. Macro — patchify stem: ResNet 7×7 stride-2 conv + 3×3 maxpool (4× downsample) → 4×4 stride-4 non-overlapping conv. → 79.4 → 79.5.
4. ResNeXt-ify: depthwise conv (groups = channels) for the 3×3, "use more groups, expand width" → widen 64 → 96 (Swin-T width). Depthwise alone drops acc to 78.3 at 2.4G; widening → 80.5 at 5.3G. Rationale: depthwise = spatial-only mixing, 1×1 = channel-only mixing; mirrors self-attention (spatial weighted sum, per-channel) + MLP separation.
5. Inverted bottleneck: MLP hidden dim 4× input (MobileNetV2 idea, expansion 4 here vs 6 in MNv2). dims per block: 96→384→96. FLOPs 5.3 → 4.6G (saves on downsample shortcut 1×1). → 80.5 → 80.6.
6. Move up depthwise conv: put dw conv first (like MSA before MLP), so the expensive spatial op acts on fewer channels. Temporary drop 80.6 → 79.9 at 4.1G.
7. Large kernel: dw kernel 3→5→7→9→11; saturates at 7. 79.9 (3×3) → 80.6 (7×7). Use 7×7 (matches Swin window ≥7).
8. Micro — ReLU → GELU: acc unchanged 80.6.
9. Micro — fewer activations: keep only ONE GELU between the two 1×1 layers (Transformer MLP has one). → 80.6 → 81.3 (matches Swin-T 81.3).
10. Micro — fewer norms: keep only ONE norm, before the first 1×1. → 81.3 → 81.4. Adding a norm at block start does not help.
11. Micro — BN → LN: LayerNorm replaces BatchNorm. → 81.4 → 81.5.
12. Separate downsampling layers: 2×2 stride-2 conv between stages (not in-block stride). Diverges without norm; add LN before each downsample, after stem, after final GAP. → 81.5 → 82.0 (Swin-T 81.3).

Final ConvNeXt-T block (channels_last impl): dwconv 7×7 → LayerNorm → Linear(dim,4dim) → GELU → Linear(4dim,dim) → LayerScale (γ, init 1e-6) → DropPath → residual add.

## Design-decision → why (with rejected alternatives)
- Why depthwise + 1×1 not grouped+dense? Separates spatial vs channel mixing → analogous to attention(spatial)+MLP(channel). Depthwise alone loses capacity → must widen (ResNeXt "expand width").
- Why expansion 4 (not 6 of MobileNetV2)? Matches Transformer MLP ratio 4× exactly; that's the connection being drawn.
- Why move dw conv up? Inverted bottleneck makes the middle wide; putting the spatial (dw, large-kernel) op where channels are FEW keeps it cheap, and the dense 1×1s do the heavy lifting — mirrors MSA-before-MLP ordering.
- Why kernel 7 not 11? Empirically saturates at 7 in both regimes; larger kernels add FLOPs with no gain. 7 also matches Swin local window.
- Why one GELU / one LN? Transformer block is "sparse" in nonlinearities/norms — one activation in MLP, two LN. Removing redundant per-conv activations/norms recovers accuracy (the residual block was over-saturated with them). Empirically a leading extra BN/LN doesn't help.
- Why LN over BN? BN has batch-dependence intricacies (small-batch, train/test mismatch, EMA interactions). LN is per-sample, plays well with the rest. Direct LN-for-BN in vanilla ResNet hurts (Wu & Johnson 2021), but with all other modernizations it's fine and slightly better.
- Why separate downsampling + extra norms? 2×2 stride-2 conv between stages (Swin-style patch merging analog). Bare downsampling diverges; LN "wherever resolution changes" stabilizes (norm at boundaries controls activation statistics through sharp resolution/channel jumps).
- Why LayerScale (γ init 1e-6)? From CaiT (Touvron et al. 2021): per-channel learnable diagonal on the residual branch, init near 0 so each block starts ≈ identity, easing optimization of deep stacks. Per-channel (not scalar) gives per-feature freedom.
- Why patchify stem 4×4 stride-4? Non-overlapping, ViT-style; ResNet stem's overlapping 7×7+maxpool is replaceable with near-identical accuracy at lower complexity; stride 4 keeps Swin's 4× initial downsample for the 4-stage hierarchy.
- Why channels_last + Linear for 1×1? 1×1 conv ≡ Linear over channel dim; permuting to (N,H,W,C) and using nn.Linear is slightly faster in PyTorch.
- DropPath rates linearly spaced 0→drop_path_rate across all blocks (stochastic depth). trunc_normal_(std .02) init; head scaled by head_init_scale.

## Ancestors (load-bearing)
- ResNet (He 2016): residual block F(x)+x; bottleneck 1×1-3×3-1×1; stages (3,4,6,3); BN; 7×7 stem.
- ResNeXt (Xie 2017): grouped conv in 3×3, cardinality; "more groups, wider"; better FLOPs/acc.
- MobileNet (Howard 2017) / Xception: depthwise separable conv (dw spatial + pw 1×1 channel).
- MobileNetV2 (Sandler 2018): inverted residual + linear bottleneck, expansion 6, narrow→wide→narrow with skip on narrow.
- ViT (Dosovitskiy 2021): patchify, no conv prior, MSA + MLP(4×) + LN, GELU; quadratic global attention.
- Swin (Liu 2021): hierarchical, local window attention (≥7), shifted windows, patch merging downsample, stage ratio 1:1:3:1 (or 1:1:9:1), separate downsampling.
- CaiT (Touvron 2021): LayerScale.
- GELU (Hendrycks & Gimpel 2016): x·Φ(x), smooth ReLU.
- LayerNorm (Ba 2016); BatchNorm (Ioffe & Szegedy 2015).

## Scaffold ↔ final code correspondence
Pre-method scaffold: a generic staged ConvNet backbone — a residual block stub (`# TODO: block internals`), a stem stub, downsample stubs, stage builder, head. Final code fills the block with dwconv→LN→Linear→GELU→Linear→LayerScale→DropPath→add, the patchify stem, separate 2×2 downsamplers with LN, custom channels_first/last LayerNorm.
