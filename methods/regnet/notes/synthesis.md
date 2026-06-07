# RegNet synthesis

## Verified arXiv id
2003.13678 "Designing Network Design Spaces" (Radosavovic, Kosaraju, Girshick, He, Dollár; FAIR, CVPR 2020).
Canonical code: github.com/facebookresearch/pycls → pycls/models/{regnet.py, anynet.py, blocks.py} (fetched to code/).

## Pain point / research question
Two paradigms exist for finding good architectures. Manual design (VGG, ResNet, ResNeXt) yields interpretable *design principles* that generalize, but is hard as choices multiply. NAS finds a strong single *instance* in a fixed search space, but the output is one network tuned to one setting — no transferable principle, no understanding of *why*. Goal: a paradigm combining both — design the design *space* itself, progressively simplifying an unconstrained space into a low-dimensional one full of good, simple, interpretable models that generalize across compute regimes. Testbed: network *structure* (widths, depths, groups), standard blocks.

## Method = Radosavovic 2019 design-space measurement tool
Quality of a design space is measured by sampling n models, training them, and looking at the *error distribution*, summarized by the error EDF: F(e) = (1/n) Σ 1[e_i < e] — fraction of sampled models with error < e. Compare design spaces by their EDFs (lower/left-shifted is better; mean error = area-related summary). Low-compute regime: 400MF, 10 epochs, n=500 (training 100 such ≈ one ResNet-50 at 4GF/100ep). Use empirical bootstrap on (statistic, error) pairs to get 95% CI for the best value of a parameter.
Bootstrap: sample with replacement 25% of pairs, take the min-error pair, repeat 1e4 times, 95% CI for that x; median = most likely best.

## General network structure (held fixed)
- Stem: stride-2 3×3 conv, w0=32 output channels.
- Body: 4 stages, progressively reduced resolution; each stage = sequence of identical blocks, first block stride-2.
- Head: global average pool + FC → n classes.
- Block = X block (standard ResNeXt residual bottleneck with group conv): 1×1 conv (to w_b = round(w·b)) → 3×3 group conv (groups = w_b/g) → 1×1 conv (to w); BN+ReLU after each conv; residual add then ReLU; projection 1×1 (stride-2) on shortcut when shape changes.

## The design-space-design trajectory (AnyNetX → RegNet)
AnyNetX_A: per-stage d_i, w_i, b_i, g_i, 4 stages → 16 dof. Sampling: d_i≤16, w_i≤1024 div by 8, b_i∈{1,2,4}, g_i∈{1..32}. ~10^18 configs. (Build on X block.)
- B: share bottleneck ratio b_i = b across stages. EDF virtually unchanged → no loss, simpler (13 dof). Bootstrap: b≤2 best.
- C: also share group width g_i = g. EDF unchanged (10 dof). Find g>1 best.
- D: constraint w_{i+1} ≥ w_i (good nets have increasing widths). EDF improves substantially.
- E: constraint d_{i+1} ≥ d_i (depths also increase, not necessarily last stage). EDF improves. Each of D,E reduces space by 4!.

Quantized linear parameterization (the core insight): plot per-block widths of top-20 AnyNetX_E models; population trend follows a line w_j = 48·(j+1). But individual models have *quantized* (piecewise-constant) widths. So:
1. linear: u_j = w_0 + w_a·j, for 0 ≤ j < d. Params: d, w_0>0, w_a>0.
2. continuous: write u_j = w_0 · w_m^{s_j} → solve s_j = log(u_j/w_0)/log(w_m). w_m>0 controls quantization.
3. quantize: round s_j to ⌊s_j⌉, then w_j = w_0 · w_m^{⌊s_j⌉}.
Per-stage: stage i has width w_i = w_0·w_m^i and depth d_i = #blocks with ⌊s_j⌉ = i.
Fit test: grid search w_0, w_a, w_m to minimize e_fit (mean log-ratio predicted/observed per-block widths). Best models all have low e_fit; e_fit improves C→E. RegNet = the design space of only such linear-structured models: 6 params (d, w_0, w_a, w_m, b, g). Sample d<64, w_0,w_a<256, 1.5≤w_m≤3. ~3×10^8 configs (10 orders smaller than AnyNetX_A). Random search efficiency much higher (~32 models suffices).

## Trends from analyzing RegNetX (100 models, 25 epochs, lr 0.1)
- Depth d of best models stable across flops, ~20 blocks (≈60 layers). (Contradicts "deeper for more flops".)
- Best bottleneck ratio b = 1.0 → removes the bottleneck. (Contradicts standard practice.)
- Width multiplier w_m ≈ 2.5 (not exactly 2 = doubling).
- g, w_a, w_0 increase with flops.
- Activations (output tensor sizes summed) ∝ sqrt(flops) for best models; params ∝ flops linearly; runtime ~ linear + sqrt term. Activations strongly affect runtime on memory-bound accelerators (GPU/TPU).
- Constrained RegNetX: b=1, d≤40, w_m≥2, limit params/activations → faster, smaller, no accuracy loss. Further 12≤d≤28 for SOTA comparison.

## Alternate-design findings (motivating, about existing ideas)
- Inverted bottleneck (b<1, MobileNetV2) degrades EDF slightly; depthwise (g=1) even worse. → For these regular nets, plain bottleneck-free b=1 with groups is better.
- Varying input resolution (EfficientNet's compound scaling) harms RegNetX; fixed 224×224 best even at higher flops.
- Y block = X + Squeeze-and-Excitation (Hu 2018): SE = global avg pool → FC reduce (to w_se = round(w_in·se_r)) → ReLU → FC expand → sigmoid → channel rescale. RegNetY = RegNetX with SE → improves EDF. se_r = 0.25.

## generate_regnet (code-grounded)
ws_cont = arange(d)*w_a + w_0; ks = round(log(ws_cont/w_0)/log(w_m)); ws_all = w_0 * w_m^ks; round to multiple of q=8; ws, ds = unique(ws_all, counts). adjust_block_compatibility makes w·b divisible by g.

## Ancestors (load-bearing)
- VGG (Simonyan 2015): stack 3×3, double width across stages.
- ResNet (He 2016): residual F(x)+x, bottleneck, stages, depth scaling.
- ResNeXt (Xie 2017): group conv, cardinality; the X block.
- MobileNetV2 (Sandler 2018): inverted residual / linear bottleneck (b<1).
- Xception (Chollet 2017): depthwise (g=1).
- EfficientNet (Tan 2019): compound scaling (depth/width/resolution), NAS baseline.
- SE (Hu 2018): squeeze-and-excitation channel attention.
- Radosavovic 2019 ("On Network Design Spaces"): EDF, distribution-level comparison, empirical bootstrap.
- BatchNorm (Ioffe 2015).

## Scaffold ↔ final code correspondence
Pre-method scaffold: a configurable staged-bottleneck-net builder (AnyNet) taking per-stage widths/depths/strides/bottleneck/group lists; stem, stages of residual bottleneck (group conv) blocks with optional SE stub, head. The contribution is the *width/depth generator* (`generate_regnet`) that maps 6 scalars → per-stage lists via the quantized linear function — a free function stub. Final code fills generate_regnet (linear → continuous → round → power → quantize-to-8 → unique) and adjust_block_compatibility, the SE channel-attention body, and wires RegNet to AnyNet.
