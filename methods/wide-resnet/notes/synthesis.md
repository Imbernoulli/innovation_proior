# Wide ResNet (WRN) — synthesis (grounded in arXiv 1605.07146 source + szagoruyko code)

## Source
- arXiv 1605.07146 (verified). BMVC 2016. LaTeX read in full.
- Canonical code: szagoruyko/wide-residual-networks pytorch/resnet.py — functional. depth=6n+4, widths [16k,32k,64k], pre-activation basic block BN-ReLU-conv-(dropout)-BN-ReLU-conv, projection 1x1 convdim when ni!=no, group downsample stride at first block of group1,group2. conv0 (3->16), group0 stride1, group1 stride2, group2 stride2, final BN-ReLU, avgpool 8, fc.

## Pain point / research question (in-frame, mid-2016)
- Deep residual nets scale to thousands of layers with improving performance. BUT each fraction-of-a-percent of accuracy costs ~doubling the layers → training very deep ResNets has DIMINISHING FEATURE REUSE → very slow to train.
- Study of ResNets so far has focused mainly on (a) the order of activations in a block and (b) depth. Goal: go beyond — explore a richer set of ResNet block architectures, examine how aspects besides activation order affect performance, and ask: how WIDE should deep residual networks be? Can decreasing depth + increasing width be a more effective way to improve performance?

## Diminishing feature reuse (the core diagnostic, from Highway / stochastic depth)
- The identity-mapping residual block that lets you train very deep nets is ALSO a weakness: as gradient flows, nothing forces it to go through the residual block weights — it can avoid learning. So possibly only a few blocks learn useful representations, or many blocks share little info with small contribution. Formulated as "diminishing feature reuse" (Highway, Srivastava et al.).
- Stochastic depth (Huang et al.) addresses it by randomly disabling residual blocks during training — a special case of dropout where each block has an identity scalar weight on which dropout is applied. Its effectiveness PROVES the hypothesis that many blocks are barely used.

## Residual block formula (Sec)
- x_{l+1} = x_l + F(x_l, W_l). (Note: paper's notation, x_{l+1} input, x_l output, but standard: x_{l+1} = x_l + F(x_l).)
- Pre-activation order: BN-ReLU-conv (changed from original conv-BN-ReLU by He et al. identity-mappings paper). Pre-activation trains faster + better → use it; don't consider original.
- Drop bottleneck: bottleneck makes blocks THINNER (used to increase #layers cheaply), opposite of studying widening → focus on BASIC block.

## Three ways to increase representational power of a block
- add more conv layers per block; widen conv layers (more feature planes); increase filter sizes.
- Don't use filters > 3x3 (small filters effective; VGG, Inception-v4). → study only the first two.
- Two factors: deepening factor l = #convs in a block; widening factor k = multiplier on #features. Baseline basic block = l=2, k=1.

## Architecture (Table 1)
- conv1: [3x3, 16] (fixed, output 32x32).
- conv2 (32x32): [3x3, 16k; 3x3, 16k] × N.
- conv3 (16x16): [3x3, 32k; 3x3, 32k] × N. downsample at first block.
- conv4 (8x8): [3x3, 64k; 3x3, 64k] × N. downsample at first block.
- avg-pool 8x8 → fc → softmax.
- Total conv layers n = 6N + 4 (the "depth"). Width by factor k. k=1 ≡ original thin pre-act ResNet.
- Notation WRN-n-k: n total conv layers, k widening factor. e.g. WRN-40-2, WRN-28-10. Append block type e.g. WRN-40-2-B(3,3).

## Block-type study B(M) (Sec, Table 3)
- B(M): M = list of kernel sizes in block. #feature planes constant across block (no bottleneck).
- Variants: B(3,3) original basic; B(3,1,3); B(1,3,1) straightened bottleneck; B(1,3); B(3,1); B(3,1,1) NIN-style.
- Result (k=2, comparable params): B(3,3) best by little margin; B(3,1) and B(3,1,3) very close with fewer params/layers; B(3,1,3) slightly faster. Blocks with comparable #params give ~same results → restrict to 3x3 convs for consistency.

## Deepening factor study l (Sec, Table 4)
- WRN-40-2, 3x3 convs, fixed params 2.2M, vary l∈{1,2,3,4}.
- B(3,3) (l=2) best at 5.43; l=1 B(3): 6.69 (worse); l=3 B(3,3,3): 5.65; l=4 B(3,3,3,3): 5.93 (worst).
- Speculation: deeper blocks (l=3,4) worse due to increased optimization difficulty from FEWER residual connections (fixed total convs + fixed params → more convs per block = fewer blocks = fewer skip connections). → B(3,3), l=2 optimal.

## Width study k (Sec, Table 2/width)
- #params and compute QUADRATIC in k; LINEAR in l and d (number of blocks). But it's more computationally effective to widen than to have thousands of small kernels — GPU is much more efficient on large tensors. Want optimal d-to-k ratio.
- Argument for width: almost all pre-ResNet architectures (Inception, VGG) were much wider than thin ResNets. WRN-22-8 / WRN-16-10 similar in width/depth/params to VGG.
- Experiments k∈[2,12], depth∈[16,40]: all of 40/22/16-layer nets gain when k increased 1→12×. At fixed k=8 or 10, depth 16→28 improves but 28→40 decreases (WRN-40-8 < WRN-22-8).
- WRN-40-4 (8.9M) ≈ ResNet-1001 (10.2M) accuracy on CIFAR but 8× faster → depth-to-width ratio of thin ResNets far from optimal; depth adds no regularization vs width at this level.
- Wide nets train with 2× (and more) params than ResNet-1001 and outperform; would need to double thin net depth (infeasible) to match.

## Dropout in residual blocks (Sec)
- Widening ↑ params → need regularization. BN already regularizes but requires heavy data augmentation (avoid). Add dropout layer INSIDE each residual block, BETWEEN convolutions (and after ReLU), NOT in the identity path. (Prior work put dropout in identity path → negative effects.)
- Rationale: perturbs BN in the next residual block, prevents it from overfitting; helps diminishing feature reuse by enforcing learning in different blocks.
- Dropout p: 0.3 CIFAR, 0.4 SVHN (cross-validated). No extra epochs needed.
- Effect: WRN-28-10 CIFAR-10/100 down 0.11%/0.4%; big on SVHN (no aug → BN overfits → dropout regularizes; WRN-16-8 SVHN 1.81%→1.54%). 16-deep WRN with dropout → 1.64% SVHN.
- Side observation (diagnostic, on existing ResNet training): after first LR drop, loss + val error suddenly rise and oscillate until next LR drop, caused by WEIGHT DECAY; lowering wd hurts accuracy; dropout partially removes this effect.

## ImageNet (Sec)
- Non-bottleneck ResNet-18/34 widened 1.0→3.0× → accuracy rises with width; comparable-param nets of different depth give similar results. But outperformed by bottleneck nets (bottleneck better suited / task needs deeper).
- WRN-50-2-bottleneck (widen inner 3x3 of ResNet-50 by 2×) outperforms ResNet-152, 3× fewer layers, faster; slightly worse than pre-act ResNet-200 but ~2× faster. ImageNet needs more width at same depth vs CIFAR. Unnecessary to go beyond 50 layers (compute).
- For ImageNet <100 layers, pre-activation doesn't help much → use original ResNet there. For CIFAR/SVHN use pre-activation basic blocks.

## Computational efficiency (Sec)
- Thin deep nets with small kernels are against GPU nature (sequential structure). Widening balances computations more optimally → wide nets many times more efficient. WRN-28-10 1.6× faster than ResNet-1001; WRN-40-4 8× faster at same accuracy.

## Implementation details (Sec)
- SGD with Nesterov momentum, cross-entropy loss. Init LR 0.1, weight decay 0.0005, dampening 0, momentum 0.9, minibatch 128.
- CIFAR: LR ×0.2 at epochs 60, 120, 160; total 200 epochs.
- SVHN: init LR 0.01, ×0.1 at 80, 120; total 160 epochs.
- CIFAR data aug: horizontal flips + random crops from 4-pixel reflect-padded image. No heavy aug. Mean/std normalization (or ZCA for some). SVHN: only /255 to [0,1].
- depth must be 6n+4 (n blocks per group).

## Load-bearing ancestors (baselines)
- ResNet (He et al. 2015) — residual y=x+F(x); deep, thin, bottleneck; the base. Diminishing feature reuse.
- Pre-activation ResNet / identity mappings (He et al. 2016, "basicblock2") — BN-ReLU-conv order; trains very deep better; the actual baseline.
- Highway networks (Srivastava et al.) — gated residual links (learned gates); formulated diminishing feature reuse.
- Stochastic depth (Huang et al.) — randomly drop residual blocks in training; proves many blocks barely used.
- Dropout (Srivastava et al. 2014) — random unit zeroing; BN (Ioffe & Szegedy 2015) — reduces internal covariate shift, regularizes; BN shown to beat dropout.
- VGG (Simonyan & Zisserman) / Inception (GoogLeNet) — wide architectures; small 3x3 filters.
- NIN (Lin et al.), DSN, FitNet — comparison methods; B(3,1,1) NIN-style.
- Circuit complexity theory (shallow circuits need exponentially more components than deep) — the pro-depth argument being pushed back on.

## Design-decision → why table
| Decision | Why this, not alternative |
|---|---|
| Increase WIDTH (factor k), decrease depth | very deep thin ResNets suffer diminishing feature reuse + slow training; widening at fixed/comparable params gives same or better accuracy far faster (GPU efficient on big tensors). |
| Basic B(3,3) block, l=2 | block-type study: B(3,3) best (comparable-param blocks ~tie); deepening l=3,4 worse (fewer residual connections → harder optimization); drop bottleneck (it thins, opposite of widening); no filters >3x3 (small filters effective). |
| Pre-activation order BN-ReLU-conv | trains faster + better than original conv-BN-ReLU (He et al. identity mappings). |
| widths 16, 16k, 32k, 64k; conv1 fixed 16 | ResNet template (double width when halving map); k scales the three groups, conv1 fixed. |
| Dropout BETWEEN convs in the block (after ReLU), p=0.3 CIFAR/0.4 SVHN | widening adds params → need regularization; BN alone needs heavy aug; dropout in identity path hurts (prior work); between-conv dropout perturbs next block's BN, fights diminishing feature reuse; biggest effect on SVHN (no aug, BN overfits). |
| SGD + Nesterov, lr 0.1, wd 5e-4, dampening 0, mom 0.9, batch 128 | standard; wd causes a post-LR-drop oscillation but lowering it hurts accuracy, so keep it (dropout partially mitigates). |
| depth = 6n+4 | 3 groups × n blocks × 2 convs + conv1 + fc-input = 6n+4 conv layers. |
| projection 1x1 only when channels change (group transitions) | identity elsewhere (ResNet convention). |
