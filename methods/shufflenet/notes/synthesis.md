# ShuffleNet — synthesis (grounded in arXiv 1707.01083 source + jaxony/ShuffleNet code)

## Source
- arXiv 1707.01083 (verified). LaTeX read in full.
- Canonical code: jaxony/ShuffleNet model.py — channel_shuffle(reshape (g,n)→transpose(1,2)→flatten), ShuffleUnit (grouped 1×1 compress→shuffle→DWConv3×3→grouped 1×1 expand, add/concat), bottleneck=out//4, stage2 first unit grouped_conv=False.

## Pain point / research question (in-frame, mid-2017)
- Trend: deeper/larger CNNs (hundreds of layers, thousands of channels, billions of FLOPs). This examines the OPPOSITE extreme: best accuracy at very limited budgets, tens-hundreds of MFLOPs (10–150 MFLOPs), for mobile platforms (drones, robots, smartphones).
- Distinct from pruning/compressing/low-bit a "basic" architecture — aim to design a highly efficient BASIC architecture for these tiny budgets.
- Diagnostic observation: SoTA efficient architectures (Xception depthwise separable, ResNeXt group conv) become LESS efficient in extremely small networks because of costly DENSE 1×1 convolutions. In ResNeXt only the 3×3 layers are grouped; the pointwise (1×1) convs occupy 93.4% of multiply-adds (cardinality=32). In tiny nets, expensive 1×1 convs force a limited number of channels under the budget → damages accuracy.

## Two new operations
### Pointwise group convolution
- Straightforward fix: apply channel-sparse connections (group convolution) ALSO on the 1×1 layers, not just the 3×3. Each conv operates only on its input channel group → big compute reduction.

### Channel shuffle (the side-effect fix)
- Side effect of stacking group convs: outputs from a group only relate to inputs within that group → blocks information flow between groups → weakens representation.
- Fix: let a group conv take input from DIFFERENT groups. Divide channels in each group into subgroups, feed each next-layer group with different subgroups → input/output channels fully related.
- Implementation: for a layer with g groups whose output has g×n channels: reshape channel dim to (g, n), transpose to (n, g), flatten back. Works even if the two convs have different #groups. Differentiable → embeddable for end-to-end training.
- cuda-convnet had a "random sparse convolution" = random channel shuffle + group conv, but different purpose, seldom exploited.

## ShuffleNet Unit (Sec 3.2, Fig 2)
- Start from ResNet bottleneck unit (residual block).
- Residual branch:
  (b) stride 1: 1×1 grouped conv (pointwise group conv, reduce to bottleneck) → BN → ReLU → channel shuffle → 3×3 DEPTHWISE conv (on bottleneck) → BN (NO ReLU after DWConv, per Xception) → 1×1 grouped conv (expand to match shortcut) → BN. Then elementwise ADD with identity shortcut → ReLU.
  (c) stride 2: (i) 3×3 average pool (stride 2) on shortcut path; (ii) replace elementwise add with channel CONCATENATION (cheaply enlarges channel dim); depthwise conv stride 2.
- Second pointwise group conv recovers channel dimension to match shortcut. No extra channel shuffle after second pointwise (comparable scores).
- BN/nonlinearity usage like ResNet/ResNeXt, EXCEPT no ReLU after depthwise conv.
- Depthwise conv only on the bottleneck feature map (intentionally), to prevent overhead — DWConv has low theoretical complexity but poor compute/memory-access ratio on low-power mobile devices, hard to implement efficiently.

## FLOPs comparison (Sec 3.2)
- Input c×h×w, bottleneck channels m:
  - ResNet unit: hw(2cm + 9m²) FLOPs.
  - ResNeXt unit: hw(2cm + 9m²/g) FLOPs.
  - ShuffleNet unit: hw(2cm/g + 9m) FLOPs.
- (2cm = two 1×1 convs c↔m; 9m² = dense 3×3 at m channels; 9m²/g = grouped 3×3; for ShuffleNet 2cm/g = grouped 1×1 convs, 9m = depthwise 3×3 (9 per channel × m channels).)
- → ShuffleNet has the least complexity; given a budget it can use WIDER feature maps. Critical for small nets (tiny nets have insufficient channels to process info).

## Architecture (Table 1)
- 3 stages of ShuffleNet units. First unit in each stage stride 2. Within a stage, same hyperparameters; next stage doubles output channels. Bottleneck channels = 1/4 of output channels (like ResNet).
- Conv1: 3×3, stride 2 → 112×112×24. MaxPool 3×3 stride 2 → 56×56.
- Stage2: 28×28, stride-2 unit then 3 stride-1 units. Output channels (g=1/2/3/4/8): 144/200/240/272/384.
- Stage3: 14×14, stride-2 then 7 stride-1 units. Channels: 288/400/480/544/768.
- Stage4: 7×7, stride-2 then 3 stride-1 units. Channels: 576/800/960/1088/1536.
- GlobalPool 7×7 → FC 1000.
- g (groups) controls connection sparsity of pointwise convs; adapt output channels to keep ~140 MFLOPs. Larger g → more output channels (more filters) for fixed complexity → encode more info, but might degrade an individual filter (fewer input channels per filter).
- NOTE: Stage2 first pointwise layer is NOT grouped (input channels relatively small, only 24).
- Scale factor s: "ShuffleNet s×" scales #filters by s → complexity ~s² ×. 1×=140M, 0.5×=38M, 0.25×=13M.

## Ablation findings (diagnostic, controlled experiments — Tables 2,3)
- Group conv: g>1 consistently better than g=1 (Xception-like). Smaller models benefit more from groups (1× best g=8 +1.2%; 0.5× +3.5%; 0.25× +4.4%). Hypothesis: gain from wider feature maps (group conv allows more channels for fixed budget). For some models large g (=8) saturates/drops (fewer input channels per filter harms representation); for smaller models larger g consistently better.
- Channel shuffle: consistently boosts scores; bigger gain when g large (g=8: 1×/0.5×/0.25× gains 5.2/5.8/3.8). Shows importance of cross-group info interchange.
- (g=1 = no pointwise group conv = "Xception-like" structure.)

## Training details (Sec 4)
- Follow ResNeXt training settings, two exceptions: (i) weight decay 4e-5 (not 1e-4), linear-decay LR from 0.5 to 0; (ii) less aggressive scale augmentation. (Small nets underfit rather than overfit — same as MobileNet.)
- 4 GPUs, batch 1024, 3×10⁵ iterations, 1–2 days. Single-crop top-1: 224×224 center crop from 256× input.

## Load-bearing ancestors (baselines)
- ResNet (He et al. 2016) — bottleneck residual unit (1×1→3×3→1×1); the base unit; FLOPs hw(2cm+9m²).
- ResNeXt (Xie et al. 2016) — grouped 3×3 (cardinality); FLOPs hw(2cm+9m²/g); but dense 1×1 dominate (93.4%) in small nets.
- Xception (Chollet 2016) — depthwise separable convs (depthwise 3×3 + pointwise 1×1); no ReLU after depthwise. Generalizes Inception separable convs.
- MobileNet (Howard et al. 2017) — depthwise separable convs for lightweight models; SoTA lightweight; the main competitor; small nets underfit.
- Group convolution: AlexNet (Krizhevsky 2012, 2-GPU split); demonstrated in ResNeXt.
- BatchNorm (Ioffe & Szegedy 2015). GoogLeNet, SqueezeNet, SENet — efficient designs.
- cuda-convnet random sparse convolution (random shuffle + group conv).

## Design-decision → why table
| Decision | Why this, not alternative |
|---|---|
| Pointwise GROUP convolution (group the 1×1 layers) | in tiny nets the dense 1×1 convs dominate FLOPs (93.4% in ResNeXt); grouping them frees budget → more channels; without it, budget caps channels and hurts accuracy. |
| Channel shuffle after the first pointwise group conv | stacking group convs blocks cross-group info flow (a group's output sees only its group's input) → weak representation; shuffle (reshape (g,n)→transpose→flatten) mixes groups, restores full channel relation, is cheap + differentiable. |
| Depthwise 3×3 on the bottleneck only | DWConv is cheap in theory but has poor compute/memory-access ratio on mobile → keep it only on the narrow bottleneck to limit overhead. |
| No ReLU after depthwise conv | following Xception — ReLU after depthwise hurts (depthwise has few params per channel; rectifying kills info). |
| Second pointwise group conv to restore channels; no shuffle after it | match shortcut dims; second shuffle gives comparable scores so omit for simplicity. |
| Stride-2 unit: avg-pool shortcut + CONCAT instead of add | concatenation enlarges channel dimension cheaply at downsampling (channels double between stages); avg pool aligns the shortcut spatial size. |
| Bottleneck channels = 1/4 output channels | follows ResNet bottleneck ratio. |
| Stage2 first 1×1 NOT grouped | input channels small (24) → grouping there saves little and over-sparsifies. |
| g adapts output channels to hold ~140 MFLOPs | larger g → more channels for fixed budget (more info), but too-large g starves each filter of input channels; sweet spot depends on model size (smaller models prefer larger g). |
| Scale factor s (s× → ~s² complexity) | simple width knob to hit a target budget. |
| weight decay 4e-5, linear LR 0.5→0, less aggressive aug | tiny nets underfit not overfit → lighter regularization, gentler augmentation. |
