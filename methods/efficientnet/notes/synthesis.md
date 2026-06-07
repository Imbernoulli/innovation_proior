# EfficientNet — synthesis notes (pre-Phase-2)

## The pain point / research question
ConvNets are routinely scaled up for more accuracy when a bigger budget is available
(ResNet-18→200 by depth; WideResNet by width; GPipe by resolution 480×480). But "how to
scale" is folklore: people bump ONE dimension (depth OR width OR resolution), or tune two/three
by hand. Hand-tuning two-three dims is tedious and yields sub-optimal accuracy/FLOPS. Question:
is there a *principled* rule for trading a fixed extra compute budget across depth, width,
resolution that beats single-dimension scaling — and is cheap to apply to any target budget?

## Three knobs and what each buys (motivating diagnostics, pre-method facts)
- **Depth d (#layers)**: deeper → richer/more-complex/hierarchical features, generalizes.
  But vanishing gradients; skip connections + BN alleviate. Diagnostic: gains saturate —
  ResNet-1000 ≈ ResNet-101 accuracy despite far more layers. (single-dim depth curve plateaus ~80%)
- **Width w (#channels)**: wider → finer-grained features, easier to train (WideResNet). Used for
  small models (MobileNet "depth multiplier"). But shallow+very-wide can't capture high-level
  features; accuracy saturates fast as w grows.
- **Resolution r (input HxW)**: higher res → more fine-grained patterns. 224→299 (Inception)→
  331 (NASNet)→480 (GPipe); detection uses 600. Gains diminish at very high res.
- **Observation 1**: scaling ANY single dim improves accuracy but the gain diminishes for bigger
  models — each curve plateaus around ~80% top-1.

## The key empirical coupling (Observation 2)
Scale width w at fixed (d=1,r=1): saturates quickly. Scale width at (d=2,r=2): much higher
accuracy at the SAME FLOPS. So the dims are NOT independent: at higher resolution you want more
depth (bigger receptive field to cover more pixels) AND more width (more channels to capture the
extra fine-grained patterns). Intuition: bigger image ⇒ need more layers (receptive field) +
more channels (finer patterns). ⇒ Must BALANCE all three, not push one.

## The compound-scaling rule + the FLOPS derivation (CORE MATH)
Define a single user knob φ ("how much extra compute"). Assign it across dims with constants α,β,γ:
  depth d = α^φ , width w = β^φ , resolution r = γ^φ , with α,β,γ ≥ 1.

FLOPS scaling of a regular conv:
- Depth: L layers ⇒ FLOPS ∝ L, so ∝ d  (linear).
- Width: a conv mapping c_in→c_out costs ∝ c_in·c_out; scaling both by w ⇒ ∝ w².
- Resolution: FLOPS ∝ spatial size H·W; scaling each side by r ⇒ ∝ r².
Convs dominate ConvNet cost, so total FLOPS ∝ d · w² · r².
Substitute: total FLOPS scales by (α^φ)·(β^φ)²·(γ^φ)² = (α·β²·γ²)^φ.
To make φ a clean "2^φ× compute" dial, IMPOSE α·β²·γ² ≈ 2. Then FLOPS ≈ 2^φ for any φ.
(Memory ∝ w·r² roughly, also bounded.) Note d enters linearly but w,r squared — that asymmetry
is exactly why the constraint has α to the 1st power and β,γ squared.

## Two-step recipe (why two-step)
- STEP 1: fix φ=1 (assume 2× resources). Small grid search over α,β,γ subject to α·β²·γ²≈2 on the
  SMALL baseline B0. Best for B0: **α=1.2, β=1.1, γ=1.15**. Check: 1.2·1.1²·1.15²
  = 1.2·1.21·1.3225 = 1.920 ≈ 2. ✓
- STEP 2: freeze α,β,γ; sweep φ to get B1..B7. Why two-step: searching α,β,γ *directly* on a big
  model is prohibitively expensive; search ONCE on the small model, reuse the ratios everywhere.
  Cost of search is ~constant in target size.
- Per-model coefficients (w,d,res,dropout): B0 (1.0,1.0,224,0.2), B1 (1.0,1.1,240,0.2),
  B2 (1.1,1.2,260,0.3), B3 (1.2,1.4,300,0.3), B4 (1.4,1.8,380,0.4), B5 (1.6,2.2,456,0.4),
  B6 (1.8,2.6,528,0.5), B7 (2.0,3.1,600,0.5). (These are α^φ etc. rounded; e.g. B7 depth
  1.2^? ≈ 3.1.)

## Why a good baseline matters / the NAS baseline (B0)
Scaling can't change the per-layer operator F_i; it only changes L_i,C_i,H_i,W_i. So the ceiling
is set by the baseline. To showcase scaling, design a strong mobile-size baseline by NAS.
- Multi-objective NAS in the MnasNet search space, optimizing ACC(m)·[FLOPS(m)/T]^w with w=-0.07,
  T = target FLOPS (400M). FLOPS (not latency) as objective since no specific device targeted.
  Produces B0, similar to MnasNet but slightly bigger (400M target).
- Building block: **MBConv** (mobile inverted bottleneck, from MobileNetV2) + **squeeze-excitation**.

## B0 stage table (the baseline to scale)
stem Conv3x3 s2 →32ch @112². Then 7 MBConv stages:
1. MBConv1 k3 r1  i32→o16  (expand=1, no expansion conv)
2. MBConv6 k3 r2  o24  (stride 2 first)
3. MBConv6 k5 r2  o40  (stride 2)
4. MBConv6 k3 r3  o80  (stride 2)
5. MBConv6 k5 r3  o112 (stride 1)
6. MBConv6 k5 r4  o192 (stride 2)
7. MBConv6 k3 r1  o320 (stride 1)
Head: Conv1x1→1280, global avg pool, FC. All MBConv use SE ratio 0.25. (canonical block strings:
r1_k3_s11_e1_i32_o16_se0.25 / r2_k3_s22_e6_i16_o24 / r2_k5_s22_e6_i24_o40 / r3_k3_s22_e6_i40_o80
/ r3_k5_s11_e6_i80_o112 / r4_k5_s22_e6_i112_o192 / r1_k3_s11_e6_i192_o320.)

## Design-decision → why table (load-bearing)
| Decision | Why this, why not the alternative |
|---|---|
| Single compound knob φ + fixed ratios α,β,γ | Reduces the per-layer L_i,C_i,H_i,W_i design space (huge) to one scalar; balance is "scale each by a constant ratio". Alternative (free per-dim tuning) = tedious manual search, sub-optimal. |
| α·β²·γ²≈2 constraint | Makes φ a clean log2-compute dial: FLOPS≈2^φ. From FLOPS ∝ d·w²·r². Without it φ has no consistent compute meaning. |
| Balance all 3 vs scale 1 | Obs.1: single-dim saturates ~80%. Obs.2: width at (d,r)=(2,2) >> width at (1,1) for same FLOPS. Dims coupled (receptive field & pattern granularity follow resolution). |
| Grid-search α,β,γ on small B0, reuse | Direct search on big models is too expensive; do it once on B0. |
| α=1.2,β=1.1,γ=1.15 | Best grid point under constraint for B0; note depth gets the largest exponent because it's only linear in FLOPS so it's "cheaper" per unit accuracy than width/res which are quadratic. |
| MBConv (inverted residual, depthwise-separable) baseline block | Depthwise-separable conv = depthwise (per-channel spatial) + pointwise (1×1 channel mix) costs ~k²+C instead of k²·C → 8-9× cheaper for k=3. Inverted residual: expand→depthwise→project, skip on the THIN bottleneck endpoints (cheap to add; memory-light: never materialize big tensor across blocks). |
| Linear bottleneck (no ReLU on projection output) | ReLU on a low-dim (bottleneck) tensor destroys information (can't recover after collapsing to a subspace); keep the projection linear. |
| Expansion ratio 6 (1 in first block) | Depthwise conv works in the EXPANDED high-dim space where ReLU is non-destructive; 6× is MnasNet/MobileNetV2 sweet spot. First block input already small → expand=1 (skip the expansion 1×1). |
| Squeeze-and-Excitation, ratio 0.25 | Channel attention: global-avg-pool → 2 FC (reduce by r then restore) → sigmoid gate per channel. Recalibrates channels by global context, ~free compute. SE reduction relative to block INPUT filters. |
| Swish/SiLU activation x·σ(x) | smooth, non-monotonic; empirically > ReLU for these nets. |
| Stochastic depth (drop-connect), survival 0.8, linearly scaled by depth | regularize deep scaled nets; drop whole residual branch with prob rising along depth. |
| Dropout 0.2→0.5 across B0→B7 | bigger models need more regularization; scale it with model size. |
| round_filters to multiple of 8 (depth_divisor=8), within 10% | width scaling must keep channel counts hardware-friendly (multiples of 8); guard against >10% rounding-down. |
| round_repeats = ceil(d · L_i) | depth scaling rounds layer-repeats up. |
| RMSProp, lr 0.256 decay 0.97/2.4ep, wd 1e-5, bn-mom 0.99, AutoAugment | MnasNet-style training recipe. |

## Ancestors (load-bearing) — for context.md baselines
- **Depthwise separable conv (MobileNetV1, Howard 2017; Xception, Chollet 2017)**: factor a
  k×k×C_in×C_out conv into depthwise (k×k per channel) + pointwise (1×1). Cost
  D_K²·M·D_F² + M·N·D_F² vs D_K²·M·N·D_F²: ratio 1/N + 1/D_K² (~1/8-1/9 for 3×3). Gap: still
  hand-designed; how to scale?
- **MobileNetV2 (Sandler 2018)**: inverted residual + linear bottleneck. Block: 1×1 expand
  (×t=6) → 3×3 depthwise → 1×1 project (linear). Residual on the narrow bottleneck endpoints.
  This is MBConv. Gap: per-dim scaling (width "multiplier") only.
- **SENet (Hu 2018)**: SE block — squeeze (GAP)→excite (FC reduce r=16→FC restore→sigmoid)→
  rescale channels. Adaptive channel-wise attention, near-free. Best single-model ImageNet 82.7%
  but 146M params. Gap: huge.
- **MnasNet (Tan 2018)**: platform-aware multi-objective NAS. Reward ACC·(LAT/T)^w, factorized
  hierarchical search space (per-block conv op/kernel/SE/skip). EfficientNet reuses this search
  space but optimizes FLOPS not latency, target 400M → B0.
- **ResNet (He 2016)** / **WideResNet (Zagoruyko 2016)**: the canonical depth- and width-scaling
  baselines; used to *demonstrate* compound scaling generalizes (MobileNetV1/V2, ResNet-50).
- **GPipe (Huang 2018)**: pushed SOTA via resolution(480)+giant model w/ pipeline parallelism;
  84.3% / 557M — the "scale up brute force" prior art compound scaling competes with.

## Evaluation settings (pre-method, no outcomes)
ImageNet-1k (1.28M train / 50k val; report top-1/top-5, single-crop single-model; minival 25k
held out for early stopping). Transfer: CIFAR-10/100, Birdsnap, Stanford Cars, Flowers, FGVC
Aircraft, Oxford-IIIT Pets, Food-101. Yardstick: accuracy vs #params and vs FLOPS; inference
latency on CPU. Train recipe: RMSProp, AutoAugment, stochastic depth, dropout.

## Scaffold ↔ code correspondence
Pre-method scaffold = generic ConvNet harness: a per-channel-cheap conv primitive exists
(depthwise separable), a residual block abstraction exists, a stage-based body, a classifier head,
SGD/RMSProp loop, ImageNet pipeline. EMPTY SLOTS: (1) the building block we'll design, (2) the
baseline stage table, (3) **the scaling rule** mapping a budget to (depth,width,resolution) —
the one big contribution slot. Final code fills: MBConv+SE block, B0 stage list, round_filters/
round_repeats + compound (α^φ,β^φ,γ^φ) scaler.
</content>
</invoke>
