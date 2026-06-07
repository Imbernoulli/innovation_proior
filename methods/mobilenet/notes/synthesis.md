# Synthesis — MobileNet derivation

## Pain point
~2016: CNN accuracy gains came from going deeper/wider (VGG 15.3B mult-adds, 138M params; GoogLeNet; Inception V3). But mobile/embedded/AR/robotics/self-driving need real-time inference on tight compute & memory budgets. Two camps existed: (a) compress a pretrained net (pruning, quantization, hashing, low-rank factorization, distillation); (b) train a small net directly. Most "small net" work optimized **parameter count / model size** but not **latency** (actual mult-adds and how implementable they are on GEMM hardware).

## Central object
A standard conv layer: input F is D_F×D_F×M, kernel K is D_K×D_K×M×N, output G is D_F×D_F×N (stride 1, same padding).
- Output: G_{k,l,n} = Σ_{i,j,m} K_{i,j,m,n} · F_{k+i-1,l+j-1,m}.
- Cost (mult-adds): D_K·D_K·M·N·D_F·D_F.
- Key observation: cost is **multiplicative** in (kernel size D_K²)(input channels M)(output channels N)(spatial D_F²). The standard conv does TWO jobs at once: spatially **filtering** each channel, and **combining** channels into N new features. Both jobs are entangled in the same kernel, which is why N and D_K² multiply.

## The factorization (depthwise separable)
Split the two jobs.
- **Depthwise conv**: one D_K×D_K filter per input channel, no cross-channel mixing.
  - Ĝ_{k,l,m} = Σ_{i,j} K̂_{i,j,m} · F_{k+i-1,l+j-1,m}, kernel K̂ is D_K×D_K×M.
  - Cost: D_K·D_K·M·D_F·D_F. (N dropped out — no channel combining.)
- **Pointwise conv**: 1×1 conv, M→N, mixes channels, no spatial extent.
  - Cost: M·N·D_F·D_F. (D_K dropped to 1.)
- Both followed by BN + ReLU.
- Total separable cost: D_K²·M·D_F² + M·N·D_F².

## Reduction ratio (derive inline)
(D_K²·M·D_F² + M·N·D_F²) / (D_K²·M·N·D_F²)
= (D_K²·M·D_F²)/(D_K²·M·N·D_F²) + (M·N·D_F²)/(D_K²·M·N·D_F²)
= 1/N + 1/D_K².
For D_K=3 and N large (N≥64), this ≈ 1/9, i.e. **8–9× less computation**. The 1/N term is negligible for typical N (64..1024); the dominant saving is 1/D_K² = 1/9.

## Architecture (Table 1)
- First layer: full conv 3×3, 3→32, stride 2.
- Then 13 depthwise-separable blocks; channel schedule 32→64→128→128→256→256→512 (×6 at 512) →1024→1024.
- Downsampling via stride-2 in depthwise conv (and first layer): strides at layers giving 224→112→56→28→14→7.
- Each conv (full, depthwise, pointwise) followed by BN+ReLU, except final FC (no nonlinearity)→softmax.
- 28 layers counting depthwise & pointwise separately.
- Global avg pool 7×7 → FC 1024→1000 → softmax.

## Why this design — design-decision → why
- **Why depthwise separable not other factorizations?** Flattened nets (Jin 2014) factor into rank-1 1D filters (fully separable in all dims) — too aggressive, large accuracy loss. Low-rank factorization of pretrained nets (Jaderberg, Lebedev) compresses after the fact, doesn't define a trainable-from-scratch family. Depthwise-separable is the "just right" factorization: separate spatial filtering from channel combining, keep full 2D spatial kernels.
- **Why put 95% of compute in 1×1 convs (and is that good)?** A 1×1 conv is exactly a GEMM (matrix multiply) on the reshaped feature map — no im2col reordering needed (unlike a general k×k conv, which Caffe maps to GEMM only after an im2col copy). GEMM is the single most optimized numerical kernel. So concentrating compute in 1×1 means the theoretical mult-add savings actually translate to wall-clock speedups. Mult-adds alone aren't enough; structured-dense beats unstructured-sparse until very high sparsity.
- **Why width multiplier α (thin) instead of fewer layers (shallow)?** Empirically narrowing beats shortening at equal compute (~3% better). α∈(0,1], typical 1, 0.75, 0.5, 0.25. M→αM, N→αN. Depthwise-sep cost becomes D_K²·αM·D_F² + αM·αN·D_F² → the pointwise term scales ~α², so cost & params drop ~α². Train from scratch, not prune.
- **Why resolution multiplier ρ?** Spatial cost scales with D_F². Set ρ implicitly via input resolution 224/192/160/128. Cost scales ρ². Combined: D_K²·αM·(ρD_F)² + αM·αN·(ρD_F)². ρ touches compute only, not params.
- **Why little/no weight decay on depthwise filters?** They have very few parameters (D_K²·M, vs pointwise M·N), so they barely contribute capacity and regularizing them just hurts. Small models overfit less → less augmentation, no label smoothing, no auxiliary heads (vs Inception V3 training).
- **Why BN+ReLU on both depthwise and pointwise?** Standard 2016 practice (BN, Ioffe & Szegedy 2015) for trainability; depthwise alone is linear-ish per channel, the nonlinearity + normalization between the two sublayers helps.
- **Example layer (Table 3)** internal layer D_K=3, M=N=512, D_F=14: full conv 462M mult-adds / 2.36M params → depthwise-sep 52.3M / 0.27M → α=0.75: 29.6M/0.15M → ρ=0.714 (i.e. 14→10): 15.1M/0.15M (ρ touches compute, not params).

## Canonical implementation
- TF-Slim `mobilenet_v1.py`: depthwise = `slim.separable_conv2d(net, None, [3,3], depth_multiplier=1, stride=s)` (filters=None → depthwise only), pointwise = `slim.conv2d(net, depth, [1,1], stride=1)`; both with BN+ReLU via arg_scope. Width multiplier = `depth_multiplier` applied as `depth = max(int(d*mult), min_depth=8)`. CONV_DEFS list mirrors Table 1.
- PyTorch (kuangliu style): depthwise = `nn.Conv2d(in,in,3,stride,1,groups=in,bias=False)`; pointwise = `nn.Conv2d(in,out,1,1,0,bias=False)`; each + BN + ReLU. `groups=in_channels` is exactly per-channel depthwise.

## Ancestors (load-bearing)
- **Sifre 2014 (rigid-motion scattering, PhD thesis)** — origin of depthwise separable convolution.
- **Inception / Szegedy (BN-Inception, Inception V3 "Rethinking")** — factorized convolutions (e.g. n×n → n×1 + 1×n), 1×1 convs as cheap channel mixers/bottlenecks; used separable convs in first layers.
- **Xception, Chollet 2016** — scaled depthwise separable filters to beat Inception V3; "extreme Inception".
- **Flattened nets, Jin 2014** — fully factorized (rank-1) convolutions, showed extreme factorization is possible.
- **Factorized nets, Wang 2016** — concurrent similar factorized conv + topological connections.
- **SqueezeNet, Iandola 2016** — bottleneck (1×1 squeeze) for tiny param count; size-focused, not latency.
- **BN, Ioffe & Szegedy 2015**; **GEMM/im2col, Caffe (Jia 2014)**; **RMSProp (Tieleman & Hinton)**; **distillation (Hinton 2015)** complementary.
