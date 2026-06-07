# Stochastic Depth synthesis

## Verified arXiv id
1603.09382 "Deep Networks with Stochastic Depth" (Huang, Sun, Liu, Sedra, Weinberger; Cornell, ECCV 2016).
Canonical code: github.com/yueatsprograms/Stochastic_Depth (Torch/Lua) — ResidualDrop.lua + main.lua (fetched to code/). Final code presented in PyTorch faithful to this module.

## Pain points (the problem)
Depth drives expressiveness, but very deep nets bring three problems:
1. Vanishing gradients: repeated mult by small weights shrinks backprop signal in early layers.
2. Diminishing feature reuse (forward info loss): input/early features washed out by repeated mult with random weights.
3. Long training time: forward+backward scale linearly with depth; 152-layer ResNet ~ weeks on ImageNet.
The dilemma: short nets train fast and flow info well but aren't expressive; deep nets are expressive but hard/slow to train. Seemingly contradictory wish: a DEEP net at test time but a SHORT net during training.

## ResNet baseline (the thing modified)
ResBlock update: H_ℓ = ReLU(f_ℓ(H_{ℓ-1}) + id(H_{ℓ-1})), f_ℓ = Conv-BN-ReLU-Conv-BN (bottleneck on ImageNet). id = identity, or linear projection when dims change. Network = composition of L ResBlocks. ResNet motivated by: deeper plain nets get HIGHER TRAINING error (degradation) → skip connections fix it.

## The method
Randomly drop entire ResBlocks during training, bypassing via the identity skip; keep connection identity. b_ℓ ∈ {0,1} Bernoulli, p_ℓ = Pr(b_ℓ=1) = survival probability.
Training update (eq res_block_drop):
  H_ℓ = ReLU( b_ℓ · f_ℓ(H_{ℓ-1}) + id(H_{ℓ-1}) ).
If b_ℓ=1 → original ResNet block. If b_ℓ=0 → H_ℓ = id(H_{ℓ-1}) = H_{ℓ-1}.
Why does b_ℓ=0 give pure identity (not ReLU(id(H)))? Because the input H_{ℓ-1} is always non-negative (it's the output of a previous ReLU, or the initial Conv-BN-ReLU stem), so ReLU acts as identity on it. Key: skipped block = exact identity, so the network is genuinely shorter that step.

## Survival probability schedule
p_ℓ new hyperparams; neighbors should be similar. Two options:
- Uniform: p_ℓ = p_L for all ℓ (one hyperparam).
- Linear decay (preferred): p_0 = 1 (input always active), decaying to p_L at last block:
  p_ℓ = 1 − (ℓ/L)(1 − p_L).
Rationale for decay: early layers extract low-level features used by all later layers → should be more reliably present (higher survival). Set p_L = 0.5 throughout (training stable wrt p_L). With p_L=0.2 same error as constant-depth on CIFAR-10 with 40% speedup.

## Expected depth and savings
Effective #blocks L̃ random; E(L̃) = Σ_{ℓ=1}^L p_ℓ.
Under linear decay with p_L=0.5: E(L̃) = (3L−1)/4 ≈ 3L/4. For 110-layer net (L=54 blocks): E(L̃)≈40. Train ~40 blocks, test with 54.
Derivation of (3L−1)/4: Σ_{ℓ=1}^L [1 − (ℓ/L)(1/2)] = L − (1/2L)·(L(L+1)/2) = L − (L+1)/4 = (4L−L−1)/4 = (3L−1)/4. ✓
Training-time savings ≈ 25% under p_L=0.5 (skipped blocks need no forward/backward).

## Test time (eq res_block_drop_test)
All f_ℓ active (full network). Recalibrate each f_ℓ by its survival prob (it was only present a fraction p_ℓ of updates, downstream weights tuned to that):
  H_ℓ^Test = ReLU( p_ℓ · f_ℓ(H_{ℓ-1}^Test; W_ℓ) + H_{ℓ-1}^Test ).
(Like Dropout weight scaling.) Code: at test, output = skip + net(input)·(1−deathRate) where 1−deathRate = p_ℓ.

## Why it improves test error (two reasons + reg)
1. Shorter expected depth during training → shorter forward/gradient chains → stronger gradients esp. in early layers.
2. Implicit ensemble: each block on/off → 2^L sub-networks with shared weights; each minibatch samples & updates one; test averages them (weighting eq). Higher diversity than same-depth ensembling.
Also a regularizer (like Dropout) even WITH BatchNorm — and unlike Dropout, which gives ~no improvement on 110-layer BN ResNets. Stochastic depth makes the net shorter, not thinner.

## Ancestors (load-bearing)
- ResNet (He 2015): residual block, identity skip, degradation observation.
- Highway Networks (Srivastava 2015): parameterized gated skip connections crossing layers.
- Dropout (Srivastava 2014): multiply hidden activations by Bernoulli, co-adaptation reduction, ensemble view, test scaling.
- DropConnect (Wan 2013), Maxout, DropIn — stochastic regularizers.
- BatchNorm (Ioffe 2015): standardize per-minibatch; reduces vanishing gradient, regularizes; Dropout loses effectiveness with it.
- ReLU (Nair 2010).

## Code grounding (ResidualDrop.lua + main.lua)
- ResidualDrop: net = Conv(3×3,stride)-BN-ReLU-Conv(3×3)-BN; skip = Identity (+AvgPool if stride>1, +zero-pad channels if widening = ResNet option A). gate boolean.
- Forward train: output = skip(input); if gate: output += net(input). (gate open ⇒ block active.)
- Forward test: output = skip(input) + net(input)·(1−deathRate).
- main.lua: linear decay sets block i deathRate = i/L · opt.deathRate (deathRate = 1−p). Per-minibatch: openAllGates(); for each block close gate if rand < deathRate. addResidualDrop adds block then a ReLU after (the outer ReLU of the residual update).
- ImageNet uses bottleneck block; CIFAR uses the 2-conv basic block; 110-layer = 54 blocks.

## Scaffold ↔ final code correspondence
Pre-method scaffold: a ResNet of residual blocks, each block = (residual branch f, identity/projection skip), update H = ReLU(f(H)+id(H)); a per-block scalar hyperparam slot and a train/test forward stub. Final code fills: per-minibatch Bernoulli gate that zeroes the residual branch (training), the linear-decay survival schedule p_ℓ across blocks, and the test-time scaling of f by p_ℓ.
