# Inception-v3 — synthesis (grounded in arXiv 1512.00567 source + torchvision)

## Source
- arXiv 1512.00567 (verified). LaTeX source read in full (all .tex + model.txt).
- Canonical code: torchvision/models/inception.py — Inception3 matches model.txt exactly. Stem 32/32/64/pool/80/192/pool; Mixed_5b/5c/5d = InceptionA (figure 5 / inceptionv2 fig, 5x5->two 3x3); Mixed_6a=InceptionB (grid reduction 35->17); Mixed_6b-6e=InceptionC (7x7 factorized 1x7+7x1); Mixed_7a=InceptionD (reduction 17->8); Mixed_7b/7c=InceptionE (expanded 8x8 modules); aux classifier on Mixed_6e (17x17); fc 2048->1000.

## Naming clarification (in-frame: never reference "the paper")
- The paper's Table 1 architecture is called "Inception-v2" in the text. "Inception-v3" = that architecture PLUS the cumulative tricks: BN-auxiliary classifier + RMSProp + Label Smoothing + factorized 7x7 (Table 3 in paper). The method we land on (the full recipe) is Inception-v3. The reasoning derives the whole package.

## Pain point / research question (in-frame, late 2015)
- Since 2014 (VGG, GoogLeNet) very deep convnets are mainstream; big gains transfer across vision tasks. But scaling up = more compute + params.
- VGG: simple/uniform but expensive (3x params of AlexNet, huge FLOPs). GoogLeNet/Inception: only 5M params (12x less than AlexNet's 60M), much lower compute — feasible for mobile/big-data.
- BUT Inception's complexity makes it hard to modify. Naively scaling up (doubling all filter banks) → 4x compute and params. Original GoogLeNet paper didn't explain WHY its design choices were made → hard to adapt while keeping efficiency.
- Goal: general principles + optimization ideas to scale up convnets efficiently — use added compute as efficiently as possible via factorized convolutions + aggressive regularization. Computational efficiency + low param count still matter (mobile, big-data).

## Four general design principles (Sec "General Design Principles")
1. Avoid representational bottlenecks, especially early. Representation size should gently decrease input→output; don't compress extremely. Dimensionality is rough proxy for information content.
2. Higher-dimensional representations are easier to process locally; more activations per tile → more disentangled features → trains faster.
3. Spatial aggregation can be done over lower-dim embeddings without much loss of representational power — reduce dimension BEFORE spatial (e.g. 3x3) conv. Reason: strong correlation between adjacent units → little info loss in dim reduction if used in spatial aggregation context; and it promotes faster learning.
4. Balance width and depth. Distribute compute budget balanced between depth and per-stage filters. Increasing both in parallel is optimal for fixed compute.

## Factorizing convolutions (Sec "Factorizing Convolutions with Large Filter Size")
### Into smaller convolutions
- Larger spatial filters disproportionately expensive: a 5x5 conv with n filters over m-filter grid is 25/9 ≈ 2.78x more expensive than a 3x3 with same n.
- Replace 5x5 by a 2-layer mini-network: each 5x5 output = small FC net sliding over 5x5 tiles → exploit translation invariance → replace FC with conv → two stacked 3x3 convs have the same receptive field as one 5x5.
- Cost: with α=1 (no expansion) two 3x3 vs one 5x5 = (9+9)/25 → 28% reduction in compute (and params, since fully conv → each weight = one mult/activation).
- α = n/m, the per-unit activation change factor; for an aggregating 5x5 α≈1.5 in GoogLeNet; expand in two steps by sqrt(α) each.
- Two questions: (a) loss of expressiveness? (b) should first layer be linear (since factorizing the linear part)? Control experiments: linear activation always inferior to ReLU in all stages. Linear+ReLU settles at 76.2% vs two ReLU 77.2% top-1 after 3.86M ops. Attribute gain to enhanced space of variations the net can learn, esp. with BatchNorm on outputs.

### Into asymmetric convolutions
- Filters larger than 3x3 always reducible to 3x3 sequence. Can we factorize 3x3 further? Factor into 2x2 → only 11% saving.
- Better: ASYMMETRIC. n×1 then 1×n. A 3x1 followed by 1x3 = same receptive field as 3x3, but 33% cheaper (if #in = #out filters).
- In general n×n → 1×n then n×1; saving grows with n.
- In practice: asymmetric factorization does NOT work well on early layers; works very well on medium grids (m×m, m∈[12,20]). There, use 1×7 then 7×1. Chose n=7 for the 17×17 grid.

## Auxiliary classifiers (Sec "Utility of Auxiliary Classifiers")
- GoogLeNet introduced aux classifiers to push useful gradients to lower layers / combat vanishing gradient. Lee et al: aux classifiers promote stable learning/convergence.
- Finding: aux classifiers do NOT improve convergence early — training with/without side head looks identical until high accuracy. Near end, net with aux branch overtakes slightly.
- Removing the LOWER aux branch had no adverse effect → original "evolve low-level features" hypothesis is misplaced.
- New interpretation: aux classifiers act as REGULARIZERS. Supported by: main classifier performs better if side branch is batch-normalized or has dropout → weak evidence BN acts as regularizer. BN of side head layers → 0.4% absolute top-1 gain.

## Efficient grid size reduction (Sec)
- Traditionally pool to reduce grid. To avoid representational bottleneck, expand filters BEFORE pooling: d×d×k → d/2×d/2×2k needs a stride-1 conv with 2k filters then pool → dominated by expensive conv on large grid = 2 d² k² ops.
- Alternative: pool then conv = 2(d/2)²k² (1/4 cost) BUT creates representational bottleneck (rep drops to (d/2)²k) → less expressive.
- Solution: two parallel stride-2 blocks P (pooling) and C (convolution), both stride 2, concatenate filter banks → cheaper AND avoids bottleneck.

## Architecture (Table 1, the "Inception-v2" layout)
- Input 299×299×3.
- Stem: conv 3×3/2 (→149×149×32); conv 3×3/1 (→147×147×32); conv 3×3/1 padded (→147×147×64); pool 3×3/2 (→73×73×64); conv 3×3/1 (→73×73×80); conv 3×3/2 (→71×71×... wait model.txt: conv 1×80 then conv 3×192). Table 1 says: conv 3×3/2, conv 3×3/1, conv padded 3×3/1, pool 3×3/2, conv 3×3/1, conv 3×3/2 [80], conv 3×3/1 [192]. model.txt: conv(3,32,s2), conv(3,32), conv(3,64,SAME), maxpool(3,s2), conv(1,80,VALID), conv(3,192,VALID), maxpool(3,s2). Note the 7×7 stem of older nets is factorized into three 3×3.
- 3× Inception (fig 5, "inceptionv2": 5x5 replaced by two 3x3) at 35×35×288.
- grid reduction → 17×17×768.
- 5× Inception (fig 6, "inceptionv3": 1×7 + 7×1 factorized n×n) at 17×17×768.
- grid reduction → 8×8×1280.
- 2× Inception (fig 7, "inceptionv4": expanded filter banks, 1×3 and 3×1 in parallel) at 8×8×2048.
- pool 8×8 → 2048; linear logits; softmax 1000.
- 42 layers deep; ~2.5× GoogLeNet compute; still cheaper than VGG.
- model.txt exact filter banks: see file. Aux head on top of mixed_7 (last 17×17): AvgPool(5,s3) → Conv(1,128) → FC(768) → Softmax(weight=0.4, label_smoothing=0.1).

## Label smoothing regularization (LSR) (Sec)
- Softmax: p(k|x) = exp(z_k)/Σ exp(z_i). Cross-entropy ℓ = −Σ_k log(p(k)) q(k). Gradient ∂ℓ/∂z_k = p(k) − q(k), bounded in [−1,1].
- Hard target q(k)=δ_{k,y}: maximizing log-likelihood approached only as z_y ≫ z_k. Two problems: (1) overfitting (full prob to ground truth doesn't guarantee generalization); (2) encourages largest logit ≫ others → with bounded gradient, reduces adaptability → model too confident.
- LSR: q'(k|x) = (1−ε) δ_{k,y} + ε u(k). Use prior u(k); experiments use uniform u(k)=1/K → q'(k) = (1−ε)δ_{k,y} + ε/K.
- Prevents largest logit from getting much larger than others; all q'(k) have positive lower bound.
- Cross-entropy interpretation: H(q',p) = (1−ε) H(q,p) + ε H(u,p). Second term penalizes deviation of p from prior u, relative weight ε/(1−ε). H(u,p) = D_KL(u‖p) + H(u).
- ImageNet K=1000, u=1/1000, ε=0.1 → consistent ~0.2% absolute improvement top-1 and top-5.

## Training methodology (Sec)
- TensorFlow distributed, 50 replicas on NVidia Kepler, batch 32, 100 epochs.
- Earlier: momentum 0.9. Best models: RMSProp decay 0.9, ε=1.0.
- Learning rate 0.045, decayed every 2 epochs by exponential factor 0.94.
- Gradient clipping threshold 2.0 stabilizes training.
- Evaluations: running average (EMA) of parameters over time.

## Lower-resolution input (Sec)
- Higher input resolution helps, but must separate effect of resolution vs model capacity/compute. Keep compute constant: reduce stride of first two layers or remove first pool for low-res input.
- 79×79 (stride1, no pool): 75.2%; 151×151 (stride1, pool): 76.4%; 299×299 (stride2, pool): 76.6% top-1 (constant compute). Low-res nearly matches high-res at equal compute. (These are diagnostic findings about resolution-vs-compute, knowable design facts; the proposed final-model wins are out of scope.)

## What "Inception-v3" = (cumulative, Table 3 of paper)
- Inception-v2 base + RMSProp + Label Smoothing + Factorized 7×7 (i.e. the stem 7×7 factorized into 3×3s included) + BN-auxiliary (BN in the side head / fully-connected layer of aux classifier). All four together = Inception-v3.

## Load-bearing ancestors (context / baselines)
- AlexNet (Krizhevsky 2012) — 60M params; first deep CNN ImageNet winner; tasks transfer.
- VGGNet (Simonyan & Zisserman 2014) — architectural simplicity (3x3 stacks), but high compute, 3x AlexNet params. Established that two 3x3 = one 5x5 receptive field with fewer params + extra nonlinearity.
- GoogLeNet / Inception-v1 (Szegedy et al. 2014/2015) — Inception modules (parallel 1x1/3x3/5x5/pool branches concatenated), generous 1x1 dimension reduction, only 5M params; aux classifiers; the architecture being scaled up. Gap: complex, hard to modify, undocumented design rationale.
- BatchNorm (Ioffe & Szegedy 2015) — normalize layer pre-activations over minibatch; faster training, large LR, mild regularizer. The "Inception-v2" baseline in their result tables is BN-Inception.
- He et al. 2015 (PReLU/delving deep) — denser higher-performing successor; the SoTA being beaten on accuracy at higher compute.
- Sutskever et al. 2013 — momentum. Tieleman & Hinton — RMSProp. Pascanu et al. 2012 — gradient clipping.
- Lee et al. 2014 — deeply-supervised nets (aux classifiers promote stable learning).

## Design-decision → why table
| Decision | Why this, not alternative |
|---|---|
| Factorize 5×5 → two stacked 3×3 | same receptive field, (9+9)/25 = 28% cheaper compute+params; extra nonlinearity adds expressiveness. |
| Use ReLU (not linear) in the first factorized layer | control experiment: linear+ReLU 76.2% < two ReLU 77.2%; ReLU enlarges the learnable function space, esp. with BN on outputs. |
| Factorize n×n → 1×n then n×1 (asymmetric), n=7 | 3×3→2×2 only 11% saving; 3×1+1×3 = 33% cheaper; asymmetric saving grows with n. Only on medium grids (12–20); fails on early layers. |
| Reduce dimension (1×1) before spatial conv | principle 3: adjacent units strongly correlated → little info loss in reduction when output feeds spatial aggregation; promotes faster learning. |
| Parallel stride-2 conv ‖ stride-2 pool for grid reduction, concatenate | pool-then-conv is 1/4 cost but creates representational bottleneck (principle 1); conv-then-pool avoids bottleneck but 3× expensive; parallel branches are cheap AND bottleneck-free. |
| Factorize 7×7 stem into three 3×3 | same idea as 5×5→two 3×3; cheaper stem, more nonlinearity. |
| Expand filter banks on coarsest 8×8 grid (parallel 1×3, 3×1) | principle 2: high-dim representations easier to process locally; coarsest grid is where high-dim sparse representation is most critical (ratio of local 1×1 processing vs spatial aggregation is highest). |
| Aux classifier kept, BN'd, only the upper one | not for low-level gradient (that hypothesis is wrong — early training identical with/without); it's a regularizer; lower aux branch removable with no harm; BN side head → +0.4% top-1. |
| Avoid early representational bottleneck; rep size decreases gently | principle 1: extreme early compression loses information that no later layer can recover. |
| Label smoothing ε=0.1, uniform u | hard one-hot → overconfidence, largest logit ≫ rest, bounded gradient reduces adaptability + overfits; mixing in ε/K floor keeps all targets positive, regularizes, +0.2% top-1/top-5. |
| RMSProp (decay 0.9, ε=1.0) over momentum | best models; adaptive per-parameter scaling. |
| LR 0.045, ×0.94 every 2 epochs; grad clip 2.0; EMA of weights | exponential decay schedule; clipping stabilizes; EMA smooths eval. |
| Input 299×299 | gives the stride/pool schedule that lands on 8×8 coarsest grid; low-res (79/151) nearly matches at equal compute. |
