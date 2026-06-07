# GoogLeNet / Inception — Phase 1 Synthesis (design-decision → why)

## Pain point at the time (2014)
- AlexNet (Krizhevsky 2012) showed deep CNNs win ImageNet. Trend since: more depth (NiN), more width / bigger layers (ZFNet/Zeiler-Fergus, OverFeat), dropout to fight overfitting.
- The naive recipe "just make it bigger" has two costs:
  1. **More params → overfitting** (ImageNet fine-grained classes, e.g. Siberian husky vs Eskimo dog, need lots of labeled data which is expensive).
  2. **Compute blows up quadratically**: chaining two conv layers, a uniform increase in #filters → quadratic increase in multiply-adds. If much of the added capacity ends near zero (weights ~0), compute is wasted.
- Constraint they imposed on themselves: **~1.5 billion multiply-adds at inference** — must be deployable, not an academic curiosity. (AlexNet ~60M params; GoogLeNet ends at ~6.8M params, 12× fewer than AlexNet, yet deeper/more accurate.)

## The theoretical seed: sparsity (Arora et al. 2013) + Hebbian
- Arora, Bhaskara, Ge, Ma 2013 "Provable bounds for learning some deep representations": if the data distribution is representable by a large, very sparse DNN, the optimal topology can be built layer-by-layer by **analyzing correlation statistics of activations and clustering highly-correlated units** into the units of the next layer.
- Resonates with **Hebbian principle** ("neurons that fire together wire together") → suggests it holds even under weaker conditions, in practice.
- So the *fundamental* fix to both overfitting and compute would be **sparse connectivity**, even inside convolutions.
- **The wall:** today's hardware is terrible at non-uniform sparse computation — even a 100× cut in arithmetic doesn't pay off because of lookup/cache-miss overhead; dense matmul on highly tuned BLAS/cuDNN is enormously faster. (Historical note: ConvNets used random/sparse connection tables since LeNet to break symmetry; AlexNet went back to *full* connections precisely to exploit GPU dense compute.)
- **The bridge / the actual idea:** sparse-matrix literature (Çatalyürek 2010) says clustering a sparse matrix into **relatively dense submatrices** gives state-of-the-art practical sparse-matmul performance. → Approximate the optimal *local sparse structure* with **dense components** that current hardware loves. That is the whole Inception thesis: "approximate the expected optimal sparse structure by readily available dense building blocks."

## From theory to the module
- Assume translation invariance → build from convolutional blocks; find the optimal *local* construction and repeat it spatially.
- Arora's layer-by-layer: cluster correlated units. In low layers correlated units concentrate in **local regions** → covered by **1×1 convs** (a cluster in one spot). Some clusters are more spatially spread → covered by **3×3** and **5×5** convs over larger patches. Fewer clusters over larger regions.
- Restrict to filter sizes **1×1, 3×3, 5×5** — to avoid patch-alignment issues (odd sizes, easy SAME padding with pad 0/1/2). "Based more on convenience than necessity."
- Pooling has been essential in SOTA convnets → add a **parallel pooling path** in each module too.
- → **Naive Inception module:** concatenate {1×1, 3×3, 5×5 convs, 3×3 max-pool} outputs along the channel axis into one tensor that feeds the next stage.
- As modules stack, higher layers capture higher abstraction → spatial concentration drops → ratio of 3×3 and 5×5 should increase with depth.

## The wall: naive module blows up — and the 1×1 bottleneck fix
- A 5×5 conv on a layer with many input channels is prohibitively expensive. Pooling path makes it worse: its #output channels = #input channels, so concatenation **monotonically grows** channel count stage to stage → compute blows up within a few stages.
- **Concrete cost math** (load-bearing): naive 5×5 on a 28×28×192 input → 32 output channels:
  `28·28·192·32·5·5 = 120,422,400` MACs.
  With a 1×1 reduction 192→16 first, then 5×5 16→32:
  `28·28·192·16·1·1 = 2,408,448` (reduce) + `28·28·16·32·5·5 = 10,035,200` (5×5) = `12,443,648` MACs.
  → ~**10×** cheaper (120.4M → 12.4M), same output shape.
- General cost of a conv = (output spatial H·W) · (out_ch) · (in_ch) · (kernel area). The expensive operator is the big kernel; its cost is linear in in_ch. Insert a cheap **1×1 conv to cut in_ch** before each big conv, and a 1×1 **projection** after the pooling path to cap its channels.
- **Why 1×1 specifically:** from NiN — a 1×1 conv is a cross-channel parametric pooling / per-pixel MLP; it linearly recombines channels at fixed spatial location, i.e. a learned channel projection. Cheap (no spatial kernel area). Dual-purpose: also carries a ReLU → adds a nonlinearity.
- Justification framed via **embeddings**: even low-dim embeddings hold lots of info about a sizeable patch; but embeddings are dense/compressed and compressed info is harder to model. So **keep representation sparse at most places** (Arora's condition) and **compress only when signals must be aggregated en masse** — i.e. apply 1×1 reductions right before the expensive 3×3/5×5.
- → **Inception module with dimension reduction:** four branches —
  (1) 1×1;
  (2) 1×1 reduce → 3×3;
  (3) 1×1 reduce → 5×5;
  (4) 3×3 max-pool → 1×1 projection.
  All convs use ReLU. Outputs concatenated.

## GoogLeNet incarnation (Table 1) — exact numbers
- Stem (kept "traditional conv" for memory efficiency, not principle): conv 7×7/2 (64) → maxpool 3×3/2 → LRN → conv 1×1 (64) → conv 3×3 (192) → LRN → maxpool 3×3/2.
- 9 Inception modules: 3a,3b | maxpool | 4a,4b,4c,4d,4e | maxpool | 5a,5b. (each module "2 layers deep" → 22 parametric layers; ~100 building blocks total.)
- Inception args (in_ch, 1×1, 3×3reduce, 3×3, 5×5reduce, 5×5, poolproj):
  3a(192,64,96,128,16,32,32) 3b(256,128,128,192,32,96,64)
  4a(480,192,96,208,16,48,64) 4b(512,160,112,224,24,64,64) 4c(512,128,128,256,24,64,64) 4d(512,112,144,288,32,64,64) 4e(528,256,160,320,32,128,128)
  5a(832,256,160,320,32,128,128) 5b(832,384,192,384,48,128,128)
- Head: **global average pooling 7×7 → 1×1×1024** (from NiN), then **dropout 40%**, then **one linear 1024→1000** + softmax.
- Original used **LRN** (local_size 5, alpha 1e-4, beta 0.75), **no BatchNorm** (BN came after, early 2015). torchvision modernizes with BN (and a known 5×5→3×3 bug in branch3).

### Design decision → why
| Decision | Why this, not the alternative |
|---|---|
| Global average pooling head, not big FC | NiN: GAP has **no parameters** → kills overfitting (AlexNet's FC head was the bulk of its 60M params) and is a structural regularizer; averaging each feature map → class-confidence vector. Gave +0.6% top-1 over FC. (Extra linear layer kept only for easy fine-tuning to other label sets — convenience.) |
| Dropout still kept (40%) even after removing FC | GAP removes most params but dropout on the 1024-vector still measurably helps — empirically remained "essential." |
| 1×1 / 3×3 / 5×5 only | Cover local→spread clusters at increasing scale; odd sizes avoid patch-alignment issues; "convenience not necessity." |
| Parallel pooling branch in every module | Pooling is essential in SOTA convnets → a parallel pool path should help too; concatenated as a 4th branch. |
| 1×1 reductions before 3×3/5×5; 1×1 proj after pool | Cap the quadratic compute blow-up (120M→12M example); keep within 1.5B-MAC budget; "compress only when aggregating en masse." ReLU on them = bonus nonlinearity. |
| Concatenate branches (multi-scale) | Process visual info at several scales in parallel, then let the next stage abstract from all scales at once. |
| 3×3/5×5 ratio rises with depth | Higher layers = higher abstraction = less spatial concentration → more spread-out clusters → bigger kernels. |
| Two auxiliary classifiers at 4a, 4d, weight 0.3, discarded at test | 22 layers deep → worry that gradients can't propagate back effectively (no BN/residuals yet). Mid-network features should already be discriminative (shallow nets do well). Aux heads: (a) inject gradient signal deep in the net (combat vanishing gradient), (b) encourage discriminative low/mid features, (c) extra regularization. Discounted by 0.3 so they don't dominate the real objective. Removed at inference (pure training aid). |
| Aux head structure: avgpool 5×5/3 → 1×1 conv 128 → FC 1024 → dropout 70% → FC 1000 softmax | Small cheap classifier; 1×1 conv reduces channels before FC; heavy dropout (0.7) because it's a small regularizing head. |
| Keep stem as plain convs (no Inception in low layers) | "Technical reasons / memory efficiency during training" — infrastructural, not principled. |
| 1.5B-MAC budget | Deployability on modest/embedded hardware; forces efficient resource distribution rather than indiscriminate growth. |
| LRN (original) | Standard post-AlexNet normalization of the day (BN didn't exist yet). |
| Async SGD, momentum 0.9, LR ×0.96 every 8 epochs, Polyak averaging | DistBelief distributed training; standard momentum (Sutskever 2013); Polyak/Juditsky averaging for a smoother final model. |

## Load-bearing ancestors (elaborate, not name-drop)
- **LeNet-5 (LeCun 1989/1998)** — the template CNN: stacked conv (+ optional contrast-norm + maxpool) then FC. Used random/sparse connection tables in feature dim to break symmetry. GoogLeNet's name is homage to it.
- **AlexNet (Krizhevsky 2012)** — deep CNN wins ImageNet 2012 (16.4% top-5). ReLU, dropout, GPU dense conv, LRN, heavy FC head (most of 60M params). Reverted to full feature-dim connections to exploit GPU dense matmul. Gap: huge params → overfitting risk + compute cost; FC head dominates params.
- **ZFNet / Zeiler-Fergus 2014** — visualizing convnets; tuned AlexNet by enlarging layers (layer size). Trend: bigger layers.
- **OverFeat (Sermanet 2013)** — bigger layers, multi-scale, detection/localization with one convnet. Trend: width + multi-scale.
- **Network-in-Network (Lin et al. 2013)** — the direct parent. (i) mlpconv: replace linear conv filter with a tiny per-patch MLP = stack of **1×1 convs** + ReLU → richer cross-channel nonlinear features, drops into standard pipelines. (ii) **Global average pooling** replacing FC: no params, structural regularizer. GoogLeNet uses both heavily but repurposes 1×1 mainly as **dimension reduction**, which NiN did not emphasize. Gap NiN leaves: still a single-scale stack, no compute-budget story, no explicit multi-scale module.
- **Arora et al. 2013** — sparsity theory (above): optimal sparse topology by correlation-clustering layer by layer. Gap: hardware can't do sparse; needs dense approximation.
- **Serre et al. 2007** — neuroscience model of primate visual cortex: a series of **fixed Gabor filters of multiple sizes** to handle multiple scales — the multi-scale-processing intuition. Gap: fixed 2-layer model; Inception *learns* all filters and repeats the module many times.
- **R-CNN (Girshick 2014)** — detection: region proposals (selective search / low-level cues) + CNN classifier per region. Used as the detection pipeline (out of reasoning scope — proposed-method evaluation).
- **Hinton dropout 2012; Sutskever momentum 2013; Polyak averaging 1992; DistBelief Dean 2012** — training machinery.

## Code grounding
- Canonical: torchvision `googlenet.py` (BasicConv2d, Inception(in, 1x1, 3x3red, 3x3, 5x5red, 5x5, poolproj) four-branch concat, InceptionAux, GAP→dropout→fc). NOTE: torchvision adds BatchNorm (not in original) and has a documented branch3 bug (5×5 implemented as 3×3). BVLC caffe `bvlc_googlenet/deploy.prototxt` = faithful original (LRN, no BN, true 5×5).
- My final code: faithful to original architecture (true 5×5, LRN optional, dropout 40% main / 70% aux, two aux heads, GAP head) but written in clean PyTorch grounded in the torchvision structure.

## In-scope vs out-of-scope
- IN: all motivation/architecture derivations, the sparsity→dense bridge, the cost math, every module/head/aux design choice.
- OUT (proposed-method evaluation): ILSVRC'14 classification results table (6.67% top-5), crop/ensemble ablations, detection results.

## Phase 2.6 Codex gate — status
Codex companion located at ~/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs.
Ran in FOREGROUND (blocking). BOTH attempts (gpt-5.5 xhigh and high) failed with:
"You've hit your usage limit ... try again at 4:27 AM."
No second-reviewer runtime available → per SKILL 2.6, fell back to careful MANUAL re-derivation of every equation + ran both code blocks end-to-end in a torch venv:
- conv cost H·W·Cin·Cout·k^2 ✓; α^2 width scaling ✓
- naive 5x5 = 28·28·192·32·25 = 120,422,400 ≈1.2e8 ✓
- reduced = 2,408,448 + 10,035,200 = 12,443,648 ≈1.2e7, ~10x ✓
- all 9 Inception channel sums match Table-1 output sizes ✓; cross-block in/out chain consistent ✓
- aux fc1 in = 128·4·4 = 2048 ✓
- answer.md code: params≈13.4M, train→(3 logits), loss.backward() ok, eval→[B,1000] ✓
- reasoning.md code: identical behavior, mutually consistent ✓
- scaffold purity (no method names / no "reference implementation"), 5 sections, no markdown headers in reasoning, no CJK, no proposed-method results, BN/residual only mentioned as not-yet-existing tools (in-frame) ✓
