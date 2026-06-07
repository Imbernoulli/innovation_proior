# jigsaw synthesis (grounding notes)

Verified arXiv: 1603.09246 "Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles" (ECCV 2016), Noroozi & Favaro (Univ. Bern). Canonical code: MehdiNoroozi/JigsawPuzzleSolver (Caffe, original); bbrattoli/JigsawPuzzlePytorch (clean PyTorch reimpl — Network + select_permutations.py used for grounding the code structure).

## Pain point / research question
- Supervised features need expensive labels. Self-supervised learning (2015): exploit labels free in the data. Want a pretext task on *single images* that yields features transferring to classification/detection.
- Prior self-supervised: Doersch (context prediction, relative position of 2 patches), Wang&Gupta (tracking-based patch similarity in video), Agrawal (egomotion). Multi-image methods learn invariance (same instance → focus on color/texture similarities, limited intraclass variability). Single-image context-prediction is closest but uses only pairs → relative position between two tiles can be ambiguous.
- Goal: a single-image pretext task that forces learning *object parts and their spatial arrangement* and transfers well.

## Load-bearing ancestors (grounded)
- Doersch et al. 2015 (context prediction): classify the relative position of a second tile w.r.t. a center tile (8 positions on 3×3 grid). Single image. Gap: only 2 tiles seen at once → relative position can be genuinely ambiguous (two similar tiles); doesn't intersect ambiguity across all tiles. Also suffers shortcuts (chromatic aberration, edge continuity). ~4 weeks training.
- Wang & Gupta 2015: triplet from tracking in video; learns patch similarity metric. Gap: multi-image, same-instance → learns low-level similarity (color/texture), not high-level structure.
- Agrawal et al. 2015 (egomotion): siamese net predicts egomotion between 2 frames using odometry labels. Gap: multi-image, invariance to same instance, limited intraclass variability.
- AlexNet (Krizhevsky 2012): the backbone architecture (conv1-5 + fc) that features are transferred to. 61M params.

## The method (grounded; Section 3 + figure + code)
- **Jigsaw puzzle pretext**: crop 225×225 window from image, 3×3 grid of 75×75 cells, sample a 64×64 tile from each cell (random shift → gap between tiles). Shuffle the 9 tiles by a permutation drawn from a *predefined permutation set*, feed to the network, predict the *index* of the permutation (classification over |permutation set| classes). 9!=362,880 possible orderings; use a subset.
- **Context-Free Network (CFN)**: siamese-ennead. Each of 9 tiles → its own AlexNet conv stack (conv1–conv5) + fc6, all 9 rows share weights up to and including fc6. The 9 fc6 outputs are concatenated → fc7 → fc8 → softmax over permutation indices. "Context-free" because each tile is processed independently (receptive field limited to one tile) until fc6; context (cross-tile arrangement) only enters at fc7+. CFN ≈ AlexNet accuracy on ImageNet (57.1 vs 57.4 top-1) but fewer params (27.5M vs 61M); fc6 is 4×4×256×512≈2M vs AlexNet's 6×6×256×4096≈37.5M. During puzzle training, stride 2 in first conv (tiles are small 64×64); for transfer, weights moved to standard AlexNet (stride 4).
- **Probabilistic view**: CFN output = p(S | A1..A9) = p(S|F1..F9) Π p(Fi|Ai), S=tile configuration, Ai=appearance of part i, Fi=feature. Want Fi to be semantic (identify relative position). Failure mode: if only 1 puzzle per image, the net learns Fi → absolute position (p(S|F)=Π p(Li|Fi)), features carry no semantics, just arbitrary 2D position. This is the "shortcut."
- **Permutation set selection (Algorithm 1)**: choose the set by *maximizing average Hamming distance*. Greedy: start from a random permutation; iteratively add the permutation (from all 9!) with maximal Hamming distance to the current set. (Variants: argmin → minimal, uniform → middle.) Set generated *before* training. Findings: larger cardinality → harder task, better transfer; larger average Hamming distance → less ambiguous, better; the minimum Hamming distance also controls ambiguity. Best = trade-off of cardinality vs dissimilarity. Conclusion: "A good self-supervised task is neither simple nor ambiguous." Default ~1000 permutations, max-Hamming.

## Shortcuts → fixes (design decisions, grounded)
- **Absolute-position shortcut**: feed *many* puzzles per image (avg 69 of 1000 configs) with high-Hamming permutations so each tile must go to many positions → can't map appearance→fixed position.
- **Low-level statistics (mean/std of pixels match across adjacent tiles)**: independently normalize the mean and std of each tile.
- **Edge continuity**: leave a random gap between tiles — sample the 64×64 tile from an 85×85 cell → up to 21px gap (3.4 ablation says 85×85→21px; intro/fig say 75×75→avg 11px; the final shortcuts-prevention setting is 85×85, 21px gap).
- **Chromatic aberration** (spatial shift between color channels increasing toward borders, a position cue): (i) crop central square, resize to 255×255; (ii) mix grayscale (30%) and color (70%) images; (iii) spatially jitter color channels by ±{0,1,2} px per tile.
- Ablation (Table shortcut, detection mAP): all three on → 52.6; no gap → 47.7; no normalization → 43.5; no color jitter → 51.1.

## Implementation (grounded)
Caffe, SGD, no batchnorm, batch 256, base lr 0.01, 350K iterations (~2.5 days, 1 Titan X). 1.3M ImageNet images (no labels), 256×256. ~69 puzzles/image avg. Transfer: copy conv weights to standard AlexNet (stride 4), fill fc with Gaussian (mean 0.1, std 0.001), fine-tune.

## Design-decision → why
- Stack-tiles-on-channels rejected: net keys on low-level texture correlations across tiles near boundaries, no global understanding → CFN delays cross-tile computation to fc7.
- Predict permutation index (classification) not absolute tile positions: avoids per-tile position regression that invites the absolute-position shortcut; the index forces joint reasoning over all 9.
- Shared weights up to fc6: forces a single tile-feature extractor (the transferable part); the puzzle-specific reasoning lives only in fc7/fc8 (discarded at transfer).
- Permutation set as the key knob: too few/too-distant perms → task too easy (net overfits ordering, weak features); too many/too-close → ambiguous (similar tiles can't be disambiguated). Max average Hamming distance + moderate cardinality balances.

## Evaluation settings (pre-method, no outcomes)
Pretrain on ImageNet-1K images (no labels). Transfer via pretrain+finetune: PASCAL VOC 2007 classification (Krähenbühl framework), VOC 2007 detection (Fast R-CNN), VOC 2012 semantic segmentation (FCN). ImageNet 2012 classification with layer-locking (lock conv1..k, retrain rest). Metrics: mAP (detection), classification accuracy, mIoU. Image retrieval (pool5 NN). Baselines for comparison: AlexNet supervised, Doersch, Wang&Gupta, Pathak context-encoder, random init.
