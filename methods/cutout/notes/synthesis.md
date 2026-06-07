# Cutout synthesis (Phase 1.5)

arXiv 1708.04552 (verified). DeVries & Taylor 2017. Canonical impl: uoguelph-mlrg/Cutout (util/cutout.py).

## Pain point
Modern CNNs (tens-hundreds of millions of params) overfit. Two standard regularizers: data augmentation (flip/crop/color) and dropout (zero hidden activations -> break co-adaptation). But: dropout is much LESS effective in conv layers than FC layers, because (a) conv layers have far fewer params (need less reg) and (b) neighboring pixels carry near-identical info, so dropping one pixel/activation barely removes the information — the neighbors still pass it on. So conv-dropout just makes feature detectors noise-robust, no model-averaging effect. Dropout variants (SpatialDropout = drop whole feature maps; max-drop = drop maximal activation) try to fix this but LOSE their advantage once batch norm or data augmentation is present — plain dropout wins again. Question: a regularizer that survives BN+augmentation and works on conv nets.

## The diagnostic about pixel correlation
The key motivating fact: in images, neighboring pixels are highly correlated, so per-pixel input noise (Bernoulli erasing, denoising-AE style) is weak — the value is recoverable from neighbors. To actually remove information you must remove a *contiguous region*.

## Load-bearing ancestors
- **Dropout (Hinton 2012; Srivastava 2014):** zero hidden activations w.p. p; test-time scale by p (or inverted). Bagging/co-adaptation story. Strong in FC, weak in conv.
- **SpatialDropout (Tompson 2015):** drop entire feature maps (channels) -> handles neighbor-correlation in feature space. But feature-level and per-map.
- **max-drop (Park & Kwak):** drop maximal activation per map/channel. Worse than dropout under BN.
- **Denoising autoencoders (Vincent 2010):** corrupt input (erase random pixels), reconstruct -> features. Per-pixel corruption = local.
- **Context encoders (Pathak 2016):** erase a large contiguous region, reconstruct -> forces GLOBAL understanding, higher-level features. The "remove a contiguous region forces context" precedent (but for self-supervised representation learning, not supervised classification).
- **Data augmentation lineage (LeCun affine; Bengio occlusions/scratches; Krizhevsky AlexNet flip/crop/PCA color):** Bengio's "occlusions" overlay scratches; closest to cutout but partial overlay vs full zero-mask.

## Derivation (insight-before-method)
Want dropout's benefit in conv nets, surviving BN+aug. Dropout fails in conv because neighbor pixels are redundant; dropping one pixel removes ~nothing (recoverable). To genuinely remove information from the input -> propagate through ALL feature maps -> network must use context, you must (1) drop at the INPUT not intermediate features (so the removed content is absent from every downstream map, unlike per-map dropout where it survives in other maps), and (2) drop a CONTIGUOUS region not scattered pixels (so neighbors can't fill it in). That's "cutout": zero a contiguous square of the input. It's dropout-in-input-space with a spatial prior (the same way CNNs add a spatial prior to MLPs). Closer to data augmentation (generates novel occluded images) than to noise.

Motivation framing: object occlusion is common in vision; simulate it -> network robust to occlusion + uses full image context, not just one telling feature (like a child told a wheels-occluded car is still a car).

## Method details (verified against §Cutout, §Implementation Details)
- Fixed-size ZERO mask at a random location per image per epoch.
- SQUARE patch: found SIZE matters more than SHAPE, so square for simplicity. One knob: side length L.
- Center = uniformly random pixel (y,x) ~ Unif over image. Mask = square of side L around center, CLIPPED to image -> patch may stick out past borders.
- **Allowing the patch to lie partly outside borders is CRITICAL** to good performance: ensures some examples keep a large visible portion (not always-occluded). Alternative that matches: constrain patch fully inside but apply cutout only w.p. 50% (sometimes unmodified image). Both work because the network must see clean/lightly-occluded images too.
- NO test-time rescaling (unlike dropout). Test = normal.
- Dataset must be NORMALIZED to ~zero mean so the zero-mask ≈ mean, not distorting batch stats.
- Applied on CPU during data loading -> effectively free.
- n_holes = 1 typically.
- Original "targeted" version: upsample max-activated feature map, threshold at mean -> binary mask over salient region. Worked well but random fixed-size square performed EQUALLY -> dropped the targeting for simplicity.

## Why removing-region-from-all-feature-maps matters
Other dropout variants treat each feature map individually -> a feature removed from one map survives in others -> noisy inconsistent representation -> only teaches noise-robustness. Cutout removes content at input -> absent from EVERY subsequent map -> the network genuinely cannot see it -> must use context. This is the structural difference from SpatialDropout/max-drop.

## Hyperparameters (settings, used to ground answer.md "typical")
- CIFAR-10: L=16; CIFAR-100: L=8 (more classes -> smaller patch, context less useful, need fine detail). SVHN: L=20. STL-10: 24 (no aug) / 32 (aug).
- Grid search over L; accuracy parabolic in L (rises to optimum then falls below baseline).
- Train: WRN-28-10, 200 epochs, batch 128, SGD Nesterov mom 0.9, wd 5e-4, lr 0.1 /5 at 60/120/160. ResNet18 too. CIFAR aug = pad4-crop + hflip.

## Code (verified against util/cutout.py)
class Cutout(n_holes, length): in __call__, h,w from img; mask=np.ones((h,w)); for each hole: y=randint(h),x=randint(w); y1=clip(y-L//2,0,h);y2=clip(y+L//2,0,h);x1,x2 same; mask[y1:y2,x1:x2]=0; mask=torch.from_numpy(mask).expand_as(img); img=img*mask. Inserted into transforms.Compose AFTER ToTensor+Normalize.

## Design-decision -> why
- Input layer (not intermediate): removed content absent from ALL maps -> forces context; vs per-map dropout where it survives elsewhere.
- Contiguous region (not pixels): neighbor pixels are redundant; scattered removal is recoverable -> ineffective.
- Square (not other shapes): size > shape in importance; square = 1 simple knob.
- Random uniform center, patch may exceed borders: keeps some images mostly-visible -> mix of corrupted+clean is essential (alt: 50% application prob).
- Zero fill + zero-mean normalization: zero ≈ dataset mean -> minimal batch-stat distortion; no info added (vs random-noise fill).
- No test-time scaling: it's augmentation (generates images), not multiplicative noise needing expectation correction.
- Fixed size, dropped the targeted/max-activation version: random equalled targeted, far simpler.

## Unsourced: none. All from egpaper_final.tex + util/cutout.py.
