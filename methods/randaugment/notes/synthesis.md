# RandAugment synthesis (Phase 1.5)

arXiv 1909.13719 (verified). Cubuk, Zoph, Shlens, Le 2019. Canonical impl: tensorflow/tpu efficientnet (+ augmentation_transforms.py from AutoAugment repo for the op magnitude mapping).

## Pain point
Learned augmentation (AutoAugment, Cubuk 2018) gives big accuracy gains but requires a SEPARATE SEARCH PHASE: train an RNN controller with reinforcement learning (PPO) on a small PROXY task (reduced dataset + small model) to find a policy, then transfer it to the real task. Three problems:
1. The search is expensive and complicated (a second optimization loop, RL controller).
2. Proxy assumption is FALSE here: optimal augmentation STRENGTH depends on model size AND dataset size. A policy tuned on a small proxy model/dataset is sub-optimal when transferred to a big model/big dataset — bigger models/datasets want STRONGER augmentation. So searching on a proxy is structurally wrong for augmentation.
3. Because the policy is fixed by the proxy search, you can't adjust regularization strength to the target model/dataset.
Goal: automated augmentation with NO separate search — fold the augmentation params into the normal training hyperparameters.

## Baseline: AutoAugment (Cubuk 2018, verified against AA PDF)
- Search space: a policy = 5 sub-policies; each sub-policy = 2 operations applied in sequence; each operation = (type from 16 ops, probability from 11 discrete values, magnitude from 10 discrete values). For each image in each minibatch, pick a sub-policy uniformly at random; each op applied w.p. its probability.
- Size: (16 × 10 × 11)^5-per-subpolicy... with 5 sub-policies ≈ (16×10×11)^10 ≈ 2.9×10^32.
- Controller = RNN trained with RL (PPO) to maximize child-network validation accuracy on a proxy task; transfer to target. 30+ parameters in the policy.
- Stochasticity at 3 levels: which sub-policy, per-op probability, direction (e.g. rotate CW/CCW) — this DIVERSITY is the main benefit (per AA/PBA/Fast AA).

## Other baselines / inspirations
- Fast AutoAugment (Lim 2019): density-matching, faster search, but still separate search.
- Population Based Augmentation (PBA, Ho 2019): evolutionary; found optimal magnitude INCREASES during training -> inspired RandAugment to use a fixed magnitude SCHEDULE rather than searching per-transform magnitudes. Still separate search.
- Standard augmentation (flip, crop), Cutout, Mixup: manual.

## The derivation (insight-before-method)
The benefit of learned policies = increased DIVERSITY of examples, not the specific learned probabilities. So: don't learn the probabilities — just ALWAYS sample transforms UNIFORMLY. Replace the learned per-transform probability with a parameter-free rule: at each application, pick a transform uniformly at random from the K=14 transforms (prob 1/K each). This removes ALL the probability parameters and the per-transform selection.
- Apply N transforms in sequence per image -> K^N possible combinations (still diverse).
- Magnitude: AA searches a separate magnitude per transform (on a 0-10 integer scale). PBA showed magnitudes follow a similar schedule. Postulate a SINGLE global magnitude M shared by ALL transforms. Ablation (Fig): changing one transform's magnitude while fixing others moves accuracy by <0.2% -> tying all magnitudes to one M barely hurts.
- Result: 2 interpretable hyperparameters, N (number of ops) and M (global magnitude), both monotone in regularization strength. Search space K^N (RandAugment), vs AA's ~10^32 — reduction of >34 orders of magnitude. So small that plain GRID SEARCH on the TARGET task (no proxy) suffices, and N,M just join the normal hyperparameter search.

## Magnitude schedule ablation (Appendix)
Four ways to set M during training: constant / random (uniform between two values) / linearly increasing / random with increasing upper bound. ALL worked equally well (CIFAR-10 WRN-28-10, ~97.2-97.3%). Chose CONSTANT because it's a single hyperparameter.

## Magnitude scale (verified augmentation_transforms.py, PARAMETER_MAX=10)
Each op maps level M∈[0,10] (or beyond, e.g. EfficientNet uses up to 28/31) to its parameter via float_parameter(level,maxval)=level*maxval/10 or int_parameter:
- Rotate: ±(M/10·30)° ; ShearX/Y: M/10·0.3 ; TranslateX/Y: M/10·(some px) ; Solarize: threshold int(M/10·256) ; Posterize: int(M/10·4) bits ; Color/Contrast/Brightness/Sharpness: M/10·1.8 + 0.1 (i.e. 0.1–1.9 enhancement). AutoContrast/Equalize/Identity take no magnitude.
- 14 transforms: Identity, AutoContrast, Equalize, Rotate, Solarize, Color, Posterize, Contrast, Brightness, Sharpness, ShearX, ShearY, TranslateX, TranslateY.

## The algorithm (verified algorithm_figure.tex)
transforms = [14 ops]
def randaugment(N, M):
  sampled_ops = np.random.choice(transforms, N)   # uniform, with replacement
  return [(op, M) for op in sampled_ops]
Apply the N returned (op, magnitude M) to the image, in addition to default flip/crop (and cutout on SVHN, applied after).

## Hyperparameters found (appendix; used for answer "typical")
- CIFAR-10: N=3. M: WRN-28-2 -> 4, WRN-28-10 -> 5, Shake-Shake -> 9, PyramidNet+ShakeDrop -> 7.
- SVHN: N=3, M~5-9 (cutout applied after RA).
- ImageNet ResNet-50: N=2, M=9. EfficientNet-B5: N=2, M=17. B7: N=2, M=28. (bigger model -> bigger optimal M, confirming strength scales with model size.)
- COCO detection: N=1, M=5-6.
- WRN train: 200 epochs, lr 0.1, batch 128, wd 5e-4, cosine decay.

## Design-decision -> why
- Uniform op sampling 1/K (drop learned probabilities): the benefit is diversity, not the exact probabilities; removes all probability params -> no search needed; K^N still diverse.
- Single global magnitude M (vs per-transform): per-transform magnitude barely matters (ablation <0.2%); PBA showed magnitudes share a schedule. One interpretable knob.
- N = number of ops applied sequentially: the other knob; larger N -> stronger reg.
- Constant magnitude schedule: all schedules tie; constant has fewest hyperparameters.
- Grid search on TARGET task (no proxy): search space K^N is tiny; proxy is wrong because optimal strength scales with model/dataset size.
- Both N,M monotone in regularization strength: human-interpretable, tune to model/dataset.

## Unsourced: none. Algorithm/ops from methods.tex+algorithm_figure.tex; magnitudes from augmentation_transforms.py; AA baseline from AA PDF (2.9×10^32, 5 sub-policies, RNN+PPO, proxy).
