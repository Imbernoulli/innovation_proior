**Problem.** The scaffold default — `RandomCrop` + `RandomHorizontalFlip` — perturbs only geometry; it injects no photometric or rotational diversity, so it under-regularizes models that have more capacity than the small datasets can constrain. Learned augmentation policies fix this but require a separate RL search on a proxy, which is expensive and structurally wrong (augmentation *strength* scales with model/data, and a proxy cannot re-tune it).

**Key idea.** Collapse the policy search space to a search-free rule by removing what the diversity benefit does not need: drop the learned per-transform probabilities and sample N transforms *uniformly* (1/K each); tie all magnitudes to one global value M. Diversity is then forced by construction (K^N combinations) with nothing to learn — and N, M behave like ordinary interpretable hyperparameters.

**Why it works.** The accuracy gain of learned policies comes from *diversity*, not the specific learned probabilities; uniform N-op sampling supplies that diversity directly. Per-transform magnitudes barely move accuracy (~0.1%) and rise together over training, so a single shared M loses almost nothing. The space shrinks from ~10^32 to K^N, small enough that no controller is needed.

**Scaffold edit / hyperparameters.** The task exposes torchvision's built-in `RandAugment`, which already implements uniform-N-ops-at-magnitude-M over its own fixed op table and internal magnitude binning. The literal edit puts it at the *front* of the PIL-space pipeline (before crop/flip and `ToTensor`) with `num_ops=2`, `magnitude=9` — the standard ResNet-on-CIFAR setting and a sensible single point given the harness offers no per-target grid search. On FashionMNIST the harness's resize+channel-repeat wrapper means `RandAugment` sees a grayscale PIL image, so the color-family ops act on grayscale content.

**What to watch.** A real lift over the geometric floor on the CIFAR pairs (largest on CIFAR-100, most regularization headroom); a near-wash on FashionMNIST (simple grayscale, already-high floor). The watch-item: N and M are *fixed* — a single un-tunable operating point with no per-image strength variation — which is exactly what the next rung removes.

```python
# EDITABLE region of custom_augment.py (lines 246-275) -- step 1: RandAugment
def build_train_transform(config):
    """RandAugment augmentation: automated policy before geometric transforms.

    Pipeline: RandAugment(2, 9) + RandomCrop + HFlip + ToTensor + Normalize.
    """
    return transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
```
