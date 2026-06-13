## Research question

Long-tail image classification: the training split is exponentially imbalanced — a few head classes have thousands of examples, many tail classes have tens — but the test set is balanced and the metric rewards every class equally. Plain cross-entropy on that training split sees the head far more often, so its gradient is dominated by head examples and the classifier collapses tail decision regions; test accuracy on the rare classes craters. The one thing being designed here is the **per-class loss weight** — a vector `w[c]` fed to `nn.CrossEntropyLoss(weight=w)` that re-scales each class's contribution to the gradient to counteract the imbalance. Everything else — the imbalanced dataset construction, the sampler, the model, the optimizer, the schedule, the evaluation metric — is fixed.

## Prior art before the first rung (re-weighting lineage)

The first rung reacts to the standard re-weighting heuristics that precede any principled treatment. These are the ideas the ladder climbs out of.

- **No re-weighting (uniform cross-entropy).** `w[c] = 1` for all `c`. The empirical-risk-minimization default: minimize mean per-example loss. Under exponential imbalance the per-example mean is a per-class mean weighted by class frequency, so the head dominates the gradient and the balanced-test metric — an *unweighted per-class* mean — is systematically mismatched to what training optimizes. Gap: optimizes the wrong average; tail classes are under-served by construction.
- **Inverse-frequency weighting (standard statistical practice).** `w[c] = N / (C · n[c])`, i.e. weight inversely proportional to class size, which exactly re-normalizes the per-example mean into a per-class mean (each class contributes equal total weight). The textbook fix for imbalance. Gap: for an imbalance ratio of 100 the rarest class gets ~100× the weight of the most frequent, so a handful of tail examples — each seen with enormous weight, repeatedly, with heavy augmentation — drive the gradient and the model over-fits them; the raw dynamic range is destabilizing, especially on architectures without skip connections.
- **Square-root inverse frequency (`w[c] ∝ 1/√n[c]`).** A smoothed variant that halves the exponent on the count, shrinking the head/tail weight ratio from `ratio` to `√ratio` (100× → 10×). Used widely as a gentler heuristic. Gap: the √ choice is arbitrary — it is a single global dampening exponent with no account of *how much* a class of size `n` is actually under-represented; it under-corrects in some regimes and over-corrects in others, and offers no principled knob.

The named baselines below all sit on this lineage: they are different mappings from a class count `n[c]` to a loss weight `w[c]`, and the question is which mapping transfers across datasets (CIFAR-10-LT vs CIFAR-100-LT), imbalance ratios (100 vs 50), and architectures (ResNet-32 vs VGG-16-BN).

## The fixed substrate

A standard supervised long-tail training pipeline is frozen and must not be touched. The imbalanced training set is built by exponential decay: class `i` keeps `n_i = n_max · (1/imbalance_ratio)^(i/(C−1))` of its examples (class 0 most frequent, class `C−1` rarest), drawn with a fixed RNG seed. The model is a CIFAR-adapted ResNet-32 (`[5,5,5]` basic blocks, 3×3 stem, global average pool) or a VGG-16-BN (adaptive-pool head), Kaiming-initialised. Training is SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, cosine-annealed over 200 epochs, batch size 128, with `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`. The loss is `nn.CrossEntropyLoss(weight=w)` where `w` is the only thing the experiment controls. The model trains on the long-tail split and is evaluated on the full **balanced** test set; the reported number is the best test accuracy reached over the run. The loop hands the editable function exactly what it needs: the per-class counts, the class total, and a small config (`imbalance_ratio`, `dataset`, `arch`, `total_samples`).

## The editable interface

Exactly one function is editable — `compute_class_weights(class_counts, num_classes, config)` in `pytorch-vision/custom_weighting.py`. It receives `class_counts` (a length-`num_classes` tensor, sorted so class 0 has the most samples), `num_classes`, and a `config` dict, and must return a length-`num_classes` 1-D tensor of non-negative weights consumed as `nn.CrossEntropyLoss(weight=...)`. The computation must be pure: no access to training data, model parameters, or test labels — only the counts and config. Every method on the ladder is a fill of this one contract: a closed-form map from class counts to a weight vector, with a chosen normalization (the baselines all normalize the weights to sum to `num_classes`, so the average per-class weight stays 1 and the effective learning rate is unchanged). Higher weight on a class = larger gradient contribution from its examples.

The starting point is the scaffold default: **uniform weights — no re-weighting.** Each method replaces exactly this function body and nothing else.

```python
# EDITABLE region of pytorch-vision/custom_weighting.py — default fill (no reweighting)
def compute_class_weights(class_counts, num_classes, config):
    """Compute per-class loss weights for imbalanced classification.

    Called after creating the imbalanced dataset, before training begins.
    The returned weights are used as: nn.CrossEntropyLoss(weight=weights).

    Args:
        class_counts: torch.Tensor [num_classes] — training samples per class
            (class 0 has the most samples, class C-1 the fewest).
        num_classes:  int — 10 for CIFAR-10, 100 for CIFAR-100.
        config: dict with keys imbalance_ratio (float), dataset (str),
            arch (str), total_samples (int).

    Returns:
        torch.Tensor [num_classes] — per-class weights for CrossEntropyLoss.
            Higher weight = more emphasis on that class during training.
    """
    # Default: uniform weights (no reweighting)
    return torch.ones(num_classes)
```

## Evaluation settings

Three settings spanning architecture and imbalance regime, each on one fixed seed (42): **ResNet-32 on CIFAR-10-LT** at imbalance ratio 100 (10 classes, head 5000 / tail 50), **ResNet-32 on CIFAR-100-LT** at ratio 100 (100 classes, head 500 / tail 5), and **VGG-16-BN on CIFAR-100-LT** at ratio 50 (100 classes, head 500 / tail 10). The metric on all three is best balanced-test top-1 accuracy (%, higher is better). A weighting rule must produce numerically stable weights compatible with cross-entropy and must not change the dataset construction, sampler, architecture, optimizer, or metric.
