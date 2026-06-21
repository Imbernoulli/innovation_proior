## Research question

A model is trained on data whose provenance the user does not control. An adversary injects a small fraction of training images carrying a fixed trigger — a corner patch or a faint blended pattern — and relabels them to an attacker-chosen target class. The trained model keeps clean test accuracy but classifies any triggered input as the target: a backdoor.

The design task is a poisoned-sample scoring rule. After the fixed harness trains the model, it provides the penultimate-layer features and logits for every training example. The defense must assign each example a suspicion score; the harness drops the highest-scoring fraction and retrains from scratch. Everything else — architecture, poison injection, removal budget, and retrain schedule — is frozen.

## Prior art / Background / Baselines

- **BadNets / Blended attacks** — the threat model. Core idea: stamp a small trigger on a fraction of one class, relabel those examples to the target class, and train normally; the model learns trigger → target while clean accuracy stays high.

- **Confidence / per-class statistics** — a simple defense baseline. Core idea: score each point by softmax confidence on its label, or by a per-class confidence z-score, because poisoned examples are often classified confidently into the target.

## Fixed substrate / Code framework

The harness is frozen. It: (1) builds a poisoned training set for a fixed trigger; (2) trains the victim model for 100 epochs (SGD, `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, cosine annealing); (3) extracts penultimate-layer features and logits for the full training set; (4) calls the defense; (5) removes the top `1.5 * eps` fraction of highest-scoring samples and retrains from scratch for 100 epochs on the filtered set; (6) evaluates clean accuracy and attack success rate on triggered test inputs.

The objective is `defense_score = 0.5 * clean_acc + 0.5 * (1 - asr)`. The filter-stage `poison_recall` is reported as a diagnostic but is **not** part of `defense_score`: a defense that under-ranks poison but still gets the retrained model to forget the backdoor can still win, and a defense that ranks poison well but does not remove enough to break the shortcut will see `asr` stay high.

## Editable interface

Only the `BackdoorDefense` class in `custom_backdoor_defense.py` is editable. The contract is:

- `fit(features, labels, poison_fraction, **kwargs)` — `features` is `(N, D)` penultimate features, `labels` is `(N,)` training labels after poisoning, and `poison_fraction` upper-bounds the poison fraction. Cache any per-fit state needed for scoring.
- `score_samples(features, logits)` — returns 1-D suspicion scores of length `N`, higher = more suspicious. `logits` is `(N, C)` and is offered by the interface but need not be used.

The starting point is the scaffold below, which scores by max softmax confidence. Each method replaces exactly this class body.

```python
# EDITABLE region of custom_backdoor_defense.py — default fill (max-confidence)
import numpy as np
import torch


class BackdoorDefense:
    """Sample-scoring defense for poisoned-example filtering.

    Default baseline: score each sample by its max softmax confidence. Higher
    scores indicate more suspicious (likely poisoned) examples; the fixed
    harness removes the top fraction and retrains.
    """

    def __init__(self):
        self.class_means = None

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features)
        labels = np.asarray(labels)
        self.class_means = {}
        for cls in np.unique(labels):
            cls_features = features[labels == cls]
            self.class_means[int(cls)] = cls_features.mean(axis=0)

    def score_samples(self, features, logits):
        probs = torch.softmax(torch.as_tensor(logits), dim=1).cpu().numpy()
        return probs.max(axis=1)              # max-confidence suspicion score
```

## Evaluation settings

Three benchmark settings, research-scale training, single seed `42`:

- `resnet20-cifar10-badnets`: ResNet-20 on full CIFAR-10, BadNets trigger, 5% poison fraction.
- `vgg16bn-cifar100-blend`: VGG-16-BN on full CIFAR-100, Blend trigger, 1% poison fraction.
- `mobilenetv2-fmnist-badnets`: MobileNetV2 on full FashionMNIST, BadNets trigger, 8% poison fraction.

Reported per setting: `clean_acc` (higher better), `asr` (attack success rate on triggered test inputs, lower better), `poison_recall` (diagnostic), and `defense_score = 0.5*clean_acc + 0.5*(1-asr)` (ranking objective, higher better).
