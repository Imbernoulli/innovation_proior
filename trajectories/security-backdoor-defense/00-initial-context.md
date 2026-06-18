## Research question

A model is trained on data whose provenance the user does not control. An adversary injects a small
number of training images that carry a fixed **trigger** (a corner pixel patch, a faint blended
pattern) and are relabeled to an attacker-chosen **target class**. The trained model keeps clean test
accuracy but classifies *any* triggered input as the target — a **backdoor**. The single thing being
designed is a **poisoned-sample scoring rule**: given the penultimate-layer features and logits the
fixed harness extracts after training, assign each training example a suspicion score so the harness
can drop the highest-scoring fraction and retrain on the survivors. Everything else — the victim
architecture, the poison injection, the removal budget, the retrain schedule — is frozen.

## Prior art before the first rung

The first rung reacts to the cheap, no-structure defenses and to the attack itself.

- **BadNets / Blended attacks (Gu, Dolan-Gavitt, Garg 2017, arXiv:1708.06733; Chen et al. 2017,
  arXiv:1712.05526)** — the threat. Stamp a trigger on a small fraction of one class, relabel to the
  target, train as usual. The model maps trigger -> target while clean accuracy looks pristine. The
  corruption is invisible from accuracy and is camouflaged at the pixel level (a tiny perturbation),
  so it must be found in the *learned representation*. Gap (as a problem): nothing in the ordinary
  training loop flags it.
- **Confidence / per-class statistics.** Score each point by softmax confidence on its label, or by a
  per-class confidence z-score; poison tends to be confidently classified into the target. Gap: a
  weak, heavily-overlapping signal — once the model fits the clean data confidently too, poison ranks
  no better than chance, and nothing about the *feature geometry* of the poison sub-population is used.

These set up the move the first rung makes: stop looking at scalar confidence and start looking at the
**geometry of the penultimate features within a class**, where the trigger-amplified poison forms a
sub-population.

## The fixed substrate

A frozen harness owns everything except the scoring rule. It: (1) builds a poisoned training set for a
fixed trigger (full dataset, no subsampling); (2) trains the victim model for 100 epochs (SGD,
`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, cosine annealing); (3) extracts penultimate-layer
features and logits for the *entire* training set; (4) calls the defense; (5) removes the top
`1.5 * eps` fraction of highest-scoring samples (an over-estimate of the poison count, per Tran et al.
2018 Sec. 4.1) and retrains from scratch for 100 epochs on the filtered set; (6) evaluates clean
accuracy and attack success rate on triggered test inputs on the *retrained* model.

The objective is `defense_score = 0.5 * clean_acc + 0.5 * (1 - asr)` (BackdoorBench convention). The
filter-stage `poison_recall` (fraction of true poison removed) is reported as a diagnostic but is
**not** part of `defense_score`: a defense that under-ranks poison but still gets the retrained model
to forget the backdoor can still win, and a defense that ranks poison well but does not remove *enough*
to break the trigger shortcut will see `asr` stay high.

## The editable interface

Exactly one region is editable — the `BackdoorDefense` class in `custom_backdoor_defense.py`. Every
method on the ladder is a fill of this same contract:

- `fit(features, labels, poison_fraction, **kwargs)` — `features` is `(N, D)` penultimate features,
  `labels` the `(N,)` training labels *after poisoning*, `poison_fraction` an upper bound on the
  poison fraction. Cache whatever per-fit state the scoring rule needs, keeping the alignment between
  examples and their training labels.
- `score_samples(features, logits)` — returns 1-D suspicion scores of length `N`, higher = more
  suspicious. `logits` is `(N, C)`; it is offered by the interface but is not necessarily the grouping
  signal.

The starting point is the scaffold default: **score by max softmax confidence** — no feature geometry
at all. Each method on the ladder replaces exactly this class body.

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
- `vgg16bn-cifar100-blend`: VGG-16-BN on full CIFAR-100, Blend trigger, 1% poison fraction (1% so the
  target class stays ~33% poisoned, within the operating regime of per-class SVD/clustering defenses;
  5% would make the target class 83% poison, a degenerate regime).
- `mobilenetv2-fmnist-badnets`: MobileNetV2 on full FashionMNIST, BadNets trigger, 8% poison fraction.

Reported per setting: `clean_acc` (higher better), `asr` (attack success rate on triggered test
inputs, lower better), `poison_recall` (diagnostic), and `defense_score = 0.5*clean_acc + 0.5*(1-asr)`
(the ranking objective, higher better).
