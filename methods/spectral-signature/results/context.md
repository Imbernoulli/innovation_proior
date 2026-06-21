## Research question

Training data for a vision model may come from a source the defender does not fully control. An
attacker can exploit that by adding a small number of examples that carry a fixed trigger and are
assigned to an attacker-chosen target label. After ordinary training, the model can retain normal
accuracy on clean validation data while mapping triggered inputs to the target. The defender needs a
data-cleaning step: identify the likely corrupted training examples, remove them, and retrain without
discarding so much clean data that the classifier is damaged.

## Available ingredients

Several lines of work are already available before designing the defense.

**Backdoor attacks.** BadNets-style attacks show that a network can behave normally on clean inputs
and fail only when a trigger appears. A trigger can be a single bright pixel, a small pattern, or a
physical sticker-like mark, and the attack can remain hidden from validation sets that contain only
clean examples.

**Classical poisoning defenses.** Prior data-poisoning defenses often study attackers that degrade
ordinary test accuracy. Some defenses remove input outliers and then run empirical risk minimization,
or certify a worst-case loss against a family of attacks.

**High-dimensional robust statistics.** Robust mean and covariance estimation asks for reliable
estimates when an adversary corrupts an `epsilon` fraction of samples. A small shifted subpopulation
can be hard to see coordinate by coordinate but can still change the covariance spectrum.

**Learned representations.** A trained classifier supplies intermediate feature vectors, often the
penultimate-layer representation, for every training example. These vectors are available to a
post-training cleaning procedure even when the training images themselves are high-variance and hard
to compare directly.

## Baselines

**Clean validation accuracy.** The standard metric for catching harmful models; often reported
alongside attack success rate in the backdoor literature.

**Input-space outlier removal.** Filtering based on distance from a class centroid in pixel space,
simple sphere or slab filters, or principal directions in raw-pixel space.

**Influence functions.** Gradient-based importance scores that quantify how much each training point
affects predictions on a held-out set.

**Scalar feature statistics.** Representation norm or correlation with a fixed direction in feature
space, used as a per-example suspicion score.

## Evaluation setting

The natural benchmark is image classification with a trigger-backdoor training set. The attacker
chooses a source class, a target label, and a trigger, stamps the trigger on a small number of source
examples, relabels them to the target, and adds them to training. The poison rate is low enough that
the target-label training set still has a clean majority.

A defense is judged by clean test accuracy after retraining, attack success or misclassification rate on
triggered test inputs after retraining, and the fraction of true poisoned training examples removed by
the cleaning step. The defender may know only an upper bound on the poison budget, so the removal
rule should tolerate modest over-removal of clean examples.

Architectures are standard convolutional networks, and the cleaning step runs after an initial training
pass. The defense has access to training labels as used during that pass and to a feature matrix
extracted from the trained model.

## Code framework

The harness supplies feature vectors, training labels, logits for interface compatibility, and an upper
bound on the poison fraction. The scoring object must preserve the alignment between rows of the
feature matrix and rows of the label vector. Higher scores mean "more suspicious"; a separate filtering
step removes the highest-scoring examples and retrains.

```python
import numpy as np


class BackdoorDefense:
    """Assigns a suspicion score to each training example.

    fit() sees penultimate-layer features and the training labels. score_samples()
    is called on the same examples and returns one scalar score per row.
    """

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features)
        labels = np.asarray(labels)
        # Cache any per-label state needed by the score.
        pass

    def score_samples(self, features, logits=None):
        features = np.asarray(features)
        # Return shape (N,), larger means more suspicious.
        pass


def run_defense(features, labels, logits, poison_fraction):
    defense = BackdoorDefense()
    defense.fit(features, labels, poison_fraction)
    scores = defense.score_samples(features, logits)
    k = int(np.ceil(1.5 * poison_fraction * len(scores)))
    remove = np.argsort(scores)[-k:]
    keep = np.setdiff1d(np.arange(len(scores)), remove)
    return keep
```
