# Context: detecting hidden-trigger training poison without trusted data

## Research Question

A deep network is trained on data the model owner did not fully control: crowdsourced labels,
scraped reviews, customer-supplied images, or a model inherited from a third party. An adversary can
inject a small number of training examples that carry a fixed trigger and are labeled as a chosen
target class. After training, the network behaves normally on ordinary test inputs, but whenever the
trigger is present at inference time, a source-class input is mapped to the target class.

The defender wants a procedure that takes only the trained model and the untrusted training set and
identifies the individual poisoned training rows. The binding constraints are severe: no verified
clean reference set, no assumption that the trigger is known, no per-candidate retraining loop, and no
acceptable degradation of ordinary clean accuracy. The defender can trust the training process and
model architecture, but not the data source. The adversary can change a minority of samples and
their labels, but not the architecture, optimizer, or final weights directly.

## Background

BadNets made the threat concrete. In a street-sign setting, a stop sign with a sticker can be
inserted into training as a speed-limit sign; the resulting model keeps high accuracy on ordinary
validation images while misclassifying triggered stop signs. In MNIST-style experiments, poisoning
10% of the training data can already produce a high-success backdoor while clean validation looks
benign. Standard validation is therefore the wrong signal: the attack is designed so that nothing
unusual happens until the hidden trigger appears.

The useful signal is more likely to live inside the trained network than in the raw input. A trigger
that succeeds during training becomes a strong predictor of the target label. A late hidden
representation can therefore amplify the trigger-related feature even when the pixel-level
difference is small. This representation-space view is also the lesson of Spectral Signatures: raw
input norms and correlations can overlap heavily, while learned representations can expose a
corrupted subpopulation.

There is still a geometric obstacle. Late-layer activations can be tens or hundreds of thousands of
coordinates after flattening. In such high dimensions, Euclidean distances lose contrast, so direct
distance-based grouping is unreliable. Any detector that uses activation geometry needs a controlled
low-dimensional representation first. PCA and ICA are natural candidates: PCA preserves
high-variance directions, while ICA searches for independent non-Gaussian components.

## Baselines

**Spectral Signatures (Tran, Li, Madry, 2018).** For each label, collect learned representations
`R(x_i)`, center them by the class mean `R_hat`, stack the centered rows into a matrix `M`, take the
top right singular vector `v`, and score each example by
`tau_i = ((R(x_i) - R_hat) . v)^2`. The defense removes the examples with the top `1.5 * epsilon`
scores and retrains. Its population condition is that, for a mixture
`F = (1 - epsilon) D + epsilon W` with covariances bounded by `sigma^2 I`, the means satisfy
`||mu_D - mu_W||_2^2 >= 6 sigma^2 / epsilon`; then the clean and poison distributions are
`epsilon`-spectrally separable. The wall is that the detector is a one-direction outlier score: it
assumes the poisoned group is exposed by the top singular direction and does not directly return a
discrete subpopulation structure.

**Fine-Pruning (Liu, Dolan-Gavitt, Garg, 2018).** Run clean validation inputs through the model,
identify neurons dormant on clean data, prune them until a tolerated accuracy drop is reached, and
fine-tune. This can repair a model, but it requires a trusted clean set, can trade off clean accuracy,
and does not identify which training samples were injected.

**General outlier, provenance, and influence defenses.** Certified or online outlier defenses need a
trusted clean distribution or become difficult to apply to DNNs. Provenance and influence methods
measure each point's effect on performance, which can require retraining at dataset scale and can
miss backdoors because ordinary performance is intentionally preserved.

## Evaluation Setting

The controlled evaluation uses poisoned versions of MNIST, LISA traffic signs, and Rotten Tomatoes
movie reviews. The image attacks follow the BadNets pattern: add a trigger to source-class samples,
label them as the target class, and append the modified samples. MNIST uses a small CNN and a
bottom-right pixel pattern; LISA uses cropped 32 x 32 signs, a VGG-style classifier, and a
post-it-like trigger that maps stop signs to speed-limit signs. The text case appends a signature to
positive reviews and labels them negative.

Ground truth is available only for evaluation. The main metrics are per-class accuracy and F1 for
the clean-vs-poison assignment, together with the post-repair behavior of the model. A natural
negative control is to run the same kind of geometric analysis on flattened raw inputs rather than
on learned representations; if raw-input geometry fails while representation geometry succeeds, that
supports the premise that the trained model has made the hidden signal visible.

## Code Framework

The detector plugs into a poison-filtering interface. The fixed machinery trains or receives the
suspect classifier, stores the untrusted training data and labels, and expects the defense to return a
report plus one clean/poison decision per training example. The open design slot is the detector
inside `detect_poison`.

```python
import numpy as np


class PoisonFilteringDefence:
    """Framework hook for a training-set poison detector.

    The detector must return `is_clean`, where 1 means keep the training
    example as clean and 0 means treat it as poison. The framework can then
    remove or relabel the flagged rows and retrain/repair the model.
    """

    def __init__(self, classifier, x_train, y_train):
        self.classifier = classifier
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)

    def detect_poison(self, **kwargs):
        # TODO: design the detector using only the suspect model and suspect data.
        # Return: (report: dict, is_clean: list[int] with 1=clean and 0=poison)
        raise NotImplementedError


def repair_from_filter(defence, **kwargs):
    report, is_clean = defence.detect_poison(**kwargs)
    is_clean = np.asarray(is_clean, dtype=bool)
    filtered_x = defence.x_train[is_clean]
    filtered_y = defence.y_train[is_clean]
    return report, filtered_x, filtered_y
```
