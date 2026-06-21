# Context: detecting hidden-trigger training poison without trusted data

## Research Question

A deep network is trained on data its owner did not fully control: crowdsourced labels, scraped reviews, customer-supplied images, or a model inherited from a third party. An adversary can inject a small fraction of training examples that carry a fixed trigger and are labeled as a chosen target class. After training, the network behaves normally on ordinary test inputs, but whenever the trigger appears at inference time, a source-class input is mapped to the target class.

The defender wants a procedure that takes the trained model and the untrusted training set and identifies the individual poisoned training rows. The defender has no verified clean reference set and no knowledge of the trigger; the defender can trust the training process and architecture, but not the data source. The adversary can alter a minority of samples and their labels, but not the architecture, optimizer, or final weights.

## Background

BadNets make the threat concrete: a stop sign with a sticker inserted into training as a speed-limit sign yields a model that keeps high clean accuracy while misclassifying triggered stop signs. In MNIST-style experiments, poisoning a small fraction of the training data already produces a high-success backdoor while clean validation looks benign. The attack is designed so that nothing unusual happens until the hidden trigger appears.

A trigger that succeeds during training becomes a strong predictor of the target label, so a late hidden representation can amplify the trigger-related feature even when the pixel-level difference is small. Late-layer activations can be tens or hundreds of thousands of coordinates after flattening. Standard linear dimensionality-reduction tools operate on such activations: PCA preserves high-variance directions, while ICA searches for independent non-Gaussian components.

## Baselines

**Spectral Signatures.** For each target class, collect learned representations, center them by the class mean, and score each example by its squared projection onto the top right singular direction of the centered matrix. The score is a single-direction outlier statistic that flags anomalous points.

**Fine-Pruning.** Run clean validation inputs through the model, identify neurons that are dormant on clean data, prune them until the tolerated accuracy drop is reached, and fine-tune.

**General outlier, provenance, and influence defenses.** Certified or online outlier defenses operate against a reference clean distribution. Provenance and influence methods measure each point's effect on performance.

## Evaluation Setting

The controlled evaluation uses poisoned versions of MNIST, LISA traffic signs, and Rotten Tomatoes movie reviews. The image attacks follow the BadNets pattern: add a trigger to source-class samples, label them as the target class, and append the modified samples. MNIST uses a small CNN and a bottom-right pixel pattern; LISA uses cropped 32 x 32 signs, a VGG-style classifier, and a post-it-like trigger that maps stop signs to speed-limit signs. The text case appends a signature to positive reviews and labels them negative.

Ground truth is available only for evaluation. The main metrics are per-class accuracy and F1 for the clean-vs-poison assignment, together with the post-repair behavior of the model. A natural negative control is to run the same kind of geometric analysis on flattened raw inputs rather than on learned representations; if raw-input geometry fails while representation geometry succeeds, that supports the premise that the trained model has made the hidden signal visible.

## Code Framework

The detector plugs into a poison-filtering interface. The fixed machinery trains or receives the suspect classifier, stores the untrusted training data and labels, and expects the defense to return a report plus one clean/poison decision per training example. The open design slot is the detector inside `detect_poison`.

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
