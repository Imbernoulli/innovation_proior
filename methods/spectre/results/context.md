## Research Question

A user has trained an image classifier on a training set whose provenance is not trusted. A small number of examples may have been modified with an attacker-chosen trigger and assigned to a target label, so the trained model keeps ordinary test accuracy but maps triggered inputs to the target class. The defender has the corrupted training set, the trained model, and an upper bound on the number or fraction of corrupted examples. The defender needs a filter that marks the suspicious training examples before retraining from scratch on the retained set.

The hard part is that the filter cannot assume a clean validation set, a known trigger, or a visually obvious label mismatch. It mainly has access to hidden representations produced by the already-trained network. Those representations may contain the only useful trace of the attack, and that trace can be weak, spread across several trigger variants, or hidden behind ordinary high-variance directions of the target class.

## Background

Backdoor attacks are a training-time poisoning threat: the attacker inserts a small population of examples that share a trigger and a target label. In the representation space of the learned network, these examples often behave like a shifted subpopulation inside the target class. If clean target-class representations have mean `mu_D`, poisoned target-class representations have mean `mu_W`, and the poisoned fraction is `epsilon`, then the class mixture covariance contains the rank-one between-population term `epsilon * (1 - epsilon) * (mu_D - mu_W)(mu_D - mu_W)^T`, in addition to the within-clean and within-poison covariances.

Spectral Signatures turns that observation into a filter: center the representations for a class, take the top singular direction, and score each point by squared projection onto that direction. This is cheap and needs no clean data. Its weakness is also clear from the covariance formula. The top singular direction is the largest-variance direction of the combined class data, not necessarily the direction that separates poison from clean data.

High-dimensional robust statistics supplies a different primitive. Under epsilon-corruption assumptions, robust mean and covariance estimators can recover a Gaussian distribution's clean mean and covariance up to small error when the sample size is large enough. These estimators are expensive and sample-hungry in the original feature dimension, so any practical use has to keep the estimation dimension modest.

Quantum Entropy scoring supplies a separate scoring primitive for outliers. Instead of using only total squared norm or only the top eigenvector, it uses a matrix exponential of the empirical covariance so that high-variance directions receive a soft, temperature-controlled emphasis.

## Baselines

- **Confidence or loss filtering.** Score examples by prediction confidence, loss, or related per-example statistics. This is not tied to the hidden representation geometry and becomes weak once the poisoned model fits both clean and triggered examples confidently.
- **Activation Clustering.** Cluster representations within a class and remove examples from the suspicious cluster. This can work when poisoned examples form a compact, separated cluster, but it is unstable when the poison signature is diffuse or when ordinary class variation dominates the clustering.
- **Spectral Signatures.** Score by the squared projection onto the top singular direction of each class's centered representations and remove the top-scoring examples. This is the strongest immediate predecessor, but it checks a single combined-covariance direction and can miss low-variance or multi-direction poison signatures.

## Evaluation Settings

The setting is image classification with CIFAR-style target-label backdoors. The model is trained on a poisoned dataset, hidden representations are saved for the training examples, a defense removes a fixed budget of suspicious target-label examples, and a fresh model is retrained on the filtered data. The main diagnostics are how many true poisons are removed, clean test accuracy after retraining, and triggered test accuracy after retraining.

The relevant attacks include fixed pixel triggers, multi-way pixel triggers that split the poison population across several trigger variants, periodic triggers, label-consistent attacks, and hidden-trigger transfer-learning attacks. The defense budget follows the prior spectral-signature convention: remove `1.5 * epsilon * n` target-label examples when `epsilon` is a fraction, equivalently `1.5 * eps_count` examples in a target-label file containing `eps_count` poisoned samples.

## Code Framework

The filter plugs into an offline representation pipeline. The training script creates a poisoned experiment, a representation saver writes one hidden-representation matrix per label, the filtering script reads the target-label matrix, and retraining consumes a Boolean mask.

```python
def filter_target_label_representations(reps, poison_count):
    """Return a Boolean keep-mask for one label's representation matrix.

    reps has shape (feature_dim, n_examples) in the filtering stage.
    poison_count is an upper bound on the number of poisoned examples in
    this label. True means keep for retraining; False means remove.
    """
    raise NotImplementedError
```
