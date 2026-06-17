# Context: detecting backdoor-poisoned training data without trusted data (circa 2018)

## Research question

A deep network is often trained on data the model owner did not fully control: crowdsourced
labels, scraped reviews, customer-supplied images, or a model inherited from a third party. An
adversary who can inject even a small fraction of training examples can plant a *backdoor*: pick a
*source* class, stamp a fixed *trigger* (a sticker, a pixel patch, a text signature) onto source
images, relabel them as a *target* class, and append them. The trained network classifies clean
inputs correctly — its test accuracy looks normal — but whenever the trigger appears at inference
time it flips the prediction to the target class. The pain is that the malicious behavior is
*hidden*: it only fires on inputs carrying a trigger that is known solely to the adversary, so
validating on a held-out clean test set reveals nothing.

The concrete goal is a procedure that, given only the (possibly poisoned) trained model and the
untrusted training set itself, decides *which individual training examples* are the injected poison
— so they can be removed and the model retrained clean. The hard constraints that rule out the
existing options: it must work **without any verified clean / trusted dataset** (in the supply-chain
threat model the defender cannot assume access to known-good data), it must not require retraining
the model once per candidate point (infeasible at DNN scale), and it must not degrade clean
accuracy, which matters in safety-critical deployments. The adversary controls some fraction of the
training samples and their labels, but not the architecture, the optimizer, or the training
process.

## Background

By 2018 the security of trained models is an active concern. Most prior work targeted *evasion
attacks* (adversarial examples: small, input-specific perturbations that flip a prediction at test
time, e.g. Carlini & Wagner 2017). A separate and arguably more dangerous threat is the *poisoning
attack*, where the adversary manipulates the *training* data. The particularly insidious variant is
the backdoor / trojan attack.

**The attack (BadNets; Gu, Dolan-Gavitt & Garg 2017).** For each target class, take images of a
source class, add a fixed backdoor trigger, label them as the target, and append them to the
training set. The trained net learns two things at once: the genuine target-class features (so
clean target inputs are classified correctly) and the trigger (so any triggered source input is
classified as target). The reported diagnostic is stark: with as little as 10% of the training set
poisoned, the test error on triggered inputs falls below ~1.5% — a small poison fraction is enough
to install a near-perfect backdoor while clean accuracy is untouched. This is the phenomenon a
defense must catch, and it establishes that benign-looking validation cannot detect the threat.

**Why poison hides in input space but surfaces in feature space (Spectral Signatures; Tran, Li &
Madry, NeurIPS 2018).** A diagnostic finding that shapes the whole design space: when a class is
corrupted, its examples are a mixture of a large clean sub-population and a small poisoned one. If
you look at *raw inputs* — l2 norms, correlations with a random vector, correlation with the top
eigenvector of the input covariance — the two sub-populations overlap heavily; the natural variance
of images swamps the difference, so input-level statistics cannot separate poison from clean. But
when the same examples are mapped to the network's *learned representation* (a late hidden layer),
the two sub-populations pull apart into separable distributions. The intuition: the trigger is a
strong, simple predictor of the target label, so the feature extractor is *incentivized to amplify*
it; as that signal is boosted, poisoned points become more and more distinguishable from clean ones
in feature space. The actionable pre-method fact is therefore: do the analysis on hidden-layer
activations, not on inputs.

**Clustering in high dimensions is fragile.** A flattened hidden-layer activation can be tens or
hundreds of thousands of dimensions. It is well established (Aggarwal, Hinneburg & Keim 2001;
Domingos 2012) that in very high-dimensional spaces the Euclidean metric loses contrast — the ratio
of nearest to farthest distances tends to one — so distance-based grouping degrades. Any geometric
analysis of activations must first reduce dimensionality, while retaining the directions that carry
the relevant structure. Two standard reductions are on the table: Principal Component Analysis,
which extracts orthogonal directions of maximal variance, and Independent Component Analysis, which
extracts maximally statistically-independent (non-Gaussian) directions.

**The defender's premise.** The defender has the trained model and the untrusted training set, and
trusts the training *process* (just not the data). A reasonable structural assumption about the
attack: the adversary poisons strictly less than half of any single class — otherwise the "target
class" is mostly trigger images and the attack is trivially conspicuous. So within any class, if a
poison sub-population exists, it is the minority.

## Baselines

**Spectral Signatures (Tran, Li & Madry, NeurIPS 2018, arXiv:1811.00636).** The closest prior
defense and the one that established the feature-space view. For each label, collect the learned
representations R(x_i) of all n examples in that class, center them by the class mean R-hat, and
stack the centered vectors into an n x d matrix M. Take v, the top right singular vector of M (the
top eigenvector of the representation covariance). Score each example by its squared projection onto
that direction, tau_i = ((R(x_i) - R-hat) . v)^2, and remove the examples with the top 1.5*epsilon
scores (epsilon an upper bound on the poison fraction; the 1.5 factor over-removes to raise recall),
then retrain. The method comes with a guarantee: model the class as a mixture F = (1 - eps) D +
eps W of a clean distribution D and a poison distribution W; if their means are separated enough
relative to the variance, ||mu_D - mu_W||_2^2 >= 6 sigma^2 / eps, the mixture is *eps-spectrally
separable* and removing the top-eps fraction along v provably removes the poison (proof by
Chebyshev bounds plus a top-eigenvector-correlation argument). **Gap:** the score is a *single
linear direction* — one projection axis, with a unimodal continuous-outlier score. It implicitly
assumes the poison shows up as a rank-one mean shift that the top singular vector captures. When the
clean class is itself multimodal, or the poison's displacement is not aligned with the single
dominant variance axis, one direction can fail to isolate it, and there is no discrete notion of
"which group each point belongs to."

**Fine-Pruning (Liu, Dolan-Gavitt & Garg 2018, arXiv:1805.12185).** Operates on neurons rather than
on per-sample scores: feed clean inputs through the model, identify neurons that stay dormant on
clean data (suspected of encoding the backdoor), prune them until a tolerated accuracy drop on a
trusted set is reached, then fine-tune. **Gap:** it requires a *trusted, verifiably clean*
validation set to know which neurons are dormant and to bound the accuracy loss; it degrades clean
accuracy by pruning; and it removes the backdoor's machinery without identifying *which training
points* are poison, so it cannot clean the dataset itself.

**Anomaly / outlier-detection defenses (Steinhardt, Koh & Liang 2017; Kloft & Laskov 2010, 2012).**
General certified or online outlier-removal defenses against poisoning. **Gap:** they rely on a
clean trusted dataset to fit the outlier model; without one, effectiveness drops sharply and a
sufficiently strong adversary (poison fraction >= 30%) can craft attacks that defeat them; the
tractable algorithms given were for SVMs, not DNNs.

**Provenance- and influence-based defenses (Nelson et al. 2009; Baracaldo et al. 2017).** Detect
poison by measuring each point's effect on classifier performance. **Gap:** evaluating that effect
requires retraining on the order of the dataset size — infeasible for DNNs — and a backdoor causes
*no* drop in standard performance, so a performance-degradation signal does not fire on it.

## Evaluation settings

The natural yardstick is per-example detection quality with known ground truth in a controlled
poisoning experiment:

- **Datasets and models:** MNIST with a small CNN (two conv + two fully-connected layers); the LISA
  traffic-sign dataset with a deeper convolutional classifier (a VGG-style / faster-R-CNN-based
  model), signs cropped to 32x32 and grouped into a few classes; and the Rotten Tomatoes movie-
  review sentiment dataset with a text CNN. The benchmark task in the harness this method plugs into
  instead uses CIFAR-10/ResNet-20 (BadNets trigger), CIFAR-100/VGG-16-BN (blend trigger), and
  FashionMNIST/MobileNetV2 (BadNets trigger), all trained for 100 epochs with SGD (lr 0.1, momentum
  0.9, weight decay 5e-4) and cosine annealing.
- **Poison construction:** BadNets-style — source-class images plus a fixed trigger, relabeled to a
  target class, at poison fractions such as 1%, 5%, 8%, 10%, 15%, 33% (kept below 50% within the
  target class so the poison stays a minority sub-population).
- **Removal budget:** an over-estimate of the poison count, removing the top 1.5*epsilon highest-
  scoring fraction (following the Spectral Signatures recommendation), then retraining on the
  filtered set.
- **Metrics:** per-class detection accuracy and F1 of the poison-vs-clean decision (poison_recall as
  a diagnostic of the filter), and post-retrain clean test accuracy together with attack success
  rate on triggered test inputs (a defense score weighting clean accuracy against 1 - ASR). A
  natural control to measure against is clustering the *raw inputs* instead of the activations.

## Code framework

The defense plugs into a fixed harness. The harness trains a victim model on the (poisoned) data,
extracts penultimate-layer features and logits for the whole training set, calls the defense to
assign a per-sample suspicion score, removes the top fraction of highest-scoring samples, and
retrains. What is *not* settled is how to turn the per-sample features (and labels) into a suspicion
score — that scoring rule is exactly what is to be designed. The substrate is only generic
numerical machinery that already exists: array operations, a generic clustering primitive, and a
dimensionality reduction primitive. The single empty slot is the scoring rule.

```python
import numpy as np


class BackdoorDefense:
    """Assigns a suspicion score to each training example from its penultimate-layer
    feature vector (higher = more likely poisoned). The harness removes the top-scoring
    fraction and retrains. Only the scoring rule is to be designed; everything around it
    (training, feature extraction, filtering, retraining) is fixed."""

    def fit(self, features, labels, poison_fraction, **kwargs):
        # features: (N, D) penultimate-layer features for the full training set
        # labels:   (N,)  training labels after poisoning
        # poison_fraction: approximate fraction of poisoned points
        # TODO: the per-class analysis we will design, fitted here and used in score_samples.
        pass

    def score_samples(self, features, logits):
        # features: (N, D) penultimate-layer features
        # logits:   (N, C) model logits
        # return: (N,) 1-D suspicion scores; higher means more suspicious
        # TODO: turn the fitted analysis into a per-sample suspicion score.
        pass


# existing fixed harness the defense plugs into
def run_defense(defense, features, labels, logits, poison_fraction, remove_fraction):
    defense.fit(features, labels, poison_fraction)       # fit the scoring rule
    scores = defense.score_samples(features, logits)     # one suspicion score per sample
    n_remove = int(remove_fraction * len(scores))        # top 1.5 * poison_fraction, set by harness
    flagged = np.argsort(scores)[::-1][:n_remove]        # highest-scoring samples
    return flagged                                       # removed, then the model is retrained
```

The harness supplies the features, labels, and logits; `fit` / `score_samples` is where the scoring
rule will live.
