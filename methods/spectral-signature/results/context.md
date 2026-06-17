## Research question

A user wants to train an image classifier but does not control the provenance of the training
set. Modern vision models are data-hungry, so the data is often scraped or sourced from a third
party that cannot be fully vetted. An adversary who can inject a small number of training examples
can mount a *backdoor* (Trojan) attack: pick a fixed trigger perturbation (a small pixel patch, a
blended pattern), stamp it onto a handful of images, relabel those images to an attacker-chosen
target class, and slip them into the training set. After training, the model behaves normally on
clean inputs — its test accuracy is indistinguishable from a model trained on clean data — but it
classifies *any* input carrying the trigger as the target class. The attack is dangerous precisely
because it is invisible from the outside: the validation accuracy looks fine, so nothing in the
ordinary training/evaluation loop flags it.

The precise goal is a defense that, given the (untrusted) training set and the trained model,
identifies and removes the corrupted examples so that the user can retrain on a filtered set while
discarding little clean data. The hard part is that the corrupted points are a small fraction of one
class, they are correctly labeled *as far as the trigger-target rule is concerned*, and at the pixel
level they differ from clean images of the target class only by a tiny perturbation. A solution has
to find a small, deliberately camouflaged sub-population hiding inside a much larger, high-variance
population.

## Background

By this time deep nets are deployed in increasingly security-sensitive settings, and the security
of the *training pipeline* — not just test-time robustness — has become a concern. Two related
threats frame the landscape:

- **Adversarial examples** (Goodfellow, Shlens, Szegedy 2014; Madry et al. 2017) perturb a *test*
  input within a small `L_p` ball to flip its prediction. A model adversarially trained to be
  robust to `L_p` perturbations of radius `eps` would, in principle, also resist any trigger that
  lives inside that ball. But a backdoor trigger that sets even a single pixel to an arbitrary
  value lies far outside any reasonable `L_p` ball, so test-time robustness does not cover it.
- **Data poisoning** (Biggio, Nelson, Laskov 2012; Xiao et al. 2015) injects corrupted *training*
  points. The classic poisoning goal is to *degrade* the model's generalization accuracy.
  Defenses for that regime — certified defenses via convex outlier removal (Steinhardt, Koh,
  Liang 2017), influence functions to trace a prediction back to influential training points
  (Koh, Liang 2017) — are built around the assumption that the poison *hurts test accuracy*.

The backdoor threat (Gu, Dolan-Gavitt, Garg 2017; Chen et al. 2017) is qualitatively different:
the poison does *not* degrade clean test accuracy, so accuracy-degradation defenses have nothing
to latch onto. A small poisoned slice can create targeted trigger behavior while leaving ordinary
clean evaluation essentially unchanged. The poison is "hidden" because the model only deviates in
the presence of the trigger.

A separate and load-bearing thread is **high-dimensional robust statistics**. The core problem
there: estimate the mean of a distribution `D` from samples, when an `eps`-fraction of the samples
have been replaced by adversarial outliers `W`. Naive approaches (coordinate-wise medians,
discarding points far from the sample mean) lose accuracy that grows with the dimension. A line of
work (Diakonikolas, Kamath, Kane, Li, Moitra, Stewart 2016, 2017; Lai, Rao, Vempala 2016;
Charikar, Steinhardt, Valiant 2017) showed how to get *dimension-independent* error by working
through the **covariance spectrum**. The key structural fact in that literature: a small
sub-population whose mean is shifted away from the bulk inflates the variance of the mixture in
the *direction of the shift*. Concretely, if `F = (1-eps) D + eps W` is the mixture of a clean
distribution `D` (mean `mu_D`) and an outlier distribution `W` (mean `mu_W`), and `Delta = mu_D -
mu_W`, the mixture covariance contains a rank-one term proportional to `Delta Delta^T`. So
contamination announces itself as an anomalous spike in the spectrum of the empirical covariance,
and the top eigenvector points (approximately) along `Delta`. The robust-statistics filters
exploit this: examine the spectrum, find the high-variance direction, and remove the points that
sit at the extreme of that direction.

A few diagnostic observations about *where* such structure can be found set up the problem:

- At the **raw input / pixel level**, a trigger is a tiny perturbation, so it barely moves the
  class mean, while the variance across natural images of a class is enormous. Comparing the
  projection of clean vs. poisoned images onto the top singular vector of the *pixel* covariance,
  the two sub-populations do not separate — the image variance dominates the small mean shift.
- Weaker per-point statistics computed on the network's **representation** — the `L2` norm of the
  representation vector, or its correlation with a fixed random vector — give *some* separation
  between clean and poisoned points but with substantial overlap, not enough to threshold
  reliably.

## Baselines

A new defense would be measured against, and reacts to, the following prior approaches.

**BadNets — the attack being defended against (Gu, Dolan-Gavitt, Garg 2017).** Choose a target
label and a trigger (e.g. a single bright pixel near a corner, a small checkerboard patch).
Stamp the trigger onto a small set of images, relabel them to the target, and add them to the
training set. Training on this set yields a model that maps the trigger -> target while leaving
all clean behavior intact. This defines the threat model: the attack is simple, the poisoned
fraction is small (a few percent of a class), and the trigger appears on essentially all of the
adversarially-added points and (almost) nowhere else. **Limitation as a *problem* the defender
faces:** the corruption is undetectable from clean accuracy and is camouflaged at the pixel level.

**Certified defenses / outlier removal for poisoning (Steinhardt, Koh, Liang 2017).** Bound the
worst-case test loss an attacker can induce, and remove points that fall outside a feasible set
defined by a convex outlier-removal procedure on the *inputs*. **Gap:** the analysis and the
removal criterion are tied to the test-accuracy-degradation regime; a backdoor leaves clean test
accuracy untouched, so this machinery does not register the poison as harmful, and input-level
outlier removal does not separate triggered images from clean ones.

**Influence functions (Koh, Liang 2017).** Trace a particular prediction back to the training
points most responsible for it, by approximating how the loss on a test point changes if a
training point is up-weighted. **Gap:** designed to explain or detect points that change typical
predictions; backdoored points do not change predictions on typical (untriggered) test examples,
so they leave a weak influence footprint there.

**Robust mean/covariance estimation filters (Diakonikolas et al. 2016, 2017; Lai, Rao, Vempala
2016; Charikar, Steinhardt, Valiant 2017).** Given samples an `eps`-fraction of which are
adversarial, estimate the true mean with dimension-independent error by iteratively inspecting the
covariance: if the empirical covariance has an unusually large eigenvalue, the corresponding
direction is contaminated, so down-weight or remove the points farthest along it; repeat until the
spectrum is benign. The guarantee rests on the mixture-covariance identity above — contamination
must create a spectral spike, and the spike's eigenvector is correlated with the contamination
direction. **Gap:** this whole line was developed and evaluated for *robust statistics / robust
optimization*, where the contamination is generic adversarial noise and the object is an accurate
mean estimate or a robustly-trained model. It had not been connected to backdoor detection, and —
as the diagnostic above shows — applied directly to the *input vectors* of poisoned images the
separation it needs does not materialize, because at the pixel level the trigger's mean shift is
swamped by image variance.

**Weaker representation statistics ( `L2` norm, random-projection correlation ).** Score each
training point by the norm of its representation, or by `|<R(x), r>|` for a fixed random `r`, and
flag the extremes. **Gap:** Cheap, but the separation is weak — clean and poisoned score
distributions overlap heavily, so any threshold either keeps poison or discards a lot of clean
data.

## Evaluation settings

The natural yardsticks already in use:

- **Image classification with a trigger backdoor.** Datasets such as CIFAR-10 (50,000 training
  images, 10 classes, 5,000 per class), CIFAR-100, and FashionMNIST. A fixed trigger pattern
  (BadNets-style pixel patch, or a blended pattern) is stamped onto a small fraction of a chosen
  source class and relabeled to a target class; typical poisoned fractions are a few percent of
  the dataset (e.g. 5% on CIFAR-10), kept low enough that within the target class the poison stays
  a minority sub-population.
- **Architectures**: standard convolutional networks of the era — ResNet variants, VGG with batch
  norm, MobileNet — trained with SGD (momentum, weight decay) and a learning-rate schedule for
  ~100 epochs.
- **The representation**: features read off a designated layer believed to capture high-level
  features, typically the penultimate (pre-logits) layer, extracted for the entire training set
  after training.
- **Metrics**: clean test accuracy (should stay near the clean baseline after defense); attack
  success rate / misclassification rate on triggered test inputs (lower is better); and, as a
  diagnostic on the filter itself, the fraction of true poisoned points that the defense removes.
- **Protocol**: train the (possibly poisoned) model, run the defense to score and remove suspected
  points, retrain from scratch on the filtered set, then measure clean accuracy and triggered
  misclassification on the *retrained* model. The poison count is not known exactly; the defender
  is given an upper bound on the poisoned fraction.

## Code framework

The defense plugs into a fixed harness. The harness trains a victim model on the poisoned training
data, extracts penultimate-layer features for every training example, calls `fit` with those
features and their training labels, calls `score_samples` with the same feature matrix and the
model logits, removes the highest-scoring `1.5 * eps` fraction, and retrains. Everything about
training, feature extraction, the removal budget, and retraining is fixed; the only thing to design
is the scoring rule that turns features into a suspicion score per training point while preserving
the alignment between examples and their training labels. What already exists is the generic
numerical toolkit — arrays, means, and a singular-value / eigen decomposition routine — plus the
empty scoring object.

```python
import numpy as np


class BackdoorDefense:
    """Assigns a suspicion score to each training example. Higher = more
    suspected poisoned. The harness removes the top-scoring fraction and
    retrains. fit() sees features/labels; score_samples() sees features/logits."""

    def __init__(self):
        # any per-fit state the scoring rule needs
        ...

    def fit(self, features, labels, poison_fraction, **kwargs):
        # features: (N, D) penultimate-layer representations of the training set
        # labels:   (N,)  training labels (after poisoning)
        # poison_fraction: an upper bound on the fraction of poisoned points
        # TODO: cache any state needed later by score_samples while preserving
        #       the alignment between examples and their training labels.
        # TODO: the statistic we will design from the fitted training set.
        pass

    def score_samples(self, features, logits):
        # features: (N, D) penultimate-layer representations
        # logits:   (N, C) model logits supplied by the harness
        # returns:  (N,) suspicion scores, higher = more suspicious
        # TODO: turn each example into a scalar suspicion score using the
        #       training-label-aligned state learned in fit().
        pass


# fixed harness the defense plugs into (sketch)
def run_defense(features, labels, logits, poison_fraction):
    defense = BackdoorDefense()
    defense.fit(features, labels, poison_fraction)
    scores = defense.score_samples(features, logits)        # one score per example
    k = int(np.ceil(1.5 * poison_fraction * len(scores)))
    remove = np.argsort(scores)[-k:]
    keep = np.setdiff1d(np.arange(len(scores)), remove)
    return keep                                              # retrain on these
```

The single empty slot is the scoring rule: what to estimate in `fit`, and how `score_samples`
maps each example's representation to a scalar suspicion score under the harness interface.
