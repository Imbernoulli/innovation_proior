# Context: class-imbalanced visual recognition and loss reweighting (circa 2018)

## Research question

Large-scale, real-world image datasets are long-tailed: a small number of "head" classes claim
most of the examples, while the majority of "tail" classes are represented by only a handful of
images each. A convolutional classifier trained on such data with the ordinary cross-entropy loss
is pulled toward the frequent classes — it minimizes average loss by getting the head right and
the tail collapses. The standard remedy is cost-sensitive reweighting: multiply each example's
loss by a per-class factor that compensates for how rare its class is, so that the tail is not
drowned out. The precise problem is which factor to use. The textbook choice — weight inversely
to class frequency, so every class contributes equally — is known to degrade badly on highly
imbalanced real data, and practitioners have fallen back on an unmotivated smoothed variant. What
is needed is a per-class weight that (1) is grounded in some principle rather than a bare
heuristic, (2) is controlled by a single interpretable knob, (3) can be dialed smoothly from "no
reweighting at all" to "full inverse-frequency reweighting" so that one form covers the whole
spectrum of imbalance regimes, (4) adapts to the character of the dataset (coarse vs. fine-grained
classes), and (5) plugs into any loss — softmax cross-entropy, sigmoid cross-entropy, or a
difficulty-aware loss — without changing the model, the optimizer, the sampler, or the evaluation
metric. Closing the gap between "reweight by something" and "reweight by *this*, for *this* reason"
is the problem.

## Background

By this time deep CNNs (AlexNet, VGG, GoogLeNet, ResNet) dominate visual recognition, and their
success rests on large annotated datasets. Curated academic benchmarks (CIFAR, ImageNet ILSVRC
2012, CUB-200) are roughly balanced by class, but datasets harvested from the real world are heavily
skewed: in species-classification data such as iNaturalist a few common species have thousands of
images while thousands of rare species have a dozen. CNNs trained on such distributions perform
poorly on the weakly represented classes (Japkowicz & Stephen 2002; He & Garcia 2008; Van Horn &
Perona 2017; Buda et al. 2018). Two broad families of remedies exist. **Re-sampling** changes the
data the network sees — over-sample the minor classes (repeat or synthesize examples, e.g. SMOTE,
ADASYN) or under-sample the major classes. In deep feature learning both have known costs:
over-sampling duplicates examples, slowing training and inviting over-fitting on the few real tail
images; under-sampling discards examples that the representation needs. **Cost-sensitive
reweighting** instead leaves the data alone and reshapes the loss, assigning higher cost to
examples from minor classes.

The principle under reweighting is cost-sensitive decision theory. Elkan (2001) "The Foundations
of Cost-Sensitive Learning" formalizes it for two classes with a cost matrix `C`: predict the
class of minimum expected cost, so for costs `c10` (false positive) and `c01` (false negative) the
optimal decision is to predict the positive class iff `P(y=1|x) ≥ p*`, with the threshold

```
p* = (c10 - c00) / (c10 - c00 + c01 - c11).
```

Since a standard learner targets the `0.5` threshold, Elkan's Theorem 1 says that to move the
operating threshold to a desired `p*` you rescale the number of negative training examples by
`(p*/(1 - p*)) · ((1 - p0)/p0)` — and, crucially, if the learner accepts example weights, you can
achieve the same shift by *weighting* each negative example by that factor instead of resampling.
This is the theoretical license for per-class loss weighting: a class's weight plays the role of
the cost of misclassifying it. For the imbalance problem the cost is taken to grow with rarity.

Two empirical facts about existing reweighting recipes set up the problem. First, weighting every
class so it contributes equally — inverse class frequency — is widely adopted (Huang et al. 2016;
Wang et al. 2017) but, as work training on large-scale real long-tailed data reports (Mikolov et
al. 2013 for word frequencies; Mahajan et al. 2018 for weakly-supervised image pretraining),
performs poorly when imbalance is extreme. Second, those same practitioners therefore switched to
weighting by the *square root* of inverse frequency, `w_c ∝ 1/sqrt(n_c)`, which empirically does
better. Together they motivate a softer count-based rule without explaining why square-root
smoothing is the right one.

A separate strand reweights by sample *difficulty* rather than class count. Focal loss (Lin et al.
2018), built for the extreme foreground/background imbalance in dense object detection, multiplies
cross-entropy by a modulating factor that shrinks the loss of well-classified examples:

```
FL(p_t) = -(1 - p_t)^gamma * log(p_t),
```

with `p_t` the predicted probability of the true class and `gamma ≥ 0` the focusing strength; it
also has an α-balanced variant `FL(p_t) = -α_t (1 - p_t)^gamma log(p_t)` where `α_t` is a per-class
weight typically set by inverse frequency or tuned by hand.

Finally, the geometric idea that sampling a region with overlapping pieces yields *saturating*
coverage comes from the random covering problem (Janson 1986, "Random coverings in several
dimensions"): cover a large set with a sequence of i.i.d. random small sets and ask for the
expected covered volume. The exact expectation depends on the shape of the small sets and the
dimension of the space and is hard to compute in general.

## Baselines

These are the prior reweighting rules a new per-class weight would be measured against and reacts
to. Throughout, `n_c` is the number of training samples in class `c` and `C` the number of classes.

**Inverse class frequency (Huang et al. 2016; Wang et al. 2017).** Set the per-class weight
inversely to the class count, `w_c ∝ 1/n_c`, normalized (e.g. so the weights sum to `C` or to `1`).
This is the direct reading of Elkan-style cost-sensitive reweighting when every class is declared
equally important: it cancels the empirical class prior, so the reweighted objective behaves as if
the data were balanced. Core math: a sample from class `c` contributes `(1/n_c)·L` to the loss, so
the total contribution of class `c` is `(n_c)·(1/n_c)·L = L`, i.e. every class contributes the same
total regardless of size. **Gap:** under high imbalance the smallest classes receive enormous
weights — a class with five images is up-weighted hundreds of times relative to a class with
thousands — and those few images are dominated by near-duplicates and label noise, so a handful of
unreliable gradients gets amplified to dominate training; the tail weight is far larger than the
information those samples actually carry, and accuracy suffers.

**Smoothed inverse square-root frequency (Mikolov et al. 2013; Mahajan et al. 2018).** Damp the
amplification with `w_c ∝ 1/sqrt(n_c)`, again normalized. By taking a square root the ratio between
the largest and smallest weight is compressed (a 1000:1 count imbalance becomes ~32:1 in weight),
so the tail is boosted without the violent over-amplification of pure inverse frequency, and it
empirically outperforms inverse frequency on real long-tailed data. **Gap:** the exponent `1/2` is
a bare heuristic — there is no derivation for it, nothing ties it to any property of the dataset,
and it is one fixed point with no interpretable parameter; it sits between "no reweighting" and
"inverse frequency" but offers no principled way to move along that axis, and no reason it should
be optimal for one dataset rather than another.

**Focal loss (Lin et al. 2018).** Reweight by *difficulty*: `FL(p_t) = -(1-p_t)^gamma log(p_t)`
down-weights examples the model already classifies confidently (`p_t` near 1) so training focuses
on hard examples, with an α-balanced form carrying an extra per-class factor `α_t`. Core idea: the
modulating factor `(1-p_t)^gamma` is ≈1 for hard examples and ≈0 for easy ones. **Gap:** difficulty
is not class count — a frequent class can have hard examples and a rare class easy ones — and
focusing on hard examples risks fixating on noisy or mislabeled data; moreover its per-class `α_t`
is left as a free hyperparameter or set by inverse frequency, so when the data is class-imbalanced
focal loss still needs a principled per-class weight and does not supply one.

## Evaluation settings

The natural yardsticks already in use for class-imbalanced visual recognition:

- **Artificially long-tailed CIFAR.** Take CIFAR-10 / CIFAR-100 and discard training images per
  class along an exponential profile `n = n_orig · μ^i` (class index `i`, `μ ∈ (0,1)`), leaving the
  balanced test set untouched. The *imbalance factor* is the largest class count divided by the
  smallest; values of 10, 20, 50, 100, 200 are standard. CIFAR-100 (and its 20-superclass
  coarse-label version, CIFAR-20) provides a fine-grained vs. coarse-grained contrast at fixed image
  content. ResNet-32 trained from scratch, batch size 128, SGD with momentum, is the conventional
  backbone here; in the imbalanced setting the learning-rate-drop factor sometimes has to be made
  gentler than the usual ×0.1 to avoid loss and validation error creeping up after the drop.
- **Real-world long-tailed datasets.** iNaturalist 2017 (579,184 images, 5,089 species, imbalance
  ≈435) and 2018 (437,513 images, 8,142 species, imbalance ≈500), using the official train/val
  splits, with ResNets of varying depth.
- **Balanced large-scale.** ILSVRC 2012 (1,281,167 train / 50,000 val, ~uniform) as a robustness
  check under near-balanced class counts.
- Metric: top-1 (and top-5) accuracy / error on the balanced validation or test set, so that tail
  classes count as much as head classes. Protocol holds the data construction, sampler, model,
  optimizer, and metric fixed and varies only the loss reweighting.

## Code framework

The per-class weight plugs into a fixed image-classification harness: a long-tailed training split,
a standard data pipeline (random-crop + horizontal-flip augmentation), a ResNet, SGD with momentum
and a scheduled learning rate, and a weighted loss criterion. Everything in that harness already
exists; the only open hook is a pure function that maps training counts to the vector handed to the
loss.

```python
import torch
import torch.nn as nn


def compute_class_weights(class_counts, num_classes, config):
    """Map per-class training counts to a 1-D weight vector for the loss.

    class_counts : 1-D tensor, class_counts[c] = number of training samples in class c.
    num_classes  : int, C.
    config       : dataset/architecture/imbalance metadata (dataset, arch, imbalance_ratio, ...).

    Returns a length-C tensor `weights` for a weighted loss: example loss for a class-c
    sample is scaled by weights[c]. The computation is pure (no access to images, model,
    or test labels).
    """
    # TODO: choose the count-to-weight rule.
    pass


def train(model, train_loader, class_counts, num_classes, config):
    weights = compute_class_weights(class_counts, num_classes, config)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4)                       # fixed optimizer
    for images, labels in train_loader:                                 # long-tailed train split
        optimizer.zero_grad()
        logits = model(images)                                          # fixed ResNet backbone
        loss = criterion(logits, labels)                               # per-class weighted loss
        loss.backward()
        optimizer.step()
```

The training loop supplies the per-class counts once; the hook returns the single vector that
re-balances the loss.
