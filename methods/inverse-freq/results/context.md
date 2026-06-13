# Context: training classifiers on class-imbalanced data

## Research question

A classifier is trained by minimizing the average loss over a labelled training set. When the
training set is *class-imbalanced* — a few "head" classes hold most of the examples while many
"tail" classes hold only a handful each, the long-tailed distribution that real-world data
almost always follows — this objective misbehaves in a specific way. The average loss is a sum
of one term per example, so it is dominated by the head: a model can drive the average down by
getting the many head examples right while the few tail terms barely register, and the learned
classifier becomes biased toward predicting frequent classes. Yet the situation that actually
matters at deployment is the opposite. The model is evaluated on a *balanced* test set, where
every class counts equally regardless of how rare it was in training — rare-class accuracy is
weighted exactly as much as common-class accuracy. So the quantity being minimized during
training (average loss under the skewed training class frequencies) is not the quantity being
scored at test time (per-class performance under a uniform class prior). The precise problem
is to close that gap **without touching the data construction, the sampler, the model, the
optimizer, or the evaluation metric** — the only open surface is the loss. A solution must make
a standard average-loss training objective serve the balanced target rather than the skewed
empirical prior, use only training-set bookkeeping available before optimization begins, and
remain stable across architectures and imbalance severities. Closing the mismatch between "the
prior the data has" and "the prior the test wants" is the problem.

## Background

The setting is supervised image classification with deep convolutional networks. The standard
training principle is **empirical risk minimization (ERM)**: minimize the average loss

```
R_emp(f) = (1/N) Σ_{i=1}^{N} L(f(x_i), y_i)
```

over the `N` training examples, where `L` is typically the softmax cross-entropy. The average
loss is a Monte-Carlo estimate of an expectation: as `N` grows, `R_emp(f) → E_{(x,y)~p}[L(f(x),y)]`,
where `p` is the distribution the training data is drawn from. Grouping the expectation by class,

```
E_{(x,y)~p}[L] = Σ_{c=1}^{C} p(y=c) · E[ L | y=c ],
```

so the objective weights each class's conditional loss by that class's *prior* `p(y=c)`. For a
training set with `n_c` examples in class `c` and `N = Σ_c n_c` total, the empirical class prior
is `p(y=c) = n_c / N`. When the counts are wildly unequal, these priors are wildly unequal, and
the per-class conditional losses enter the objective in those same wildly unequal proportions.

Two facts about the world frame the problem, both knowable before any fix is chosen. First, the
phenomenon: deep classifiers trained on long-tailed data perform poorly on the weakly
represented classes (Japkowicz & Stephen 2002; He & Garcia 2008; Buda et al. 2018) — this is the
class-imbalance problem, documented across many learners and datasets. Second, the relevant
statistical machinery already exists in a neighbouring literature. In econometrics and
epidemiology one routinely faces *choice-based* / *endogenous-stratified* sampling, where the
class frequencies in the sample do not match the class frequencies in the population of interest
(e.g. a study deliberately collects equal numbers of a rare and a common outcome). Standard
estimation theory has a name and a correction for the resulting distortion — the sample prior
differs from the target prior — and the general principle that an expectation under one
distribution can be evaluated from samples drawn under another is the basic content of importance
sampling. These are pre-existing tools; the open question is how, and with what magnitude, they
apply to the long-tail loss-weighting problem above.

## Baselines

The prior approaches a loss-side correction would be measured against and reacts to. Throughout,
`n_c` is the number of training examples in class `c`, `N = Σ_c n_c`, and `C` the number of
classes.

**Plain cross-entropy / ERM.** Minimize the unweighted average loss `R_emp(f)` above. Every
example contributes equally, so each *class* contributes in proportion to its count `n_c`. Core
behaviour: the objective is exactly the empirical expectation `Σ_c (n_c/N) E[L|y=c]`, i.e. it
optimizes performance under the *training* class prior. **Gap:** on imbalanced data that prior
is far from uniform, so the head classes dominate the sum and the tail classes are nearly
ignored; the resulting classifier is biased toward frequent classes and scores poorly on the
balanced test set, where the tail counts as much as the head. The objective is faithfully
minimized — it is simply minimizing the wrong expectation.

**Re-sampling (over- / under-sampling).** Change the data the network sees so the classes are
balanced: over-sample the minority classes by duplicating or synthesizing examples (e.g. SMOTE,
ADASYN), or under-sample the majority classes by discarding examples, or both (Kubat & Matwin
1997; Japkowicz 2000). Core idea: if the network sees roughly equal numbers per class, the
average-loss objective is no longer dominated by the head. **Gap:** over-sampling duplicates the
*few real* tail images, so in deep feature learning it invites over-fitting on those handful of
images and slows training; under-sampling throws away head examples the representation needs.
It also alters the data pipeline and the number of effective gradient steps — which the present
setting forbids changing — rather than acting purely through the loss.

**Cost-sensitive decision theory (Elkan 2001, "The Foundations of Cost-Sensitive Learning").**
For two classes with a cost matrix `C(i,j)` (cost of predicting `i` when the truth is `j`), the
optimal prediction minimizes expected cost; the optimal decision is to predict the positive
class iff `P(y=1|x) ≥ p*` with

```
p* = (c10 - c00) / (c10 - c00 + c01 - c11).
```

A standard accuracy-maximizing learner targets the threshold `0.5`, so to make it act at a
different `p*` one rebalances the training set. Elkan's Theorem 1: to move the operating
threshold from `p0` to `p*`, multiply the number of negative training examples by

```
(p* / (1 - p*)) · ((1 - p0) / p0),
```

and — the load-bearing remark — *if the learner accepts example weights, the same shift is
achieved by weighting each negative example by that factor instead of resampling*. This is the
theoretical license that a per-class loss weight plays the role of a misclassification cost:
declaring a class costlier is equivalent to up-weighting its examples. **Gap:** it is a two-class
result phrased through a cost matrix, and it supplies the *mechanism* (reweight by class) and the
*equivalence* (weighting ≡ resampling), but for the imbalance problem it does not fix the
magnitude — one still has to decide what the per-class cost should be as a function of how rare
the class is, and Elkan further argues that for Bayesian and decision-tree learners rebalancing
has little effect, leaving open exactly when and by how much weighting helps.

**Prior-correction / weighted-likelihood estimators for distorted sampling (Manski & Lerman 1977;
King & Zeng 2001, "Logistic Regression in Rare Events Data").** When a sample is collected under a
class prior `ȳ` that differs from the target population prior `τ`, a logistic model fit by plain
maximum likelihood is biased for the target. Two consistent corrections exist. *Prior correction*
subtracts `ln[((1-τ)/τ)(ȳ/(1-ȳ))]` from the fitted intercept. *Weighting* (the weighted exogenous
sampling MLE) instead maximizes a weighted log-likelihood

```
ln L_w(β) = w_1 Σ_{Y_i=1} ln π_i + w_0 Σ_{Y_i=0} ln(1 - π_i),
            w_1 = τ / ȳ,   w_0 = (1 - τ) / (1 - ȳ),
```

i.e. it weights each observation by a ratio of the target prior to the observed prior, so the
weighted objective is consistent for the target population rather than the sampled one. Core
idea: re-weighting the likelihood by (target prior)/(observed prior) compensates for a sampling
distortion in the label distribution. **Gap:** these results are developed for binary logistic
regression in the econometric / rare-events setting and assume the target prior `τ` is known or
estimable from external information; they have not been stated for multi-class deep image
classification on long-tailed data, where the "distortion" is the natural class imbalance of the
training set and the target is the balanced test prior.

## Evaluation settings

The natural yardsticks already in use for class-imbalanced visual recognition:

- **Artificially long-tailed CIFAR.** Take CIFAR-10 / CIFAR-100 and trim the per-class training
  counts to an exponential profile, leaving the balanced test set untouched. The *imbalance
  ratio* is the largest class count divided by the smallest; ratios of 50 and 100 are standard.
  ResNet-32 trained from scratch is the conventional backbone; VGG-16-BN (a backbone without
  skip connections) is a contrasting architecture.
- **Fixed training pipeline.** SGD with learning rate `0.1`, momentum `0.9`, weight decay `5e-4`;
  cosine-annealed learning rate over `200` epochs; `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`
  augmentation. Training is on the long-tailed train split; evaluation is on the balanced test
  set, so tail classes count as much as head classes.
- **Metric.** Best test accuracy (%) on the balanced test set, higher is better. The protocol
  holds the data construction, sampler, model, optimizer, and metric fixed, and varies only the
  loss-side adjustment.

## Code framework

The open loss hook plugs into a fixed image-classification harness: a long-tailed training split,
a standard augmentation pipeline, a ResNet (or VGG-BN) backbone, SGD with momentum and a cosine
schedule, and a cross-entropy criterion that accepts a length-`C` vector. Everything else in that
harness stays fixed; the empty slot is a pure function that fills the vector handed to the loss.

```python
import torch
import torch.nn as nn


def compute_class_weights(class_counts, num_classes, config):
    """Map training-set class bookkeeping to a 1-D vector for the loss.

    class_counts : 1-D tensor, class_counts[c] = number of training samples in class c.
    num_classes  : int, C.
    config       : dataset/architecture/imbalance metadata
                   (dataset, arch, imbalance_ratio, total_samples).

    Returns a length-C tensor `weights` for nn.CrossEntropyLoss(weight=...). The computation
    must be pure — no access to the images, the model parameters, or any test label.
    """
    # TODO: choose the loss-side adjustment.
    pass


def train(model, train_loader, class_counts, num_classes, config):
    weights = compute_class_weights(class_counts, num_classes, config)
    criterion = nn.CrossEntropyLoss(weight=weights)                      # fixed loss hook
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4)                       # fixed optimizer
    for images, labels in train_loader:                                 # long-tailed train split
        optimizer.zero_grad()
        logits = model(images)                                          # fixed backbone
        loss = criterion(logits, labels)                               # loss-side adjustment
        loss.backward()
        optimizer.step()
```

The training loop supplies the class bookkeeping once; the hook returns the single vector consumed
by the loss.
