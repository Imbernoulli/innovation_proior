# Context: loss functions for classification under label noise (circa 2017-2018)

## Research question

We train a deep classifier `f: X -> R^c` with a softmax output on a dataset whose labels are
partially corrupted. A fraction of the training labels have been flipped to a wrong class, and we do
not know which ones. The classifier is a high-capacity DNN, which is exactly the regime where the
problem bites: such networks can fit *arbitrary* label assignments, including random ones, so when
some labels are wrong the network will happily memorize the wrong targets and its clean-test accuracy
collapses. We want training to be robust to this corruption.

The constraint that shapes the whole problem is *minimal intervention*. We want robustness that comes
purely from the **objective** — the per-example loss summed over the minibatch — leaving the model,
the optimizer, the data pipeline, and the training schedule untouched. Concretely, no confusion-matrix
estimation, no auxiliary clean validation set, no label-cleaning network, no second model, no change
to the architecture. A practitioner should be able to swap one loss function into an existing training
script and get a more noise-tolerant model. The precise goal: a loss that (1) is a drop-in replacement
for the standard classification loss, (2) demonstrably limits how much a confidently-wrong label can
distort training, yet (3) still trains a deep network to high accuracy at a normal rate on a hard
dataset, and (4) carries a theoretical guarantee of noise tolerance rather than being a pure heuristic.

## Background

**Memorization makes label noise dangerous for DNNs.** Zhang et al. (2016, "Understanding deep
learning requires rethinking generalization") showed empirically that standard image networks can
drive training error to zero on data with *completely random* labels — their capacity to memorize is
essentially unbounded. Arpit et al. (2017, "A closer look at memorization in deep networks") refined
the picture: when trained on partially noisy data, a network first learns the simple, generalizable
patterns (largely from the clean labels) and only later overfits the noisy labels. So early in
training the model is implicitly "right" about many corrupted examples; the damage is done when
continued training forces it to memorize the wrong targets. This is the pre-method fact the problem
turns on: the danger is not that noise exists, but that a high-capacity model will eventually fit it.

**Risk minimization and noise tolerance.** Training is empirical risk minimization: with a loss `L`,
the risk of a classifier is `R_L(f) = E_D[L(f(x), y)]`, and we descend it by backprop. With label
noise the available data is drawn from a corrupted distribution `D_η`, giving the noisy risk
`R^η_L(f) = E_{D_η}[L(f(x), ỹ)]`. Following Manwani & Sastry and Ghosh et al., a loss `L` is called
**noise-tolerant** if a global minimizer of the clean risk `R_L` is also a global minimizer of the
noisy risk `R^η_L` — i.e. minimizing the loss you can actually compute (on corrupted labels) lands you
at the classifier you would have gotten from clean labels. The noise model usually assumed is
conditionally independent of the input given the true label, `p(ỹ = k | y = j, x) = η_{jk}`. Noise is
**uniform/symmetric** with rate `η` if a wrong label is equally likely to be any of the other classes:
`η_{jj} = 1 - η`, `η_{jk} = η/(c-1)` for `k ≠ j`. (The label-flip corruption used here — replace the
true class by a fixed permutation of it — is a structured special case of class-dependent noise.)

**Symmetric losses are provably noise-tolerant (Ghosh, Manwani & Sastry 2015; Ghosh, Kumar & Sastry,
AAAI 2017).** Call a loss **symmetric** if, for some constant `C`,

```
sum_{j=1}^c L(f(x), j) = C    for all x and all f.
```

Ghosh et al. proved that any symmetric loss is noise-tolerant under uniform noise with `η < (c-1)/c`,
*independently of the data distribution*. The mechanism is worth seeing because it exposes a useful
design target: preserve the ordering of clean risks after label corruption. Under uniform noise, expand
the noisy risk by conditioning on the true label:

```
R^η_L(f) = E_x E_{y|x} [ (1-η) L(f(x), y) + (η/(c-1)) sum_{i≠y} L(f(x), i) ]
         = E_x E_{y|x} [ (1-η) L(f(x), y) + (η/(c-1)) ( C - L(f(x), y) ) ]
         = Cη/(c-1) + ( 1 - ηc/(c-1) ) R_L(f),
```

where the second line uses symmetry to replace `sum_i L = C`. So the noisy risk is an *affine,
order-preserving* transform of the clean risk: a constant plus a positive multiple of `R_L(f)`,
positive because `1 - ηc/(c-1) > 0` exactly when `η < (c-1)/c`. An affine increasing function has the
same minimizer as its argument, so the clean-risk minimizer also minimizes the noisy risk — noise
tolerance, with no estimation of anything. Under class-dependent noise, analogous sufficient conditions
also require the correct label to remain more likely than any particular wrong label, and the clean-risk
transfer statement uses the zero-clean-risk case (`R_L(f*) = 0`, i.e. separable classes).

**Mean absolute error is symmetric; categorical cross entropy is not.** For a DNN with a softmax
output and one-hot target `e_j`, mean absolute error is `L_MAE(f(x), e_j) = ||e_j - f(x)||_1 = 2 - 2
f_j(x)` (using `sum_k f_k = 1`); up to a constant this is the unhinged loss `1 - f_j(x)` (van Rooyen
et al. 2015). It is symmetric: `sum_{j=1}^c (2 - 2 f_j) = 2c - 2`, a constant. By the theorem above MAE
is therefore noise-tolerant. Categorical cross entropy, `L_CCE(f(x), e_j) = -log f_j(x)`, is neither
symmetric nor bounded, and is sensitive to label noise.

**The diagnostic that sets up the problem: the gradient-weighting behavior of CCE vs MAE.** Reading
off the gradients with respect to the network parameters for a softmax output,

```
d L_CCE / dθ  =  -(1 / f_y) ∇_θ f_y        (cross entropy)
d L_MAE / dθ  =  -∇_θ f_y                    (MAE / unhinged)
```

CCE carries the factor `1/f_y`: a sample whose softmax probability on its given label is small — a
sample the model currently "disagrees" with — gets a *large* gradient. So CCE implicitly puts more
weight on the hard, low-confidence examples. On clean data this is exactly what you want (focus effort
on the difficult points). On noisy data it is the failure mode: a poisoned example, on which the model
is confidently *wrong* relative to the corrupted label, has small `f_ỹ` and therefore a large
gradient, so cross entropy drives the network to memorize precisely the corrupted labels. MAE has no
`1/f_y` factor, so it weights every sample equally — that is the source of its noise robustness.
These are pre-method facts about the two existing losses, read directly from their gradients.

## Baselines

**Categorical cross entropy (CCE), standard ERM.** `L = -log f_y`, minimized by backprop. The default
classification objective; trains DNNs fast and accurately on clean data because its `1/f_y` gradient
weighting concentrates learning on hard examples. **Limitation:** unbounded and nonsymmetric, so it is
not noise-tolerant; the very gradient weighting that helps on clean data causes the network to overfit
confidently-mislabeled examples, and on a high-capacity model continued training memorizes the noise
and clean-test accuracy degrades.

**Mean absolute error / unhinged loss (Ghosh et al. 2015, 2017; van Rooyen et al. 2015).** `L_MAE = 2
- 2 f_y` for softmax outputs, equal up to a constant to the unhinged loss `1 - f_y`. Symmetric, hence
provably noise-tolerant under uniform noise for `η < (c-1)/c`; for class-dependent noise the analogous
condition needs separability plus the correct label being more likely than any particular wrong label.
**Limitation:** because its gradient `-∇_θ f_y` treats every sample equally, it provides no
extra pull on the difficult, low-confidence examples; on large, hard datasets trained with stochastic
gradient methods this makes optimization slow and can leave final accuracy substantially lower than
CCE on hard image datasets. So MAE pays for its robustness with a learning-dynamics collapse that makes
it unreliable as the only objective on challenging data.

**Forward correction with a confusion matrix (Sukhbaatar & Fergus 2014; Patrini et al. 2017).** Model
the noise explicitly: the loss uses `p(ỹ = ỹ_n | x) = sum_i p(ỹ_n | y=i) p(y=i | x)` with an estimated
or known noise transition (confusion) matrix. **Limitation:** the confusion matrix is usually unknown
and must be estimated, and estimates are hard to get right; moreover, with a high-capacity DNN and a
diagonally-dominant confusion matrix, minimizing the corrected cross entropy collapses back to
minimizing ordinary CCE (the network drives each softmax to a one-hot on the noisy label), so even the
*true* matrix can fail to help. It also changes the objective in a way that requires extra machinery
beyond a one-line loss swap.

**Label-cleaning / re-weighting / latent-true-label methods (Veit et al. 2017; Reed et al. 2014;
Tanaka et al. 2018; Jiang et al. 2017 MentorNet; Ren et al. 2018; Northcutt et al. 2017).** A broad
family that gradually replaces noisy labels with model predictions, prunes likely-corrupted samples by
softmax confidence, or learns per-sample weights with an auxiliary network. **Limitation:** these add
real algorithmic machinery — a second network, an EM loop, or, crucially, a small *clean* validation
set to anchor the cleaning/re-weighting. That violates the minimal-intervention goal and is not
available when no trusted labels exist.

## Evaluation settings

The natural yardsticks for noise-robust classification, all pre-method facts (datasets, corruption
processes, and metrics that exist independently of any new loss):

- **Datasets / architectures.** CIFAR-10 and CIFAR-100 (32×32 natural images, 10 and 100 classes) and
  FASHION-MNIST (28×28 grayscale, 10 classes), trained with standard convolutional networks (e.g.
  ResNet-family, VGG-family, MobileNet-family).
- **Synthetic label corruption.** True labels are artificially corrupted at a known rate so the
  experiment is controlled. Two standard regimes: **uniform/symmetric** noise (a corrupted label is
  drawn uniformly from the other classes) and **class-dependent** noise (corruption follows a fixed
  per-class pattern). A simple structured instance is a deterministic **label flip**, replacing the
  true class `y` by a fixed function of it (e.g. `(y + 1) mod c`). Noise rates in the 10-40% range are
  typical. Test labels are always clean.
- **Optimization protocol.** Identical architectures and optimizers across all losses, changing *only*
  the loss function; standard SGD (or Adam) with a cosine/step learning-rate schedule over a fixed
  epoch budget, minibatch size on the order of 128. Networks use ReLU hidden units and a softmax
  output. A fraction of the training data may be held out for validation; only training/validation
  labels are corrupted.
- **Metrics.** Clean-test classification accuracy is the primary quantity. To probe robustness
  directly one also measures how much the model fits the corrupted targets — the fraction of poisoned
  samples on which the model predicts the *wrong (poisoned)* label — so that a good method both raises
  clean accuracy and lowers the degree of noise-memorization.

## Code framework

The loss is a small, swappable object that the fixed training harness calls once per minibatch. The
harness injects label corruption into the training set, runs the model forward, asks this object for a
scalar loss, and backpropagates — none of which the loss object controls. What the object computes from
the logits and the (possibly corrupted) labels is the single open slot.

```python
import torch
import torch.nn.functional as F


class RobustLoss:
    """Per-minibatch training objective for classification under label noise.

    The fixed harness corrupts a fraction of the training labels, runs the model,
    calls compute_loss(...) once per minibatch on the resulting logits and the
    (possibly corrupted) labels, and backpropagates the returned scalar. Only the
    objective computed here is open; the model, optimizer, schedule, and the data
    pipeline are fixed.
    """

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        # logits: (batch, num_classes) raw model outputs for this minibatch.
        # labels: (batch,) possibly-corrupted integer class labels.
        # epoch:  current training epoch (0-indexed).
        # Return: a scalar loss tensor to backpropagate.
        #
        # TODO: replace this default with the noise-robust objective.
        #       The default below is the standard classification loss.
        return F.cross_entropy(logits, labels)
```

The single empty slot is the body of `compute_loss`: the per-example objective, averaged over the
minibatch, that the design will fill in.
