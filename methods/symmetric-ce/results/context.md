# Context: learning deep classifiers from label-corrupted data (circa 2017-2019)

## Research question

Modern deep networks have millions of parameters and need large labelled datasets, but
labels collected at scale — by crowdsourcing, web scraping, or surrounding-text heuristics —
are routinely wrong for a sizeable fraction of examples. The problem is to train an
accurate classifier when a portion of the *training* labels have been flipped to incorrect
classes, while the *test* set is clean. A solution should recover a classifier whose accuracy
on clean data is close to what it would be without corruption, without an architectural
overhaul, without auxiliary clean data, and without a known noise-transition matrix.

## Background

The dominant training recipe is empirical risk minimization with the cross entropy (CE) loss.
For a `K`-class problem with one-hot ground-truth distribution `q(k|x)` and softmax
prediction `p(k|x) = e^{z_k} / sum_j e^{z_j}`, the sample-wise CE loss is

```
ell_ce = - sum_{k=1}^K q(k|x) log p(k|x) = - log p(y|x)    (single label y).
```

CE is the maximum-likelihood objective and, via the identity `KL(q||p) = H(q,p) - H(q)` with
`H(q)` constant for a fixed label distribution, minimizing CE is minimizing the KL divergence
`KL(q||p)` from the observed label distribution `q` to the prediction `p`.

Several diagnostic findings about how networks behave under label noise were established before
any robust-loss design. Zhang et al. (ICLR 2017, "Understanding deep learning requires
rethinking generalization") showed DNNs can memorize arbitrary (even random) labels, and that
in practice they fit clean, easy patterns first and only later memorize the wrongly-assigned
labels — a *memorization* effect. Arpit et al. (ICML 2017) corroborated this: networks learn
simple shared patterns early and overfit corrupted labels late. Ma et al. (ICML 2018) framed
the same arc as subspace-dimensionality compression followed by expansion. The standard read
of these results was that label-noise damage is essentially *overfitting* to the noisy labels.

There is also a more refined, per-class picture available from simply watching class-wise test
accuracy during CE training. Even with perfectly clean labels, the per-class accuracy curves
span a wide band throughout training: some classes ("easy") converge fast, others ("hard")
lag — a *class-biased* learning dynamic, attributable to intrinsic differences in how
separable each class's patterns are. When labels are corrupted, this band widens sharply: easy
classes reach high accuracy and then begin to drop, while hard classes plateau far below their
clean-label ceiling. On the clean *portion* of a hard class, the network's average confidence
on the correct class can sit around 0.5 with several percent of mass leaking to
visually-similar classes, and the hard classes contribute far fewer true positives at every
stage.

A separate, theory-side background concerns *which loss functions are inherently tolerant to
label noise*, independent of the architecture or any noise estimate. Ghosh, Kumar & Sastry
(AAAI 2017) formalized noise-tolerance through risk minimization: write the clean risk
`R(f) = E_{x,y} L(f(x), y)` and the noisy risk `R^eta(f) = E_{x,yhat} L(f(x), yhat)` at noise
rate `eta`; `L` is *noise-tolerant* if the global minimizer of `R^eta` is also a global
minimizer of `R`. Their central lemma identifies a sufficient condition: for some constant `C`,

```
sum_{i=1}^{K} L(f(x), i) = C    for all x and all classifiers f.
```

Under this condition, for symmetric/uniform noise of rate `eta`,

```
R^eta(f) = (1-eta) R(f) + (eta/(K-1)) ( C - R(f) )
         = C*eta/(K-1) + ( 1 - eta*K/(K-1) ) R(f),
```

which is an increasing affine function of `R(f)` whenever `eta < (K-1)/K = 1 - 1/K`, so it has
the same argmin as `R(f)` — robustness, distribution-free.

Label smoothing (Szegedy et al. 2016; Pereyra et al. 2017) — replacing one-hot targets
with `(1-epsilon)` on the true class and `epsilon/(K-1)` spread elsewhere — is a known way to
damp over-confident predictions and ease overfitting.

## Baselines

These are the prior approaches a new objective would be measured against and would react to.

**Cross entropy (standard ERM).** Train with `ell_ce = -log p(y|x)` directly on the (possibly
corrupted) labels. Fast, well-conditioned gradients, the default everywhere.

**Mean Absolute Error / `L1` loss (Ghosh et al. 2017).** Use `ell_mae = sum_k |p(k|x) -
q(k|x)|`. For a one-hot label this is `(1 - p_y) + sum_{k!=y} p_k = 2(1 - p_y)`. Summed over
classes it equals the constant `sum_i ell_mae(f(x), i) = 2(K-1)`, independent of `x` and `f`, so
MAE satisfies the Ghosh condition and is provably noise-tolerant for `eta < 1 - 1/K`.

**Generalized Cross Entropy / `L_q` (Zhang & Sabuncu, NeurIPS 2018).** Interpolate between CE
and MAE with a Box-Cox transform of the true-class probability,

```
ell_q = (1 - p(y|x)^q) / q,    q in (0, 1].
```

As `q -> 0` this tends to CE (good gradients, fast convergence); at `q = 1` it is `1 - p_y`,
i.e. MAE up to scale. A single `q` tunes the trade-off, and the loss behaves
like a `p`-weighted MAE that down-weights low-confidence examples.

**Bootstrapping (Reed et al., ICLR-WS 2015).** Replace the target with a convex mix of the
observed label and the model's own prediction: soft variant target `beta*q + (1-beta)*p`, hard
variant uses the one-hot of `argmax p`; then take CE against that target. The idea is that as
the model becomes competent its prediction can correct a wrong label.

**Loss correction with a noise-transition matrix (Forward/Backward, Patrini et al. 2017).**
Multiply predictions (or the loss) by an estimated class-to-class noise matrix `T`.

**Label Smoothing Regularization (Szegedy 2016; Pereyra 2017).** CE against softened targets.
Eases over-confidence and some overfitting.

## Evaluation settings

The natural yardsticks for a label-noise objective:

- **Datasets and corruption.** MNIST (LeCun et al. 1998), CIFAR-10 and CIFAR-100 (Krizhevsky
  2009), and a real-world web dataset, Clothing1M (Xiao et al. 2015, ~1M images, 14 classes,
  labels from surrounding text, intrinsically noisy). Synthetic corruption comes in two
  protocols: **symmetric/uniform** noise, flipping a chosen fraction of training labels to one
  of the other classes uniformly, at rates `eta` swept from 0 up to 0.8; and
  **asymmetric/class-conditional** noise, flipping only within confusable pairs (e.g. CIFAR-10
  TRUCK->AUTOMOBILE, CAT<->DOG; CIFAR-100 within 20 super-classes), at rates up to 0.4. The
  test set is always clean.
- **Architectures.** A 4-layer CNN for MNIST, an 8-layer CNN (6 conv + 2 dense) for CIFAR-10,
  ResNet-44 for CIFAR-100, and an ImageNet-pretrained ResNet-50 for Clothing1M.
- **Optimization protocol.** SGD with momentum 0.9, weight decay (around `1e-4` to `5e-3`),
  initial learning rate 0.1 (or `1e-3` for the pretrained Clothing1M run), with step decays at
  fixed epoch milestones; simple augmentation (shift, horizontal flip) on CIFAR.
- **Metrics and diagnostics.** Clean-test classification accuracy; per-class test-accuracy
  curves over training (to read class-biased dynamics); prediction confidence on the clean
  portion of a class and per-class true-positive counts; t-SNE 2D embeddings of
  penultimate-layer features.

## Code framework

The objective plugs into a fixed supervised-classification training harness: a data pipeline
that yields minibatches where some labels have been corrupted upstream, a standard network
producing class probabilities, an SGD optimizer, and a training loop that evaluates a scalar
loss, backpropagates, and steps. Everything except the loss is settled. The single empty slot is
the per-minibatch objective: given one-hot labels and predicted class probabilities, return one
scalar to minimize.

```python
import tensorflow as tf
from keras import backend as K


def make_classification_loss(**params):
    """Per-minibatch training objective for classification under label corruption.

    The data pipeline, network, optimizer, and training loop are fixed; only this
    loss is open. The loss receives one-hot labels, predicted class probabilities,
    and returns a scalar tensor to minimize."""

    def loss(y_true, y_pred):
        # y_true: (batch, num_classes) one-hot targets, some corrupted upstream
        # y_pred: (batch, num_classes) model class probabilities
        # TODO: the training objective we will design.
        pass

    return loss


# existing training harness the objective plugs into
model.compile(loss=make_classification_loss(), optimizer=optimizer, metrics=["accuracy"])
```

The training harness supplies one-hot labels and class probabilities; the returned `loss`
function is where the objective will live.
