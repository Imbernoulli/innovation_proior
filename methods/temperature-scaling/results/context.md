# Context

## Research question

A classifier used inside a larger decision pipeline must not only be accurate, it must
*know how often it is right*. When a network reports a probability of 0.8 for its predicted
class, we want that to mean something concrete: of all the inputs on which it says 0.8, about
80% should actually be correct. This property is **calibration**. A self-driving perception
stack that brakes on the network's confidence, a diagnosis system that defers to a human
doctor when unsure, a speech recognizer whose acoustic scores are fused with a language model
— all of them treat the reported confidence as a real probability, and all of them break if
that confidence is systematically too high.

The precise problem: take a trained multi-class classifier whose softmax output is *not* a
faithful probability of correctness, and produce, as cheaply as possible, a confidence score
that *is* faithful — using a held-out validation set, without retraining the network, and
without hurting the classification accuracy that was so expensive to obtain. Formally, writing
$\hat Y$ for the predicted class and $\hat P$ for its confidence, the ideal target is
$\mathbb{P}(\hat Y = Y \mid \hat P = p) = p$ for every $p\in[0,1]$.

## Background

A decade earlier, neural networks were observed to produce well-calibrated probabilities on
binary classification tasks (Niculescu-Mizil & Caruana 2005). The same contrast appears in the
CIFAR-100 diagnostic: a shallow 5-layer LeNet has its average confidence closely tracking its
accuracy, and its reliability diagram sits roughly on the diagonal.

Several diagnostic observations about *modern* networks (deep/wide architectures with Batch
Normalization and little weight decay) show that this is no longer true — they are accurate but
badly **overconfident**, with average confidence well above accuracy. These observations
associate miscalibration with capacity and regularization choices rather than proving a single
cause:

- **Capacity.** Increasing depth or width lowers classification error but *raises* the
  expected calibration error. On a CIFAR-100 ResNet, sweeping depth (at 64 filters/layer) or
  width (at 14 layers) shows calibration error growing substantially with capacity, even
  though the smallest models already show some miscalibration.
- **Batch Normalization.** Models trained with BatchNorm tend to be more miscalibrated than
  those without, even when BatchNorm slightly improves accuracy, and this holds across
  learning-rate choices.
- **Weight decay.** As regularization has fallen out of fashion (top ImageNet models use an
  order of magnitude less weight decay than earlier ones), calibration has worsened.
  Calibration keeps improving as weight decay is increased, well past the point of optimal
  accuracy — accuracy and calibration are not optimized by the same setting.
- **The NLL/accuracy disconnect.** During training, once a model classifies (almost) all
  training points correctly, the negative log-likelihood can still be driven down by making the
  correct-class probabilities sharper. Watching a 110-layer ResNet on CIFAR-100: after the
  learning rate drops at epoch 250, test error and NLL both fall, but then NLL *overfits* for
  the rest of training while test error keeps slowly improving (29% → 27%). The network can
  buy accuracy at the expense of well-modeled probabilities — overfitting manifests in the
  probabilities, not the 0/1 loss.

The load-bearing concepts: the softmax $\sigma_\text{SM}(\mathbf z)^{(k)} = e^{z_k}/\sum_j
e^{z_j}$ turning a logit vector into a distribution; the predicted class
$\hat y=\arg\max_k z_k$ and confidence $\hat p=\max_k\sigma_\text{SM}(\mathbf z)^{(k)}$;
the notion of a *proper scoring rule* (in expectation, NLL is minimized exactly when the
predicted distribution equals the true conditional $\pi(Y|X)$); and the idea of a held-out
validation set as a place to fit a small post-processing model without touching the network.

## Baselines

A new calibration method would be measured against the existing post-processing techniques,
all of which fit a map from uncalibrated scores to calibrated probabilities on a held-out set.

**Histogram binning** (Zadrozny & Elkan 2001). Partition the predicted probabilities into $M$
bins $B_1,\dots,B_M$; assign each bin a single calibrated value $\theta_m$ chosen to minimize
the bin-wise squared loss $\sum_m\sum_i \mathbf1(\hat p_i\in B_m)(\theta_m-y_i)^2$, whose
solution is the empirical accuracy in the bin. Simple and non-parametric, but the bin
boundaries are fixed by hand and the output is piecewise-constant.

**Isotonic regression** (Zadrozny & Elkan 2002). Fit a piecewise-constant *non-decreasing*
function $f$ minimizing $\sum_i(f(\hat p_i)-y_i)^2$, jointly choosing the bin boundaries and
values subject to $\theta_1\le\dots\le\theta_M$. A strict generalization of histogram binning.
Still non-parametric; can overfit small validation sets and produces a non-smooth map.

**Bayesian Binning into Quantiles (BBQ)** (Naeini et al. 2015). Bayesian model averaging over
*all* binning schemes $s=(M,\mathcal I)$. With a uniform prior over schemes and Beta priors on
the per-bin parameters, the marginal likelihood $\mathbb P(D\mid S{=}s)$ has a closed form, and
the calibrated probability is $\sum_s \mathbb P(\hat q\mid \hat p,s,D)\,\mathbb P(s\mid D)$.
Heavier machinery; still fundamentally a binning model.

**Platt scaling** (Platt 1999; for nets, Niculescu-Mizil & Caruana 2005). The only *parametric*
member: fit scalars $a,b$ and output $\hat q=\sigma(a z+b)$ (sigmoid of an affine transform of
the logit), with $a,b$ chosen by NLL on the validation set, the network fixed. Binary only as
stated; needs a multiclass extension.

The natural multiclass extensions of Platt scaling work on the full logit vector $\mathbf z$:
**matrix scaling** applies an affine map $\mathbf W\mathbf z+\mathbf b$ before the softmax, with
$\mathbf W,\mathbf b$ fit by NLL; and **vector scaling** restricts $\mathbf W$ to be diagonal.
Matrix scaling's parameter count grows as $K(K+1)$ with the number of classes $K$.

## Evaluation settings

The yardsticks are standard classification benchmarks where a state-of-the-art network is
trained, a portion of data is held out for fitting the calibration map, and calibration is
measured on the test set.

- **Vision**: CIFAR-10 and CIFAR-100 (32×32 color, 10 / 100 classes), SVHN,
  Caltech-UCSD Birds (200 classes), Stanford Cars (196 classes), and ImageNet,
  using modern architectures — ResNet, ResNet with stochastic depth, Wide-ResNet, DenseNet —
  and a shallow LeNet as a historical reference point.
- **NLP / documents**: 20 Newsgroups, Reuters, and the Stanford Sentiment Treebank, with
  models such as deep averaging networks and tree-LSTMs.

Protocol: train the network normally; hold out a validation split (the same one used for
hyperparameter tuning) on which to fit the calibration map; evaluate on test. Calibration
metrics, computed by binning predictions into $M=15$ equal-width confidence bins:
$\acc(B_m)=\frac1{|B_m|}\sum_{i\in B_m}\mathbf1(\hat y_i=y_i)$ and
$\conf(B_m)=\frac1{|B_m|}\sum_{i\in B_m}\hat p_i$, summarized by

$$\text{ECE}=\sum_{m=1}^M\frac{|B_m|}{n}\bigl|\acc(B_m)-\conf(B_m)\bigr|,\qquad
\text{MCE}=\max_m\bigl|\acc(B_m)-\conf(B_m)\bigr|,$$

together with NLL $=-\sum_i\log\hat\pi(y_i|\mathbf x_i)$, reliability diagrams that plot
bin accuracy against bin confidence (the diagonal is perfect; the diagram does not show how
many samples fall in a bin), and the top-1 error.

## Code framework

The pieces that already exist: a trained classifier that emits a logit vector, a validation
data loader, a cross-entropy (NLL) loss, a standard optimizer, and a binned ECE metric for
diagnosis. A calibration method is a *post-processing decorator* that wraps the frozen network,
holds whatever small set of calibration parameters it needs, transforms the logits before the
softmax, and fits those parameters on the validation set. The empty slots are the parameter
container, the logit transform, and the fitting routine.

```python
import torch
from torch import nn, optim
from torch.nn import functional as F


class CalibratedModel(nn.Module):
    """Wrap a trained classifier with a learned post-hoc calibration map on the logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model                       # frozen, already trained
        # TODO: the calibration parameters this method introduces.

    def transform(self, logits):
        # TODO: map raw logits -> calibrated logits.
        raise NotImplementedError

    def forward(self, input):
        logits = self.model(input)               # network stays fixed
        return self.transform(logits)

    def fit(self, valid_loader):
        # Collect validation logits/labels once (network in eval mode so BN stats are frozen).
        training_mode = self.model.training
        self.model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in valid_loader:
                logits_list.append(self.model(input))
                labels_list.append(label)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        self.model.train(training_mode)
        # TODO: optimize the calibration parameters on (logits, labels).
        raise NotImplementedError


def ece(logits, labels, n_bins=15):
    """Expected Calibration Error: |confidence - accuracy| averaged over equal-width bins."""
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    out = torch.zeros(1, device=logits.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = conf.gt(lo) & conf.le(hi)
        prop = m.float().mean()
        if prop.item() > 0:
            out += (conf[m].mean() - correct[m].mean()).abs() * prop
    return out
```
