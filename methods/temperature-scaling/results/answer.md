# Temperature Scaling

## Problem

A trained multi-class classifier is accurate but **miscalibrated**: its reported confidence
(the max softmax probability) does not match its true correctness frequency — modern
high-capacity, lightly-regularized networks are systematically *overconfident*. We want a
faithful confidence score, $\mathbb P(\hat Y=Y\mid \hat P=p)=p$, obtained cheaply, without
retraining the network and without changing its accuracy.

## Key idea

The classifier's class *ranking* is fine; only the *scale* of its logits is inflated (once the
training set is fit, NLL can only drop by sharpening the correct-class probabilities). So
correct the scale with a single shared scalar: divide every logit by a temperature $T>0$ before
the softmax, and fit $T$ on a held-out validation set.

## Method

Given the logit vector $\mathbf z_i$ for input $\mathbf x_i$, the calibrated confidence is

$$\hat q_i = \max_k\,\sigma_\text{SM}(\mathbf z_i/T)^{(k)},\qquad
\sigma_\text{SM}(\mathbf u)^{(k)}=\frac{e^{u_k}}{\sum_j e^{u_j}}.$$

Because $T$ is shared across all classes, $\arg\max_k z_k/T=\arg\max_k z_k$: **the predicted
class — and therefore the accuracy — is unchanged.** $T=1$ recovers the original probabilities;
$T\to\infty$ flattens them to the uniform $1/K$ (maximum uncertainty); $T\to0^+$ collapses them
to a one-hot point mass. With the network frozen, $T$ is the single parameter optimized to
minimize the validation negative log-likelihood (a smooth proper scoring rule; the binned ECE
is non-differentiable and only used for evaluation).

This is the simplest extension of Platt scaling. Matrix scaling ($\mathbf W\mathbf z+\mathbf b$,
$K(K+1)$ params) and vector scaling (diagonal $\mathbf W$, $2K$ params) are the richer cousins,
but matrix scaling overfits for large $K$ and can change the argmax; the single scalar cannot.

**Maximum-entropy characterization.** Temperature scaling is the unique solution of

$$\max_q -\sum_{i}\sum_k q_i^{(k)}\log q_i^{(k)}\quad\text{s.t.}\quad
q_i^{(k)}\ge0,\;\sum_k q_i^{(k)}=1,\;\sum_i z_i^{(y_i)}=\sum_i\sum_k z_i^{(k)} q_i^{(k)}.$$

Lagrangian stationarity gives $\log q_i^{(k)}=\lambda z_i^{(k)}+\beta_i-1$, so
$q_i^{(k)}\propto e^{\lambda z_i^{(k)}}$; normalizing yields
$q_i^{(k)}=\sigma_\text{SM}(\lambda\mathbf z_i)^{(k)}$ with $T=1/\lambda$. The single scalar is
the lone multiplier of the single moment constraint (average true-class logit = average weighted
logit). Raising $T$ raises output entropy back to what the data support — undoing overconfidence.

## Code

```python
import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """Wrap a trained classifier; divide its logits by a learned scalar temperature T > 0.
    `model` must output raw logits (not softmax / log-softmax)."""
    def __init__(self, model):
        super().__init__()
        self.model = model                                    # frozen, already trained
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # init > 1: expect overconfidence

    def temperature_scale(self, logits):
        T = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / T                                     # monotone => argmax unchanged

    def forward(self, input):
        return self.temperature_scale(self.model(input))

    def set_temperature(self, valid_loader):
        nll = nn.CrossEntropyLoss()
        self.model.eval()                                     # freeze BatchNorm running stats
        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in valid_loader:
                logits_list.append(self.model(input))
                labels_list.append(label)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)

        def closure():
            optimizer.zero_grad()
            loss = nll(self.temperature_scale(logits), labels)  # softmax(z / T) cross-entropy
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


def ece(logits, labels, n_bins=15):
    """Expected Calibration Error: |confidence - accuracy| averaged over equal-width bins."""
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    out = torch.zeros(1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = conf.gt(lo) & conf.le(hi)
        if m.float().mean() > 0:
            out += (conf[m].mean() - correct[m].mean()).abs() * m.float().mean()
    return out
```

Fit once on validation (`set_temperature`), then `forward` returns calibrated probabilities at
test time. Accuracy is identical to the uncalibrated model; only the confidences are softened.
