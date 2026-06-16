# Temperature Scaling

## Problem

A trained multi-class classifier is accurate but **miscalibrated**: its reported confidence
(the max softmax probability) does not match its true correctness frequency — modern
high-capacity, lightly-regularized networks are systematically *overconfident*. We want a
faithful confidence score, $\mathbb P(\hat Y=Y\mid \hat P=p)=p$, obtained cheaply, without
retraining the network and without changing its accuracy.

## Key idea

The post-hoc goal is to preserve the classifier's class ranking and correct a dominant
miscalibration mode: logits whose overall scale is too large. Once most training examples are
classified correctly, NLL can still be reduced by sharpening the correct-class probabilities, so
fit a single shared rescaling on held-out validation data: divide every logit by a temperature
$T>0$ before the softmax.

## Method

Given the logit vector $\mathbf z_i$ for input $\mathbf x_i$, the calibrated confidence is

$$\hat q_i = \max_k\,\sigma_\text{SM}(\mathbf z_i/T)^{(k)},\qquad
\sigma_\text{SM}(\mathbf u)^{(k)}=\frac{e^{u_k}}{\sum_j e^{u_j}}.$$

Because $T$ is shared across all classes, $\arg\max_k z_k/T=\arg\max_k z_k$: **the predicted
class — and therefore the accuracy — is unchanged.** $T=1$ recovers the original probabilities;
$T\to\infty$ flattens them to the uniform $1/K$ (maximum uncertainty); $T\to0^+$ collapses them
onto the argmax class, up to ties. With the network frozen, $T$ is the single parameter optimized to
minimize the validation negative log-likelihood (a smooth objective whose population optimum is
the true conditional distribution; the binned ECE is non-differentiable and only used for
evaluation).

This is the simplest extension of Platt scaling. Matrix scaling ($\mathbf W\mathbf z+\mathbf b$,
$K(K+1)$ params) and vector scaling (diagonal $\mathbf W$, $2K$ params) are the richer cousins,
but matrix scaling can overfit for large $K$ and can change the argmax; the single positive
scalar has far less capacity and cannot change the argmax.

## Metrics

Calibration is visualized with reliability diagrams: predictions are split into equal-width
confidence bins, each bin plots empirical accuracy against average confidence, and perfect
calibration lies on the diagonal. For $B_m=\{i:\hat p_i\in I_m\}$,

$$\acc(B_m)=\frac1{|B_m|}\sum_{i\in B_m}\mathbf1(\hat y_i=y_i),\qquad
\conf(B_m)=\frac1{|B_m|}\sum_{i\in B_m}\hat p_i,$$

and the scalar Expected Calibration Error is

$$\text{ECE}=\sum_{m=1}^M\frac{|B_m|}{n}\left|\acc(B_m)-\conf(B_m)\right|.$$

**Maximum-entropy characterization.** Temperature scaling is the unique solution of

$$\max_q -\sum_{i}\sum_k q_i^{(k)}\log q_i^{(k)}\quad\text{s.t.}\quad
q_i^{(k)}\ge0,\;\sum_k q_i^{(k)}=1,\;\sum_i z_i^{(y_i)}=\sum_i\sum_k z_i^{(k)} q_i^{(k)}.$$

Lagrangian stationarity gives $\log q_i^{(k)}=\lambda z_i^{(k)}+\beta_i-1$, so
$q_i^{(k)}\propto e^{\lambda z_i^{(k)}}$; normalizing yields
$q_i^{(k)}=\sigma_\text{SM}(\lambda\mathbf z_i)^{(k)}$; writing the positive scale as
$\lambda=1/T$ gives the temperature form. The single scalar is the lone multiplier of the single
moment constraint (average true-class logit = average weighted logit). For an overconfident
model, raising $T$ raises output entropy back to what the data support.

## Code

```python
import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """A thin decorator that wraps a classifier with temperature scaling.
    The wrapped model must output raw logits, not softmax or log-softmax."""
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.log_temperature = nn.Parameter(torch.log(torch.ones(1) * 1.5))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        wrapper_training = self.training
        model_training = self.model.training
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = self.log_temperature.device
        self.to(device)
        self.eval()                                           # freeze BatchNorm running stats

        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                label = label.to(device)
                logits_list.append(self.model(input))
                labels_list.append(label)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' %
              (before_temperature_nll, before_temperature_ece))

        optimizer = optim.LBFGS([self.log_temperature], lr=0.01, max_iter=200)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' %
              (after_temperature_nll, after_temperature_ece))

        self.train(wrapper_training)
        self.model.train(model_training)
        return self


class _ECELoss(nn.Module):
    """Expected Calibration Error over equally spaced confidence bins."""
    def __init__(self, n_bins=15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
```

Fit once on validation (`set_temperature`), then `forward` returns calibrated logits; applying
softmax gives calibrated probabilities at test time. Accuracy is identical to the uncalibrated
model because the predicted class is unchanged; only the confidences are softened.
