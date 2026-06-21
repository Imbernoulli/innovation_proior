A trained multi-class classifier can be accurate and yet lie about how often it is right. I keep seeing the same thing on a modern deep network — a ResNet with batch normalization on CIFAR-100, say: it is accurate, but its reported confidence, the maximum softmax probability, runs systematically above its true correctness frequency. Bin the predictions by reported confidence and ask, in each bin, what fraction were actually correct, and the answer sits stubbornly below the bin's confidence; the reliability diagram bows away from the diagonal. This matters because everything downstream treats that number as a real probability — a perception stack that brakes on it, a diagnosis system that defers to a human when unsure, a recognizer whose scores are fused with a language model — and all of them break when the confidence is systematically too high. What I want is a faithful confidence: writing $\hat Y$ for the predicted class and $\hat P$ for its confidence, I want $\mathbb{P}(\hat Y = Y \mid \hat P = p) = p$ for every $p$, and I want it cheaply, fit on a held-out validation set, without retraining the network and without spending a single point of the accuracy that was so expensive to obtain.

The failure is new, and that is the clue. A shallow LeNet from a decade ago was well calibrated on this kind of task — its average confidence tracked its accuracy. Something in the modern recipe is correlated with worse calibration even as accuracy improves: sweeping depth or width down the classification error but drives the calibration error up; batch normalization helps accuracy but hurts calibration; weight decay fell out of fashion, and dialing it back up keeps improving calibration long after accuracy has stopped, so accuracy and calibration are simply not optimized by the same setting. The cleanest view is in the training curves of a 110-layer ResNet: after the learning-rate drop the test error keeps inching down — 29% to 27% — while the NLL turns around and climbs. The network overfits the negative log-likelihood while still improving the 0/1 loss. Mechanically this is forced: once almost every training point is classified correctly, cross-entropy can still be reduced only by pushing the correct-class probability toward 1, that is, by inflating the logits, by becoming more confident than the small accuracy gain warrants. So the class ranking is not what needs repair — accuracy is fine, even improving. The visible failure is that the *magnitude* of the logits has run ahead of what the probabilities should be, a dominant *scale* error.

That diagnosis tells me what a fix must and must not do. I should not retrain, should not change which class wins, should not relearn the boundary. The existing post-hoc options each miss in their own way. The non-parametric binning methods — histogram binning, isotonic regression, BBQ — are flexible and fine on a binary problem, but for a multiclass net the standard move is to run $K$ one-versus-all problems and renormalize, which throws away the joint softmax structure and multiplies the fitting burden by $K$; they are flexible enough to overfit a modest validation set, produce a non-smooth map, and ignore the evidence that the dominant error is a shared scale. The parametric line acts where the scale problem lives, on the logits: Platt scaling outputs $\sigma(az+b)$ but is binary, and its multiclass generalizations — matrix scaling $\mathbf{W}\mathbf{z}+\mathbf{b}$ with $K(K+1)$ parameters, vector scaling with a diagonal $\mathbf{W}$ and $2K$ — are richer than the shared-scale hypothesis calls for. Matrix scaling has roughly ten thousand parameters on CIFAR-100 and a million on ImageNet, enough to overfit, and a full $\mathbf{W}$ can rotate the logits and *change the argmax*, degrading the accuracy I was promised I would keep. Vector scaling is better but can still rescale classes differently and reorder them.

I propose temperature scaling. The minimal transform that corrects a uniform scale error, and only that, is to divide every logit by one shared positive number $T$ — the temperature, borrowing the name from statistical mechanics and from the soft targets of distillation. The calibrated confidence is

$$\hat q_i = \max_k\,\sigma_\text{SM}(\mathbf z_i / T)^{(k)},\qquad \sigma_\text{SM}(\mathbf u)^{(k)}=\frac{e^{u_k}}{\sum_j e^{u_j}},$$

with a single scalar $T>0$ shared across all classes and all examples. The reason this is exactly right, rather than merely small, is in its two properties. Because $T$ is the *same* for every class, dividing by a positive $T$ is a monotone transform that never reorders the logits: $\arg\max_k z_k/T = \arg\max_k z_k$, so the predicted class never changes and the accuracy is preserved *exactly*, not approximately — the property matrix scaling could not promise. And the effect on the probabilities is purely a softening: as $T\to\infty$ the scaled logits flatten and $\hat q_i\to 1/K$, maximal uncertainty; at $T=1$ I recover the original probabilities; as $T\to0^+$ the largest logit dominates and the distribution collapses onto the argmax class, up to ties. Sweeping $T$ upward from 1 does exactly one thing — bleeds confidence out of an overconfident model toward uniform — which is the one knob the shared-scale hypothesis needs, with almost no capacity to overfit.

I pick $T$ by minimizing the validation negative log-likelihood. The ECE I care about is binned and non-differentiable, a poor thing to optimize directly; NLL is its smooth, well-behaved cousin, the same objective whose over-optimization exposed the scale problem in the first place, and so the right lever to pull the logits back. Freeze the network, run it once over the validation set in eval mode to cache the logits and labels — eval mode matters so batch norm uses its frozen running statistics rather than recomputing batch statistics, otherwise the very thing being calibrated shifts underneath me — then minimize NLL over the single scalar. Because argmax-preservation requires $T>0$, I optimize an unconstrained $\log T$ and exponentiate inside the scaling step; I initialize $T$ above 1 (1.5), since an overconfident model needs softening; and L-BFGS on the cached logits is ample for a smooth one-parameter problem.

What makes me trust this beyond the overconfidence story is that the same operation falls out of a clean principle. Calibration is about being no more confident than the evidence warrants — being as high-entropy as possible while consistent with the data. So maximize the entropy of the predicted distributions subject to one coupling constraint, that the average logit assigned to the *true* class equals the average logit under $q$ — a moment-matching condition that pins the distributions to the data without dictating their shape:

$$\max_q -\sum_{i}\sum_k q_i^{(k)}\log q_i^{(k)}\quad\text{s.t.}\quad q_i^{(k)}\ge0,\;\sum_k q_i^{(k)}=1,\;\sum_i z_i^{(y_i)}=\sum_i\sum_k z_i^{(k)} q_i^{(k)}.$$

Introducing a multiplier $\lambda$ for the coupling constraint and $\beta_i$ for each normalization, the entropy term contributes $-(\log q_i^{(k)}+1)$, the coupling term $\lambda z_i^{(k)}$, the normalization term $\beta_i$, so stationarity gives $\log q_i^{(k)}=\lambda z_i^{(k)}+\beta_i-1$, hence $q_i^{(k)}=\exp(\lambda z_i^{(k)}+\beta_i-1)$ — automatically positive, so the nonnegativity I set aside is satisfied for free. The factor $e^{\beta_i-1}$ enters every class identically and is divided away by normalization, leaving

$$q_i^{(k)}=\frac{e^{\lambda z_i^{(k)}}}{\sum_{j} e^{\lambda z_i^{(j)}}}=\sigma_\text{SM}(\lambda\mathbf z_i)^{(k)},$$

and writing the positive scale as $\lambda = 1/T$ gives the temperature form exactly. So temperature scaling is the unique maximum-entropy distribution that matches the average true-class logit, and the lone scalar is precisely the lone Lagrange multiplier of the single moment constraint — which is why one parameter is principled, not just convenient. For an overconfident model, raising $T$ (lowering $\lambda$) raises the output entropy back to what the validation data support.

End to end: train the network normally and freeze it; hold out a validation split; push it through the frozen network once, in eval mode, caching logits and labels; minimize validation NLL over $T$ by dividing every cached logit by $T$ and stepping L-BFGS on softmax cross-entropy; at test time the calibrated confidence is $\max_k\sigma_\text{SM}(\mathbf z/T)^{(k)}$ with the fitted $T$, and the predicted label is unchanged. The code is a thin decorator around the trained model holding one learnable scalar.

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
