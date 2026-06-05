OK, let me think this through from scratch. I have a trained classifier — say a deep ResNet with batch normalization on CIFAR-100 — and it is *accurate*. But I keep seeing the same thing when I look at the probabilities it reports: on the test set its predicted-class confidence averages well above its actual accuracy. If I bin the predictions by reported confidence and ask, in each bin, what fraction were actually right, the answer sits stubbornly below the bin's confidence. The reliability diagram bows away from the diagonal. The net says 0.9 and is right maybe 0.7 of the time. It is overconfident, and I want to fix that.

Before I reach for a fix, let me understand *why* this happens, because the cure should attack the cause. The striking thing is that this is new. A shallow LeNet from a decade ago, on the same kind of task, was well calibrated — its average confidence tracked its accuracy. So something about how I train modern nets broke calibration while improving accuracy. Let me stare at what changed.

Depth and width went up enormously. When I sweep depth on a fixed-width ResNet, or width on a fixed depth, classification error goes down — but the expected calibration error goes *up*, monotonically, with capacity. Batch normalization came in; nets with it are more miscalibrated than nets without, even when it nudges accuracy up. Weight decay fell out of fashion; and when I dial weight decay back up, calibration keeps improving long after accuracy has stopped improving — accuracy and calibration just aren't optimized by the same regularization setting. Three different knobs, all pointing the same way: more capacity and less regularization buy accuracy and cost calibration.

There's a cleaner way to see the mechanism, and it's in the training curves. Take that 110-layer ResNet and watch the negative log-likelihood and the test error as training proceeds. After the learning-rate drop, both fall. But then something telling happens: the test error keeps inching down — 29% to 27% — while the NLL turns around and *climbs*. The network is overfitting the NLL while still improving the 0/1 loss. Think about what that means mechanically. Once almost every training example is classified correctly, the cross-entropy can still be reduced — but only by pushing the correct-class probability closer and closer to 1, i.e. by inflating the logits, by becoming more confident. The optimizer has run out of mistakes to fix, so it spends its remaining gradient making the right answers sharper. On test data that sharpening doesn't correspond to any real increase in correctness; it's pure overconfidence. So the disease isn't that the *ranking* of classes is wrong — accuracy is fine, even improving. The disease is that the *magnitude* of the logits has been inflated relative to what the probabilities should be.

That last sentence is the whole game, and it reframes the cure. If accuracy is fine and only the scale of the logits is off, then I do not want to retrain the network, I do not want to change which class wins, and I do not want to relearn the decision boundary. I want a cheap post-processing step that takes the already-trained logits and rescales them so the resulting probabilities are honest — fit on a small held-out set, leaving the expensive network untouched.

So let me survey what's on the table for post-hoc calibration and ask, for each, whether it fits a problem that is essentially "the logits are inflated."

The non-parametric binning methods. Histogram binning chops the predicted probabilities into bins and replaces each bin with its empirical accuracy. Isotonic regression does the smarter version — a piecewise-constant non-decreasing function fit by least squares, choosing both the bin edges and the values. BBQ goes Bayesian and averages over all binning schemes. These are flexible, and on a binary problem they're fine. But two things bother me for the multiclass net. First, they're built for one probability at a time; the standard multiclass trick is to run $K$ one-versus-all binning problems and renormalize, which throws away the joint structure of the softmax and multiplies the fitting burden by $K$. Second, they're flexible enough to overfit a held-out set that isn't huge, and they produce a non-smooth map — which is awkward if I later want to fuse these probabilities into another probabilistic model. They don't use the fact that I *know* the problem is just a scale issue.

Then there's the parametric line. Platt scaling: take the logit $z$ and output $\sigma(az+b)$, fitting $a,b$ by likelihood on the validation set. That's binary. The multiclass generalizations operate on the whole logit vector $\mathbf z$ before the softmax. Matrix scaling applies an affine map $\mathbf W\mathbf z+\mathbf b$ and refits the softmax on top; vector scaling restricts $\mathbf W$ to a diagonal. These are appealing because they're parametric and they act exactly where I diagnosed the problem — on the logits. But matrix scaling has $K(K+1)$ parameters. On CIFAR-100 that's already ~10,000 parameters fit on a validation set of a few thousand; on ImageNet with $K=1000$ it's a million. That will overfit, and worse, a full $\mathbf W$ can rotate the logits and *change the argmax*, i.e. degrade the accuracy I was promised I'd keep. Vector scaling is better — diagonal, $2K$ parameters — but it can still rescale classes differently and reorder them, and it's still more capacity than my diagnosis calls for.

Here's where I want to be disciplined. My diagnosis was very specific: the *ranking* is right, only the overall *scale* of the logits is inflated, uniformly enough that confidence runs ahead of accuracy across the board. The minimal transform that fixes a uniform scale problem, and *only* that, is dividing every logit by one shared positive number. Let me write the calibrated confidence as

  $\hat q_i = \max_k\,\sigma_\text{SM}(\mathbf z_i / T)^{(k)}$,

with a single scalar $T>0$ shared across all classes and all examples. Borrow the name from where this operation already lives — statistical mechanics, and the soft targets in distillation — and call $T$ the temperature.

Let me check it has exactly the properties my diagnosis demands, before I get attached to it. Because $T$ is the *same* for every class, dividing by it is a monotone transform of the logits that doesn't reorder them: $\arg\max_k z_k/T = \arg\max_k z_k$. So the predicted class never changes, which means **the accuracy is exactly preserved** — not approximately, exactly. That's the property matrix scaling couldn't promise. And the effect on the probabilities is precisely a softening: as $T\to\infty$ the scaled logits flatten and $\hat q_i\to 1/K$, the maximally uncertain uniform distribution; at $T=1$ I recover the original probabilities; as $T\to 0^+$ the largest logit dominates and the distribution collapses to a one-hot point mass. So sweeping $T$ from 1 upward does exactly one thing — bleeds confidence out of an overconfident model toward uniform — which is the one knob my overconfidence diagnosis says I need. With a single parameter there is essentially nothing to overfit on the held-out set.

How do I pick $T$? I want the probabilities to be honest, and the clean differentiable objective that rewards honest probabilities is the negative log-likelihood — it's a proper scoring rule, minimized exactly when the predicted distribution matches the truth. The expected-calibration-error metric I care about is binned and non-differentiable, a bad thing to optimize directly; NLL is its smooth, well-behaved cousin and it's what pushed the logits to inflate in the first place, so it's the right lever to pull them back. So: freeze the network, run it once over the validation set to cache the logits and labels, and minimize the validation NLL over the single scalar $T$. One parameter, a couple hundred optimizer iterations, done.

Now I want to know whether "divide the logits by a constant and minimize NLL" is a principled thing or just a lucky hack that happened to match my overconfidence story. Let me see if temperature scaling falls out of a clean optimization principle, because if it does I'll trust it far more.

The intuition is: calibration is about *not being more confident than the evidence warrants*, i.e. being as uncertain — as high-entropy — as possible while still being consistent with the data. So let me literally maximize the entropy of the predicted distributions subject to a constraint that ties them to the observed logits and labels. For validation logit vectors $\mathbf z_1,\dots,\mathbf z_n$ with labels $y_1,\dots,y_n$, look for distributions $q(\mathbf z_i)^{(k)}$ solving

  maximize $\;-\sum_{i=1}^n\sum_{k=1}^K q(\mathbf z_i)^{(k)}\log q(\mathbf z_i)^{(k)}$
  subject to $\;q(\mathbf z_i)^{(k)}\ge 0$, $\;\sum_k q(\mathbf z_i)^{(k)}=1$ for each $i$,
  and one coupling constraint $\;\sum_{i=1}^n z_i^{(y_i)}=\sum_{i=1}^n\sum_{k=1}^K z_i^{(k)}q(\mathbf z_i)^{(k)}$.

The first two constraints just say each $q(\mathbf z_i)$ is a probability vector. The third is the substantive one: the average logit assigned to the *true* class must equal the average logit under the distribution $q$ — a moment-matching condition that pins the distributions to the data without dictating their shape. Maximum entropy subject to a moment constraint is exactly the recipe that produces exponential-family / softmax forms, so let me turn the crank with Lagrange multipliers and see what drops out.

Introduce one multiplier $\lambda$ for the coupling constraint and one $\beta_i$ for each normalization constraint (I'll handle nonnegativity by checking it at the end), and write the Lagrangian

  $L=-\sum_i\sum_k q_i^{(k)}\log q_i^{(k)}+\lambda\sum_i\Bigl[\sum_k z_i^{(k)}q_i^{(k)}-z_i^{(y_i)}\Bigr]+\sum_i\beta_i\Bigl[\sum_k q_i^{(k)}-1\Bigr]$,

abbreviating $q_i^{(k)}=q(\mathbf z_i)^{(k)}$. Differentiate with respect to a single $q_i^{(k)}$. The entropy term contributes $-\frac{d}{dq}(q\log q)=-(\log q_i^{(k)}+1)$; the coupling term contributes $\lambda z_i^{(k)}$; the normalization term contributes $\beta_i$. So

  $\dfrac{\partial L}{\partial q_i^{(k)}}=-\log q_i^{(k)}-1+\lambda z_i^{(k)}+\beta_i$.

Set it to zero: $\log q_i^{(k)}=\lambda z_i^{(k)}+\beta_i-1$, hence

  $q_i^{(k)}=\exp\!\bigl(\lambda z_i^{(k)}+\beta_i-1\bigr)$.

This is positive automatically, so the nonnegativity constraint I set aside is satisfied for free — good. Now impose $\sum_k q_i^{(k)}=1$ to solve for the per-example multiplier: $\beta_i$ and the $-1$ enter every class identically, so they form a common factor $e^{\beta_i-1}$ that the normalization simply divides away,

  $q_i^{(k)}=\dfrac{e^{\lambda z_i^{(k)}}}{\sum_{j=1}^K e^{\lambda z_i^{(j)}}}=\sigma_\text{SM}(\lambda\mathbf z_i)^{(k)}.$

That is a softmax of the logits scaled by $\lambda$. Identify $\lambda = 1/T$, and it is *exactly* temperature scaling, $q_i^{(k)}=\sigma_\text{SM}(\mathbf z_i/T)^{(k)}$. So temperature scaling is not a hack — it is the unique maximum-entropy distribution consistent with matching the average true-class logit. The single scalar is the lone Lagrange multiplier of the single coupling constraint, which is why one parameter is the *right* number, not just a convenient small one. And it makes the overconfidence story precise: an overconfident model has low-entropy outputs, raising $T$ (lowering $\lambda$) raises the entropy back toward what the data support.

A couple of things to nail down for the implementation. When I cache the validation logits, I must run the network in eval mode so batch-norm uses its frozen running statistics rather than recomputing batch statistics — otherwise the very thing I'm calibrating shifts under me. And I'll optimize the single $T$ with a quasi-Newton method (L-BFGS) on the cached logits, since it's a smooth one-dimensional (well, one-parameter) problem and L-BFGS converges in a handful of steps. I'll initialize $T$ above 1 (say 1.5), since I diagnosed overconfidence and expect the optimum to soften, not sharpen.

Let me trace the whole procedure end to end. Train the network normally and freeze it. Hold out a validation split. Push that split through the frozen network once, in eval mode, and stash the logits and labels. Minimize the validation NLL over the scalar $T$ by dividing every cached logit vector by $T$, taking the softmax cross-entropy against the labels, and stepping L-BFGS. At test time, the calibrated confidence for an input is $\max_k\sigma_\text{SM}(\mathbf z/T)^{(k)}$ with the fitted $T$; the predicted label is unchanged. That's the entire method, and the code mirrors it: a thin decorator around the trained model holding one learnable scalar.

```python
import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """Decorator: divide a frozen classifier's logits by a learned scalar temperature T > 0.
    Output of `model` must be raw logits, not softmax."""
    def __init__(self, model):
        super().__init__()
        self.model = model                                 # already trained, kept fixed
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # expect overconfidence -> T > 1

    def temperature_scale(self, logits):
        # one shared scalar across classes and examples => argmax (accuracy) is preserved
        T = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / T

    def forward(self, input):
        logits = self.model(input)                         # network untouched
        return self.temperature_scale(logits)

    def set_temperature(self, valid_loader):
        nll = nn.CrossEntropyLoss()
        # eval mode so BatchNorm uses frozen running stats while we cache logits
        self.model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in valid_loader:
                logits_list.append(self.model(input))
                labels_list.append(label)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # minimize validation NLL over the single parameter T (smooth proper scoring rule)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)

        def closure():
            optimizer.zero_grad()
            loss = nll(self.temperature_scale(logits), labels)   # softmax(z / T) cross-entropy
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
```

The causal chain, start to finish: a modern high-capacity, lightly-regularized network is accurate but overconfident, because once it classifies the training set it can only lower NLL by inflating the correct-class logits — the *ranking* stays right while the *scale* of the logits runs ahead of true correctness. The minimal transform that corrects a uniform scale error without touching the ranking is dividing every logit by a single shared positive temperature $T$, which provably leaves the argmax — and hence the accuracy — unchanged and simply softens the distribution toward uniform as $T$ grows. Fitting $T$ by minimizing the held-out negative log-likelihood (a smooth proper scoring rule) needs one parameter and nothing to overfit, and the same operation falls out exactly as the unique maximum-entropy distribution that matches the average true-class logit — so the lone parameter is the lone Lagrange multiplier, and raising it restores exactly the entropy the data warrant.
