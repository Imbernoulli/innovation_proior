# Context

## Research question

Adaptive stochastic gradient methods — the family that rescales each coordinate of the gradient by a
running estimate of its second moment — have become the optimizer of choice for training deep
networks, prized for their fast early progress and their relative insensitivity to per-parameter
scaling. In many important settings (notably attention-based sequence models and large-scale language
pretraining) the standard practice is to run them with a **learning-rate warmup**: deliberately shrink
the step size for the first few hundred or few thousand updates, then ramp it up to the target rate.
Warmup is used as a heuristic, re-tuned by trial and error for each task; its shape and duration are
chosen empirically.

The question this setting raises is mechanistic: *why* does an adaptive optimizer behave differently in
its opening updates than later in training, and *what* is warmup doing during that early phase?

## Background

**The adaptive update and its denominator.** All of these optimizers form, per coordinate, a running
estimate of the squared gradient and divide the step by its square root. The early progress comes from
this rescaling: directions with persistently small gradients get amplified, large ones damped. The
second-moment estimate is the heart of the method — a *statistical* estimate, formed from however many
gradient samples have been seen so far.

**The early phase has few samples.** At step `t`, the second-moment estimate has effectively averaged
on the order of a handful of squared gradients. The per-coordinate denominator is built from very few
samples in the opening updates. This is a basic fact about any second-moment-based adaptive rule,
independent of architecture.

**Diagnostic observations on existing systems.** Several measurements about *existing* adaptive
optimizers describe the setting and are knowable without any new method:

- On a Transformer trained for German→English translation (IWSLT'14), removing warmup leaves the
  training loss around 10; adding warmup drives it below 3. Similar behavior appears in
  bidirectional-transformer language-model pretraining.
- A histogram of absolute gradient values (log scale) over training shows that *without* warmup the
  gradient distribution shifts within the first ~10 updates, its mass moving toward small values, and
  stays in that regime through training. *With* warmup the distribution remains spread out.
- Two controlled interventions probe the early phase. (i) Freeze the parameters and the first moment
  for the first ~2000 iterations while still accumulating the second-moment estimate, then resume
  normal updates: training proceeds normally and the gradient distribution stays spread out — i.e.
  collecting more samples for the denominator before stepping changes the outcome. (ii) Inflate the
  numerical-stability constant in the denominator from a negligible `1e-8` to a non-negligible `1e-4`,
  which caps how large the adaptive rate can get and likewise avoids the early collapse, though the
  large additive constant biases the rate and changes final quality.

**Distributions of squared Gaussians.** At initialization the weights are drawn mean-zero, so the
per-coordinate gradients are roughly mean-zero; a reasonable starting description treats the early
gradients in a coordinate as i.i.d. zero-mean Gaussian `N(0, σ²)`. Standard distribution theory relates
sums and reciprocals of squared Gaussians to the chi-square family, whose PDFs and moments are known in
closed form.

**EMA bookkeeping.** The second-moment estimate is an exponential moving average (EMA) with decay `β₂`,
weighting sample `i` at step `t` by `β₂^{t−i}`; the geometric-sum identities for such weights are
standard. A classical result in forecasting (Nau, 2014) relates an EMA to a simple moving average (SMA)
of a finite window via the *center of mass* of the two weightings.

**A basic scaling fact.** Scaling a random quantity by `α` scales its variance by `α²`
(`Var[αx]=α²Var[x]`).

## Baselines

**SGD with momentum.** Accumulate a (bias-uncorrected) EMA of the gradient `m_t = β₁m_{t-1}+(1-β₁)g_t`
and step `θ_t = θ_{t-1} − α m_t`. No per-coordinate rescaling, hence no second-moment estimate.

**Adagrad** (Duchi et al., 2011). Per-coordinate rate `∝ 1/√(Σ_{i≤t} g_i²)`, accumulating *all* past
squared gradients. The denominator grows without bound, so the effective learning rate decays
monotonically; well suited to sparse and convex problems.

**RMSprop** (Tieleman & Hinton, 2012). Replace Adagrad's growing sum by an EMA of squared gradients,
`v_t = β₂v_{t-1}+(1-β₂)g_t²`, and step by `g_t/√v_t`, keeping the rate from decaying to zero.

**Adam** (Kingma & Ba, 2014). Combine an EMA first moment and EMA second moment, each
**bias-corrected** for their zero initialization:
```
m_t = β₁ m_{t-1} + (1-β₁) g_t ,   m̂_t = m_t / (1-β₁^t)
v_t = β₂ v_{t-1} + (1-β₂) g_t² ,  v̂_t = v_t / (1-β₂^t)
θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)
```
The de-facto default. Variants Nadam (Dozat, 2016) and Adadelta (Zeiler, 2012) share the same
EMA-second-moment core.

**Generic adaptive framework** (Reddi et al., 2019). All of the above are instances of a single
template: at step `t`, compute a momentum `m_t = φ(g_1,…,g_t)` and an adaptive rate `l_t = ψ(g_1,…,g_t)`,
then update `θ_t = θ_{t-1} − α_t m_t l_t`, all element-wise. Adam corresponds to the bias-corrected
EMA choices of `φ` and `ψ`. This template is the level of abstraction for reasoning about the adaptive
rate `ψ` in isolation.

**Learning-rate warmup** (Vaswani et al., 2017; Goyal et al., 2017; Popel & Bojar, 2018). Set `α_t`
small for the first `T_w` steps (e.g. linear `α_t = (t/T_w)·α_0`) before settling to the schedule. The
common workaround for the early-phase behavior above; it was found to stabilize deep layers (Gotmare et
al., 2018) and requires choosing `T_w` per task.

**Other stabilizers.** Gradient clipping (Bengio et al., 2013), careful initialization (Balduzzi et
al., 2017; Zhang et al., 2019), and normalization layers (Ioffe & Szegedy, 2015; Ba et al., 2016)
address training instability from the architecture/gradient side rather than from the optimizer's
adaptive rate.

## Evaluation settings

The natural yardsticks for an optimizer at this time, spanning the regimes where warmup is used:

- **Neural machine translation.** Transformer encoder-decoder on IWSLT'14 De↔En (small) and WMT'16
  En→De (large), via a standard sequence-to-sequence toolkit; tokenized text, label-smoothed
  cross-entropy, BLEU for evaluation, with the Adam-plus-warmup recipe as the reference protocol.
- **Language modeling.** Word-level LSTM language models (e.g. on the One Billion Word benchmark) with
  adaptive softmax, dropout, and gradient clipping; perplexity / negative log-likelihood as the metric.
- **Image classification.** ResNet architectures on CIFAR-10 and ImageNet, standard data augmentation
  (random crop / horizontal flip), step-decay schedules, top-1 accuracy.
- **Diagnostic protocol.** Track training loss and gradient-magnitude histograms over the first few
  thousand updates with and without warmup, and compare the controlled interventions above, to localize
  where and when the early-phase behavior occurs.

## Code framework

The primitives below already exist: a base optimizer class that owns per-parameter state and exposes a
`step()` called once per minibatch, and the standard EMA bookkeeping for first and second moments. The
open slot is the rule that turns those moments into a step in the early, sample-starved phase.

```python
import math
import torch
from torch.optim.optimizer import Optimizer

class AdaptiveOptimizer(Optimizer):
    """Generic per-coordinate adaptive optimizer: EMA first & second moments + a step rule."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:                       # known: zero-init EMA state
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)      # m_t
                    state['exp_avg_sq'] = torch.zeros_like(p.data)   # v_t
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # known: EMA updates of the first and second moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                # ---- the slot the method fills ----
                # Given m_t, v_t, t, β₁, β₂: decide the per-coordinate step in the early,
                # few-sample phase (the part warmup currently patches).
                step_direction, step_scale = self._step_rule(exp_avg, exp_avg_sq, t, beta1, beta2, group)
                p.data.add_(step_direction, alpha=-group['lr'] * step_scale)

        return loss

    def _step_rule(self, exp_avg, exp_avg_sq, t, beta1, beta2, group):
        # TODO: the contribution. Return (direction_tensor, scalar_scale).
        raise NotImplementedError
```
