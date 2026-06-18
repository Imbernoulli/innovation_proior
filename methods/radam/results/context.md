# Context

## Research question

Adaptive stochastic gradient methods — the family that rescales each coordinate of the gradient by a
running estimate of its second moment — have become the optimizer of choice for training deep
networks, prized for their fast early progress and their relative insensitivity to per-parameter
scaling. Yet in many of the most important settings (notably attention-based sequence models and
large-scale language pretraining) they are fragile at the very start of training: run with a constant
learning rate from step one and the loss frequently settles into a bad local optimum or diverges. The
field's standard remedy is a hand-tuned **learning-rate warmup** — deliberately shrink the step size
for the first few hundred or few thousand updates, then ramp it up — but warmup is a heuristic with no
theory behind it. There is no guarantee it helps in a new setting and no principled way to choose its
shape or duration; practitioners re-tune it by trial and error for every task.

The question this raises is mechanistic: *why* does an adaptive optimizer need a small learning rate
in its opening updates, and *what exactly* does warmup fix? A satisfying answer would (a) identify the
concrete failure mode of the adaptive update in the early phase, (b) explain warmup as addressing that
specific failure, and (c) ideally replace the heuristic with a rule derived from the failure itself —
one with no free schedule to tune, that adapts automatically to the optimizer's own hyperparameters.

## Background

**The adaptive update and its denominator.** All of these optimizers form, per coordinate, a running
estimate of the squared gradient and divide the step by its square root. The early progress comes from
this rescaling: directions with persistently small gradients get amplified, large ones damped. The
estimate of the second moment is therefore the heart of the method — and it is a *statistical*
estimate, formed from however many gradient samples have been seen so far.

**The early phase has almost no samples.** At step `t`, the second-moment estimate has effectively
averaged on the order of a handful of squared gradients. An estimate of a variance-like quantity from
a handful of samples is itself a high-variance random quantity; taking its reciprocal square root —
which is what the adaptive rate does — makes a small, noisily-underestimated denominator blow the step
up. So the per-coordinate denominator is built from very few samples in the opening updates. This is a
pre-method fact about any second-moment-based adaptive rule, independent of architecture.

**Diagnostic observations on existing systems.** Several measurements about *existing* adaptive
optimizers set up the problem and are knowable without any new method:

- On a Transformer trained for German→English translation (IWSLT'14), removing warmup leaves the
  training loss stuck around 10; adding warmup drives it below 3. The same brittleness appears in
  bidirectional-transformer language-model pretraining.
- A histogram of absolute gradient values (log scale) over training shows that *without* warmup the
  gradient distribution is **distorted within the first ~10 updates** — its mass collapses toward small
  values — and the optimizer never recovers. The damage is done almost immediately.
- Two controlled interventions isolate the cause. (i) Freeze the parameters and the first moment for
  the first ~2000 iterations while still accumulating the second-moment estimate, then resume normal
  updates: the convergence failure disappears, and the gradient distribution is no longer distorted —
  i.e. simply *collecting more samples* for the denominator before stepping fixes it. (ii) Inflate the
  numerical-stability constant in the denominator from a negligible `1e-8` to a non-negligible `1e-4`:
  this also avoids the catastrophic failure (a large additive constant caps how large the adaptive
  rate can get), but it noticeably *hurts* final quality because the large constant biases the rate.

Together these say: the early failure is reduced either by gathering more samples for the denominator
or by crudely capping how large the adaptive rate can get — but a crude cap pays a bias penalty.

**Modeling the gradients at initialization.** At initialization the weights are drawn mean-zero, so the
per-coordinate gradients are roughly mean-zero too; a reasonable starting description is to treat the
early gradients in a coordinate as i.i.d. zero-mean Gaussian `N(0, σ²)`. Standard distribution theory
relates sums and reciprocals of squared Gaussians to the chi-square family, and these laws have known
PDFs and moments.

**EMA bookkeeping.** The second-moment estimate is an exponential moving average (EMA) with decay `β₂`,
weighting sample `i` at step `t` by `β₂^{t−i}`; the geometric-sum identities for such weights are
standard. A classical result in forecasting (Nau, 2014) relates an EMA to a simple moving average (SMA)
of a finite window via the *center of mass* of the two weightings.

**A basic scaling fact.** Scaling a random quantity by `α` scales its variance by `α²`
(`Var[αx]=α²Var[x]`).

## Baselines

**SGD with momentum.** Accumulate a (bias-uncorrected) EMA of the gradient `m_t = β₁m_{t-1}+(1-β₁)g_t`
and step `θ_t = θ_{t-1} − α m_t`. No per-coordinate rescaling, hence no second-moment estimate and none
of the early instability that comes with it — but also none of the fast, scale-free early progress that
makes adaptive methods attractive, and it is sensitive to learning-rate and per-parameter scaling.

**Adagrad** (Duchi et al., 2011). Per-coordinate rate `∝ 1/√(Σ_{i≤t} g_i²)`, accumulating *all* past
squared gradients. The denominator grows without bound, so the effective learning rate decays
monotonically — good for sparse/convex problems, but it stalls on long nonconvex training.

**RMSprop** (Tieleman & Hinton, 2012). Replace Adagrad's growing sum by an EMA of squared gradients,
`v_t = β₂v_{t-1}+(1-β₂)g_t²`, and step by `g_t/√v_t`. Keeps the rate from decaying to zero, but `v_t`
is a noisy estimate built from few samples early on — exactly the high-variance-denominator issue.

**Adam** (Kingma & Ba, 2014). Combine an EMA first moment and EMA second moment, each
**bias-corrected** for their zero initialization:
```
m_t = β₁ m_{t-1} + (1-β₁) g_t ,   m̂_t = m_t / (1-β₁^t)
v_t = β₂ v_{t-1} + (1-β₂) g_t² ,  v̂_t = v_t / (1-β₂^t)
θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)
```
The de-facto default. Its early second-moment estimate `v̂_t` is formed from few samples, and in
practice it is exactly Adam that exhibits the early instability the question is about — the gap this
leaves open. Variants Nadam (Dozat, 2016) and Adadelta (Zeiler, 2012) share the same EMA-second-moment
core and the same early-phase exposure.

**Generic adaptive framework** (Reddi et al., 2019). All of the above are instances of a single
template: at step `t`, compute a momentum `m_t = φ(g_1,…,g_t)` and an adaptive rate `l_t = ψ(g_1,…,g_t)`,
then update `θ_t = θ_{t-1} − α_t m_t l_t`, all element-wise. Adam corresponds to the bias-corrected
EMA choices of `φ` and `ψ`. This template is the right level of abstraction for reasoning about "the
adaptive rate `ψ`" in isolation.

**Learning-rate warmup** (Vaswani et al., 2017; Goyal et al., 2017; Popel & Bojar, 2018). Set `α_t`
small for the first `T_w` steps (e.g. linear `α_t = (t/T_w)·α_0`) before settling to the schedule. The
prevailing-wisdom workaround for the early instability above. It works empirically and was found to
stabilize deep layers (Gotmare et al., 2018), but it has no derivation, requires choosing `T_w` per
task, and offers no principle for its shape.

**Other stabilizers.** Gradient clipping (Bengio et al., 2013), careful initialization (Balduzzi et
al., 2017; Zhang et al., 2019), and normalization layers (Ioffe & Szegedy, 2015; Ba et al., 2016)
attack training instability from the architecture/gradient side rather than from the optimizer's
adaptive rate, and are orthogonal to it.

## Evaluation settings

The natural yardsticks for an optimizer at this time, spanning the regimes where warmup is known to
matter:

- **Neural machine translation.** Transformer encoder-decoder on IWSLT'14 De↔En (small) and WMT'16
  En→De (large), via a standard sequence-to-sequence toolkit; tokenized text, label-smoothed
  cross-entropy, BLEU for evaluation, with the Adam-plus-warmup recipe as the reference protocol.
- **Language modeling.** Word-level LSTM language models (e.g. on the One Billion Word benchmark) with
  adaptive softmax, dropout, and gradient clipping; perplexity / negative log-likelihood as the metric.
- **Image classification.** ResNet architectures on CIFAR-10 and ImageNet, standard data augmentation
  (random crop / horizontal flip), step-decay schedules, top-1 accuracy.
- **Diagnostic protocol.** Track training loss and gradient-magnitude histograms over the first few
  thousand updates with and without warmup, and compare the controlled interventions above, to localize
  where and when the early failure occurs.

## Code framework

The primitives below already exist: a base optimizer class that owns per-parameter state and exposes a
`step()` called once per minibatch, and the standard EMA bookkeeping for first and second moments. What
does **not** yet exist is the rule that decides how to turn those moments into a step in the early,
sample-starved phase — that is the empty slot the method will fill.

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
                # Given m_t, v_t, t, β₁, β₂: decide the per-coordinate step in a way that is
                # well-behaved in the early, few-sample phase (the part warmup currently patches).
                step_direction, step_scale = self._step_rule(exp_avg, exp_avg_sq, t, beta1, beta2, group)
                p.data.add_(step_direction, alpha=-group['lr'] * step_scale)

        return loss

    def _step_rule(self, exp_avg, exp_avg_sq, t, beta1, beta2, group):
        # TODO: the contribution. Return (direction_tensor, scalar_scale).
        raise NotImplementedError
```
