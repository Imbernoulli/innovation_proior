# Context: learning-rate scheduling for deep network training (circa 2016-2017)

## Research question

A deep convolutional classifier is trained by stochastic gradient descent with momentum on a
fixed pipeline: the optimizer type (SGD, momentum 0.9, weight decay 5e-4), the data
augmentation, the weight initialization (Kaiming normal), the loss (cross-entropy), and the
epoch budget (200 epochs) are all held constant. The one knob left free is the learning rate,
and more precisely the *learning rate as a function of training time* — the value `eta(t)` fed
to SGD at each epoch `t`. The learning rate is the single most consequential hyperparameter in
this setup, and a single constant value is wrong at both ends of training for opposite reasons.
Immediately after random initialization the network sits in a region of the loss surface whose
local curvature is high, so a large step overshoots and the loss can blow up; late in training,
near a minimum, a large step cannot settle and the parameters rattle around at the
gradient-noise floor instead of converging. The precise goal is therefore to design a schedule
`eta(t)` — a curve over the 200 epochs — that keeps early steps safe, makes fast progress in
the bulk of training, and lets the model settle cleanly at the end, all without touching the
optimizer, the data, or any other hyperparameter, and ideally without introducing a pile of
new schedule hyperparameters that themselves need tuning. The yardstick is the best test
accuracy reached during the run.

## Background

By this time the dominant recipe for training deep residual image classifiers is plain SGD
with (Nesterov or heavy-ball) momentum and a hand-tuned learning rate that is *dropped* at a
few points during training; adaptive optimizers (AdaDelta, Adam) exist but the
state-of-the-art results on CIFAR/ImageNet with residual nets are obtained with momentum SGD,
not with them. Several facts about the loss landscape and the optimizer are load-bearing here.

First, the **local stability of gradient descent** is governed by curvature. On a local
quadratic model of the loss with Hessian `H`, gradient descent with step `eta` contracts along
an eigendirection of curvature `lambda` only if `eta < 2/lambda`; once `eta` exceeds
`2/lambda_max(H)` the iteration *diverges* along the sharpest direction. Freshly initialized
networks are observed to occupy sharp, high-curvature regions, so `2/lambda_max` is small there
— a full-size learning rate can exceed this threshold and produce a sharp transient increase in
the loss, or outright divergence, before the dynamics drift to a flatter region where the same
rate is admissible. This is the mechanism behind early-training instability and divergence at
high learning rates.

Second, **momentum carries state across steps.** Heavy-ball / Nesterov SGD maintains a velocity
`v_{t+1} = mu*v_t - eta*grad`, `x_{t+1} = x_t + v_{t+1}`, with `mu = 0.9`. The velocity is an
exponentially weighted history of past (rate-scaled) gradients; an abrupt change in `eta`
leaves `v` mismatched to the new rate, which matters when the rate changes sharply.

Third, the practical wisdom is that the learning rate should be **large early and small late.**
A large rate early makes rapid progress while the model is far from any minimum and the gradient
points consistently downhill; a small rate late is needed to resolve the minimum below the
stochastic-gradient noise floor. This is why every competitive recipe of the era *reduces* the
rate over the course of training rather than holding it fixed.

Fourth, two motivating empirical observations about *existing* schedules are well documented.
(a) Training a large/aggressively-configured model from scratch with the full target rate from
the very first step degrades or destabilizes early training — the training-error curve starts
worse and, in the worst case, spikes and never recovers. (b) Holding the rate at a low constant
value for the first few epochs and *then* switching abruptly to the full target rate does not
fix this: the abrupt switch itself produces a training-error spike right at the transition,
because the network has descended into a region whose curvature still does not admit the full
rate. Both are observations about how *existing* schedules behave early in training, before any
new schedule is proposed.

## Baselines

These are the prior schedule families a new schedule would be measured against and would react
to. Each is a real, used recipe; the gap noted for each is where it stalls.

**Constant learning rate.** Hold `eta(t) = base_lr` for all epochs. This is the degenerate
baseline. **Gap:** it is simultaneously too large at the start (early instability / loss spike on
the high-curvature initial region) and too large at the end (cannot settle into the minimum,
parameters oscillate at the noise floor). No single value serves both phases.

**Step decay (He et al. 2016; Wide-ResNet recipe, Zagoruyko & Komodakis 2016).** Hold the rate
constant, then multiply it by a fixed factor at a few hand-picked milestones — divide by 10 at,
say, 50% and 75% of the budget in the ResNet recipe; the Wide-ResNet recipe multiplies by 0.2
at epochs 60/120/160 of a 200-epoch run. This is the workhorse that produced the leading
residual-net results. It captures "large early, small late" in a piecewise-constant way.
**Gaps:** (1) the milestone epochs are extra hyperparameters tied to the specific budget and
must be retuned if the budget or model changes; (2) each drop is a *discontinuity* — the rate
falls by a large factor in one step, which shocks the training dynamics (and, with momentum,
mismatches the velocity to the new rate) at exactly the moments it fires; (3) it leaves the
*start* of training untreated — the very first step already uses the full rate, so it does
nothing about the early-instability observations above.

**Constant low-rate prefix (He et al. 2016).** Prepend a short phase of a low *constant* rate
before resuming the normal (e.g. step-decay) schedule, to get past early instability. **Gap:**
as observed above, the abrupt return from the low prefix to the full rate re-creates the
instability it was meant to prevent — the training error spikes at the transition out of the
prefix and training can fail to recover. The prefix treats the symptom for a few epochs but the
discontinuity at its end reintroduces the disease.

**Restart-based schedules in optimization (O'Donoghue & Candes 2012; conjugate-gradient
restarts, Fletcher-Reeves / Powell 1977).** A separate lineage: in (accelerated) gradient and
conjugate-gradient methods, periodically *resetting* the method — discarding accumulated
momentum/history and restarting — provably restores the optimal convergence rate on
ill-conditioned problems, because overused momentum induces a periodic oscillation whose period
scales with the square root of the condition number. **Gap relative to this setting:** these are
restart *mechanisms* for deterministic (accelerated) optimization; they say to reset periodically
but do not, by themselves, prescribe the within-phase learning-rate curve for noisy minibatch
SGD on a deep net, nor address the start-of-training instability.

**Cyclical learning rates (Smith 2015/2017).** Let the rate oscillate between bounds on a
repeating (triangular) cycle rather than only decrease, on the argument that periodically
raising the rate helps traverse saddle-point plateaus. **Gap:** the triangular shape has sharp
corners (slope discontinuities at the peaks and troughs), and the method is framed around
repeated cycling rather than a single clean anneal to a small final rate; the cycle bounds and
period are themselves hyperparameters.

## Evaluation settings

The natural yardsticks already in use at the time, all pre-existing:

- **Datasets / models.** CIFAR-10 (32x32 color, 10 classes, 50k train / 10k test) with a
  CIFAR-style ResNet-20; CIFAR-100 (100 classes) with ResNet-56; FashionMNIST (28x28
  grayscale, upsampled, 10 classes) with a CIFAR-adapted MobileNetV2. These are the standard
  small-image classification benchmarks; the schedule must work across architectures (residual
  vs inverted-residual) and dataset difficulties without per-case retuning.
- **Optimizer / protocol (fixed).** SGD with momentum 0.9 and weight decay 5e-4, minibatch 128,
  `base_lr = 0.1`, Kaiming-normal initialization, cross-entropy loss, 200 epochs. Data
  augmentation: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` + per-channel
  normalization. There is no built-in PyTorch scheduler — the schedule is realized by setting
  the optimizer's learning rate manually, once per epoch.
- **Metric.** Best test accuracy (%) achieved during training (higher is better). The schedule
  may not modify model code, augmentation, loss, optimizer type, weight decay, or evaluation.

## Code framework

The schedule plugs into a fixed single-GPU training harness. The model, the data pipeline, the
optimizer (SGD + momentum + weight decay), and the per-epoch loop already exist and are not to
be touched. Each epoch the loop calls a single function to obtain the learning rate for that
epoch and writes it into every parameter group before training on the epoch. The one empty slot
is the body of that function: given the epoch index, the total budget, the base rate, and a
small config describing the architecture/dataset, return the rate to use. Nothing about the
*shape* of `eta(t)` is settled — that curve is exactly what is to be designed.

```python
import math


def get_lr(epoch, total_epochs, base_lr, config):
    """Return the learning rate for this epoch.

    Called once per epoch; the returned float is written into every SGD
    parameter group for the epoch.

    epoch:        current epoch, 0-indexed (0 .. total_epochs-1)
    total_epochs: total number of training epochs (e.g. 200)
    base_lr:      the configured base learning rate (e.g. 0.1)
    config:       {'arch': str, 'dataset': str}
    """
    # TODO: the learning-rate curve eta(epoch) we will design.
    return base_lr


# ---- the surrounding harness already exists; not edited ----

def main_loop(model, train_loader, test_loader, optimizer, criterion,
              device, total_epochs, base_lr, config):
    best_acc = 0.0
    for epoch in range(total_epochs):
        lr = get_lr(epoch, total_epochs, base_lr, config)   # the slot above
        for pg in optimizer.param_groups:                   # apply to SGD
            pg['lr'] = lr
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        best_acc = max(best_acc, test_acc)
    return best_acc
```

The harness supplies `epoch`, `total_epochs`, `base_lr`, `config` and consumes a single float;
`get_lr` is where the schedule will live.
