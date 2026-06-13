# Context: learning-rate scheduling for training deep networks (circa 2015-2017)

## Research question

Training a deep convolutional network with SGD is dominated, in practice, by one knob: the
learning rate. Set it too high and training diverges; set it too low and it crawls, and the
characteristic curve for a fixed rate is the same every time — a burst of progress, then a
long plateau where each iteration buys almost nothing. The standard remedy is to run a large
rate for many epochs until the test accuracy plateaus, drop the rate by a factor of ten, and
repeat two or three times. That works, but it spends the overwhelming majority of its
iterations in those flat plateaus, it demands hand-chosen milestones, and it commits to the
prevailing wisdom that the learning rate must stay *small* — below some "maximum safe value"
— for the whole run.

The precise goal is a learning-rate schedule that (1) reaches a given test accuracy in far
fewer iterations than the piecewise-constant recipe; (2) avoids trading speed for final
generalization; (3) needs little per-dataset, per-architecture tuning, with a principled way
to pick its bounds from a single short pre-run rather than a grid search; (4) keeps the optimizer type,
the loss, the data pipeline, and the per-step update rule of SGD-with-momentum exactly as
they are — only the per-iteration rate (and, optionally, the momentum coefficient) changes;
and (5) generalizes well, not merely trains fast. The existing schedules below each get part
of this; none resolves the speed, tuning, and generalization requirements together. Closing
that gap is the problem.

## Background

By this point SGD with a hand-tuned, hand-scheduled learning rate is the engine of deep
learning, and there is an extensive literature on it (Goodfellow et al. 2016; Bottou 2012)
and on the geometry of the loss surfaces it traverses (Chaudhari et al. 2016). Several
threads in that background are load-bearing here.

**The loss-topology picture.** Goodfellow, Vinyals & Saxe (2014), characterizing the loss
landscape, visualized a training trajectory as a path that makes steep, fast progress in the
first few iterations, then enters a long, nearly flat valley where the slope — and therefore
the per-iteration progress — is tiny, and only at the very end maneuvers through a narrow
trough to a local minimum. Dauphin et al. (2014) argued that the difficulty of minimizing
these losses comes from *saddle points* — regions of small gradient that stall descent —
rather than from poor local minima. A saddle has small gradients in some directions; a small
learning rate barely moves through it, while a larger rate traverses it quickly.

**Large learning rates and generalization.** A growing body of work tied the learning rate
to the *width* of the minimum SGD finds, and width to generalization. Keskar et al. (2016)
reported that small mini-batches lead to wide, flat minima and large batches to sharp minima,
with flat minima generalizing better (echoing Hochreiter & Schmidhuber 1997). Jastrzębski et
al. (2017) showed that the ratio of learning rate to batch size, together with the gradient
variance, controls the width of the minimum, and that *higher* SGD noise drives solutions
toward better-generalizing regions. Smith & Le (2017) made the noise quantitative, deriving
a gradient-noise scale

```
g  ≈  eps * N / ( B * (1 - m) )
```

where `eps` is the learning rate, `N` the training-set size, `B` the batch size, and `m` the
momentum coefficient — so a larger learning rate, a smaller batch, or a larger momentum all
increase the same effective noise, and Smith et al. (2017) / Jastrzębski et al. (2017)
independently showed that *decaying the learning rate* and *increasing the batch size* are
two ways to do the same thing. The relevant quantity for training is the noise scale `g`, not
`eps` or `B` alone. A definition of regularization in common use (Goodfellow et al. 2016) is
"any modification to a learning algorithm intended to reduce its generalization error but not
its training error" — a lens that will matter when reading the diagnostic curves below.

**A diagnostic phenomenon, knowable from a single pre-run.** It is possible to learn a great
deal about a network's tolerance for learning rates from one short run in which the rate is
swept linearly upward from near zero. Plotting test accuracy against the rising rate gives a
curve that climbs, holds, and then — once the rate is too large — turns ragged and falls.
The position of that turn is a per-architecture fact about the largest usable rate. Two
observations from such sweeps are the empirical seeds of everything that follows. First, on
some deep residual networks the accuracy stays high over a strikingly wide band of large
rates, far past the conventional "safe" ceiling. Second, over a band of large rates the
*training* loss rises while the *test* loss falls — the generalization gap shrinks as the
rate grows, which, by the definition above, is the signature of regularization happening *by
the learning rate itself*.

**Curriculum and annealing.** Two old ideas frame how a rate should move over a run.
Curriculum learning (Bengio et al. 2009) starts easy and ramps up difficulty; simulated
annealing (Aarts & Korst 1988) injects large perturbations early and cools to a quiet,
low-temperature search at the end. A learning-rate schedule that rises and then falls is a
discrete cousin of both.

**Momentum, mechanically.** SGD with momentum maintains a velocity and steps along it,

```
v_{i+1}     = m * v_i  -  eps * grad L(theta_i)
theta_{i+1} = theta_i  +  v_{i+1}
```

so the velocity is a moving average of the gradient and the momentum coefficient `m` scales
how much past gradients carry forward. Because the update magnitude depends on both `eps` and
`m`, momentum and learning rate are coupled — they push on the same lever. Sutskever et al.
(2013) had already shown that a *scheduled* momentum, ramped and then eased back, matters for
deep nets.

## Baselines

These are the prior schedules a new one would be measured against and would react to.

**Piecewise-constant / step decay (He et al. 2016).** Hold a global rate (commonly ≈0.1) for
many epochs until the test accuracy plateaus, then divide it by ten and continue, repeating
two or three times. This is the default training regime for residual networks. **Gap:** each
constant-rate segment shows the same shape — a quick rise then a long plateau — so most of
the iterations are spent in plateaus making little progress; the milestone epochs are
hand-chosen per dataset and architecture; and the whole schedule keeps the rate at
conventionally small values throughout, never probing the large-rate band where the
diagnostic sweeps show high accuracy and a shrinking generalization gap.

**Cyclical learning rates, triangular policy (Smith 2015).** Rather than a single fixed or
monotonically decaying rate, let the rate oscillate linearly between a lower and an upper
bound, back and forth, for the whole run. With `stepsize` the number of iterations in a
half-cycle, the triangular schedule is

```
cycle = floor( 1 + iter / (2 * stepsize) )
x     = | iter / stepsize  -  2 * cycle  +  1 |
lr    = base_lr + (max_lr - base_lr) * max(0, 1 - x)
```

so `lr` sweeps from `base_lr` up to `max_lr` and back every `2 * stepsize` iterations. The
motivating insight is that a temporary increase in the rate can hurt in the short term yet
help overall — a larger rate traverses saddle-point plateaus that a small rate stalls on.
This work also introduced the linear *LR range test*: sweep the rate upward over one short
run and read the largest usable rate off the accuracy curve. Variants halve the amplitude
each cycle (`triangular2`) or decay the bounds geometrically (`exp_range`). **Gap:** it runs
many full up-and-down cycles and oscillates between two fixed bounds for the entire run; the
rate never descends far *below* the lower bound, and on its own the policy does not deliver an
order-of-magnitude reduction in iterations.

**Cosine annealing with warm restarts, SGDR (Loshchilov & Hutter 2016).** Decay the rate
along a cosine from a high value to a small one, then jump it back up to the top and decay
again — a sawtooth of cosine descents. **Gap:** the periodic jumps back to the maximum are a
restart pattern, not a single rise-hold-fall traversal; the schedule still spends the run
cycling rather than ramping once and then driving the rate far down at the end.

**Linear warmup then decay (He et al. 2016; Goyal et al. 2017).** Begin with a small rate,
increase it linearly over the first few epochs to a target, then decay (step or cosine).
Warmup stabilizes the high rates needed for large-batch training, and is in effect a
discretized version of an increasing-rate phase. **Gap:** it is only the up-ramp followed by
ordinary decay; there is no single cycle that rises, falls, *and then* anneals the rate orders
of magnitude below where it started, and the momentum coefficient is left constant throughout.

**Adaptive per-parameter methods (AdaGrad, Duchi et al. 2011; AdaDelta, Zeiler 2012; Adam,
Kingma & Ba 2014; Nesterov momentum, Sutskever et al. 2013).** These set effective
per-parameter rates from gradient statistics instead of a single schedule. **Gap:** the
effective rates they use, where they work well, are not in the very-large-rate band; they are
designed around small, stable steps and do not, by themselves, exploit the large-rate regime
that the diagnostic sweeps reveal.

## Evaluation settings

The natural yardsticks already in use for image-classification schedules:

- **CIFAR-10** (50,000 training / 10,000 test 32×32 color images, 10 classes) and
  **CIFAR-100** (100 classes), trained with residual networks of varying depth (ResNet-20,
  -56, -110, and the `20 + 9n` family), wide ResNets, and DenseNets, with standard
  augmentation (random crop with 4-pixel padding, random horizontal flip).
- **MNIST** (handwritten digits) with a small LeNet-style convolutional network.
- **ImageNet** (ILSVRC, ~1.28M training images, 1000 classes) with ResNet-50 and
  Inception-ResNet-v2, top-1 accuracy.
- Optimizer: SGD with momentum (commonly `m = 0.9`) and weight decay; mini-batch training.
  The schedule is the only thing that varies — same model, loss, augmentation, and update
  rule across compared runs.
- Metric: best / final test accuracy, read against the *number of iterations or epochs* to
  expose training speed. The diagnostic LR range test is run as a short pre-training sweep.
- Protocol: identical initialization across compared schedules; learning-rate bounds are
  chosen by a single short LR range test rather than a full grid search.

## Code framework

The schedule plugs into the SGD-with-momentum training loop already used for the baselines.
What shape the per-iteration rate should take over the run — and whether and how the momentum
coefficient should move with it — is exactly what is to be designed, so the substrate is only
the generic machinery that already exists: an SGD-with-momentum optimizer whose `lr` (and
`momentum`) can be set each step, an outer loop that draws a batch and steps, and the
single empty slot that maps the current training progress to the rate to use. A second stub
holds the short diagnostic pre-run that would supply the schedule's bounds.

```python
import math


class SGD:
    """Existing SGD-with-momentum optimizer. lr and momentum are settable each step;
    the update rule (v <- m*v - lr*grad ; theta += v) is fixed and not what we design."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = [zeros_like(p) for p in self.params]

    def step(self, grads):
        for p, g, v in zip(self.params, grads, self.v):
            if self.weight_decay:
                g = g + self.weight_decay * p
            v *= self.momentum
            v -= self.lr * g
            p += v


def get_schedule_value(step, total_steps, low_lr, high_lr):
    """Map the current training progress to the (learning rate[, momentum]) to use
    this step. The shape of this map over the whole run is the contribution."""
    # TODO: the schedule we will design.
    pass


def lr_range_test(model, loss_fn, data_loader, lr_start, lr_end, num_steps):
    """One short pre-run that sweeps the learning rate upward to probe how the
    network responds across a range of rates."""
    # TODO: the diagnostic sweep we will design, and how to read its result.
    pass


# existing minibatch training loop the schedule plugs into
def train(model, loss_fn, data_loader, optimizer, total_steps):
    step = 0
    for inputs, targets in cycle(data_loader):
        if step >= total_steps:
            break
        lr = get_schedule_value(step, total_steps, low_lr, high_lr)   # set rate for this step
        optimizer.lr = lr
        grads = backprop(model, loss_fn, inputs, targets)             # forward + backward
        optimizer.step(grads)                                         # SGD-with-momentum update
        step += 1
```

The outer loop hands `get_schedule_value` the current step and the run length; that function,
and the diagnostic sweep that sets its bounds, are where the contribution lives.
