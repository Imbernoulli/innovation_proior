# Context: learning-rate scheduling for deep-net training (circa 2015-2016)

## Research question

Training a deep convolutional classifier — a wide residual network on CIFAR-10/100, say — is, in
practice, governed by two hand-tuned knobs: the learning-rate *schedule* and the amount of L2 weight
decay. The optimizer itself is almost always plain stochastic gradient descent with momentum; the model,
the data augmentation, the batch size, and the weight decay are held fixed by convention. What is left to
decide is how the scalar learning rate `eta_t` should evolve over the run. A single constant rate is
unsuitable: too large and training is unstable or never settles, too small and it crawls. So the rate is
made to *change* over training, and the question is the precise shape of that change. The optimizer, loop,
and other hyperparameters stay fixed; only the schedule changes.

## Background

Deep nets are the best-performing method for image and speech recognition, and their training on large
datasets is the computational bottleneck — often days even on high-end GPUs — so any acceleration of
training is valuable. The training of a net with `n` parameters is the minimization of `f: R^n -> R`,
done by the stochastic gradient step

```
x_{t+1} = x_t - eta_t * grad f_t(x_t),
```

where `grad f_t` is the gradient on a small `t`-th minibatch and `eta_t` is the learning rate. One could
use second-order information, `x_{t+1} = x_t - eta_t H_t^{-1} grad f_t(x_t)`, but forming and
inverting the Hessian `H_t` is intractable for large `n`, and limited-memory quasi-Newton (L-BFGS, Liu &
Nocedal 1989) is little used in deep learning given the stochasticity of `grad f_t`, the
ill-conditioning of `f`, and the abundance of saddle points produced by the hierarchical structure of the
parameter space (Fukumizu & Amari 2000; Dauphin et al. 2014). Diagonal adaptive methods such as AdaDelta
(Zeiler 2012) and Adam (Kingma & Ba 2014) approximate the inverse Hessian cheaply. The state-of-the-art
residual networks (He et al. 2015; Huang et al. 2016; Zagoruyko & Komodakis 2016) were trained with plain
SGD with (Nesterov) momentum,

```
v_{t+1} = mu_t * v_t - eta_t * grad f_t(x_t),
x_{t+1} = x_t + v_{t+1},
```

with `v` initialized to zero, `eta_t` a decreasing learning rate, and `mu_t` a momentum coefficient
trading off the current gradient against the accumulated velocity. The lever that moves
state-of-the-art results in this regime is the *schedule* of `eta_t`.

Several conceptual frames sit in the background.

**The two-phase intuition of decaying rates.** A high learning rate takes large steps that traverse the
loss landscape broadly and quickly cross flat, low-gradient regions; a low learning rate takes small steps
that settle into and fine-tune a minimum. Any decreasing schedule is, at heart, a transition from a
coarse exploratory phase to a fine refining phase. The staircase implements this transition as a few
sudden drops.

**Restarts in gradient-free optimization.** When minimizing multimodal functions with evolution
strategies such as CMA-ES, a standard device is to *restart* the search repeatedly, and to increase the
search effort across restarts — e.g. start with a small population `lambda` and double it after each
restart (Hansen 2009). Starting small and growing the budget per restart gives a decent solution quickly,
while the later, larger restarts do the global search. This "increase the per-restart budget, often by
doubling" pattern is well established in that community.

**Restarts in gradient-based optimization.** Classical first-order methods restart too. The conjugate
gradient method flushes its history every `n` or `n+1` iterations (Fletcher & Reeves 1964), and Powell
(1977) proposed restarting when consecutive gradients lose enough orthogonality. The most directly
relevant analysis is O'Donoghue & Candès (2012) on accelerated (momentum / heavy-ball) gradient schemes
for smooth strongly-convex `f`. They observe that such a scheme has two regimes depending on the momentum
coefficient `beta`: below a critical value `beta_i* = (1 - sqrt(lambda_i/L)) / (1 + sqrt(lambda_i/L))` the
mode is over-damped and converges slowly and monotonically; above it the mode is under-damped and the
iterates *ripple* — the objective shows regular bumps. Analyzing the scheme on a quadratic `f(x) = (1/2)
x^T A x` as a linear dynamical system, they find the under-damped modes oscillate, and the frequency of
the mode tied to the smallest eigenvalue `mu` is `psi_mu ≈ sqrt(mu/L)` — so the *period* of the ripples is
proportional to the square root of the condition number `L/mu`. Because the right momentum depends on a
condition number that is unknown and varies locally across the landscape, one usually runs in the
high-momentum rippling regime. Their device is to **restart** — reset the momentum to zero and take the
current iterate as a fresh starting point — periodically. If `L` and `mu` were known, a fixed restart
every `k* = e * sqrt(8L/mu)` iterations recovers the optimal linear rate
`O(sqrt(L/mu) log(1/eps))`; and two
adaptive heuristics restart whenever a cheap signal fires — the *function scheme* when `f(x^k) >
f(x^{k-1})`, the *gradient scheme* when `grad f(y^{k-1})^T (x^k - x^{k-1}) > 0` (the momentum and the
negative gradient make an obtuse angle, i.e. momentum is carrying us uphill). The lesson carried by this
work: restarting a momentum method can recover fast convergence when the curvature is unknown and
drifts, with fixed restarts exposing the square-root-condition-number timescale and adaptive restarts
removing the need to know `mu`; a restart is fundamentally a *reset of the acceleration phase*.

**Cyclical learning rates.** Smith (2015) demonstrated an empirical phenomenon for deep nets:
letting the global learning rate *rise and fall* cyclically between a lower and an upper bound — rather
than only decrease — is beneficial overall, even though each rise temporarily harms performance. His
triangular policy ramps `eta` linearly up from `base_lr` to `max_lr` and linearly back down over a cycle
of length `2 * stepsize`, with optional variants that shrink the band or decay it exponentially per cycle.
The rationale leans on the saddle-point picture (Dauphin et al. 2014): the difficulty in deep
optimization is plateaus around saddle points, whose small gradients stall progress, and a *raised*
learning rate accelerates traversal of those plateaus. The bounds are found with a cheap "LR range test"
(sweep `eta` upward for a few epochs and watch where accuracy starts and stops improving).

## Baselines

These are the schedules a new schedule would be measured against. They share a fixed
substrate: SGD with momentum, fixed weight decay, the per-epoch (or per-iteration) learning rate set by
the schedule.

**Multi-step (staircase) decay — ResNet (He et al. 2015).** Hold `eta` constant, divide it by 10 when the
error plateaus. Concretely on CIFAR-10: start at `eta_0 = 0.1`, divide by 10 at 32k and 48k iterations,
terminate at 64k, with momentum 0.9 and weight decay 1e-4. Core idea: implement the coarse-to-fine
transition as two or three discrete drops.

**Multi-step decay — Wide Residual Networks (Zagoruyko & Komodakis 2016).** The same family, tuned for
WRNs and the standard CIFAR setup in use: SGD with Nesterov momentum 0.9, initial
`eta_0 = 0.1`, weight decay 5e-4, dampening 0, minibatch 128, the learning rate dropped by a factor of
0.2 at epochs 60, 120 and 160, for a total of 200 epochs; data augmentation is horizontal flips and
4-pixel-pad random crops. This is the concrete state-of-the-art recipe.

**Cyclical (triangular) learning rates (Smith 2015).** Oscillate `eta` linearly between `base_lr` and
`max_lr` with cycle length `2 * stepsize`; pick the bounds with the LR range test. Core idea: a varying
rate that periodically goes *up* helps overall, with the bounds set by a cheap sweep rather than fine
tuning.

**Adaptive diagonal optimizers (AdaDelta, Zeiler 2012; Adam, Kingma & Ba 2014).** Rescale each
coordinate's step by running gradient statistics, reducing manual rate tuning. They still run with some
global `eta_t` schedule; in this regime the strongest image-classification results came from plain
SGD+momentum with a hand-scheduled rate.

## Evaluation settings

The natural yardsticks already in use for this kind of schedule, all pre-existing:

- **Wide Residual Networks on CIFAR-10 and CIFAR-100** (Krizhevsky 2009): 32x32 color images, 10 and 100
  classes, 50,000 train / 10,000 test. The reference architecture is WRN-28-10 (depth 28, width factor
  10). Preprocessing is per-pixel mean subtraction (or global contrast normalization + ZCA whitening);
  augmentation is horizontal flips and 4-pixel-pad random crops. Optimizer: SGD with momentum 0.9, weight
  decay 5e-4, minibatch 128. Total budget on the order of 200 epochs. Metric: test classification error.
- **Wider / deeper variants** (e.g. WRN-28-20) under the same protocol, to probe whether a faster schedule
  lets one train a larger network in the same wall-clock budget.
- **A convolutional pipeline on a dataset of EEG recordings** (Schirrmeister et al. 2017) — right/left
  hand and foot movement classification, ~1000 trials per subject across 14 subjects — as an
  out-of-domain check that a schedule generalizes beyond vision.
- **A downsampled (32x32) ImageNet** with all 1000 classes, same augmentation as CIFAR, as a larger-scale
  stress test; metric top-1 / top-5 error.
- Protocol: identical model, augmentation, weight decay, optimizer, and batch size across schedules;
  only `eta_t` differs; runs reported as medians over a few seeds; the schedule called once per epoch (or
  per batch) to set the rate.

## Code framework

The schedule plugs into the existing SGD-with-momentum training harness already used for the staircase
baseline. The optimizer, the model, the loss, the data pipeline, and the weight decay are all fixed and
pre-existing; the only thing not yet decided is the function that maps the current point in training to a
scalar learning rate. So the substrate is just the generic training loop plus one empty slot: a
`get_lr(...)` callable that, given how far along training is, returns the rate to use, which the loop
writes into the optimizer before each update.

```python
def get_lr(epoch, total_epochs, base_lr, config):
    """Return the scalar learning rate to use for this epoch.

    Inputs available before any schedule is chosen:
      epoch        : how far into training we are (0-indexed; updated each epoch,
                     and can be made fractional if updated per batch)
      total_epochs : the total training budget
      base_lr      : the initial / reference learning rate
      config       : architecture / dataset descriptors, if the schedule wants them
    """
    # TODO: the learning-rate schedule we will design.
    #       Map the current training progress to a scalar rate.
    pass


# existing SGD+momentum training loop the schedule plugs into
def train(model, loss_fn, data_loader, optimizer, total_epochs, base_lr, config):
    for epoch in range(total_epochs):
        lr = get_lr(epoch, total_epochs, base_lr, config)   # set this epoch's rate
        for group in optimizer.param_groups:
            group['lr'] = lr
        for inputs, targets in data_loader:                 # draw a minibatch
            optimizer.zero_grad()
            outputs = model(inputs)                          # forward through the fixed model
            loss = loss_fn(outputs, targets)                 # fixed loss
            loss.backward()                                  # backprop fills gradients
            optimizer.step()                                 # SGD+momentum update with this lr
```

The loop supplies the current progress; `get_lr` is where the schedule will live.
