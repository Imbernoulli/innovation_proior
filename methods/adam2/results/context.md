## Research question

The objectives that drive machine learning are scalar functions `f(theta)` we can only ever
observe through noise, and we have to minimize them over millions of parameters. The noise is
not a single thing. Most of it comes from *minibatch subsampling*: the true objective is an
expectation over data, `f(theta) = E_x[f_x(theta)]`, and on each step we evaluate the gradient
on a random handful of examples, so each `g_t` is an unbiased but noisy single draw of the
gradient of `E[f]`. But some of it is *injected deliberately* — dropout, for instance,
randomizes which units are active on every forward pass, so even re-running the same example
gives a different gradient each time. Either way, the per-step gradient is an unreliable
sample of the quantity we actually want to descend. On top of that the models are enormous, so
anything that forms, stores, or inverts a matrix the size of `theta × theta`, or estimates a
Hessian, is impossible on a memory-bounded GPU — we are confined to first-order information
with memory that stays roughly linear in the number of parameters.

What a satisfactory optimizer would have to deliver, all at once: (1) only first-order
gradients and little memory; (2) a separate step size *per parameter* rather than one global
rate; (3) robustness to very noisy gradients; (4) good behavior on *sparse, high-dimensional*
features, where a rare-but-informative gradient should still drive a large step; (5) good
behavior on *non-stationary* objectives, whose gradient scale and curvature drift as training
proceeds (minibatch noise, drifting data, and dropout all cause this); and (6) very little
tuning, ideally with a single step-size knob that is not tightly coupled to the arbitrary
magnitude of the loss. Each existing method below covers a *subset* of these and is known to
break on at least one of the others. Covering all six with one cheap, first-order rule is the
open problem.

## Background

By this point stochastic gradient-based optimization is the engine behind the rapid progress
in deep learning — large image classifiers (Krizhevsky et al. 2012), deep speech systems
(Hinton et al. 2012a; Deng et al. 2013; Graves et al. 2013) — and the workhorse recipe is SGD
with a hand-tuned global learning rate, usually with momentum. The recurring pain points,
documented across this work:

- **One global learning rate is structurally wrong, not just inconvenient.** A single step
  size has to be hand-picked and frequently re-scheduled by hand: too large diverges, too
  small crawls. Worse, the *right* scale genuinely differs across coordinates — weight-shared
  convolutional layers produce gradients of a very different magnitude than dense layers,
  because a conv filter's gradient is summed over every spatial location it touches, so no
  single rate can be correct for all parameters at once.
- **Gradients are very noisy**, from both minibatching and dropout (Hinton et al. 2012b), so a
  direction read off any one step is unreliable and some temporal smoothing is needed to
  recover the trend.
- **Sparse, high-dimensional features** are common in NLP / bag-of-words models: most features
  are zero on most examples, and when a rare feature finally fires its gradient carries a lot
  of information and should produce a large step. This is observed to be exactly the regime
  where plain SGD with a global rate is slow.
- **Non-stationary objectives.** Minibatch noise, curvature that shifts as the weights move,
  and dropout reshuffling the effective network make the relevant gradient scale drift over
  training, so any statistic accumulated over the *entire* history goes stale and stops
  describing the current geometry.
- **Second-order / quasi-Newton methods do not scale** on a GPU: curvature estimates are
  accurate but memory-hungry, and methods that store per-minibatch curvature information grow
  in memory with the number of minibatch partitions.

Several conceptual frames are on the table. The **moment** view of gradient statistics: an
exponentially decaying running average of the gradient is an estimate of its first moment (the
mean direction), and an EMA of the squared gradient is an estimate of its second raw moment
(the uncentered variance / per-coordinate RMS). A general fact about any such EMA is that its
effective window length is set by its decay rate — a decay `beta` keeps an effective memory of
roughly `1/(1-beta)` recent samples, so `beta = 0.9` averages on the order of ten samples and
`beta = 0.999` on the order of a thousand. The **online convex optimization / regret** frame
(Zinkevich 2003): treat the optimizer as committing to a point `theta_t` before an adversary
reveals a convex cost `f_t`, and measure it by regret against the best fixed point over the
whole sequence,

```
R(T) = sum_{t=1}^{T} [ f_t(theta_t) - f_t(theta*) ],   theta* = argmin_theta sum_t f_t(theta).
```

Zinkevich showed projected online gradient descent with step `eta_t = t^{-1/2}` attains
`O(sqrt(T))` regret on any bounded-gradient convex sequence, so average regret `R(T)/T -> 0`. A
standard tool in this frame is the first-order convexity bound `f_t(theta_t) - f_t(theta*) <=
g_t^T(theta_t - theta*)` (a convex function lies above its tangent), which converts regret into
a sum over the algorithm's own updates. The **natural gradient** frame (Amari 1998; Pascanu &
Bengio 2013): the diagonal of the Fisher information is the expected per-coordinate squared
gradient, and natural-gradient preconditioning multiplies the step by the inverse Fisher.

## Baselines

The prior methods a new optimizer would be measured against and would react to.

**SGD with momentum / Nesterov (Sutskever et al. 2013).** Classical momentum keeps an
exponentially decaying running average of the gradient and steps along it:

```
v_t     = mu * v_{t-1} + g_t
theta_t = theta_{t-1} - alpha * v_t
```

In moment language `v_t` is a (biased, un-normalized) estimate of the gradient's first moment:
it averages out minibatch noise so one bad draw doesn't yank the trajectory, builds speed along
consistently downhill directions, and damps oscillation across a narrow ravine because the
sideways components alternate sign and average to near zero while the downhill component
accumulates. Nesterov accelerated gradient evaluates the gradient at a look-ahead point
`theta + mu*v`, which is more responsive and more stable at high momentum. Sutskever, Martens,
Dahl & Hinton (ICML 2013) showed that a well-initialized net with a carefully *scheduled*
momentum — ramped up early, then *reduced toward the end of training* — can rival Hessian-free
second-order optimization. **Gap:** still one global learning rate, no per-parameter scaling;
the average is over the *raw* gradient, so coordinates of very different gradient magnitude are
all stepped at the same scale.

**AdaGrad (Duchi, Hazan & Singer, JMLR 2011).** Give every parameter its own rate, shrunk by
the *accumulated* sum of that coordinate's squared gradients:

```
theta_{t+1,i} = theta_{t,i} - alpha * g_{t,i} / sqrt( sum_{s=1}^{t} g_{s,i}^2 )
```

A rarely-active coordinate keeps a small accumulated denominator and so takes a large step when
it finally fires; a frequently-active coordinate takes small steps — excellent for sparse,
high-dimensional data, where most of the information lives in the rare nonzero gradients.
Analyzed in the regret framework it attains `O(sqrt(T))` regret, and for sparse data the
adaptive bound is roughly `O(log d · sqrt(T))` rather than `O(sqrt(d·T))` for non-adaptive SGD.
**Gap:** the denominator is a *monotonically growing sum* that never forgets. Over a long run
`sqrt(sum g^2)` keeps climbing, so the effective per-coordinate learning rate
`alpha / sqrt(sum g^2)` decays monotonically toward zero and learning stalls — tolerable
annealing on a stationary convex problem, but fatal on the non-stationary, non-convex
objectives of deep training, where the optimizer must keep moving as the geometry drifts.

**RMSProp (Tieleman & Hinton, Coursera Lecture 6.5, 2012; momentum variant Graves 2013).**
Cure AdaGrad's vanishing rate by replacing the growing sum with an exponential moving average
of the squared gradient:

```
v_t     = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
theta_t = theta_{t-1} - alpha * g_t / ( sqrt(v_t) + eps )
```

The EMA forgets old gradients, so the denominator tracks the *recent* gradient scale (a
windowed second-moment estimate rather than a cumulative one) — good on non-stationary and
online problems. Graves (2013) added a momentum term applied to the already-rescaled gradient.
**Gaps:** (1) the `v_t` recursion is started at `v_0 = 0`, and in practice when `beta_2` is
pushed close to 1 — which a smooth, reliable second-moment estimate on sparse gradients
demands — RMSProp's first steps are observed to be very large and training can diverge early;
(2) the momentum, where used, sits on the *rescaled* gradient rather than on a separately
maintained estimate of the gradient direction itself.

**AdaDelta (Zeiler, 2012).** Same EMA-of-squared-gradient denominator as RMSProp, plus an EMA
of the squared *parameter updates* whose square root goes in the numerator, so the update is
dimensionally consistent and no global learning rate is needed. **Gap:** the unit-matching is a
heuristic, and like RMSProp it inherits the same zero-initialized EMA denominator.

**Quasi-Newton on minibatches (Roux & Fitzgibbon 2010; Sohl-Dickstein et al. 2014).** The
Sum-of-Functions Optimizer (SFO) estimates per-minibatch curvature for near-second-order
progress. **Gap:** memory grows linearly in the number of minibatch partitions — often
infeasible on a memory-constrained GPU — and it assumes deterministic subfunctions, so it does
not cope with dropout-style stochastic regularization.

## Evaluation settings

The natural yardsticks already in use, into which a new optimizer would be dropped:

- **Logistic regression (convex)** on MNIST (784-dim image vectors, 10 classes), with `L2`
  regularization and minibatch size 128; metric is training negative log-likelihood vs.
  iterations / epochs over the dataset. A convex objective isolates optimizer behavior from
  local-minimum effects, and the step size is often annealed as `alpha_t = alpha / sqrt(t)` to
  match the `t^{-1/2}` schedule from online-convex theory.
- **Sparse-feature logistic regression** on the IMDB movie-review dataset (Maas et al. 2011),
  pre-processed into 10,000-dim bag-of-words vectors (highly sparse), commonly with 50% dropout
  on the BoW features (Wang & Manning 2013) — the natural test for the sparse-gradient regime.
- **Multilayer fully-connected nets** (e.g. two hidden layers of 1000 ReLU units each),
  minibatch 128, with `L2` weight decay, evaluated both deterministically and with dropout
  stochastic regularization; training cost vs. epochs. A non-convex setting where convergence
  theory no longer formally applies.
- **Convolutional nets** on CIFAR-10 (e.g. three 5×5 conv + 3×3 max-pool stages then a
  1000-unit dense layer), whitened inputs, dropout on input and dense layers, minibatch 128;
  training cost over early and longer runs. Weight sharing makes per-layer gradient scales
  differ sharply, stressing per-parameter adaptation.
- **Variational autoencoder** (Kingma & Welling 2013), single hidden layer of 500 softplus
  units, 50-dim spherical-Gaussian latent — used as a setting to sweep the decay rates and
  `log10(alpha)` over a dense grid and watch the effect of design choices on training loss.
- Protocol: identical parameter initialization across optimizers; learning rate / momentum
  searched over a dense grid; comparisons read off the best hyperparameter setting.

## Code framework

The optimizer plugs into the same minibatch-SGD training harness already used for the
baselines. Nothing about the per-parameter update rule is settled — that rule is exactly what
is to be designed — so the substrate is only the generic first-order machinery that already
exists: an `Optimizer` object that owns per-parameter state and exposes a `step()` consuming
the freshly computed gradients and writing a small update back into the parameters, plus an
outer loop that draws a minibatch, runs the existing model and loss, backpropagates to fill in
each parameter's gradient, and calls `step()`. The single empty slot is the update rule itself.

```python
import torch


class Optimizer:
    """Generic first-order stochastic optimizer. Owns per-parameter state and applies a
    per-parameter update from the current gradient. Memory must stay ~linear in the number
    of parameters (no theta x theta matrix, no Hessian)."""

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        # lazily-initialized per-parameter buffers, if the update rule needs any
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad                       # one noisy gradient sample for this parameter
            state = self.state[id(p)]
            # TODO: the per-parameter update rule we will design.
            #       Given the stream of noisy gradients g (and any per-parameter state
            #       we choose to keep), compute the update and apply it:
            #       p += <update>(g, state)
            pass


# existing minibatch training loop the optimizer plugs into
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:        # draw a minibatch
        optimizer.zero_grad()
        outputs = model(inputs)                # forward through the existing model
        loss = loss_fn(outputs, targets)       # existing loss (may include dropout noise)
        loss.backward()                        # backprop fills p.grad for every parameter
        optimizer.step()                       # apply the per-parameter update rule
```

The outer loop supplies one gradient sample per parameter; `step()` is where the per-parameter
rule will live.
