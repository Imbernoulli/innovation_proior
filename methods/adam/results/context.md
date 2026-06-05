# Context: first-order stochastic optimization landscape (circa 2012-2014)

## Research question

Many objectives in machine learning are scalar functions `f(theta)` we can only ever see
through noise. The noise has two distinct sources. The first is minibatch subsampling: the
gradient is computed on a random handful of datapoints, so each `g_t` is a noisy, unbiased
draw of the true gradient of the expected loss `E[f(theta)]`. The second is *deliberately
injected* stochasticity such as dropout regularization, which randomizes the network on
every forward pass, so even the very same training example yields a different gradient on
each pass. Both make the per-step gradient an unreliable single sample of the quantity we
actually want to descend. On top of that the models are large — millions of parameters — so
anything that forms, stores, or inverts a matrix the size of `theta × theta`, or needs a
Hessian, is ruled out on a memory-constrained GPU. We are confined to first order, with
memory that stays close to linear in the number of parameters.

The precise goal is a single optimizer that simultaneously: (1) uses only first-order
gradients and little memory; (2) gives each parameter its own step size rather than one
global rate; (3) is robust to very noisy gradients; (4) handles sparse, high-dimensional
features, where a rare but informative gradient should still produce a large, useful step;
(5) handles *non-stationary* objectives whose gradient scale and curvature drift over the
course of training (minibatching, dropout, and changing data all cause drift); and (6) needs
little hyperparameter tuning, ideally with a step-size knob that is interpretable as a
distance in parameter space rather than something coupled to the arbitrary magnitude of the
loss. Each of the existing methods below achieves a subset of these; none achieves all six at
once. Closing that gap is the problem.

## Background

By this time stochastic gradient-based optimization is the engine driving the rapid progress
in deep learning — large image classifiers (Krizhevsky et al. 2012), deep speech systems
(Hinton et al. 2012a; Graves et al. 2013) — and the dominant recipe is SGD with a hand-tuned
learning rate, usually with momentum. The pain points:

- **Learning-rate tuning is painful and structurally wrong with one global rate.** A single
  step size has to be hand-picked and often re-scheduled by hand; too large diverges, too
  small crawls. Worse, the *right* scale differs across coordinates: weight-shared
  convolutional layers produce gradients of a very different magnitude than dense layers, so
  one global rate cannot be right for all parameters at once.
- **Gradients are very noisy**, from both minibatching and dropout (Hinton et al. 2012b), so
  a direction read off a single step is unreliable; some temporal smoothing is needed to
  recover the trend.
- **Sparse, high-dimensional features** are common in NLP / bag-of-words models: most
  features fire rarely, and when a rare feature finally produces a gradient, that gradient
  carries a lot of information and should drive a large step.
- **Non-stationary objectives**: minibatch noise and drifting data make the gradient
  signal's scale and curvature change over training, so any statistic accumulated over the
  *entire* history goes stale and stops reflecting the current geometry.
- **Second-order / quasi-Newton methods don't scale** on a GPU: estimating curvature is
  accurate but memory-hungry, and curvature methods that store per-minibatch information grow
  in memory with the number of minibatch partitions.

Two conceptual frames sit in the background and will matter throughout. First, the
**moment** view of gradient statistics: an exponentially decaying running average of the
gradient is an estimate of its first moment (the mean direction); a running average of the
*squared* gradient is an estimate of the second raw moment (the uncentered variance) — i.e. a
per-coordinate estimate of gradient magnitude. The *window length* of such an EMA is set by
its decay rate: a decay `beta` keeps an effective memory of roughly `1/(1-beta)` recent
samples, so `beta = 0.9` averages on the order of ten samples and `beta = 0.999` averages on
the order of a thousand. This window/precision tradeoff will govern the choice of decay
rates. Second, the **online convex optimization / regret** frame (Zinkevich 2003): treat the
optimizer as committing to a point `theta_t` before an adversary reveals a convex cost `f_t`,
and measure it by regret against the best fixed point for the whole sequence,

```
R(T) = sum_{t=1}^{T} [ f_t(theta_t) - f_t(theta*) ],   theta* = argmin_theta sum_t f_t(theta).
```

Zinkevich showed that projected online gradient descent with step `eta_t = t^{-1/2}`
achieves `O(sqrt(T))` regret for any bounded-gradient convex sequence, so average regret
`R(T)/T -> 0`. The analytic entry point is the first-order convexity bound
`f_t(theta_t) - f_t(theta*) <= g_t^T (theta_t - theta*)` (a convex function lies above its
tangent), which converts regret into a sum over the algorithm's own updates — the lever any
convergence proof for a new method would pull. A related frame is **natural gradient** (Amari
1998; Pascanu & Bengio 2013): a second-moment estimate of the gradient is related to the
diagonal of the Fisher information matrix, so dividing a step by the square root of that
estimate is a cheap, diagonal, *more conservative* cousin of natural-gradient preconditioning
(square root of inverse diagonal Fisher, rather than full inverse Fisher).

## Baselines

These are the prior methods a new optimizer would be measured against and would react to.

**SGD with momentum / Nesterov (Sutskever et al. 2013).** Classical momentum keeps an
exponentially decaying running average of the gradient and steps along it:

```
v_t     = mu * v_{t-1} + g_t
theta_t = theta_{t-1} - alpha * v_t
```

In moment language `v_t` is a (biased, un-normalized) estimate of the gradient's first
moment: it averages out minibatch noise, builds speed along consistently downhill
directions, and damps oscillation across ravines. Nesterov accelerated gradient evaluates the
gradient at a look-ahead point `theta + mu*v`, which is more responsive and more stable at
high momentum. Sutskever, Martens, Dahl & Hinton (ICML 2013) showed that well-initialized
nets with a carefully *scheduled* momentum — ramped up early, then *reduced toward the end of
training* — can rival Hessian-free second-order optimization; this is direct evidence both
that a smoothed first moment is genuinely part of the answer and that the momentum
coefficient should *decay* late in training. **Gap:** still one global learning rate; no
per-parameter scaling. The average is over the *raw* gradient, so coordinates with very
different gradient magnitudes are all stepped at the same scale.

**AdaGrad (Duchi, Hazan & Singer, JMLR 2011).** Give every parameter its own rate, shrunk by
the *accumulated* squared gradient of that coordinate:

```
theta_{t+1,i} = theta_{t,i} - alpha * g_{t,i} / sqrt( sum_{s=1}^{t} g_{s,i}^2 )
```

Rarely-active coordinates keep a small accumulated denominator and so take large steps when
they finally fire; frequently-active coordinates take small steps — excellent for sparse,
high-dimensional data. Analyzed in the regret framework, it attains `O(sqrt(T))` regret, and
for sparse data the adaptive bound is roughly `O(log d · sqrt(T))` versus `O(sqrt(d·T))` for
non-adaptive SGD. **Gap:** the denominator is a *monotonically growing sum* that never
forgets. Over a long run `sqrt(sum g^2)` keeps climbing, so the effective per-coordinate
learning rate decays monotonically toward zero and learning stalls — acceptable annealing on
a stationary convex problem, but fatal on the non-stationary, non-convex objectives of deep
training, where the optimizer must keep moving as the geometry drifts.

**RMSProp (Tieleman & Hinton, Coursera Lecture 6.5, 2012; momentum variant Graves 2013).**
Cure AdaGrad's vanishing rate by replacing the growing sum with an exponential moving average
(EMA) of the squared gradient:

```
v_t     = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
theta_t = theta_{t-1} - alpha * g_t / ( sqrt(v_t) + eps )
```

The EMA forgets old gradients, so the denominator tracks the *recent* gradient scale (a
windowed second-moment estimate, not a cumulative one) — good on non-stationary and online
problems. Graves (2013) added a momentum term, applied to the already-rescaled gradient.
**Gaps:** (1) no bias-correction term — because `v_0 = 0`, the early `v_t` is biased toward
zero, and when `beta_2` is pushed close to 1 (which a reliable second-moment estimate on
sparse gradients demands) this bias makes the early denominator far too small, so the first
steps are huge and training can diverge; (2) the momentum sits on the *rescaled* gradient
rather than on a clean, separately maintained first-moment estimate — there is no unifying
"estimate the first and second moments and combine them" story.

**AdaDelta (Zeiler, 2012).** Same EMA-of-squared-gradient denominator as RMSProp, plus an EMA
of the squared *parameter updates* whose square root goes in the numerator, making the update
dimensionally consistent so that no global learning rate is needed. **Gap:** still no
moment-estimation / bias-correction perspective; the unit-matching is a heuristic and shares
RMSProp's lack of principled de-biasing.

**Quasi-Newton on minibatches (Roux & Fitzgibbon 2010; Sohl-Dickstein et al. 2014).** The
Sum-of-Functions Optimizer (SFO) estimates per-minibatch curvature for near-second-order
progress. **Gap:** memory grows linearly in the number of minibatch partitions — often
infeasible on a memory-constrained GPU — and it assumes deterministic subfunctions, so it
does not cope with dropout-style stochastic regularization.

## Evaluation settings

The natural yardsticks already in use:

- **Logistic regression (convex)** on MNIST (784-dim image vectors, 10 classes), with `L2`
  regularization and minibatch size 128; metric is training negative log-likelihood vs.
  iterations / epochs over the dataset. A convex objective isolates optimizer behavior from
  local-minimum effects. The step size is often annealed as `alpha_t = alpha / sqrt(t)` to
  match the `t^{-1/2}` schedule from online-convex theory.
- **Sparse-feature logistic regression** on the IMDB movie-review dataset (Maas et al. 2011),
  pre-processed into 10,000-dim bag-of-words vectors (highly sparse), commonly with 50%
  dropout on the BoW features (Wang & Manning 2013) — the natural test for the sparse-gradient
  regime.
- **Multilayer fully-connected nets** (e.g. two hidden layers of 1000 ReLU units), minibatch
  128, with `L2` weight decay, evaluated both deterministically and with dropout stochastic
  regularization; training cost vs. epochs. A non-convex setting where convergence theory no
  longer formally applies.
- **Convolutional nets** on CIFAR-10 (e.g. three 5×5 conv + 3×3 max-pool stages then a
  1000-unit dense layer), whitened inputs, dropout on input and dense layers, minibatch 128;
  training cost over early epochs and over a longer run. Weight sharing makes per-layer
  gradient scales differ sharply, stressing per-parameter adaptation.
- **Variational autoencoder** (Kingma & Welling 2013), single hidden layer of 500 softplus
  units, 50-dim spherical-Gaussian latent — used as a setting to sweep the decay rates and
  `log10(alpha)` over a dense grid and observe the effect of design choices on training loss.
- Protocol: identical parameter initialization across optimizers; learning rate / momentum
  searched over a dense grid; comparisons read off the best hyperparameter setting.

## Code framework

The optimizer plugs into the same minibatch-SGD training harness already used for the
baselines. Nothing about the per-parameter update rule is settled yet — that rule is exactly
what is to be designed — so the substrate is only the generic first-order stochastic
optimization machinery that already exists: an `Optimizer` object that owns per-parameter
state and exposes a `step()` that consumes the freshly computed gradients and writes a small
update back into the parameters, and an outer training loop that draws a minibatch, runs the
existing model and loss, backpropagates to fill in each parameter's gradient, and calls
`step()`. The single empty slot is the update rule itself.

```python
import torch


class Optimizer:
    """Generic first-order stochastic optimizer. Owns per-parameter state and
    applies a per-parameter update from the current gradient. Memory must stay
    ~linear in the number of parameters (no theta x theta matrix, no Hessian)."""

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
            #       Given the stream of noisy gradients g (and any per-parameter
            #       state we choose to keep), compute the update and apply it:
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

The outer loop supplies one gradient sample per parameter; `step()` is where the
per-parameter rule will live.
