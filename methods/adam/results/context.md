# Context: first-order stochastic optimization (circa 2012–2014)

## Research question

Many machine-learning objectives are scalar functions `f(theta)` we can only observe
through noise. The noise has two sources. Minibatch subsampling makes each gradient `g_t`
a noisy, unbiased draw of the true gradient of the expected loss `E[f(theta)]`. Deliberately
injected stochasticity such as dropout randomizes the network on every forward pass, so even
the same training example yields a different gradient each time. The models are also large —
millions of parameters — so forming, storing, or inverting a `theta × theta` matrix, or
computing a Hessian, is infeasible on a memory-constrained GPU; we work with first-order
gradients and per-parameter state that stays close to linear in the number of parameters.

The question is how to turn this stream of noisy first-order gradients into good parameter
updates: how to set the step size for each parameter, using only the gradients and a little
per-parameter memory.

## Background

The dominant recipe is stochastic gradient descent with a hand-tuned learning rate, usually
with momentum, and it is the engine behind the recent progress in deep learning — large image
classifiers (Krizhevsky et al. 2012), deep speech systems (Hinton et al. 2012a; Graves et al.
2013). A few features of the setting are worth stating plainly. The right step scale can
differ across coordinates: weight-shared convolutional layers produce gradients of a very
different magnitude than dense layers. Gradients are noisy, from both minibatching and dropout
(Hinton et al. 2012b). Sparse, high-dimensional features are common in NLP / bag-of-words
models, where most features fire rarely and a rare feature's gradient carries a lot of
information. Minibatch noise and drifting data make the gradient's scale and curvature change
over the course of training. Curvature estimation is accurate but memory-hungry on a GPU.

Several framings are available. The **moment** view of gradient statistics: an exponentially
decaying running average (EMA) of a gradient is an estimate of its first moment, the mean
direction, and the window length of any EMA is set by its decay — a decay `beta` keeps an
effective memory of roughly `1/(1-beta)` recent samples, so `beta = 0.9` averages on the order
of ten samples and `beta = 0.999` on the order of a thousand. The **online convex optimization
/ regret** frame (Zinkevich 2003): treat the optimizer as committing to a point `theta_t`
before an adversary reveals a convex cost `f_t`, and measure it by regret against the best
fixed point for the whole sequence,

```
R(T) = sum_{t=1}^{T} [ f_t(theta_t) - f_t(theta*) ],   theta* = argmin_theta sum_t f_t(theta).
```

Zinkevich showed projected online gradient descent with step `eta_t = t^{-1/2}` achieves
`O(sqrt(T))` regret for any bounded-gradient convex sequence, so average regret `R(T)/T -> 0`;
a standard tool is the first-order convexity bound `f_t(theta_t) - f_t(theta*) <= g_t^T
(theta_t - theta*)`. The **natural gradient** frame (Amari 1998; Pascanu & Bengio 2013): the
diagonal of the Fisher information matrix is the expected per-coordinate squared gradient, and
natural-gradient preconditioning multiplies the step by the inverse Fisher.

## Baselines

**SGD with momentum / Nesterov (Sutskever et al. 2013).** Keep an exponentially decaying
running average of the gradient and step along it:

```
v_t     = mu * v_{t-1} + g_t
theta_t = theta_{t-1} - alpha * v_t
```

In moment language `v_t` is a (biased, un-normalized) estimate of the gradient's first moment:
it averages out minibatch noise, builds speed along consistently downhill directions, and damps
oscillation across ravines. Nesterov accelerated gradient evaluates the gradient at a look-ahead
point `theta + mu*v`, which is more responsive at high momentum. Sutskever, Martens, Dahl &
Hinton (ICML 2013) showed that well-initialized nets with a carefully scheduled momentum —
ramped up early, then reduced toward the end of training — can rival Hessian-free second-order
optimization.

**AdaGrad (Duchi, Hazan & Singer, JMLR 2011).** Give every parameter its own rate, shrunk by
the accumulated squared gradient of that coordinate:

```
theta_{t+1,i} = theta_{t,i} - alpha * g_{t,i} / sqrt( sum_{s=1}^{t} g_{s,i}^2 )
```

Rarely-active coordinates keep a small accumulated denominator and so take large steps when they
finally fire; frequently-active coordinates take small steps — well suited to sparse,
high-dimensional data. Analyzed in the regret framework it attains `O(sqrt(T))` regret, and for
sparse data the adaptive bound is roughly `O(log d · sqrt(T))` versus `O(sqrt(d·T))` for
non-adaptive SGD.

**RMSProp (Tieleman & Hinton, Coursera Lecture 6.5, 2012; momentum variant Graves 2013).**
Replace AdaGrad's growing sum with an exponential moving average of the squared gradient:

```
v_t     = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
theta_t = theta_{t-1} - alpha * g_t / ( sqrt(v_t) + eps )
```

The EMA forgets old gradients, so the denominator tracks the recent gradient scale — a windowed
second-moment estimate rather than a cumulative one. Graves (2013) added a momentum term applied
to the already-rescaled gradient. The `v_t` recursion is initialized at `v_0 = 0`.

**AdaDelta (Zeiler, 2012).** The same EMA-of-squared-gradient denominator as RMSProp, plus an
EMA of the squared parameter updates whose square root goes in the numerator, making the update
dimensionally consistent so that no global learning rate is needed.

**Quasi-Newton on minibatches (Roux & Fitzgibbon 2010; Sohl-Dickstein et al. 2014).** The
Sum-of-Functions Optimizer (SFO) estimates per-minibatch curvature for near-second-order
progress; its memory grows linearly in the number of minibatch partitions, and it assumes
deterministic subfunctions.

## Evaluation settings

The yardsticks in use:

- **Logistic regression (convex)** on MNIST (784-dim image vectors, 10 classes), with `L2`
  regularization and minibatch size 128; metric is training negative log-likelihood vs.
  iterations. A convex objective isolates optimizer behavior from local-minimum effects, and the
  step size is often annealed as `alpha_t = alpha / sqrt(t)` to match the `t^{-1/2}` schedule.
- **Sparse-feature logistic regression** on the IMDB movie-review dataset (Maas et al. 2011),
  pre-processed into 10,000-dim bag-of-words vectors, commonly with 50% dropout on the BoW
  features (Wang & Manning 2013) — the sparse-gradient regime.
- **Multilayer fully-connected nets** (e.g. two hidden layers of 1000 ReLU units), minibatch
  128, with `L2` weight decay, evaluated both deterministically and with dropout; training cost
  vs. epochs. A non-convex setting.
- **Convolutional nets** on CIFAR-10 (e.g. three 5×5 conv + 3×3 max-pool stages then a 1000-unit
  dense layer), whitened inputs, dropout on input and dense layers, minibatch 128. Weight sharing
  makes per-layer gradient scales differ sharply.
- **Variational autoencoder** (Kingma & Welling 2013), single hidden layer of 500 softplus
  units, 50-dim spherical-Gaussian latent — a stochastic generative-model objective.
- Protocol: identical parameter initialization across optimizers; method-specific step-size and
  momentum/adaptation hyperparameters searched over a dense grid; comparisons read off the best
  setting for each method.

## Code framework

The optimizer plugs into the same minibatch-SGD training harness already used for the baselines.
The per-parameter update rule is exactly what is to be designed, so the substrate is only the
generic first-order stochastic-optimization machinery: an optimizer object owns any per-parameter
state it chooses to keep, receives the fresh stochastic gradient for each parameter, and writes
an update back into that parameter. The outer training loop draws a minibatch, evaluates the
model and loss, backpropagates one gradient sample per parameter, and calls the optimizer.

```python
class Optimizer:
    """Generic first-order stochastic optimizer.

    It may keep O(number_of_parameters) state, but it cannot form a Hessian,
    a theta-by-theta matrix, or any minibatch-partition curvature table.
    """

    def __init__(self, params, step_size):
        self.params = list(params)
        self.step_size = step_size
        self.state = {id(param): {} for param in self.params}

    def step(self, gradients):
        for param, grad in zip(self.params, gradients):
            slot = self.state[id(param)]
            # TODO: design the per-parameter first-order update rule.
            # param <- param + update(grad, slot, step_size)
            pass


def train(model, loss_fn, data_stream, optimizer):
    for inputs, targets in data_stream:
        loss = loss_fn(model(inputs), targets)
        gradients = backward(loss, model.parameters())
        optimizer.step(gradients)
```

The outer loop supplies one noisy gradient sample per parameter; `step()` is the only empty slot.
