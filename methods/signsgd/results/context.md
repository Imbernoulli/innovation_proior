# Context: communication-bottlenecked distributed first-order optimization (circa 2016-2017)

## Research question

Training a large neural network is distributed across many workers (GPUs in one box, or several
machines linked by a parameter server). Each worker computes a stochastic gradient on its slice
of the minibatch; those gradients have to be aggregated before the optimizer can take a step, and
then the aggregate has to be sent back. For a `d`-parameter model with `M` workers, the usual
32-bit upload plus 32-bit return path moves `64 M d` bits per iteration.
For modern nets with `d > 10^8`, the gradient exchange — not the arithmetic — is the wall-clock
bottleneck, and it gets worse as you add workers. The question is how to reduce the number of bits
communicated per coordinate while preserving useful convergence behavior on the non-convex
objectives that deep learning presents.

## Background

By this time stochastic gradient descent (Robbins & Monro, 1951) — `x_{k+1} = x_k - delta g_k`,
with `g_k` a noisy unbiased gradient on a minibatch — is the workhorse of deep learning, usually
with momentum and a hand-tuned, scheduled learning rate. Three pieces of prior knowledge frame the
problem.

**The sign of the gradient as a robust update direction.** A weight's gradient magnitude can be
wildly different from another's and can drift during training, which makes a single global learning
rate structurally hard to pick. A classical response is to throw the magnitude away and act only on
the *sign* of each coordinate's gradient, adapting a separate step size per weight. This is robust
and fast in full-batch training. The canonical illustration (Hinton's
lecture notes on neural-net optimization): a weight that receives a gradient of `+0.1` on nine
minibatches and `-0.9` on the tenth has average gradient zero and should stay put — but a sign rule
increments it nine times and decrements it once by a comparable amount, so the weight drifts a long
way. Curing this by keeping a moving average of the squared gradient and dividing by its root
(`MeanSquare(w,t) = 0.9 MeanSquare(w,t-1) + 0.1 (dE/dw)^2`, step `~ g / sqrt(MeanSquare)`) gave
the per-coordinate adaptive optimizers — which transmit full-precision rescaled gradients.

**The geometry of high-dimensional vectors.** A way to summarize how "spread out" a vector
`v ∈ R^d` is across its coordinates is the ratio `phi(v) := ||v||_1^2 / (d ||v||_2^2)`. It equals
`1` for a fully dense vector (all coordinates equal in magnitude) and `~1/d ≈ 0` for a fully sparse
one (a single nonzero). It gives the exact relation `||v||_1^2 = phi(v) d ||v||_2^2`, and also
the useful bound `||v||_1^2 ≤ phi(v) d^2 ||v||_inf^2`.
Whether a coordinate-blind, magnitude-blind update helps or hurts plausibly depends on the relative
density of the gradient, the gradient noise, and the curvature — but the empirical densities of
these objects in real networks were not, at this point, something anyone had measured.

**A diagnostic measurement that *can* be made before committing to any method.** Welford's
single-pass, numerically stable algorithm computes the exact mean and per-coordinate variance of
the stochastic gradient at a fixed point in parameter space. At every epoch of training a small
convolutional network (Resnet-20 on CIFAR-10), one can do an extra full pass over the data to
compute the true gradient `g` and its per-coordinate standard deviation `sigma`, and then evaluate
`phi(g)` and `phi(sigma)`. That diagnostic would say whether gradient signal and stochastic noise
are spread across many coordinates or concentrated in a few, without yet committing to any
particular compression rule.

**Convex-optimization geometry available on the shelf.** For a norm `||.||`, the normalized
steepest-descent direction is `argmin{ g^T v : ||v|| <= 1 }` (Boyd & Vandenberghe, Convex
Optimization, §9.4). The non-convex setting of deep learning makes global guarantees hopeless, so
theory settles for convergence to a stationary point — bounding the gradient norm in expectation
(Ghadimi & Lan, 2013). The standard smoothness assumption is scalar: `|f(y) - [f(x)+g(x)^T(y-x)]|
≤ (L/2)||y-x||_2^2`. A finer, coordinate-wise version assigns a separate constant to each direction.

## Baselines

**Full-precision distributed SGD (Robbins & Monro, 1951; parameter server, Li et al.).** Each
worker sends its 32-bit gradient up, the server averages and sends the 32-bit average down. Under
`L`-smoothness and bounded total variance `sigma^2 := ||sigma||_2^2`, with step `delta = 1/L` and a
growing batch (or `delta = 1/(L sqrt K)`, batch 1), the non-convex rate is
`E[(1/K) sum_k ||g_k||_2^2] ≤ (1/sqrt N)[2L(f_0 - f*) + sigma^2]` in `N` gradient calls.

**Resilient backpropagation, Rprop (Riedmiller & Braun, 1993).** Per-weight adaptive step that
ignores gradient magnitude and reads only the sign: multiply a weight's step by `~1.2` if its last
two gradient signs agree, by `~0.5` if they flip. Robust and fast in full batch, and the ancestor of
RMSprop and Adam.

**RMSprop / Adam (Tieleman & Hinton, 2012; Kingma & Ba, 2015).** Restore minibatch averaging by
dividing the gradient by an exponential moving average of its root-mean-square; Adam adds a momentum
average in the numerator, so its step is roughly `<g>_{beta1} / sqrt(<g^2>_{beta2})` — mean over RMS.

**1-bit SGD (Seide et al., 2014).** Quantize each gradient coordinate to one bit (a thresholded
sign), but *carry the quantization error forward* into the next minibatch (error feedback) so the
discarded magnitude is not lost. Empirically near-lossless on speech DNNs and dramatically cheaper
to communicate.

**QSGD (Alistarh et al., 2017).** Stochastically round each coordinate to one of a few discrete
levels so the compressed gradient is an *unbiased* estimate of the true gradient; this lets standard
SGD theory carry over directly. At 1-bit precision the return path from the server picks up log
factors in the bit count, `(2 + log(2M+1)) M d`.

**TernGrad (Wen et al., 2017).** Unbiased ternary quantization to `{0, ±1}`.

**Stochastic sign descent under an `l_inf` majorization (Carlson et al., 2016).** Studies a
stochastic signed update under `l_inf`-geometry smoothness assumptions.

## Evaluation settings

The natural yardsticks already in use, all pre-existing datasets, architectures, and protocols:

- **Resnet-20 on CIFAR-10** (He et al., 2016; Krizhevsky, 2009): the small-network workhorse;
  standard `{train/val/test}` split, learning-rate schedule decimated at fixed epochs, weight decay
  and momentum tuned on a held-out validation split. Test accuracy is the metric.
- **Resnet-50 v2 on ImageNet** (He et al., 2016b; Russakovsky et al., 2015): the large-scale test,
  where communication cost actually bites; train/test top-1 accuracy, with augmentation switched off
  near the end.
- **A controlled toy quadratic** for isolating geometry: `f(x) = (1/2)||x||^2` on `R^100` (so
  `g(x) = x`, a fully dense gradient), with Gaussian noise added to *only the first coordinate*
  (extreme sparse noise); per-algorithm constant learning rates tuned separately. The point is to
  exhibit a regime, not to benchmark.
- **Gradient-statistics protocol:** at each epoch, a full extra pass over the data with Welford's
  algorithm to record the exact gradient `g` and per-coordinate noise `sigma`, then `phi(g)` and
  `phi(sigma)`. Histograms of the per-coordinate noise across random parameters check whether the
  noise looks unimodal and symmetric as the batch grows.
- **Communication accounting:** bits transmitted per iteration as a function of `d` and `M`, up and
  down, for each scheme.

## Code framework

The scheme plugs into the existing parameter-server training loop. Each worker already runs the
model and loss, backpropagates to fill a gradient buffer, and hands that gradient to a
**communication layer** before the optimizer applies a step. The communication layer is where bytes
go on the wire, so it is the natural place to interpose: a worker *encodes* its gradient into
whatever will be transmitted, the server *aggregates* the encoded messages from all workers and
broadcasts a result, and each worker *decodes* the broadcast back into an update direction. None of
the encode/aggregate/decode rule is settled — that rule is exactly what is to be designed — so the
substrate is only the generic harness plus an empty codec.

```python
import torch


class GradientCodec:
    """Encodes a gradient tensor into the message communicated over the network,
    and decodes a received aggregate back into an update direction. Owns any
    per-parameter state it chooses to keep (`name` identifies the parameter)."""

    def __init__(self):
        self.state = {}

    def encode(self, grad, name):
        # grad: the full-precision local stochastic gradient for this parameter.
        # Returns (message, ctx): `message` is what goes on the wire (the thing we
        # want to make small); `ctx` is local-only side information for decoding.
        # TODO: the compression map we will design.
        pass

    def aggregate(self, messages):
        # Combine the messages received from all workers into the quantity the
        # server broadcasts back. (Single-worker: identity.)
        # TODO: the server-side combine we will design.
        pass

    def decode(self, received, ctx):
        # Turn the broadcast aggregate back into a same-shape update direction.
        # TODO: the decode rule we will design.
        pass


# existing parameter-server training loop the codec plugs into
def train_step(model, loss_fn, batch, codec, params, lr):
    inputs, targets = batch
    loss = loss_fn(model(inputs), targets)        # existing model + loss
    loss.backward()                               # backprop fills p.grad
    for p, name in params:
        message, ctx = codec.encode(p.grad, name) # worker -> wire
        received = codec.aggregate([message])     # server combines worker messages
        direction = codec.decode(received, ctx)   # wire -> update direction
        p.data.add_(direction, alpha=-lr)         # apply the step
        p.grad = None
```

The harness supplies one full-precision gradient per parameter; `encode`/`aggregate`/`decode` are
the slots the scheme will fill. The loop exposes the learning rate, while the codec may own any
state or local hyperparameters its rule requires.
