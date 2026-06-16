## Research question

In data-parallel distributed training, every worker computes a stochastic gradient on its
local minibatch, and before the optimizer can take a step the workers must agree on the
aggregate gradient — typically an all-reduce that sums each worker's `g` into the global
gradient. That gradient is a single dense vector the size of the whole model: for modern
networks, millions of 32-bit floats, exchanged *every iteration*. As clusters and models grow,
this synchronization — not the floating-point arithmetic — becomes the wall-clock bottleneck;
the impressive compute of a data center sits idle waiting on the network. The precise goal is
to drive down the number of bits each worker must send per iteration — by one, two, even three
orders of magnitude — while preserving the convergence behavior and final model quality that
full-gradient training is valued for. The tension is sharp: any lossy reduction of the
gradient perturbs the descent direction, and a perturbed direction can slow convergence, bias
the solution, or stop it converging at all. A solution must compress aggressively *and* come
with a reason to believe the optimizer still reaches the same place.

## Background

**The setting.** Minimize `f(x) = E_i[f_i(x)]` over `x ∈ R^d` by stochastic gradient descent,
`x_{t+1} = x_t − γ g_t`, where `g_t` is an unbiased stochastic gradient, `E[g_t] = ∇f(x_t)`
(Robbins & Monro 1951). In the distributed case `g_t` is itself a sum of per-worker
stochastic gradients that must be communicated and aggregated each step. Standard analytic
assumptions are `L`-smoothness of `f` (the gradient is `L`-Lipschitz) and a bounded second
moment `E‖g_t‖² ≤ σ²`; under these, SGD with `γ = O(1/√T)` reaches
`min_t E‖∇f(x_t)‖² = O(1/√T)` on non-convex objectives, and `O(1/√T)` suboptimality on
convex ones. Any communication-reduction scheme is measured against *this* rate: matching it
means compression cost nothing asymptotically.

**A structural fact about gradients that invites compression.** Gradient updates in deep
networks are strongly *positively skewed* — the overwhelming majority of coordinates carry
tiny values and a small fraction carry large ones (Aji & Heafield 2017; Dryden et al. 2016).
In an NMT embedding matrix, for instance, only the handful of vocabulary rows touched by a
minibatch get a substantial gradient and the rest are near zero. Concretely this means most
of the gradient's "energy" — its squared `ℓ₂` norm — is concentrated in a few coordinates, so
in principle a few coordinates could stand in for the whole vector with little loss of energy.
Different parameter blocks (a convolutional layer vs. a large embedding) also live on very
different scales, so the notion of "large" coordinate is only meaningful *within* a block; a
single global magnitude cutoff lets the large-scale blocks dominate and silences the rest
(Aji & Heafield 2017, who found per-block thresholds, or layer normalization, necessary for a
global threshold to work).

**The bias/variance distinction that governs which compressors are safe.** A compressor `C`
applied to the gradient is *unbiased* if `E[C(g)] = g`; then `C(g)` is still a valid
stochastic gradient and the entire SGD analysis goes through unchanged except that the
gradient's variance is inflated by `C`. It is *biased* if `E[C(g)] ≠ g`; then `C(g)` is not a
stochastic gradient of `f` at all, and none of the standard guarantees apply. This split is
the central organizing fact of the field at the time: unbiased schemes are theoretically
clean but limited in how far they can compress; the most aggressive schemes used in practice
are biased, work strikingly well empirically, and have essentially no convergence theory.

**The empirical pull of aggressive, biased compression.** Practitioners had already found
that extreme, biased compression — sending only the sign of each coordinate (1-bit SGD,
Seide et al. 2014), or sending only a tiny top fraction of coordinates by magnitude (gradient
dropping, Aji & Heafield 2017; Dryden et al. 2016; Lin et al. 2018) — could cut communication
by 1-3 orders of magnitude with little or no loss in final accuracy. A recurring, load-bearing
empirical detail across all of these practical schemes: a coordinate that gets suppressed is
*not* simply discarded. 1-bit SGD used local error accumulation, and gradient dropping kept
local dropped values after observing that "small gradients can accumulate over time" and that
zeroing them damages convergence. These engineering choices were effective, but their
mathematical role was not settled: the field had strong empirical recipes and only partial
theory for why biased compressors with local bookkeeping should behave like SGD.

## Baselines

The prior compression schemes a new method would be measured against and reacts to.

**Unbiased stochastic quantization — QSGD (Alistarh, Grubic, Li, Tomioka, Vojnovic, NeurIPS
2017) and TernGrad (Wen et al. 2017).** Quantize each coordinate by *randomized* rounding to
one of `s` discrete levels, scaled so the result is unbiased: `E[Q(g)] = g`. Because it is
unbiased, `Q(g)` plugs directly into standard SGD analysis; the only price is a variance
blow-up, `E‖Q(g)‖² ≤ κ‖g‖²` with `κ` growing as the levels coarsen (up to a `√d` factor at the
most aggressive setting), and the convergence slows by exactly that factor `κ`. **Gap:** the
unbiased construction has a bits floor — even at the coarsest level QSGD must transmit the
sign and index of order `√d` coordinates, so it cannot reach the constant-coordinates-per-step
regime that sparsification reaches, and pushing toward extreme compression directly inflates
the variance and the iteration count.

**Sign compression — 1-bit SGD (Seide et al. 2014), signSGD / Signum (Bernstein et al. 2018).**
Send only the sign of each coordinate: `x_{t+1} = x_t − γ sign(g_t)`, one bit per coordinate.
Extremely cheap, and close kin to adaptive methods (sign of a momentum-smoothed gradient is
the `Signum` variant, which mirrors Adam's behavior). It converges only under benign
conditions — Gaussian gradient noise, or a batch size that grows with the iteration count.
**Gap:** the sign operator is biased, `E[sign(g)] ≠ ∇f`, and that bias is not benign. It
forgets the *magnitude* of the gradient: on a one-dimensional problem with bimodal stochastic
gradient (value `+4` with probability `1/4`, `−1` with probability `3/4`, mean `1/4`), the
expected sign points the *wrong way*, so the objective increases in expectation for any
positive step size. And it forgets *direction*: on the convex problem
`f(x) = ε|x₁+x₂| + |x₁−x₂|`, started at `(1,1)`, the subgradient's sign is always `±(1,−1)`,
so the iterates never leave the line `x₁+x₂ = 2` and `f(x_t) ≥ f(x_0)` for *every* step-size
schedule, even with the full deterministic (sub)gradient. The discarded component
`ε(1,1)` — the part the sign throws away — is exactly the direction toward the optimum, and it
is thrown away again at every step. So a biased compressor can stall an otherwise convergent
optimizer outright; sign methods are not merely slower, they can fail to converge.

**Magnitude sparsification / gradient dropping — top-k (Aji & Heafield 2017; Dryden et al.
2016; Strom 2015).** Of the `d` coordinates of `g`, keep only the `k` with the largest
absolute value (transmit their values and their indices) and zero the rest; `k` is a tiny
fraction of `d` (drop ratios of 99% or 99.9% are typical). Because gradients are positively
skewed, the kept coordinates hold most of the energy, and empirically the final accuracy is
nearly untouched at very high drop rates. The communicated payload is `k` floats plus `k`
indices (`k log d` bits). **Gap:** top-k is biased, `E[top_k(g)] ≠ g`, so the clean unbiased
analysis does not apply and no convergence guarantee was known for it. A naive implementation
has a concrete failure mode: a coordinate whose magnitude is *persistently* small never enters
the top-k, so its gradient signal is never transmitted and that direction is permanently
starved — the optimizer is driven by only a biased subset of the gradient. Practical gradient
dropping avoided simply zeroing small coordinates, but the convergence mechanism and the exact
rate remained open.

**Mem-SGD (Stich, Cordonnier, Jaggi, NeurIPS 2018).** The first attempt to put sparsified SGD
with the local-stash mechanism (which they call "memory") on a theoretical footing in the
strongly convex case. It formalizes a class of compressors by a *contraction* property and
analyzes SGD that compresses the stepped, memory-augmented gradient and carries the
uncompressed remainder forward in a memory vector `m`. **Gap:** the analysis is restricted to
the *smooth, strongly convex* setting; the non-convex objectives of deep learning, and the
non-smooth case, are not covered, and the precise sense in which a *general* biased compressor
(sign, top-k, low-rank, …) can be made to match SGD's rate is left open.

## Evaluation settings

The yardsticks already in use for distributed-training compression:

- **Image classification on CIFAR-10 / CIFAR-100** (Krizhevsky 2009) with the standard
  augmentation (random `32×32` crop with padding 4, horizontal flip, per-channel
  normalization). Networks of varying size and depth: ResNet-style residual networks
  (He et al. 2016) at a few depths, and a VGG-style network with batch norm (Simonyan &
  Zisserman 2014). Trained with SGD plus momentum (`β = 0.9`), weight decay `5×10⁻⁴`, a
  learning-rate schedule (step decays, or a cosine schedule) over a fixed budget of epochs
  (e.g. 200), with the initial learning rate tuned on a baseline batch size.
- **Convex / strongly convex logistic regression** on standard sparse and dense datasets
  (e.g. RCV1, `epsilon`) with `ℓ₂` regularization, as the setting where convergence theory is
  exact and the compression-vs-rate tradeoff can be read off cleanly; learning rates of the
  form `η_t = γ/(λ(t+a))`.
- **Sequence models** (attentional encoder-decoder NMT, LSTM language models) where embedding
  gradients are extremely skewed and communication is a large share of step time — the
  setting where magnitude sparsification originated.
- **Protocol.** Compression is applied per parameter tensor (layer-wise), with a fixed
  compression ratio (e.g. retain 1% of coordinates, "100×"). Compared algorithms share
  initialization and schedule; the comparison is read off the best learning rate per method.
  Metrics: test accuracy, training loss vs. iteration, and the
  realized communication volume.

## Code framework

The compressor is a drop-in layer between backprop and the all-reduce: the training loop
already exists, the model and loss already exist, the optimizer step already exists. What sits
in the middle is a `Compressor` object with three methods: an `__init__` that fixes the target
compression ratio, one method that turns a gradient tensor into a small payload to communicate,
and one method that reconstructs a full-shape tensor from that payload. Exactly what the
payload contains and how it is chosen are open.

```python
import torch


class Compressor:
    """Lossy gradient compressor used between backprop and all-reduce.

    compress() maps a full gradient tensor to a small payload that will be
    communicated; decompress() rebuilds a tensor of the original shape from that
    payload. `name` identifies the parameter tensor being compressed.
    `compress_ratio` is the target fraction of information retained
    (0.01 = keep 1% = 100x compression)."""

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        pass

    def compress(self, tensor, name):
        # TODO: fill in the communication-reduction scheme.
        pass

    def decompress(self, compressed_tensors, ctx):
        pass


# existing distributed data-parallel training loop the compressor plugs into
def train(model, loss_fn, data_loader, optimizer, compressor):
    for inputs, targets in data_loader:          # draw a local minibatch
        optimizer.zero_grad()
        outputs = model(inputs)                   # forward through the existing model
        loss = loss_fn(outputs, targets)
        loss.backward()                           # backprop fills p.grad for every parameter
        for name, p in model.named_parameters():  # compress -> communicate -> decompress
            if p.grad is None:
                continue
            payload, ctx = compressor.compress(p.grad, name)
            # ... all-reduce the small `payload` across workers here ...
            p.grad = compressor.decompress(payload, ctx)
        optimizer.step()                          # the usual optimizer (e.g. SGD+momentum)
```

The loop hands `compress()` one full gradient tensor per parameter and expects back a small
payload to communicate and a tensor of the original shape after decompression. The whole
design lives in those three method bodies.
