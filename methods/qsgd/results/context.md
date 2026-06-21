# Context: communication-efficient data-parallel SGD (circa 2015-2016)

## Research question

In data-parallel stochastic gradient descent, a set of `K` workers collectively minimize a
function `f: R^n -> R`. Each worker keeps a local copy of the `n`-dimensional parameter vector
`x`, draws an independent stochastic gradient `g` of `f` on its own data shard, broadcasts that
gradient to its peers, and every worker aggregates the `K` received gradients and applies the
same descent step. With dense gradients, each worker must send and receive an `n`-dimensional
vector of 32-bit floats *every iteration* — `32n` bits per peer per round. As models grow into
the tens of millions of parameters (a ResNet-152 has ~60M, AlexNet ~62M), and as `K` grows, the
gradient-*exchange* phase stops being a small overhead and becomes the dominant cost of an
iteration: on commodity multi-GPU setups the communication of gradients can take a larger share
of wall-clock time than the gradient computation itself, and that share only worsens as more
GPUs are added. The bottleneck is bytes on the wire, not flops.

The question is how to drive down the number of bits each worker transmits per iteration —
ideally far below `32n` — while keeping the optimization on track, meaning with a provable
convergence guarantee under the standard assumptions used for SGD.

## Background

By this time stochastic gradient descent is the workhorse for large-scale learning, and its
parallel variants have received heavy attention precisely because they scale (Bekkerman, Bilenko
& Langford 2011; Dean et al. 2012; Recht, Re, Wright & Niu 2011, "Hogwild!"; Li et al. 2014,
parameter server). The relevant theory is the standard convergence guarantee for SGD on a
convex, `L`-smooth objective. Treat the optimizer as having access to stochastic gradients `g(x)`
with `E[g(x)] = grad f(x)` and a variance bound `sigma^2`; one often also keeps a second-moment
bound `E[||g(x)||^2] <= B`, which implies the same scale of variance control because
`E[||g - grad f||^2] <= E[||g||^2]` when `g` is unbiased. With a suitable constant step size,
after `T` iterations the averaged iterate satisfies

```
E[ f( (1/T) sum_t x_t ) ] - min_x f(x)  <=  R * sqrt( 2 sigma^2 / T )  +  L R^2 / T,
```

where `R^2 = sup_x ||x - x_0||^2` (Bubeck 2015, Thm 6.3; the SGD lineage traces to Robbins &
Monro 1951). For data-parallel SGD, averaging `K` independent worker gradients is a minibatch of
size `K`, which divides the variance term by `K`. Rephrased with a second-moment bound `B`, the
iteration count for error `epsilon` has the familiar form

```
T = O( R^2 * max( 2B / (K epsilon^2), L / epsilon ) ),
```

i.e. **linear in the stochastic-gradient noise scale** in the regime where the first term
dominates. The important constraint is just as much about the hypothesis as the rate: the theorem
is an unbiased-gradient theorem. Extra random noise can be accounted for through its variance or
second moment, but a biased gradient surrogate is no longer covered by this bound.

A second background thread is the empirical observation, by practitioners building distributed
training systems, that gradient communication is *the* scaling bottleneck for large models, and
that lossy compression of gradients can relieve it — reduced-precision representations, and more
aggressive schemes that send drastically fewer bits per coordinate, were already in use in
production frameworks (DistBelief; CNTK; TensorFlow).

A third thread is integer coding. The values produced by an aggressive gradient representation
are not uniformly distributed — small magnitudes are common, large ones rare — so a fixed-width
encoding wastes bits. Universal codes for the positive integers exploit exactly this: Elias's
recursive (omega) coding represents an integer `k` in `|Elias(k)| <= (1 + o(1)) log k + 1` bits
(`log` base 2), prepending the length of `k`'s binary representation and recursing on that
length, so frequent small integers get short codes and the rare large ones pay a logarithmic
premium (Elias 1975). Encoding and decoding are linear in the codeword length and need no prior
bound on `k`.

Finally, there is a known floor on what compression can buy. The communication complexity of
`n`-dimensional convex optimization is `Omega(n (log n + log 1/epsilon))` bits (Tsitsiklis & Luo
1987), and for the closely related distributed-mean-estimation problem, any scheme that keeps the
estimator's variance blowup bounded by a constant must communicate `Omega(n)` bits per round
(Zhang, Duchi, Jordan & Wainwright 2013; Suresh, Yu, McMahan & Kumar 2016). Whatever the
bits-vs-variance tradeoff turns out to be, it cannot be pushed past this information-theoretic
wall.

## Baselines

These are the prior gradient-compression methods a new scheme would be measured against.

**1-bit SGD with error feedback (Seide, Fu, Droppo, Li & Yu, Interspeech 2014).** The most
aggressive scheme in production. Reduce each gradient coordinate to a single bit — its position
relative to a fixed threshold of 0 — and reconstruct it with a small number of recomputed
floating-point values shared across a weight-matrix column. Raw, this loses too much: the
reconstruction systematically misrepresents the gradient and the training diverges. The fix that
makes it usable is *error feedback*: keep a residual buffer `Delta`, and before quantizing the
current gradient `G(t)` add back the residual carried from the previous time it was sent,

```
G_quant(t) = Q( G(t) + Delta(t - N) ),
Delta(t)   = G(t) - Q_inverse( G_quant(t) ),
```

so the error suppressed in one round is folded into a later round and "all gradients are
eventually added up into the model (in the limit)." Combined with AdaGrad, automatic minibatch
sizing, double buffering, and model parallelism, this enabled state-of-the-art scaling of speech
DNNs.

**Low-precision / reduced-precision SGD, analyzed by "Buckwild!" (De Sa, Zhang, Olukotun & Re,
NIPS 2015).** The first work to put convergence guarantees on low-precision SGD. Using a
martingale framework that treats various perturbations (asynchrony, rounding) as forms of noise
in a unified model, it derives convergence rates for SGD when the gradients are represented in
low-precision fixed-point arithmetic, assuming the quantization is unbiased, the problem is
convex, and the gradients are sparse, and it bounds the error probability.

**Plain parallel SGD (no compression).** Send the full 32-bit gradient. This is the correctness
baseline: it is exactly the standard SGD/minibatch theorem, no extra variance, but `32n` bits per
worker per round — the cost the others are trying to avoid.

## Evaluation settings

The natural yardsticks already in use for distributed DNN training:

- **Image classification.** Convolutional networks — AlexNet (Krizhevsky et al. 2012), VGG
  (Simonyan & Zisserman 2014), ResNet (He et al. 2016), and BN-Inception (Ioffe & Szegedy 2015) —
  trained on ImageNet (Deng et al. 2009) and on the smaller CIFAR-10 (Krizhevsky & Hinton 2009),
  with standard per-network sizes and the hyperparameters tuned for the 32-bit-precision run.
- **Speech recognition.** An LSTM acoustic model (Hochreiter & Schmidhuber 1997) on the CMU AN4
  corpus.
- **Hardware / system.** Multi-GPU servers (e.g. up to 16 NVIDIA K80 GPUs on an EC2 instance)
  with fast MPI-based GPU-to-GPU communication, running on a deep-learning framework such as
  Microsoft CNTK; double buffering to overlap communication with computation.
- **Metrics.** Test/validation accuracy (top-1 / top-5 for ImageNet, word error rate for speech),
  the per-epoch wall-clock time broken into computation vs communication, and the number of bits
  communicated per iteration. The protocol contrasts these against the full-precision run and
  against the production 1-bit scheme while sweeping the number of GPUs and the compression budget.

## Code framework

A data-parallel SGD harness already exists; the only undecided piece is how a gradient is
turned into something small enough to put on the wire and then turned back. The substrate is the
generic machinery: a training step that computes a local gradient for each parameter tensor, a
`Compressor` object with a fixed interface that maps a gradient tensor to a payload-to-send plus a
piece of local context and maps a received payload back to a tensor of the original shape, and the
all-reduce exchange loop that compresses each worker's gradient, sums the decompressed peer
gradients, and steps. What goes *inside* the compressor — how to represent the gradient compactly
and how to reconstruct it — is exactly what is to be designed, so its body is an empty slot.

```python
import torch


class Compressor:
    """Maps a gradient tensor to a compact payload to communicate, and back.

    `budget` is a generic communication setting for the representation. The
    compressor may keep per-parameter local state across calls (`name` identifies
    the parameter); only the returned payload is sent over the network, while
    `ctx` stays local for decompression. The decompressed tensor must match the
    input tensor's shape.
    """

    def __init__(self, budget=None):
        self.budget = budget
        # TODO: any configuration / per-parameter buffers the representation needs.

    def compress(self, tensor, name):
        # TODO: the gradient representation we will design.
        #       Return (payload_to_send, ctx_kept_local).
        pass

    def decompress(self, compressed_tensors, ctx):
        # TODO: reconstruct a tensor of the original shape from the payload + ctx.
        pass


# existing data-parallel exchange loop the compressor plugs into
def all_reduce_step(local_grads, compressor, world_size):
    aggregated = []
    for name, g in local_grads:                 # one gradient tensor per parameter
        payload, ctx = compressor.compress(g, name)   # shrink before sending
        # broadcast `payload` to peers; receive each peer's payload (omitted)
        peer_payloads = exchange(payload)             # network communication
        summed = None
        for p in peer_payloads:
            gp = compressor.decompress(p, ctx)        # reconstruct each peer gradient
            summed = gp if summed is None else summed + gp
        aggregated.append((name, summed / world_size))
    return aggregated                                 # fed to the optimizer step
```

The exchange loop supplies one gradient tensor per parameter; `compress` / `decompress` are where
the representation will live.
