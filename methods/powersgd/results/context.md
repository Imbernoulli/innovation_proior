## Research Question

Synchronous data-parallel training repeats the same expensive operation every iteration: each worker
computes a stochastic gradient on its local minibatch, and the workers must aggregate those gradients
before the optimizer updates the shared model. For modern neural networks this communicated object
is a dense vector with millions or billions of floating-point entries. When the model is large or the
network is slow relative to local compute, gradient exchange can dominate wall-clock time.

The goal is to reduce communicated data by one or two orders of magnitude while preserving the
behavior of the uncompressed optimizer. The compressed representation must keep a fixed shape and
aggregate by addition or averaging so that it fits the collective communication primitives that
optimized distributed-training systems already use.

## Distributed SGD And Collectives

The baseline optimization loop minimizes `f(x) = E_i[f_i(x)]` with SGD. At step `t`, worker `w`
computes a stochastic gradient `g_{t,w}` and the distributed update uses the average

`g_t = (1/W) sum_{w=1}^W g_{t,w}`.

Dense SGD communicates this average with all-reduce. The bandwidth term for a well-implemented
ring all-reduce is proportional to the tensor size with only a small worker-count factor, and tree
variants similarly exploit associativity of addition. The message stays a dense tensor of the same
shape throughout the collective.

This implementation property becomes a mathematical constraint on a compressor. If the workers
send `C(g_{t,w})`, then the system wants the average of those compressed objects to be a compressed
object of the average gradient, or at least a fixed-shape representation from which the same update
can be reconstructed.

## Compression Baselines

Unbiased quantization schemes randomly round coordinates to a small number of levels so that
`E[C(g)] = g`. This makes the optimization analysis resemble SGD with extra variance.

Sign methods send one bit per coordinate, sometimes with majority vote across workers. They are
appealingly small.

Sparsification keeps a subset of coordinates, often the largest magnitudes, and uses a memory vector
to reinsert dropped coordinates later. Error compensation can preserve optimizer quality over time.

Spectral or atomic compression treats a matrix gradient through atoms such as singular-vector outer
products. This matches the observation that many neural-network gradient matrices have rapidly
decaying spectra.

## Mathematical Ingredients

Error feedback is the standard correction for biased compression. In its simplest form, each worker
keeps a residual `e_t`, compresses the corrected update `p_t = gamma g_t + e_t` or the equivalent
raw-gradient quantity used before the optimizer applies `gamma`, and then stores

`e_{t+1} = p_t - C(p_t)`.

For a compressor satisfying a contraction condition

`||C(x) - x||^2 <= (1 - delta) ||x||^2`,

the residual remains bounded. The corrected sequence `x_tilde = x_t - e_t` follows an SGD-like
recurrence, and smoothness lets the convergence rate match SGD asymptotically up to higher-order
terms. The guarantee is not that every compressed step equals the dense step; it is that omitted
information is delayed in a controlled way.

Matrix low-rank approximation offers a different structure from coordinate sparsity. A matrix
`M in R^{m x n}` can be approximated as `P Q^T` with `P in R^{m x r}` and `Q in R^{n x r}`, using
only `(m+n)r` numbers. Classical subspace iteration estimates such a rank-`r` subspace with
matrix multiplications and an orthogonalization rather than a full singular value decomposition.

## Code Framework

The compressor sits between backpropagation and the communication step. It may keep state per
parameter tensor, and it may need parameter names to keep that state aligned across iterations.
Vector-shaped tensors can be passed through unchanged because they are small and have no useful
matrix rank. Convolutional and linear weight gradients can be viewed as matrices by flattening every
dimension except the output-feature dimension.

```python
import torch


class Compressor:
    """Lossy gradient compressor between backprop and gradient aggregation."""

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        # Return a payload plus enough context to reconstruct the update.
        pass

    def decompress(self, compressed_tensors, ctx):
        pass


def train(model, loss_fn, data_loader, optimizer, compressor):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            payload, ctx = compressor.compress(p.grad, name)
            # The compressor may expose ordinary payloads for a single collective,
            # or it may perform a multi-stage collective internally.
            p.grad = compressor.decompress(payload, ctx)
        optimizer.step()
```
