# QSGD, distilled

QSGD (Quantized SGD) is a family of lossy gradient-compression schemes for communication-bound
data-parallel SGD. Each worker quantizes its stochastic gradient by *randomized rounding* to a
discrete set of `s` levels in a way that is **exactly unbiased** (`E[Q_s(g)] = g`), then encodes
the result with a universal integer code. Because the quantized gradient stays unbiased, it is
itself a legitimate stochastic gradient: it changes nothing in the standard SGD analysis except
the variance, so convergence guarantees carry over with the variance multiplied by a bounded
blowup factor — no error feedback and no extra model-sized buffer. A single knob `s` (the number
of levels, often chosen as `2^b` for a `b`-bit payload) smoothly trades communicated bits against
added variance.

## Problem it solves

In data-parallel SGD, `K` workers each broadcast an `n`-dimensional gradient every iteration —
`32n` bits per peer. For models with tens of millions of parameters, this gradient exchange,
not the gradient computation, dominates iteration time. QSGD cuts bits-per-iteration by a large
factor while preserving a provable convergence rate.

## Key idea

Bias is what put prior aggressive schemes (1-bit SGD) outside SGD's convergence theory and forced
the error-feedback patch. The standard SGD bound charges only for the gradient's *variance* — and
only because the gradient is unbiased. So compress *unbiasedly*: then the quantized gradient is a
stochastic gradient whose only effect is to inflate the variance, which plugs directly into the
existing theorem.

**Stochastic quantization.** For `v != 0`, with `s >= 1` levels, `a_i = |v_i|/||v||_2 in [0,1]`,
and `ell` the integer with `a_i in [ell/s, (ell+1)/s]`:

```
Q_s(v_i) = ||v||_2 * sgn(v_i) * xi_i(v, s),

xi_i = (ell+1)/s  with probability  p(a_i, s) = a_i * s - ell   (the fractional part of a_i*s)
       ell/s       otherwise.
```

The round-up probability `p = a*s - ell` is forced by demanding `E[xi_i] = a_i`, i.e. exact
unbiasedness. `s = 1` recovers the crude ternary `{-||v||, 0, +||v||}` quantizer.

## Properties (per gradient vector `v in R^n`)

- **Unbiasedness:** `E[Q_s(v)] = v`. (`E[xi_i] = ell/s + p/s = a_i`.)
- **Variance bound:** `E[ ||Q_s(v) - v||^2 ] <= min(n/s^2, sqrt(n)/s) * ||v||^2`, equivalently
  second moment `E[ ||Q_s(v)||^2 ] <= (1 + min(n/s^2, sqrt(n)/s)) ||v||^2`.
  The two branches come from `p <= 1` (giving `n/s^2`) and `p <= a*s` (giving `sqrt(n)/s` via
  `||v||_1 <= sqrt(n)||v||_2`). At `s = 1` the added-variance factor is `sqrt(n)`; at
  `s = sqrt(n)` the added-variance factor is `1` and the second-moment factor is `<= 2`.
- **Sparsity / density:** `E[ ||Q_s(v)||_0 ] = O(s(s + sqrt(n)))`; at `s = 1`,
  `E[nnz] <= ||v||_1/||v||_2 <= sqrt(n)`.

## Encoding

The integer levels `s*xi_i` are mostly small (large integers are rare since `||v||_2` supports
few large coordinates), so use **Elias recursive (omega) coding**, `|Elias(k)| <= (1+o(1))log k + 1`
bits. Send `(||v||_2 in 32 bits, signs, levels)`, coding nonzero position-gaps and values. Bounds
(via Jensen on the concave `x log(C/x)` and `log(1+x)`):

- **Per-iteration bits (general `s`):**
  `(3 + (3/2 + o(1)) log( 2(s^2+n) / (s(s+sqrt(n))) )) * s(s+sqrt(n)) + 32`.
- **Sparse regime `s = 1`:** `O(sqrt(n) log n)` bits, variance blowup `sqrt(n)`.
- **Dense regime `s = sqrt(n)`:** `<= 2.8 n + 32` bits (vs `32n`), added variance
  `<= ||v||^2`, and second moment `<= 2||v||^2`.

This tradeoff is essentially optimal: any scheme with constant variance blowup must send
`Omega(n)` bits/round (distributed-mean-estimation lower bound).

## Convergence guarantees

Let `alpha = min(n/s^2, sqrt(n)/s)`. If the original stochastic gradients have second moment
at most `B`, then the quantized gradients have second moment at most `B_q = (1 + alpha)B`;
the analysis is otherwise unchanged.

- **Convex, `L`-smooth.** Parallel QSGD on `K` workers reaches error `epsilon` in
  `T = O( R^2 * max( 2B_q/(K epsilon^2), L/epsilon ) )` iterations, with the bit cost above
  (`2.8n + 32` at `s = sqrt(n)`).
- **Smooth non-convex** (via Ghadimi-Lan): with constant step `eta = O(1/L)` and a random
  stopping iterate,
  `(1/L) E[||grad f(x_R)||^2] <= O( sqrt(L (f(x_1) - f*)) / N + (1 + alpha) B / L )`.
  The compression enters through the same noise-scale term.

## Practical variants

- **Bucketing.** Flatten the gradient, split into buckets of `d` consecutive entries, quantize
  each bucket independently with its own scale. The blowup becomes `min(d/s^2, sqrt(d)/s)`,
  controlled by the chosen `d` rather than the full dimension `n`. Example: `d = 512`, `4` bits
  (`s = 16`) gives blowup `<= sqrt(512)/16 ≈ 1.41`. `d = 1` is no quantization; `d = n` is the
  full-tensor scheme.
- **Max-normalization.** Scale by `max|v_i|` instead of `||v||_2`; preserves more per-iteration
  signal, forfeits the sparsity guarantee (irrelevant in the dense `Theta(sqrt(d))`-level regime).

## Working code

The per-tensor `L2`-normalized form, matching the GRACE `grace_dl/torch/compressor/qsgd.py`
kernel. Payload sent = signed integer levels + one norm float; `ctx` (shape) stays local. No
residual buffer.

```python
import torch

from grace_dl.torch import Compressor


class QSGDCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = previous_level + is_next_level

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(
            torch.int8 if self.quantum_num < 128 else torch.half
        )
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
```
