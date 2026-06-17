# PowerSGD, Distilled

PowerSGD compresses each matrix-shaped gradient update with a warm-started rank-`r` subspace
iteration and error feedback. For a local corrected matrix `M_w`, all workers share the same query
factor `Q`, compute `P_w = M_w Q`, all-reduce the mean `P`, orthogonalize that shared `P`, compute
`Q_w = M_w^T P`, all-reduce the mean `Q`, and reconstruct `P Q^T`.

The two all-reduces are essential. Computing both factors locally and then all-reducing them once is
only correct in the one-worker case.

## Algorithm

For each tensor name:

```text
local residual e[name] = 0
shared warm-start factor Q[name] initialized identically on every worker

M = reshape(g + e[name])              # raw-gradient units; optimizer applies lr later
orthonormalize Q[name]
P_local = M @ Q[name]
P = all_reduce_mean(P_local)
P = orthonormalize(P)
Q_local = M.T @ P
Q = all_reduce_mean(Q_local)
M_hat = P @ Q.T
e[name] = M - M_hat                   # one-worker exact residual; distributed reference invariant
Q[name] = Q                           # warm start for next step
return M_hat
```

Vector tensors are sent dense. Convolutional tensors are viewed as `(out_channels, in_channels * kh *
kw)`. The original method and reference implementation use a fixed rank as the main knob; the `compress_ratio`
fallback below only translates a ratio-style scaffold into a rank.

## Code

```python
import torch


class Compressor:
    """PowerSGD with warm-started low-rank factors and error feedback.

    `all_reduce_mean` is an optional callable that averages a tensor across workers
    in-place or returns the averaged tensor. If omitted, the compressor is the exact
    W=1 version. In distributed use, this compressor performs the two required
    collectives internally; do not all-reduce the returned payload a third time.
    """

    def __init__(
        self,
        compress_ratio=0.01,
        rank=None,
        min_compression_rate=2.0,
        random_seed=0,
        all_reduce_mean=None,
    ):
        self.compress_ratio = compress_ratio
        self.rank = rank
        self.min_compression_rate = min_compression_rate
        self.all_reduce_mean = all_reduce_mean
        self.residuals = {}
        self.qs = {}
        self.generators = {}
        self.random_seed = random_seed

    def _generator(self, device):
        key = str(device)
        if key not in self.generators:
            self.generators[key] = torch.Generator(device=device).manual_seed(self.random_seed)
        return self.generators[key]

    def _rank(self, m, n):
        if self.rank is not None:
            return min(int(self.rank), m, n)
        r = int(self.compress_ratio * m * n / (m + n))
        return min(max(1, r), m, n)

    def _should_compress(self, m, n, r):
        dense = m * n
        compressed = (m + n) * r
        return dense / compressed >= self.min_compression_rate

    def _orthogonalize(self, matrix):
        if matrix.shape[1] == 1:
            eps = torch.finfo(matrix.dtype).eps
            return matrix / matrix.norm(dim=0, keepdim=True).clamp_min(eps)
        return torch.linalg.qr(matrix, mode="reduced").Q

    def _all_reduce_mean(self, tensor):
        if self.all_reduce_mean is None:
            return tensor
        reduced = self.all_reduce_mean(tensor)
        return tensor if reduced is None else reduced

    def compress(self, tensor, name):
        shape = tensor.shape
        corrected = tensor
        if name in self.residuals:
            corrected = corrected + self.residuals[name]

        matrix = corrected.reshape(shape[0], -1)
        m, n = matrix.shape

        if m == 1 or n == 1:
            dense = self._all_reduce_mean(matrix.clone())
            self.residuals.pop(name, None)
            self.qs.pop(name, None)
            return [dense], (shape, False)

        r = self._rank(m, n)
        if not self._should_compress(m, n, r):
            dense = self._all_reduce_mean(matrix.clone())
            self.residuals.pop(name, None)
            self.qs.pop(name, None)
            return [dense], (shape, False)

        q = self.qs.get(name)
        if q is None or q.shape != (n, r) or q.device != matrix.device or q.dtype != matrix.dtype:
            q = torch.randn(
                n,
                r,
                device=matrix.device,
                dtype=matrix.dtype,
                generator=self._generator(matrix.device),
            )

        q = self._orthogonalize(q)
        p = matrix @ q
        p = self._all_reduce_mean(p)
        p = self._orthogonalize(p)

        q = matrix.t() @ p
        q = self._all_reduce_mean(q)
        self.qs[name] = q.detach()

        approx = p @ q.t()
        self.residuals[name] = (matrix - approx).reshape(shape).detach()
        return [p, q], (shape, True)

    def decompress(self, compressed_tensors, ctx):
        shape, low_rank = ctx
        if not low_rank:
            return compressed_tensors[0].reshape(shape)
        p, q = compressed_tensors
        return (p @ q.t()).reshape(shape)
```

The code mirrors the NeurIPS Algorithm 1 description and the official `epfml/powersgd` implementation:
matrix reshape by output dimension, identical warm-start query state, orthogonalized factors,
two fixed-shape reductions, dense pass-through for vectors or non-saving tensors, and residual
feedback equal to corrected input minus reconstructed update. The convergence statement should be
read in the error-feedback sense: under a contractive compressor, the residual is bounded and the
corrected-iterate analysis recovers SGD's asymptotic rate; the practical one-step warm-started
low-rank approximation is empirically justified rather than proved as an exact dense-SGD
replacement.
