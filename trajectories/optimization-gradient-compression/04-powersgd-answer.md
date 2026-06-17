**Problem.** The three baselines have saturated single-node accuracy at 100× (sign-with-EF
strongest: ResNet-56 94.14, VGG 70.77, ResNet-20 92.52), all bought by the same error-feedback
memory. But every one of them breaks the property that makes data-parallel training fast: the
`W`-independent all-reduce. Top-k supports differ across workers (sum of `W` `k`-sparse vectors
has up to `Wk` nonzeros → all-gather, cost grows with `W`); majority-vote and quantized codes are
non-linear collectives. The real headroom is the systems axis the benchmark cannot score: match
the baselines' accuracy with a *linear*, all-reduceable compressor.

**Key idea (warm-started low-rank + error feedback).** A weight gradient is a matrix
`M ∈ R^{m×n}` (conv folded to `(out, in·kh·kw)`), and gradient matrices are approximately
*low-rank* — a spectral concentration, distinct from top-k's coordinate skew and crucially
*linearly accessible*. Compress to rank-`r` `M ≈ P Q^T` via **one power-iteration step**:
`P = M Q`, orthonormalize `P` (QR), `Q = M^T P`. Both multiplies are linear in `M`, so with the
fixed factor shared across workers they all-reduce (`Σ P_w = (Σ M_w)Q`, `Σ Q_w = (Σ M_w)^T P`) —
two small, fixed-shape, `W`-independent collectives.

**Why one step suffices.** **Warm-start** `Q` from the previous step: the gradient subspace
drifts slowly, so one step continues a power iteration running across the whole run, and
warm-started rank-1/2 matches full-rank accuracy (cold-restart random low-rank does not).
**Error feedback** carries the deleted trailing spectrum (`e = M − P Q^T`, added back as `g + e`),
so the virtual iterate runs exact SGD — the same memory that rescued the prior two rungs.

**This task's fill.** Per-name residual `e[name]` and warm-start `Q[name]`, both local (only
`(P, Q)` go on the wire). `r = max(1, int(ratio·mn/(m+n)))` capped at `min(m,n)` — per-layer
target compression. 1-D tensors sent dense (rank meaningless, tiny). Random init is the only
randomness, once per tensor; `Q` is the post-projection factor.

**Hyperparameters.** `compress_ratio = 0.01` → per-layer `r` (≈1–2 on these layers); per-name
residual + warm-start state; QR orthonormalization.

**Reference.** Vogels, Karimireddy, Jaggi, "PowerSGD: Practical Low-Rank Gradient Compression for
Distributed Optimization", NeurIPS 2019, arXiv:1905.13727 (official `epfml/powersgd`); error
feedback from Karimireddy et al., ICML 2019, arXiv:1901.09847.

```python
class Compressor:
    """PowerSGD: warm-started rank-r low-rank gradient compression with error feedback.

    Reshape each matrix-shaped gradient to M (m x n); add the error-feedback
    residual; do ONE power-iteration step from the previous step's Q (warm start):
    P = M Q, orthonormalize P, Q = M^T P. Communicate (P, Q) -- (m+n)*r floats.
    Reconstruct M_hat = P Q^T; carry M - M_hat in a per-tensor residual. P = M Q
    and Q = M^T P are linear in M, so across workers they all-reduce with the cheap
    W-independent collective. 1-D tensors are sent dense."""

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}     # e[name]: error-feedback memory, NOT communicated
        self.qs = {}            # Q[name]: warm-start subspace, NOT communicated

    def _rank(self, m, n):
        r = max(1, int(self.compress_ratio * m * n / (m + n)))
        return min(r, m, n)

    def compress(self, tensor, name):
        shape = tensor.shape
        # error correction: compress g + e (raw-gradient units; optimizer applies lr)
        if name in self.residuals:
            tensor = tensor + self.residuals[name]

        matrix = tensor.view(tensor.shape[0], -1)
        m, n = matrix.shape
        if m == 1 or n == 1:                      # vectors: send dense, no low-rank, no EF
            return [matrix.clone(), None], (shape, False)

        r = self._rank(m, n)
        q = self.qs.get(name)
        if q is None or q.shape != (n, r):        # one-time random init; else warm start
            q = torch.randn(n, r, device=matrix.device, dtype=matrix.dtype)

        p = matrix @ q                            # P = M Q   (linear -> all-reduceable)
        p, _ = torch.linalg.qr(p)                 # orthonormalize the left factor
        q = matrix.t() @ p                        # Q = M^T P (linear -> all-reduceable)
        self.qs[name] = q                         # warm start next step

        approx = p @ q.t()                        # M_hat = P Q^T
        self.residuals[name] = (matrix - approx).view(shape)   # carry trailing spectrum
        return [p, q], (shape, True)

    def decompress(self, compressed_tensors, ctx):
        shape, low_rank = ctx
        if not low_rank:
            return compressed_tensors[0].view(shape)
        p, q = compressed_tensors
        return (p @ q.t()).view(shape)
```
