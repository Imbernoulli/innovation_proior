Synchronous data-parallel training spends much of its wall-clock time repeating one expensive exchange: every worker computes a stochastic gradient on its local minibatch, and the workers must aggregate those gradients before the optimizer updates the shared model. For a large network this communicated object is a dense vector of millions or billions of entries, and when the model is big or the network is slow relative to local compute, gradient exchange dominates. We want to cut the communicated volume by one or two orders of magnitude without changing how the underlying optimizer behaves. The catch is not purely statistical. The reason dense SGD is tolerable at all is that the object being communicated is a fixed-shape tensor and aggregation is addition, which lets ring or tree all-reduce exploit associativity so the message does not grow more structurally complicated as workers are added. Any compressor that sends a small message but breaks this property quietly fails: if the compressed object's structure depends on each worker's local gradient, the sum of those objects is no longer a compressed object of the same type, and the system is forced into gather, decode, merge, or re-encode logic where the nominal compression ratio stops predicting real speed.

That constraint is exactly what disqualifies the familiar coordinate schemes. Top-$k$ sparsification with a memory vector is a good optimizer story — a residual carries the dropped coordinates forward — but worker A's largest coordinates and worker B's largest coordinates are different sets, so their sum has the union support, and that union grows with the number of workers; the aggregate is not a $k$-sparse object of the same size. Sign methods send one bit per coordinate but the sign operator is nonlinear and biased, and majority vote across workers is not ordinary averaging of fixed-shape tensors. Stochastic quantization is easier to analyze because it is unbiased, $\mathbb{E}[C(g)] = g$, but quantized codes and their scaling metadata are not generally closed under addition the way a dense tensor is. What we actually need is a compressed representation built out of operations that commute with summing the workers.

The structure of the gradients themselves points at the right idea. A fully connected layer produces a matrix gradient, and a convolutional filter gradient can be flattened into a matrix by keeping output channels as rows and folding the input and kernel dimensions into columns. Matrices can be low-rank even when their entries are dense, and neural-network gradient matrices empirically have rapidly decaying spectra. If an $m \times n$ matrix $M$ is well approximated by rank $r$, then a factorization $PQ^\top$ with $P \in \mathbb{R}^{m\times r}$ and $Q \in \mathbb{R}^{n\times r}$ costs $(m+n)r$ numbers instead of $mn$ — exactly the scale needed for aggressive compression. But the naive route repeats the same structural failure in spectral coordinates: the best rank-$r$ approximation from an SVD is not a linear function of $M$ and is expensive to compute, so if every worker truncates its own SVD, averaging the factors does not give the truncated SVD of the averaged gradient.

I propose PowerSGD: compress each matrix-shaped gradient with a warm-started rank-$r$ subspace iteration whose expensive steps are linear in $M$, made aggregatable by ordering those steps around the collective, and corrected with error feedback so nothing thrown away is truly lost. The reason subspace iteration is the right primitive is that its work is two matrix multiplications, and a matrix multiplication is linear in the local gradient as long as the *other* factor is shared across workers. So all workers hold the same query factor $Q$, each computes $P_w = M_w Q$ locally, and we all-reduce the mean to get $P = \frac{1}{W}\sum_w M_w Q$, which equals $\bar{M}Q$ for the averaged gradient $\bar M$. Only after that aggregated $P$ exists does every worker orthonormalize the same $P$, then each computes $Q_w = M_w^\top P$ against that common left factor, and a second all-reduce yields $Q = \frac{1}{W}\sum_w M_w^\top P = \bar M^\top P$. The reconstructed update is $PQ^\top$. The full step on a local corrected matrix $M$ is

$$P = \mathrm{orth}(M Q), \qquad Q \leftarrow M^\top P, \qquad \hat M = P Q^\top,$$

with each of the two matrix products preceded by its all-reduce in the distributed case. The order of operations is the load-bearing implementation constraint: a one-shot hook that computes both local $P_w$ and local $Q_w$ before communicating and then all-reduces both factors once is *not* this algorithm — it only coincides with it when $W=1$, because $Q_w$ must be formed against the already-aggregated, already-orthonormalized $P$, not against a local $P_w$. Two fixed-shape all-reduces, in sequence, are what make the compressed message both small and additively aggregatable.

Two further design choices make the one-step approximation actually usable. First, error feedback, because a rank-$r$ subspace iteration is biased: it drops trailing singular directions and a single iteration can miss part of the leading subspace, so $\hat M$ cannot be treated as an unbiased stochastic gradient. Each worker keeps a residual $e$, and before compression forms the corrected matrix $M = \mathrm{reshape}(g + e)$ in the same raw-gradient units the optimizer will later scale by the learning rate; after reconstruction it stores $e \leftarrow M - \hat M$, i.e. exactly what this step failed to send. Under a contraction condition $\|C(x)-x\|^2 \le (1-\delta)\|x\|^2$ the residual stays bounded, the corrected iterate $\tilde x_t = x_t - e_t$ follows an SGD-like recurrence, and smoothness lets the rate match SGD asymptotically up to higher-order terms; the guarantee is not that every compressed step equals the dense step but that omitted information is merely delayed in a controlled way. Second, warm starting: starting from a fresh random $Q$ each step throws away information, but under small SGD steps the gradient distribution drifts slowly, so the dominant subspace at this step is close to the previous one. Reusing the previous $Q$ turns each step into a continuation of one long power iteration rather than a restart — an Oja-like stochastic subspace tracking — which is what makes a single left and a single right multiplication per step suffice.

The orthogonalization discipline keeps the factors from absorbing arbitrary scale. We normalize the warm-started $Q$ before multiplying, compute and reduce $P$, orthonormalize the reduced $P$, compute and reduce $Q$, store that reduced $Q$ for the next step, and on the next iteration normalize it again before use; for a rank-one factor "orthonormalize" is just dividing by the column norm, and otherwise it is a reduced QR. Vector-shaped tensors carry no useful matrix rank and are small, so they are sent dense; convolutional weight gradients are viewed as $(\text{out\_channels}, \text{in\_channels}\cdot k_h\cdot k_w)$. Rank is the natural knob — rank one, two, four — so a fixed `rank` is exposed first; a `compress_ratio` is only a translation layer, solving $(m+n)r \le \text{ratio}\cdot mn$ and capping at $\min(m,n)$, and matrices whose factorization would not save at least a minimum compression rate are simply passed through dense. What the method delivers, stated exactly: it trades one dense gradient all-reduce for two fixed-shape low-rank all-reduces, uses warm-started subspace iteration to make those factors useful, and uses error feedback so the omitted part stays in local memory instead of vanishing.

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
