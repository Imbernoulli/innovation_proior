SignSGD came back as the strongest baseline: ResNet-56/CIFAR-10 recovered to $\{94.33, 93.98, 94.12\}$, mean $94.14$, above EF-TopK's $93.85$ and QSGD's $94.08$; VGG-11-BN/CIFAR-100 held and edged up to $\{70.82, 70.82, 70.66\}$, mean $70.77$; ResNet-20 came in tightest at $92.52$. But that is the tell: all three baselines have converged to roughly the same accuracy ceiling, all of it bought by the same error-feedback memory, differing only in which information they delete and delay. The benchmark simulates a *single* worker, and so it has been hiding the cost that matters most at scale. Look at what each does at the all-reduce. EF-TopK keeps the $k$ largest coordinates — but worker A's top-$k$ and worker B's touch *different* coordinates, so the sum of $W$ sparse vectors has up to $Wk$ nonzeros; the aggregate is not $k$-sparse, the cheap ring all-reduce (cost independent of $W$) does not apply, and the framework falls back to an all-gather whose cost grows with $W$. SignSGD's scale-version compresses the return trip by majority vote $\text{sign}[\sum_w \text{sign}(g_w)]$, a non-linear collective; QSGD is worse still, since summing two workers' quantized codes is not a quantized code. Every baseline breaks the one property that makes data-parallel training fast — the $W$-independent all-reduce. Accuracy is saturated; aggregateability is not.

So the requirement sharpens: a compressor $C$ that is aggressive enough for 100×, accurate enough to match sign-with-EF, and *linear in the gradient*, so that $\sum_w C(g_w) = C(\sum_w M_w)$ is itself a compressed object of fixed shape that all-reduces cheaply. Linearity rules out coordinate sparsity — "which coordinates are largest" is data-dependent per worker, so top-$k$ is fundamentally non-linear across workers — and pushes me to look at what a gradient *actually is* rather than a flat vector of $d$ numbers. A weight gradient for a conv or linear layer is naturally a matrix $M \in \mathbb{R}^{m\times n}$ (fold a conv's output channels against everything else). Besides coordinate sparsity, matrices have a second low-complexity structure: low *rank*. Gradient matrices are approximately low-rank — their singular spectra decay fast, so most of the Frobenius energy lives in the top few singular directions. A rank-$r$ factorization $M \approx P Q^\top$ with $P \in \mathbb{R}^{m\times r}$, $Q \in \mathbb{R}^{n\times r}$ captures that in $(m+n)r$ numbers instead of $mn$, easily 100× at small $r$ — and, crucially, it is a *linearly accessible* structure.

I propose **PowerSGD — warm-started low-rank compression with error feedback**. The exact best rank-$r$ approximation is the truncated SVD, but that is *not* linear in $M$: the singular vectors of $M_A + M_B$ are not those of $M_A$ plus $M_B$, so summing per-worker SVDs reintroduces the support-mismatch disease in singular-vector space. I need an *approximate* low-rank scheme whose two factors come from *linear* operations, and the power method / randomized range finder gives exactly that. To find the top-$r$ subspace of $M$: pick an $n\times r$ matrix $Q$, form $P = MQ$ (its columns approximate $M$'s top-$r$ column space), orthonormalize $P$, project back $Q = M^\top P$, and reconstruct $M \approx P Q^\top$. The two heavy operations $P = MQ$ and $Q = M^\top P$ are *both linear in $M$*. If $Q$ is held fixed and identical across workers, $\sum_w P_w = (\sum_w M_w)\,Q$ — all-reduce the small $P_w$ and get $P$ for the *aggregate* gradient; once $P$ is agreed (orthonormalized identically from the all-reduced $P$), $\sum_w Q_w = (\sum_w M_w)^\top P$ all-reduces too. The two matrix multiplies of one power step are precisely the two linear, all-reduceable operations the systems constraint demands, each a small fixed-shape $W$-independent collective. *One step* of subspace iteration, not a full SVD, is the compressor.

One power step from a *random* $Q$ is a poor approximation — random subspace iteration needs several passes to land on the top-$r$ subspace, and I can afford only one matrix-multiply pair per gradient. So a single step gives a crude low-rank approximation that deletes real gradient energy — a biased, contractive compressor, exactly the kind the last two rungs taught me to rescue with **error feedback**. Keep a per-tensor residual $e$, compress $p = g + e$, send the low-rank $C(p)$, stash $e \leftarrow p - C(p)$ — the deleted trailing spectrum is the residual, paid back next step. The virtual-iterate identity is the same one that made EF-TopK and sign-with-EF converge: $\tilde x = x - e$ runs exact SGD, the residual stays bounded because the rank-$r$ projection is contractive, and the leading-order rate matches uncompressed SGD. Error feedback is compressor-agnostic, which is why I can reuse it a third time.

The second idea is what makes *one* power step actually good, and it has no analog in the baselines. The waste in one-step subspace iteration is starting $Q$ from a fresh random matrix every step, discarding everything learned last step. But the gradient's top-$r$ subspace does not jump between consecutive SGD steps — it drifts slowly. So **warm-start**: reuse the previous step's $Q$ as this step's starting matrix. Then a single power step is no longer one iteration from scratch — it is one step of a power iteration *running continuously across the whole training run*, because the input gradients change slowly. The subspace estimate compounds across iterations, so warm-started rank-1 or rank-2 reaches the accuracy that cold-restart random low-rank would need many power passes to get; warm-start buys the accuracy of many iterations at the cost of one. The two ideas lock together: warm-start makes the single-step approximation good enough that the error-feedback residual is small, and error feedback catches whatever the single step still misses. That is how a method spending only $(m+n)r$ numbers matches a dense gradient's accuracy — what the saturated baselines do, but now *linearly* and so cheaply aggregateable.

The fill carries per-name state: the residual $e[\text{name}]$ and the warm-start $Q[\text{name}]$, both local — only $(P,Q)$ go on the wire. On `compress`, error-correct $M = (g + e[\text{name}])$ reshaped to a matrix; if the tensor is 1-D (bias, BN scale/shift — rank is meaningless and they are tiny), pass it through dense with no residual; otherwise pick rank $r = \max(1, \lfloor \text{ratio}\cdot mn/(m+n)\rfloor)$ capped at $\min(m,n)$ so each layer hits the target compression rather than a global fixed $r$; initialize $Q$ random $n\times r$ only if missing or shape-changed — the only randomness, once per tensor, every later step warm-starts. Then $P = MQ$; orthonormalize $P$ with QR (keep only the orthonormal factor); $Q = M^\top P$; stash $Q[\text{name}] = Q$ for next step; reconstruct $\hat M = P Q^\top$ and set $e[\text{name}] = M - \hat M$, reshaped back. The payload is $[P, Q]$ — $(m+n)r$ floats — plus a context with shape and a low-rank-vs-passthrough flag; `decompress` forms $P Q^\top$ and reshapes, or returns the dense passthrough. The invariant to hold is that the residual is exactly $M - \text{decompress}(\text{compress}(M))$, corrected-input minus reconstructed-output, and the stored $Q$ is the *post-projection* factor, so the warm-start starts from the subspace this step found.

This is the rare compressor that is aggressive, convergent, *and* fast on real clusters. The bar is sign-with-EF's real numbers. On ResNet-56/CIFAR-10, warm-started low-rank with EF should match or beat $94.14$ — the dense gradient is approximately low-rank, so $P Q^\top$ captures it about as well as a dense one-bit signal and the warm-start tracks the drifting subspace. On VGG-11-BN/CIFAR-100 it should hold the $\sim\!70.8$ ceiling with no collapse — the residual carries the trailing spectrum, and the VGG weights are the largest matrices where low-rank has the most room, so I expect $\sim\!70.5$–$71$. On ResNet-20, $\sim\!92.5$. The accuracy claim is "match the strongest baseline at 100×"; the claim the benchmark cannot score but that motivated the whole rung is "do it with a *linear*, all-reduceable compression that keeps the $W$-independent collective top-$k$ and majority-vote forfeit" — a single warm-started power step, $P = MQ$, orthonormalize, $Q = M^\top P$, built precisely from the two linear maps that survive aggregation, with error feedback paying back the deleted spectrum, so it earns the baselines' accuracy without paying their communication tax. Earlier error feedback is from Karimireddy et al. (2019).

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
