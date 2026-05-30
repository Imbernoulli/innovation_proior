# Context: low-memory adaptive optimization for large neural networks

## Research question

Adaptive gradient methods divide each parameter's step by a running estimate of the magnitude of that parameter's recent gradients. This per-coordinate rescaling is what makes them robust on the badly-conditioned, heterogeneous objectives that arise when training deep networks — different parameters see gradients of wildly different scales, and dividing each by its own gradient history puts them all on a common footing. It is the reason these methods empirically outperform plain SGD on language and translation models.

The robustness comes at a memory price. To rescale every parameter individually, the optimizer must store one running statistic *per parameter* — a buffer the same size as the model. Methods that also keep momentum store a second such buffer. So the optimizer state is one or two full copies of the model, on top of the parameters and their gradients.

This matters because the binding constraint on model size has shifted. The compute available for training has grown far faster than the memory available to hold the model and its training state, and task quality on data-rich problems (language modeling, machine translation) keeps improving as models grow into the billions of parameters. At that scale the optimizer's auxiliary buffers are no longer a rounding error — they are a primary reason a model does not fit. The precise problem: keep the per-coordinate adaptivity that makes these methods work, but pay *sublinear* extra memory in the number of parameters of a large weight matrix — `O(n + m)` instead of `O(nm)` for an `n × m` matrix — without losing model quality. A solution must also remain cheap to compute (the memory-saving step cannot become a training bottleneck) and must produce a nonnegative estimate, since the step divides by its square root.

## Background

**The adaptive-method family.** A line of methods rescales the gradient componentwise by the square root of an accumulated statistic of past squared gradients. Adagrad (Duchi et al., 2011) sums squared gradients over all of history, `G_t = G_{t-1} + g_t²`, and steps `x_t = x_{t-1} − α g_t / (√G_t + ε)`; this helps most when gradients are sparse, but the denominator grows without bound so the effective step decays to zero. RMSProp (Tieleman & Hinton, 2012) and Adadelta (Zeiler, 2012) replace the unbounded sum with an exponential moving average `v_t = β₂ v_{t-1} + (1−β₂) g_t²`, so the estimate tracks the *current* gradient scale rather than the whole past. Adam (Kingma & Ba, 2015) combines this second-moment EMA with a first-moment (momentum) EMA and a bias correction for the zero initialization. Across these methods the second-moment accumulator has exactly the shape of the parameters: a per-coordinate memory of recent gradient magnitude.

**The memory diagnostic.** Each EMA buffer is one full model-sized array. Adam keeps two — first and second moment — so its optimizer state is twice the parameter count; together with the parameters and the gradient, training holds several full copies of the model. For a large weight or embedding matrix this is the dominant consumer of auxiliary memory. Removing even one of these buffers, or shrinking it from `O(nm)` to `O(n+m)`, directly raises the largest model that fits.

**Structure to exploit.** Much of a network's parameter count lives in two-dimensional weight matrices used for linear transformations and embeddings. A matrix has structure a flat vector does not: rows and columns. The second-moment accumulator `V ∈ ℝ^{n×m}` for such a matrix is, entry by entry, an exponential moving average of squared gradients, hence **nonnegative**. A nonnegative matrix is exactly the object that nonnegative matrix factorization and low-rank approximation study, which makes "store a compressed `V` and reconstruct it" a concrete question rather than a vague hope.

**Low-rank approximation, and why the obvious tool fails here.** The classical optimal low-rank approximation is the truncated singular value decomposition: keeping the top `k` singular values/vectors gives the best rank-`k` approximation in Frobenius norm (Eckart & Young, 1936). Two facts make it the wrong tool for a *running* accumulator. First, the SVD of a sum is not built from the SVDs of the summands, so the factors of an exponentially smoothed `V` cannot be obtained by smoothing the factors of the instantaneous squared gradients — the moving average of the factors is not the factors of the moving average. Second, SVD factors can be negative, and a negative reconstructed second moment cannot be square-rooted. Both are disqualifying for an accumulator that must be updated by exponential smoothing and then fed into an inverse square root.

**Nonnegative matrix factorization and the I-divergence.** NMF (Lee & Seung, 1999) factors a nonnegative matrix into nonnegative factors under a cost function chosen to respect nonnegativity. Beyond the Frobenius norm, a standard NMF cost is the generalized Kullback–Leibler divergence, also called the I-divergence: for nonnegative scalars, `d(p, q) = p log(p/q) − p + q`, with `0/0 = 0`, `0 log 0 = 0`, and `p/0 = ∞` for `p > 0`. It is nonnegative and zero only when `p = q`, which follows from the elementary inequality `x log x ≥ x − 1`. General rank-`k` NMF under the I-divergence has no closed form and is solved by alternating minimization (Finesso & Spreij, 2006).

**Out-of-date second-moment estimates.** A separate, known difficulty with EMA-based adaptive methods concerns the decay rate `β₂`. Reddi et al. (2018), analyzing the convergence of Adam, show that a *fast*-decaying second moment (small `β₂`) can cause nonconvergence. But a *slow*-decaying second moment (large `β₂`) bases the denominator on gradients from far in the past; if the model is moving quickly, that stale estimate no longer matches the current gradient and the resulting step can be much larger than intended. A direct way to watch for this on a weight matrix `X` is the root-mean-square of the *unscaled* update `u = −g/√v̂` over the matrix's entries, `RMS(U) = √(mean(g²/v̂))`: when the estimator is doing its job, `g²/v̂ ≈ 1` and `RMS(U) ≈ 1`, so values far from `1` flag a stale, miscalibrated denominator.

**Gradient clipping.** A long-standing stabilizer (Pascanu et al., 2013) rescales the gradient down before the step whenever its norm exceeds a threshold. For plain SGD the step direction *is* the gradient, so this also bounds the distance moved. For an adaptive method the step is the gradient times per-coordinate scaling, so bounding the gradient norm does not bound the size of the actual update.

**Parameter scale and relative change.** A practitioner observation (attributed to Hinton): training behaves well when the magnitude of each parameter update is roughly `10⁻²`–`10⁻³` times the magnitude of the parameter itself — a *relative* notion of step size. Standard adaptive methods instead specify an *absolute* target step `α`. Making one absolute `α` work across parameter groups on very different scales (for example, an embedding matrix versus a deep weight matrix) typically requires hand-engineering the initialization scales — e.g. initializing embeddings small and multiplying them up in the forward pass — so that a single `α` is appropriate everywhere.

**Prior low-memory attempts.** Saving optimizer memory by sharing or averaging accumulators across structurally related parameters had been tried: averaging Adagrad accumulators across embedding vectors (Gupta et al., 2014), and a brief mention of a factored accumulator in the appendix of the sparsely-gated mixture-of-experts work (Shazeer et al., 2017). These establish that compressing the accumulator is plausible without committing to the optimal way to do it.

## Baselines

**Adam (Kingma & Ba, 2015).** The reference adaptive method. State: first-moment EMA `m_t = β₁ m_{t-1} + (1−β₁) g_t` and second-moment EMA `v_t = β₂ v_{t-1} + (1−β₂) g_t²`; bias-corrected `m̂_t = m_t/(1−β₁ᵗ)`, `v̂_t = v_t/(1−β₂ᵗ)`; update `x_t = x_{t-1} − α_t m̂_t/(√v̂_t + ε)`. Strong and robust on language/translation models. Gap: two model-sized buffers — optimizer state twice the parameter count — which is prohibitive for very large matrices.

**RMSProp / Adadelta.** Adam without the first moment (RMSProp) and a related unit-correcting variant (Adadelta). One model-sized buffer for the second moment. Gap: still `O(nm)` for a matrix; still an absolute step size; the second-moment decay-rate dilemma (fast decay can fail to converge, slow decay can produce stale, oversized updates) is unaddressed.

**Adagrad (Duchi et al., 2011).** Unbounded sum of squared gradients in the denominator; strong under sparsity. Gaps: the denominator only grows, so steps decay to zero; one full model-sized accumulator.

**SGD (with optional momentum).** Zero or one auxiliary buffer — the memory baseline to beat. Gap: no per-coordinate adaptivity, so on heterogeneous, badly-conditioned objectives it needs careful per-problem learning-rate tuning and underperforms adaptive methods on language/translation models.

**Increasing-decay-rate schedules (Reddi et al., 2018).** Proposed as a fix to Adam's nonconvergence: let the effective second-moment decay rate increase over training rather than stay fixed. The relevant prior idea to build on, not a full optimizer.

## Evaluation settings

The natural yardstick is a large attention-based sequence model (the Transformer of Vaswani et al., 2017) trained on the WMT 2014 English-to-German translation task, the setting where expensive models and large embedding/weight matrices make optimizer memory bite. Quality is reported as **BLEU** on the newstest2013 development set, decoded with beam search (beam size 4, length penalty 0.6). Training runs a fixed step budget (on the order of 100k steps) with batches of a few thousand tokens per side. Two learning-rate regimes probe stability: one with a linear warmup followed by inverse-square-root decay, `s_t = min(10⁻⁶·t, 1/√t)`, and one with the warmup removed, `s_t = min(10⁻², 1/√t)`, since instabilities surface when the early warmup is absent. The decay-rate dilemma is probed by sweeping `β₂` (e.g. fast `0.9` versus slow `0.999`) with and without warmup. Plain SGD across a range of learning-rate multipliers is included as the zero-extra-memory reference. The comparison of interest is matched-budget quality and training stability at a given memory cost, not raw state-of-the-art.

## Code framework

A standard training stack already supplies minibatches, autodiff gradients, and a base optimizer abstraction that stores per-parameter state and applies an update inside the training loop. The missing rule lives inside `step`: how to represent the second-moment statistic when the parameter is a matrix, how to reconstruct the per-coordinate denominator, and how to turn the rescaled direction into a stable parameter update.

```python
import math
import torch
from torch.optim import Optimizer


class LowMemoryAdaptiveOptimizer(Optimizer):
    """Per-coordinate adaptive optimizer with sublinear extra memory.

    For each parameter, maintain a running second-moment statistic of the
    squared gradient and step along the gradient rescaled by its inverse
    square root. The implementation must decide how to store that statistic
    for a matrix-shaped parameter and how to reconstruct the per-coordinate
    denominator from it.
    """

    def __init__(self, params, lr=None, eps=None,
                 clip_threshold=None, decay_rate=None, beta1=None):
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, beta1=beta1)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _get_lr(group, state):
        # TODO: choose the step size for this parameter
        pass

    @staticmethod
    def _approx_sq_grad(stored_stat_a, stored_stat_b):
        # TODO: reconstruct an inverse square-root denominator from the stored
        #       representation of the second-moment statistic
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                # TODO: initialize per-parameter state -- a representation of
                #       the second-moment statistic whose size is sublinear in
                #       the size of a matrix-shaped parameter
                # TODO: update that statistic from grad**2
                # TODO: reconstruct the per-coordinate denominator and form the
                #       rescaled update direction from grad
                # TODO: turn the direction into the actual step and apply it
                pass
        return loss


# training loop (pre-existing)
def train(model, data, optimizer, steps):
    for _ in range(steps):
        x, y = next(data)
        loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
