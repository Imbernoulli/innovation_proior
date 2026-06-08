# Context

## Research question

Optimizers for large-model training fall into two camps. Diagonal adaptive methods (Adam/AdamW) keep
one scalar preconditioner per coordinate — cheap, but blind to correlations between coordinates of a
weight matrix. Non-diagonal (second-order) methods precondition with an actual matrix that captures
those correlations and converge in far fewer steps, but the matrices involved are enormous and have to
be inverted or eigendecomposed, which is expensive enough that practitioners refresh them only
occasionally — and the staler the matrix, the more the benefit erodes. The question is whether the
convergence advantage of a non-diagonal preconditioner can be retained at close to diagonal-method cost
and stability: an optimizer that preconditions with cross-coordinate curvature, costs only a little
more than Adam, adds essentially no new hyperparameters, and — crucially — does not fall apart when its
expensive part is computed infrequently.

## Background

**Full-matrix Adagrad (Duchi et al. 2011).** For a weight matrix W ∈ ℝ^{m×n} with vectorized gradient
g = vec(G) ∈ ℝ^{mn}, accumulate H = Σ g gᵀ ∈ ℝ^{mn×mn} and step w ← w − η H^{-1/2} g. This is the ideal
non-diagonal preconditioner — it whitens the gradient using the full second-moment matrix — but H is
mn×mn and its inverse square root is hopeless at scale.

**Adam as a diagonal approximation (Kingma & Ba 2015).** Adam keeps only the diagonal of the
second-moment matrix: an EMA of the elementwise squared gradient Vₜ, and steps W ← W − η Mₜ/√Vₜ (Mₜ the
first-moment EMA). Cheap, but it sees no correlation between coordinates.

**Adafactor (Shazeer & Stern 2018; Zhai et al. variant).** A memory-light Adam: replace the
second-moment matrix Vₜ by its best rank-1 approximation V'ₜ (outer product of a row factor and a
column factor) and step W ← W − η Mₜ/√V'ₜ. Still diagonal in spirit, but factored to save memory.

**Shampoo (Gupta et al. 2018; Anil et al. 2020).** A structured (Kronecker) approximation of
full-matrix Adagrad. Maintain two per-side preconditioners Lₜ = Σ Gₜ Gₜᵀ ∈ ℝ^{m×m} and
Rₜ = Σ Gₜᵀ Gₜ ∈ ℝ^{n×n}, and step W ← W − η Lₜ^{-1/4} Gₜ Rₜ^{-1/4}. Vectorized this is preconditioning by
(Lₜ ⊗ Rₜ)^{-1/4} — a Kronecker factorization of the mn×mn preconditioner into an m×m and an n×n piece,
which captures left- and right-side correlations of the weight matrix at tractable size. Practical
findings the method rests on: using power 1/2 instead of 1/4 (i.e. L^{-1/2} G R^{-1/2}) performs better
and aligns with the optimal Kronecker approximation of the Adagrad preconditioner (Anil et al. 2020;
Morwani et al. 2024); and a scalar per-layer learning-rate correction normalizes the
Kronecker factor as L ⊗ R / Trace(L), so each eigen-coordinate is scaled by
(λ_i μ_j / Trace(L))^{-1/2}. The attraction is that Shampoo is a practical non-diagonal optimizer
already used in optimization-efficiency benchmarks and large training runs. The drawbacks: it carries
extra hyperparameters (the exponent, learning-rate grafting), and the L^{-1/2}, R^{-1/2} require an
eigendecomposition / inverse-root of L and R, which is costly enough that implementations compute it
only every f steps — between refreshes the preconditioner is stale, so the diagnostic observation is
that Shampoo's effective adaptivity is limited to the cadence at which L and R are refreshed, and its
performance degrades as f grows.

**K-FAC and E-KFAC (Martens & Grosse 2015; George et al. 2018).** A separate second-order family that
preconditions with a Kronecker-factored Fisher. E-KFAC's refinement is the structural idea that matters
here: it inserts a *diagonal* preconditioner that is updated *between* the (expensive) eigen-refreshes,
in the eigenbasis of the second-order factor — i.e. it runs a cheap diagonal method in a slowly-changing
basis provided by a second-order method, getting per-step adaptivity for free between refreshes.

**GaLore (Zhao et al. 2024).** Projects Adam's momentum into a low-rank subspace from the SVD of the
*current* gradient to save memory. Two structural choices contrast with a curvature-basis approach: the
subspace comes from the instantaneous gradient SVD (not an accumulated GGᵀ/GᵀG average), and momentum is
kept in the projected space and not rotated when the subspace changes; it also projects only one side.

The standing tension among these: the diagonal methods (Adam, Adafactor, Lion, Sophia) are cheap but
do not surpass AdamW for LLM pretraining (Kaddour et al. 2023; Zhao et al. 2024), pointing to non-diagonal
preconditioning as the lever; but the non-diagonal method (Shampoo) pays for its eigen-refresh and
loses adaptivity when that refresh is infrequent.

## Baselines

**AdamW (Loshchilov & Hutter 2019).** Mₜ = β₁Mₜ₋₁ + (1−β₁)Gₜ; Vₜ = β₂Vₜ₋₁ + (1−β₂)Gₜ²; bias-correct;
W ← (1−ηλ)W − η Mₜ̂/(√Vₜ̂ + ε). The robust diagonal default. Gap: per-coordinate preconditioning only;
no cross-coordinate curvature, so more steps to a given loss on large-model pretraining.

**Shampoo, power-1/2 variant (Gupta et al. 2018; Anil et al. 2020).** Lₜ = ΣGGᵀ, Rₜ = ΣGᵀG;
W ← W − η Trace(Lₜ)^{1/2} Lₜ^{-1/2} Gₜ Rₜ^{-1/2}, equivalently scaling eigen-coordinate (i,j) by
(λ_i μ_j / Trace(Lₜ))^{-1/2}. Non-diagonal Kronecker preconditioning; strong convergence. Gap: needs an
eigendecomposition / inverse-root of L and R, computed only every f steps; its adaptivity is tied to
that cadence and degrades as f grows; extra hyperparameters (exponent, grafting).

**Adafactor (Shazeer & Stern 2018).** W ← W − η Mₜ/√V'ₜ with V'ₜ the rank-1 factorization of the
second moment. Memory-light diagonal method. Gap: still diagonal — no cross-coordinate structure.

**E-KFAC (George et al. 2018).** Diagonal preconditioner updated between eigen-refreshes in K-FAC's
eigenbasis. Establishes the run-a-diagonal-method-in-a-second-order-eigenbasis template. Gap: its
between-refresh diagonal preconditioner is not Adam, and it is built for the Fisher, not for the
Shampoo (GGᵀ/GᵀG) factors.

## Evaluation settings

The natural yardstick, all pre-existing: language-model pretraining at fixed model and data budget
(decoder-only transformers of a few hundred million parameters; standard web-text corpora; a fixed
token budget, e.g. Chinchilla-optimal ≈20× model size or a longer 100× budget), comparing optimizers by
training/validation cross-entropy loss versus training iterations and versus wall-clock time, under a
cosine learning-rate schedule with warmup and a per-optimizer learning-rate sweep. Secondary axes are
robustness to the preconditioner-refresh frequency and behavior across batch sizes (the critical batch
size). The cost side is measured as the multiplicative wall-clock / FLOP overhead of the optimizer
relative to AdamW.

## Code framework

The substrate is a per-layer optimizer plugged into a transformer pretraining loop, plus the
already-existing primitives: per-layer EMA buffers, a routine to extract an orthonormal eigenbasis of a
small PSD matrix (eigendecomposition, or power-iteration-plus-QR), and rotation of a gradient into and
out of such a basis. The open slot is the *per-layer preconditioned update*: what second-order
statistics to accumulate, what coordinate system to express the adaptive step in, and how often to
refresh that coordinate system.

```python
import torch
from torch.optim.optimizer import Optimizer


def eigenbasis(P, Q_prev=None):
    """Orthonormal eigenvectors of a small PSD matrix P (m×m or n×n).
    First call: full eigendecomposition. Later: one power-iteration step + QR from Q_prev."""
    if Q_prev is None:
        _, Q = torch.linalg.eigh(P)
        return torch.flip(Q, dims=[1])           # descending eigenvalue order
    Q, _ = torch.linalg.qr(P @ Q_prev)           # power iteration + QR
    return Q


class PreconditionedOptimizer(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=0.01, precondition_frequency=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd, f = (group["lr"], group["eps"], group["weight_decay"],
                              group["precondition_frequency"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)       # first moment
                    state["exp_avg_sq"] = torch.zeros_like(p)    # second moment
                    # TODO: per-side curvature accumulators and their eigenbases
                state["step"] += 1

                # TODO: accumulate cross-coordinate curvature on each side of the
                #       weight matrix; express an adaptive (Adam-style) step in a
                #       slowly-changing coordinate system derived from that curvature,
                #       refreshing the coordinate system only every f steps; then map
                #       the step back and apply it with decoupled weight decay.
                pass
        return None
```
