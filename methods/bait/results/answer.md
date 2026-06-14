# BAIT, distilled

BAIT (Batch Active learning via Information maTrices) is a pool-based batch active-learning rule
for neural networks. It views the network as a probability model `p(y | x, θ)`, takes the
Fisher-information / MLE-error objective from classical estimation theory, and makes it tractable
at neural scale. Each round it greedily selects the batch whose accumulated last-layer Fisher best
"covers" the pool Fisher, in the A-optimality (trace) sense, using a forward-oversample /
backward-prune greedy made efficient by the Woodbury identity and a trace rotation. It applies
unchanged to classification and regression, and to convex models.

## Problem it solves

Batch active learning for deep networks: at each round pick `B` pool points to label, retrain, and
repeat, to reach low loss with the fewest labels. Needs to trade off uncertainty, diversity, and
representativeness, be principled (so its behavior is predictable and extensible), and run at
neural dimensionality — properties no prior method had together.

## Key idea

Read the network as `p(y | x, θ)` with loss `ℓ = -log p`; fitting is MLE. The expected excess
MLE error (and, equivalently, the Bayes risk in the linear-Gaussian case) of labeling a set `S` is
governed by the **weighted trace of the inverse Fisher**:

```
S* = argmin_{S ⊂ U, |S| ≤ B}  tr( (Σ_{x ∈ S} I(x; θ))^{-1}  I_U(θ) ),
```

where `I(x; θ) = E_{y∼p(·|x,θ)} ∇²ℓ(x,y;θ)` is the per-example Fisher (label-independent for GLMs
and Gaussian regression) and `I_U(θ) = (1/|U|) Σ_{x∈U} I(x; θ)` is the pool Fisher. This is
A-optimality with the pool Fisher as the target metric.

Why trace (A-optimality) and not determinant (D-optimality): the trace objective *is* the MLE
error / Bayes risk, and it genuinely couples the selected Fisher to the pool via `I_U`. A
determinant cannot: `det(I(x;θ)^{-1} I(θ)) = det(I(x;θ)^{-1}) · det(I(θ))` and `det(I(θ))` is
constant in `x`, so `argmax_x det(I(x;θ)^{-1} I(θ)) = argmax_x det(I(x;θ)^{-1})` for *any* `I(θ)` —
a determinant is structurally blind to the pool geometry. The pool term `I_U` is what makes
selection robust even on poor / random feature bases; dropping it reduces BAIT to per-point Fisher
maximization (a known, weaker objective).

## Where the objective comes from

- **Bayesian linear regression (Bayes risk).** Prior `θ* ~ N(0, λ^{-1}I)`, noise `σ²`; MAP = ridge
  with regularizer `λσ²`. For `Λ_S = Σ_{x∈S} xx^T + λσ²I` and pool second moment
  `Σ = (1/n)Σ_i x_i x_i^T`, the Bayes risk is exactly `BayesRisk(S) = σ² tr(Λ_S^{-1} Σ)` — and the
  RHS has no labels, so it can be minimized before querying.
- **Frequentist MLE error** (Chaudhuri, Kakade, Netrapalli & Sanghavi, NeurIPS 2015). The expected
  excess log-likelihood error of the MLE on `m` labels from distribution `Γ` is, with matched
  upper/lower bounds, `tr(I_Γ(θ*)^{-1} I_U(θ*)) / m`. Same weighted trace.

The two coincide in linear regression (`I(x;θ) = xx^T/σ²`), differing only by the regularizer `λ`.

## Three neural obstacles → three fixes

1. Full-network `I(x;θ)` is enormous → use the **last-layer Fisher** `I(x;θ^L)`; `θ^L = θ` for
   linear models.
2. The representation shifts every round → **recompute** the Fisher each round (iterative, unlike
   the convex one-shot two-phase scheme).
3. Exact selection is an SDP, infeasible in high dimensions → **greedy** selection.

## Algorithm (forward-backward greedy)

The trace functional is **not submodular**, so plain forward greedy is suboptimal. BAIT
oversamples forward to `2B`, then prunes backward to `B` (oversample factor 2 chosen for the
compute/quality trade-off; larger gave no gain).

```
Init S with B_0 random labels; fit θ_1.
for t = 1..T:
    I(θ_t^L) = (1/|U|) Σ_{x∈U} I(x; θ_t^L)                 # pool Fisher
    M_0 = λI + (1/|S|) Σ_{x∈S} I(x; θ_t^L)                 # seed Fisher + ridge prior
    for i = 1..2B:   # forward (oversample)
        x̃ = argmin_{x∈U} tr( (M_i + I(x;θ_t^L))^{-1} I(θ_t^L) );  M_{i+1} = M_i + I(x̃;θ_t^L)
    for i = 2B..B+1: # backward (prune)
        x̃ = argmin_{x∈S} tr( (M_i - I(x;θ_t^L))^{-1} I(θ_t^L) );  M_{i-1} = M_i - I(x̃;θ_t^L)
    query labels for S; retrain θ_t on S.
```

## Efficient per-candidate evaluation (Woodbury + trace rotation)

Write `I(x;θ^L) = V_x V_x^T`, with `V_x` a `dk × k` matrix whose columns are the per-class
last-layer loss gradients scaled by `√(class probability)` (so `V_x V_x^T = x^L x^L^T ⊗
(diag(π) - π π^T)`, the exact multiclass-logistic Fisher). Then with `A = I + V_x^T M_i^{-1} V_x`
(a `k × k`, easily inverted matrix):

```
tr((M_i + V_x V_x^T)^{-1} I(θ))
   = tr(M_i^{-1} I(θ))                              # constant in x
   - tr(M_i^{-1} V_x A^{-1} V_x^T M_i^{-1} I(θ))

⇒ argmin = argmax_x tr( V_x^T (M_i^{-1} I(θ) M_i^{-1}) V_x · A^{-1} )      # cyclic trace rotation
```

The rotation puts `V_x` outside, so `M_i^{-1} I(θ) M_i^{-1}` is precomputed once per step and every
candidate is scored with small (`k × k`, `dk × k`) products — all candidates in one batched GPU
matmul, never forming a `dk × dk` matrix per point. After selecting `x̃`, update the inverse with
the same identity: `M_{i+1}^{-1} = M_i^{-1} - M_i^{-1} V_{x̃} A^{-1} V_{x̃}^T M_i^{-1}`. The backward
pass is identical with `A = -I + V_x^T M_i^{-1} V_x`.

## Regression reduction

For `k`-output regression `y = Wx + N(0, Σ)`, `I(x;W) = xx^T ⊗ Σ^{-1}`. By Kronecker algebra the
noise covariance cancels:

```
tr( (Σ_{x∈S} I(x;θ))^{-1} I_U(θ) ) = k · tr( (Σ_{x∈S} x^L x^L^T)^{-1} (Σ_{x∈U} x^L x^L^T) ).
```

So regression uses **rank-one** `x^L x^L^T` instead of rank-`k` `V_x V_x^T` — even cheaper,
roughly coreset-speed. (BADGE cannot do regression at all: its hallucinated-gradient embedding
needs a most-likely class.)

## Relation to BADGE

BADGE's gradient embedding `g_x = ∇_{θ_out} ℓ(x, ŷ; θ_out)` (at the model's most-likely label `ŷ`)
is a **single column** of `V_x`, *unscaled* by `√p` — a rank-one shadow of `I(x;θ)`. BADGE
maximizes the Gram determinant of these `g_x` (≈ k-DPP / D-optimality), via k-means++ seeding.
BAIT's two gains: (i) the full rank-`k` Fisher `V_x V_x^T` instead of one gradient column, and
(ii) the pool Fisher `I(θ)` (impossible inside a determinant). BAIT pays ~`k×` more compute per
selected point, justified when label cost dominates compute.

## Working code

Fills the `query` slot of the harness. `get_exp_grad_embedding` returns `V_x` for each point
(per-class last-layer gradients scaled by `√p`); `select` runs the A-optimal forward-backward
greedy with Woodbury + trace-rotation scoring.

```python
import numpy as np
import torch


def select(X, K, fisher, iterates, lamb=1, nLabeled=0):
    """Greedy A-optimal batch selection: minimize tr((sum_S V V^T)^{-1} I(theta)).
    X[i] = V_{x_i}, shape (rank, dim), with V V^T = I(x; theta^L).
    fisher = I(theta^L) (pool Fisher); iterates = labeled-set Fisher seeding M_0.
    Forward-oversample to 2K, then backward-prune to K (the trace is not submodular)."""
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    # M_0^{-1} = (lamb*I + labeled_Fisher * nLabeled/(nLabeled+K))^{-1};
    # scale candidate factors by sqrt(K/(nLabeled+K)) so the new batch joins the seed in proportion.
    currentInv = torch.inverse(lamb * torch.eye(dim) + iterates * nLabeled / (nLabeled + K))
    X = X * np.sqrt(K / (nLabeled + K))

    # ---- forward selection, over-sample by 2x ----
    over_sample = 2
    for i in range(int(over_sample * K)):
        # rotated-trace score for ALL candidates at once:
        #   score(x) = tr( V_x^T M^{-1} I(theta) M^{-1} V_x  (I + V_x^T M^{-1} V_x)^{-1} )
        innerInv = torch.inverse(torch.eye(rank) + X @ currentInv @ X.transpose(1, 2))
        traceEst = torch.diagonal(
            X @ currentInv @ fisher @ currentInv @ X.transpose(1, 2) @ innerInv,
            dim1=-2, dim2=-1).sum(-1)

        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:        # largest score = best decrease of the objective
            if j not in indsAll:
                ind = j
                break
        indsAll.append(ind)

        # Woodbury low-rank inverse update: M^{-1} <- M^{-1} - M^{-1} V A^{-1} V^T M^{-1}
        xt_ = X[ind].unsqueeze(0)
        innerInv = torch.inverse(torch.eye(rank) + xt_ @ currentInv @ xt_.transpose(1, 2))
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv)[0]

    # ---- backward pruning: remove K extras, deleting the least-useful each time ----
    for i in range(len(indsAll) - K):
        xt_ = X[indsAll]
        innerInv = torch.inverse(-1 * torch.eye(rank) + xt_ @ currentInv @ xt_.transpose(1, 2))
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
            dim1=-2, dim2=-1).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()

        xt_ = X[indsAll[delInd]].unsqueeze(0)
        innerInv = torch.inverse(-1 * torch.eye(rank) + xt_ @ currentInv @ xt_.transpose(1, 2))
        currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv)[0]
        del indsAll[delInd]

    return indsAll


class BaitSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BaitSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.lamb = args['lamb']                       # ridge / prior precision (default 1)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # rank-k Fisher factors V_x for the whole pool: per-class last-layer grads scaled by sqrt(p)
        xt = self.get_exp_grad_embedding(self.X, self.Y)   # [n_pool, k, d*k], V_x V_x^T = I(x; theta^L)

        # pool Fisher I(theta^L) = (1/|U|) sum_x V_x V_x^T
        batchSize = 1000
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        for i in range(int(np.ceil(len(self.X) / batchSize))):
            xt_ = xt[i * batchSize:(i + 1) * batchSize]
            fisher = fisher + torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / len(xt), 0)

        # seed Fisher from the already-labeled points (for M_0)
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[self.idxs_lb]
        for i in range(int(np.ceil(len(xt2) / batchSize))):
            xt_ = xt2[i * batchSize:(i + 1) * batchSize]
            init = init + torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / len(xt2), 0)

        chosen = select(xt[idxs_unlabeled], n, fisher, init,
                        lamb=self.lamb, nLabeled=np.sum(self.idxs_lb))
        return idxs_unlabeled[chosen]
```

Defaults: `λ = 1` (`0.01` for CIFAR-10), seed with ~100 random labels, retrain from scratch each
round (no warm-starting), oversample factor 2.
