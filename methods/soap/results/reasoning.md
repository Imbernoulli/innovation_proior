Let me start from the thing that actually bugs me about the optimizer everyone trains large models with. AdamW keeps, per coordinate, an EMA of the gradient and an EMA of the squared gradient, and it divides one by the square root of the other. That square root is a diagonal preconditioner — it gives every coordinate its own learning rate, and it is completely blind to how coordinates of a weight matrix relate to one another. A hidden layer's weight W ∈ ℝ^{m×n} is not a bag of mn independent scalars; it is a linear operator, and its gradient G has structure — its rows are correlated, its columns are correlated — that a diagonal method throws away. The thing that would use that structure is the full second-moment matrix.

So write down the ideal. Full-matrix Adagrad vectorizes the gradient, g = vec(G) ∈ ℝ^{mn}, accumulates H = Σ g gᵀ, and steps w ← w − η H^{-1/2} g. That H^{-1/2} whitens the gradient using every pairwise correlation between the mn coordinates — the genuine non-diagonal preconditioner. And it is hopeless: H is mn×mn, so for even a modest m = n = 1024 that is a million-by-million matrix whose inverse square root I would have to keep current. Nobody does this. Adam's answer is to keep only the diagonal of H; that is exactly why it is cheap and exactly why it is blind.

Shampoo's answer is the interesting middle. Instead of the full H or just its diagonal, approximate H by a Kronecker product of two small matrices: a left factor L = Σ G Gᵀ ∈ ℝ^{m×m} that captures row correlations and a right factor R = Σ Gᵀ G ∈ ℝ^{n×n} that captures column correlations. The preconditioned step is W ← W − η L^{-1/4} G R^{-1/4}, which in vectorized form is preconditioning by (L ⊗ R)^{-1/4}. So I have factored the mn×mn preconditioner into an m×m piece and an n×n piece — tractable — while still seeing both sides' correlations. Practical Shampoo work also gives me two refinements I should not ignore: use power 1/2 rather than 1/4 (so L^{-1/2} G R^{-1/2}), which matches the optimal Kronecker approximation of the Adagrad preconditioner; and normalize the Kronecker factor as L ⊗ R / Trace(L), so the inverse-square-root step carries a Trace(L)^{1/2} scalar.

But Shampoo has a cost that bites in exactly the regime I care about. To form L^{-1/2} and R^{-1/2} I need an eigendecomposition (or an inverse matrix root) of L and of R. That is expensive — cubic in the side length — so no one does it every step; they do it every f steps and reuse the stale roots in between. And here is the diagnostic that bothers me: between refreshes, Shampoo's preconditioner is frozen. Its only adaptivity is the periodic recomputation of L^{-1/2}, R^{-1/2}. So as I push f up to save compute, the preconditioner gets staler and performance degrades — the very lever I would pull to make it affordable is the one that kills it. Plus it drags extra hyperparameters (the exponent, learning-rate grafting). I want Shampoo's non-diagonal preconditioning without paying the eigen-refresh every step and without losing adaptivity when I refresh rarely.

Let me look harder at what Shampoo is actually computing, because I have a hunch it is secretly a diagonal method wearing a costume, and if so I can attack it where it is cheap. Take the idealized power-1/2 Shampoo with dataset-average factors. Rotate the gradient into the eigenbasis of the two factors: let Q_L hold the eigenvectors of L (with eigenvalues λ₁,…,λ_m) and Q_R the eigenvectors of R (eigenvalues μ₁,…,μ_n), and define G' = Q_Lᵀ G Q_R. In this rotated frame, what does Shampoo's step do to each coordinate? The preconditioner L^{-1/2} ⊗ R^{-1/2}, with the Trace(L) scalar correction, scales the (i,j) coordinate of the rotated gradient by (λ_i μ_j / Σ_i λ_i)^{-1/2}. That is a diagonal rescaling — in the eigenbasis, Shampoo is just dividing each coordinate by the square root of λ_i μ_j (up to the trace scalar). So in the right coordinate system Shampoo is diagonal.

That diagonal rescaling is suspiciously close to the row-column scaling in Adafactor. Adafactor, the memory-light Adam, scales coordinate (i,j) by (A_i C_j / Σ A)^{-1/2}, where A is the row-marginal and C the column-marginal of the squared rotated gradient — A_i = E_B[Σ_j (G'_B)²_{ij}] and C_j = E_B[Σ_i (G'_B)²_{ij}]. So Adafactor in this rotated frame scales by (A_i C_j / ΣA)^{-1/2}, and Shampoo scales by (λ_i μ_j / Σλ)^{-1/2}. These are the same if A_i = λ_i and C_j = μ_j. Let me check A_i, because if it holds the whole picture collapses into something I can exploit. With u_i the i-th column of Q_L (the i-th eigenvector of L) and v_j the columns of Q_R,

  A_i = E_B[ Σ_j (G'_B)²_{ij} ] = E_B[ Σ_j (u_iᵀ G_B v_j)² ].

The v_j form an orthonormal basis, so Σ_j (u_iᵀ G_B v_j)² = ‖u_iᵀ G_B‖² = u_iᵀ G_B G_Bᵀ u_i. Take the expectation: A_i = u_iᵀ E_B[G_B G_Bᵀ] u_i = u_iᵀ L u_i = λ_i, because u_i is the eigenvector of L = E[GGᵀ] with eigenvalue λ_i. The same argument on the other side gives C_j = μ_j. So A_i = λ_i, C_j = μ_j exactly, and the two rescalings coincide.

So Shampoo (power 1/2, with the trace correction and dataset-average factors) is identical to running Adafactor in the eigenbasis of Shampoo's preconditioner. The expensive eigendecomposition was never the adaptivity — it was just computing the basis. Once I am in that basis, the actual preconditioning is a cheap, elementwise, diagonal rescaling. This is the same shape as E-KFAC, which runs a diagonal preconditioner in K-FAC's eigenbasis; here the second-order method providing the basis is Shampoo.

The idealized equivalence is not the practical algorithm yet. In practice I use running averages of L and R, not dataset averages, so there is a small online-estimation mismatch. The bigger issue is that I do not recompute the eigenbasis every step. In Shampoo, because the preconditioner is the inverse root of L and R, freezing the basis freezes the entire rescaling — the λ_i, μ_j it divides by are stale until the next eigen-refresh. But in the equivalent "diagonal-in-the-eigenbasis" picture, the rescaling is the second moment of the rotated gradient, and that I can update every single step — it is just an elementwise EMA, no eigendecomposition required. The basis changes slowly (it only needs refreshing every f steps), but the diagonal preconditioner living in that basis can keep adapting continuously. Shampoo couldn't do this because it never separated "find the basis" from "rescale in the basis"; the equivalence does exactly that separation.

So here is the design the equivalence hands me. Keep Shampoo's eigenbasis Q_L, Q_R — refresh it only every f steps, paying the eigendecomposition rarely. But between refreshes, instead of reusing a frozen rescaling, run a full diagonal adaptive optimizer in that basis, updating its second moment every step on the rotated gradient. And since I am free to pick the diagonal optimizer, I will use Adam rather than Adafactor's rank-1 factorization — Adafactor only helps memory and the factorization is itself an approximation; with Adam I keep the true elementwise second moment V in the rotated space and gain a bit of generality (any diagonal optimizer would slot in here). I can run AdamW in the eigenbasis given by Shampoo's preconditioner: Adam in a rotated space.

Let me write the per-layer step out, because the order of operations carries the whole idea. For an m×n weight W I keep four things: the two Shampoo factors L (m×m), R (n×n); their eigenbases Q_L, Q_R; and Adam's two moments M, V (m×n). At step t with gradient G:

rotate the gradient into the basis, G' = Q_Lᵀ G Q_R; this is the coordinate system where the preconditioner is diagonal. Update the first moment as the same original-space momentum expressed in the current basis, M' = Q_Lᵀ(β₁ M + (1−β₁)G)Q_R. If I store M' for speed, the invariant is still the original-space momentum: before a basis change I project M' back, and after the new basis is ready I project it in again. Update the second moment directly in the rotated space, every step, V = β₂ V + (1−β₂) (G' ⊙ G') — this is the elementwise rescaling that Shampoo froze and I do not. Take the Adam step in the rotated frame, N' = M' / (√V + ε), and fold the bias corrections into the scalar step size as √(1−β₂ᵗ)/(1−β₁ᵗ). Rotate the result back to the original space, N = Q_L N' Q_Rᵀ, apply the adaptive update, and then apply decoupled weight decay. Then update the Shampoo factors with their own EMA, L = β₂ L + (1−β₂) G Gᵀ, R = β₂ R + (1−β₂) Gᵀ G; and only when t is a multiple of f, refresh the eigenbases from the updated factors.

The only new hyperparameter relative to AdamW is the preconditioning frequency f — everything else (lr, betas, eps, weight decay) is AdamW's, because in the rotated frame this is AdamW. When the basis is stale, V is still being updated every step in that basis, so the preconditioner keeps adapting to the current gradient's scale even between refreshes. Shampoo, whose adaptivity was the eigen-refresh, has nothing to fall back on when f is large; this does. The expensive operation has been demoted from "the source of adaptivity" to "an occasional basis recalibration."

Now the refresh itself — I should not pay a full eigendecomposition every f steps if I can avoid it, since I already have a good estimate of the eigenvectors from last time. The basis drifts slowly, so one step of power iteration warm-started from the previous Q is enough: form S = L Q_prev (one matmul), then orthonormalize by a QR decomposition, Q = QR(S). QR is faster than a full symmetric eigendecomposition in practice. I only need the genuine eigendecomposition on the very first refresh, to initialize Q; after that it is matmul-plus-QR. There is a small bookkeeping detail before the QR step: sort the old basis by the estimated eigenvalues diag(Qᵀ L Q), and reorder the corresponding axis of V the same way, so the diagonal variance table follows the coordinates it belongs to. The stored momentum goes through the original space across this refresh, while V only follows the axis permutation that preserves coordinate alignment.

A few boundary cases that the structure forces. For 1D parameters — norm gains, biases — there is no left/right matrix structure, no GGᵀ and GᵀG to speak of, so I just run the AdamW update with the identity projection; trying to maintain a preconditioner there is pointless overhead. For layers with a huge dimension — the vocabulary side of the embedding and the LM head, where m or n is the vocab size — maintaining and eigendecomposing an (vocab × vocab) factor is space- and time-prohibitive, so I fix that side's rotation to the identity (Q = I) and only precondition the other side. If I set both Q_L and Q_R to the identity for a 2D layer, the rotation is trivial and the update reduces exactly to Adam, which is a reassuring sanity check that this really is Adam in a rotated space. And the very first step I use only to initialize the preconditioner from the first gradient and I skip the parameter update, so I never project a gradient through a basis derived from that same gradient. Weight decay and bias correction are taken straight from AdamW, applied identically — decoupled decay shrinking the weights, and the √(1−β₂ᵗ)/(1−β₁ᵗ) bias-correction factor folded into the step size.

Let me write it the way it actually runs, as a per-layer optimizer with the four state matrices, the rotate-in / Adam / rotate-back step, and the power-iteration+QR refresh.

```python
import torch
from torch.optim.optimizer import Optimizer


def _eigh_basis(P):
    # Full eigendecomposition on first touch; return eigenvectors in descending order.
    P32 = P.float()
    eye = torch.eye(P32.shape[0], device=P32.device, dtype=P32.dtype)
    _, Q = torch.linalg.eigh(P32 + 1e-30 * eye)
    return torch.flip(Q, dims=[1]).to(P.dtype)


def _project_2d(X, QL, QR):
    if QL is not None:
        X = QL.t() @ X
    if QR is not None:
        X = X @ QR
    return X


def _project_back_2d(X, QL, QR):
    if QL is not None:
        X = QL @ X
    if QR is not None:
        X = X @ QR.t()
    return X


def _refresh_basis(P, Q_prev, V, dim):
    # Sort old approximate eigenvectors, reorder V on the matching axis, then do one QR step.
    if P is None or Q_prev is None:
        return None, V
    P32, Q32 = P.float(), Q_prev.float()
    est_eig = torch.diag(Q32.t() @ P32 @ Q32)
    idx = torch.argsort(est_eig, descending=True)
    V = V.index_select(dim, idx)
    Q32 = Q32[:, idx]
    Q32, _ = torch.linalg.qr(P32 @ Q32)
    return Q32.to(Q_prev.dtype), V


class SOAP(Optimizer):
    """AdamW run in the eigenbasis of Shampoo's preconditioner.

    Per 2D layer, small sides get Shampoo factors L/R and bases QL/QR; sides larger
    than max_precond_dim use the identity. Non-2D parameters run the same AdamW update
    with identity projection.
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=0.01, precondition_frequency=10,
                 max_precond_dim=10000, shampoo_beta=-1.0, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        max_precond_dim=max_precond_dim, shampoo_beta=shampoo_beta,
                        correct_bias=correct_bias)
        super().__init__(params, defaults)

    def _init_state(self, p, G, group):
        beta2 = group["betas"][1]
        shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else beta2
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)       # first moment in the current basis
        state["exp_avg_sq"] = torch.zeros_like(p)    # rotated-basis second moment
        state["use_precond"] = G.dim() == 2
        state["shampoo_beta"] = shampoo_beta
        if G.dim() != 2:
            return

        m, n = G.shape
        max_dim = group["max_precond_dim"]
        state["L"] = None if m > max_dim else torch.zeros(m, m, device=G.device, dtype=G.dtype)
        state["R"] = None if n > max_dim else torch.zeros(n, n, device=G.device, dtype=G.dtype)
        if state["L"] is not None:
            state["L"].mul_(shampoo_beta).add_(G @ G.t(), alpha=1 - shampoo_beta)
            state["QL"] = _eigh_basis(state["L"])
        else:
            state["QL"] = None
        if state["R"] is not None:
            state["R"].mul_(shampoo_beta).add_(G.t() @ G, alpha=1 - shampoo_beta)
            state["QR"] = _eigh_basis(state["R"])
        else:
            state["QR"] = None

    def _step_size(self, group, t):
        if not group["correct_bias"]:
            return group["lr"]
        beta1, beta2 = group["betas"]
        return group["lr"] * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

    def _apply_adamw_update(self, p, update, step_size, lr, wd):
        p.add_(update, alpha=-step_size)
        if wd > 0:
            p.add_(p, alpha=-lr * wd)                # decoupled decay, as in the implementation

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd, f = (group["lr"], group["eps"],
                              group["weight_decay"], group["precondition_frequency"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                state = self.state[p]

                if "step" not in state:               # initialize from this gradient, skip update
                    self._init_state(p, G, group)
                    continue

                M, V = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if state["use_precond"]:
                    QL, QR = state["QL"], state["QR"]
                    G_rot = _project_2d(G, QL, QR)
                    M.mul_(beta1).add_(G_rot, alpha=1 - beta1)
                    M_rot = M
                else:
                    G_rot = G
                    M.mul_(beta1).add_(G, alpha=1 - beta1)
                    M_rot = M

                V.mul_(beta2).addcmul_(G_rot, G_rot, value=1 - beta2)
                denom = V.sqrt().add_(eps)
                N_rot = M_rot / denom

                if state["use_precond"]:
                    N = _project_back_2d(N_rot, state["QL"], state["QR"])
                else:
                    N = N_rot
                self._apply_adamw_update(p, N, self._step_size(group, t), lr, wd)

                if state["use_precond"]:
                    sb = state["shampoo_beta"]
                    if state["L"] is not None:
                        state["L"].mul_(sb).add_(G @ G.t(), alpha=1 - sb)
                    if state["R"] is not None:
                        state["R"].mul_(sb).add_(G.t() @ G, alpha=1 - sb)
                    if t % f == 0:
                        M_orig = _project_back_2d(M, state["QL"], state["QR"])
                        state["QL"], V = _refresh_basis(state["L"], state["QL"], V, dim=0)
                        state["QR"], V = _refresh_basis(state["R"], state["QR"], V, dim=1)
                        state["exp_avg"] = _project_2d(M_orig, state["QL"], state["QR"])
                        state["exp_avg_sq"] = V
        return loss
```

The chain now fits together. AdamW preconditions diagonally and is blind to the row/column correlations of a weight matrix; full-matrix Adagrad would use them but is mn×mn; Shampoo approximates that by a Kronecker product L⊗R, but its L^{-1/2}, R^{-1/2} need an eigendecomposition I can only afford every f steps, and because its adaptivity is that eigen-refresh, it goes stale and degrades as f grows. Rotating into the eigenbasis of L and R, I find Shampoo with power 1/2 and the trace correction is exactly Adafactor run in that basis — the eigendecomposition was only computing the basis, and the actual preconditioning there is a cheap diagonal rescaling, namely the second moment of the rotated gradient. So I separate the two: refresh the basis rarely with warm-started power iteration plus QR, but run a full Adam in that basis and update its second moment every step, re-projecting the original-space momentum when the basis drifts. That gives me Shampoo's non-diagonal preconditioning with only the refresh frequency added over AdamW, with plain AdamW for 1D parameters and an identity rotation on huge dimensions.
