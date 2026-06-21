The thing that bugs me about the optimizer everyone uses to train large models is that AdamW is structurally blind. Per coordinate it keeps an EMA of the gradient and an EMA of the squared gradient and divides one by the square root of the other, which is a diagonal preconditioner: every coordinate gets its own learning rate, and nothing about how coordinates relate to one another is ever used. But a hidden-layer weight $W \in \mathbb{R}^{m\times n}$ is not a bag of $mn$ independent scalars — it is a linear operator, its gradient $G$ has correlated rows and correlated columns, and a diagonal method throws all of that away. The object that would actually use it is the full second-moment matrix. Full-matrix Adagrad writes this down honestly: vectorize $g = \mathrm{vec}(G) \in \mathbb{R}^{mn}$, accumulate $H = \sum g g^\top$, and step $w \leftarrow w - \eta\, H^{-1/2} g$, where $H^{-1/2}$ whitens the gradient using every pairwise correlation among the $mn$ coordinates. It is also hopeless: $H$ is $mn\times mn$, a million-by-million matrix for a modest $m=n=1024$, whose inverse square root I would have to keep current. Adam's escape is to keep only the diagonal of $H$ — which is exactly why it is cheap and exactly why it is blind.

Shampoo is the interesting middle. Instead of the full $H$ or its diagonal, it approximates $H$ by a Kronecker product of two small matrices — a left factor $L = \sum G G^\top \in \mathbb{R}^{m\times m}$ for row correlations and a right factor $R = \sum G^\top G \in \mathbb{R}^{n\times n}$ for column correlations — and steps $W \leftarrow W - \eta\, L^{-1/2} G R^{-1/2}$, which vectorized is preconditioning by $(L\otimes R)^{-1/2}$ (using the practical power $1/2$ that matches the optimal Kronecker approximation of the Adagrad preconditioner, plus a $\mathrm{Trace}(L)^{1/2}$ scalar normalization). The $mn\times mn$ preconditioner is factored into an $m\times m$ and an $n\times n$ piece — tractable — while still seeing both sides' correlations. But the cost bites precisely where I care. To form $L^{-1/2}$ and $R^{-1/2}$ I need an eigendecomposition or inverse matrix root of $L$ and $R$, cubic in the side length, so nobody does it every step; they do it every $f$ steps and reuse the stale roots in between. The diagnostic that bothers me is that between refreshes Shampoo's preconditioner is frozen — its only adaptivity *is* the periodic recomputation of $L^{-1/2}, R^{-1/2}$ — so the lever I would pull to make it affordable, pushing $f$ up, is the very thing that makes it stale and degrades it. I want Shampoo's non-diagonal preconditioning without paying the eigen-refresh every step and without losing adaptivity when I refresh rarely.

So I looked harder at what Shampoo actually computes, suspecting it is a diagonal method in disguise. Take idealized power-$1/2$ Shampoo with dataset-average factors and rotate the gradient into the joint eigenbasis: let $Q_L$ hold the eigenvectors of $L$ (eigenvalues $\lambda_1,\dots,\lambda_m$), $Q_R$ those of $R$ (eigenvalues $\mu_1,\dots,\mu_n$), and define $G' = Q_L^\top G Q_R$. In this frame the preconditioner $L^{-1/2}\otimes R^{-1/2}$ with the trace correction scales coordinate $(i,j)$ of the rotated gradient by $(\lambda_i \mu_j / \sum_i \lambda_i)^{-1/2}$ — a pure diagonal rescaling. And that rescaling is suspiciously close to Adafactor's, which scales $(i,j)$ by $(A_i C_j / \sum A)^{-1/2}$ with $A_i$ the row-marginal and $C_j$ the column-marginal of the squared rotated gradient. They coincide if $A_i = \lambda_i$ and $C_j = \mu_j$, so I check $A_i$ directly. With $u_i$ the $i$-th eigenvector of $L$ and $\{v_j\}$ the orthonormal eigenvectors of $R$,
$$A_i = \mathbb{E}_B\Big[\textstyle\sum_j (u_i^\top G_B v_j)^2\Big] = \mathbb{E}_B\big[\, u_i^\top G_B G_B^\top u_i\,\big] = u_i^\top\, \mathbb{E}_B[G_B G_B^\top]\, u_i = u_i^\top L\, u_i = \lambda_i,$$
using that the $v_j$ form an orthonormal basis so $\sum_j (u_i^\top G_B v_j)^2 = \|u_i^\top G_B\|^2$, and that $u_i$ is the eigenvector of $L = \mathbb{E}[GG^\top]$. The same argument gives $C_j = \mu_j$. So power-$1/2$ Shampoo (with the trace correction and dataset-average factors) is *exactly* Adafactor run in the eigenbasis of its own preconditioner. The expensive eigendecomposition was never the adaptivity — it only computed the basis; the actual preconditioning in that basis is a cheap elementwise rescaling, namely the second moment of the rotated gradient. This is the same shape as E-KFAC, which runs a diagonal preconditioner in K-FAC's eigenbasis; here the second-order method supplying the basis is Shampoo.

That separation is the whole lever, and it is what I propose to exploit. I call the method SOAP — ShampoO with Adam in the Preconditioner's eigenbasis. Keep Shampoo's eigenbasis $Q_L, Q_R$, but refresh it only every $f$ steps, paying the eigendecomposition rarely; between refreshes, instead of reusing a frozen rescaling, run a full diagonal adaptive optimizer in that basis and update its second moment every single step on the rotated gradient — just an elementwise EMA, no eigendecomposition needed. The basis drifts slowly and only needs occasional recalibration, but the diagonal preconditioner living in it keeps adapting continuously. Shampoo couldn't do this because it never separated "find the basis" from "rescale in the basis"; the equivalence does exactly that. And since I am free to pick the diagonal optimizer, I use Adam rather than Adafactor's rank-1 factorization — Adafactor's factorization only saves memory and is itself an approximation, whereas Adam keeps the true elementwise second moment $V$ in the rotated space and generalizes (any diagonal optimizer would slot in). SOAP is therefore AdamW run in the eigenbasis given by Shampoo's preconditioner.

The per-layer step is where the order of operations carries the idea. For an $m\times n$ weight I keep four things: the Shampoo factors $L$ ($m\times m$) and $R$ ($n\times n$); their eigenbases $Q_L, Q_R$; and Adam's two moments $M, V$ ($m\times n$, stored in the rotated basis). At step $t$ with gradient $G$, I first rotate into the basis, $G' = Q_L^\top G Q_R$, the coordinate system where the preconditioner is diagonal; update the first moment $M \leftarrow \beta_1 M + (1-\beta_1) G'$ (the original-space momentum expressed in the current basis); update the second moment in the rotated space every step, $V \leftarrow \beta_2 V + (1-\beta_2)(G' \odot G')$ — this is precisely the rescaling Shampoo froze and I do not; take the Adam step in the rotated frame, $N' = M / (\sqrt{V} + \epsilon)$; fold the bias corrections into the scalar step size as $c_t = \sqrt{1-\beta_2^t}/(1-\beta_1^t)$; rotate back, $N = Q_L N' Q_R^\top$; apply $W \leftarrow W - \eta\, c_t\, N$ and then decoupled weight decay $W \leftarrow W - \eta\lambda W$ outside the Adam moments. Finally update the Shampoo factors with their own EMA, $L \leftarrow \beta_2 L + (1-\beta_2) G G^\top$, $R \leftarrow \beta_2 R + (1-\beta_2) G^\top G$, and only when $t$ is a multiple of $f$ refresh the bases. The only new hyperparameter over AdamW is the preconditioning frequency $f$; lr, betas, eps, and weight decay are AdamW's, because in the rotated frame this *is* AdamW. When the basis is stale, $V$ is still updated every step in that basis, so the preconditioner keeps adapting to the current gradient's scale between refreshes — Shampoo, whose adaptivity was the eigen-refresh, has nothing to fall back on when $f$ is large; this does. The expensive operation has been demoted from the source of adaptivity to an occasional basis recalibration.

The refresh itself I make cheap. Since the basis drifts slowly and I already have a good estimate from last time, one step of power iteration warm-started from the previous $Q$ suffices: form $S = L\, Q_\text{prev}$ (one matmul) and orthonormalize by QR, $Q = \mathrm{QR}(S)$, which is faster than a full symmetric eigendecomposition; the genuine eigendecomposition is only needed on the very first refresh to initialize $Q$. One bookkeeping detail precedes the QR: sort the old basis by the estimated eigenvalues $\mathrm{diag}(Q^\top L Q)$ and reorder the matching axis of $V$ the same way, so the diagonal variance table follows the coordinates it belongs to; the stored momentum is carried through the original space across the refresh, while $V$ only follows that axis permutation. The structure also forces a few boundary cases. For 1D parameters — norm gains, biases — there is no left/right matrix structure, so I just run AdamW with identity projection; a preconditioner there is pointless overhead. For huge (vocabulary-sized) dimensions, where an $m\times m$ or $n\times n$ factor is space- and time-prohibitive, I fix that side's rotation to the identity and only precondition the other side. Setting *both* rotations to identity for a 2D layer reduces the update exactly to Adam — a reassuring sanity check that this really is Adam in a rotated space. And the very first step is used only to initialize the factors and basis; the parameter update is skipped, so a gradient is never projected through a basis derived from that same gradient. Weight decay and bias correction are taken straight from AdamW and applied identically.

```python
import torch
from torch.optim.optimizer import Optimizer


def _eigh_basis(P):
    P32 = P.float()
    eye = torch.eye(P32.shape[0], device=P32.device, dtype=P32.dtype)
    _, Q = torch.linalg.eigh(P32 + 1e-30 * eye)
    return torch.flip(Q, dims=[1]).to(P.dtype)     # descending eigenvalue order


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
    """AdamW run in Shampoo's eigenbasis; identity projection gives plain AdamW."""

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
        state["exp_avg"] = torch.zeros_like(p)       # stored in the current basis
        state["exp_avg_sq"] = torch.zeros_like(p)
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
            p.add_(p, alpha=-lr * wd)

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

                if "step" not in state:
                    self._init_state(p, G, group)
                    continue

                M, V = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if state["use_precond"]:
                    G_rot = _project_2d(G, state["QL"], state["QR"])
                    M.mul_(beta1).add_(G_rot, alpha=1 - beta1)
                    M_rot = M
                else:
                    G_rot = G
                    M.mul_(beta1).add_(G, alpha=1 - beta1)
                    M_rot = M
                V.mul_(beta2).addcmul_(G_rot, G_rot, value=1 - beta2)

                denom = V.sqrt().add_(eps)
                N_rot = M_rot / denom
                N = _project_back_2d(N_rot, state["QL"], state["QR"]) if state["use_precond"] else N_rot
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
