We want to train large models fast, and the single most powerful lever I know of is *preconditioning* the gradient — premultiplying it by a matrix that reshapes the loss geometry so descent stops crawling along anisotropic, correlated directions. In the online-learning world there is even a provably optimal choice for that matrix: full-matrix AdaGrad accumulates the outer-product sum $M_t = \sum_{s\le t} g_s g_s^\top$ and preconditions with $H_t = M_t^{1/2}$, giving the update $w_{t+1} = w_t - \eta\, H_t^{-1} g_t$ and a regret that scales like $\operatorname{tr}\big((\sum g g^\top)^{1/2}\big)$, the best attainable in this family. It captures *every* pairwise correlation between coordinates, not merely a per-coordinate rescaling. The reason no one uses it at scale is brutal: for a single weight matrix $W \in \mathbb{R}^{m\times n}$ the flattened gradient $g = \operatorname{vec}(G)$ has dimension $d = mn$, so $H_t$ is $mn \times mn$ — $m^2 n^2$ numbers to store and $O(m^3 n^3)$ work to take its inverse root. For $m = n = 1000$ that is a $10^6 \times 10^6$ matrix and $10^{18}$ operations per step. The escape everyone actually takes is to throw away the off-diagonal entirely: diagonal AdaGrad and Adam keep only $\operatorname{diag}(\sum g\odot g)$, one scalar per coordinate, $O(d)$ memory and a trivial inverse. They work well, but a diagonal preconditioner is axis-aligned — it can only stretch and shrink along the coordinate axes and is structurally blind to the very correlations that made full-matrix preconditioning powerful. Architecture-specific factored methods like K-FAC sit elsewhere on the trade-off but must know the network graph and the statistics of backpropagated gradients, are intricate to implement, and carry no convex-case regret guarantee. So I am stuck between a full matrix that is impossibly large and a diagonal that discards all cross-coordinate structure, and I want something in between that keeps real matrix structure but pays nowhere near $m^2 n^2$.

The observation that breaks the impasse is that the parameter is not intrinsically a flat vector — it is a *matrix* (or, in general, a tensor). Flattening is something I do only to write down $H_t$. So I propose Shampoo: instead of modeling correlations between every pair of the $mn$ entries, model only the geometry *within each axis*. For a matrix I keep one small left (row) matrix and one small right (column) matrix, each a second-moment statistic of the gradients contracted down to that axis,
$$L_t = \varepsilon I_m + \sum_{s\le t} G_s G_s^\top \in \mathbb{R}^{m\times m}, \qquad R_t = \varepsilon I_n + \sum_{s\le t} G_s^\top G_s \in \mathbb{R}^{n\times n},$$
with a small ridge $\varepsilon I$ keeping everything positive-definite. These are the two axis-wise contractions of $\sum g g^\top$: $G_t G_t^\top$ contracts away the column index to leave a row-by-row second moment, and $G_t^\top G_t$ contracts away the row index. They cost $m^2 + n^2$ memory and $O(m^3 + n^3)$ for any root, versus $m^2 n^2$ and $O(m^3 n^3)$ — completely tractable. I then precondition the gradient on *both* sides and step,
$$W_{t+1} = W_t - \eta\, L_t^{-1/4}\, G_t\, R_t^{-1/4}.$$
What makes this exactly a full-matrix method in disguise is the vec–Kronecker identity $(L \otimes R^\top)\operatorname{vec}(G) = \operatorname{vec}(L G R)$; checking it on a rank-one $G = uv^\top$, the left side is $(Lu)\otimes(R^\top v)$ by the mixed-product rule and the right side is $\operatorname{vec}\big((Lu)(R^\top v)^\top\big) = (Lu)\otimes(R^\top v)$, and linearity does the rest. Since $R_t$ is symmetric, $(R_t^{-b})^\top = R_t^{-b}$, so two-sided multiplication $L_t^{-a} G_t R_t^{-b}$ is precisely the flattened mirror-descent step $w_{t+1} = w_t - \eta\, H_t^{-1} g_t$ with the implicit Kronecker preconditioner $H_t = L_t^{a} \otimes R_t^{b}$, never formed explicitly.

The entire method hinges on the exponent, and the naive guess is wrong. One is tempted to say: $L_t$ and $R_t$ are the row and column covariances, full AdaGrad takes the square root of a covariance, so use $a = b = 1/2$. To see why that is too aggressive I need to know what the flat covariance $\sum g g^\top$ is actually bounded by in terms of $L_t$ and $R_t$. Take the SVD $G = \sum_{i=1}^r \sigma_i u_i v_i^\top$, so $g = \operatorname{vec}(G) = \sum_i \sigma_i (u_i \otimes v_i)$ and $g g^\top$ is the outer product of a sum, whose cross terms couple the two axes and resist any clean Kronecker form. I decouple them with the inequality $\big(\sum_i w_i\big)\big(\sum_i w_i\big)^\top \preceq r \sum_i w_i w_i^\top$ — which follows from $(\sum_i \alpha_i)^2 \le r \sum_i \alpha_i^2$ (Cauchy–Schwarz) applied to $\alpha_i = x^\top w_i$ — paying a factor equal to the rank $r$. With $w_i = \sigma_i(u_i\otimes v_i)$ this gives $g g^\top \preceq r \sum_i \sigma_i^2 (u_i u_i^\top)\otimes(v_i v_i^\top)$, a clean sum of Kronecker products. Bounding one factor by the identity — $v_i v_i^\top \preceq I_n$ because the $v_i$ are orthonormal — collapses it to $\tfrac1r\, g g^\top \preceq (G G^\top)\otimes I_n$, and symmetrically $\tfrac1r\, g g^\top \preceq I_m \otimes (G^\top G)$. Summing over $t$ and folding in the ridge, the *same* matrix $\varepsilon I_{mn} + \tfrac1r\sum_t g_t g_t^\top$ lies below both $L_T \otimes I_n$ and $I_m \otimes R_T$. These two upper bounds commute (each acts on a different factor of the tensor product, and $(L_T\otimes I_n)(I_m\otimes R_T) = L_T \otimes R_T$ either way), and for commuting PSD matrices the geometric mean is operator-monotone: $X \preceq Y_1$ and $X \preceq Y_2$ with $Y_1, Y_2$ commuting gives $X \preceq Y_1^{1/2} Y_2^{1/2}$. Hence
$$\varepsilon I_{mn} + \tfrac1r \sum_t g_t g_t^\top \;\preceq\; (I_m \otimes R_T)^{1/2}(L_T \otimes I_n)^{1/2} = L_T^{1/2} \otimes R_T^{1/2}.$$
Read carefully, this kills the naive guess: $L_T^{1/2}\otimes R_T^{1/2}$ is the analogue of the *un-rooted* covariance $\sum g g^\top$, not of its square root — the two halves there are the price of splitting one covariance across two axes (each $L_T, R_T$ already grows like $t$, so $L_T^{1/2}\otimes R_T^{1/2}$ grows like $t^{1/2}\cdot t^{1/2} = t$, matching $\sum g g^\top$). Full AdaGrad preconditions with the *square root* of the covariance, so I take one more root: $H_t = (L_t^{1/2}\otimes R_t^{1/2})^{1/2} = L_t^{1/4}\otimes R_t^{1/4}$, using $(A\otimes B)^s = A^s\otimes B^s$. The exponent is $1/4 = \tfrac12\cdot\tfrac12$: one half from splitting the covariance across the two axes, one half from the AdaGrad root. An independent check makes the $1/4$ feel inevitable — since $L_t, R_t \sim t$, the effective step scale is $\|L_t^{-1/4}\cdot R_t^{-1/4}\| \sim t^{-1/4}\cdot t^{-1/4} = t^{-1/2}$, exactly the canonical $O(1/\sqrt t)$ stochastic step decay; the naive $1/2$ would have given a far-too-fast $t^{-1}$.

The same lemma delivers a guarantee, not just an analogy. The $H_t = L_t^{1/4}\otimes R_t^{1/4}$ are monotonically increasing (each step adds a PSD term and the Kronecker product preserves order), so the standard adaptive-mirror-descent bound's first term telescopes: with $D = \max_t \|W_t - W^*\|_F$ and $x^\top M x \le \|x\|^2\operatorname{tr}(M)$, it is at most $\tfrac{D^2}{2\eta}\operatorname{tr}(H_T)$. For the second term I compare against the ideal full preconditioner $\hat H_t = (r\varepsilon I + \sum_{s\le t} g_s g_s^\top)^{1/2}$; operator-monotonicity of $x\mapsto x^{1/2}$ applied to $r\varepsilon I + \sum g g^\top \preceq r\,H_t^2$ gives $\hat H_t \preceq \sqrt r\, H_t$, hence $H_t^{-1}\preceq \sqrt r\,\hat H_t^{-1}$ and $(\|g\|^*_{H_t})^2 \le \sqrt r\,(\|g\|^*_{\hat H_t})^2$. The adaptive-regularization lemma with potential $\Phi(H) = \operatorname{tr}(H) + r\varepsilon\operatorname{tr}(H^{-1})$ — whose minimizer is exactly $\hat H_t$, since $\nabla_H\operatorname{tr}(A H^{-1} + H) = I - A H^{-2}$ vanishes at $H = A^{1/2}$ — gives $\sum_t (\|g_t\|^*_{\hat H_t})^2 \le 2\operatorname{tr}(\hat H_T)$, so $\sum_t (\|g_t\|^*_{H_t})^2 \le 2\sqrt r\operatorname{tr}(\hat H_T) \le 2r\operatorname{tr}(H_T)$. Combining, $R_T \le \big(\tfrac{D^2}{2\eta} + \eta r\big)\operatorname{tr}(H_T)$, and $\eta = D/\sqrt{2r}$ yields
$$R_T \le \sqrt{2r}\, D\, \operatorname{tr}(L_T^{1/4})\operatorname{tr}(R_T^{1/4}),$$
using $\operatorname{tr}(A\otimes B) = \operatorname{tr}(A)\operatorname{tr}(B)$. Under $\|G_t\|_2 \le 1$ we have $L_T \preceq T I_m$ and $R_T \preceq T I_n$, so each trace is $O(T^{1/4})$ and their product is $O(\sqrt T)$ — optimal for online/stochastic convex optimization. The $1/4$ is precisely what makes the two traces each scale as $T^{1/4}$ so that they multiply to the optimal $\sqrt T$.

Everything lifts to tensors with one preconditioner per *mode*. For an order-$k$ tensor with dimensions $n_1\times\cdots\times n_k$, matricize on axis $i$ (that axis as rows, the rest flattened into columns) and accumulate the mode-$i$ contraction $G^{(i)} = \operatorname{mat}_i(G)\operatorname{mat}_i(G)^\top$, so $H^i_t = \varepsilon I_{n_i} + \sum_{s\le t} G_s^{(i)}$; for $k=2$ this is exactly $L$ and $R$. The tensor vec–Kronecker identity $\big(\bigotimes_i M_i\big)\operatorname{vec}(G) = \operatorname{vec}(G\times_1 M_1\cdots\times_k M_k)$ and the generalized lemma $\varepsilon I_n + \sum_t g_t g_t^\top \preceq r\bigotimes_i (H^i_T)^{1/k}$ (with $r = (\prod_i r_i)^{1/k}$) show that $\bigotimes_i (H^i_T)^{1/k}$ plays the covariance role, so the preconditioner is $H_t = \big(\bigotimes_i H^i_t\big)^{1/(2k)}$ and along each mode I apply $(H^i_t)^{-1/(2k)}$ via the tensor-matrix product $\times_i$; the exponent $-\tfrac1{2k} = \tfrac1k\cdot\tfrac12$ is the $k$-way axis split times the AdaGrad root, recovering $-1/4$ at $k=2$, and the regret is again $O(\sqrt T)$. The inverse $p$-th roots are computed from the eigen/SVD of the symmetric PSD statistic — $(H^i)^{\alpha} = \sum_j \lambda_j^\alpha u_j u_j^\top$, with the ridge guaranteeing $\lambda_j > 0$ — and since this $O(n_i^3)$ root is the only nontrivial cost I refresh it only every 20–100 steps and reuse it in between. I add a momentum-style running average $\bar G_t = \alpha \bar G_{t-1} + (1-\alpha) G_t$ with $\alpha\approx 0.9$, treat each parameter tensor independently (a block-diagonal preconditioner across tensors, capturing all intra-tensor correlations while keeping the optimizer fully architecture-oblivious — it needs only the tensor shapes), and fall back to a diagonal preconditioner $H^i_t = \sum \operatorname{diag}(G^{(i)})$ on any axis too large to store or factorize, for which the same analysis holds with the entrywise $D_\infty$ in place of the Frobenius $D$.

```python
import torch
from torch.optim.optimizer import Optimizer


def _matrix_power(matrix, power):
    # Inverse p-th root of a symmetric PSD matrix via SVD:
    #   H = U diag(s) V^T  ->  H^power = U diag(s^power) V^T.
    u, s, v = torch.svd(matrix)
    return u @ s.pow(power).diag() @ v.t()


class Shampoo(Optimizer):
    """Per-axis Kronecker-factored preconditioning for tensor parameters.

    For a parameter of order k, keep one statistics matrix H^i per axis i
    (the mode-i gradient contraction summed over steps) and precondition the
    i-th mode of the gradient by (H^i)^{-1/(2k)}. For a matrix (k=2) this is
    the two-sided update W <- W - lr * L^{-1/4} G R^{-1/4}.
    """

    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                 epsilon=1e-4, update_freq=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()          # tensor order k
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # H^i = eps * I  and a cache for (H^i)^{-1/(2k)}.
                        state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim))
                        state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)
                if weight_decay > 0:
                    grad.add_(p.data, alpha=weight_decay)

                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # Matricize on axis dim_id: this axis becomes the rows.
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    # H^i <- H^i + mat_i(g) mat_i(g)^T.
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / (2 * order)))

                    if dim_id == order - 1:
                        grad = grad_t @ inv_precond          # right-multiply last mode
                        grad = grad.view(original_size)
                    else:
                        grad = inv_precond @ grad            # left-multiply other modes
                        grad = grad.view(transposed_size)

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])

        return loss
```
