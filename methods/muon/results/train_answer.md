The cost of training a large language model is dominated by the optimizer's work over trillions of tokens, and the overwhelming bulk of a transformer's trainable parameters live in 2D hidden weight matrices — the attention and feed-forward projections. AdamW is the near-universal default for these, but its treatment of them is structurally wrong. With its moving averages switched off and $\epsilon$ ignored, Adam's per-coordinate update collapses to $g/\sqrt{g^2} = \mathrm{sign}(g)$: it normalizes each scalar entry in isolation. A hidden weight matrix, though, is not a bag of independent scalars; it is a linear operator on the layer's input/hidden space, and what we care about is how the update changes that operator's action on vectors. An entrywise rule is blind to this. The diagnostic that makes the worry concrete is the spectrum of the updates themselves: for transformer 2D weights, the SGD-momentum and Adam update matrices are nearly low-rank — high condition number, a few singular directions carrying almost all the magnitude, and a long tail of small-singular-value directions, which can still matter for learning, receiving almost no step. Shampoo does take the matrix structure seriously, preconditioning with $L^{-1/4} G R^{-1/4}$ where $L = \sum GG^\top$ and $R = \sum G^\top G$, but forming and inverse-fourth-rooting the two preconditioners is $O(A^3+B^3)$ time and $O(A^2+B^2)$ memory per matrix and demands higher precision — far too expensive to run every step at billion-parameter scale. What we need is an update that respects the matrix's operator structure, runs every step on GPUs in low precision using only cheap primitives, and keeps AdamW's practical conveniences: a stable update magnitude and transferable learning-rate and weight-decay settings.

I propose Muon, a matrix-aware optimizer for the hidden 2D weights. The way to see what it should do is to write the optimizer step as a small local problem. To first order the loss changes like $\langle G, \Delta W\rangle$ where $G$ is the gradient or a momentum estimate; I do not trust that linear model arbitrarily far, so I penalize the step, $\min_{\Delta}\, \langle G,\Delta\rangle + \tfrac{\lambda}{2}\|\Delta\|^2$. Writing $\Delta = c\,T$ with $c\ge 0$ and $\|T\|=1$ separates direction from scale: the optimal direction is the unit object most aligned with $G$ (with a minus sign for descent) and the scale is the dual norm of $G$ over $\lambda$. So everything reduces to the choice of norm on the step. If I pick the flattened infinity norm, the maximizer of $\langle G,T\rangle$ subject to $|T_{ij}|\le 1$ is $T_{ij}=\mathrm{sign}(G_{ij})$ and the dual norm is the entrywise $\ell_1$ norm — exactly Adam's coordinatewise sign update, controlled by its largest entry rather than by its action as an operator. For a dense hidden linear map the natural geometry on both sides is (RMS-scaled) Euclidean, so the induced operator norm is the spectral norm, and I should instead solve $\arg\max_{\|T\|_2\le 1}\langle G,T\rangle$. With $G = U\Sigma V^\top = \sum_i \sigma_i u_i v_i^\top$, we have $\langle G,T\rangle = \sum_i \sigma_i\, u_i^\top T v_i \le \sum_i \sigma_i$ whenever $\|T\|_2\le 1$, and this bound is attained by

$$T = U V^\top,$$

so the dual norm is the nuclear norm and the steepest unit-spectral-norm direction is the polar factor $UV^\top$. This is the equalization I was after: the update keeps the singular vectors of the momentum, so it still follows the directions the gradient history suggests, but replaces every singular value by one, so the dominant directions no longer swallow the whole step and the small-but-useful ones are lifted to comparable size. The same object arrives by a second route — among semi-orthogonal matrices, minimizing $\|O-G\|_F^2$ is the same as maximizing $\langle O,G\rangle$ since $\|O\|_F^2$ is fixed, giving $O=UV^\top$ for full-rank $G$ — and by a third: accumulation-free Shampoo with $L=GG^\top$, $R=G^\top G$ computes $(GG^\top)^{-1/4}G(G^\top G)^{-1/4} = U\Sigma^{-1/2}\Sigma\Sigma^{-1/2}V^\top = UV^\top$. So the target is well-motivated; the only problem is that an explicit SVD or inverse roots are too slow and too precision-fragile to run on every hidden matrix every step.

The cheap GPU primitive is matrix multiplication, especially in bfloat16, and odd matrix polynomials act on singular values while leaving singular vectors untouched: if $X = U S V^\top$ then $XX^\top X = U S^3 V^\top$, so any odd scalar polynomial applied through this form reshapes $S$ alone. The clean Newton–Schulz cubic $X_{k+1} = 1.5\,X_k - 0.5\,X_k X_k^\top X_k$ applies $f(s)=1.5s-0.5s^3$ to each singular value, whose fixed point at $s=1$ attracts every positive singular value in the basin below $\sqrt{3}$; dividing $X$ by its Frobenius norm first guarantees $\sigma_{\max}\le 1$, safely inside the basin, and a positive rescaling never changes the polar direction. The cubic, though, is slow near zero — exactly where the many tiny singular values of these updates sit — and the convergence speed there is just the slope of the scalar map at the origin. A quintic $g(s)=as+bs^3+cs^5$ buys a second free parameter. Insisting on exact convergence to one everywhere would force a small slope at zero, but exact orthogonality is not the practical requirement — roughly equalizing the singular values is — so I choose coefficients with a large slope at the origin and accept an output like $U S' V^\top$ with $S'$ clustered around one rather than exactly equal to it. The tuned coefficients are $(a,b,c)=(3.4445,\,-4.7750,\,2.0315)$, and five bfloat16 iterations suffice.

That leaves a scale issue, because a semi-orthogonal update does not have the same entry RMS for every shape. For an orthogonalized update of shape $[A,B]$ and rank $r$, $O = U_{:,:r}V_{:r,:}^\top$, expanding $\mathrm{RMS}(O)^2 = \frac{1}{AB}\sum_{ij}(\sum_k U_{ik}V_{kj})^2$ and using orthonormality to kill the cross terms gives

$$\mathrm{RMS}(O) = \sqrt{\tfrac{r}{AB}},$$

which in the common full-rank case $r=\min(A,B)$ becomes $1/\sqrt{\max(A,B)}$. So a bare orthogonalized update under-steps on matrices with a larger long side and a split parameter over-steps; multiplying by $\sqrt{\max(A,B)}$ cancels this full-rank shape dependence, and a further factor of $0.2$ brings the update RMS into AdamW's empirical range so one learning rate and one weight decay can be shared across both kinds of parameter. Weight decay must be decoupled — folding it into the gradient would push it through the orthogonalizer and change its meaning — so I use the AdamW-style form $W \leftarrow (1-\eta\lambda)W - \eta\,\text{update}$, which also keeps the weight RMS and layer-output RMS from drifting upward at long scale, where bfloat16 precision becomes a real constraint. Muon is applied only where the geometry matches the derivation: hidden 2D linear weights (and convolutional filters once flattened to matrices). Embeddings are 2D but are one-hot lookups, not dense operators; the LM head is 2D but empirically does better on the usual adaptive rule; gains and biases have no matrix singular geometry. All of those stay on AdamW. Sharding changes the implementation but not the math — a shard's singular directions are not the full matrix's — so in a ZeRO-1 setup the local shard updates its momentum, the momentum shards are gathered into the full matrix, Newton–Schulz runs on the full matrix in bfloat16, and each rank keeps only its local slice of the update, all while storing a single momentum buffer instead of AdamW's two. So the complete method is: smooth the hidden-matrix gradient with (Nesterov) momentum, replace that matrix by its approximate polar factor via the tuned Newton–Schulz quintic, scale by $0.2\sqrt{\max(A,B)}$ to remove the shape dependence and match AdamW's update RMS, apply decoupled weight decay, and leave embeddings, heads, gains, and biases to AdamW.

```python
import math
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate the polar factor / zeroth power of G with bfloat16 matmuls."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """Keller-style EMA momentum plus Newton-Schulz orthogonalization."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    return zeropower_via_newtonschulz5(update, steps=ns_steps)


def adamw_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    m_hat = exp_avg / (1 - betas[0] ** step)
    v_hat = exp_avg_sq / (1 - betas[1] ** step)
    return m_hat / (v_hat.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for hidden matrices; AdamW for embeddings, heads, gains, and biases."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 1e-3)
                group["momentum"] = group.get("momentum", 0.95)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                group["nesterov"] = group.get("nesterov", True)
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.1)
        super().__init__(param_groups, dict())

    @staticmethod
    def report_adjusted_lr(lr, shape):
        A, B = shape[:2]
        return lr * 0.2 * math.sqrt(max(A, B))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    g = p.grad.view(p.grad.size(0), -1) if p.grad.ndim > 2 else p.grad
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    update = muon_update(
                        g,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    p.mul_(1 - lr * wd)
                    p.add_(update.reshape(p.shape), alpha=-self.report_adjusted_lr(lr, p.shape))
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = adamw_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
        return loss
```
