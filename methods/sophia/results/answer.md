# Sophia: Second-order Clipped Stochastic Optimization

## Problem

Reach the same language-model pre-training loss in fewer optimizer steps while keeping average step cost and memory close to AdamW. The obstacle is anisotropic, non-convex curvature: Newton-style scaling is attractive, but full or frequently refreshed curvature is too expensive and unsafe.

## Method

Maintain a gradient EMA \(m_t\) and an infrequently refreshed diagonal-curvature EMA \(h_t\):

- \(m_t=\beta_1m_{t-1}+(1-\beta_1)g_t\)
- every \(k\) steps, \(h_t=\beta_2h_{t-k}+(1-\beta_2)\hat h_t\); otherwise \(h_t=h_{t-1}\)
- apply decoupled weight decay, then

\[
\theta_{t+1}=\theta_t-\eta_t\,\operatorname{clip}\!\left(\frac{m_t}{\max\{\gamma h_t,\epsilon\}},1\right).
\]

Here clipping is elementwise. The identity

\[
\eta\,\operatorname{clip}(m/\max\{\gamma h,\epsilon\},1)
=\frac{\eta}{\gamma}\operatorname{clip}(m/\max\{h,\epsilon/\gamma\},\gamma)
\]

separates saturated update size from clip fraction: \(\eta\) sets the size of clipped coordinates, while \(\gamma\) controls how many coordinates escape clipping. If \(h_i\le 0\) or is tiny, the positive floor makes the ratio have the sign of \(m_i\), and clipping falls back to a bounded momentum-SignGD coordinate step.

## Curvature Estimators

**Sophia-H / Hutchinson.** Draw \(u\sim\mathcal N(0,I_d)\) and return

\[
\hat h=u\odot(\nabla^2L(\theta)u).
\]

It is unbiased for \(\operatorname{diag}(\nabla^2L)\) because \(E[u_i(Hu)_i]=\sum_jH_{ij}E[u_iu_j]=H_{ii}\). It needs a Hessian-vector product, \(\nabla_\theta\langle\nabla_\theta L,u\rangle\), and may produce negative coordinates.

**Sophia-G / Gauss-Newton-Bartlett.** For \(\ell(\theta,(x,y))=\operatorname{ce}(f(\theta,x),y)\),

\[
\nabla_\theta^2\ell=J_\theta f\,S\,J_\theta f^\top+J_{\theta\theta}f[q],
\]

and the method drops the second term, estimating the diagonal of the PSD Gauss-Newton term. For softmax cross-entropy, \(S=\operatorname{diag}(p)-pp^\top\) depends only on logits. Bartlett's identities give

\[
\operatorname{diag}(J_\theta f\,S\,J_\theta f^\top)
=E_{\hat y\sim\operatorname{Cat}(p)}
[\nabla_\theta\operatorname{ce}(f,\hat y)\odot\nabla_\theta\operatorname{ce}(f,\hat y)].
\]

For a minibatch loss averaged over \(B\) sampled labels,

\[
E[B\,\nabla\widehat L\odot\nabla\widehat L]
=E\left[\frac1B\sum_b\nabla\operatorname{ce}_b\odot\nabla\operatorname{ce}_b\right],
\]

because the sampled-label score has zero mean and independent cross terms vanish. Thus Sophia-G needs one extra ordinary backward pass on model-sampled labels every \(k\) steps.

## Convex Check

The simplified theorem analyzes exact full-Hessian clipped Newton, not the stochastic diagonal implementation:

\[
\theta_+=\theta-\eta V^\top\operatorname{clip}(V(\nabla^2L)^{-1}\nabla L,\rho).
\]

Under strict convexity and the multiplicative Hessian-continuity condition, if \(\eta\rho\le R/\sqrt d\),

\[
L(\theta_+)-L(\theta)
\le-(\eta-\eta^2)\sum_i
\min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\}.
\]

The clipped branch gives safe sign-like progress; the unclipped branch gives Newton progress. With \(\eta=1/2\) and \(\rho=R/(2\sqrt d)\),

\[
T\lesssim d\frac{L(\theta_0)-\min L}{\mu R^2}
+\log\frac{\mu R^2}{32d\epsilon},
\]

which has no condition-number or largest-eigenvalue dependence in that idealized model. The SignGD proxy lower bound on \(L_{\mu,\beta}=\frac{\mu}{2}\theta_1^2+\frac{\beta}{2}\theta_2^2\) scales as \(\Omega(\sqrt{\beta/\mu})\).

## Reference-Code Faithful Sophia-G

This is the non-capturable SophiaG path, reduced to the essential PyTorch logic. It stores the unscaled sampled-label gradient square in `hessian`; the `bs` argument supplies the \(B\) factor from the GNB estimator.

```python
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1):
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, p):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = torch.tensor(0.)
            state["exp_avg"] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
            state["hessian"] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
        elif "hessian" not in state:
            state["hessian"] = torch.zeros_like(
                p, memory_format=torch.preserve_format)
        return state

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_state(p)
                state["hessian"].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, _ = group["betas"]
            lr = group["lr"]
            rho = group["rho"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SophiaG does not support sparse gradients")

                state = self._init_state(p)
                state["step"] += 1
                m = state["exp_avg"]
                h = state["hessian"]

                p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                ratio = (m.abs() / (rho * bs * h + 1e-15)).clamp(max=1.0)
                p.addcmul_(m.sign(), ratio, value=-lr)
        return loss


optimizer = SophiaG(model.parameters(), lr=peak_lr, betas=(0.965, 0.99),
                    rho=0.05, weight_decay=0.2)
k = 10

for it in range(max_iters):
    X, Y = get_batch("train")
    logits, loss = model(X, Y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(bs=total_bs * block_size)
    optimizer.zero_grad(set_to_none=True)

    if it % k == k - 1:
        Xh, _ = get_batch("train")
        logits, _ = model(Xh, None)
        y_sample = torch.distributions.Categorical(logits=logits).sample()
        loss_h = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_sample.view(-1),
            ignore_index=-1,
        )
        loss_h.backward()
        optimizer.update_hessian()
        optimizer.zero_grad(set_to_none=True)
```

The defaults stated above are \(\beta_1=0.96\), \(\beta_2=0.99\), \(\epsilon=10^{-12}\), \(k=10\); the released training configuration uses \(\beta_1=0.965\), `1e-15` in the denominator, and tunes `rho` by the unclipped fraction (`win_rate`, roughly 0.1-0.5).
