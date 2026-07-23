The economics of pre-training a large language model are brutally simple: the run is essentially a long sequence of optimizer steps, and its cost is the number of steps times the cost of one step. So the only way to make a real dent in the time and money is to reach the same validation loss in fewer steps without paying more per step. That second clause is what rules out almost everything attractive. The landscape of a Transformer loss has wildly anisotropic curvature — direct measurement of the diagonal Hessian on a 125M-parameter model shows positive entries spread across many orders of magnitude, some directions very sharp and others very flat — and a single shared learning rate cannot serve both. It must be small enough not to diverge in the sharpest direction, which starves the flat ones, so progress is governed by the condition number. Curvature is exactly the information that fixes this: Newton's method, $\theta\leftarrow\theta-\eta H^{-1}\nabla L$, equalizes loss decrease across coordinates and converges at a rate independent of the condition number. But the classical ways of using it are unusable at scale. The Hessian is $d\times d$, impossible to form or invert; away from a minimum it is indefinite, so the raw step $-g/h$ points uphill along negative-curvature directions and can blow up; and it drifts along the trajectory, so a stale or local quadratic model misleads. The standard remedies — trust regions, line search, cubic regularization — all add machinery and per-step cost, and diagonal-Hessian preconditioners like AdaHessian re-estimate curvature every single step, which by itself costs more than an extra gradient and has prevented any wall-clock speedup on decoder-only language models. Adam, the incumbent, sidesteps curvature entirely: strip away its moving averages and its update is $\eta\cdot\operatorname{sign}(\nabla L)$, a fixed-magnitude step in every coordinate. Equal displacement is not equal progress on a heterogeneous landscape, so the sharp coordinates reach the valley and bounce while the flat ones crawl; on a quadratic even idealized SignGD pays the square root of the condition number.

I propose Sophia, a second-order clipped stochastic optimizer that takes the cheap part of Newton and pairs it with a cheap safety valve. The reasoning that motivates it is a local quadratic $q(\theta)=\tfrac12 h(\theta-\theta^*)^2$: there $g=h(\theta-\theta^*)$, the step $-g/h$ lands exactly at the minimizer, and the removed loss is $\tfrac12 g^2/h$. Curvature is therefore precisely the per-coordinate scale I want — sharp directions get small displacement, flat directions get large displacement — and equalizing loss decrease rather than coordinate displacement is the whole point. So Sophia keeps a diagonal curvature estimate (one extra vector, one scale per parameter, the affordable fragment of Newton) and clips the preconditioned update per coordinate to bound it where the curvature estimate cannot be trusted. Concretely I maintain a gradient EMA for the numerator, $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$, and a diagonal-curvature EMA $h_t$ that is refreshed only every $k$ steps, $h_t=\beta_2 h_{t-k}+(1-\beta_2)\hat h_t$, and otherwise carried forward; after decoupled weight decay the update is

$$\theta_{t+1}=\theta_t-\eta_t\,\operatorname{clip}\!\left(\frac{m_t}{\max\{\gamma h_t,\epsilon\}},1\right).$$

Every piece earns its place. The denominator uses a positive floor, $\max\{\gamma h_t,\epsilon\}$, so a negative or tiny $h_i$ is replaced by $\epsilon$; then $m_i/\epsilon$ keeps the sign of the momentum and the elementwise clip caps it at $\operatorname{sign}(m_i)$ — exactly the behavior I want when curvature is untrustworthy, a bounded momentum-SignGD step, while in the trustworthy regime where $m_i/h_i$ is moderate the clip is inactive and I recover the curvature-scaled step. Clipping alone would only bound magnitude; it is the positive floor together with the clip that handles the sign correctly, which is why both are needed and why mere magnitude clipping of a raw Newton step would not be safe. Refreshing $h_t$ only every $k$ steps is what preserves the cost target — refreshing every step would erase the wall-clock benefit — and this infrequency is tolerable only because any coordinate that has gone stale or wrong is clipped down into a bounded sign step rather than allowed to explode. The role of $\gamma$ is made precise by the reparameterization identity

$$\eta\,\operatorname{clip}(m/\max\{\gamma h,\epsilon\},1)=\frac{\eta}{\gamma}\operatorname{clip}(m/\max\{h,\epsilon/\gamma\},\gamma),$$

which separates two knobs that would otherwise be conflated: in the left-hand form saturated coordinates have update magnitude $\eta$, so $\eta$ sets the size of clipped steps, while $\gamma$ is the clip threshold on the raw curvature-scaled ratio and therefore controls how many coordinates escape clipping. In practice $\gamma$ (or its implementation surrogate $\rho$) is tuned by that unclipped fraction.

The remaining question is how to get a diagonal curvature estimate cheap enough to refresh rarely, and there are two routes. The Hutchinson route (Sophia-H) draws $u\sim\mathcal N(0,I_d)$ and returns $\hat h=u\odot(\nabla^2 L\,u)$; this is unbiased for the true diagonal because $E[u_i(Hu)_i]=\sum_j H_{ij}E[u_iu_j]=H_{ii}$, and the product $Hu$ is a Hessian-vector product obtained as $\nabla_\theta\langle\nabla_\theta L,u\rangle$, so the Hessian is never materialized — at the price that individual coordinates can come out negative and must rely on the floor-and-clip fallback. The Gauss-Newton-Bartlett route (Sophia-G), which is the one I carry into the implementation, exploits the loss structure instead. For $\ell(\theta,(x,y))=\operatorname{ce}(f(\theta,x),y)$ the chain rule gives

$$\nabla_\theta^2\ell=J_\theta f\,S\,J_\theta f^\top+J_{\theta\theta}f[q],$$

with $S=\partial_t^2\operatorname{ce}(t,y)$ and $q=\partial_t\operatorname{ce}(t,y)$ at $t=f(\theta,x)$. Dropping the second term introduces a bias but yields a positive semidefinite surrogate, since cross-entropy is convex in the logits so $S\succeq 0$ — and positivity is exactly what keeps the denominator safe. For softmax cross-entropy $S=\operatorname{diag}(p)-pp^\top$ with $p=\operatorname{softmax}(f)$, which depends only on the logits and not on the realized label. Sampling $\hat y\sim\operatorname{Cat}(p)$, Bartlett's second identity rewrites $S$ as the expected outer product of the score, and multiplying through by the logit Jacobian gives

$$\operatorname{diag}(J_\theta f\,S\,J_\theta f^\top)=E_{\hat y\sim\operatorname{Cat}(p)}[\nabla_\theta\operatorname{ce}(f,\hat y)\odot\nabla_\theta\operatorname{ce}(f,\hat y)].$$

The practical snag is that autodiff hands back the averaged minibatch gradient, not per-example gradients. Writing $\widehat L(\theta)=\tfrac1B\sum_b\operatorname{ce}(f(\theta,x_b),\hat y_b)$ with independent sampled labels, Bartlett's first identity makes each sampled-label score have zero mean, so the cross terms vanish and

$$E[B\,\nabla\widehat L\odot\nabla\widehat L]=E\left[\frac1B\sum_b\nabla\operatorname{ce}_b\odot\nabla\operatorname{ce}_b\right],$$

the diagonal of the minibatch Gauss-Newton matrix. This is the estimator I use: sample labels from the model, run one ordinary backward pass, square the averaged gradient, and multiply by the batch size $B$. It is nonnegative and gradient-only — no Hessian-vector product at all — at the cost of estimating the Gauss-Newton diagonal rather than the full Hessian diagonal.

It is worth being precise about what the theory actually guarantees, because the clean convergence statement is not a theorem about the stochastic diagonal implementation; it analyzes exact full-Hessian clipped Newton, clipped in the Hessian eigenbasis, $\theta_+=\theta-\eta V^\top\operatorname{clip}(V(\nabla^2 L)^{-1}\nabla L,\rho)$ with $\nabla^2 L=V^\top\Sigma V$. Under strict convexity and a multiplicative Hessian-continuity condition, with $\eta\rho\le R/\sqrt d$, a one-step descent bound follows from $f(t)=L(t\theta_++(1-t)\theta)$ staying inside the radius-$R$ neighborhood so that $f''(t)\le 2f''(0)$ and $f(1)\le f(0)+f'(0)+f''(0)$. The first derivative is $f'(0)=-\eta\sum_i\min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\}$ and the second is bounded by the same sum, so

$$L(\theta_+)-L(\theta)\le-(\eta-\eta^2)\sum_i\min\{\rho|v_i^\top\nabla L|,\sigma_i^{-1}|v_i^\top\nabla L|^2\},$$

whose two branches are exactly the design intent: unclipped Newton progress where the local ratio is safe, clipped sign-like progress where it is not. A two-phase argument finishes it — either the summed decrement is small and the iterate is already near the minimizer, or the descent lemma forces a fixed loss drop — and once $L(\theta)-\min L\le\mu\rho^2/8$ no coordinate clips and the update becomes damped Newton with exponential decay. With $\eta=1/2$ and $\rho=R/(2\sqrt d)$ the step count is

$$T\lesssim d\,\frac{L(\theta_0)-\min L}{\mu R^2}+\log\frac{\mu R^2}{32d\epsilon},$$

with no dependence on the condition number or the largest curvature in that idealized model, against the SignGD proxy's $\Omega(\sqrt{\beta/\mu})$ lower bound on $L_{\mu,\beta}=\tfrac{\mu}{2}\theta_1^2+\tfrac{\beta}{2}\theta_2^2$ — the sign step pays the square root of the condition number that Sophia avoids.

In the implementation the ordinary steps use the real-label gradient to update the momentum and apply the clipped, denominator-scaled update, while every $k$ steps a separate sampled-label backward pass refreshes only the curvature EMA. One detail to read carefully: the code stores the unscaled sampled-label gradient square in `hessian`, so the $B$ factor from the Gauss-Newton-Bartlett estimator is supplied at use time by the `bs` argument inside the denominator `rho * bs * h`, which is why the ratio is faithful to the estimator even though `update_hessian` never multiplies by the batch size. The stated defaults are $\beta_1=0.96$, $\beta_2=0.99$, $\epsilon=10^{-12}$, $k=10$; the implementation ships $\beta_1=0.965$, a `1e-15` denominator floor, and tunes `rho` by the unclipped fraction.

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
