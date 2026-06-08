# Nesterov's Accelerated Gradient Method

## Problem

Minimize a convex, $\beta$-smooth function $f:\mathbb{R}^n\to\mathbb{R}$ (gradient $\beta$-Lipschitz) using only a first-order oracle (value + gradient), one gradient per step. Plain gradient descent gives $f(x_k)-f^*=O(1/k)$ in the convex case and oracle complexity $O(\kappa\log\frac1\varepsilon)$ in the $\alpha$-strongly convex case ($\kappa=\beta/\alpha$). Neither is optimal.

## Key idea

Read the gradient at an **extrapolated look-ahead point**, not at the current iterate. Take a gradient step, then push past it with momentum:
$$x_{k+1}=y_k-\tfrac1\beta\nabla f(y_k),\qquad y_{k+1}=x_{k+1}+\beta_k\,(x_{k+1}-x_k).$$
The extrapolation-*before*-gradient ordering is what makes the method certifiable on the whole smooth convex class (heavy ball, which applies momentum and reads the gradient at the same point, accelerates only on quadratics and can fail to converge in general). The momentum coefficient ramps as $\beta_k=\tfrac{a_k-1}{a_{k+1}}\to\tfrac{k-1}{k+2}$ in the convex case, with $a_0=1$, $a_{k+1}=\tfrac{1+\sqrt{1+4a_k^2}}2$; in the strongly convex case it is the constant $\beta_k=\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$.

## Convergence theorems

**Smooth convex ($\mu=0$).** With $\gamma_0=3\beta$,
$$f(x_k)-f^*\le\frac{8\beta\|x_0-x^*\|^2}{3(k+1)^2}=O\!\Big(\frac1{k^2}\Big).$$
(The line-search variant gives $f(x_k)-f^*\le\tfrac{C}{(k+2)^2}$, $C=4\beta\|y_0-x^*\|^2$; $C=2\beta\|y_0-x^*\|^2$ if $\beta$ is known and the accepted step is always $1/\beta$.)

**Smooth strongly convex ($\mu>0$).** With $\gamma_0=\mu$, $\alpha_k\equiv\sqrt{\mu/\beta}=1/\sqrt\kappa$, $\beta_k\equiv\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$,
$$f(x_k)-f^*\le\tfrac{\beta+\mu}2\|x_0-x^*\|^2\Big(1-\tfrac1{\sqrt\kappa}\Big)^k\approx\tfrac{\beta+\mu}2\|x_0-x^*\|^2\,e^{-k/\sqrt\kappa},$$
oracle complexity $O(\sqrt\kappa\log\frac1\varepsilon)$.

**Matching lower bounds.** For any first-order black-box method (queries in the span of observed gradients), there is a $\beta$-smooth convex $f$ with $\min_{s\le t}f(x_s)-f^*\ge\tfrac{3\beta}{32}\tfrac{\|x_1-x^*\|^2}{(t+1)^2}=\Omega(1/t^2)$, and a $\beta$-smooth $\alpha$-strongly convex $f$ with $f(x_t)-f^*\ge\tfrac\alpha2\big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^{2(t-1)}\|x_1-x^*\|^2\approx\tfrac\alpha2 e^{-4(t-1)/\sqrt\kappa}\|x_1-x^*\|^2=\Omega(\sqrt\kappa\log\frac1\varepsilon)$. The accelerated method attains both floors up to constants — it is **optimal**.

## Why it works (estimate-sequence proof, $\mu\ge0$)

An *estimating sequence* is $\{\phi_k\},\{\lambda_k\}$ with $\lambda_k\to0$ and $\phi_k(x)\le(1-\lambda_k)f(x)+\lambda_k\phi_0(x)$ for all $x$. If iterates satisfy $f(x_k)\le\phi_k^*:=\min_x\phi_k(x)$, then evaluating at $x^*$:
$$f(x_k)-f^*\le\phi_k(x^*)-f^*\le\lambda_k\big(\phi_0(x^*)-f^*\big),$$
so the rate equals the decay of $\lambda_k$.

*Construction.* $\phi_0(x)=\phi_0^*+\tfrac{\gamma_0}2\|x-v_0\|^2$, $\lambda_0=1$, and recurse with $\alpha_k\in(0,1)$, $\lambda_{k+1}=(1-\alpha_k)\lambda_k$,
$$\phi_{k+1}(x)=(1-\alpha_k)\phi_k(x)+\alpha_k\big[f(y_k)+\nabla f(y_k)^\top(x-y_k)+\tfrac\mu2\|x-y_k\|^2\big].$$
The bracket is a valid lower model by ($\mu$-strong) convexity, so this stays an estimating sequence. Each $\phi_k$ keeps the form $\phi_k^*+\tfrac{\gamma_k}2\|x-v_k\|^2$ with $\gamma_{k+1}=(1-\alpha_k)\gamma_k+\alpha_k\mu$, $v_{k+1}=\tfrac1{\gamma_{k+1}}[(1-\alpha_k)\gamma_k v_k+\alpha_k\mu y_k-\alpha_k\nabla f(y_k)]$.
The minimum value obeys
$$\phi_{k+1}^*=(1-\alpha_k)\phi_k^*+\alpha_k f(y_k)-\tfrac{\alpha_k^2}{2\gamma_{k+1}}\|\nabla f(y_k)\|^2+\tfrac{\alpha_k(1-\alpha_k)\gamma_k}{\gamma_{k+1}}\big(\tfrac\mu2\|y_k-v_k\|^2+\nabla f(y_k)^\top(v_k-y_k)\big).$$

*Maintaining $f(x_{k+1})\le\phi_{k+1}^*$.* Lower-bounding the $\phi_{k+1}^*$ recurrence (drop the strong-convexity term; use $\phi_k^*\ge f(x_k)\ge f(y_k)+\nabla f(y_k)^\top(x_k-y_k)$) leaves a gradient term and a cross term. Two choices close it:
- **Step:** set $\beta\alpha_k^2=\gamma_{k+1}$. Then $\tfrac{\alpha_k^2}{2\gamma_{k+1}}=\tfrac1{2\beta}$, and the gradient step $x_{k+1}=y_k-\tfrac1\beta\nabla f(y_k)$ (smoothness descent) covers the gradient term: $f(y_k)-\tfrac1{2\beta}\|\nabla f(y_k)\|^2\ge f(x_{k+1})$.
- **Look-ahead:** the cross term has no sign, so choose $y_k$ to kill it: $y_k=\tfrac{\alpha_k\gamma_k v_k+\gamma_{k+1}x_k}{\gamma_k+\alpha_k\mu}$ — a convex blend of $x_k$ and the lower-model minimizer $v_k$. This is the extrapolation.

Eliminating $v_k,\gamma_k$ gives the two-sequence momentum form with $\beta_k=\tfrac{\alpha_k(1-\alpha_k)}{\alpha_k^2+\alpha_{k+1}}$ and $\alpha_{k+1}^2=(1-\alpha_{k+1})\alpha_k^2+q_f\alpha_{k+1}$, $q_f=\mu/\beta$.

*Rates.* $\mu=0$: the recurrence forces $a_k=1/\alpha_{k-1}$-type growth $a_k\ge1+\tfrac k2$, so $\lambda_k\sim 1/k^2$, giving $O(1/k^2)$. $\mu>0$ with $\gamma_0=\mu$: $\alpha_k\equiv\sqrt{q_f}$ is the fixed point, $\lambda_k=(1-\sqrt{q_f})^k$, giving $e^{-k/\sqrt\kappa}$.

*Equivalent Lyapunov proof (convex case).* The potential $\Phi_t=\lambda_{t-1}^2(f(y_t)-f^*)+\tfrac\beta2\|u_t\|^2$, $u_t=\lambda_t x_t-(\lambda_t-1)y_t-x^*$, is non-increasing (from the two smoothness inequalities at $x_t$ plus the telescoping identity the $x_{t+1}$-update is designed to produce), so $\lambda_{t-1}^2(f(y_t)-f^*)\le\tfrac\beta2\|x_1-x^*\|^2$, and $\lambda_{t-1}\ge t/2$ yields $f(y_t)-f^*\le\tfrac{2\beta\|x_1-x^*\|^2}{t^2}$.

## Final algorithm (code)

```python
import numpy as np

def accelerated_gradient(obj, x0, n_iters):
    """Nesterov's accelerated gradient method.

    obj.beta : gradient-Lipschitz constant (smoothness)
    obj.mu   : strong-convexity parameter (0 if merely convex)
    obj.grad(x) : the first-order oracle
    Reads the gradient at the extrapolated look-ahead point y, then steps.
    """
    beta, mu = obj.beta, obj.mu
    x = np.array(x0, dtype=float)
    y = x.copy()                       # x1 = y1
    x_prev = x.copy()

    if mu > 0:
        sqrt_kappa = np.sqrt(beta / mu)
        momentum_const = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)
    a = 1.0                            # a_0 = 1 (convex case)

    for _ in range(n_iters):
        x_prev = x
        x = y - (1.0 / beta) * obj.grad(y)                 # gradient step at look-ahead
        if mu > 0:
            mom = momentum_const                           # exp(-k/sqrt(kappa))
        else:
            a_next = (1.0 + np.sqrt(1.0 + 4.0 * a * a)) / 2.0   # a_{k+1}^2 - a_{k+1} = a_k^2
            mom = (a - 1.0) / a_next                        # -> (k-1)/(k+2)
            a = a_next
        y = x + mom * (x - x_prev)                          # extrapolate before next gradient

    return x
```

If $\beta$ is unknown, replace `1/beta` by a backtracking line search that, starting from the previous accepted step, halves $h$ until $f(y)-f(y-h\nabla f(y))\ge\tfrac h2\|\nabla f(y)\|^2$; the accepted $h_k\ge\tfrac1{2\beta}$ and the total extra halvings are logarithmic in the initial overestimate, preserving the rate (with constant $C=4\beta\|y_0-x^*\|^2$). If $\mu$ is known but the convex coefficient is used, restart the convex method every $\lceil4\sqrt{\beta/\mu}\rceil-1$ iterations; each block halves the gap, giving $O(\sqrt\kappa\log\frac1\varepsilon)$.
