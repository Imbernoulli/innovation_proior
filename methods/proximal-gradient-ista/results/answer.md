# Proximal Gradient (ISTA) and its acceleration (FISTA)

## Problem

Minimize a composite convex objective

$$\min_x\ F(x)=f(x)+g(x),$$

with $f$ smooth convex ($\nabla f$ Lipschitz with constant $L$) and $g$ convex but possibly nonsmooth and "simple" (cheap proximal map). The motivating instance is $\ell_1$-regularized least squares (lasso / wavelet deblurring)

$$\min_x\ \|Ax-b\|^2+\lambda\|x\|_1,$$

which is large-scale and dense, so interior-point methods are infeasible and the dominant cost must be matvecs with $A,A^\top$.

## Key idea

Split smooth and nonsmooth. At the current point $y$, replace $f$ by its quadratic upper model from the descent lemma and keep $g$ exact:

$$x^+=\arg\min_x\ \Big[f(y)+\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|^2\Big]+g(x).$$

Completing the square turns this into a gradient step followed by the proximal map of $g$ (forward–backward):

$$x^+=\mathrm{prox}_{tg}\big(y-t\nabla f(y)\big),\qquad \mathrm{prox}_{t g}(v)=\arg\min_x\ g(x)+\tfrac{1}{2t}\|x-v\|^2,\qquad t=\tfrac1L .$$

Its fixed points are exactly the minimizers: $0\in\nabla f(x^\*)+\partial g(x^\*)\iff x^\*=\mathrm{prox}_{tg}(x^\*-t\nabla f(x^\*))$, using the prox optimality $v-\mathrm{prox}_{t g}(v)\in t\,\partial g(\mathrm{prox}_{t g}(v))$.

## The ℓ₁ prox is soft-thresholding

For $g=\lambda\|\cdot\|_1$ the prox separates coordinatewise; solving $\min_{x_i}\lambda|x_i|+\tfrac{1}{2t}(x_i-v_i)^2$ gives

$$\big[\mathrm{prox}_{t\lambda\|\cdot\|_1}(v)\big]_i=\mathrm{sign}(v_i)\,\big(|v_i|-\lambda t\big)_+=T_{\lambda t}(v)_i .$$

So ISTA for the lasso is $x_{k+1}=T_{\lambda t}\big(x_k-2tA^\top(Ax_k-b)\big)$ (with $f=\|Ax-b\|^2$, $\nabla f=2A^\top(Ax-b)$, $L=2\lambda_{\max}(A^\top A)$).

## Algorithms

**ISTA (proximal gradient).** Input $L\ge L(f)$; $x_0$. For $k\ge1$: $x_k=p_L(x_{k-1})$ where $p_L(y)=\arg\min_x Q_L(x,y)$ and $Q_L(x,y)=f(y)+\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|^2+g(x)$.

**FISTA (accelerated).** $y_1=x_0$, $t_1=1$. For $k\ge1$:

$$x_k=p_L(y_k),\qquad t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},\qquad y_{k+1}=x_k+\frac{t_k-1}{t_{k+1}}\,(x_k-x_{k-1}).$$

One gradient and one prox per iteration in both. When $L(f)$ is unknown, backtrack: inflate $L$ by $\eta>1$ until $F(p_L(y))\le Q_L(p_L(y),y)$; under the standard initialization the accepted values satisfy $L_k\le\eta L(f)$.

## Convergence guarantees

Both rest on one inequality. **Engine (Lemma).** If $L$ satisfies $F(p_L(y))\le Q_L(p_L(y),y)$ (true for $L\ge L(f)$), then for all $x$,

$$F(x)-F(p_L(y))\ \ge\ \tfrac{L}{2}\|p_L(y)-y\|^2+L\langle y-x,\,p_L(y)-y\rangle.$$

*Proof.* From the domination, $F(x)-F(p_L(y))\ge F(x)-Q_L(p_L(y),y)$. Convexity gives $f(x)\ge f(y)+\langle x-y,\nabla f(y)\rangle$ and $g(x)\ge g(p_L(y))+\langle x-p_L(y),\gamma\rangle$ with $\gamma=-\nabla f(y)-L(p_L(y)-y)\in\partial g(p_L(y))$ (prox optimality). Subtracting $Q_L(p_L(y),y)$ and substituting $\nabla f(y)+\gamma=-L(p_L(y)-y)$ yields the bound after $\langle x-p_L(y),y-p_L(y)\rangle=\|p_L(y)-y\|^2+\langle y-x,p_L(y)-y\rangle$. ∎

**ISTA: $O(1/k)$.** With $y=x_n$, $x=x^\*$: $\tfrac{2}{\alpha L(f)}(F(x^\*)-F(x_{n+1}))\ge\|x^\*-x_{n+1}\|^2-\|x^\*-x_n\|^2$, using $L_{n+1}\le\alpha L(f)$ and $F(x^\*)-F(x_{n+1})\le0$. Telescoping over $n=0,\dots,k-1$ and using the monotonicity from the same engine at $x=y=x_n$, so $\sum_{n=1}^kF(x_n)\ge kF(x_k)$,

$$F(x_k)-F(x^\*)\le\frac{\alpha L(f)\,\|x_0-x^\*\|^2}{2k},\qquad \alpha=1\ (\text{const}),\ \eta\ (\text{backtracking}).$$

**FISTA: $O(1/k^2)$.** Let $v_k=F(x_k)-F^\*$, $u_k=t_kx_k-(t_k-1)x_{k-1}-x^\*$. Applying the engine twice at $y=y_{k+1}$ (with $x=x_k$ and $x=x^\*$), combining with weights $(t_{k+1}-1)$ and $1$, multiplying by $t_{k+1}$, using $t_k^2=t_{k+1}^2-t_{k+1}$ and the polarization identity gives the variable-step potential recursion

$$\tfrac{2}{L_k}t_k^2 v_k-\tfrac{2}{L_{k+1}}t_{k+1}^2 v_{k+1}\ge\|u_{k+1}\|^2-\|u_k\|^2 .$$

Hence $a_k+b_k$ is nonincreasing for $a_k=\tfrac{2}{L_k}t_k^2 v_k$, $b_k=\|u_k\|^2$. The base case $a_1+b_1\le\|x_0-x^\*\|^2$ follows from the engine at $y=y_1=x_0$, $x=x^\*$. Since $t_{k+1}=\tfrac{1+\sqrt{1+4t_k^2}}{2}\ge t_k+\tfrac12\Rightarrow t_k\ge\tfrac{k+1}{2}$,

$$F(x_k)-F(x^\*)\le\frac{2\,\alpha L(f)\,\|x_0-x^\*\|^2}{(k+1)^2}.$$

This matches the optimal first-order rate (Nemirovsky–Yudin lower bound) for smooth convex minimization, extended here to the nonsmooth composite case. FISTA is not a descent method; the decreasing quantity is the potential, not $F(x_k)$.

## Implementation (ℓ₁ deblurring / lasso)

```python
import numpy as np

def soft_threshold(v, thr):
    # prox of thr*||.||_1 : sign(v)*max(|v|-thr, 0)
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)

def grad_f(x, A, b):
    return 2.0 * (A.T @ (A @ x - b))          # f(x)=||Ax-b||^2

def ista(A, b, lam, x0, n_iter):
    L = 2.0 * np.linalg.eigvalsh(A.T @ A)[-1] # Lipschitz const of grad f
    t = 1.0 / L
    x = x0.copy()
    for _ in range(n_iter):
        x = soft_threshold(x - t * grad_f(x, A, b), lam * t)
    return x

def fista(A, b, lam, x0, n_iter):
    L = 2.0 * np.linalg.eigvalsh(A.T @ A)[-1]
    t = 1.0 / L
    x = x0.copy(); y = x0.copy()
    tk = 1.0
    for _ in range(n_iter):
        x_new  = soft_threshold(y - t * grad_f(y, A, b), lam * t)
        tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk**2)) / 2.0
        y      = x_new + ((tk - 1.0) / tk_new) * (x_new - x)
        x, tk = x_new, tk_new
    return x
```
