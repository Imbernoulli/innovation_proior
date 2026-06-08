# Local Minimax: local optimality for nonconvex-nonconcave minimax optimization

## Problem

For $\min_{\mathbf x\in\mathcal X}\max_{\mathbf y\in\mathcal Y} f(\mathbf x,\mathbf y)$ with $f$ smooth but
nonconvex–nonconcave (GANs, adversarial training, multi-agent RL), classical equilibrium notions break:
Nash/saddle points need not exist and need not match what algorithms find, and because
$\min\max\neq\max\min$ here, simultaneous-game notions ignore the load-bearing *order* of play. The right
global object is the **Stackelberg/global minimax** point — the leader minimizing the follower's worst-case
value $\phi(\mathbf x):=\max_{\mathbf y} f(\mathbf x,\mathbf y)$ — but it is NP-hard to find. **Local
minimax** is the tractable local surrogate for it.

## Definition (the landing artifact)

**Local minimax.** $(\mathbf x^\star,\mathbf y^\star)$ is a *local minimax point* of $f$ if there exist
$\delta_0>0$ and a function $h$ with $h(\delta)\to0$ as $\delta\to0$, such that for every
$\delta\in(0,\delta_0]$ and every $(\mathbf x,\mathbf y)$ with $\|\mathbf x-\mathbf x^\star\|\le\delta$ and
$\|\mathbf y-\mathbf y^\star\|\le\delta$,
$$f(\mathbf x^\star,\mathbf y)\ \le\ f(\mathbf x^\star,\mathbf y^\star)\ \le\ \max_{\mathbf y':\,\|\mathbf y'-\mathbf y^\star\|\le h(\delta)} f(\mathbf x,\mathbf y').$$
The arbitrary vanishing $h$ lets the follower's reaction neighborhood shrink at its own rate
($h(\delta)/\delta$ may $\to\infty$), encoding the leader-commits / follower-best-responds asymmetry while
keeping the notion truly local. Equivalently: $\mathbf y^\star$ is a local max of $f(\mathbf x^\star,\cdot)$,
and there is $\epsilon_0>0$ with $\mathbf x^\star$ a local min of
$g_\epsilon(\mathbf x)=\max_{\|\mathbf y-\mathbf y^\star\|\le\epsilon}f(\mathbf x,\mathbf y)$ for all
$\epsilon\in(0,\epsilon_0]$. Local minimax is a strict weakening of local Nash (every local Nash is local
minimax via $h\equiv0$; the converse fails, e.g. $f=-x^2+5xy-y^2$ at the origin).

## Optimality-condition theorem

Write $\mathbf A=\nabla^2_{\mathbf x\mathbf x}f$, $\mathbf B=\nabla^2_{\mathbf y\mathbf y}f$,
$\mathbf C=\nabla^2_{\mathbf x\mathbf y}f$ (so $\mathbf C^\top=\nabla^2_{\mathbf y\mathbf x}f$) at the point.

- **First-order necessary.** Any local minimax point is stationary:
  $\nabla_{\mathbf x}f=\nabla_{\mathbf y}f=0$.
- **Second-order necessary.** $\mathbf B\preceq0$; and if $\mathbf B\prec0$ then the Schur complement
  $$\mathbf A-\mathbf C\mathbf B^{-1}\mathbf C^\top=\nabla^2_{\mathbf x\mathbf x}f-\nabla^2_{\mathbf x\mathbf y}f\,(\nabla^2_{\mathbf y\mathbf y}f)^{-1}\nabla^2_{\mathbf y\mathbf x}f\ \succeq\ 0.$$
- **Second-order sufficient (strict local minimax).** A stationary point with $\mathbf B\prec0$ and
  $\mathbf A-\mathbf C\mathbf B^{-1}\mathbf C^\top\succ0$ is a local minimax point.

When $\mathbf B$ is nondegenerate, necessary and sufficient conditions match up to $\succeq$ vs $\succ$.
Contrast local Nash's sufficient condition ($\mathbf A\succ0$, $\mathbf B\prec0$): local minimax replaces
$\mathbf A\succ0$ by the Schur complement, bringing in the interaction $\mathbf C$ and breaking the
$\mathbf x\leftrightarrow\mathbf y$ symmetry — the order of play now lives in the algebra.

*Proof sketch.* Stationarity from the first-order argument. With $\mathbf B\prec0$, the inner local max of
the second-order expansion $\tfrac12\boldsymbol\delta_x^\top\mathbf A\boldsymbol\delta_x+\boldsymbol\delta_x^\top\mathbf C\boldsymbol\delta_y+\tfrac12\boldsymbol\delta_y^\top\mathbf B\boldsymbol\delta_y$
is attained at $\boldsymbol\delta_y^\star=-\mathbf B^{-1}\mathbf C^\top\boldsymbol\delta_x$ (norm
$O(\|\boldsymbol\delta_x\|)$, hence within the local radius), giving inner-max value
$\tfrac12\boldsymbol\delta_x^\top(\mathbf A-\mathbf C\mathbf B^{-1}\mathbf C^\top)\boldsymbol\delta_x+o(\|\boldsymbol\delta_x\|^2)$;
the leader's local-min requirement makes this $\ge0$ ($\succeq$) or, for sufficiency, $>0$ ($\succ$). $\square$

## Connection to gradient descent ascent (the algorithmic meaning)

$\gamma$-GDA flow ($\dot{\mathbf x}=-\tfrac1\gamma\nabla_{\mathbf x}f$, $\dot{\mathbf y}=\nabla_{\mathbf y}f$)
has Jacobian $\mathbf J_\gamma=\begin{pmatrix}-\tfrac1\gamma\mathbf A & -\tfrac1\gamma\mathbf C\\ \mathbf C^\top & \mathbf B\end{pmatrix}$;
a fixed point is strictly stable iff all eigenvalues have negative real part. With $\epsilon=1/\gamma\to0$ the
eigenvalues split into $d_2$ values $\approx\mathrm{eig}(\mathbf B)$ and $d_1$ values
$\approx-\epsilon\,\mathrm{eig}(\mathbf A-\mathbf C\mathbf B^{-1}\mathbf C^\top)$ (via Schur factorization +
continuity of polynomial roots). Hence for large $\gamma$, strict stability $\iff \mathbf B\prec0$ and
$\mathbf A-\mathbf C\mathbf B^{-1}\mathbf C^\top\succ0$ $\iff$ strict local minimax. Formally, with
$\underline{\infty\text{-GDA}}=\liminf_\gamma$ and $\overline{\infty\text{-GDA}}=\limsup_\gamma$ of the strict
stable sets,
$$\textsf{local-minimax}\subset\underline{\infty\text{-GDA}}\subset\overline{\infty\text{-GDA}}\subset\textsf{local-minimax}\cup\{\text{stationary},\ \nabla^2_{\mathbf y\mathbf y}f\ \text{degenerate}\}.$$
So $\infty$-GDA $=$ local minimax away from the degenerate follower-Hessian boundary. For fixed $\gamma$
neither inclusion holds — explicit quadratic counterexamples — so the $\gamma\to\infty$ separation is
essential.

## Structural facts

- A **global** minimax point can be *neither* local minimax *nor* stationary (e.g. $f=0.2xy-\cos y$, global
  solutions $(0,\pm\pi)$ have nonzero gradient) — unlike minimization, where global minima are local minima.
- Local minimax points **may not exist** in general (e.g. $f=y^2-2xy$ on a box), but **do exist** when
  $f(\mathbf x,\cdot)$ is strongly concave near its maxima with all local maxima global; then the global
  minimax point is also local minimax.
- **Max-oracle case.** When $\max_{\mathbf y}f(\mathbf x,\cdot)$ is solvable, the problem reduces to
  minimizing the $\ell$-weakly-convex $\phi=\max_{\mathbf y}f$; GD-with-max-oracle reaches an approximate
  stationary point measured by the Moreau-envelope gradient, at rate
  $\mathbb E\|\nabla\phi_{1/(2\ell)}(\bar{\mathbf x})\|^2\le 2\frac{(\phi_{1/(2\ell)}(\mathbf x_0)-\min\phi)+\ell L^2\gamma^2}{\gamma\sqrt{T+1}}+4\ell\epsilon$.
- **Mixed strategies** restore the minimax theorem (Glicksberg 1952): order becomes irrelevant and mixed
  Nash always exists; one empirical side reduces to an $N$-fold pure global minimax with
  $N\gtrsim d_2(LD/\epsilon)^2\log(LD/\epsilon)$, and the symmetric maximin construction gives the other
  empirical side.

## Minimal numerical skeleton

```python
import numpy as np

def _as_2d(matrix):
    return np.atleast_2d(np.asarray(matrix, dtype=float))

def schur_complement(A, B, C):
    A, B, C = map(_as_2d, (A, B, C))
    return A - C @ np.linalg.solve(B, C.T)

def is_negative_definite(matrix, tol=1e-9):
    return np.max(np.linalg.eigvalsh(_as_2d(matrix))) < -tol

def is_positive_definite(matrix, tol=1e-9):
    return np.min(np.linalg.eigvalsh(_as_2d(matrix))) > tol

def is_stationary(grad_x_value, grad_y_value, tol=1e-8):
    return np.linalg.norm(grad_x_value) < tol and np.linalg.norm(grad_y_value) < tol

def second_order_test(A, B, C, tol=1e-9):
    """Strict Hessian test: B < 0 and A - C B^{-1} C^T > 0."""
    B = _as_2d(B)
    if not is_negative_definite(B, tol):
        return False
    return is_positive_definite(schur_complement(A, B, C), tol)

def is_local_optimum(grad_x_value, grad_y_value, A, B, C, tol=1e-8):
    return (
        is_stationary(grad_x_value, grad_y_value, tol)
        and second_order_test(A, B, C, tol)
    )

def gda_step(x, y, grad_x, grad_y, eta):
    return x - eta * grad_x(x, y), y + eta * grad_y(x, y)

def order_aware_step(x, y, grad_x, grad_y, eta, ratio):
    return x - (eta / ratio) * grad_x(x, y), y + eta * grad_y(x, y)

def flow_jacobian(A, B, C, ratio):
    A, B, C = map(_as_2d, (A, B, C))
    return np.block([[-(1.0 / ratio) * A, -(1.0 / ratio) * C],
                     [                 C.T,                  B]])

def linearly_stable(jacobian, tol=1e-9):
    return np.all(np.real(np.linalg.eigvals(jacobian)) < -tol)
```

As `ratio` grows, `linearly_stable(flow_jacobian(A, B, C, ratio))` matches `second_order_test(A, B, C)` for
nondegenerate $\mathbf B$, which is exactly the sourced $\infty$-GDA correspondence.
