# Context: minimizing a smooth-plus-nonsmooth convex objective at scale

## Research question

Across signal and image processing, statistical inference, and optics one repeatedly faces a linear inverse problem

$$Ax = b + w,$$

with $A \in \mathbb{R}^{m\times n}$ and $b\in\mathbb{R}^m$ known, $w$ unknown noise, and $x$ the signal/image to recover. In image deblurring $b$ is the blurred, noisy picture, $A$ is the (convolution) blur operator, and $x$ is the sharp image. When $A$ is ill-conditioned — which is the typical case — the naive least-squares estimate $\arg\min_x \|Ax-b\|^2$ has enormous norm and is meaningless; regularization is mandatory.

The regularizer that has drawn the most interest is the $\ell_1$ norm, giving

$$\min_x\ F(x) \equiv \|Ax-b\|^2 + \lambda\|x\|_1 .$$

In wavelet-based restoration $A = RW$ ($R$ the blur, $W$ an inverse wavelet transform), $x$ holds the wavelet coefficients, and the $\ell_1$ term induces sparsity because most natural images are sparse in the wavelet domain. The $\ell_1$ penalty is also less sensitive to outliers (sharp edges) than an $\ell_2$ penalty.

The precise goal: solve this convex but **nonsmooth** problem when it is **large-scale** (image deblurring can reach millions of decision variables) and the matrices are **dense**. A solver must therefore (i) handle the nondifferentiable $\ell_1$ term exactly, (ii) keep the per-iteration cost down to a few matrix–vector products with $A$ and $A^\top$, and (iii) reach a target accuracy in as few iterations as possible. Any method whose iteration requires a matrix factorization or solving a dense linear system is excluded at this scale.

More generally the same shape recurs whenever one minimizes a sum

$$\min_x\ f(x) + g(x),$$

with $f$ smooth convex and $g$ convex but possibly nonsmooth. A good method for the $\ell_1$ case should not be tied to $\ell_1$; it should work for any such pair.

## Background

**Why interior-point methods do not scale here.** The $\ell_1$ problem can be recast as a second-order cone program and solved by interior-point methods, which converge in very few (Newton-type) iterations. But each iteration solves a large, generally dense linear system. For deblurring with millions of variables and a dense $A$, forming and factoring that system is out of reach. This is the pressure that pushes the field toward **first-order methods** — methods that use only function values and gradients, where the dominant cost is a cheap matvec with $A$ and $A^\top$. First-order methods need many more iterations, so their *rate of convergence* becomes the decisive quantity.

**The descent lemma for smooth $f$.** If $f$ is continuously differentiable with $L$-Lipschitz gradient ($\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$), then for all $x,y$

$$f(x) \le f(y) + \langle x-y,\nabla f(y)\rangle + \tfrac{L}{2}\|x-y\|^2 .$$

A convex quadratic with curvature $L$ sits above $f$ and touches it at $y$. This is the single most-used fact in first-order analysis: it certifies that the gradient step $y \mapsto y-\tfrac1L\nabla f(y)$ — which is exactly the minimizer of that upper quadratic — decreases $f$. The classical gradient method $x_k = x_{k-1}-t\nabla f(x_{k-1})$ can be read as the proximal regularization of the *linearized* $f$: $x_k=\arg\min_x f(x_{k-1})+\langle x-x_{k-1},\nabla f(x_{k-1})\rangle+\tfrac{1}{2t}\|x-x_{k-1}\|^2$ (Levitin–Polyak 1966; Martinet 1970; Bertsekas).

**The subgradient $\partial g$ of a nonsmooth convex $g$.** For convex $g$, $v\in\partial g(z)$ means $g(x)\ge g(z)+\langle x-z,v\rangle$ for all $x$. For $g(x)=\lambda|x|$ in one variable, $\partial g(0)=[-\lambda,\lambda]$ and $\partial g(x)=\lambda\,\mathrm{sign}(x)$ for $x\ne0$. Optimality of an unconstrained convex problem is Fermat's rule $0\in\partial g(x)$. A direct subgradient method for $F$ exists but is slow — its worst-case rate for a Lipschitz nonsmooth objective is only $O(1/\sqrt{k})$ — precisely because it ignores the smoothness of the $f$ part and smears the nonsmooth part into one subgradient.

**The Moreau proximity operator (Moreau, 1962).** For convex $\varphi$, the Moreau envelope of index $\gamma$ is $\;{}^{\gamma}\varphi(x)=\inf_y \varphi(y)+\tfrac{1}{2\gamma}\|x-y\|^2$, and the unique minimizer

$$\mathrm{prox}_{\varphi}(x) = \arg\min_y \ \varphi(y) + \tfrac12\|y-x\|^2$$

is the proximity operator. It is characterized by the inclusion $x-\mathrm{prox}_{\varphi}(x)\in\partial\varphi(\mathrm{prox}_{\varphi}(x))$, it generalizes Euclidean projection (when $\varphi=\iota_C$ is the indicator of a convex set $C$, $\mathrm{prox}_{\varphi}=P_C$), and it is firmly nonexpansive.

**Sparsity and wavelet shrinkage.** That thresholding small wavelet coefficients denoises a signal goes back to Donoho–Johnstone (1995) and the variational/Besov-penalty denoising of Chambolle, DeVore, Lee, Lucier (1998). The empirical fact the field already takes as given: natural images have sparse wavelet expansions, an $\ell_1$ penalty promotes sparse minimizers, and soft-thresholding a coefficient is exactly the one-dimensional solution of "data fit plus $\lambda\times$ absolute value." This shrinkage step is cheap, separable, and the obvious nonlinear building block for sparse recovery.

**Nesterov's optimal first-order method (1983).** For *smooth* convex minimization, the plain gradient method achieves $F(x_k)-F^* = O(1/k)$. Nesterov (1983) exhibited a gradient method achieving $O(1/k^2)$ using no more than one gradient per iteration plus one extra, cheaply computed auxiliary point. By the Nemirovsky–Yudin (1983) complexity theory this $O(1/k^2)$ is the best attainable rate for any method that sees only first-order information at a sequence of points — it is "optimal." The catch: this acceleration was developed for smooth problems, and its standard analysis (estimate sequences) is built around a differentiable objective.

## Baselines

**Iterative shrinkage-thresholding (ISTA) — the proximal forward-backward lineage.** The popular workhorse for the $\ell_1$ problem. Each step is a gradient step on the data term followed by a soft-threshold:

$$x_{k+1} = T_{\lambda t}\!\big(x_k - 2t\,A^\top(Ax_k-b)\big), \qquad T_\alpha(x)_i = (|x_i|-\alpha)_+\,\mathrm{sign}(x_i).$$

It has been derived independently by several groups under different names (threshold Landweber, iterative denoising, EM-type wavelet restoration; Figueiredo–Nowak 2003; Chambolle et al. 1998; Hale–Yin–Zhang 2007; Vonesch–Unser 2007; Wright–Nowak–Figueiredo 2008). In the optimization literature it traces to the **proximal forward-backward** splitting of Bruck (1977), Passty (1979), and is analyzed in depth by **Combettes & Wajs (2005)**, whose fixed-point view is $0\in\nabla f(x)+\partial g(x)\iff x=\mathrm{prox}_{tg}(x-t\nabla f(x))$, with the iteration $x_{n+1}=\mathrm{prox}_{tg}(x_n-t\nabla f(x_n))$ converging under $t\in(0,2/L)$-type conditions. **Daubechies, Defrise & De Mol (2004)** derive the same iteration by *optimization transfer*: they add a "surrogate" term $\Xi(f;a)=C\|f-a\|^2-\|Kf-Ka\|^2$ (with $C>\|K^*K\|$) to decouple the coupling operator $K^*Kf$, so the surrogate functional minimizes coordinatewise and the per-coordinate solution is exactly soft-thresholding; they prove the iterates converge in norm to a minimizer. Convergence of the sequence $\{x_k\}$ to a minimizer of $F$ holds (e.g. for $t\in(0,1/\|A^\top A\|)$). **Gap:** ISTA is simple and cheap per iteration, but slow. Its analyses focus on *whether* the sequence converges, not on a nonasymptotic rate in function value; in practice it crawls, and recent analysis shows its asymptotic rate can be arbitrarily bad (Bredies–Lorenz 2008).

**Two-step / multistep ISTA accelerations (concurrent).** TWIST (Bioucas-Dias & Figueiredo 2007) builds the next iterate from two previous iterates and shows experimental speedups for $\|Ax-b\|^2+\varphi(x)$; subspace-optimization methods (Elad, Matalon, Zibulevsky 2007) minimize over an affine subspace spanned by past iterates and the current gradient. **Gap:** both demonstrate speedups empirically but establish *no* global nonasymptotic rate of convergence.

**Direct subgradient method on $F$.** Treat $F=\|Ax-b\|^2+\lambda\|x\|_1$ as one nonsmooth convex function and step along a subgradient. **Gap:** rate only $O(1/\sqrt{k})$; it throws away the smoothness of the data term that a splitting method exploits.

**Interior-point / SOCP solvers.** Recast as a cone program (Ben-Tal–Nemirovski 2001). **Gap:** each iteration needs a dense linear solve; infeasible at the million-variable, dense-$A$ scale.

## Evaluation settings

The natural test problems are $\ell_1$/wavelet-based image deblurring. A representative protocol: take a grayscale test image (e.g. the $256\times256$ "cameraman", or a simple synthetic image from a regularization toolbox), scale pixels to $[0,1]$, blur it (e.g. a $9\times9$ Gaussian, standard deviation $4$, via standard image-filtering routines), and add zero-mean white Gaussian noise (e.g. standard deviation $10^{-3}$). Use reflexive (Neumann) boundary conditions. Set $A=RW$ with $R$ the blur matrix and $W$ an inverse multi-stage Haar wavelet transform, choose a regularization parameter $\lambda$, and initialize at the blurred image. Because $A^\top A$ here is diagonalizable by a two-dimensional cosine transform, its eigenvalues — hence the Lipschitz constant — are computable, allowing constant-step runs. The yardsticks are the objective value $F(x_k)$ as a function of iteration $k$ and the function-value error $F(x_k)-F^*$ (when an optimal value is known, e.g. a noiseless $\lambda=0$ case whose optimum is $0$), tracked over hundreds to tens of thousands of iterations, plus visual quality of the restored image. The methods to compare against are ISTA and the two-step accelerations above.

## Code framework

A generic first-order solver harness for $\min_x f(x)+g(x)$. It exposes the problem data, an update placeholder, and a driver loop.

```python
import numpy as np

# ---- problem data: l1-regularized least squares  f(x)=||Ax-b||^2,  g(x)=lam||x||_1 ----
def make_problem(A, b, lam):
    AtA = A.T @ A
    def f(x):       return np.sum((A @ x - b)**2)
    def grad_f(x):  return 2.0 * (A.T @ (A @ x - b))      # cheap: two matvecs
    def g(x):       return lam * np.sum(np.abs(x))
    L = 2.0 * np.linalg.eigvalsh(AtA)[-1]                 # Lipschitz const of grad f
    return f, grad_f, g, L, lam

# ---- one update of the first-order loop ----
def update(state, f, grad_f, g, t, lam):
    # state may hold current and previous iterates plus scalar recurrences.
    # TODO: fill in
    pass

# ---- the driver: a plain first-order loop, generic to any update ----
def solve(f, grad_f, g, L, lam, x0, n_iter):
    t = 1.0 / L
    state = {"x": x0.copy(), "x_prev": x0.copy(), "k": 0}
    history = []
    for _ in range(n_iter):
        state = update(state, f, grad_f, g, t, lam)
        history.append(f(state["x"]) + g(state["x"]))
    return state["x"], history
```
