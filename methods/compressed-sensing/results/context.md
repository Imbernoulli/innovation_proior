# Context: Recovering a sparse signal from far too few linear measurements

## Research question

We are handed a linear measurement model

$$ y = A x + w, \qquad A \in \mathbb{R}^{m\times n}, \quad m \ll n, $$

where $x \in \mathbb{R}^n$ is the unknown signal (or its coefficients in some fixed
basis), $A$ is a known measurement matrix, $y \in \mathbb{R}^m$ are the observations,
and $w$ is noise. The system is **underdetermined**: there are fewer equations than
unknowns, so $\{x : Ax = y\}$ is an affine subspace of dimension at least $n-m$, with
infinitely many exact solutions. In application after application the signal of interest is
**sparse** — only a handful $k \ll n$ of its entries are nonzero — or sparse after a known
transform (a natural image is sparse in a wavelet basis; a sum of a few sinusoids is sparse
in Fourier). The question is whether sparsity is enough side information to pin down the
*right* $x$ from $m \ll n$ measurements, and how to compute it at the relevant scale, where
$n$ runs to the millions and $A$ is dense.

## Background

**The least-squares response.** The textbook response to an underdetermined
system is the minimum-norm solution $\hat x = A^{\!\top}(AA^{\!\top})^{-1}y$, i.e. the
$\ell_2$-smallest $x$ with $Ax=y$. It spreads energy across all coordinates. Tikhonov
regularization, $\min_x \|Ax-y\|_2^2 + \lambda\|Lx\|_2^2$, adds a smooth quadratic penalty
that shrinks coefficients toward zero.

**Sparsity as a count.** A direct formalization of "find the
sparsest consistent signal" uses the $\ell_0$ "norm" $\|x\|_0 = \#\{i : x_i \neq 0\}$:

$$ (P_0)\qquad \min_x \|x\|_0 \quad\text{s.t.}\quad Ax = y \quad(\text{or } \|Ax-y\|_2\le\varepsilon). $$

$\|\cdot\|_0$ is not a norm and is nonconvex; $(P_0)$ is a combinatorial problem. To solve
it exactly one runs over candidate supports $T \subseteq \{1,\dots,n\}$
and tests whether a consistent signal lives on each — exponentially many subsets. Natarajan
(1995, *Sparse Approximate Solutions to Linear Systems*, SIAM J. Comput. 24:227–234)
proved this is **NP-hard**: given $A$, $b$, $\varepsilon$, computing the fewest-nonzero $x$
with $\|Ax-b\|_2\le\varepsilon$ is intractable in the worst case.

**The $\ell_1$ relaxation.** Replacing the count by the
$\ell_1$ norm $\|x\|_1 = \sum_i |x_i|$ turns the combinatorial problem into a convex one:

$$ (P_1)\qquad \min_x \|x\|_1 \quad\text{s.t.}\quad Ax = y, $$

a linear program (basis pursuit; Chen, Donoho & Saunders 1998, *Atomic decomposition by
basis pursuit*, SIAM J. Sci. Comput. 20:33–61). The observed phenomenon is that the
$\ell_1$ solution is frequently *identical* to the $\ell_0$ solution.
Candès, Romberg & Tao (2006, *Robust Uncertainty Principles*, arXiv:math/0409186) made this
precise for the Fourier-sampling model: for an overwhelming fraction of supports $T$ and
random frequency sets $\Omega$ with $|T| \le \alpha\,|\Omega|/\log N$, the solutions of
$(P_0)$ and $(P_1)$ coincide, so the convex program recovers the sparse signal *exactly*.
Donoho (2006, *Compressed Sensing*, IEEE Trans. Inf. Theory 52:1289–1306) established the
companion sampling-rate statement: if an ambient object has $N$ samples but is compressible
in a known transform, then nonadaptive linear measurements far below $N$ can be enough. In
the abstract image model, certain $N$-pixel image classes need only
$O(N^{1/4}\log^{5/2}N)$ nonpixel measurements for faithful recovery; in the sparse-coefficient
model, on the order of $S\log N$ random measurements can match the error of keeping the
$S$ largest transform coefficients. The structural property of $A$ that makes this work is
that $A$ acts as a near-isometry on sparse vectors (a *restricted isometry*): every small set
of columns is approximately orthonormal, so distinct sparse signals stay separated under $A$
and the sparse solution is the unique one $\ell_1$ can find. Random and Fourier-subsampling
matrices satisfy this with high probability.

**Why $\ell_1$ induces sparsity (the geometry).** Tibshirani (1996, the lasso; reviewed
in *Regression shrinkage and selection via the lasso: a retrospective*, J. R. Statist.
Soc. B 73:273–282) gave the picture in the statistical setting: minimizing
$\|y-Ax\|_2^2$ subject to $\|x\|_1 \le t$, or equivalently the penalized form
$\min_x \tfrac12\|Ax-y\|_2^2 + \lambda\|x\|_1$. The $\ell_1$ ball is a cross-polytope whose
vertices and low-dimensional faces sit *on the coordinate axes*; the level sets of the
data term, growing until they touch the ball, generically first touch at such a vertex,
where many coordinates are exactly zero. The $\ell_2$ ball, by contrast, is round and has
no such corners, which is precisely why ridge shrinks but never selects. The same
penalized objective is a standard workhorse for recovery: the $\ell_1$ term is
less sensitive than $\ell_2$ to the large outlier coefficients that, in imaging, encode
sharp edges.

**Solving the convex program at scale.** $(P_1)$ and its penalized cousin can be cast as a
second-order cone / linear program and handed to an interior-point method, whose Newton
steps form and factor systems involving $A$. At the application scale the variables number
in the millions and $A$ is dense (a blur convolution, a partial Fourier operator); the
operations that are cheap are applications of $A$ and $A^{\!\top}$ — matrix–vector products.
**First-order** methods are built from $\nabla\big(\tfrac12\|Ax-y\|_2^2\big) =
A^{\!\top}(Ax-y)$, which is exactly one $A$ followed by one $A^{\!\top}$.

**First-order pieces on the table.** Gradient descent, $x_{k+1} = x_k - t\nabla g(x_k)$,
minimizes a smooth convex $g$ at rate $g(x_k)-g^\star = O(1/k)$ and assumes
differentiability. For nonsmooth convex problems the subgradient method
converges at $O(1/\sqrt k)$. From convex analysis there is also
Moreau's *proximal mapping*
$\operatorname{prox}_h(v) = \arg\min_u\, h(u) + \tfrac12\|u-v\|_2^2$, the resolvent that
generalizes Euclidean projection (for an indicator function it *is* projection onto the
set). For *smooth* convex minimization Nesterov
(1983, *A method for solving the convex programming problem with convergence rate
$O(1/k^2)$*, Dokl. Akad. Nauk SSSR 269:543–547) showed that an extrapolation step
accelerates gradient descent
to $O(1/k^2)$, which Nemirovsky & Yudin (1983) had shown is the best possible rate for any
first-order method on this class. These are the first-order pieces available off the shelf.

## Baselines

- **Minimum-$\ell_2$ / pseudoinverse solution.** $\hat x = A^{\!\top}(AA^{\!\top})^{-1}y$,
  the least-norm exact solution. Core idea: pick the solution closest to the origin in
  Euclidean norm.

- **Tikhonov / ridge regularization.** $\min_x \|Ax-y\|_2^2 + \lambda\|Lx\|_2^2$. Core
  idea: a smooth quadratic penalty stabilizes an ill-posed inverse.

- **$\ell_0$ / combinatorial search ($P_0$).** $\min\|x\|_0$ s.t. $Ax=y$. Core idea: the
  exact statement of "sparsest consistent signal", NP-hard (Natarajan 1995), with
  exponentially many supports to test.

- **Greedy support selection (matching pursuit / OMP).** Iteratively add the column most
  correlated with the current residual and re-fit. Core idea: a cheap heuristic for $(P_0)$.

- **Basis pursuit / lasso via interior-point ($P_1$ as SOCP/LP).** $\min\|x\|_1$ s.t.
  $Ax=y$, or $\min\tfrac12\|Ax-y\|_2^2+\lambda\|x\|_1$, solved by a second-order
  interior-point method whose Newton steps form and factor systems involving $A$. Core idea:
  the convex relaxation, solved to high accuracy.

- **Plain (sub)gradient on the penalized objective.** Gradient descent handles the
  smooth term; the subgradient method handles the nonsmooth $\ell_1$ term at $O(1/\sqrt k)$.
  Core idea: stay first-order and cheap, using only $A$, $A^{\!\top}$ products.

## Evaluation settings

The natural proving ground is **linear inverse problems in imaging**, where the unknown is
sparse in a wavelet basis. A representative setup: a known image (e.g. the $256\times256$
*cameraman*) is blurred by a spatially invariant kernel (a $9\times9$ Gaussian, std. 4) and
corrupted by additive white Gaussian noise (std. $\sim 10^{-3}$); the operator is $A = RW$
with $R$ the blur matrix and $W$ an (inverse) wavelet transform, so $x$ holds the wavelet
coefficients and the objective is $\tfrac12\|Ax-y\|_2^2 + \lambda\|x\|_1$ with a small
$\lambda$ (e.g. $10^{-5}$–$10^{-4}$). Reflexive (Neumann) boundary conditions are used so
that the eigenvalues of $A^{\!\top}\!A$ — and hence the Lipschitz constant of the gradient —
are computable via the cosine transform. A second standard testbed is **synthetic sparse
recovery**: draw a random $A$, a $k$-sparse $x^\star$, set $y = Ax^\star$ (optionally with
noise), and attempt to recover $x^\star$. The figures of merit are the optimization
accuracy $F(x_k)-F^\star$ versus iteration count (and versus wall-clock, where each
iteration is on the order of one application each of $A$ and $A^{\!\top}$),
the support/reconstruction error against the planted $x^\star$, and how the required number
of measurements $m$ scales with the sparsity $k$.

## Code framework

The pieces that already exist: dense or operator-form $A$ with cheap $A$, $A^{\!\top}$
products; the smooth least-squares data term and its gradient; and a generic first-order
iteration loop. How to incorporate the sparsity prior into the iteration is the empty slot.

```python
import numpy as np

def apply_A(A, x):   return A @ x          # cheap matrix-vector product
def apply_AT(A, r):  return A.conj().T @ r # adjoint, also cheap

def smooth_value(A, y, x):                  # least-squares fit, 1/2 ||Ax-y||^2
    r = apply_A(A, x) - y
    return 0.5 * float(np.real(np.vdot(r, r)))

def grad_smooth(A, y, x):                   # gradient of the smooth data term
    return apply_AT(A, apply_A(A, x) - y)

def lipschitz_constant(A):
    # TODO: estimate the largest eigenvalue of A.conj().T @ A
    pass

def solve(A, y, lam, x0=None, step=None, n_iter=500):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    x = x0.copy()
    for k in range(n_iter):
        # TODO: design the per-iteration update that drives x toward a sparse
        #       minimizer of 1/2||Ax-y||^2 + lam*||x||_1 using only grad_smooth
        #       (i.e. A, A^T products)
        pass
    return x
```
