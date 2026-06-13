# Context: Recovering a sparse signal from far too few linear measurements

## Research question

We are handed a linear measurement model

$$ y = A x + w, \qquad A \in \mathbb{R}^{m\times n}, \quad m \ll n, $$

where $x \in \mathbb{R}^n$ is the unknown signal (or its coefficients in some fixed
basis), $A$ is a known measurement matrix, $y \in \mathbb{R}^m$ are the observations,
and $w$ is noise. The system is **underdetermined**: there are fewer equations than
unknowns, so $\{x : Ax = y\}$ is an affine subspace of dimension at least $n-m$ — an
infinitude of exact solutions, and classical linear algebra has no reason to prefer any
one of them. Yet in application after application the signal of interest is **sparse** —
only a handful $k \ll n$ of its entries are nonzero — or sparse after a known transform
(a natural image is sparse in a wavelet basis; a sum of a few sinusoids is sparse in
Fourier). The question is whether sparsity is enough side information to pin down the
*right* $x$ from $m \ll n$ measurements, and — crucially — whether we can compute it.

A solution has to clear two bars at once. First, **identifiability**: under what
condition on $A$ and on the sparsity level does the sparse signal become the *unique*
sparse member of the solution set, so that asking for "the sparsest $x$ consistent with
$y$" is well posed? Second, **tractability**: the sparsest-solution problem is, on its
face, a combinatorial search over which entries are nonzero, and the application sizes
($n$ in the millions, $A$ dense) rule out anything that scales worse than a few
matrix–vector products per step. A method that recovers the signal but needs to enumerate
supports, or that needs to factor or even form $A^{\!\top}\!A$, is useless at scale.

## Background

**Why the least-squares reflex fails.** The textbook response to an underdetermined
system is the minimum-norm solution $\hat x = A^{\!\top}(AA^{\!\top})^{-1}y$, i.e. the
$\ell_2$-smallest $x$ with $Ax=y$. It spreads energy across *all* coordinates — it is the
least sparse reasonable answer — and when $A$ is ill-conditioned its norm blows up and it
becomes meaningless. Tikhonov regularization, $\min_x \|Ax-y\|_2^2 + \lambda\|Lx\|_2^2$,
stabilizes the norm but, being a smooth quadratic penalty, still only *shrinks*
coefficients toward zero; it never sets them exactly to zero, so it cannot produce a
sparse estimate.

**Sparsity as a count, and why that is hard.** The honest formalization of "find the
sparsest consistent signal" uses the $\ell_0$ "norm" $\|x\|_0 = \#\{i : x_i \neq 0\}$:

$$ (P_0)\qquad \min_x \|x\|_0 \quad\text{s.t.}\quad Ax = y \quad(\text{or } \|Ax-y\|_2\le\varepsilon). $$

$\|\cdot\|_0$ is not a norm and is nonconvex; $(P_0)$ is a combinatorial problem. To solve
it exactly one would, in effect, run over every candidate support $T \subseteq \{1,\dots,n\}$
and test whether a consistent signal lives on it — exponentially many subsets. Natarajan
(1995, *Sparse Approximate Solutions to Linear Systems*, SIAM J. Comput. 24:227–234)
proved this is **NP-hard**: given $A$, $b$, $\varepsilon$, computing the fewest-nonzero $x$
with $\|Ax-b\|_2\le\varepsilon$ is intractable in the worst case. So $(P_0)$ states exactly
what we want and is exactly what we cannot afford to solve directly.

**The empirical surprise that reframed the field.** Replacing the count by the
$\ell_1$ norm $\|x\|_1 = \sum_i |x_i|$ turns the combinatorial problem into a convex one:

$$ (P_1)\qquad \min_x \|x\|_1 \quad\text{s.t.}\quad Ax = y, $$

a linear program (basis pursuit; Chen, Donoho & Saunders 1998, *Atomic decomposition by
basis pursuit*, SIAM J. Sci. Comput. 20:33–61). The observed, and at first startling,
phenomenon is that the $\ell_1$ solution is frequently *identical* to the $\ell_0$ solution.
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
penalized objective is the noise-robust workhorse for recovery: the $\ell_1$ term is also
less sensitive than $\ell_2$ to the large outlier coefficients that, in imaging, encode
sharp edges.

**Solving the convex program at scale.** $(P_1)$ and its penalized cousin can be cast as a
second-order cone / linear program and handed to an interior-point method. That is
theoretically fine but practically wrong here: the problems are huge (millions of
variables) and $A$ is *dense* (a blur convolution, a partial Fourier operator), so forming
or factoring the Newton system is hopeless. The only operations one can afford are
applications of $A$ and $A^{\!\top}$ — cheap matrix–vector products. This pushes the whole
solution toward **first-order** methods built from $\nabla\big(\tfrac12\|Ax-y\|_2^2\big) =
A^{\!\top}(Ax-y)$ alone.

**First-order pieces on the table.** Gradient descent, $x_{k+1} = x_k - t\nabla g(x_k)$,
minimizes a smooth convex $g$ at rate $g(x_k)-g^\star = O(1/k)$ — but it needs
differentiability, and our objective has the nondifferentiable $\ell_1$ term. The classical
escape for nonsmooth convex problems, the subgradient method, converges only at $O(1/\sqrt
k)$ and has no coordinatewise dead-zone that deliberately snaps small entries to zero. Also
on the shelf from convex analysis is
Moreau's *proximal mapping*
$\operatorname{prox}_h(v) = \arg\min_u\, h(u) + \tfrac12\|u-v\|_2^2$, the resolvent that
generalizes Euclidean projection (for an indicator function it *is* projection onto the
set). Proximal mappings are firmly nonexpansive, so
iterations built from them are stable. Finally, for *smooth* convex minimization Nesterov
(1983, *A method for solving the convex programming problem with convergence rate
$O(1/k^2)$*, Dokl. Akad. Nauk SSSR 269:543–547) showed that an extrapolation step — one
gradient evaluation plus a cheaply computed look-ahead point — accelerates gradient descent
to $O(1/k^2)$, which Nemirovsky & Yudin (1983) had shown is the best possible rate for any
first-order method on this class. These are the first-order pieces available off the shelf;
how they bear on the sparse-recovery objective is not settled here.

## Baselines

- **Minimum-$\ell_2$ / pseudoinverse solution.** $\hat x = A^{\!\top}(AA^{\!\top})^{-1}y$,
  the least-norm exact solution. Core idea: pick the solution closest to the origin in
  Euclidean norm. Gap: it is the *anti-sparse* answer (energy spread over all $n$
  coordinates) and explodes for ill-conditioned $A$; it ignores the sparsity prior entirely.

- **Tikhonov / ridge regularization.** $\min_x \|Ax-y\|_2^2 + \lambda\|Lx\|_2^2$. Core
  idea: a smooth quadratic penalty stabilizes an ill-posed inverse. Gap: a differentiable
  $\ell_2$ penalty shrinks but never zeroes coefficients, so the estimate is dense; it
  regularizes the *norm*, not the *support*.

- **$\ell_0$ / combinatorial search ($P_0$).** $\min\|x\|_0$ s.t. $Ax=y$. Core idea: the
  exact statement of "sparsest consistent signal". Gap: NP-hard (Natarajan 1995),
  exponentially many supports to test; infeasible beyond toy sizes.

- **Greedy support selection (matching pursuit / OMP).** Iteratively add the column most
  correlated with the current residual and re-fit. Core idea: a cheap heuristic for $(P_0)$.
  Gap: greedy, can pick the wrong atom early with no recourse, and lacks the global recovery
  guarantees that the convex relaxation enjoys.

- **Basis pursuit / lasso via interior-point ($P_1$ as SOCP/LP).** $\min\|x\|_1$ s.t.
  $Ax=y$, or $\min\tfrac12\|Ax-y\|_2^2+\lambda\|x\|_1$, solved by a second-order
  interior-point method. Core idea: the convex relaxation, solved to high accuracy. Gap:
  Newton steps require forming/factoring systems involving $A$ — impossible for the large,
  dense $A$ of imaging and compressed sensing; does not exploit that only $A$, $A^{\!\top}$
  products are cheap.

- **Plain (sub)gradient on the penalized objective.** Gradient descent ignores the
  nonsmooth term; the subgradient method handles it but at $O(1/\sqrt k)$ and without a
  coordinatewise dead-zone. Core idea: stay first-order and cheap. Gap: too slow, and no
  clean mechanism for producing exact zeros.

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
accuracy $F(x_k)-F^\star$ versus iteration count (and versus wall-clock, since each
iteration is one application each of $A$ and $A^{\!\top}$ plus a coordinatewise shrinkage),
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
