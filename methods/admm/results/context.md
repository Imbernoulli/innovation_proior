# Context: Solving a large convex problem whose objective splits into two coupled blocks

## Research question

We are handed a convex optimization problem in which the objective separates into two pieces
tied together by a single linear constraint:

$$ \min_{x,z}\ f(x) + g(z) \qquad \text{s.t.}\qquad Ax + Bz = c, $$

with $x \in \mathbb{R}^n$, $z \in \mathbb{R}^m$, $A \in \mathbb{R}^{p\times n}$,
$B \in \mathbb{R}^{p\times m}$, $c \in \mathbb{R}^p$, and $f,g$ convex (allowed to be
nondifferentiable and to take the value $+\infty$, so that constraints can be folded into the
objective as indicator functions). This template is more general than it looks. A great many
problems of interest — regularized loss minimization $\ell(x)+\lambda\|x\|_1$ (split into a smooth
data term and a separable penalty by writing $x=z$); a sum of $N$ data-block losses
$\sum_i f_i(x)$ that share one parameter vector (split by giving each block a private copy
$x_i$ and constraining $x_i=z$); model fitting where a regularizer sits on the shared variable —
all fall into this two-block form once the variable is duplicated.

A solution method has to clear two bars **at the same time**, and the difficulty is that the two
bars are in tension. **Robustness:** the method must converge under weak assumptions — no strict
convexity, no finiteness of $f$, no full rank of $A$ or $B$ — because the objectives we care about
(an $\ell_1$ norm, an indicator of a polytope, an affine loss) routinely violate all of those.
**Decomposition:** the dataset and the variable are large, often physically distributed across many
machines, so the per-iteration work must split into independent pieces — each block $x_i$, or each
data partition, updated on its own processor, with only a cheap gather/broadcast tying them
together. The pain is that the existing tool that gives robustness destroys decomposition, and the
existing tool that gives decomposition is not robust. We want both.

## Background

The setting is large-scale convex optimization driven by "big data": datasets too large or too
distributed to gather centrally, objectives that are sums over data blocks or over features, and
nonsmooth regularizers (the $\ell_1$ norm, total variation, indicators of constraint sets) that are
ubiquitous in statistics and signal processing. The relevant theory is Lagrangian duality and the
calculus of subgradients; the relevant pain points are inherited from three classical algorithms.

**Dual ascent.** For the equality-constrained problem $\min f(x)$ s.t. $Ax=b$, the Lagrangian is
$L(x,y)=f(x)+y^\top(Ax-b)$ and the dual function is
$g(y)=\inf_x L(x,y) = -f^*(-A^\top y)-b^\top y$, where $f^*$ is the convex conjugate. Maximizing the
dual by gradient ascent gives the iteration
$x^{k+1}=\arg\min_x L(x,y^k)$, $y^{k+1}=y^k+\alpha^k(Ax^{k+1}-b)$, because $\nabla g(y)=Ax^+ - b$ is
exactly the constraint residual at the inner minimizer. It is elegant but **fragile**: the
$x$-minimization is only well-defined when $f$ is strictly convex and finite. The cleanest way to
see the failure: if $f$ is a nonzero affine function of any component of $x$, then for most $y$ the
Lagrangian is unbounded below in $x$ and the $x$-update has no solution. When the dual is
nondifferentiable one must fall back on the dual subgradient method, whose step sizes are delicate
and whose dual ascent is nonmonotone.

**Dual decomposition.** Dual ascent's one great virtue is that it can decentralize. If the objective
is separable, $f(x)=\sum_{i=1}^N f_i(x_i)$, and $A=[A_1\;\cdots\;A_N]$ is partitioned conformably,
then $L(x,y)=\sum_i\big(f_i(x_i)+y^\top A_i x_i - \tfrac1N y^\top b\big)$ is separable in $x$, so the
$x$-minimization splits into $N$ independent subproblems
$x_i^{k+1}=\arg\min_{x_i}L_i(x_i,y^k)$ solved in parallel; each iteration is a broadcast of the price
$y$ and a gather of the residual contributions $A_i x_i^{k+1}$. This idea goes back to Everett and to
the large-scale linear-programming decompositions of Dantzig–Wolfe and Benders in the early 1960s.
But it inherits dual ascent's fragility wholesale: the same strict-convexity/finiteness requirements
that break the $x$-update are still in force.

**The method of multipliers (augmented Lagrangian).** Robustness was bought, in the late 1960s, by
Hestenes (1969) and Powell (1969), who added a quadratic penalty on the constraint residual to the
Lagrangian:
$$ L_\rho(x,y) = f(x) + y^\top(Ax-b) + (\rho/2)\|Ax-b\|_2^2, \qquad \rho>0. $$
This is exactly the ordinary Lagrangian of the equivalent problem
$\min f(x)+(\rho/2)\|Ax-b\|^2$ s.t. $Ax=b$ (the added term vanishes on the feasible set). The point
of the penalty is that the associated dual function $g_\rho(y)=\inf_x L_\rho(x,y)$ is differentiable
under mild conditions even when the original dual was not. Dual ascent on the modified problem is the
**method of multipliers**:
$$ x^{k+1}=\arg\min_x L_\rho(x,y^k), \qquad y^{k+1}=y^k+\rho(Ax^{k+1}-b), $$
where the dual step size is taken to be the penalty parameter $\rho$ itself. That particular step
size is not arbitrary: since $x^{k+1}$ minimizes $L_\rho(\cdot,y^k)$, stationarity gives
$0=\nabla f(x^{k+1})+A^\top(y^k+\rho(Ax^{k+1}-b))=\nabla f(x^{k+1})+A^\top y^{k+1}$, so each iterate is
automatically *dual feasible*, and as the primal residual $Ax^{k+1}-b$ drives to zero the pair
becomes optimal. The method of multipliers converges under far weaker conditions than dual ascent —
including when $f$ is nonsmooth, takes the value $+\infty$, or is not strictly convex. The penalty
bought robustness. The diagnostic failure that this whole development runs into, and that the next
method must repair, is that the **quadratic penalty couples the variables**: with $f$ separable,
$(\rho/2)\|\sum_i A_i x_i - b\|^2$ contains cross terms $\rho\,(A_i x_i)^\top(A_j x_j)$, so $L_\rho$
is *not* separable, the joint $x$-minimization can no longer be split across processors, and
decomposition is lost. Robustness and decomposition cannot both be had from these classical pieces.

**Operator-splitting ancestors.** Sitting underneath all of this is a second, more abstract line of
work: solving monotone-operator problems by *splitting*. Many convex problems reduce to finding a
zero of a sum of two maximal monotone operators $S+T$ (for a closed proper convex $h$, the
subdifferential $\partial h$ is maximal monotone, and a zero of $\partial h$ is a minimizer of $h$).
Douglas and Rachford (1956), originally for discretized heat equations, and then Lions and Mercier
(1979) for general monotone operators, gave a splitting iteration that alternates the *resolvents*
$(I+\lambda S)^{-1}$, $(I+\lambda T)^{-1}$ of the two operators rather than handling $S+T$ at once.
The resolvent of a subdifferential is exactly Moreau's **proximal operator**
$\operatorname{prox}_{h,\rho}(v)=\arg\min_x h(x)+(\rho/2)\|x-v\|_2^2$, which generalizes Euclidean
projection (for the indicator of a convex set it *is* the projection). Rockafellar's proximal point
algorithm (1976) is the umbrella: it finds a zero of a maximal monotone operator by repeated
resolvent steps, and both the method of multipliers and Douglas–Rachford splitting turn out to be
instances. This operator-splitting view is the source of the convergence machinery the eventual
two-block method will lean on.

## Baselines

- **Dual ascent.** $x^{k+1}=\arg\min_x L(x,y^k)$, $y^{k+1}=y^k+\alpha^k(Ax^{k+1}-b)$ with
  $L=f(x)+y^\top(Ax-b)$. Core idea: gradient-ascend the dual, recovering the primal at the inner
  $\arg\min$. Gap: the inner minimization needs strict convexity/finiteness of $f$ — it simply fails
  (unbounded below) for affine or nonstrictly-convex $f$; step sizes are delicate and ascent
  nonmonotone in the nondifferentiable case.

- **Dual decomposition.** Dual ascent with $f=\sum_i f_i(x_i)$, $A=[A_1\cdots A_N]$, so the
  $x$-update splits into $N$ parallel subproblems with a price broadcast and residual gather. Core
  idea: exploit separability to decentralize. Gap: inherits dual ascent's fragility unchanged — no
  robustness to nonsmooth/affine/non-strictly-convex $f$.

- **Method of multipliers (augmented Lagrangian).** $x^{k+1}=\arg\min_x L_\rho(x,y^k)$,
  $y^{k+1}=y^k+\rho(Ax^{k+1}-b)$ with $L_\rho=f(x)+y^\top(Ax-b)+(\rho/2)\|Ax-b\|^2$. Core idea: a
  quadratic penalty makes the dual differentiable and yields convergence without strict
  convexity/finiteness; using $\rho$ as the dual step keeps every iterate dual feasible. Gap: the
  penalty's cross terms couple a separable $f$, so $L_\rho$ is not separable and the $x$-update can
  no longer be decomposed across processors — robustness at the cost of parallelism.

- **Douglas–Rachford / proximal-point splitting.** Find a zero of $S+T$ (maximal monotone) by
  alternating resolvents $(I+\lambda S)^{-1},(I+\lambda T)^{-1}$ (Lions–Mercier 1979; Rockafellar's
  proximal point algorithm 1976). Core idea: never touch $S+T$ jointly — split into per-operator prox
  steps. Gap (at this point): stated abstractly for operators, not yet connected to the concrete
  two-block constrained optimization template or to a distributed implementation; it is the
  convergence theory waiting for the algorithm that instantiates it.

## Evaluation settings

The natural proving grounds are large-scale statistical and machine-learning fits where the
objective is a sum of a loss and a nonsmooth regularizer, or a sum over data blocks. The canonical
small testbed is the **lasso**, $\min \tfrac12\|Cx-b\|_2^2 + \lambda\|x\|_1$, on synthetic data: draw
a wide design $C\in\mathbb{R}^{q\times d}$ with $d\gg q$, plant a sparse $x^\star$, set
$b=Cx^\star+\text{noise}$, choose $\lambda$ as a fraction of $\lambda_{\max}=\|C^\top b\|_\infty$
(above which the solution is zero). Related $\ell_1$ testbeds: basis pursuit
($\min\|x\|_1$ s.t. $Cx=b$), $\ell_1$-regularized logistic regression, total-variation denoising,
group lasso, and sparse inverse covariance selection. The distributed testbeds are **consensus**
problems — $\min\sum_i f_i(x)$ split over data partitions, each $f_i$ a block loss, fit on its own
worker — and **sharing/exchange** problems where agents trade a shared resource. Figures of merit
are the objective value and the primal/dual residual norms versus iteration count, the number of
iterations to reach a modest accuracy, and how the wall-clock and communication scale with the
number of workers; each iteration costs one (often cached/factored) block solve plus a coordinatewise
or projection step plus a gather/broadcast.

## Code framework

What already exists: routines to apply the blocks $A,B$ and their adjoints; the two objective
pieces $f,g$ and whatever simple sub-solver each admits (a cached factorization for a quadratic $f$,
a closed-form prox for a separable $g$); a generic iteration loop holding a primal block, a coupled
block, and a dual vector. The block updates, the dual update, and the residual-based stopping test
are the empty slots.

```python
import numpy as np


def x_update(z, u, rho):
    # TODO: minimize f(x) + (rho/2) || A x + B z - c + u ||^2 over x
    pass


def z_update(x, u, rho):
    # TODO: minimize g(z) + (rho/2) || A x + B z - c + u ||^2 over z
    pass


def prox_g(v, rho):
    # TODO: the simple per-block solve for g (closed form when g is separable / an indicator)
    pass


def primal_residual(x, z):
    # TODO: the equality-constraint residual A x + B z - c
    pass


def dual_residual(z, z_prev, rho):
    # TODO: the residual of the second optimality condition, from the change in z
    pass


def solve(rho=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    x = None  # primal block
    z = None  # coupled block
    u = None  # (scaled) dual vector
    for k in range(n_iter):
        z_prev = z
        x = x_update(z, u, rho)
        z = z_update(x, u, rho)
        # TODO: dual update from the current primal residual
        # TODO: form primal and dual residuals; stop when both are below their tolerances
        pass
    return x, z
```
