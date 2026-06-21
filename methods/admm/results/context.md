# Context: Solving a large convex problem whose objective splits into two coupled blocks

## Research question

We are handed a convex optimization problem in which the objective separates into two pieces
tied together by a single linear constraint:

$$ \min_{x,z}\ f(x) + g(z) \qquad \text{s.t.}\qquad Ax + Bz = c, $$

with $x \in \mathbb{R}^n$, $z \in \mathbb{R}^m$, $A \in \mathbb{R}^{p\times n}$,
$B \in \mathbb{R}^{p\times m}$, $c \in \mathbb{R}^p$, and $f,g$ convex (allowed to be
nondifferentiable and to take the value $+\infty$, so that constraints can be folded into the
objective as indicator functions). This template is more general than it looks. A great many
problems of interest — regularized loss minimization $\ell(x)+\lambda\|x\|_1$ (a smooth
data term plus a separable penalty); a sum of $N$ data-block losses
$\sum_i f_i(x)$ that share one parameter vector; model fitting where a regularizer sits on a shared
variable — can be cast into this two-block form by an appropriate change of variables.

Two properties matter for the solution method. **Robustness:** the objectives we care about (an
$\ell_1$ norm, an indicator of a polytope, an affine loss) are often nondifferentiable, can take the
value $+\infty$, are not strictly convex, and the maps $A,B$ need not be full rank, so we want
convergence under such weak assumptions. **Decomposition:** the dataset and the variable are large,
often physically distributed across many machines, so the per-iteration work should split into
independent pieces — each block $x_i$, or each data partition, updated on its own processor, with
only a cheap gather/broadcast tying them together.

## Background

The setting is large-scale convex optimization driven by "big data": datasets too large or too
distributed to gather centrally, objectives that are sums over data blocks or over features, and
nonsmooth regularizers (the $\ell_1$ norm, total variation, indicators of constraint sets) that are
ubiquitous in statistics and signal processing. The relevant theory is Lagrangian duality and the
calculus of subgradients. Three classical algorithms supply the building blocks.

**Dual ascent.** For the equality-constrained problem $\min f(x)$ s.t. $Ax=b$, the Lagrangian is
$L(x,y)=f(x)+y^\top(Ax-b)$ and the dual function is
$g(y)=\inf_x L(x,y) = -f^*(-A^\top y)-b^\top y$, where $f^*$ is the convex conjugate. Maximizing the
dual by gradient ascent gives the iteration
$x^{k+1}=\arg\min_x L(x,y^k)$, $y^{k+1}=y^k+\alpha^k(Ax^{k+1}-b)$, because $\nabla g(y)=Ax^+ - b$ is
exactly the constraint residual at the inner minimizer. The $x$-minimization is well-defined when $f$
is strictly convex and finite; when the dual is nondifferentiable one uses the dual subgradient
method.

**Dual decomposition.** Dual ascent decentralizes when the objective is separable. If
$f(x)=\sum_{i=1}^N f_i(x_i)$ and $A=[A_1\;\cdots\;A_N]$ is partitioned conformably, then
$L(x,y)=\sum_i\big(f_i(x_i)+y^\top A_i x_i - \tfrac1N y^\top b\big)$ is separable in $x$, so the
$x$-minimization splits into $N$ independent subproblems
$x_i^{k+1}=\arg\min_{x_i}L_i(x_i,y^k)$ solved in parallel; each iteration is a broadcast of the price
$y$ and a gather of the residual contributions $A_i x_i^{k+1}$. This idea goes back to Everett and to
the large-scale linear-programming decompositions of Dantzig–Wolfe and Benders in the early 1960s.

**The method of multipliers (augmented Lagrangian).** Hestenes (1969) and Powell (1969) added a
quadratic penalty on the constraint residual to the Lagrangian:
$$ L_\rho(x,y) = f(x) + y^\top(Ax-b) + (\rho/2)\|Ax-b\|_2^2, \qquad \rho>0. $$
This is exactly the ordinary Lagrangian of the equivalent problem
$\min f(x)+(\rho/2)\|Ax-b\|^2$ s.t. $Ax=b$ (the added term vanishes on the feasible set). The
associated dual function $g_\rho(y)=\inf_x L_\rho(x,y)$ is differentiable under mild conditions even
when the original dual was not. Dual ascent on the modified problem is the **method of multipliers**:
$$ x^{k+1}=\arg\min_x L_\rho(x,y^k), \qquad y^{k+1}=y^k+\rho(Ax^{k+1}-b), $$
where the dual step size is taken to be the penalty parameter $\rho$ itself. That particular step
size is not arbitrary: since $x^{k+1}$ minimizes $L_\rho(\cdot,y^k)$, stationarity gives
$0=\nabla f(x^{k+1})+A^\top(y^k+\rho(Ax^{k+1}-b))=\nabla f(x^{k+1})+A^\top y^{k+1}$, so each iterate is
automatically *dual feasible*, and as the primal residual $Ax^{k+1}-b$ drives to zero the pair
becomes optimal. The method of multipliers converges under far weaker conditions than dual ascent —
including when $f$ is nonsmooth, takes the value $+\infty$, or is not strictly convex. With $f$
separable, the penalty $(\rho/2)\|\sum_i A_i x_i - b\|^2$ contains cross terms
$\rho\,(A_i x_i)^\top(A_j x_j)$, so $L_\rho$ is not separable in $x$.

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
instances. This operator-splitting view is a second body of convergence theory standing alongside
the augmented-Lagrangian one.

## Baselines

- **Dual ascent.** $x^{k+1}=\arg\min_x L(x,y^k)$, $y^{k+1}=y^k+\alpha^k(Ax^{k+1}-b)$ with
  $L=f(x)+y^\top(Ax-b)$. Core idea: gradient-ascend the dual, recovering the primal at the inner
  $\arg\min$; the inner minimization is well-defined for strictly convex finite $f$.

- **Dual decomposition.** Dual ascent with $f=\sum_i f_i(x_i)$, $A=[A_1\cdots A_N]$, so the
  $x$-update splits into $N$ parallel subproblems with a price broadcast and residual gather. Core
  idea: exploit separability to decentralize.

- **Method of multipliers (augmented Lagrangian).** $x^{k+1}=\arg\min_x L_\rho(x,y^k)$,
  $y^{k+1}=y^k+\rho(Ax^{k+1}-b)$ with $L_\rho=f(x)+y^\top(Ax-b)+(\rho/2)\|Ax-b\|^2$. Core idea: a
  quadratic penalty makes the dual differentiable and yields convergence without strict
  convexity/finiteness; using $\rho$ as the dual step keeps every iterate dual feasible.

- **Douglas–Rachford / proximal-point splitting.** Find a zero of $S+T$ (maximal monotone) by
  alternating resolvents $(I+\lambda S)^{-1},(I+\lambda T)^{-1}$ (Lions–Mercier 1979; Rockafellar's
  proximal point algorithm 1976). Core idea: never touch $S+T$ jointly — split into per-operator prox
  steps.

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
are the objective value and the constraint-residual norms versus iteration count, the number of
iterations to reach a modest accuracy, and how the wall-clock and communication scale with the
number of workers.

## Code framework

What already exists: routines to apply the blocks $A,B$ and their adjoints; the two objective
pieces $f,g$ and whatever simple sub-solver each admits; vector norms, factorizations, and a generic
driver that can repeat a candidate update rule. The open slot is the iteration that uses those pieces
to solve $\min f(x)+g(z)$ s.t. $Ax+Bz=c$.

```python
import numpy as np


def solve_split_problem(A, B, c, n, m, p, rho=1.0,
                        n_iter=1000, abstol=1e-4, reltol=1e-2):
    # TODO: fill in the iteration that solves  min f(x) + g(z)  s.t. A x + B z = c
    pass


def lasso(C, b, lam, rho=1.0, alpha=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    # TODO: instantiate it on  min 1/2||C x - b||^2 + lam||x||_1
    pass
```
