# Context

## Research question

I want to minimize a convex function $f:\mathbb{R}^n\to\mathbb{R}$ whose gradient is Lipschitz — the *smooth convex* class. Concretely, $f$ is $\beta$-smooth: $\|\nabla f(x)-\nabla f(y)\|\le \beta\|x-y\|$ for all $x,y$, and there is a minimizer $x^*$ with $f^*=f(x^*)$. I am restricted to a *first-order black-box* oracle: at any queried point $x$ I may read $f(x)$ and $\nabla f(x)$, nothing else (no Hessian, no global structure), and I build the next query from the history of points and gradients seen so far. This is the setting of essentially every large-scale problem in statistics and machine learning — regularized least squares, logistic regression, SVMs, matrix completion — where the dimension $n$ is large enough that anything beyond gradients is unaffordable.

The precise question is two-sided. **(1) How fast can the optimality gap $f(x_k)-f^*$ be driven to zero as a function of the number of oracle calls $k$, for the worst function in the class?** Is there a fundamental floor — a rate no first-order method can beat — and what is it? **(2) Is that floor attainable by an explicit, cheap, gradient-only rule?** A solution must use *only* the information a first-order oracle returns, cost a constant number of gradient evaluations per step, and provably match the floor — not on nice quadratics, but on every smooth convex function. The pain is that the obvious method, gradient descent, only gives $O(1/k)$ in the smooth convex case, while the natural fix (adding momentum) works on quadratics but loses its guarantee on general convex functions.

## Background

The objects are convex sets and convex functions. $f$ is convex iff it lies below its chords, equivalently iff it lies above each of its tangent (sub)gradient lines: $f(x)-f(y)\le \nabla f(x)^\top(x-y)$. The minimizer is characterized by $\nabla f(x^*)=0$.

**Smoothness as a quadratic sandwich.** $\beta$-smoothness is equivalent to the inequality
$$0\le f(x)-f(y)-\nabla f(y)^\top(x-y)\le \tfrac{\beta}{2}\|x-y\|^2,$$
i.e. at every $y$ the function is trapped under the quadratic $q_y^+(x)=f(y)+\nabla f(y)^\top(x-y)+\tfrac{\beta}{2}\|x-y\|^2$. Minimizing this upper model over $x$ gives the gradient step and the central one-step inequality
$$f\!\Big(x-\tfrac{1}{\beta}\nabla f(x)\Big)-f(x)\le -\tfrac{1}{2\beta}\|\nabla f(x)\|^2,$$
which says a gradient step buys at least $\tfrac{1}{2\beta}\|\nabla f(x)\|^2$ of descent.

**Strong convexity as the dual quadratic sandwich.** $f$ is $\alpha$-strongly convex if $f(x)-f(y)\le \nabla f(x)^\top(x-y)-\tfrac{\alpha}{2}\|x-y\|^2$, equivalently $x\mapsto f(x)-\tfrac{\alpha}{2}\|x\|^2$ is convex; now each point sits *above* a quadratic lower model $q_x^-(y)=f(x)+\nabla f(x)^\top(y-x)+\tfrac{\alpha}{2}\|y-x\|^2$. Smoothness ($\beta$) and strong convexity ($\alpha$) are dual curvature bounds with $\beta\ge\alpha$; their ratio $\kappa=\beta/\alpha$ is the *condition number*.

**The diagnostic fact about gradient descent.** With the descent inequality above, gradient descent on a $\beta$-smooth convex $f$ provably converges, but only at $f(x_k)-f^*=O(1/k)$ — and on a $\beta$-smooth, $\alpha$-strongly convex $f$ it converges linearly but at a rate governed by $\kappa$ (oracle complexity $O(\kappa\log\frac1\varepsilon)$). These are *upper* bounds the field has had in hand. The open, load-bearing question is whether they are tight: the prevailing intuition is that gradient descent, being the canonical first-order method, ought to be close to optimal — but no matching lower bound pins it down, and there are hints (the conjugate-gradient method, momentum on quadratics) that one can do better.

**Momentum on quadratics works.** It is known empirically and analytically that on a strongly convex quadratic, adding a momentum/inertia term accelerates convergence to a rate set by $\sqrt\kappa$ rather than $\kappa$ — a quadratic improvement. This is the tantalizing evidence that $O(\kappa)$ is *not* the floor.

## Baselines

**Gradient descent on smooth convex $f$ (the workhorse).** Iterate $x_{t+1}=x_t-\tfrac{1}{\beta}\nabla f(x_t)$. Using the one-step descent and convexity, with $\delta_s=f(x_s)-f^*$ one gets $\delta_{s+1}\le \delta_s-\tfrac{1}{2\beta\|x_1-x^*\|^2}\delta_s^2$ (the distance to $x^*$ is non-increasing), which telescopes via $\tfrac{1}{\delta_{t}}-\tfrac{1}{\delta_1}\ge \omega(t-1)$ to
$$f(x_t)-f^*\le \frac{2\beta\|x_1-x^*\|^2}{t-1}=O(1/t).$$
**Gap:** the step size is the smallest the upper model allows; far from the optimum the method crawls, and the $1/t$ rate means reaching accuracy $\varepsilon$ costs $\Theta(1/\varepsilon)$ gradients. Nothing in this analysis suggests it is best possible, but nothing improves it either.

**Gradient descent on smooth strongly convex $f$.** With $\eta=\tfrac1\beta$, strong convexity upgrades the contraction to $\|x_{t+1}-x^*\|^2\le(1-\tfrac{\alpha}{\beta})\|x_t-x^*\|^2$, giving $\|x_{t+1}-x^*\|^2\le e^{-t/\kappa}\|x_1-x^*\|^2$ — a *linear* rate, but with exponent $1/\kappa$. A slightly larger step $\eta=\tfrac{2}{\alpha+\beta}$ improves the constant to $\big(\tfrac{\kappa-1}{\kappa+1}\big)^2$ per step, i.e. exponent $\approx 4/\kappa$. **Gap:** the dependence on $\kappa$ is linear; for ill-conditioned problems ($\kappa$ as large as the sample size in regularized ML), this is slow, and the momentum-on-quadratics evidence says $\sqrt\kappa$ should be reachable.

**Polyak's heavy-ball method (momentum, 1964).** Add inertia: $x_{k+1}=x_k-\alpha\nabla f(x_k)+\beta_{\!H}(x_k-x_{k-1})$ — a discrete heavy ball rolling through the landscape, the momentum term carrying it across narrow valleys instead of zig-zagging. On a strongly convex *quadratic*, tuning $\alpha,\beta_{\!H}$ from the Hessian's eigenvalues yields the optimal local rate with $\beta_{\!H}=\big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2$ — the coveted $\sqrt\kappa$ acceleration. **Gap:** the momentum is applied at, and the gradient is read at, the *same* current point $x_k$; the analysis is a spectral (eigenvalue) argument that lives on quadratics. On a general (non-quadratic) smooth convex $f$ the method carries no global convergence guarantee and can fail to converge (it can settle into a limit cycle). So acceleration is, at this point, a quadratic-only phenomenon.

**Conjugate gradient / Nemirovski–Yudin optimal schemes.** On quadratics, conjugate gradient attains the optimum in $n$ steps and exhibits $\sqrt\kappa$-type behavior; earlier optimal-complexity schemes for the smooth class were known but required richer oracles — a *plane-search* or at least a *line-search* oracle, not pure first-order gradient information. **Gap:** these either are special to quadratics or assume an oracle stronger than "one gradient per step", so they do not answer whether a *plain* first-order method can be optimal on the whole smooth convex class.

## Evaluation settings

The natural yardstick is *oracle complexity*: for a function class (smooth convex; smooth $\alpha$-strongly convex) and a black-box first-order procedure, count the number of gradient/function evaluations needed to guarantee $f(x_k)-f^*\le\varepsilon$ in the worst case over the class. Two regimes matter: the convex case (rate as a power of $1/k$, complexity as a power of $1/\varepsilon$) and the strongly convex case (linear rate $e^{-ck}$, complexity $\propto\log\frac1\varepsilon$ with the constant $c$ the prize). The worst-case is taken over the whole class, with the geometry fixed by $\beta$, $\alpha$, and the initial distance $R=\|x_1-x^*\|$. The standard test problems that make these bounds bite are quadratics $f(x)=\tfrac12 x^\top A x-b^\top x$ with $0\preceq A\preceq\beta I$ (and $\succeq\alpha I$ in the strongly convex case), particularly tridiagonally-coupled $A$, because the coupling controls how fast information about the optimum can propagate through coordinates under a gradient-span-restricted method.

## Code framework

The available computational pieces are the oracle, the curvature constants, a plain gradient step, a line-search routine, and one unresolved iterate-combination rule.

```python
import numpy as np

class Objective:
    """Smooth (and optionally strongly) convex objective with a first-order oracle."""
    def __init__(self, beta, mu=0.0):
        self.beta = beta          # gradient Lipschitz constant (smoothness)
        self.mu = mu              # strong-convexity parameter (0 if merely convex)
    def value(self, x):      raise NotImplementedError
    def grad(self, x):       raise NotImplementedError

def gradient_step(obj, x, step=None):
    """One plain gradient-descent step x -> x - (1/beta) grad f(x)."""
    if step is None:
        step = 1.0 / obj.beta
    return x - step * obj.grad(x)

def backtracking_step(obj, x, alpha_prev):
    """Find a step that satisfies the smoothness descent inequality (line search stub)."""
    # TODO: shrink alpha from alpha_prev until
    #       f(x) - f(x - alpha*grad) >= 0.5*alpha*||grad||^2
    pass

class FirstOrderMethod:
    """Iterative first-order minimizer. The update rule is the open slot."""
    def __init__(self, obj, x0):
        self.obj = obj
        self.x = x0
        # TODO: any auxiliary sequences / coefficients a faster rule needs
    def step(self):
        # TODO: the iterate-combination update we are trying to discover --
        #       plain gradient descent is the trivial fill; can the rate improve?
        pass

def run(obj, x0, n_iters):
    method = FirstOrderMethod(obj, x0)
    for _ in range(n_iters):
        method.step()
    return method.x
```
