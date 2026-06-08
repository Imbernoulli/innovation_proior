# Context: cheap stochastic optimization with a linear rate for finite sums

## Research question

A great many machine-learning problems reduce to minimizing a finite average of smooth losses,

  min_w P(w),  P(w) = (1/n) Σ_{i=1}^n ψ_i(w),

where each ψ_i is a per-example loss (squared loss, logistic loss, possibly with an added
0.5λ‖w‖² regularizer), n is the number of training examples, and w ∈ R^d. The two standard
ways to attack this sit at opposite ends of a frustrating trade-off.

Full-batch gradient descent computes the exact gradient ∇P(w) = (1/n) Σ_i ∇ψ_i(w) every step.
When each ψ_i is L-smooth and P is γ-strongly convex (L ≥ γ > 0), a constant step size η < 1/L
gives **linear** (geometric) convergence: P(w_t) − P(w*) = O((1 − γ/L)^t). But every single
step costs n gradient evaluations, so reaching accuracy ε costs on the order of n·κ·ln(1/ε)
gradient evaluations, where κ = L/γ is the condition number — prohibitive when n is large.

Stochastic gradient descent samples one index i_t per step and moves along ∇ψ_{i_t}(w). The
per-step cost is now independent of n, which is why it dominates large-scale learning. But it
converges only **sublinearly**, O(1/k), and needs a decaying step size η_t = O(1/t) to do even
that. The precise question is: on a *finite* training set, can one method have SGD's cheap
O(1)-per-step iterations **and** gradient descent's linear convergence with a constant step
size — without paying a memory cost that rules out large or nonconvex models?

## Background

**Smoothness and strong convexity.** Each ψ_i is convex and L-smooth:
ψ_i(w) − ψ_i(w') − 0.5L‖w − w'‖² ≤ ∇ψ_i(w')^T(w − w'). P is γ-strongly convex:
P(w) − P(w') − 0.5γ‖w − w'‖² ≥ ∇P(w')^T(w − w'). These two inequalities are the load-bearing
analytic facts. Strong convexity gives the quadratic lower bound P(w) − P(w*) ≥ 0.5γ‖w − w*‖²
that converts function-value progress into iterate progress; smoothness gives the upper bound
that lets a gradient step provably decrease the value, and — crucially — lets one bound a
per-example gradient *difference* by a function-value *suboptimality*. For an L-smooth convex
ψ_i the descent lemma gives ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤
2L[ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*)]; averaged over i with ∇P(w*) = 0 this becomes
(1/n)Σ_i‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L[P(w) − P(w*)]. This is
the bridge that ties "how big are the per-sample gradients" to "how far from optimal are we".

**The variance floor of SGD.** The stochastic gradient is
unbiased, E_i[∇ψ_i(w)] = ∇P(w), but it carries variance. At the optimum the full gradient
vanishes, ∇P(w*) = 0, yet the individual gradients do **not**: in general ∇ψ_i(w*) ≠ 0, so
E_i‖∇ψ_i(w*)‖² =: σ² > 0. Feed that into the SGD recursion. With strong convexity the
one-step contraction reads, schematically, E‖w_t − w*‖² ≲ (1 − 2ηγ)‖w_{t−1} − w*‖² + η²σ²;
its fixed point has nonzero squared distance on the order of ησ²/γ. So with a constant step size
SGD does not converge to w* — it settles into a noise ball around it, with squared radius set by
ησ²/γ. The only way to shrink the ball to zero is η → 0, but η → 0 also kills the contraction factor,
which is exactly why one is forced onto η_t = O(1/t) and the resulting O(1/k) rate. The
behaviour is well documented: with a large fixed η, SGD's loss drops fast then oscillates above
the minimum forever; with a small η it crawls. This is a property of the estimator, not of
tuning — it is the variance, not the bias, that caps the rate.

**The unbiased-oracle lower bound.** This O(1/k) is not a failure of cleverness. In the model
where an algorithm only accesses P through *unbiased* measurements of its gradient, O(1/k) is
the minimax-optimal rate for strongly convex optimization (Nemirovski & Yudin 1983; Nemirovski,
Juditsky, Lan & Shapiro 2009; Agarwal et al. 2012). You cannot beat it with unbiased gradient
samples alone. The escape hatch is that on a *finite* sum the samples are **not** an endless
stream of fresh draws — there are only n of them, fixed, and they may be revisited. That extra
structure is information the lower bound does not assume, and it is what any linear-rate method
for finite sums must exploit.

**Control variates (classical variance reduction).** To estimate E[X] by Monte Carlo when one
has a second variable Y that is highly correlated with X *and whose mean E[Y] is known*, the
control-variate estimator is g = X − Y + E[Y]. It is unbiased for any such Y, E[g] = E[X], and
its variance is Var(g) = Var(X) + Var(Y) − 2Cov(X, Y), which is small precisely when X and Y
are strongly correlated. The leverage is entirely in finding a Y that tracks X and has a
computable mean. This is standard Monte Carlo machinery, predating its use in optimization.

## Baselines

**Full-batch gradient descent (Cauchy 1847; Nesterov 2004).** w_{k+1} = w_k − η∇P(w_k). Linear
convergence O(ρ^k), ρ < 1 depending on κ, with constant η for smooth strongly convex P. The
gap it leaves: iteration cost scales with n, so it is unappealing exactly in the large-n regime
that motivates stochastic methods.

**Stochastic gradient descent (Robbins & Monro 1951; Bottou).** w_{k+1} = w_k − η_k∇ψ_{i_k}(w_k).
Cost independent of n; sublinear O(1/k) with η_k = O(1/k). Pegasos (Shalev-Shwartz, Singer &
Srebro 2007) and stochastic linear-prediction solvers (Zhang 2004) made it the default for
large-scale learning. The gap: the variance floor above — a constant step cannot converge, and
the forced decay yields only sublinear progress, slow when an accurate solution is needed.

**Incremental Aggregated Gradient, IAG (Blatt, Hero & Gauchman 2007).** Keeps a table of the
most recent gradient computed for each i and steps along their average, updating one entry per
iteration. Deterministic cycling; achieves linear convergence under conditions but with a step
size that must be very small, and its analysis is delicate.

**Stochastic Average Gradient, SAG (Le Roux, Schmidt & Bach 2012).** A randomized IAG. It keeps
a memory y_i of the last gradient seen for each example and steps along the running average:
x_{k+1} = x_k − (α/n) Σ_i y_i, where on each step one random i_k is refreshed, y_{i_k} ←
∇ψ_{i_k}(x_k), and the rest are held. Per-iteration cost is one gradient, yet it attains a
**linear** rate on smooth strongly convex finite sums — the first method to combine SGD's cheap
step with GD's geometric rate. Two gaps. (1) It must store all n gradients (an n×d table);
for least squares that compresses to n scalars, but for general structured-prediction or
neural-network models the full table is impractical. (2) Its update direction is **biased**
(the stored average is not, in expectation, ∇P(x_k)), which makes the convergence proof
intricate and the constants loose.

**Stochastic Dual Coordinate Ascent, SDCA (Shalev-Shwartz & Zhang 2012; Hsieh et al. 2008 for
the SVM case).** For P(w) = (1/n)Σφ_i(w) + 0.5λ‖w‖², it ascends the dual one coordinate at a
time, maintaining dual variables α_i with w = Σ_i α_i. At the optimum α_i* = −(1/λn)∇φ_i(w*),
so the effective per-step direction ∇φ_i(w) + λnα_i vanishes as (w, α) → (w*, α*). This lets it
keep a constant-order step and converge linearly; it is the solver inside popular linear-SVM
packages. Gaps: it is tied to the regularized-ERM / dual structure, and like SAG it stores a
per-example quantity (the duals) — again impractical for complex nonconvex models.

The common thread among the linear-rate methods: each one carries a *per-example memory* (a
stored gradient, or a dual) that, near the optimum, cancels the offending offset and drives the
update's variance to zero — which is what permits a constant step size. The open gap is to get
that vanishing-variance effect **without** the per-example table, so it applies to models where
storing n gradients is out of the question.

## Evaluation settings

The natural yardstick is the training objective itself, P(w) − P(w*), the loss residual,
plotted against computational cost measured in gradient evaluations divided by n (i.e. effective
passes over the data) so that SGD's n cheap steps and a batch method's one expensive step are
compared fairly. A complementary quantity is the empirical variance of the per-step update,
which directly exposes whether a method's noise actually decays. Test error rate is tracked as
the downstream quantity of real interest. Standard datasets and tasks of the time: L2-regularized
multiclass and binary logistic regression on MNIST, rcv1.binary and covtype.binary (LIBSVM),
the protein dataset, and CIFAR-10; and, for the nonconvex regime, a small fully-connected neural
net (one hidden layer, sigmoid activations, softmax outputs, L2 regularization) on MNIST and
CIFAR-10. Step-size schedules for the SGD baseline are themselves part of the protocol:
fixed η, exponential decay η_0 a^{⌊t/n⌋}, and t-inverse η_0(1 + b⌊t/n⌋)^{−1}, each tuned.

## Code framework

A finite-sum training harness. The data and the per-example loss/gradient already exist; the
optimizer is the empty slot. The full-gradient and plain-SGD steps are the reference points.

```python
import numpy as np

class FiniteSumProblem:
    """min_w P(w) = (1/n) sum_i psi_i(w). Provides per-example and full gradients."""
    def __init__(self, X, y, reg):
        self.X, self.y, self.reg = X, y, reg
        self.n, self.d = X.shape

    def grad_i(self, w, i):
        """Gradient of psi_i at w. e.g. logistic: phi'(w^T x_i) x_i + reg*w."""
        raise NotImplementedError

    def full_grad(self, w):
        """(1/n) sum_i grad_i(w, i): one full pass over the data."""
        g = np.zeros(self.d)
        for i in range(self.n):
            g += self.grad_i(w, i)
        return g / self.n

    def value(self, w):
        raise NotImplementedError


def gd_step(problem, w, eta):
    # full-batch reference: linear rate, but n gradients per step
    return w - eta * problem.full_grad(w)


def sgd_step(problem, w, eta_t):
    # cheap O(1) reference: unbiased but high-variance; needs eta_t -> 0
    i = np.random.randint(problem.n)
    return w - eta_t * problem.grad_i(w, i)


def finite_sum_optimize(problem, w0, eta, m, n_outer):
    """Placeholder for a cheap finite-sum optimizer with constant-step inner updates."""
    w = w0.copy()
    for s in range(n_outer):
        # TODO: any outer-loop bookkeeping needed before cheap stochastic updates.
        # TODO: inner loop of m cheap updates.
        # TODO: choose the next outer iterate from the inner trajectory.
        pass
    return w
```
