## Research question

A great many learning problems reduce to minimizing a finite average of per-example losses,

```
min_w  P(w) = (1/n) Σ_{i=1}^n ψ_i(w),
```

where each ψ_i is the loss (often plus regularization) on the i-th of n training examples. The
two textbook ways to descend this objective sit at opposite ends of a brutal trade-off, and the
problem is that *neither* is acceptable when n is large and a reasonably accurate solution is
wanted. Full-batch gradient descent (GD) takes a clean step along ∇P(w) = (1/n) Σ_i ∇ψ_i(w) and,
for a smooth strongly convex P, contracts the error by a fixed factor every iteration — a linear
(geometric) rate — but each of those iterations touches all n examples, so its cost scales with n
and it is unappealing for large datasets. Stochastic gradient descent (SGD) replaces the full
gradient with the gradient of a single randomly drawn example, so each step costs O(1) independent
of n; the price is that this rate is only sublinear, O(1/t), and the step size has to be driven to
zero for the method to converge at all. The precise goal is to break that dichotomy: an optimizer
whose per-iteration cost is that of SGD (a constant number of single-example gradients, independent
of n), but whose convergence rate is that of GD (linear, with a step size that does *not* have to
decay), for smooth strongly convex finite sums — and to do so with memory that stays close to
linear in the dimension d, so that it remains usable on problems where storing one auxiliary
gradient vector per example would be out of the question.

## Background

The setting is large-scale empirical risk minimization, and the prevailing practical engine is
SGD (Robbins & Monro 1951; Bottou & LeCun 2003), used precisely because its cost per step does not
grow with n. The relevant theory and the relevant pain points:

- **The two regimes, made precise.** Assume each ψ_i is convex and L-smooth and that P is
  γ-strongly convex with L ≥ γ > 0; write the condition number κ = L/γ. With a constant step
  η < 1/L, GD enjoys linear convergence O((1 − γ/L)^t) (Nesterov 2004), so to reach accuracy ε it
  needs O(κ ln(1/ε)) iterations — but each iteration is n gradient evaluations, i.e. O(nκ ln(1/ε))
  gradient computations overall. SGD, w^(t) = w^(t-1) − η_t ∇ψ_{i_t}(w^(t-1)) with i_t uniform,
  uses one gradient per step and is unbiased (E_i[∇ψ_i(w)] = ∇P(w)), but converges only at the
  sublinear O(1/t) rate.

- **Why SGD is stuck at sublinear: the variance floor.** A single example's gradient equals the
  full gradient only in expectation; the spread around that mean is variance, and it does not go
  away on its own. The sharp way to see the consequence is at the optimum w*: there ∇P(w*) = 0,
  but the individual ∇ψ_i(w*) are generally *not* zero — only their average is. The quantity
  σ² = (1/n) Σ_i ‖∇ψ_i(w*)‖² is positive in general. So even if SGD were handed w = w* it would
  take a nonzero step and walk away. With a constant step size η the iterate therefore cannot
  settle: its expected squared distance to w* contracts toward a positive floor of order η σ²/γ
  rather than toward zero (Nedić & Bertsekas 2000 split the constant-step SG rate into a part that
  decays linearly in t and a part, independent of t, that does not vanish). To push that floor to
  zero one is forced to send η_t → 0, and a vanishing step is exactly what drags the rate down to
  O(1/t). The noise σ², not any bias, is what caps the rate.

- **An optimality barrier — and the structural loophole.** For strongly convex optimization in
  the oracle model where an algorithm may only query *unbiased* noisy gradient measurements, the
  O(1/t) rate is optimal: no method restricted to that information can do better (Nemirovski &
  Yudin 1983; Nemirovski et al. 2009; Agarwal et al. 2012). The loophole is that the finite-sum
  problem gives *more* than an unbiased-measurement oracle: the same n functions recur, sampling
  is from a fixed finite set rather than a fresh draw each step, and the population is small enough
  to occasionally do something with all of it. Whatever beats O(1/t) must exploit that the dataset
  is finite and fixed, not anonymous fresh noise.

- **Control variates (classical Monte Carlo).** A standard variance-reduction device for
  estimating E[X] by sampling: if a second random variable, correlated with X, has a mean that is
  cheaply known, that knowledge can be turned into a corrected estimator with the same mean as X
  but smaller variance, by an amount that grows with the correlation. Variance-reduction
  techniques in this spirit are well established in stochastic estimation, including for gradient
  estimates (e.g. Greensmith, Bartlett & Baxter 2004 in the reinforcement-learning setting).

- **Smoothness ⇄ suboptimality.** A workhorse fact for smooth convex functions: the squared
  gradient norm is controlled by the function-value gap. For an L-smooth convex f with minimizer
  at value f*, one gradient step from x with step 1/L decreases the value by at least
  (1/2L)‖∇f(x)‖², which rearranges to ‖∇f(x)‖² ≤ 2L[f(x) − f*]. This is the standard bridge from
  "the gradient is small" to "the value is near optimal" and back, and it is the kind of bound a
  convergence proof in this area is built from.

## Baselines

The methods a new finite-sum optimizer would be measured against, and where each stalls.

**Gradient descent / accelerated GD (Cauchy 1847; Nesterov 1983, 2004).** The exact full gradient
gives a constant-step linear rate; accelerated GD improves the κ-dependence. **Limitation:** every
iteration costs n example-gradients, so on large datasets even one pass is expensive and many
passes are needed; the per-iteration cost scaling with n is the thing one is trying to escape.

**Plain SGD with a decreasing step (Robbins & Monro 1951; Bottou).** One example-gradient per
step, cost independent of n. **Limitation:** the per-example gradients do not vanish at w*, so a
constant step leaves the method jittering in a ball around the optimum; convergence requires
η_t → 0, which yields only the sublinear O(1/t) rate. With a *fixed* large step it descends fast
at first but then oscillates above the minimum and never reaches it; with a fixed small step it
crawls. The usual practical patch is hand-tuned step-size scheduling (e.g. exponential or 1/t
decay), which trades one difficulty for a tuning burden and still does not cross into a linear
rate.

**Momentum / gradient averaging / iterate averaging (Tseng 1998; Nesterov 2009 dual averaging;
Polyak & Juditsky 1992).** Momentum reweights past stochastic gradients geometrically; gradient
averaging uses their running mean; iterate averaging averages the w^(t). These improve constants,
robustness, or asymptotic efficiency. **Limitation:** all still need decreasing step sizes and
none is known to push the rate below O(1/t) on the finite-sum problem.

**Incremental aggregated gradient, IAG (Blatt, Hero & Gauchman 2007).** Keep one stored gradient
y_i per example; each step refreshes the single y_{i_t} just visited and moves along the *average*
of the whole stored table, x^(k+1) = x^(k) − (α/n) Σ_i y_i. The step thus carries information about
every example at SGD-like per-step cost. **Limitation:** indices are visited *cyclically*, which
limits the analysis to strongly convex quadratics (with no explicit rate) and treats a full pass
as one iteration; it also requires storing the full n-gradient table.

**Stochastic average gradient, SAG (Le Roux, Schmidt & Bach 2012).** The randomized cousin of IAG:
same table of stored per-example gradients and the same averaged step, but i_t is *sampled*
uniformly instead of cycled. This randomization is enough to prove an explicit linear convergence
rate for general strongly convex finite sums while keeping the per-step cost of a single
example-gradient — the first method to combine SGD's cheap step with GD's linear rate.
**Limitation:** it must store one gradient (or, exploiting fi(aᵢᵀx) structure for linear models,
one scalar) per example — an O(n d) table in general, reducible to O(n) only in the linear-model
special case. For models without that structure (structured-prediction losses, neural networks)
the per-example table is impractical. The convergence analysis is also notably intricate (a joint
Lyapunov bound on gradients and iterates), and the proof's intuition for *why* the linear rate
appears is not transparent.

**Stochastic dual coordinate ascent, SDCA (Shalev-Shwartz & Zhang 2012; Hsieh et al. 2008 for the
linear-SVM instance).** For L2-regularized linear losses, optimize the dual by picking one dual
coordinate α_i at a time and maximizing along it; the primal iterate is reconstructed from the
duals. It attains a linear rate and is the workhorse behind popular linear-SVM/logistic solvers.
**Limitation:** it is formulated through convex conjugates and dual coordinates, tying it to the
regularized-linear-predictor form; it stores the n dual variables, and the dual machinery does not
extend cleanly to general (e.g. nonconvex, structured) objectives.

The common shape of the gap: SAG and SDCA *prove* that linear convergence at SGD's per-step cost
is achievable on finite sums — but both buy it by carrying per-example state (a gradient table or
dual variables) whose size grows with n, which rules them out exactly where one most wants a cheap
fast optimizer. Their convergence is also analyzed in ways that obscure the mechanism.

## Evaluation settings

The natural yardsticks for finite-sum optimizers:

- **L2-regularized multiclass logistic regression on MNIST** (handwritten-digit images, 10
  classes), a convex objective; small regularization (e.g. λ ≈ 1e-4). The clean convex case where
  optimizer behavior is isolated from local minima.
- **L2-regularized binary logistic regression** on standard LIBSVM benchmark datasets — e.g.
  rcv1 (high-dimensional sparse text), covtype, protein — and on CIFAR-10 features; convex, with
  small λ (e.g. 1e-3 to 1e-5). Image inputs normalized to [0,1]; some datasets standardized; for
  sets without a held-out split, the training data is halved into train/test.
- **Small fully-connected neural networks** (e.g. one hidden layer of ~100 units, sigmoid
  activation, softmax output, L2 regularization) on MNIST and CIFAR-10, trained with mini-batches
  (e.g. size 10). A nonconvex setting where the convex theory does not formally apply.
- **Protocol and accounting.** The fair cost axis is the *number of gradient computations divided
  by n* (effective passes over the data), since gradient evaluation dominates compute; a method
  that does extra full passes must pay for them on this axis. The comparison baseline is best-tuned
  SGD, including hand-scheduled learning-rate decay (exponential η_0 a^⌊t/n⌋ and 1/t-type
  η_0 (1 + b⌊t/n⌋)^{-1} schedules tuned over a grid), plus the stored-table linear-rate methods
  where they apply. Reported quantities are the training-loss residual P(w) − P(w*) (with P(w*)
  estimated by running GD a long time), the test error rate, and — as a diagnostic — the variance
  of the update direction over the run.

## Code framework

The optimizer plugs into the same finite-sum training harness used for the baselines. What already
exists is a problem object that can hand back the gradient of a single example, the average
gradient over all examples, and the loss; a loop that can take a fixed number of stochastic updates
by drawing example indices; and a fixed, constant step size. What is *not* settled is how to form
the update direction from those gradient queries — that is the slot to be designed. Filling it must
not change the model, the loss, or the per-example gradient oracle.

```python
import torch


class FiniteSumProblem:
    """The objective P(w) = (1/n) Σ_i ψ_i(w) and its gradient oracles (already given)."""

    def __init__(self, params, n, batch_size):
        self.params = list(params)        # model parameters w
        self.n = n                        # number of training examples
        self.batch_size = batch_size

    def grad_batch(self, w_state, idx):
        """∇ over a mini-batch of example indices, evaluated at parameters w_state.
        Returns a list of gradient tensors, one per parameter. (provided)"""
        ...

    def full_grad(self):
        """∇P(w) = (1/n) Σ_i ∇ψ_i(w) at the current parameters; one full pass. (provided)"""
        ...

    def loss_batch(self, idx):
        """Scalar loss on a mini-batch at the current parameters. (provided)"""
        ...


class FiniteSumOptimizer:
    """Owns whatever auxiliary state the update rule needs (kept ~O(d)), and applies a
    constant-step update from single-example/mini-batch gradient queries."""

    def __init__(self, problem, lr, inner_steps):
        self.problem = problem
        self.lr = lr
        self.inner_steps = inner_steps
        # TODO: any auxiliary state the update rule we design will maintain.

    def train_one_epoch(self):
        # a fixed number of cheap stochastic updates
        total_loss, n_batches = 0.0, 0
        for _ in range(self.inner_steps):
            idx = torch.randint(self.problem.n, (self.problem.batch_size,))
            # TODO: the update direction we will design — form it from the available
            #       gradient queries, then apply the constant-step in-place update
            #       w <- w - lr * <direction>.
            pass
            total_loss += float(self.problem.loss_batch(idx))
            n_batches += 1
        return {"avg_loss": total_loss / max(n_batches, 1)}
```

The harness supplies single-example/mini-batch gradients, an average-gradient pass, and a constant
step over a fixed number of stochastic inner updates; the update direction is the empty slot.
