# Context: Automatically setting the step size for first-order convex optimization

## Research question

Consider unconstrained convex minimization

    min_{x ∈ R^p} f(x),

where f is convex, has Lipschitz constant G (every subgradient satisfies ‖g‖ ≤ G), and has a
non-empty set of minimizers. The workhorse is the subgradient method: from a starting point x₀,

    x_{k+1} = x_k − γ_k g_k,   g_k ∈ ∂f(x_k),

and one returns an average iterate x̂_n = (1/(n+1)) Σ_{k=0}^n x_k. The single quantity that decides
whether and how fast this converges is the step size (learning rate) γ_k. Too large and the iterates
oscillate around the solution; too small and progress crawls.

The pain point is sharp and quantitative. The step size that achieves the best possible convergence
for this class is

    γ_k = D / (G √n),   where D = ‖x₀ − x*‖ is the distance from the start to a minimizer x*,

and it yields the worst-case-optimal rate

    f(x̂_n) − f* = O(D G / √n).

Two problem constants enter: G and D. Adaptivity to G is essentially solved (see Background). But D
depends on x*, the very thing optimization is trying to find — so D is not available, and in practice
nobody knows it. The standard workaround is to bracket D with loose lower/upper bounds d₀ and d_max
and run a log-spaced grid search over the step scale, training the model many times. This grid search
is the dominant cost of hyperparameter tuning in most machine-learning pipelines.

The question this context sets up: can the step size be set automatically — with no learning-rate
constant to tune, no extra function-value or gradient evaluations per step, no line search or
back-tracking — while still attaining the optimal O(DG/√n) rate, and ideally with no extra
logarithmic factor over the rate one would get if D were known?

## Background

**The optimal step size and why D is the obstacle.** For convex G-Lipschitz f the rate O(DG/√n) is
worst-case optimal for this complexity class (Nesterov, *Lectures on Convex Optimization*, 2018). It
is attained by γ_k = D/(G√n), and more flexibly by any-time variants that do not need n fixed in
advance. The obstruction is purely that D = ‖x₀ − x*‖ is unknown: it is a property of the (unknown)
solution.

**Adaptivity to the gradient scale G — AdaGrad-Norm.** The Lipschitz constant G can be removed from
the step size by accumulating observed gradient norms. The AdaGrad-Norm step size (Streeter &
McMahan, "Less Regret via Online Conditioning," 2010; Duchi, Hazan & Singer, "Adaptive Subgradient
Methods," JMLR 2011; Ward, Wu & Bottou, 2019) is

    γ_k = D / √( Σ_{i=0}^{k} ‖g_i‖² ),

combined with projection onto the D-ball around the origin. The load-bearing technical fact behind
its analysis is a telescoping bound: with γ_k = 1/√(Σ_{i<k}‖g_i‖²),

    Σ_{k=0}^n γ_k ‖g_k‖² ≤ 2 √( Σ_{k=0}^n ‖g_k‖² ),

so the accumulated gradient-error term stays controlled without knowing G ahead of time. The
coordinate-wise version of AdaGrad keeps a per-coordinate denominator a_(k+1)i = √(G²+Σ g_ki²) and
measures distance in the ℓ∞ norm, D∞ = ‖x₀ − x*‖∞. The crucial limitation: AdaGrad-Norm still has D
in its numerator, and performs poorly when D is estimated wrong.

**Dual averaging.** An alternative to plain subgradient descent that pairs naturally with any-time
step sizes is dual averaging (Nesterov; Orabona, "A Modern Introduction to Online Learning," 2019).
It accumulates a weighted gradient sum s_{k+1} = s_k + λ_k g_k and forms

    x_{k+1} = x₀ − γ_{k+1} s_{k+1}.

The classical dual-averaging suboptimality bound, with weights λ_k = d_k, reads

    Σ_{k=0}^n d_k (f(x_k) − f*) ≤ ½ γ_{n+1}^{-1} D² + Σ_{k=0}^n (γ_k/2) d_k² ‖g_k‖².

Here D enters as D². A useful tightening (Carmon & Hinder, below) replaces the D² term by D‖x₀−x_{n+1}‖
using the triangle inequality, which is never larger than ½γ^{-1}D² and is the same when D = ‖x₀−x_{n+1}‖.

**Hyper-gradients.** A separate line tracks the inner product between the gradient g_k and the current
step direction (Bengio 2000; Domke 2012; Pedregosa 2016; Feurer & Hutter 2019; Chandra et al. 2022).
When the gradient points the same way as the previous step, the learning rate is nudged up; otherwise
down. This inner product — the (negative) hyper-gradient — is a directional too-large/too-small
signal. Methods using it need an extra hyper-learning-rate to control how fast the step size changes,
and the signal carries only sign information, not the magnitude of the right step.

**Motivating empirical observations.** On a
toy problem f(x) = |x| with x₀ = 1.0, the distance to the solution is exactly D = 1.0; this is the
kind of one-dimensional instance where one can watch any distance-estimation scheme behave. AdaGrad-
Norm with a mis-set D either oscillates (D too big) or stalls (D too small). Grid search over a
log-spaced step scale is the de-facto standard precisely because no cheaper reliable alternative is in
common use for deep learning. Among methods that need no learning rate, very few have ever been
demonstrated on large deep-learning problems at all.

## Baselines

**Polyak step size** (Polyak, *Introduction to Optimization*, 1987). γ_k = (f(x_k) − f*)/‖g_k‖²
gives the optimal rate with no log factor — but trades knowledge of D for knowledge of f*, the
optimal value, which is usually also unknown. Plugging in estimates of f* tends to make convergence
unstable; restart schemes that maintain lower bounds on f* recover convergence but only within a
multiplicative log factor of optimal (Hazan & Kakade, 2019).

**Exact line search** (Drori & Taylor, 2020; Goujaud, Taylor & Dieuleveut, 2022). A scheme that
chooses γ_{k+1} by an exact one-dimensional minimization along the averaged direction attains the
optimal rate with no knowledge of problem constants. The cost: an exact line search per step, and
relaxing it to an approximate search without a smoothness assumption is non-trivial and threatens to
reintroduce dependence on problem constants.

**Bisection over the step grid** (Carmon & Hinder, "Making SGD Parameter-Free," COLT 2022). Rather
than running the full method at every grid point from d₀ to d_max, run a bisection. The starting
point is the observation that the "ideal" SGD step size η satisfies an implicit fixed-point equation

    η = φ(η),   φ(η) = ‖x₀ − x*‖ / √( Σ_i ‖g_i(η)‖² ) ,

whose right-hand side is computable up to the unknown distance, and which can be bracketed by finding
an interval [η, 2η] where η ↦ φ(η) − η changes sign. Solving by bisection on log η gives a rate

    f(x_n) − f* = O( D G · loglog(d_max/d₀) / √(n+1) ),

improving the log dependence on the grid range to loglog. The limitation: it is still a search /
double-loop procedure with a residual loglog factor, and is framed through regret minimization.

**DoG — Distance over Gradients** (Ivgi, Hinder & Carmon, 2023; concurrent). Estimate the distance by
the largest movement seen,

    r̄_k = max_{i ≤ k} ‖x_i − x₀‖,

and use a step size r̄_k / √(Σ‖g_i‖²) — distance over gradients, no multiplicative learning rate. The
limitation that has to be patched: r̄_k is not necessarily bounded (there is a convex counterexample
where it diverges), so a tamed variant adds dampening in the denominator; the resulting rate carries
extra multiplicative log factors compared with the D-known rate.

**Coin betting / COCOB** (Orabona & Tommasi, "Training Deep Networks without Learning Rates Through
Coin Betting," NeurIPS 2017; McMahan & Orabona, 2014). Assuming knowledge of G but not D, online
parameter-free algorithms based on coin betting achieve, among D-unknown methods, the best possible
regret

    Regret_n = O( D G √( (n+1) log(1 + D) ) ),

a √-log factor worse than the best regret achievable with knowledge of D; online-to-batch conversion
gives O( DG √(log(1+D)) / √(n+1) ) in function value. Limitations: the √-log penalty; asymptotic
rates for coin betting are not known; and the method fixes its own implicit schedule, so it cannot
accept the explicit learning-rate schedules (warmup, cosine, inverse-sqrt) that large deep-learning
models rely on. The COCOB variant is the one parameter-free method that has actually been run on deep
networks, making it the natural empirical point of comparison.

**Reward doubling** (Streeter & McMahan, 2012). In one dimension, track Σ x_k g_k against η·H̄, where
H̄ pre-bounds the total sum of squared gradients; whenever the reward exceeds the threshold, double
the step size and restart the optimizer from x₀. Rates similar to coin betting; the resets and the
need for H̄ are the drawbacks.

Common thread: every baseline either keeps a residual log / loglog penalty, swaps one unknown for
another (Polyak's f*), pays for line searches, or gives up the ability to use a standard learning-rate
schedule. None achieves complete asymptotic independence from problem constants at the optimal rate
with no extra log factor.

## Evaluation settings

The natural yardstick spans convex problems and deep learning, comparing an automatic step size
against a hand-tuned learning rate found by grid search.

- **Convex.** Logistic regression on 12 standard LIBSVM benchmark datasets; 100 epochs, batch size
  16, no weight decay, a stage-wise schedule with 10-fold decreases at 60/80/95 epochs; baseline LR
  chosen by grid search for highest final accuracy; 10 seeds. A sweep of the initial estimate d₀ over
  10^-16 … 10^-2 to test sensitivity.
- **Image classification.** CIFAR-10 (Wide ResNet 16-8), CIFAR-100 (DenseNet), ImageNet (ResNet-50),
  with the standard SGD-with-momentum pipelines, step-decay schedules, and standard augmentation.
- **Sequence and language.** IWSLT14 German→English machine translation (LSTM, inverse-sqrt
  schedule); masked language modelling with a 110M-parameter RoBERTa on Book-Wiki; auto-regressive
  language modelling with a GPT decoder-only transformer on Book-Wiki; polynomial-decay schedules
  with warmup.
- **Vision transformers** (ViT-tiny, 300 epochs, cosine schedule, heavy regularization);
  **object detection** (Faster-RCNN with a ResNeXt-101 backbone, COCO 2017); **fastMRI** knee
  reconstruction (VarNet 2.0); **recommendation** (DLRM on Criteo Kaggle click-through).

Metrics are the field-standard ones (accuracy, BLEU/perplexity, mAP, reconstruction SSIM, log-loss).
The protocol of interest is: take the schedule normally used for each problem, set only the *base*
learning rate automatically, and compare to the hand-tuned base learning rate.

## Code framework

The pieces below already exist: a PyTorch training loop, the `torch.optim.Optimizer` base class, the
data pipelines and models for the benchmarks above. What is missing is the optimizer that sets its own
step size. The scaffold lays out the empty optimizer stub; the contribution fills its `step`.

```python
import torch
import math

# --- already exists: a standard supervised training loop ---
def train(model, loader, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()        # the optimizer decides the step size internally

# --- already exists: AdaGrad-Norm style accumulation of gradient norms ---
# gsq = sum of ||g_i||^2 over steps; step scale = numerator / sqrt(gsq)
# the only unknown is the numerator that should equal the distance to the solution.

class AutoStepOptimizer(torch.optim.Optimizer):
    """
    A first-order optimizer that requires no learning-rate constant from the user.
    Drop-in for SGD/Adam: same loop, same schedules. The user passes only an
    (almost arbitrary) tiny initial scale d0 and, optionally, the problem's usual
    learning-rate *schedule* applied as a multiplier with base value 1.0.
    """
    def __init__(self, params, lr=1.0, d0=1e-6, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, d0=d0, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # TODO: the running state needed to set the step scale automatically.

    def step(self, closure=None):
        # The gradients g_k are available as p.grad for each parameter p.
        #
        # TODO: maintain whatever running state is needed across steps, and
        # TODO: set the step-scale numerator from observed quantities only
        #       (no x*, no f*, no true D), then
        # TODO: update the parameters with that automatically-set step.
        raise NotImplementedError
```
