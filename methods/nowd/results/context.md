# Context: regularizing adaptive-gradient training of deep nets (circa 2017)

## Research question

Adaptive gradient methods — AdaGrad, RMSProp, and especially Adam — have become the default
optimizers for training feed-forward and recurrent networks, because they remove most of the
pain of per-coordinate learning-rate tuning. Yet on the problems where it matters most for a
practitioner, they underperform: the strongest published results on image-classification
benchmarks (CIFAR-10/100, ImageNet) are still obtained with plain SGD with momentum, and
across a diverse set of tasks the adaptive methods reach *worse test error than SGD even when
their training loss is as good or better*. So a practitioner is forced into a per-task choice
between two optimizers — Adam for convenience and fast training, SGD-with-momentum for final
generalization — and there is no principled reason this should be necessary. The precise
problem: identify what, concretely, makes a tuned adaptive method generalize worse than a
tuned SGD-with-momentum on these regularization-sensitive tasks, and fix it with a change
small enough that the resulting optimizer keeps Adam's ease of use while matching SGD's
test-time quality. A solution has to (1) explain the gap mechanistically, not just patch it;
(2) leave Adam's adaptive per-coordinate step intact; and (3) ideally make the regularization
hyperparameter easier to set, not harder.

## Background

The field state at the time. Stochastic first-order optimization is the engine of deep
learning. SGD with momentum is the conservative default; the adaptive family —
AdaGrad (Duchi, Hazan & Singer 2011), RMSProp (Tieleman & Hinton 2012), Adam (Kingma & Ba
2014), and the recent AMSGrad (Reddi, Kale & Kumar 2018) — gives every parameter its own
effective step size by dividing the update by a per-coordinate running magnitude of the
gradient. Concretely an adaptive optimizer applies a *diagonal preconditioner*: its update
has the form `theta_{t+1} = theta_t - alpha M_t grad f_t(theta_t)` with `M_t` a diagonal
matrix that is emphatically *not* a scalar multiple of the identity — that non-uniformity
across coordinates is the whole point of being adaptive.

The motivating empirical findings. Wilson, Roelofs, Stern, Srebro & Recht (2017) ran tuned
adaptive methods against tuned SGD/SGD-momentum across image classification (CIFAR-10),
character-level language modeling, and constituency parsing, and reported that the adaptive
methods generalized *worse — often significantly worse — than SGD, even when the adaptive
solution achieved equal or better training performance*. They also exhibited a constructed
linearly-separable binary problem on which (S)GD reaches zero test error while AdaGrad,
Adam, and RMSProp are driven to test error near chance, demonstrating that adaptive methods
can settle on a qualitatively different and worse-generalizing solution at identical training
loss. Separately, the strongest CIFAR/ImageNet numbers in the literature (e.g. the
Shake-Shake-regularized residual networks of Gastaldi 2017, and the AutoAugment results of
Cubuk et al. 2018) were all obtained with SGD-with-momentum, not with any adaptive method.
The state of the world is a documented, reproducible generalization gap, and a set of
hypotheses for it that are on the table but unsettled — that adaptive methods are drawn to
sharp minima that generalize poorly (Keskar et al. 2016; with the caveat from Dinh et al.
2017 that sharpness is reparameterization-dependent), or that there is some intrinsic defect
of adaptive preconditioning (Wilson et al. 2017).

Two regularization mechanisms that the field treats as interchangeable. The first is
**L2 regularization**: augment the batch loss with a quadratic penalty,
`f^reg_t(theta) = f_t(theta) + (lambda'/2) ||theta||_2^2`, whose gradient adds `lambda' theta`
to the loss gradient before the optimizer sees it. The second is **weight decay** in the
sense of Hanson & Pratt (1988): at every step, shrink the weights toward zero by a constant
multiplicative factor as a step *separate from* the loss gradient,

```
theta_{t+1} = (1 - lambda) theta_t - alpha grad f_t(theta_t),
```

where `lambda` is the per-step decay rate. In deep-learning libraries the two are used as
synonyms — the L2 coefficient is routinely *called* "weight decay" — and for plain SGD this
is harmless: the two coincide. The relationship between them under a *non-identity*
preconditioner has not been examined; the field carries the SGD intuition over to the
adaptive methods without checking it.

EMA / moment machinery that the adaptive methods rest on. Adam maintains two
exponential-moving-average vectors per parameter — `m_t` of the gradient (first moment) and
`v_t` of the squared gradient (second raw moment) — bias-corrects each for its zero
initialization, and steps along the ratio `m_hat_t / (sqrt(v_hat_t) + eps)`. The denominator
`sqrt(v_hat_t)` is the per-coordinate adaptive scale: it is large for coordinates whose
gradients have historically been large, small for coordinates whose gradients have been
small. This is exactly the diagonal preconditioner `M_t` above, with the large-historic-
gradient coordinates getting the smallest effective step.

Scheduling and restarts that exist for SGD. A global learning-rate schedule — annealing the
rate over a run, or the cyclic cosine-annealing-with-warm-restarts of SGDR (Loshchilov &
Hutter 2016), which cools the rate along a cosine and periodically resets it while keeping the
weights — is known to improve both the final and the anytime performance of SGD on image
classification. It is *not* commonly applied to adaptive methods, on the folk reasoning that
an adaptive method already sets its own per-parameter rates and so does not need a global
schedule.

A Bayesian-filtering view of adaptive optimization (Aitchison 2018). Stochastic optimization
of `n` parameters can be cast as tracking, by Bayesian filtering, a distribution over the
optimal value of each `theta_i` given the current values of the others. With a Gaussian
state-transition prior and an approximate conjugate likelihood, the filtering mean updates as
`mu_post = mu_prior + Sigma_post * g` (with `g` the mini-batch log-likelihood gradient), so
the gradient preconditioner *is* the posterior covariance `Sigma_post`: larger steps for
parameters we are more uncertain about. Adam, RMSProp and Kronecker-factored methods all fall
out as special cases. The state-transition prior is taken to be
`P(theta_{t+1} | theta_t) = N((I - A) theta_t, Q)`, where `Q` is the weight-perturbation
covariance and `A` is a regularizer keeping the values from growing unboundedly.

## Baselines

These are the prior methods a new regularization scheme would be measured against and react to.

**SGD with momentum, with L2 regularization (the strong incumbent).** The default workhorse
for image classification. Maintains a momentum buffer and steps along it; L2 regularization
enters as `lambda' theta` added to the gradient. Because the preconditioner is a scalar
(`M_t = I` up to the global rate), the added penalty behaves exactly as a uniform
multiplicative shrink of the weights — adding `lambda' theta` to the gradient and stepping by
`-alpha` decays `theta` by `alpha lambda'` per step. Empirically this is what delivers the
state-of-the-art generalization on CIFAR/ImageNet. **Limitation:** one fact that is usually
overlooked even here is that the L2 coefficient `lambda'` that reproduces a given decay
strength depends on the learning rate (`lambda' = lambda/alpha`); the best L2 setting is
therefore tied to the best learning rate, so the two hyperparameters cannot be searched
independently — their joint optimum lies on a diagonal in the `(alpha, lambda')` plane, and
changing one without the other degrades results. This coupling is part of why SGD has a
reputation for being finicky to tune.

**Adam with L2 regularization (the convenient method that underperforms here).** Adam with
the L2 penalty implemented the standard library way: `lambda' theta` is added to the batch
gradient `g_t`, after which the gradient — now carrying both the loss term and the regularizer
— is fed through the moment EMAs and the adaptive normalization. The full update is
`theta_t = theta_{t-1} - alpha (beta_1 m_{t-1} + (1-beta_1) g_t) / (sqrt(beta_2 v_{t-1} +
(1-beta_2) g_t^2) + eps)` with `g_t = grad f_t(theta_{t-1}) + lambda' theta_{t-1}`.
**Limitation:** on the regularization-sensitive image-classification tasks where L2 is what
makes SGD shine, Adam-with-L2 does not match SGD-with-momentum; in tuning sweeps its best
results with a nonzero L2 coefficient are barely better than its results with no L2 at all, as
if the penalty is doing little. The shape of its `(alpha, lambda')` tuning landscape shows the
same diagonal coupling as SGD's, so the two hyperparameters are again entangled.

**AdaGrad / RMSProp / AMSGrad (the rest of the adaptive family).** The same diagonal-
preconditioner structure with different choices of how the per-coordinate scale is
accumulated (AdaGrad: a growing sum of squared gradients; RMSProp: an EMA; AMSGrad: a running
max of the EMA to fix an Adam convergence issue). They share Adam's library treatment of
regularization. **Limitation:** they inherit whatever Adam-with-L2 suffers from on these
tasks, and likewise trail SGD-with-momentum on image classification generalization.

**A global learning-rate schedule for adaptive methods (the road not usually taken).** Cosine
annealing / SGDR-style warm restarts as a *global* multiplier on top of an adaptive method.
**Limitation:** it is rarely used with adaptive optimizers — the assumption being that
per-parameter adaptation already does the job of a schedule — so its potential benefit for
Adam is largely unexplored, and a previous attempt to give Adam warm restarts improved its
anytime behavior but failed to make it competitive with SGD's warm restarts.

## Evaluation settings

The natural yardsticks already in use for this question, all pre-existing:

- **CIFAR-10** (Krizhevsky 2009) image classification with the standard data augmentation,
  batch size 128, using the Shake-Shake-regularized residual networks of Gastaldi (2017)
  built on `fb.resnet.torch` — a 26 2x64d ResNet (~11.6M params) and a 26 2x96d ResNet
  (~25.6M params). Metric: Top-1 test error. These are exactly the regularization-sensitive
  models where SGD-with-momentum holds the state of the art, so they are the place an
  adaptive method's regularization gap would show up.
- **ImageNet32x32** (Chrabaszcz, Loshchilov & Hutter 2017), a downsampled 32x32 version of
  ImageNet (1.2M images), same network family and batch size; Top-1/Top-5 test error. An
  epoch here is far longer than on CIFAR-10, which makes it the natural place to check whether
  a regularization setting transfers across very different training-set sizes.
- **Training budgets** spanning short to very long runs (on the order of 100 up to 1800
  epochs), to test whether the best regularization strength is stable as the number of weight
  updates grows by an order of magnitude — a known confound, since the *total* amount of decay
  accumulated over a run scales with the number of updates.
- **Learning-rate schedules**: fixed, step-drop, and cosine annealing, applied as a global
  multiplier, to test interactions between the schedule and the regularizer.
- **Protocol**: identical network/data/batch-size across the optimizers being compared;
  the initial learning rate `alpha` and the regularization coefficient swept over a 2-D
  logarithmic grid; comparisons read off the best grid setting (and the *shape* of the
  best-setting basin in the `(alpha, regularizer)` plane, which reveals whether the two
  hyperparameters are coupled).

## Code framework

The open regularization mechanism plugs into the existing Adam training harness. Everything
about Adam's adaptive step — the two moment EMAs, the bias correction, the
`m_hat / (sqrt(v_hat) + eps)` ratio — already exists and is fixed; what is *not* settled is
how a regularization coefficient should enter the update, and there is one global schedule
multiplier `eta_t` available that can scale things per step. The substrate is the generic
adaptive-optimizer loop with one empty slot: where and how the regularization acts.

```python
import math
import torch


class AdaptiveOptimizer:
    """Existing adaptive optimizer (Adam-style): per-parameter first/second-moment
    EMAs, bias correction, and a diagonal-preconditioned step. The adaptive step
    itself is settled; the open question is how the regularizer should act."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @staticmethod
    def schedule_multiplier(t):
        """Global per-step multiplier eta_t; fixed at 1.0 unless a schedule is used."""
        return 1.0

    @torch.no_grad()
    def step(self, t):
        beta1, beta2 = self.betas
        lr_t = self.lr * self.schedule_multiplier(t)
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("sparse gradients are not supported")
            state = self.state[id(p)]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1

            # TODO: how should the regularization coefficient enter the update?
            #       Decide what, if anything, it modifies and where it acts.
            #       This empty slot is the contribution.

            # --- settled adaptive (Adam) step ---
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = exp_avg_sq.sqrt().add_(self.eps)
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = lr_t * math.sqrt(bias_correction2) / bias_correction1
            p.addcdiv_(exp_avg, denom, value=-step_size)


# existing minibatch training loop the optimizer plugs into
def train(model, loss_fn, data_loader, optimizer):
    t = 0
    for inputs, targets in data_loader:
        t += 1
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)   # existing task loss
        loss.backward()                          # fills p.grad
        optimizer.step(t)
```

The `# TODO` block flags the one open question — how the regularization coefficient enters
the update — without presupposing the answer; the final code fills exactly this slot.
