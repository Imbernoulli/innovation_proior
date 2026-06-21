# Context

## Research question

We want to train large models fast, and we have a strong prior that *preconditioning* the gradient — premultiplying it by a matrix that reshapes the geometry of the loss — can dramatically accelerate convergence over plain gradient descent. The trouble is cost. For a parameter vector of dimension `d`, the ideal preconditioners (the local Hessian for Newton's method, the gradient second-moment matrix for adaptive methods) are `d × d`. In modern models `d` is in the millions, so a dense `d × d` matrix is both impossible to store (`d²` numbers) and impossible to invert or take a root of (`O(d³)` work). The question is whether we can keep most of the *power* of a full preconditioner — its ability to model correlations between coordinates, not just rescale them one at a time — while paying something close to first-order cost in memory and compute.

A key observation reframes the problem. In almost every model the parameters do not live in a flat vector; they live in a *tensor*. A multiclass linear model has a weight matrix of size (features × classes). A fully-connected layer has a weight matrix of size (outputs × inputs). A convolutional layer has an order-4 tensor (in-depth × width × height × out-depth). The tooling (Torch, TensorFlow) is built around exactly this tensor structure. So the question sharpens: can the natural tensor structure of the parameter space be exploited to build a preconditioner that retains more than per-coordinate rescaling without paying the cost of the full `d × d` matrix?

## Background

**Preconditioned gradient methods.** The general template is `w_{t+1} = w_t − η H_t^{-1} g_t`, where `g_t = ∇f_t(w_t)` and `H_t ≻ 0` is a preconditioner. Newton's method takes `H_t` to be the Hessian; quasi-Newton methods (BFGS, L-BFGS; Fletcher; Nocedal 1980; Lewis & Overton 2013) build curvature estimates from gradient differences. These bend the gradient to the local geometry and can converge far faster than steepest descent, but maintaining and inverting `H_t` is the bottleneck at scale.

**Online convex optimization (OCO).** The clean analysis framework (Shalev-Shwartz 2012; Hazan 2016). A learner plays `w_t ∈ W ⊆ R^d`, then an adversary reveals convex `f_t`; the learner suffers `f_t(w_t)` and observes `f_t`. Performance is measured by *regret*, `R_T = Σ_t f_t(w_t) − min_{w∈W} Σ_t f_t(w)`. An online-to-batch conversion (Cesa-Bianchi et al. 2004) turns a regret-`R_T` algorithm into a stochastic optimizer converging at rate `O(R_T/T)`, so a `O(√T)` regret bound is the gold standard and yields `O(1/√T)` stochastic convergence.

**Adaptive Online Mirror Descent.** With a time-varying PSD regularizer `H_t`, the update is `w_{t+1} = argmin_{w∈W} { η g_t·w + ½‖w−w_t‖²_{H_t} }`, where `‖x‖²_A = x^T A x`. On `W = R^d` this is exactly `w_{t+1} = w_t − η H_t^{-1} g_t`. Its regret obeys a standard bound (e.g. as used by Duchi et al. 2011):
```
R_T ≤ (1/2η) Σ_t ( ‖w_t−w*‖²_{H_t} − ‖w_{t+1}−w*‖²_{H_t} ) + (η/2) Σ_t ( ‖g_t‖*_{H_t} )² ,
```
where `‖x‖*_A = √(x^T A^{-1} x)` is the dual norm. The whole game in adaptive methods is choosing `H_t` to make the right-hand side small.

**The adaptive-regularization principle.** A general recipe (Gupta, Koren & Singer 2017) selects `H_t = argmin_{H≻0} { M_t • H^{-1} + Φ(H) }` for `M_t = Σ_{s≤t} g_s g_s^T` and a potential `Φ`, and shows `Σ_t (‖g_t‖*_{H_t})² ≤ Σ_t (‖g_t‖*_{H_T})² + Φ(H_T) − Φ(H_0)`. With `Φ(H) = tr(H)`, the minimizer is `H_t ∝ (Σ_{s≤t} g_s g_s^T)^{1/2}` — the square root of the accumulated gradient covariance — and the second regret term collapses to `tr((Σ g g^T)^{1/2})`. This is the optimal full-matrix preconditioner in this family.

**Matrix tools available.** Spectral calculus: for symmetric PSD `A = Σ_i λ_i u_i u_i^T` and any `α`, `A^α = Σ_i λ_i^α u_i u_i^T`, computed via eigen/SVD. Operator monotonicity (Löwner 1934): `x ↦ x^α` is operator-monotone for `α ∈ [0,1]`, i.e. `0 ⪯ X ⪯ Y ⇒ X^α ⪯ Y^α`. A geometric-mean inequality for commuting PSD matrices (Ando, Li & Mathias 2004): if `0 ⪯ X_i ⪯ Y_i`, all `X_i` commute and all `Y_i` commute, and `Σ α_i = 1` with `α_i ≥ 0`, then `∏_i X_i^{α_i} ⪯ ∏_i Y_i^{α_i}`. The Kronecker product `A ⊗ B` and the flattening `vec(·)`, with the identities `(A⊗B)(A'⊗B') = (AA')⊗(BB')`, `(A⊗B)^s = A^s⊗B^s` for PSD `A,B`, `tr(A⊗B) = tr(A) tr(B)`, `vec(uv^T) = u⊗v`, and the mixed-product/vec identity `(L ⊗ R^T) vec(G) = vec(L G R)`.

## Baselines

**Diagonal AdaGrad (Duchi, Hazan & Singer 2011).** Take `H_t = diag(Σ_{s≤t} g_s ⊙ g_s)^{1/2}`, i.e. per-coordinate accumulated squared gradients, root-ed. Update `w_{t+1,i} = w_{t,i} − η g_{t,i} / (√(Σ_s g_{s,i}²) + ε)`. Memory and compute are `O(d)`. It adapts the per-coordinate learning rate, which is enormously effective on sparse, badly-scaled problems.

**Adam (Kingma & Ba 2014).** Diagonal AdaGrad with an exponential moving average of the squared gradients (and of the gradients, for momentum) instead of a sum, plus bias correction: `m_t = β₁m_{t-1}+(1−β₁)g_t`, `v_t = β₂v_{t-1}+(1−β₂)g_t²`, `w_t ← w_t − η m̂_t/(√v̂_t+ε)`. Same `O(d)` cost.

**Full-matrix AdaGrad (Duchi, Hazan & Singer 2011).** The non-diagonal version: `H_t = (Σ_{s≤t} g_s g_s^T)^{1/2}`, with update `w_{t+1} = w_t − η H_t^{-1} g_t`. This is the member of the adaptive family that captures all pairwise correlations, and it enjoys a strong regret bound `~ tr((Σ g g^T)^{1/2})`. For a weight matrix `W ∈ R^{m×n}`, `d = mn`, so `H_t` has `m²n²` entries and the root/inverse costs `O(m³n³)`.

**Sketched / sub-sampled second-order methods (Gonen & Shalev-Shwartz 2015; Pilanci & Wainwright 2017; Erdogdu & Montanari 2015; Agarwal, Bullins & Hazan 2016; Xu et al. 2016).** Approximate the Hessian or its action by random projection or sub-sampling.

**K-FAC (Martens & Grosse 2015).** A factored second-order method for neural nets: it approximates each layer's Fisher-information matrix as a Kronecker product of two smaller matrices, using independence assumptions on the backpropagated activations and gradients. It is genuinely Kronecker-factored and relies on the statistics of the backpropagated gradients and samples from the model's predictive distribution, making it architecture-aware.

## Evaluation settings

The natural yardsticks are standard supervised-training benchmarks where one swaps only the optimizer and watches training loss / test metric versus steps and wall-clock:
- **CIFAR-10 / CIFAR-100** image classification, with standard convolutional networks: residual networks (e.g. a 32-layer and a 55-layer ResNet) and a small (20-layer) inception-style network. Metric: training cross-entropy loss and test error rate, as a function of the number of mini-batch steps.
- **LM1B** (the One Billion Word benchmark; Chelba et al. 2013) statistical language modeling, with an attention-based model (the architecture of Vaswani et al. 2017) whose parameters are order-≤2 tensors. Metric: test (log-)perplexity versus steps.
- Protocol in use at the time: mini-batches (size 128), a learning-rate sweep per optimizer choosing the best by training loss/error, and reporting steps-per-second on a single GPU to compare per-step runtime against SGD, AdaGrad, and Adam.

## Code framework

```python
import torch
from torch.optim.optimizer import Optimizer


def _matrix_power(matrix, power):
    """Symmetric-PSD matrix power via eigen/SVD: A = sum_i lambda_i u_i u_i^T,
    A^power = sum_i lambda_i^power u_i u_i^T."""
    u, s, v = torch.svd(matrix)
    return u @ s.pow(power).diag() @ v.t()


class PreconditionedOptimizer(Optimizer):
    """Generic preconditioned first-order optimizer scaffold.

    Template step:  w <- w - lr * P(grad), where P(.) applies some preconditioner
    built from accumulated gradient second-moment statistics. The specific
    statistics kept and the exact preconditioning map P are the open design slot.
    """

    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                 epsilon=1e-4, update_freq=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq)
        super().__init__(params, defaults)

    def _init_state(self, state, grad, epsilon):
        # TODO: allocate whatever second-moment statistics the method needs,
        #       given the shape of `grad`, and their root/inverse caches.
        pass

    def _accumulate_statistics(self, state, grad):
        # TODO: fold the new gradient into the accumulated second-moment statistics.
        pass

    def _precondition(self, state, grad, update_freq, step):
        # TODO: form the preconditioner from the statistics and apply it to grad,
        #       returning the preconditioned gradient.
        pass

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = grad.clone()
                    self._init_state(state, grad, group["epsilon"])

                if group["momentum"] > 0:
                    grad.mul_(1 - group["momentum"]).add_(
                        state["momentum_buffer"], alpha=group["momentum"])
                if group["weight_decay"] > 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                self._accumulate_statistics(state, grad)
                grad = self._precondition(state, grad, group["update_freq"], state["step"])

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])
        return loss
```
