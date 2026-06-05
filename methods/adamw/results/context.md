# Context

## Research question

When we train deep networks, we almost always add some form of weight regularization to improve generalization. Two recipes are in circulation and are widely treated as the same thing: *weight decay*, which shrinks every weight toward zero by a small multiplicative factor at each step, and *L2 regularization*, which adds a squared-norm penalty to the loss. For plain stochastic gradient descent they coincide, and as a result essentially every deep-learning library implements "weight decay" by adding an L2 term to the gradient.

At the same time, the field has converged on adaptive gradient methods — optimizers that rescale each coordinate's step by its own historical gradient magnitude — as the default tool for training neural networks, because they are fast and forgiving about learning-rate choice. Yet on the benchmarks people care most about (image classification on CIFAR and ImageNet), the best results are still obtained with ordinary SGD with momentum, and adaptive methods are observed to generalize worse even when they reach a lower training loss.

The question, then: is the "weight decay = L2 regularization" identity actually valid for adaptive gradient methods, or only for SGD? If it breaks, does that break explain part of the generalization gap, and what is the right way to regularize an adaptive optimizer? A satisfying answer must (i) say precisely under what conditions the two recipes diverge, (ii) connect that divergence to the observed under-performance of adaptive methods, and (iii) give a modification simple enough that practitioners will actually use it — ideally one that also makes the regularization strength and the learning rate independent to tune.

## Background

**Weight decay (Hanson & Pratt, 1988).** The original mechanism shrinks the weights multiplicatively each step, then takes the gradient step:
  θ_{t+1} = (1 − λ) θ_t − α ∇f_t(θ_t),
where λ is the decay rate per step and α the learning rate. The bias is toward small-norm networks.

**L2 regularization.** Add a penalty to the objective, f^reg(θ) = f(θ) + (λ'/2)‖θ‖²₂, so the gradient carries an extra term, ∇f^reg = ∇f + λ'θ. This extra term is then fed to whatever optimizer is in use. Because it lives inside the gradient, libraries implement it by a single line `grad += wd * param` and call it "weight decay."

**Plain and momentum SGD.** θ_{t+1} = θ_t − α ∇f_t(θ_t); with momentum, m_t = β₁ m_{t−1} + α ∇f_t, θ_{t+1} = θ_t − m_t (sign/scaling conventions vary). The update direction is the (smoothed) raw gradient, applied with a single global learning rate.

**Adaptive gradient methods.** The line that motivates the present setting:
- *AdaGrad* (Duchi et al., 2011) gives each coordinate its own rate by dividing the gradient by the square root of the running *sum* of its past squared gradients. Effective per-coordinate rates only ever shrink, eventually toward zero.
- *RMSProp* (Tieleman & Hinton, 2012) replaces that growing sum with an exponential moving average v_t = β₂ v_{t−1} + (1−β₂) g²_t, so rates stay responsive: θ −= α g/(√v_t + ε).
- *Adam* (Kingma & Ba, 2014) adds momentum and bias correction on top:
    m_t = β₁ m_{t−1} + (1−β₁) g_t,  v_t = β₂ v_{t−1} + (1−β₂) g²_t,
    m̂_t = m_t/(1−β₁ᵗ),  v̂_t = v_t/(1−β₂ᵗ),
    θ_t = θ_{t−1} − α m̂_t/(√v̂_t + ε),
  with defaults α=0.001, β₁=0.9, β₂=0.999, ε=10⁻⁸. The bias correction is needed because m and v are initialized at zero and would otherwise be biased toward zero early in training.
- *AMSGrad* (Reddi et al., 2018) modifies the v_t accumulation to fix a convergence flaw; it shares the same preconditioned structure.

The structural fact common to all of these: after whatever first-moment smoothing they use, the update is scaled coordinate-wise by a diagonal preconditioner M_t whose entries are inverse RMS gradient magnitudes (for Adam, the adaptive denominator is √v̂_t + ε). The simplified algebraic form is θ_{t+1} = θ_t − α M_t d_t, where d_t is the raw or smoothed gradient direction. In general M_t ≠ k I for any scalar k — that non-proportionality to the identity is exactly what makes these methods "adaptive."

**Cosine annealing and warm restarts (Loshchilov & Hutter, 2016).** A learning-rate-multiplier schedule
  η_t = η_min + 0.5 (η_max − η_min)(1 + cos(π T_cur/T_i)),
that cools the rate down a cosine curve over T_i epochs and then "restarts" by resetting T_cur to 0 (η jumps back up) while keeping the current weights, multiplying the next budget by a factor T_mult. It improves anytime performance and underlies several state-of-the-art image-classification results when combined with strong models. It had been demonstrated for SGD; whether it transfers to adaptive methods was open.

**A Bayesian-filtering view of adaptive methods (Aitchison, 2018).** Stochastic optimization of one parameter given the others can be cast as a tracking/filtering problem: a state-transition prior P(θ_{t+1}|θ_t) describes a small data-independent drift, a likelihood from the next mini-batch updates it, and the posterior-mean update takes the form μ_post = μ_prior + Σ_post g, so the *preconditioner is the posterior covariance* — larger steps where we are more uncertain. Adam, RMSProp and Kronecker-factored methods arise as special cases. The state-transition prior is written P(θ_{t+1}|θ_t) = N((I − A)θ_t, Q), where A is a regularizer keeping weights from growing without bound.

**Motivating empirical observations about existing systems.** Adaptive methods are the default for feed-forward and recurrent nets, yet the strongest CIFAR-10/100 and ImageNet results come from SGD with momentum (e.g. with Shake-Shake regularization). Wilson et al. (2017) reported that across image classification, character-level language modeling and constituency parsing, adaptive gradient methods generalize worse than well-tuned SGD with momentum, sometimes even while achieving lower training loss. Candidate explanations under discussion included sharp vs. flat minima (Keskar et al., 2016; Dinh et al., 2017) and intrinsic limitations of adaptation (Wilson et al., 2017), none of which had been reduced to a concrete, fixable mechanism. A further observation: for these optimizers the value of the regularization strength that works best changes with the training budget — longer runs (more weight updates) prefer a smaller value — and with batch size, since smaller batches at fixed epochs perform more updates and thus more cumulative shrinkage (related to Li et al., 2017).

## Baselines

**SGD with momentum + L2 regularization.** The standard recipe behind the strongest image-classification results at the time. The L2 term λ'θ is added to the gradient and smoothed through the momentum buffer. Core idea is sound and generalizes well; the gap it leaves is hyperparameter coupling — the regularization coefficient that works best is tied to the learning rate (changing one without the other degrades results), which contributes to SGD's reputation for hyperparameter sensitivity.

**Adam with L2 regularization.** The dominant "easy" optimizer, regularized the way the libraries offer: λ'θ added to the gradient before the moment accumulation and the √v̂ normalization. Core idea: per-coordinate adaptive rates + momentum + bias correction. The gap: on datasets where L2 clearly helps SGD, Adam barely benefits from non-zero λ' at all, and its best results trail SGD's. The structural reason is that the regularization term, living inside the gradient, is rescaled by the same per-coordinate preconditioner as the loss gradient — so the way it shrinks each weight depends on that weight's parameter and gradient history rather than being uniform.

**Adam without regularization.** The unregularized adaptive baseline; on image classification its best settings are roughly as good as Adam-with-L2's, which is itself the symptom that L2 is doing little for Adam.

**AdaGrad / RMSProp / AMSGrad.** Earlier or contemporary adaptive methods sharing the preconditioned structure; they inherit the same coupling between regularization and adaptation, so they are the family the analysis is expected to generalize to.

## Evaluation settings

- **Datasets.** CIFAR-10 (Krizhevsky, 2009) with standard data augmentation; ImageNet32×32 (Chrabaszcz et al., 2017), a downsampled 32×32 version of ImageNet with 1.2M images and an epoch roughly 24× longer than CIFAR-10's — useful for testing whether a regularization setting transfers across very different budgets and datasets.
- **Models.** Residual networks with Shake-Shake regularization following the fb.resnet.torch / Shake-Shake setup (Gastaldi, 2017): a 26-layer 2×64d ResNet (≈11.6M parameters) and a 26-layer 2×96d ResNet (≈25.6M parameters).
- **Protocol.** Batch size 128. Training budgets spanning a wide range, from 100 up to 1800 epochs, to expose budget dependence. Learning-rate schedules compared: fixed, step-drop (drops at chosen epochs), and cosine annealing. Hyperparameters explored on a 2-D grid over the initial learning rate α and the regularization strength, so the *shape* of the good-hyperparameter region (its alignment with the axes vs. the diagonal) can be read off — that shape diagnoses whether the two hyperparameters are coupled.
- **Metrics.** Top-1 test error (CIFAR-10) and Top-5 test error (ImageNet32×32); training loss (cross-entropy) and test-error learning curves over epochs for anytime behavior; for fixed training-loss levels, the corresponding test error, to separate "optimizes better" from "generalizes better."

## Code framework

The optimizer is the unit being modified. Below is a minimal Adam-style step inside a per-parameter loop, a momentum-SGD step, and a training loop, with the regularization slot left empty for whatever mechanism we settle on. The library primitives are parameter tensors, gradients via autograd, moment buffers, and a learning-rate schedule hook.

```python
import torch

class AdaptiveOptimizer:
    """Per-coordinate adaptive optimizer (Adam-style): momentum + RMS preconditioner + bias correction."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, reg=0.0, schedule=None):
        self.params = list(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.reg = reg                       # regularization coefficient placeholder
        self.schedule = schedule or (lambda step: 1.0)
        self.state = {id(p): dict(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                      for p in self.params}

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])

            # TODO: incorporate the chosen regularization mechanism here.
            pass

            st["m"].mul_(b1).add_(g, alpha=1 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            denom = (st["v"] / bc2).sqrt().add_(self.eps)
            step_size = eta * self.lr / bc1

            p.addcdiv_(st["m"], denom, value=-step_size)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class MomentumOptimizer:
    """SGD with momentum; same open regularization slot."""
    def __init__(self, params, lr, momentum=0.9, reg=0.0, schedule=None):
        self.params = list(params)
        self.lr, self.momentum, self.reg = lr, momentum, reg
        self.schedule = schedule or (lambda step: 1.0)
        self.state = {id(p): dict(step=0, m=torch.zeros_like(p)) for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])

            # TODO: incorporate the chosen regularization mechanism here.
            pass

            st["m"].mul_(self.momentum).add_(g, alpha=eta * self.lr)
            p.add_(st["m"], alpha=-1)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def train(model, loader, optimizer, loss_fn, epochs):
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
```
