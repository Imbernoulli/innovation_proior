# Context

## Research question

Pre-training a large language model is one long sequence of optimizer steps: hundreds of thousands of parameter updates, each consuming a forward and backward pass over a large minibatch. The wall-clock cost and dollar cost of a training run are, to first order, proportional to the number of steps multiplied by the cost per step. A flagship model can take months on thousands of accelerators and cost millions of dollars. Consequently, a better optimizer is not a marginal convenience — it is a direct lever on time and money.

The precise problem is: **find an update rule that reaches a target pre-training loss in substantially fewer steps than the incumbent, while keeping the average per-step compute and memory essentially unchanged.** Curvature information (the Hessian) is known to accelerate optimization in terms of step count, but classical ways of using it cost more per step than a plain gradient. The right yardstick is run-time (or total compute, or steps at fixed per-step cost) to reach a fixed loss — not loss at a fixed step, which can mislead when per-step costs differ.

## Background

**Heterogeneous curvature.** The loss surfaces of deep networks, and Transformers in particular, do not have uniform curvature. Empirical Hessian studies (Sagun et al. 2016; Ghorbani et al. 2019; Yao et al. 2020, PyHessian; Zhang et al. 2020) report eigenvalue spectra that span many orders of magnitude. A direct measurement on a 125M-parameter Transformer language model shows the distribution of positive diagonal Hessian entries to be widely dispersed: some parameter directions are very sharp, others very flat. An optimizer with a single shared learning rate must satisfy the sharpest direction (or diverge there), which starves the flat directions of progress.

**Why curvature helps, in principle.** For a convex quadratic, the optimal per-coordinate learning rate is the inverse of that coordinate's curvature; the ideal preconditioner is the Hessian itself (Boyd & Vandenberghe 2004). Newton's method, θ ← θ − H⁻¹∇L, moves each direction to the minimum of the local quadratic in one step, and its convergence does not degrade with the ratio of largest to smallest curvature (the condition number). This is exactly the property a heterogeneous landscape calls for.

**Why curvature is challenging in practice.** Three obstacles face the naive use of the Hessian in deep learning. (1) *Size and cost*: H is d×d for d parameters; forming, storing, or inverting it is impossible at LLM scale, and even cheaper structured approximations are expensive if recomputed every step. (2) *Indefiniteness*: away from a minimum the loss is non-convex, so H has negative eigenvalues; along a direction of negative curvature the Newton step −g/h points uphill, toward a maximum or saddle. (3) *Non-stationarity*: H changes rapidly along the trajectory, so a step extrapolated from a stale or local quadratic model can be badly wrong. Classical numerical optimization addresses (2) and (3) with trust regions (Conn et al. 2000), backtracking line search (Boyd & Vandenberghe 2004), and cubic regularization (Nesterov & Polyak 2006).

**Tools for probing curvature.** Automatic differentiation supports more than the plain gradient. A Hessian-vector product H u for an arbitrary vector u is available by differentiating the scalar ⟨∇L, u⟩ a second time (a double backward), at the cost of a few gradients and without ever forming H (Pearlmutter 1994). Hessian-free and randomized-probing techniques in numerical linear algebra (Hutchinson 1989; Roosta-Khorasani & Ascher 2015) use such matrix-vector products with random vectors to probe matrices that are too large to materialize. Separately, for a composed loss ℓ = ce(f(θ,x), y) the Hessian can be decomposed by the chain rule, and the Gauss-Newton / Fisher literature (Schraudolph 2002; Martens 2010, 2020; Pascanu & Bengio 2013; Sankar et al. 2021) studies such decompositions and their statistical structure; classical identities (Bartlett 1953) relate score, Fisher information, and the Hessian of a negative log-likelihood.

**Sign-based and clipped updates.** A standard simplification of Adam, used for analysis (Balles & Hennig 2018; Bernstein et al. 2018; Kunstner et al. 2023), is SignGD: dropping the moving averages, Adam's update reduces to η·sign(∇L), an idea tracing back to RProp (Riedmiller & Braun 1992) and RMSProp (Hinton 2012). Separately, gradient clipping by global norm is standard practice in LM pre-training (Merity et al. 2017; Radford et al. 2019; Zhang et al. 2022) for stability, and per-coordinate gradient clipping has been observed to behave like adaptivity (Zhang et al. 2020; Crawshaw et al. 2022).

## Baselines

**SGD / GD.** Update θ ← θ − η ∇L. On the toy two-coordinate problem L(θ₁,θ₂)=L₁(θ₁)+L₂(θ₂) with L₁ sharp (curvature h₁) and L₂ flat (curvature h₂≪h₁), the optimal learning rates are ≈1/h₁ and ≈1/h₂. A single shared η must be ≤≈1/h₁ to avoid diverging in the sharp direction, so the flat direction converges at a rate set by the condition number h₁/h₂. On Transformers, SGD underperforms Adam by a wide margin (Liu et al. 2020; Kunstner et al. 2023).

**Adam / AdamW.** m ← β₁m + (1−β₁)g; v ← β₂v + (1−β₂)g²; θ ← θ − η·m̂/(√v̂+ε), with decoupled weight decay (Loshchilov & Hutter 2017). The preconditioner uses only first-order information (a running estimate of the gradient's second moment). Dropping the averages reduces it to SignGD, θ ← θ − η·sign(∇L): the update has magnitude η in every coordinate. On a quadratic, idealized SignGD has a convergence rate that grows with the square root of the condition number.

**Newton's method.** θ ← θ − η H⁻¹∇L. Optimal for convex quadratics; the per-coordinate update g/h equalizes loss decrease across curvatures and the rate is condition-number-free. H is d×d, so the remedies for indefiniteness and non-stationarity (trust region, line search, cubic regularization) add cost.

**Diagonal-Hessian preconditioners (e.g. AdaHessian, and earlier Becker & Le Cun 1988; Schaul et al. 2013).** Use a running estimate of diag(H) — typically via the Hutchinson estimator — as the preconditioner, often with spatial averaging and an EMA across steps (Yao et al. 2021, AdaHessian; Jahani et al. 2021). This captures per-coordinate curvature with the memory of a single extra vector.

**Structured second-order methods (K-FAC, Shampoo).** Approximate the Hessian or Fisher with Kronecker-factored block structure (Martens & Grosse 2015; and follow-ups) or with full-matrix preconditioners over reshaped gradients (Gupta et al. 2018, Shampoo). These methods involve factor inversions and curvature updates.

**Searched first-order optimizers (Lion).** A symbolically discovered sign-momentum update (Chen et al. 2023). Substantially faster than Adam on vision Transformers and diffusion models, with more limited gains on language models.

## Evaluation settings

The natural testbed is autoregressive language-model pre-training with decoder-only Transformers. Models: GPT-2-style architectures across a range of sizes (tens of millions up to a few billion parameters), with the standard configuration (GELU activations, biases and dropout disabled during pre-training) using a nanoGPT-style codebase, and larger NeoX-style models. Data: OpenWebText (tokenized with the GPT-2 tokenizer; standard train/validation split) for the GPT-2 models, and the Pile for the NeoX models. Training: bfloat16, distributed/sharded data parallelism with gradient accumulation to reach large effective batch sizes, cosine learning-rate schedules with warmup, and global gradient clipping. The primary metric is validation (log-)perplexity / cross-entropy loss, plotted against number of steps, against total compute, and against wall-clock time — comparing optimizers by the run-time (or compute, or steps at matched per-step cost) needed to reach the same loss. Optional downstream probes use a handful of few-shot SuperGLUE subtasks with greedy decoding. Hyperparameters (peak learning rate, β₁, β₂, weight decay, and any optimizer-specific knobs) are selected by grid search on a small model and transferred to larger ones, with peak learning rate re-tuned per size.

## Code framework

The pieces that already exist: a tokenized text corpus and batch iterator, a decoder-only Transformer, a cross-entropy loss over the vocabulary, automatic differentiation that yields the averaged minibatch gradient (and supports Hessian-vector products via a second backward), and a generic optimizer/training-loop skeleton. The contribution will occupy the optimizer's per-parameter `step` and whatever extra bookkeeping it needs.

```python
import torch
from torch.optim.optimizer import Optimizer

# --- existing: model, data, loss ---
model = GPT(config)                      # decoder-only Transformer (logits over vocab V)
get_batch = make_loader(data, block_size, batch_size)
# loss: F.cross_entropy(logits.view(-1, V), targets.view(-1))

class TheOptimizer(Optimizer):
    """Per-parameter update rule to be designed."""
    def __init__(self, params, lr, **hparams):
        defaults = dict(lr=lr, **hparams)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_curvature(self):
        # Refresh whatever per-coordinate curvature estimate the method keeps,
        # from gradients already populated on the parameters.
        pass

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                # TODO: maintain whatever per-parameter running state the method
                #       needs, form the per-parameter update, and apply it to p.
                pass

def estimate_curvature(model, X):
    # TODO: cheap per-coordinate curvature estimate (cost ~ one gradient),
    #       to be designed.
    pass

# --- existing: training loop skeleton ---
opt = TheOptimizer(model.parameters(), lr=peak_lr)
for it in range(max_iters):
    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # global norm clip (stability)
    # TODO: any extra per-step bookkeeping the method needs before stepping.
    opt.step()
    opt.zero_grad(set_to_none=True)
```
