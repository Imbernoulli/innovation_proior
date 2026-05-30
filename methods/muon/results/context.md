# Context

## Research question

Training large language models is dominated by the cost of the optimizer's work over trillions of
tokens, and one optimizer — AdamW — is the near-universal default. The question is whether a
fundamentally better update rule exists for the bulk of a transformer's parameters: the 2D weight
matrices of the attention and feed-forward layers, which hold the overwhelming majority of the
trainable weights and the compute.

The pain point is structural. AdamW maintains, per scalar weight, a running first and second moment
and steps with `m̂/√v̂`; with the moving averages switched off this is just `sign(g)`. It therefore
treats a weight **matrix** as a flat bag of independent scalars and normalizes the gradient one
entry at a time. But a hidden weight matrix is not a bag of scalars — it is a **linear operator** on
the layer's input/hidden space, and what we actually care about is how the update changes that
operator's action. An entrywise rule is blind to the matrix's singular structure.

This matters because of an observed property of the updates themselves: for the 2D parameters of a
transformer, the SGD-momentum and Adam update matrices have very high condition number — they are
close to low-rank. A few singular directions carry almost all the magnitude of the update; the many
small-singular-value directions, which can still be important for learning, receive almost no step.
A satisfactory answer would (a) define a notion of a "unit-sized" step that respects the matrix's
operator structure rather than its individual entries, (b) spread the update across all singular
directions instead of letting a few dominate, (c) be cheap enough to run every step at LLM scale on
GPUs in low precision, and (d) keep AdamW's practical conveniences — a stable update magnitude and
transferable learning-rate / weight-decay settings.

## Background

**SGD with momentum.** The workhorse first-order update accumulates an exponential moving average
of the gradient, `M_t = μ M_{t-1} + G_t`, and steps `W_t = W_{t-1} − η M_t`. Momentum smooths
mini-batch noise and accelerates along consistent directions (Polyak; Nesterov's look-ahead variant
evaluates the gradient after a partial momentum step, Sutskever et al., 2013). The raw momentum
update inherits whatever anisotropy the gradient has.

**Adam (Kingma & Ba, 2015) and AdamW (Loshchilov & Hutter, 2019).** Adam keeps per-coordinate
first and second moments `m_t, v_t` and steps with the bias-corrected ratio `m̂_t/(√v̂_t + ε)`,
which normalizes each coordinate by its own recent gradient scale. A useful empirical property is
that its update RMS sits around 0.2–0.4 regardless of the parameter, which is why one learning rate
works across very differently shaped tensors. AdamW changes Adam's L2 penalty into **decoupled
weight decay**: the decay `W ← (1 − ηλ) W` is applied directly to the weights, not folded into the
gradient and thus not divided by `√v̂`, which makes the regularization independent of the adaptive
scaling. With the moving averages removed, Adam reduces to sign gradient descent, i.e. a strictly
entrywise rule.

**Steepest descent under a norm.** A unifying way to read a first-order optimizer is as the
solution to a local model of the loss: minimize over the update `ΔW`
`⟨G, ΔW⟩ + (λ/2)‖ΔW‖²` for a chosen norm `‖·‖` and sharpness `λ`. The choice of norm fixes the
optimizer. Different norms induce different updates; for matrix parameters the relevant norms are
the **induced operator norms** `‖M‖_{α→β} = max_{x≠0} ‖Mx‖_β/‖x‖_α`, which depend on what norms one
puts on the input and output spaces. This perspective (Bernstein & Newhouse, 2024) makes explicit
that an entrywise rule corresponds to a particular, arguably accidental, choice of norm on the
flattened weights, and that other norms — ones that respect the operator structure — are available.

**Preconditioned / second-order-flavored methods.** Shampoo (Gupta et al., 2018) maintains left and
right preconditioners `L_t = Σ G G^T`, `R_t = Σ G^T G` and steps with
`W ← W − η L_t^{-1/4} G R_t^{-1/4}`, a Kronecker-factored approximation to a full-matrix
preconditioner. It captures correlations across rows and columns that an entrywise method cannot,
but it must form and invert (fourth-root) the `A×A` and `B×B` preconditioners every so often, which
is `O(A³ + B³)` compute and `O(A² + B²)` memory per matrix — prohibitive at LLM scale.

**The matrix sign function and the polar decomposition.** Any real matrix `M` factors as
`M = U Σ Vᵀ` (SVD). Its orthogonal **polar factor** is `U Vᵀ` — the SVD with the singular values
replaced by ones — equal to `(M Mᵀ)^{-1/2} M`. This is the **matrix sign** of `M` in the sense of
applying `sign` to each singular value, and it is exactly the **closest semi-orthogonal matrix** to
`M` in Frobenius norm. Numerical-analysis texts (Higham) compute such functions without an SVD via
**Newton–Schulz iterations**: iterate an odd matrix polynomial — e.g. the cubic
`X_{k+1} = 1.5 X_k − 0.5 X_k X_kᵀ X_k`, equivalently applying `f(s) = 1.5 s − 0.5 s³` to each
singular value — which, once the matrix is normalized so its singular values lie in the basin of the
fixed point at 1, drives all singular values toward 1 using only matrix multiplications. Such
iterations are stable in low precision because they are bounded and matmul-only, unlike
inverse-root Newton iterations which require higher precision and misbehave on near-singular
matrices.

**Diagnostic observation on update spectra.** A direct measurement motivating the whole line: the
update matrices produced by entrywise optimizers on transformer 2D weights are nearly low-rank
(high condition number), so the effective rank of each step is much smaller than the matrix's
dimension. Flattening the singular spectrum of the update — giving every direction a comparable
step — is the concrete intervention this suggests.

## Baselines

**SGD with momentum.** `M_t = μ M_{t-1} + G_t`; `W_t = W_{t-1} − η M_t`. Cheap, one state buffer.
*Gap:* the update keeps the gradient's anisotropy, so a few dominant directions absorb the step and
it is sensitive to per-layer scale; needs careful learning-rate tuning per tensor shape.

**Adam / AdamW.** Per-coordinate `m̂/(√v̂+ε)` with decoupled weight decay (AdamW). *Core math:*
`m_t = β₁ m_{t-1} + (1−β₁) g`, `v_t = β₂ v_{t-1} + (1−β₂) g²`, update `m̂_t/(√v̂_t+ε)`, then
`W ← (1−ηλ)W − η·update`. *Strengths:* stable update RMS (~0.2–0.4), transferable hyperparameters,
two cheap state buffers. *Gap:* purely entrywise — it normalizes each scalar in isolation and
ignores that the weight is a matrix operator, so it cannot equalize the update across the matrix's
singular directions.

**Shampoo.** Kronecker-factored preconditioning `W ← W − η L^{-1/4} G R^{-1/4}` with
`L = Σ GGᵀ`, `R = Σ GᵀG`. *Strength:* genuinely matrix-aware; captures row/column correlations.
Notably, **with the accumulators disabled** it reduces to `(GGᵀ)^{-1/4} G (GᵀG)^{-1/4} = U Vᵀ` — an
orthogonalized update. *Gap:* forming and inverse-fourth-rooting the two preconditioners is
`O(A³+B³)` time and `O(A²+B²)` memory per matrix and needs higher precision — too expensive to run
at every step for billion-parameter models.

## Evaluation settings

- **Pre-training corpus and scale.** Large-scale autoregressive language-model pre-training on a
  multi-trillion-token corpus; dense and Mixture-of-Experts transformer architectures (e.g. a
  DeepSeek-V3-style MoE). Parameter counts from sub-billion (scaling-law sweeps) to ~16B total /
  ~3B activated.
- **Scaling-law protocol.** Under a fixed compute budget `C ≈ 6ND` (N parameters, D tokens),
  fit the compute-optimal model size, token count, learning rate, and batch size, and compare
  optimizers at matched compute; the natural yardstick is validation/training loss versus training
  FLOPs (and the FLOPs ratio at equal loss).
- **Downstream benchmarks.** Standard LLM evaluation suites — English knowledge/reasoning (MMLU,
  MMLU-pro, BBH, TriviaQA), code (HumanEval, MBPP), and math (GSM8K, MATH) — used to compare models
  at matched or differing training compute.
- **Diagnostic measurements** tracked during training: per-matrix update RMS and weight RMS, layer
  output RMS, gradient norm, maximum attention logit, and the singular-value distribution of weight
  matrices.
- **Optimizer-cost yardsticks.** Optimizer state memory (number of buffers per parameter),
  per-step communication volume in a sharded (ZeRO-1) setting, and optimizer latency as a fraction
  of the forward/backward pass.

## Code framework

The pieces already in hand: a data pipeline producing token batches, a transformer model whose
parameters split by role (2D weight matrices vs. embeddings, the output head, and 1D gains/biases),
an autoregressive loss, a known per-coordinate adaptive optimizer (Adam/AdamW) with decoupled weight
decay, and a training loop. The optimizer below is a generic `step()` skeleton with one empty slot
where a *new* matrix-aware update rule will go — the part to be designed.

```python
import torch


def adaptive_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    """Known per-coordinate adaptive update (AdamW-style, decoupled decay applied separately)."""
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    m_hat = exp_avg / (1 - betas[0] ** step)
    v_hat = exp_avg_sq / (1 - betas[1] ** step)
    return m_hat / (v_hat.sqrt() + eps)


def matrix_update(grad, momentum, momentum_coef):
    """The matrix-aware update for a 2D weight matrix — TO BE DESIGNED.

    Inputs: the current gradient, a running momentum buffer, the momentum coefficient.
    Returns: the update to subtract from the weight matrix.
    """
    # TODO: form a momentum of the gradient.
    # TODO: turn that momentum matrix into the update we actually want for a matrix-shaped
    #       parameter (the contribution of this work).
    # TODO: any shape-dependent rescaling needed to keep the update magnitude consistent.
    pass


class MatrixAwareOptimizer(torch.optim.Optimizer):
    """Generic optimizer: route each parameter to either the new matrix-aware rule or the
    known adaptive rule, then apply decoupled weight decay and step. Per-group `use_matrix_rule`."""

    def __init__(self, param_groups):
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_matrix_rule"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = matrix_update(p.grad, state["momentum_buffer"], group["momentum"])
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = adaptive_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                             state["step"], group["betas"], group["eps"])
                p.mul_(1 - lr * wd)                 # decoupled weight decay
                p.add_(update.reshape(p.shape), alpha=-lr)
        return loss


# Parameter routing decided up front: 2D hidden weight matrices vs. everything else
# (embeddings, output head, 1D gains/biases). Which rule serves which group is part of the design.
```
