# Meta-Curvature (MC), distilled

Meta-Curvature is an inner-loop adaptation rule for gradient-based meta-learning. It keeps MAML's
two-loop structure but **replaces the raw inner gradient with a learned, Kronecker-factored
preconditioning of it**, meta-trained jointly with the initialization. Instead of `θ' = θ − α ∇L`, the
inner step is `θ' = θ − α · MC(∇L)`, where `MC` transforms each layer's gradient tensor along its three
natural modes (output channels, input channels, filter positions) by three small learned matrices.
Initialized to identity it is exactly MAML. A diagonal full preconditioner is the Meta-SGD direction;
diagonal MC factors are the separable Kronecker version of that idea, so MC should be described as a
structured learned preconditioner rather than an arbitrary Meta-SGD vector.

## Problem it solves

In gradient-based meta-learning the inner step is plain SGD — one global rate, the direction locked to
the raw gradient — and the inner optimizer's choice is known to affect performance. Standard gradient
descent is poor on ill-conditioned loss surfaces; preconditioning the gradient (Newton, natural
gradient, K-FAC) fixes the geometry, but classical curvature is *computed from the task's few support
points*, so it overfits the support set (exactly the few-shot failure mode), and a full preconditioner
is `dim(θ)²` and does not scale. MC supplies a *learned*, *structured-but-cheap*,
*generalization-oriented* preconditioner.

## Key idea

There exist curvatures broadly applicable across a task family (as a shared initialization is). So
*learn* the preconditioner `M` across the task distribution, by the same meta-objective that learns
`θ` — shaping `M` to make the *post-adaptation query* loss generalize, not to fit the support faster.
Afford the otherwise-impossible full `M` by factoring it over each layer's tensor modes.

For a layer with weight/gradient tensor `G ∈ R^{C_out × C_in × d}` (`d = h·w` for conv filters, `d = 1`
for linear), define three meta-curvature matrices `M_o ∈ R^{C_out×C_out}`, `M_i ∈ R^{C_in×C_in}`,
`M_f ∈ R^{d×d}`. The transform is a series of `n`-mode products:

  MC(G) = G ×_3 M_f ×_2 M_i ×_1 M_o.

`×_3 M_f` recombines elements *within* a filter; `×_2 M_i` recombines across input channels; `×_1 M_o`
recombines across output channels (linear combinations of all filters). Products on distinct modes
commute, so the result does not depend on whether the output, input, or filter mode is applied first.
For the `[C_out, C_in, d]` vectorization used here, the same transform can be written as the
never-materialized Kronecker product `M_mc = M_o ⊗ M_i ⊗ M_f`, `vec(MC(G)) = M_mc vec(G)`;
the direct Kronecker factor order is still convention-dependent. This is related to K-FAC's
Kronecker Fisher blocks (`A ⊗ G`), but MC splits the input/filter block into separate `M_i` and `M_f`
(avoiding K-FAC's large `C_in d × C_in d` block) and *learns* the factors for generalization rather
than estimating them from the training loss and inverting them.

## The MC update

  θ' = θ − α · MC(∇L_train(θ)) = θ − α · (∇L ×_3 M_f ×_2 M_i ×_1 M_o),

per layer. The released sinusoid/Omniglot code reuses the same factors across repeated inner steps;
the paper's WRN-feature appendix reports separate meta-curvature matrices for each inner update.
Meta-parameters: the initialization `θ` **and** all the `{M_o, M_i, M_f}` across layers, meta-trained
together.

## Relation to prior methods (special cases)

- **MAML** (Finn et al. 2017): all `M = I` ⇒ `MC(G) = G` ⇒ `θ' = θ − α ∇L`. MC starts here (identity
  init) and deforms away.
- **Meta-SGD** (Li et al. 2017): a learned coordinate-wise vector `α` gives a diagonal full
  preconditioner. MC is best read as a structured, usually non-diagonal preconditioner; if its mode
  factors are diagonal, the resulting diagonal is separable across modes, not an arbitrary Meta-SGD
  vector.
- **K-FAC** (Martens & Grosse 2015): a related Kronecker-factored Fisher preconditioner, but computed
  from training-loss statistics and inverted; MC learns its factors and never inverts.

## Why it generalizes (not just fits faster)

The meta-gradient of a single (flattened) `M`, with inner update `θ^τ = θ − α M ∇L_train(θ)`, is

  ∇_M L_val^τ(θ^τ) = −α ∇_θ L_val^τ(θ^τ) · ∇_θ L_train^τ(θ)^⊤,

an outer product resembling the empirical Fisher `F = E[∇log p · ∇log p^⊤]` — but with three decisive
differences: separate train/val sets (the validation set never enters a within-task Fisher), the
validation gradient evaluated at the *adapted* parameters `θ^τ`, and no forced positive-definiteness (MC
may be indefinite, trading the descent guarantee for freedom to transform the gradient for
generalization).

## Meta-training

Initialize all `M` to identity (so the gradients are unchanged at the start — it begins as MAML); use
Adam for the outer loop; meta-optimize `θ` and all `M` simultaneously. The `n`-mode products are
differentiable and fold into the existing autodiff outer loop — no matrix inversion, and no
higher-order machinery beyond the Hessian-vector product MAML already pays.

## Working code

The canonical released implementation is the author's TensorFlow MAML fork (`silverbottlep/maml`,
commit `e9ce6546bbf2f10a29d699f0302594829a277251`). Its MC path is: `--mc` flag in `main.py:49-50`;
identity/ones factor creation in `maml.py:70-90`; gradient transformation in `maml.py:99-117`;
inner updates in `maml.py:130-151`; Adam meta-optimization in `maml.py:191-195`. That code uses
TensorFlow layouts `[kh, kw, Cin, Cout]` and `[Cin, Cout]`. The PyTorch equivalent below uses the
paper/PyTorch layout `[C_out, C_in, ...]`, so output-mode multiplication appears on the left rather
than as the right-multiply used by the TensorFlow code.

```python
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

INNER_LR = 0.5


class MetaCurvature(nn.Module):
    """Per-parameter Kronecker-factored meta-curvature: identity-initialized factor
    matrices (output / input / filter modes) transform the gradient as
    MC(G) = G x3 M_f x2 M_i x1 M_o before the SGD step."""

    def __init__(self, param: Tensor):
        super().__init__()
        kw = {"device": param.device, "dtype": param.dtype}  # factors on param's device
        if param.dim() >= 2:                       # conv [Co,Ci,kh,kw] or linear [Co,Ci]
            c_out, c_in = param.shape[0], param.shape[1]
            d = int(param[0, 0].numel()) if param.dim() > 2 else 1
            self.c_out, self.c_in, self.d = c_out, c_in, d
            self.M_o = nn.Parameter(torch.eye(c_out, **kw))
            self.M_i = nn.Parameter(torch.eye(c_in, **kw))
            self.M_f = nn.Parameter(torch.eye(d, **kw))
            self.kind = "tensor"
        else:                                      # released code uses elementwise 1D scaling
            self.scale = nn.Parameter(torch.ones_like(param))
            self.kind = "vector"

    def transform(self, g: Tensor) -> Tensor:
        if self.kind == "vector":
            return self.scale * g
        G = g.reshape(self.c_out, self.c_in, self.d)
        G = torch.einsum("oid,fd->oif", G, self.M_f)         # x3 M_f  (filter mode)
        G = torch.einsum("oid,ji->ojd", G, self.M_i)         # x2 M_i  (input mode)
        G = torch.einsum("oid,jo->jid", G, self.M_o)         # x1 M_o  (output mode)
        return G.reshape(g.shape)


class InnerLoopOptimizer(nn.Module):
    """MAML inner loop with a learned Kronecker-factored gradient preconditioner.
    Identity init gives MAML; dense mode factors give the MC transform."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        super().__init__()
        self.inner_lr = inner_lr
        self.mc = nn.ModuleList([MetaCurvature(p) for p in model.parameters()])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), retain_graph=True, create_graph=True
            )
            updates = [-self.inner_lr * mc.transform(g)
                       for mc, g in zip(self.mc, grads)]
            model = self._apply_update(model, updates)
        return model

    def _apply_update(self, model: nn.Module, updates: List[Tensor]) -> nn.Module:
        i = 0
        for p in model.parameters():
            p.update = updates[i]
            i += 1

        def reroute(module: nn.Module) -> nn.Module:
            for name in module._parameters:
                p = module._parameters[name]
                if p is not None and getattr(p, "update", None) is not None:
                    module._parameters[name] = p + p.update
                    p.update = None
            for name in module._modules:
                module._modules[name] = reroute(module._modules[name])
            return module

        return reroute(model)

    def meta_parameters(self) -> List[Tensor]:
        return list(self.mc.parameters())
```

The factors are identity-initialized (so adaptation starts as MAML), the gradient is preconditioned by
three `n`-mode products per weight tensor (a Kronecker-factored transform, never materialized as the
full matrix), the update is differentiable so the outer loop backprops into both `θ` and the `M`
matrices, and `meta_parameters()` hands the factors to the meta-optimizer. For a linear layer the
spatial mode collapses (`d = 1`, `M_f` is `1×1`); 1D bias and batch-norm parameters use an
elementwise scale, matching the released TensorFlow code.
