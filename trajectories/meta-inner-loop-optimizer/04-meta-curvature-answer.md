**Problem.** Across the ladder I have learned *where to start* (MAML), a per-coordinate *rate*
(Meta-SGD), and *which parameters* adapt (ANIL) — but never the *direction*. Every inner step still
follows the raw gradient (Meta-SGD only rescales it diagonally). ANIL's surviving head loop is plain SGD
on an ill-conditioned 1600-dim logistic regression, where the step direction matters most and ten steps
from one example per class are easily wasted on a zig-zag.

**Key idea.** Precondition the inner gradient with a *learned* curvature, meta-trained for
generalization. Replace `θ' = θ − α ∇L` with `θ' = θ − α · MC(∇L)`. For a layer with gradient tensor
`G ∈ R^{C_out×C_in×d}` (`d = kh·kw` for conv, `1` for linear), define three factor matrices `M_o`
(`C_out²`), `M_i` (`C_in²`), `M_f` (`d²`) and transform by three `n`-mode products,
`MC(G) = G ×_3 M_f ×_2 M_i ×_1 M_o` — equivalently the Kronecker product `M_o ⊗ M_i ⊗ M_f`, never
materialized. This captures dependencies among filter elements, input channels, and output channels with
`d²+C_in²+C_out²` numbers instead of `(C_out C_in d)²`.

**Why it generalizes / special cases.** Identity factors ⇒ `MC(G)=G` ⇒ **MAML**; diagonal factors ⇒
per-coordinate rescaling ⇒ **Meta-SGD**; the full Kronecker `M` keeps the off-diagonal coordinate
dependencies a diagonal cannot. The factors are *learned across tasks* (not computed from the support,
which would overfit as Meta-SGD did at 5-shot): the meta-gradient of `M` is `−α ∇_θL_val(θ^τ)·∇_θL_train
(θ)^⊤`, a Fisher-shaped outer product but with separate train/val sets, evaluated at the *adapted*
point, and not forced positive-definite — shaped for held-out loss, not support fitting. Same K-FAC
Kronecker structure but learned and never inverted.

**Hyperparameters / harness constraints.** All factors **identity-initialized** (diagonal factors to
ones) so the method *starts as MAML*. Inner step is differentiable (`autograd.grad` with the graph kept,
re-route `p ← p − α MC(g)`); `meta_parameters()` returns all factors, optimized by the same outer
`Adam(0.003)`; 5 inner steps train / 10 eval. **Budget fix (the one deviation from the textbook
transform):** dense factors are affordable on every conv mode (`C_in, C_out ≤ 64`), but a dense input
factor for the head (`C_in=1600`) would be a `1600×1600 ≈ 2.56M` matrix — 20× the model, over budget
(the cost the factorization exists to avoid). So any mode larger than `MAX_DENSE=256` degrades its factor
to a **diagonal** (per-element rate); the head keeps a dense `5×5` output factor and a diagonal input
rate. Verified: total optimizer state ≈ 80K vs model ≈ 121K, inside the ~291K budget; identity init
reproduces `MC(g)=g`; the inner loop is differentiable and the meta-gradient reaches both θ and all
factors.

**What to watch (no feedback; bar = ANIL's real numbers).** Largest expected gain at **1-shot** — the
most ill-conditioned regime where a preconditioned step should beat the zig-zag — must clear ANIL's
0.4815; failure there would falsify the central claim. At **5-shot** match/edge past ANIL's 0.6366
(miniImageNet) and 0.7138 (CIFAR-FS), since the structured step is a superset of MAML's and Meta-SGD's
and is meta-trained for generalization; the risk to watch is meta-overfitting from the larger factor
count (Meta-SGD's 5-shot failure mode), contained by the identity init and the diagonal fallback.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — finale: Meta-Curvature
class _MetaCurvatureFactor(nn.Module):
    """Per-parameter Kronecker-factored meta-curvature.

    Conv/linear weight G in R^{C_out x C_in x d} (d = kh*kw for conv, 1 for linear):
    MC(G) = G x3 M_f x2 M_i x1 M_o, with M_o, M_i, M_f identity-initialized (so the
    transform starts as the identity and the inner step starts as MAML). A mode
    larger than MAX_DENSE (the classifier head's C_in=1600) degrades its factor to
    a diagonal per-element rate to stay inside the parameter budget; 1D params
    (bias / BatchNorm) carry only the output-mode factor.
    """
    MAX_DENSE = 256  # dense factor only for modes up to this size

    def __init__(self, param: Tensor):
        super().__init__()
        # build factors on the parameter's device/dtype (the model is on CUDA;
        # the harness never .to()'s the optimizer, so factors must inherit here).
        kw = {"device": param.device, "dtype": param.dtype}
        if param.dim() >= 2:                          # conv [Co,Ci,kh,kw] or linear [Co,Ci]
            self.c_out, self.c_in = param.shape[0], param.shape[1]
            self.d = int(param[0, 0].numel()) if param.dim() > 2 else 1
            self.kind = "tensor"
            self.dense_o = self.c_out <= self.MAX_DENSE
            self.dense_i = self.c_in <= self.MAX_DENSE
            self.M_o = nn.Parameter(torch.eye(self.c_out, **kw) if self.dense_o
                                    else torch.ones(self.c_out, **kw))
            self.M_i = nn.Parameter(torch.eye(self.c_in, **kw) if self.dense_i
                                    else torch.ones(self.c_in, **kw))
            self.M_f = nn.Parameter(torch.eye(self.d, **kw))
        else:                                         # bias / BatchNorm: output mode only
            self.kind = "vector"
            self.c_out = param.shape[0]
            self.dense_o = self.c_out <= self.MAX_DENSE
            self.M_o = nn.Parameter(torch.eye(self.c_out, **kw) if self.dense_o
                                    else torch.ones(self.c_out, **kw))

    def transform(self, g: Tensor) -> Tensor:
        if self.kind == "vector":
            return (self.M_o @ g) if self.dense_o else (self.M_o * g)
        G = g.reshape(self.c_out, self.c_in, self.d)
        G = torch.einsum("oid,fd->oif", G, self.M_f)                       # x3 M_f (filter)
        G = (torch.einsum("oid,ji->ojd", G, self.M_i) if self.dense_i      # x2 M_i (input)
             else G * self.M_i.view(1, self.c_in, 1))
        G = (torch.einsum("oid,jo->jid", G, self.M_o) if self.dense_o      # x1 M_o (output)
             else G * self.M_o.view(self.c_out, 1, 1))
        return G.reshape(g.shape)


class InnerLoopOptimizer(nn.Module):
    """MAML inner loop with a learned Kronecker-factored gradient preconditioner
    (Meta-Curvature, Park & Oliva 2019). Identity init => MAML; diagonal => Meta-SGD."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        super().__init__()
        self.inner_lr = inner_lr
        self.mc = nn.ModuleList([_MetaCurvatureFactor(p) for p in model.parameters()])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            updates = [-self.inner_lr * mc.transform(g)
                       for mc, g in zip(self.mc, grads)]
            model = self._apply_update(model, updates)
        return model

    def _apply_update(self, model: nn.Module, updates: List[Tensor]) -> nn.Module:
        # differentiable p <- p + u; re-route parameter tensors so the outer loop
        # backprops into BOTH the initialization theta AND the meta-curvature factors.
        for p, u in zip(model.parameters(), updates):
            p.update = u

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
