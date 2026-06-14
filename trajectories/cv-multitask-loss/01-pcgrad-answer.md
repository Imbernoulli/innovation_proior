**Problem.** A shared backbone feeds a fine (100-way) and a coarse (20-way) head; the only knob is how
their two cross-entropy losses combine. The scaffold default sums them, so SGD steps along
`g_fine + g_coarse` on the shared trunk. When the two task gradients conflict (negative cosine), the
sum cancels in the overlap and joint training can land below single-task — and no loss *reweighting*
can fix it, because the cancellation is directional, not a matter of magnitude.

**Key idea (gradient surgery).** Detect conflict by the sign of the inner product, and when
`g_fine · g_coarse < 0`, project each gradient onto the *normal plane* of the other, removing exactly
the component that undoes the other task while keeping the rest:
`g_fine ← g_fine − (g_fine·g_coarse)/‖g_coarse‖² · g_coarse` (and symmetrically for coarse). Step
along the sum of the de-conflicted gradients. When the gradients cooperate (`cos φ ≥ 0`), do nothing,
so positive transfer is preserved. With only two tasks this is a single symmetric projection — no
loop, no random task ordering.

**Why it should help / why it might not here.** Under conflict the de-conflicted update amplifies each
task's own direction, escaping the cancellation that flattens the plain sum; it provably reaches a
lower loss than the summed step when conflict, magnitude dominance, and high curvature co-occur. But
coarse is a *coarsening* of fine, so the two gradients are often correlated — when `cos φ ≥ 0` PCGrad
reduces to the plain sum at the cost of two extra backward passes, and it never touches the
fine/coarse *magnitude imbalance*, which is the more likely bottleneck on this hierarchical pair.

**This task's implementation (not the generic wrapper).** The interface gives only the two scalar
losses and must *return a scalar*, so PCGrad runs entirely inside `forward`: it walks the autograd
graph from `fine_loss` to recover the shared parameters (the interface never hands them over), computes
the two per-task gradients with `torch.autograd.grad(retain_graph=True)`, does the single symmetric
projection, then returns a **surrogate loss** `Σ_p (g_pcgrad_p.detach() · p).sum()` whose backward
pass deposits the de-conflicted gradient into every `p.grad`. No external optimizer wrapper, no
`.grad` read-modify-write, no shuffle.

**Hyperparameters.** None to tune. Conflict threshold is the fixed sign test `dot < 0`;
`1e-12` floors guard the norm divisions; shared parameters are cached after the first graph walk.

```python
# EDITABLE region of pytorch-vision/custom_mtl.py (lines 195-216) — step 1: PCGrad (2-task, in-forward)
class MultiTaskLoss(nn.Module):
    """PCGrad: Gradient Surgery for Multi-Task Learning (Yu et al., 2020).

    Projects conflicting task gradients onto the normal plane of the
    other when their cosine similarity is negative, reducing gradient
    interference between tasks.
    """

    def __init__(self, num_tasks=2):
        super().__init__()
        self._shared_params = None

    def _get_shared_params(self, loss):
        """Extract shared model parameters from the computation graph."""
        if self._shared_params is not None:
            return self._shared_params
        # Walk the computation graph to find leaf parameters
        params = []
        seen = set()
        def _walk(grad_fn):
            if grad_fn is None:
                return
            for child, _ in grad_fn.next_functions:
                if child is None:
                    continue
                cid = id(child)
                if cid in seen:
                    continue
                seen.add(cid)
                if hasattr(child, 'variable'):
                    p = child.variable
                    if p.requires_grad:
                        params.append(p)
                _walk(child)
        _walk(loss.grad_fn)
        self._shared_params = params
        return params

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        params = self._get_shared_params(fine_loss)
        if len(params) == 0:
            return fine_loss + coarse_loss

        # Compute per-task gradients
        grads_fine = torch.autograd.grad(
            fine_loss, params, retain_graph=True, allow_unused=True,
        )
        grads_coarse = torch.autograd.grad(
            coarse_loss, params, retain_graph=True, allow_unused=True,
        )

        # Flatten gradients into vectors
        g0 = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grads_fine, params)
        ])
        g1 = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grads_coarse, params)
        ])

        # PCGrad: project conflicting gradients when cosine similarity < 0
        dot = torch.dot(g0, g1)
        if dot < 0:
            # Project each gradient onto the normal plane of the other.
            # Use originals for both projections (symmetric dot product).
            g0_norm_sq = torch.dot(g0, g0) + 1e-12
            g1_norm_sq = torch.dot(g1, g1) + 1e-12
            g0_proj = g0 - (dot / g1_norm_sq) * g1
            g1_proj = g1 - (dot / g0_norm_sq) * g0
            g0 = g0_proj
            g1 = g1_proj

        # Combined projected gradient
        g_pcgrad = g0 + g1

        # Construct a surrogate loss whose gradient equals g_pcgrad.
        # loss = sum_i (g_pcgrad_i * param_i), so grad w.r.t. param_i = g_pcgrad_i
        offset = 0
        surrogate = torch.tensor(0.0, device=fine_loss.device)
        for p in params:
            numel = p.numel()
            chunk = g_pcgrad[offset:offset + numel].reshape(p.shape).detach()
            surrogate = surrogate + (chunk * p).sum()
            offset += numel
        return surrogate
```
