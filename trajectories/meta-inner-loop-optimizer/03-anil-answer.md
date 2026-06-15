**Problem.** Adding capacity to the *step* (scalar → per-parameter rate) fixed 1-shot but meta-overfit
at 5-shot, because the extra rates tuned body parameters that barely move per task. The ceiling at
5-shot is not the step's expressiveness — it is *which* parameters the inner loop touches at all.

**Key idea.** The success of a MAML-trained net is dominated by **feature reuse**, not rapid learning:
the conv body's features are learned by the outer loop and reused as-is per task; the inner loop barely
changes the body's function. Diagnostics on this exact setup: freezing all four conv layers (everything
but the head) at test leaves MiniImageNet-1shot at 46.3 ± 0.4 vs 46.9 unfrozen; CCA of body
representations pre/post inner loop is > 0.9 (head < 0.5); body weight movement is tiny despite far
more parameters — all from early in training. So the inner loop on the body is a near-no-op. The head
*must* adapt because each five-way task relabels the output neurons. **ANIL: adapt only the head in the
inner loop, freeze the body, at both train and test.** The body stays learnable — only by the outer
loop.

**Why it is not first-order MAML.** First-order MAML keeps the full inner loop but drops the Hessian;
ANIL keeps the second-order machinery and shrinks *which* parameters adapt to the head. A second-order
term **survives**: the head's inner gradient is computed through a forward pass over the body, so
differentiating the query loss w.r.t. the body still differentiates through the head's update. That
retained curvature is why ANIL matches MAML's accuracy at a fraction of the per-task cost (at
evaluation the body is one forward pass; only the linear head iterates 10 steps).

**Hyperparameters (harness).** No learnable optimizer state, so `meta_parameters()` is `[]`. Inner
step uses `l2l.update_module` with a full-length list: $-\alpha g$ for head params (name contains
"classifier"), `zeros_like(p)` for body. Gradient taken w.r.t. head only, `create_graph=True` (keep the
surviving second-order term). **Head re-identified by name each inner step** because `update_module`
replaces parameter objects every call — stale references would make every update silently zero. Same
outer `Adam(0.003)`, 5 inner steps train / 10 eval, `inner_lr=0.5`.

**What to watch.** 5-shot should *recover* what Meta-SGD lost and edge past MAML (freezing the body
reverses the meta-overfitting): miniImageNet 5-shot above 0.6379, CIFAR-FS 5-shot toward/past 0.7067,
CIFAR spread tighter than Meta-SGD's 0.0113. 1-shot should at least hold Meta-SGD's 0.4760 (the head
still gets a second-order-aware inner loop). A 1-shot drop would falsify feature reuse in the thin
regime.

```python
# EDITABLE region of learn2learn/custom_maml.py (lines 177–254) — step 3: ANIL
class InnerLoopOptimizer:
    """ANIL inner-loop optimizer (Raghu et al., 2019).

    Almost No Inner Loop: only adapts the final classification head
    during inner-loop adaptation. The feature extractor backbone is
    frozen, relying on feature reuse from the meta-initialization.
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr
        # Identify head parameters (the last linear layer: classifier)
        # CNN4 structure: features (CNN4Backbone) -> classifier (Linear)
        self._head_param_names = set()
        for name, _ in model.named_parameters():
            if "classifier" in name:
                self._head_param_names.add(name)

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            # Re-identify head params each step because l2l.update_module
            # replaces parameter objects (new ids), so stale references
            # from a previous step would cause all updates to be zero.
            head_params = []
            head_ids = set()
            for name, p in model.named_parameters():
                if name in self._head_param_names:
                    head_params.append(p)
                    head_ids.add(id(p))

            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, head_params, create_graph=True
            )
            grad_map = {id(p): g for p, g in zip(head_params, grads)}
            updates = [
                -self.inner_lr * grad_map[id(p)] if id(p) in head_ids
                else torch.zeros_like(p)
                for p in model.parameters()
            ]
            l2l.update_module(model, updates=updates)
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []
```
