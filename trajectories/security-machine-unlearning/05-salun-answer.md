**Problem (from step 4).** SCRUB is strong but edits *all* weights and must *schedule* a min-max (push on
`D_f`, pull on `D_r`) through the shared trunk to keep forget pressure from corroding the retained classes —
which is exactly why it needs the `msteps` schedule and a small step, and why it sits near the same
instability that collapsed Bad Teacher. Its retain MIA AUC (0.4393 / 0.4648) also leaves forgetting
headroom. The structural fix: cut the leakage at the source by *localizing the edit*.

**Key idea.** SalUn: compute a **weight saliency mask** once from the magnitude of the forget-loss gradient
at the original weights, `m_S = 𝟙(|∇_θ ℓ_f(θ_o)| ≥ γ)` with `γ` = the median (top 50% most forget-salient
weights kept, the rest frozen). On the salient weights, run a *bounded* **random-labeling** forget loss
(fresh uniform random labels on `D_f`, descent toward a finite target — not unbounded ascent) plus a
true-label retain loss, and **mask the gradients** before each optimizer step so only salient weights move.

**Why it beats step 4.** Freezing the non-salient weights protects `D_r` *by construction* — no min-max, no
`msteps`, no teacher to capture — so the shared-trunk interference SCRUB had to schedule around is removed,
which both lifts retain accuracy and damps the high-variance instability. The random-label target is bounded
(cannot explode like NegGrad) and lands the forget set near generalization behaviour, so the forget MIA AUC
should drop toward 0.5.

**Implementation in this harness.** The mask is built lazily on the first step from the first forget
minibatch's gradient *at the original weights* (median thresholding only needs the weight ranking, robust to
minibatch noise), then `zero_grad` so it does not contaminate the first update. The forward uses the full
model; only the *gradient* is masked (`p.grad.mul_(mask[n])`) post-backward, pre-step. `num_classes` is read
from the model output width, so the fill works on cifar10/cifar100/fmnist. `torch` and `F` come from the
harness header — no extra imports.

**Hyperparameters.** `sparsity = 0.5` (`γ` = median of `|∇_θ ℓ_f|`); forget = random-label CE, retain =
true-label CE, equal weight. Optimizer (`Adam`, `lr=0.001`), batch, epochs fixed by the harness.

```python
class UnlearningMethod:
    """SalUn: saliency-masked random-labeling unlearning."""

    def __init__(self):
        self.sparsity = 0.5     # keep the top (1 - sparsity) most forget-salient weights
        self.mask = None        # {param_name: 0/1 tensor}, built lazily at the original weights
        self.num_classes = None

    def _build_saliency_mask(self, model, forget_x, forget_y):
        # Weight saliency = magnitude of the forget-loss gradient at the ORIGINAL weights.
        model.zero_grad()
        loss = F.cross_entropy(model(forget_x), forget_y)
        loss.backward()
        grads = {n: p.grad.detach().abs() for n, p in model.named_parameters() if p.grad is not None}
        flat = torch.cat([g.flatten() for g in grads.values()])
        gamma = flat.quantile(self.sparsity)           # median when sparsity = 0.5
        self.mask = {n: (g >= gamma).float() for n, g in grads.items()}
        model.zero_grad()

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        forget_x, forget_y = forget_batch
        if self.num_classes is None:
            self.num_classes = int(model(forget_x[:1]).shape[1])
        if self.mask is None:
            self._build_saliency_mask(model, forget_x, forget_y)   # at the original weights, first step

        # Bounded forget target: fresh uniform random labels on D_f.
        rand_y = torch.randint(0, self.num_classes, forget_y.shape, device=forget_y.device)
        forget_loss = F.cross_entropy(model(forget_x), rand_y)
        retain_loss = F.cross_entropy(model(retain_x), retain_y)
        loss = forget_loss + retain_loss

        optimizer.zero_grad()
        loss.backward()
        # Localize the edit: zero the gradient of every non-salient (frozen) weight.
        for n, p in model.named_parameters():
            if p.grad is not None and n in self.mask:
                p.grad.mul_(self.mask[n])
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "forget_loss": float(forget_loss.item()),
            "retain_loss": float(retain_loss.item()),
        }
```
