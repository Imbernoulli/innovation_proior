**Problem (from step 1).** Retain finetuning held utility but barely forgot — the membership signal on
`D_f` survived (MIA AUC ≈ 0.45–0.48, essentially un-erased) because nothing in the loss pushed against
the forget set. Forgetting needs an *active* pressure.

**Key idea.** NegGrad+: descend the retain cross-entropy and *ascend* the forget cross-entropy in one
combined loss, `L = retain_loss - forget_weight · forget_loss`. The negated forget gradient makes the
model wrong on `D_f`; the retain term defends the retained classes against the leakage through the shared
trunk.

**Why it forgets but breaks utility.** Cross-entropy to the true label is *unbounded above* — there is no
fixed point to "be wrong," so the ascent term keeps producing gradient while the bounded retain term
saturates near its minimum. The ascent wins the late dynamics, runs the weights off, and corrupts the
shared features the retained classes depend on. Expect `forget_acc → 0` but `retain_acc` to crash below
the step-1 ceiling, sinking `unlearn_score` below the passive baseline. There is also no notion of *how
much* to forget: hard-label ascent overshoots into the confidently-wrong regime, which is itself
MIA-conspicuous — motivating a bounded, reference-anchored forgetting signal next.

**Hyperparameters.** `forget_weight = 0.5` (the standard NegGrad+ balance). One combined backward/step
per call; optimizer/batch/epochs fixed by the harness.

```python
class UnlearningMethod:
    """Descend retain loss while ascending forget loss."""

    def __init__(self):
        self.forget_weight = 0.5

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        forget_x, forget_y = forget_batch
        retain_loss = F.cross_entropy(model(retain_x), retain_y)
        forget_loss = F.cross_entropy(model(forget_x), forget_y)
        loss = retain_loss - self.forget_weight * forget_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "retain_loss": retain_loss.item(), "forget_loss": forget_loss.item()}
```
