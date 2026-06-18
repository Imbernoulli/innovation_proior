**Problem.** Edit an already-pretrained classifier so it forgets one class `D_f` while keeping accuracy
on the retained classes `D_r`, using only a fixed `Adam(lr=0.001)` and 20 epochs of (retain, forget)
minibatch pairs. Retraining from scratch on `D_r` is the gold standard but too expensive per request.

**Key idea (the scaffold default).** The most conservative rule: continue finetuning on the retained
minibatch only, ignore the forget minibatch. This is pure passive unlearning — it hopes catastrophic
forgetting erodes `D_f`'s influence as a side effect of reinforcing `D_r`.

**Why (and why it is only the floor).** The retain cross-entropy is the original objective restricted to
`D_r`, run from weights that already minimize it, so `retain_acc` stays high — this fixes the utility
ceiling later rungs cannot exceed. But there is *no* term pushing against `D_f`: the shared features
that recognize class 0 are load-bearing for adjacent retained classes, so 20 Adam epochs do not erase a
class memorized over 80. The membership signal survives — `forget_mia_auc` stays above 0.5 — and that
residual memorization is the work every later rung must do.

**Hyperparameters.** No new ones; `forget_weight = 0.0` is an unused placeholder. The optimizer (Adam,
`lr=0.001`), batch size (128), and epoch count (20) are fixed by the harness.

```python
class UnlearningMethod:
    """Continue training on retained data only."""

    def __init__(self):
        pass

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        logits = model(retain_x)
        loss = F.cross_entropy(logits, retain_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
```
