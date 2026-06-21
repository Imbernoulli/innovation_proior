The task is to edit an already-pretrained classifier $f(\cdot;w)$ so that it forgets one whole class — the forget set $D_f$ — while keeping its accuracy on the retained classes $D_r$, and to do this with a fixed budget: a frozen `Adam(lr=0.001)`, twenty epochs of an unlearning loop, and one retain minibatch plus one forget minibatch handed to me on each step. Retraining from scratch on $D_r$ is the gold standard a method like this is trying to imitate, but it costs a full training run per deletion request, which is exactly the cost we are trying to avoid. The only thing I actually author is the body of one `unlearn_step`. Before I add any active forgetting machinery, I want the cheapest conceivable rule on the board as a measured floor, because that number is what every later rung has to out-do.

So I propose to start with the most conservative possible update: **retain-only finetuning**. Keep training the model on the retained minibatch with the ordinary cross-entropy objective, and ignore the forget minibatch entirely. Each step computes $\mathcal{L} = \mathrm{CE}(f(\text{retain\_x}), \text{retain\_y})$, backpropagates, and takes one Adam step — `forget_x, forget_y` never enter the forward pass. This is pure *passive* unlearning: the only mechanism by which class-0 knowledge could erode is catastrophic forgetting, the hope that reinforcing the other classes long enough lets the representation drift until class-0 competence decays on its own.

What this rung is really for is to pin down the two ends of the problem. The reason to expect it to *preserve utility* is direct: the retain loss is exactly the original training objective restricted to $D_r$, run from weights that already minimize it well, with a small Adam step. There is no force in this loss that pushes the model off the function it already computes on the retained classes — I am only reinforcing what it already does — so `retain_acc` should stay high. Whatever passive finetuning preserves is roughly the utility ceiling later methods can hope to hold, because every active-forgetting term I add from here will only ever *cost* retain accuracy, never add it.

The reason to expect it to *fail at forgetting* is just as direct, and it is the crux that motivates the whole ladder. There is no term anywhere in this objective that pushes against $D_f$; the forget images contribute nothing to the loss and nothing to the gradient. Catastrophic forgetting is the only candidate eraser, but it works against the architecture: the early layers that recognize class 0 compute generic edges and textures that *every* class needs, and retain finetuning has every incentive to keep them sharp because the retained classes depend on them. The retain objective therefore actively *protects* the very features that still recognize class 0. Twenty epochs of Adam is nowhere near enough drift to erase a class the model spent eighty epochs memorizing.

That makes one metric the honest one to watch, because the headline `forget_acc` will mislead here. The forget class is class 0, and the model still predicts class 0 confidently on held-out class-0 images, so `forget_acc` on the test split can read near zero *for the wrong reason* — the class is intact, not erased. The real privacy leak is `forget_mia_auc`: the membership attack compares the model's max-softmax confidence on forget-*train* images (members, seen during the eighty-epoch run) against forget-*test* images (non-members) and reports the Mann-Whitney AUC. A model that genuinely never saw $D_f$ would be equally (un)confident on both, giving AUC $\approx 0.5$; a model that memorized the forget training images is systematically more confident on them. Retain finetuning never touches class 0, so it does nothing to close that train/test confidence gap, and the residual memorization survives — the `(1 - \text{forget\_mia\_auc})$ term is where this rung leaves work undone, and the gap between this rung's AUC and $0.5$ is precisely the privacy work the next rung must do.

The scaffold's `forget_weight = 0.0` field is a placeholder advertising that the forget batch *could* be weighted in; the default leaves it at zero. No new hyperparameters are introduced — the optimizer (`Adam`, `lr=0.001`), batch size (128), and epoch count (20) are all fixed by the harness.

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
