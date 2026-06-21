Retain finetuning did exactly what its design promised: it held utility and barely forgot. On resnet20-cifar10 `retain_acc` came in at 0.8758 with `forget_mia_auc` 0.4512; on vgg16bn-cifar100, 0.5345 and 0.4765; on the hidden mobilenetv2-fmnist, 0.9373 and 0.4817. Read against the only value that matters for forgetting — $0.5$, the AUC of a model that genuinely never saw $D_f$ — those MIA numbers sit *just below* 0.5, which is the fingerprint of un-erased memorization: the attack barely separates forget-train from forget-test confidences only because the original model already sat near that operating point, not because any forgetting happened. The `unlearn_score` of 0.8082 / 0.6860 / 0.8185 is capped on the forgetting axis, with utility near the harness ceiling. The diagnosis is clean and it dictates the move: I need an *active* forgetting pressure, a term in the loss that actually pushes against $D_f$, because catastrophic forgetting alone does not erode a class memorized over eighty epochs.

The textbook active move, and the one I propose here, is **NegGrad+**: gradient *ascent* on the forget loss, paired with continued descent on the retain loss, in a single combined objective. Where descent on a minibatch makes the model more right, I negate the gradient on $D_f$ and make it more wrong — climb the cross-entropy on the forget examples to drive the model off the class-0 answer it memorized. The combined per-step loss is

$$\mathcal{L} = \mathcal{L}_{\text{retain}} - \beta \cdot \mathcal{L}_{\text{forget}},$$

with $\mathcal{L}_{\text{retain}} = \mathrm{CE}(f(\text{retain\_x}), \text{retain\_y})$, $\mathcal{L}_{\text{forget}} = \mathrm{CE}(f(\text{forget\_x}), \text{forget\_y})$, and $\beta = \text{forget\_weight} = 0.5$ the standard NegGrad+ balance. It is one backward and one step on this summed loss — not two separate updates.

The retain term is not optional, and the reason is the same shared representation that doomed the previous rung. $D_f$ and $D_r$ are not processed by disjoint weights: the lower convolutional layers compute features used by every class, so ascending the forget loss sends gradients back through that shared trunk and perturbs the retained classes too. The retain descent term is the defensive counter-pressure that holds the line on $D_r$ while the forget ascent does its damage on class 0. Pure ascent on $D_f$ alone would have no such guard.

But I have to be honest about the structural danger in the ascent term, because it is what will set this rung's number. Cross-entropy to the true label is *unbounded above*: as the model's predicted probability on the true class 0 approaches zero, $-\log p$ runs to infinity, so the ascent term has no fixed point to settle at — it keeps demanding more probability mass be pushed off class 0, forever. The retain descent term, by contrast, *is* bounded (cross-entropy bottoms out at a finite floor when the model is correct), so the two terms are asymmetric in the late dynamics: the retain term saturates near its minimum and stops producing large gradients, while the unbounded forget-ascent term keeps producing gradient as long as the model is not infinitely wrong. The ascent therefore *wins the late-training dynamics*. With $\beta = 0.5$ the forget term is half-weighted, which slows this but does not change the asymptotics — there is no $\beta$ that turns an unbounded ascent into a bounded one; $\beta$ only rescales how fast the weights run off. And they will run off: the natural endpoint of an unbounded ascent objective is the weights diverging, the representation getting torn apart, and the *retained* accuracy collapsing as collateral, because the very features the ascent corrupts are the shared features the retained classes depend on.

So I expect the diagnostic signature to invert the previous rung's. Retain finetuning failed *softly* — high utility, weak forgetting. NegGrad+ will fail *hard* in the opposite direction: `forget_acc` will go to zero (it absolutely forgets), and the MIA AUC may even drop *below* the prior rung as the model becomes confidently, abnormally wrong on class 0 — but it pays by crashing `retain_acc` well below the 0.8758 / 0.5345 / 0.9373 ceiling. Because the score averages utility and forgetting, a crashed retain term sinks `unlearn_score` below the passive baseline even though forgetting itself succeeds. The deepest, most-shared-trunk architecture, vgg16bn on cifar100, should crash the hardest.

There is a subtler problem on the privacy axis that points straight at the next rung. Driving `forget_acc` to exactly zero — making the model *confidently wrong* — is not what forgetting should look like. A model that genuinely never trained on class 0 does not confidently shout some other class; it sits at generalization-level uncertainty. A model taught a sharp anti-fact about class 0 has not forgotten; it has learned a new inverted competence, and that inverted competence is itself a fingerprint an attacker can read. NegGrad+ has no notion of *how much* to forget, because hard-label cross-entropy ascent has no "stop at generalization-level uncertainty" fixed point. That precise weakness motivates moving, next, to a *bounded, reference-anchored* forgetting signal.

The only hyperparameter I own is `forget_weight = 0.5`; the optimizer, batch size, and epoch count are fixed by the harness. I report `retain_loss` and `forget_loss` alongside the combined `loss` so the dynamics are visible — I expect `forget_loss` to climb without bound across the twenty epochs (the tell that the ascent never settles) and `retain_loss` to creep up as the shared trunk degrades.

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
