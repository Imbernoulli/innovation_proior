**Problem (from step 8).** With value embeddings landing 3.2751 in 1530 steps (≈4.41 min), the tanh logit
softcap is a constant chosen for safety and never reconsidered: `logits = 30*torch.tanh(logits/30)`. A cap
of 30 is so loose that `30*tanh(logits/30)` ≈ `logits` for almost the entire short run — the logits never
climb near 30 — so the cap is nearly inert and isn't actually regularizing anything.

**Key idea (lower the cap from 30 to 15).** Halve the softcap. `15*tanh(logits/15)` starts bending much
earlier, so the cap becomes active over the range of logit values the run actually produces. A tighter cap
forces the logit distribution to be more compact — it imposes structure the network would otherwise have to
learn — and in the small-scale, time-starved regime, free imposed structure is a net win.

**Why it works.** When you train for minutes on limited data, the network can't afford to discover all the
useful structure itself; any correct structure handed to it for free is structure it doesn't have to spend
scarce steps learning. Keeping the logit distribution compact is a good inductive bias for next-token
prediction, and forcing it via a tighter cap lets the optimizer spend its gradient budget elsewhere. 15 is
a moderate halving — active across the run without being so tight it distorts the probabilities the data
demands — so val_loss holds under the bar while the step count comes down.

**Change / code.** One literal edit, the constant 30 → 15 in both places.

```python
# before:
logits = 30 * torch.tanh(logits / 30)
# after (this record):
logits = 15 * torch.tanh(logits / 15)
```
