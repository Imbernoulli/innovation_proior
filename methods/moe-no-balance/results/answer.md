**Problem.** Train a Mixture-of-Experts model and measure, with **no** load-balancing loss at all,
what the router does to the token allocation and to model quality. The MoE has `N=8` experts, top-`K=2`
routing; the router is trained only by the language-model cross-entropy. Score the run by held-out
cross-entropy `L_CE` (perplexity `exp(L_CE)`), the load imbalance `L_imb = ½ Σ_i |f_i − 1/N|`, and
the fitness `r = −(L_CE + L_imb)`.

**Key idea.** There is no idea — this is the control. The cross-entropy objective contains nothing
that rewards spreading tokens across experts, so the router is free to (and does) concentrate
traffic on a few experts. The rung exists to establish the honest floor: the imbalance every later
balancing loss must cut, and the fitness `r` every later rung must beat.

**Why these choices.** Returning a zero balancing loss isolates the failure mode cleanly. Any
imbalance measured here is intrinsic to unregularized MoE routing, not an artifact of a weak
penalty. The same tiny model, data, and optimizer are reused for every later rung so the only
variable is the loss.

**Hyperparameters / contract.** `balance_loss` returns the scalar `0`. The loss must be
differentiable in the router parameters (trivially satisfied) and must not drop tokens or change the
architecture. `N=8`, `K=2`, 2 MoE layers, `d=64`; trained on the synthetic latent-topic next-token
task; `L_imb` and `L_CE` measured on 20 fresh held-out batches.

```python
import torch

def balance_loss(probs_list, topi_list, N):
    """Control: no load balancing. The router is free to collapse."""
    return torch.tensor(0.0)
```
