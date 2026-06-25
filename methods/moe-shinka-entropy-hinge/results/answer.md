**Problem.** Same MoE (`N=8`, top-`K=2`). Global-batch LBL balances average usage while preserving
specialization, but it gives no extra pull on the experts that have fallen *below* their fair share
— the nearly-dead tail of collapse — because the smooth `f·P` term has a weak gradient there.
Install ShinkaEvolve's **discovered** load-balancing loss, which adds an entropy-weighted
under-utilization hinge to the global-batch term, and measure `L_CE` (perplexity),
`L_imb = ½ Σ_i |f_i − 1/N|`, `r = −(L_CE + L_imb)`.

**Key idea.** Keep the global-batch term and **add** a hinge that fires only for under-used experts,
scaled by how peaked the router is:

```
L = N_E · (1/L) Σ_ℓ Σ_i f_{ℓ,i} P_{ℓ,i}                      [global-batch LBL]
    + (0.1/L) Σ_ℓ s(P_ℓ) · Σ_i max(0, τ − f_{ℓ,i})           [discovered hinge]

  s(P_ℓ) = 0.5 + (1 − H(P_ℓ) / log N_E)        entropy-complement weight
  τ       = 0.064 / N_E                          per-expert usage floor
```

`max(0, τ − f_{ℓ,i})` is nonzero **only** for experts below the floor `τ`, so healthy experts are
untouched. `s(P_ℓ)` reads the router's peakedness: low entropy `H` (collapse-prone) gives a large
`s` and a strong rescue; a near-uniform router gives a small `s` and the hinge barely fires. This is
targeted dead-expert rescue, gated to act exactly when and where collapse is happening, without
flattening a healthy router.

**Why these choices.** This loss was *evolved*, not hand-designed: an evolutionary search over the
Python of the balancing loss under fitness `r = −(L_CE + L_imb)`. The
two pieces are pinned down exactly: first-term coefficient `N_E/L`, hinge
coefficient `0.1/L`, weight `s(P_ℓ)=0.5+(1−H(P_ℓ)/log N_E)`, floor `τ=0.064/N_E`. The hinge gradient
must reach the router, so it is applied through the differentiable `P` of the under-used experts
(the `f`-based gate selects *which* experts are under the floor; the penalty that flows gradient
raises their probability mass).

**Hyperparameters / contract.** Global term coefficient `N_E/L`; hinge coefficient `0.1/L`; floor
`τ = 0.064/N_E`; weight `s(P_ℓ) = 0.5 + (1 − H(P_ℓ)/log N_E)`, `H` the entropy of the normalized
mean router distribution. `N_E = N = 8` experts, `L = 2` layers in this reproduction. Same
model/data/optimizer as every rung. (The loss was originally found at scale: 556M/82M-active MoE, `N_E=64`,
top-8, ~2.10B FineWeb tokens, `λ=0.01`; this is a small reproduction of the mechanism.)

```python
import math
import torch

def layer_f_P(probs, topi, N):
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()
    P = probs.mean(0)
    return f, P

def balance_loss_shinka(probs_list, topi_list, N):
    """ShinkaEvolve discovered loss:
       global-batch LBL + entropy-weighted under-utilization hinge."""
    L = len(probs_list)
    tau = 0.064 / N
    term1 = 0.0
    term2 = 0.0
    for probs, topi in zip(probs_list, topi_list):
        f, P = layer_f_P(probs, topi, N)
        term1 = term1 + N * (f.detach() * P).sum()                 # global-batch LBL
        Pn = P / (P.sum() + 1e-9)
        H = -(Pn * (Pn + 1e-9).log()).sum()                        # router entropy
        s = 0.5 + (1.0 - H / math.log(N))                          # entropy-complement weight
        under = (tau - f.detach() > 0).float()                     # experts below the floor
        # hinge gradient flows through P of the under-used experts (raise their mass)
        term2 = term2 + s.detach() * (under * torch.clamp(tau - P, min=0.0)).sum()
    return (term1 / L) + (0.1 / L) * term2
