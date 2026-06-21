## Research question

A Mixture-of-Experts (MoE) layer replaces a dense feed-forward block with `N` expert FFNs and a small **router** that, for every token, emits a probability vector over the experts and sends the token to its **top-K** experts. Sparse scaling lets model capacity grow with `N` while per-token compute stays at `K` experts. The router is trained jointly with the language model by the cross-entropy objective.

In practice, MoE training often couples an auxiliary **load-balancing loss** `L_balance` added to the cross-entropy:

```
L_total = L_CE + λ · L_balance
```

`L_balance` is a differentiable penalty, computed from the router's own statistics, that encourages the router to spread token assignments across experts. What formulations of `L_balance` are effective for this role?

## Prior art / Background / Baselines

- **Switch Transformer / GShard auxiliary loss (Lepikhin et al. 2020; Fedus et al. 2021).** The standard formulation is a penalty proportional to `N · Σ_i f_i · P_i`, where `f_i` is the fraction of tokens dispatched to expert `i` and `P_i` is the mean router probability mass assigned to expert `i`. The product gives a differentiable proxy for load imbalance, and `f_i` is computed on the local micro-batch.

- **DeepSeek auxiliary-loss-free balancing (Wang et al. 2024).** This approach maintains a per-expert bias added only to routing scores for top-K selection, updating it once per step based on recent load: over-used experts lose bias, under-used experts gain bias. The router weights receive no gradient from this balancing mechanism.

- **Global-batch load-balancing loss (Qiu et al. 2025).** This keeps the Switch penalty form but computes `f_i` over the **global batch** rather than the local micro-batch, on the premise that load balance is a corpus-level property.

## Fixed substrate / Code framework

The loss plugs into an otherwise fixed MoE training stack: token embeddings, `N`-expert top-K MoE blocks with a linear router producing a softmax probability vector `P` per token, a dispatch/combine around the expert FFNs, a language-model head, and a cross-entropy objective. From each MoE layer the router exposes exactly two statistics the loss may use:

- `P_{ℓ,i}` — the **mean router probability mass** on expert `i` in layer `ℓ`, averaged over the tokens in scope. Differentiable in the router weights.
- `f_{ℓ,i}` — the **fraction of (token, slot) assignments** routed to expert `i` in layer `ℓ`. A hard count; non-differentiable (its gradient must enter through the `P` it multiplies).

`N`, `K`, the model, the data, the optimizer, and the CE objective are frozen. Only the function that turns `{P_{ℓ,i}, f_{ℓ,i}}` into the scalar `L_balance` is editable.

## Editable interface

Exactly one function is editable: the per-step balancing loss. Every rung on the ladder is a different body for it. The control (no balancing) returns zero.

```python
import torch

N_EXPERTS = 8     # N
TOP_K = 2         # K
# probs_list[ℓ]: [tokens, N]  router softmax (P, differentiable)
# topi_list[ℓ]:  [tokens, K]  chosen expert ids per token (hard, for the counts f)

def layer_f_P(probs, topi, N):
    """f_i = fraction of (token,slot) assignments to expert i; P_i = mean router prob mass."""
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()        # hard counts -> non-differentiable
    P = probs.mean(0)                # mean router probability -> differentiable
    return f, P

# ---- EDITABLE: the load-balancing loss added to L_CE. Default = no balancing. ----
def balance_loss(probs_list, topi_list, N):
    return torch.tensor(0.0)         # control: router is free to collapse
```

The loss must be a single scalar, differentiable in the router parameters through `P`, and must not drop tokens or alter the architecture. Everything else (how `f`, `P` are computed; the model; the data) is fixed.

## Evaluation settings

Two numbers are measured on held-out data after training, and the design is judged on their trade-off:

- **Language-model cross-entropy** `L_CE` — the model's actual task quality. A balancing loss that destroys specialization shows up as a higher CE.
- **Load imbalance** `L_imb`, the L1 deviation of token allocation from uniform:

  ```
  L_imb = ½ · Σ_i | f_i − 1/N |,
  ```

  where `f_i` is the fraction of routed (token, slot) assignments that land on expert `i`. `L_imb` is `0` for perfectly uniform routing and approaches `1 − 1/N` for a fully collapsed one.

The two are combined into a single fitness to be **maximized**:

```
r = − ( L_CE + L_imb ).
```

So a good balancing loss is one that drives `L_imb` toward `0` while keeping `L_CE` as low as a well-trained model allows — the best `r` is the best joint point, not the lowest imbalance alone.

Recent large-scale MoE load-balance evaluations train models on the order of 556M parameters with `N=64` total experts and `K=8` active, using roughly 2B FineWeb tokens and a balancing weight near `λ = 0.01`, and compare baselines on downstream benchmarks. **This trajectory uses a deliberately small reproduction:** a tiny Transformer-style MoE (`N=8`, top-`K=2`, 2 MoE layers, `d=64`) on a synthetic latent-topic next-token task with genuine structure for experts to specialize on, run for a few thousand steps on CPU. Every reported `L_CE`, perplexity, and `L_imb` is the mean over 20 fresh held-out evaluation batches of a really-trained model; the fitness is `r = −(L_CE + L_imb)`. The fixed yardstick is the no-balancing control (router collapses, large `L_imb`) at the bottom and the strongest existing baseline at the top.
