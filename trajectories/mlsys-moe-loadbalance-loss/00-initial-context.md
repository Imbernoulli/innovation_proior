## Research question

A Mixture-of-Experts (MoE) layer replaces a dense feed-forward block with `N` expert FFNs and a
small **router** that, for every token, emits a probability vector over the experts and sends the
token to its **top-K** experts. The promise is sparse scaling — capacity grows with `N` while the
per-token compute stays at `K` experts — but the promise has a failure mode baked into it. The
router is trained only by the language-model loss, and the language-model loss has no reason to
spread tokens evenly: a router that learns to send almost everything to a handful of experts lowers
training loss just as well, because those few experts simply get more gradient and more capacity.
Left alone, the router **collapses** — a few experts soak up the traffic, the rest receive almost
no tokens, never train, and become dead weight. Collapse wastes the very parameters the MoE was
built to add, and under expert parallelism it also wrecks throughput, because the most-loaded
expert's device gates the all-to-all.

The single thing being designed is the **load-balancing loss** `L_imb-pen` that is added to the
language-model cross-entropy during training:

```
L_total = L_CE + L_balance
```

`L_balance` is a differentiable penalty, computed from the router's own statistics, whose gradient
pushes the router toward using all experts — without dropping tokens, without changing the
architecture, and (this is the hard part) **without flattening the specialization** that makes a
sparse model worth training in the first place. A penalty strong enough to perfectly equalize usage
also erases the router's learned preferences and hands back a model no better than a dense one of
`K` experts' width. The whole problem lives in that tension: balance the load *and* keep the
experts specialized.

## How the score is defined

Two numbers are measured at the end of training, and the design is judged on their trade-off.

- **Language-model cross-entropy** `L_CE` on held-out data (equivalently perplexity `exp(L_CE)`) —
  the model's actual task quality. A balancing loss that destroys specialization shows up here as a
  *higher* CE.
- **Load imbalance** `L_imb`, the L1 deviation of the token allocation from uniform:

  ```
  L_imb = ½ · Σ_i | f_i − 1/N |,
  ```

  where `f_i` is the fraction of routed (token, slot) assignments that land on expert `i`. `L_imb`
  is `0` for a perfectly uniform router and approaches `1 − 1/N` for a fully collapsed one. Note
  `L_imb` is a *measurement* (the count fractions are non-differentiable); it is not the training
  penalty itself.

Following the ShinkaEvolve evaluation (Lange et al. 2025, arXiv:2509.19349), the two are combined
into a single fitness to be **maximized**:

```
r = − ( L_CE + L_imb ).
```

So a good balancing loss is one that drives `L_imb` toward `0` while keeping `L_CE` as low as a
well-trained model allows — the best `r` is the best joint point, not the lowest imbalance alone.

| Reference point | what it is |
|---|---|
| No balancing (control) | router collapses: low-ish CE, large `L_imb` — the floor for `r` |
| Switch/GShard aux loss (Fedus 2021; Lepikhin 2020) | the standard fix: cuts `L_imb` sharply |
| Global-batch LBL (Qiu et al. 2025, "Demons in the Detail") | same penalty, `f_i` over the global batch — balances while preserving specialization |
| **ShinkaEvolve discovered loss** (Lange et al. 2025) | global-batch term + a novel entropy-weighted under-use hinge — the SOTA endpoint |

## Prior art before the first rung

- **Switch Transformer / GShard auxiliary loss (Lepikhin et al. 2020; Fedus et al. 2021).** The
  load-balancing loss everyone starts from: `L_aux = α · N · Σ_i f_i · P_i`, where `f_i` is the
  fraction of tokens dispatched to expert `i` (the hard count, non-differentiable) and `P_i` is the
  *mean router probability mass* assigned to expert `i` (differentiable). The product `f_i · P_i`
  is minimized, for fixed `f`, when probability is moved off the over-used experts; multiplying by
  `N` makes the uniform optimum `Σ f_i P_i = 1/N · N · (1/N) = 1/N` scale-free, and `α ≈ 1e-2`
  weights it against the CE. *Gap:* `f_i` is computed on the **micro-batch** seen by one
  device/forward — a small, noisy sample. Forcing every micro-batch toward uniform over-constrains
  the router: it penalizes a token-distribution being skewed even when the skew is the legitimate
  topic structure of that micro-batch, so it buys balance by suppressing specialization.

- **DeepSeek auxiliary-loss-free balancing (Wang et al. 2024, arXiv:2408.15664).** Drop the aux
  gradient entirely; instead keep a per-expert bias `b_i` used **only for top-K selection** (added
  to the routing score for ranking, excluded from the gate weights), and nudge it once per step:
  `b_i ← b_i + u · sign( c̄ − c_i )`, where `c_i` is expert `i`'s recent load, `c̄` the mean, and
  `u ≈ 1e-3`. Over-used experts get their bias lowered and lose tokens; under-used experts get
  raised. *Gap:* it is a control loop, not a differentiable objective — it balances counts well but
  gives the router no gradient signal about balance, and on its own it does not address the
  *specialization-vs-balance* trade-off, only the count.

- **Global-batch LBL — "Demons in the Detail" (Qiu et al. 2025, arXiv:2501.11873; Qwen
  global-load-balance blog).** Keep the Switch penalty form `N · Σ_i f_i P_i`, but compute `f_i`
  over the **global batch** (all micro-batches, synchronized) rather than the local micro-batch.
  The insight: balance is a property the whole corpus should satisfy, not every tiny slice. A
  global `f_i` lets any single micro-batch be as specialized as its content demands — a
  code-heavy micro-batch can lean on code experts — as long as usage evens out across the corpus.
  Reported to preserve specialization and lower perplexity at equal balance. *Gap:* it equalizes
  *average* usage but does nothing special for the experts that fall *below* their fair share — the
  dead-expert tail that collapse produces is treated the same as a slightly-cold expert, so the
  rescue of nearly-unused experts is only as strong as the smooth `f·P` gradient there, which is
  weak.

- **Program-evolution search over the loss itself (ShinkaEvolve, Lange et al. 2025,
  arXiv:2509.19349; AutoEvolver, Liu et al.).** Rather than hand-design the penalty, *evolve* the
  Python of the loss function with an LLM-driven evolutionary search, scored by the fitness `r`
  above on real MoE pretraining. *Gap / opportunity:* this is the method that produced the endpoint
  of this ladder — a loss no one wrote by hand — so it is not a gap but the frontier the ladder
  climbs to.

## The fixed substrate

The loss plugs into an otherwise-fixed MoE training stack: token embeddings, `N`-expert top-K MoE
blocks with a linear router producing a softmax probability vector `P` per token, a dispatch/combine
around the expert FFNs, a language-model head, and a cross-entropy objective. From each MoE layer
the router exposes exactly two statistics the loss may use:

- `P_{ℓ,i}` — the **mean router probability mass** on expert `i` in layer `ℓ`, averaged over the
  tokens in scope. Differentiable in the router weights.
- `f_{ℓ,i}` — the **fraction of (token, slot) assignments** routed to expert `i` in layer `ℓ`. A
  hard count; non-differentiable (its gradient must enter through the `P` it multiplies).

`N` (experts), `K` (top-K), the model, the data, the optimizer, and the CE objective are frozen.
Only the function that turns `{P_{ℓ,i}, f_{ℓ,i}}` into the scalar `L_balance` is editable.

## The editable interface

Exactly one function is editable: the per-step balancing loss. Every rung on the ladder is a
different body for it. The control (no balancing) returns zero.

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

The loss must be a single scalar, differentiable in the router parameters through `P`, and must not
drop tokens or alter the architecture. Everything else (how `f`, `P` are computed; the model; the
data) is fixed.

## Evaluation settings

The endpoint is grounded in the ShinkaEvolve MoE evaluation (Lange et al. 2025, arXiv:2509.19349,
Sec. 4.4): a **556M-parameter MoE with `N=64` total experts of which `K=8` are active (82M active
params/token)**, pretrained on **~2.10B FineWeb tokens**, with the balancing loss weighted at
`λ = 0.01` and fitness `r = −(L_CE + L_imb)`; a 2.7B-param / ~30B-token run is used for scaling
validation. The discovered loss is compared against DeepSeek/Qwen global-batch LBL on seven
downstream benchmarks (PIQA, ARC, OpenBookQA, WinoGrande, SocialIQA, CommonsenseQA, HellaSwag).

**This trajectory is a deliberately small reproduction**, not the paper's scale. We train a tiny
Transformer-style MoE (`N=8` experts, top-`K=2`, 2 MoE layers, `d=64`) on a synthetic
latent-topic next-token task with genuine structure for experts to specialize on, a few thousand
steps on CPU. The point is to reproduce the **ordering and mechanism** of the rungs — does the loss
cut `L_imb`, and does the discovered loss do so without sacrificing CE — at a scale that runs in
minutes and reports honest measured numbers. Every reported `L_CE`, perplexity, and `L_imb` is the
mean over 20 fresh held-out evaluation batches of a really-trained model; the fitness is
`r = −(L_CE + L_imb)`. The frozen yardstick is the no-balancing control (router collapses,
large `L_imb`) at the bottom and the discovered entropy-hinge loss at the top.
