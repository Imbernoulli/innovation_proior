**Problem.** Same MoE (`N=8`, top-`K=2`). The micro-batch aux loss balances the router but
over-constrains each slice toward uniform, suppressing specialization. Replace it with the
**global-batch** load-balancing loss — same penalty form, frequency `f_i` measured over the whole
batch — and measure `L_CE` (perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`, `r = −(L_CE + L_imb)`.
Optionally pair with DeepSeek's auxiliary-loss-free selection bias.

**Key idea.** Keep `L_LBL = α · N · Σ_i f_i · P_i` (`α = 1e-2`), but compute `f_i` over the
**global batch** rather than the micro-batch. Balance becomes a corpus-level property: any single
micro-batch may be as specialized as its content demands, as long as usage evens out globally. In
this single-process reproduction the training batch *is* the global batch, so this rung computes `f`
on the full batch (vs the Switch rung's 4 micro-splits). The optional DeepSeek loss-free bias `b_i`
is added only to the top-K selection scores (not the gate weights) and updated by
`b_i ← b_i + u · sign(c̄ − c_i)`, `u = 1e-3`, with no auxiliary gradient — a complementary
count-level controller.

**Why these choices.** The penalty form is unchanged; only the *scope* of `f` moves from a noisy
micro-batch sample to the full-batch estimate. That single change is what preserves specialization
(Qiu et al. 2025): it stops penalizing the legitimate skew of individual slices. The loss-free bias
is kept gradient-free by design so it adjusts counts without telling the router anything about
balance through `P`.

**Hyperparameters / contract.** `α = 1e-2`, `f` over the global batch, averaged over `L=2` layers.
Loss-free bias (optional): `u = 1e-3`, selection-only, updated per step from per-expert load.

```python
import torch

def layer_f_P(probs, topi, N):
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()
    P = probs.mean(0)
    return f, P

def balance_loss_global(probs_list, topi_list, N, alpha=1e-2):
    """Global-batch LBL: alpha * N * sum_i f_i P_i, f computed over the GLOBAL batch."""
    total = 0.0
    for probs, topi in zip(probs_list, topi_list):
        f, P = layer_f_P(probs, topi, N)          # f over the full (global) batch
        total = total + N * (f.detach() * P).sum()
    return alpha * total / len(probs_list)

# Optional DeepSeek auxiliary-loss-free bias (selection-only, no aux gradient):
#   biased top-k uses (router_logits + b);  after each step, per layer:
#   f, _ = layer_f_P(probs, topi, N);  c = f * N;  cbar = c.mean()
#   b += u * torch.sign(cbar - c)         # u = 1e-3
