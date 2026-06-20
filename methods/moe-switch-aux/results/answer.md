**Problem.** Same MoE (`N=8`, top-`K=2`), now add the standard Switch/GShard auxiliary
load-balancing loss to the cross-entropy and measure whether it cuts the load imbalance the control
exhibited, and at what cost to cross-entropy. Score by `L_CE` (perplexity), `L_imb = ½ Σ_i |f_i − 1/N|`,
and `r = −(L_CE + L_imb)`.

**Key idea.** Add the differentiable penalty `L_aux = α · N · Σ_i f_i · P_i` with `α = 1e-2`, where
`f_i` is the fraction of (token, slot) assignments dispatched to expert `i` (a hard count) and `P_i`
is the mean router softmax mass on expert `i` (differentiable). For fixed counts `f`, minimizing
`Σ f_i P_i` moves probability mass off the over-used experts — the gradient flows through `P`. The
factor `N` makes the balanced optimum scale-free (`Σ f_i P_i = 1/N`), and `α` sets the weight against
CE. Crucially, `f_i` is computed on the **micro-batch** (here, on 4 micro-splits of the training
batch, averaged), which is the defining property of the classical aux loss.

**Why these choices.** `f·P` is the cheapest differentiable surrogate for "are the experts the
router favors also the ones it overloads"; `f` cannot be differentiated, so the penalty must reach
the router through `P`. `α = 1e-2` is the textbook weight — large enough to break collapse, small
enough not to dominate CE. Computing `f` per micro-batch is what later rungs will diagnose as the
limitation: it over-constrains each slice toward uniform.

**Hyperparameters / contract.** `α = 1e-2`, averaged over `L=2` layers and 4 micro-splits. Same
model/data/optimizer as the control.

```python
import torch

def layer_f_P(probs, topi, N):
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()          # hard counts (non-differentiable)
    P = probs.mean(0)                  # mean router prob mass (differentiable)
    return f, P

def balance_loss_switch(probs_list, topi_list, N, alpha=1e-2):
    """Switch/GShard aux loss: alpha * N * sum_i f_i P_i, f over the micro-batch."""
    total = 0.0
    for probs, topi in zip(probs_list, topi_list):
        f, P = layer_f_P(probs, topi, N)
        total = total + N * (f.detach() * P).sum()   # gradient enters via P
    return alpha * total / len(probs_list)

# At training time f is computed per micro-batch: split the batch into 4 micro-splits,
# evaluate balance_loss_switch on each, and average — the micro-batch locality penalty.
