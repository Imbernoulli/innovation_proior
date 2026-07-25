The Switch auxiliary loss cut the MoE imbalance sharply, which confirmed that pairing the
non-differentiable token count with the differentiable router probability is the right way to steer
the router toward uniform. But it measured that count over the micro-batch, the small noisy slice one
forward pass sees, and I had argued that demanding uniformity of every slice punishes legitimate
per-slice specialization — a code-heavy slice should lean on code experts, and flattening that skew
costs cross-entropy. The diagnosis pointed at exactly one knob, the scope over which the frequency is
measured, and the method here turns only that knob.

The thing I actually care about is a corpus-level property: across all the data, every expert should
pull roughly its fair share so that none dies and the capacity is used. I do not care that any
particular micro-batch is internally uniform. Specialization is per-slice skew that averages out to
balance over the corpus, and the micro-batch penalty conflates that benign skew with the
pathological skew of collapse. So I keep the penalty form exactly as it was — the same product of
token fraction and mean probability, summed over experts, multiplied by the number of experts,
weighted by the same coefficient — and change only the set of tokens the fraction is computed over,
from the micro-batch to the global batch. Now the penalty asks whether usage is uniform across all
the data and says nothing about whether any individual slice is, which gives the router the freedom
to keep its learned structure while still satisfying the balance constraint.

Alongside this differentiable loss I add a complementary lever that is not a loss at all: DeepSeek's
auxiliary-loss-free bias. It keeps a per-expert bias used only to break ties in the top-K selection —
added to the routing scores for ranking, but excluded from the gate weights that combine the experts
— and nudges it once per step by a small constant in the direction that cools overloaded experts and
warms underloaded ones. It carries no gradient; it is a slow control loop on the counts that runs
beside the penalty, acting on the hard selection surface the smooth gradient is slow to fix. The two
mechanisms are orthogonal, so I run the global-batch loss both alone and with the bias added.

What this method leaves undone, and what I carry forward, is that it equalizes the average usage but
has nothing special to say about the experts that have fallen well below their fair share. To the
smooth penalty, an expert at a tenth of its share and one at nine-tenths are both just small terms;
the gradient that would resurrect a nearly-dead expert is weakest precisely in the tail where rescue
matters most. Balancing the mean is not resurrecting the dying, and that under-utilized tail is the
gap a targeted, discovered term has to attack next.

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

```
