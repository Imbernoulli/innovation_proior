When I set out to understand load balancing in a Mixture-of-Experts model, I forced myself to start
with the case that has no load balancing at all, because I realized I had no way to judge any fix
without first measuring the disease. A MoE layer routes each token, through a small softmax router,
to its top two of eight experts, and the router is trained by nothing but the language-model
cross-entropy. The trouble is that the cross-entropy is a sum over tokens of how well each next
token was predicted, and a single token has no reason to care whether the expert it used is shared
by a million other tokens or by ten — it cares only that the expert it used predicted well. There is
no term anywhere in that loss that sums over experts and asks whether the usage is spread out. Load
balance is a global property of the routing distribution, and the per-token loss is structurally
blind to it.

That blindness has a direction. If, early in training, the router by chance sends a few extra tokens
to one expert, that expert gets more gradient, trains faster, becomes more useful, and so the router
learns to send it even more tokens. It is a positive feedback loop with nothing to oppose it, and
its attractor is collapse: a few experts soak up the traffic while the rest receive almost no
gradient, never specialize, and become dead weight. So the method here is deliberately to do nothing
— the load-balancing loss is a literal zero — and to measure what falls out.

I measure two things on held-out data. The first is the cross-entropy itself, read also as
perplexity. The second is the load imbalance, which I define as the L1 deviation of the token
allocation from uniform: take the fraction of routed assignments each expert receives, and sum half
the absolute gaps between those fractions and the uniform share. That number is zero for a perfectly
balanced router and climbs toward its ceiling as routing concentrates. I combine the two into the
single fitness used to score every later method — the negative of cross-entropy plus imbalance — so
that fixes are judged on the joint point and not on imbalance alone, because crushing the imbalance
by wrecking the router would be a hollow win. What this control establishes is the honest floor: a
tolerable cross-entropy sitting on a clearly skewed allocation, which is exactly the signature that
says the cure must come from outside the cross-entropy, from a term added by hand that looks at the
routing distribution and pushes it toward uniform.

```python
import torch

def balance_loss(probs_list, topi_list, N):
    """Control: no load-balancing loss. The router, trained only by the LM
    cross-entropy, is free to collapse onto a few experts.

    probs_list[l]: [tokens, N]  router softmax mass P (differentiable)
    topi_list[l]:  [tokens, K]  chosen expert ids per token (hard counts -> f)
    """
    return torch.tensor(0.0)


# --- the two router statistics every balancing loss is built from ---
def layer_f_P(probs, topi, N):
    """f_i = fraction of (token, slot) assignments to expert i (hard, non-diff);
       P_i = mean router probability mass on expert i (differentiable)."""
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()
    P = probs.mean(0)
    return f, P


def load_imbalance(f):
    """L_imb = 0.5 * sum_i |f_i - 1/N|  (L1 deviation from uniform; 0 = balanced)."""
    N = f.numel()
    return 0.5 * (f - 1.0 / N).abs().sum()
```
