# Switch Transformer

## Problem

A dense Transformer welds parameter count to per-token compute: every parameter
fires on every token, so adding capacity adds proportional FLOPs. Scaling-law
evidence says parameter count is a separately valuable axis. The Switch
Transformer decouples them — scaling parameters by orders of magnitude while
holding FLOPs per token fixed — by replacing the dense feed-forward sub-layer
with a sparsely-activated bank of expert FFNs and a router that sends each token
to exactly **one** expert. It keeps the capacity-without-compute property of
Mixture-of-Experts while being simpler, cheaper to communicate, and stable enough
to train in bfloat16 at trillion-parameter scale.

## Key ideas

1. **Top-1 (Switch) routing.** The router computes \(p_i(x)=\mathrm{softmax}(W_r x)_i\)
   and sends the token to its single argmax expert, with output
   \(y = p_i(x)\,E_i(x)\). The gate value \(p_i(x)\) multiplies the expert output,
   so the router stays differentiable even with \(k=1\) — refuting the prior
   belief that \(k\ge 2\) is required for a trainable router. Top-1 halves the
   router selection work, the per-expert capacity, and the all-to-all routing
   communication relative to top-2.

2. **Expert capacity with token dropping.** Static accelerator shapes require a
   fixed per-expert buffer:
   \[
     \text{expert capacity} = \Big(\tfrac{\text{tokens per batch}}{\text{number of experts}}\Big)\times \text{capacity factor}.
   \]
   A capacity factor \(>1\) buffers routing imbalance. Tokens that overflow an
   expert are dropped: their FFN is skipped and the representation passes through
   the residual connection unchanged. Switch tolerates low capacity factors
   (1.0–1.25), where memory is scarce, better than top-2 MoE.

3. **Differentiable load-balancing loss (single aux term).**
   \[
     \text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i\, P_i,\qquad
     f_i = \tfrac{1}{T}\!\sum_{x\in\mathcal{B}}\mathbf{1}\{\arg\max p(x)=i\},\quad
     P_i = \tfrac{1}{T}\!\sum_{x\in\mathcal{B}} p_i(x).
   \]
   \(f_i\) is the (non-differentiable) fraction of tokens routed to expert \(i\);
   \(P_i\) is the (differentiable) mean router probability to expert \(i\). The
   product is minimized under uniform routing (\(f_i=P_i=1/N\), giving
   \(\sum_i f_i P_i = 1/N\), so loss \(=\alpha\)); the factor \(N\) makes the loss
   magnitude scale-invariant in the number of experts. Gradient flows through
   \(P_i\), weighted by the observed load \(f_i\), pushing probability away from
   overloaded experts. \(\alpha=10^{-2}\).

4. **Selective precision.** Cast only the router's logits/softmax to float32
   (the numerically fragile op in bf16), then recast the dispatch/combine tensors
   to bfloat16 before the cross-device all-to-all. Stability of fp32, with bf16
   communication cost.

5. **Smaller init + targeted regularization.** Initialize weights from a
   truncated normal with \(\sigma=\sqrt{s/n}\) (fan-in \(n\)) and reduce the
   default scale \(s\) by 10× (to \(\approx0.1\)) to tame hard-switch-amplified
   early variance. During fine-tuning use *expert dropout*: low dropout (0.1) on
   non-expert layers, high dropout (0.4) inside the experts. A small multiplicative
   *input jitter* during training adds exploration to the bandit-like routing.

## Algorithm (per Switch FFN layer)

1. Compute router logits in float32; (training) apply input jitter.
2. Softmax → take argmax expert and its gate value \(p_i(x)\).
3. Assign each token a slot in its expert's buffer via a cumulative count; drop
   tokens whose slot exceeds the capacity (they go to the residual).
4. Dispatch tokens to experts (all-to-all), run each expert FFN, gather outputs
   (all-to-all), scale each by its gate value, scatter home.
5. Add \(\alpha\cdot N\sum_i f_i P_i\) to the training loss.

## Code

Faithful to the canonical implementation (top-1 router, fp32 router softmax,
cumulative-count capacity with overflow drop, and the \(N\sum_i f_i P_i\)
balancing loss).

```python
import torch
import torch.nn as nn


class Expert(nn.Module):
    """Per-token FFN; replicated num_experts times with independent weights.
    expert_dropout is set higher (e.g. 0.4) during fine-tuning."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.wo(self.dropout(self.act(self.wi(x))))


class SwitchTop1Router(nn.Module):
    def __init__(self, d_model, num_experts, capacity_factor=1.25,
                 jitter=1e-2, router_dtype=torch.float32):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter = jitter
        self.router_dtype = router_dtype
        self.classifier = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):                          # x: [tokens, d_model]
        in_dtype = x.dtype
        x = x.to(self.router_dtype)                # selective precision (fp32, local)
        if self.training and self.jitter > 0:      # input jitter -> exploration
            x = x * torch.empty_like(x).uniform_(1.0 - self.jitter, 1.0 + self.jitter)

        logits = self.classifier(x)                # [tokens, N]
        probs = torch.softmax(logits, dim=-1)      # fp32 softmax (the fragile op)
        gate, expert_index = probs.max(dim=-1)     # top-1: gate value + argmax expert

        tokens = x.shape[0]
        capacity = int((tokens / self.num_experts) * self.capacity_factor)

        one_hot = nn.functional.one_hot(expert_index, self.num_experts)
        position = ((torch.cumsum(one_hot, dim=0) - 1) * one_hot).sum(dim=-1)
        keep = position < capacity                 # overflow -> dropped to residual
        gate = gate * keep

        return gate.to(in_dtype), expert_index, keep, probs.to(in_dtype)


def switch_load_balancing_loss(probs, expert_index, num_experts):
    # loss = N * sum_i f_i * P_i ; minimized at uniform routing (f_i = P_i = 1/N).
    f = nn.functional.one_hot(expert_index, num_experts).float().mean(dim=0)  # counts
    P = probs.mean(dim=0)                                                     # mean prob
    return num_experts * torch.sum(f * P)


class SwitchFFN(nn.Module):
    """Sparse drop-in replacement for the dense FFN sub-layer."""
    def __init__(self, d_model, d_ff, num_experts, capacity_factor=1.25,
                 expert_dropout=0.1, alpha=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.router = SwitchTop1Router(d_model, num_experts, capacity_factor)
        self.experts = nn.ModuleList(
            Expert(d_model, d_ff, expert_dropout) for _ in range(num_experts))

    def forward(self, x):                          # x: [batch, seq, d_model]
        b, s, d = x.shape
        x = x.reshape(-1, d)
        gate, expert_index, keep, probs = self.router(x)

        y = torch.zeros_like(x)                    # dropped tokens stay 0 here;
        for i in range(self.num_experts):          # outer residual add is the identity
            sel = keep & (expert_index == i)
            if sel.any():
                y[sel] = self.experts[i](x[sel]) * gate[sel].unsqueeze(-1)

        aux = self.alpha * switch_load_balancing_loss(probs, expert_index,
                                                      self.num_experts)
        return y.reshape(b, s, d), aux
```

In a Transformer block this replaces the dense FFN: `out = x + dropout(SwitchFFN(layer_norm(x)))`,
and the returned `aux` losses from all Switch layers are summed into the total
training loss alongside the language-modeling cross-entropy.
