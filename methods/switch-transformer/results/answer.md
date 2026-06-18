# Switch Transformer

## Problem

Dense Transformer FFNs make parameter count and per-token FLOPs grow together. The method decouples them by replacing selected dense FFN sublayers with sparse expert FFNs: a token chooses one expert, runs only that expert, and rejoins the residual stream. Parameter count grows with the number of experts; the dominant per-token expert compute stays approximately that of one FFN.

## Method

For token representation \(x\), the router computes
\[
  h(x)=W_r x,\qquad p_i(x)=\frac{\exp h_i(x)}{\sum_{j=1}^N \exp h_j(x)}.
\]
The token is routed to
\[
  e(x)=\arg\max_i p_i(x)
\]
and the sparse FFN contribution is
\[
  y=p_{e(x)}(x)\,E_{e(x)}(x).
\]
The hard expert choice is not differentiable, but the selected gate value remains in the computation graph, so router gradients flow through \(p_{e(x)}(x)\).

Static-shape execution gives each expert a fixed capacity
\[
  C=\left\lfloor\frac{T}{N}\cdot\text{capacity factor}\right\rfloor
\]
up to implementation padding/minimum-capacity details, where \(T\) is the number of routed tokens in the group and \(N\) is the expert count. Tokens whose chosen expert already has \(C\) earlier tokens are dropped from the sparse FFN; the Transformer block's outer residual path carries them forward unchanged.

For a batch or routing group \(\mathcal B\), define
\[
  f_i=\frac{1}{T}\sum_{x\in\mathcal B}\mathbf 1\{\arg\max_j p_j(x)=i\},
  \qquad
  P_i=\frac{1}{T}\sum_{x\in\mathcal B}p_i(x).
\]
The auxiliary load-balancing term is
\[
  L_{aux}=\alpha\,N\sum_{i=1}^N f_iP_i,\qquad \alpha=10^{-2}.
\]
At uniform routing, \(f_i=P_i=1/N\), so \(\sum_i f_iP_i=1/N\) and \(L_{aux}=\alpha\). The factor \(N\) keeps the uniform-value scale stable as the expert count changes. In the Mesh TensorFlow implementation, `reduce_mean(f * P) * N**2` is the same quantity because `reduce_mean` divides the expert sum by \(N\).

The stability recipe is local: cast router inputs/logits/softmax to float32, build top-1 assignments and combine weights there, then cast combine/dispatch tensors back to the model dtype before distributed expert communication. The paper also uses a 10x smaller initialization scale than the default Transformer scale, multiplicative input jitter for routing exploration, and higher dropout inside experts during fine-tuning.

## Reference-Faithful Code

This is a single-process PyTorch sketch of the paper and Mesh TensorFlow code path. It is intentionally not a performant all-to-all implementation; it preserves the routing math, capacity cases, and auxiliary loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.wo(self.dropout(F.relu(self.wi(x))))


class SwitchTop1Router(nn.Module):
    def __init__(self, d_model, num_experts, capacity_factor=1.25,
                 min_expert_capacity=1, jitter=1e-2,
                 router_dtype=torch.float32):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.jitter = jitter
        self.router_dtype = router_dtype
        self.classifier = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):  # x: [tokens, d_model]
        input_dtype = x.dtype
        router_x = x.to(self.router_dtype)
        if self.training and self.jitter > 0:
            router_x = router_x * torch.empty_like(router_x).uniform_(
                1.0 - self.jitter, 1.0 + self.jitter
            )

        weight = self.classifier.weight.to(self.router_dtype)
        logits = F.linear(router_x, weight)
        probs = F.softmax(logits, dim=-1, dtype=self.router_dtype)
        gate, expert_index = probs.max(dim=-1)

        tokens = x.shape[0]
        capacity = int((tokens * self.capacity_factor) / self.num_experts)
        capacity = max(self.min_expert_capacity, capacity)

        expert_mask = F.one_hot(expert_index, self.num_experts).to(torch.int64)
        # Mesh _switch_gating uses exclusive cumsum: positions are 0, 1, ..., C-1.
        position = (torch.cumsum(expert_mask, dim=0) - 1) * expert_mask
        position = position.sum(dim=-1)
        keep = position < capacity
        gate = gate * keep.to(gate.dtype)

        return gate.to(input_dtype), expert_index, keep, probs


def switch_load_balancing_loss(router_probs, expert_index, num_experts):
    f = F.one_hot(expert_index, num_experts).to(torch.float32).mean(dim=0)
    P = router_probs.to(torch.float32).mean(dim=0)
    return num_experts * torch.sum(f * P)


class SwitchFFN(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, capacity_factor=1.25,
                 expert_dropout=0.1, alpha=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.router = SwitchTop1Router(
            d_model, num_experts, capacity_factor=capacity_factor
        )
        self.experts = nn.ModuleList(
            Expert(d_model, d_ff, expert_dropout) for _ in range(num_experts)
        )

    def forward(self, x):  # x: [batch, sequence, d_model]
        batch, seq, d_model = x.shape
        tokens = x.reshape(batch * seq, d_model)
        gate, expert_index, keep, router_probs = self.router(tokens)

        sparse_update = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            selected = keep & (expert_index == expert_id)
            if selected.any():
                out = expert(tokens[selected])
                sparse_update[selected] = out * gate[selected].unsqueeze(-1)

        aux = self.alpha * switch_load_balancing_loss(
            router_probs, expert_index, self.num_experts
        )
        return sparse_update.reshape(batch, seq, d_model), aux
```

Use it inside a pre-norm Transformer block as
`x = x + dropout(switch_ffn(layer_norm(x))[0])`, and add the returned auxiliary losses from sparse layers to the language-modeling loss. Do not include router z-loss in this reconstruction; it appears in later library variants and optional code paths, not in the core method equations.
