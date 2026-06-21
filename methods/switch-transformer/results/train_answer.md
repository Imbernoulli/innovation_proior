I will describe the Switch Transformer, a sparse expert routing design that decouples total parameter count from per-token feed-forward computation in a Transformer. The canonical name I propose for this method is Switch Transformer, or simply Switch, with top-one expert selection.

A dense Transformer uses the same feed-forward network for every token. Widening that network adds both parameters and FLOPs, and once the layer no longer fits on one device it also adds communication. The idea behind Switch is to keep the per-token FLOPs close to one feed-forward pass while making the total parameter budget much larger by maintaining many separate expert feed-forward networks and activating only one of them for each token. This is a specific form of mixture-of-experts layer, but it is deliberately stripped down: every token is routed to exactly one expert, not two or more, and a single differentiable auxiliary loss keeps the load spread across experts.

For a token representation x, the router first computes logits h = W_r x and then a softmax probability over the N experts. The token is assigned to the expert with the highest probability, so the output contribution from the sparse layer is the selected expert applied to the token, scaled by the selected softmax probability. The hard argmax decision is not differentiable, but the scalar gate that multiplies the expert output is still a smooth function of the router logits, so gradients can reach the router through that gate. This is the key simplification compared with earlier top-k mixture-of-experts layers: the second expert is removed entirely, which cuts the dominant expert FLOPs, the capacity buffer size, and the all-to-all communication roughly in half.

Because the expert choice depends on the input, the number of tokens sent to each expert is random during training. Accelerators prefer static tensor shapes, so each expert is given a fixed capacity. The capacity is computed from the number of routed tokens, the number of experts, and a capacity factor that leaves headroom above the average load. Tokens are placed in their expert's buffer in order of arrival; once an expert's buffer is full, additional tokens assigned to that expert are dropped from the sparse layer. This dropping sounds dangerous, but the sparse layer output is added to the residual stream of the Transformer block, so a dropped token simply receives zero update from that layer and continues forward unchanged. The capacity factor is a tunable tradeoff: larger values drop fewer tokens but waste more padded slots, while smaller values save memory and communication but risk overflow.

Without a balancing mechanism, the router would quickly collapse and send most tokens to a few experts. The auxiliary loss prevents this by coupling two quantities. The first is the hard load f_i, the actual fraction of tokens routed to expert i by argmax. The second is the soft assignment P_i, the average softmax probability mass placed on expert i by the router. The loss is proportional to the sum over experts of f_i times P_i. Because f_i is computed from the argmax it has no gradient, so the stop-gradient is applied to it; P_i is differentiable through the softmax. At uniform routing both quantities equal one over N, and the normalization by N keeps the uniform loss value constant as the expert count changes. The gradient pushes down logits for overloaded experts and pushes up logits for underloaded experts, which stabilizes training without requiring a second expert for counterfactual feedback.

Several smaller design choices are important for stability at scale. The router softmax and the top-one selection are performed in float32, while the surrounding model may run in bfloat16 or another lower-precision format. Only the small router tensors need to move at higher precision, so the extra bandwidth is localized rather than global. The router weight initialization uses a scale ten times smaller than the default Transformer scale, which keeps early router logits small and prevents early routing collapse. During training, small multiplicative jitter is applied to the router input before projection, which can flip borderline decisions and encourage exploration without the high variance of sampling from the softmax. During fine-tuning, experts typically use higher dropout than the dense baseline because the much larger parameter count can overfit small downstream tasks.

The method is evaluated by pretraining a T5-style Transformer on C4 with span corruption, matching FLOPs to a dense baseline of the same compute budget. At equal FLOPs, Switch reaches better negative log perplexity because the extra parameters improve representation quality without increasing the per-token arithmetic. Wall-clock comparisons depend on the capacity factor and the efficiency of the all-to-all dispatch and combine collectives. After pretraining, the model is fine-tuned on standard NLP benchmarks such as GLUE, SuperGLUE, summarization, question answering, and reasoning tasks to confirm that the sparse pretraining transfers well.

The following PyTorch sketch captures the routing math, capacity handling, and auxiliary loss. It is not a distributed implementation; it loops over experts on a single process so that the numerical behavior remains clear.

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

    def forward(self, x):
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

    def forward(self, x):
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

To use this in a Transformer block, apply layer normalization to the input, pass the normalized input through SwitchFFN, add the result back to the residual stream with dropout, and add the returned auxiliary loss from each sparse layer to the language-modeling loss. The result is the Switch Transformer: a sparse feed-forward layer with one expert per token, fixed per-expert capacity, overflow dropping, load-balancing auxiliary loss, and localized float32 precision for the router.
