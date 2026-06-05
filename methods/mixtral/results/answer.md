# Mixtral (Sparse Mixture of Experts)

## Problem

In a dense Transformer every parameter is used for every token, so increasing capacity (which improves quality) increases per-token compute and latency proportionally. Decouple **total capacity** from **per-token compute**: grow the parameter count while each token only touches a small, fixed fraction of it.

## Key idea

Replace each layer's feed-forward (FFN) sub-block — applied independently per token — with a **Sparse Mixture of Experts (SMoE)**: $n$ expert FFNs plus a router that, per token, selects $K$ of them via a top-$K$ softmax gate. Zero-gated experts are not computed, so adding experts (raising $n$) grows the **sparse** (total) parameter count without growing the **active** (per-token) parameter count, which is set by $K$.

## Method

For $n$ experts $E_0,\dots,E_{n-1}$ and a gating network $G$, the MoE layer output for token $x$ is
$$y = \sum_{i=0}^{n-1} G(x)_i \cdot E_i(x).$$
The gate is a top-$K$ softmax over a linear router:
$$G(x) := \mathrm{Softmax}(\mathrm{TopK}(x\cdot W_g)),\qquad \mathrm{TopK}(\ell)_i = \begin{cases}\ell_i & \ell_i \text{ in top-}K\\ -\infty & \text{otherwise}\end{cases}$$
so non-top-$K$ gates are exactly 0 (their experts are skipped) and the $K$ surviving gates form a normalized convex combination. Each expert is a standard **SwiGLU** FFN, and **every** FFN sub-block is replaced by an MoE layer (cf. GShard, which uses MoE in alternating layers and a more elaborate second-expert gate). With $K=2$:
$$y = \sum_{i=0}^{n-1} \mathrm{Softmax}(\mathrm{Top2}(x\cdot W_g))_i \cdot \mathrm{SwiGLU}_i(x).$$
$K=2$ blends two experts (softer routing, mutual error-cushioning) at minimal compute.

**Configuration:** dim 4096, 32 layers, 32 heads, 8 KV heads (grouped-query), head_dim 128, FFN hidden_dim 14336, vocab 32000, context 32768, $n=8$ experts, $K=2$. This gives **~47B total parameters** but **~13B active per token** — ~5× fewer active parameters than a 70B dense model. Memory scales with the 47B sparse count, compute with the 13B active count, so it is most efficient in batched serving. Efficient execution uses block-sparse matrix-multiply kernels (Megablocks) and Expert Parallelism across GPUs, where load balancing is the key concern.

## Code

```python
import dataclasses, torch, torch.nn as nn, torch.nn.functional as F

@dataclasses.dataclass
class MoeArgs:
    num_experts: int          # n = 8
    num_experts_per_tok: int  # K = 2

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoeLayer(nn.Module):
    def __init__(self, experts, gate, args: MoeArgs):
        super().__init__()
        self.experts = nn.ModuleList(experts)        # n SwiGLU FFNs
        self.gate = gate                             # nn.Linear(dim, n, bias=False)
        self.args = args
    def forward(self, x):                            # x: (num_tokens, dim)
        gate_logits = self.gate(x)                   # x @ W_g
        weights, sel = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(x.dtype)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_idx, nth = torch.where(sel == i)   # tokens routed to expert i
            out[batch_idx] += weights[batch_idx, nth, None] * expert(x[batch_idx])
        return out                                   # only K experts run per token

def build_moe_ffn(args):
    experts = [SwiGLU(args.dim, args.hidden_dim) for _ in range(args.num_experts)]
    gate = nn.Linear(args.dim, args.num_experts, bias=False)
    return MoeLayer(experts, gate, MoeArgs(args.num_experts, args.top_k_experts))
```
