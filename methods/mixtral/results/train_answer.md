The constraint I start from is structural. A dense decoder-only Transformer in the Mistral-7B mold — grouped-query attention and a SwiGLU feed-forward block per layer — is strong, and the field has shown that quality keeps climbing as you add parameters: LLaMA-2 goes 7B, 13B, 70B and gets steadily better. But in a dense model *every* parameter touches *every* token, so the feed-forward sub-block of every layer runs the same full weight matrix over each position. That welds two numbers together that I now want to separate: the **total** parameter count, which sets the memory footprint and tracks quality, and the **active** parameter count — what a single token actually runs through — which sets the FLOPs, latency, and serving cost. In a dense model these are identical, so the moment I add capacity to chase the quality of a 70B model I also inherit its per-token compute. The 70B dense model is the quality I want and the serving cost I don't. Scaling the dense baseline therefore cannot answer the real question, which is not "how do I add capacity" but "how do I add capacity *without* adding per-token compute." GShard-style sparse MoE points in the right direction but only replaces the FFN in every other layer and leans on a more elaborate second-expert gating rule; single-expert routing is the cheapest sparse setting but gives a token no blending at all. I want every layer to get the capacity boost, with a gate simple enough to apply everywhere.

I propose Mixtral, a sparse mixture-of-experts Transformer. The idea is to pry the total parameter count apart from the active one by making each token *touch* only a fraction of the model's weights. The place to do this is the feed-forward sub-block: attention mixes across tokens and is comparatively lean, whereas the FFN holds most of the per-layer parameters and FLOPs and, crucially, is applied *independently per token* — position $t$'s FFN never looks at position $t'$. That independence is exactly what makes per-token parameter selection safe there. So instead of one FFN per layer I place $n$ separate expert FFNs $E_0,\dots,E_{n-1}$ and a gating network $G$ that, per token, decides which experts to use. The layer output for token $x$ is the gate-weighted combination

$$y = \sum_{i=0}^{n-1} G(x)_i \cdot E_i(x).$$

The load-bearing observation is that if $G(x)$ is *sparse* — most coordinates exactly zero — then for every $i$ with $G(x)_i = 0$ the term $G(x)_i E_i(x)$ vanishes and $E_i(x)$ need never be computed. So FLOPs are paid only for the experts with nonzero gate. Keeping only $K$ of the $n$ gates nonzero means each token runs through exactly $K$ experts no matter how large $n$ grows. That is the lever: raise $n$ to add total parameters (capacity, memory) while holding $K$ fixed to hold active parameters (compute).

To make $G(x)$ both sparse and learnable, the router is a bias-free linear projection to $n$ logits, $x\cdot W_g$, one per expert; $W_g$ is tiny, just $\text{dim}\times n$. I turn those logits into a sparse normalized gate by keeping the top-$K$ logits, sending the rest to $-\infty$, and softmaxing over what remains:

$$G(x) := \mathrm{Softmax}(\mathrm{TopK}(x\cdot W_g)),\qquad \mathrm{TopK}(\ell)_i = \begin{cases}\ell_i & \ell_i \text{ in top-}K\\ -\infty & \text{otherwise.}\end{cases}$$

The $-\infty$ entries exponentiate to $0$, so the non-top-$K$ gates are exactly zero — those experts are skipped — and the $K$ survivors form a normalized convex combination summing to one. This is exactly the structure I wanted: sparse where it must be, and a clean weighting over the chosen experts. The remaining dial is $K$, which sets active compute. $K=1$ rebuilds a token's hidden state from a single expert with no blending and forces the router into an all-or-nothing bet; $K=2$ is the smallest setting that still blends, letting each token combine two experts' outputs as a learned convex mixture while still touching only a small fraction of $n$. Higher $K$ buys more mixing for linearly more FFN compute, so $K=2$ is the sweet spot. The experts themselves need nothing exotic — each $E_i$ is just the standard SwiGLU FFN the dense model already uses, $\mathrm{SwiGLU}(x) = (\mathrm{SiLU}(xW_1)\odot xW_3)W_2$, instantiated as $n$ independent copies. I also depart from GShard by replacing *every* FFN sub-block with an MoE layer rather than every other one, with attention, RMSNorm, and residuals unchanged and the routing decision made independently for each token at each layer. With $K=2$ the per-token output is

$$y = \sum_{i=0}^{n-1} \mathrm{Softmax}(\mathrm{Top2}(x\cdot W_g))_i \cdot \mathrm{SwiGLU}_i(x).$$

The parameter arithmetic confirms the decoupling pays off. On a Mistral-shaped backbone — dim 4096, 32 layers, 32 heads with 8 KV heads, head_dim 128, FFN hidden_dim 14336, vocab 32000, context 32768 — with $n=8$ experts and $K=2$, attention and embeddings stay shared while the FFN multiplies by eight. The total (sparse) count lands around 47B, but since only two of the eight experts fire per token, the active count is about 13B. So the model carries 47B of capacity in memory while spending roughly 13B of compute per token — about $5\times$ fewer active parameters than a 70B dense model, with the 47B total still *below* 70B. If quality tracks capacity and compute tracks active count, this should match or beat the 70B dense model at a fraction of the per-token cost. The asymmetry to keep in mind is that memory scales with the 47B sparse count and compute with the 13B active count, so the win is on compute and is most attractive in batched serving where routing and expert-gather overhead amortize. In practice the "skip the zero-gated experts" math meets a throughput wall, because different experts receive different numbers of tokens and ragged batches stall a GPU; casting the per-expert FFN computation as large block-sparse matrix multiplications (the Megablocks approach) handles the variable token counts without padding waste, and spreading experts across GPUs via Expert Parallelism scales it further, with load balancing — keeping the router from over-subscribing a few experts — the thing to watch.

```python
import dataclasses
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)        # n SwiGLU FFNs
        self.gate = gate                             # nn.Linear(dim, n, bias=False)
        self.args = moe_args
    def forward(self, inputs: torch.Tensor):         # inputs: (num_tokens, dim)
        gate_logits = self.gate(inputs)              # inputs @ W_g
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += (
                weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
            )
        return results                               # only K experts run per token

def build_moe_ffn(args):
    experts = [SwiGLU(args.dim, args.hidden_dim) for _ in range(args.num_experts)]
    gate = nn.Linear(args.dim, args.num_experts, bias=False)
    return MoeLayer(experts, gate, MoeArgs(args.num_experts, args.top_k_experts))
```
