Let me start from the thing that actually constrains me. I have a dense decoder-only Transformer — the Mistral 7B recipe, grouped-query attention and a SwiGLU feed-forward block per layer — and it's good. The obvious way to make it better is to make it bigger, and the field has shown quality keeps climbing with parameters: LLaMA-2 goes 7B, 13B, 70B and gets steadily stronger. But in a dense model *every* parameter touches *every* token. So when I add parameters to get more quality, I raise the FLOPs per token and push per-token cost and latency up with it. Quality and per-token cost are welded together. A 70B dense model is the quality I want and the serving cost I don't. So the real question isn't "how do I add capacity" — it's "how do I add capacity *without* adding per-token compute."

What would it even mean to add parameters that a token doesn't pay for? It means each token must only *touch* a fraction of the total parameters. Some parameters are used for this token, others for that token, and the union is large but the per-token slice is small. That immediately splits the parameter count into two numbers that I've been conflating: the **total** parameter count — everything in the model, which sets the memory footprint — and the **active** parameter count — what a single token actually runs through, which sets the compute. In a dense model these are equal. I want to pry them apart: grow total, hold active.

Where in the block should the extra capacity live? Attention mixes across tokens and is comparatively lean; the feed-forward sub-block is where most of the per-layer parameters and FLOPs sit, and crucially it's applied *independently per token* — position $t$'s FFN doesn't look at position $t'$. That independence is exactly what makes it safe to make per-token decisions there. So the FFN is the place to introduce token-dependent parameter selection.

So: instead of one FFN per layer, put $n$ separate FFNs — call them experts $E_0,\dots,E_{n-1}$ — and have each token use only some of them. The classic way to make a layer use a subset of sub-networks per input is a gating network. Let the layer output, for token $x$, be a weighted combination of expert outputs,

  y = Σ_{i=0}^{n-1} G(x)_i · E_i(x),

where $G(x)$ is an $n$-dimensional vector of gate weights. Now the key observation: if $G(x)$ is *sparse* — most of its coordinates are exactly zero — then for every $i$ with $G(x)_i = 0$ the term $G(x)_i E_i(x)$ is zero, and I never have to compute $E_i(x)$ at all. So the FLOPs are paid only for the experts with nonzero gate. If I keep, say, only $K$ of the $n$ gates nonzero, then each token runs through $K$ experts regardless of how large $n$ is. That's the lever: crank $n$ up to add total parameters (capacity, memory), hold $K$ fixed to hold active parameters (compute). Increasing $n$ at fixed $K$ raises capacity at essentially constant per-token cost.

Now I have to make $G(x)$ actually sparse and learnable. The simplest thing that learns *which* experts a token wants is a linear router: project the token to $n$ logits, $x\cdot W_g$, one per expert. To turn logits into a sparse, normalized gate, take the top-$K$ logits, discard the rest, and softmax over what remains. Formally, define $\mathrm{TopK}(\ell)_i = \ell_i$ if $\ell_i$ is among the top-$K$ coordinates of $\ell$ and $-\infty$ otherwise; then

  G(x) := Softmax(TopK(x · W_g)).

The $-\infty$ entries exponentiate to 0 in the softmax, so the non-top-$K$ gates are exactly zero (I skip those experts) and the surviving $K$ gates are a normalized convex combination summing to 1. That's exactly the structure I wanted: sparse where it needs to be, and a clean weighting over the chosen experts. The router weights $W_g$ are tiny (just $\text{dim}\times n$), so the gate itself is only a bias-free linear projection followed by top-$K$ selection and normalization.

How many experts should each token get — what's $K$? $K$ is the dial that sets active compute: $K=1$ means each token's hidden state is rebuilt by a *single* expert. That's the cheapest, but it gives the token no mixing — its representation is whatever one expert says, with no blending, and the router has to make an all-or-nothing bet. $K=2$ lets each token combine two experts' outputs, so the representation is a learned convex blend rather than a hard single-expert choice, while still using only a small fraction of $n$. Going higher buys more mixing for linearly more FFN compute. So $K=2$ is the smallest setting that still blends: the two selected logits are softmax-normalized and their two expert outputs are added as a weighted sum. With $n=8$ experts and $K=2$, each token sees 2 of 8 FFNs.

Let me make the expert concrete. The experts don't need to be anything exotic — an expert is just a standard feed-forward block, the same SwiGLU FFN the dense model already uses. So $E_i = \mathrm{SwiGLU}_i$, eight independent copies of the FFN with their own weights, and the rest of the block (attention, norms, residuals) is unchanged. This routing decision happens independently for each token at each layer. Putting it together, the per-token output is

  y = Σ_{i=0}^{n-1} Softmax(Top2(x · W_g))_i · SwiGLU_i(x).

Let me do the parameter arithmetic to confirm the decoupling actually pays off. Take the Mistral-7B-shaped backbone: dim 4096, 32 layers, FFN hidden dim 14336, 32 heads with 8 KV heads (grouped-query), vocab 32000, and now 8 experts per layer with top-2 routing. The attention and embedding parameters are shared as before; what multiplies is the FFN — instead of one SwiGLU per layer there are eight. The *total* (sparse) parameter count lands around 47B. But per token only $K=2$ of the 8 experts fire, so the FFN compute per token corresponds to roughly two experts, not eight — the *active* count is about 13B. So I'm carrying 47B worth of capacity in memory while spending ~13B worth of compute per token. The comparison I care about: LLaMA-2 70B uses 70B active parameters per token; this uses ~13B, about 5× fewer, while the 47B of total capacity is still *smaller* than 70B. If quality tracks capacity-ish and compute tracks active count, I'd expect to roughly match or beat the 70B dense model at a fraction of the per-token cost. (Note the asymmetry I'm buying: memory cost scales with the 47B sparse count, compute with the 13B active count — so this wins on compute, and is most attractive in batched serving where the routing/expert-gather overhead amortizes.)

There's a subtlety in *where* to put the MoE layers and how to route. GShard, which this resembles, replaces the FFN in only *every other* layer with an MoE layer and uses a more elaborate gating strategy for the second expert. I'll go the other way: replace *every* FFN sub-block with an MoE layer (so every layer gets the capacity boost) and keep the gate dead simple — plain top-2 softmax for both chosen experts, no special second-expert logic. Simpler and applied throughout.

Now the wall this hits in practice: efficiency and balance. The math says "skip the zero-gated experts," but a naive implementation that loops over experts, gathering the tokens routed to each, is awkward on a GPU because different experts receive *different numbers of tokens* — ragged batches kill throughput. The fix is to cast the whole per-expert FFN computation as large block-sparse matrix multiplications (the Megablocks approach), which handles variable token counts per expert without padding waste and runs fast on one GPU. And when experts are spread across GPUs — Expert Parallelism, where each token is shipped to the GPU holding its chosen expert and the result shipped back — the danger is load imbalance: if the router sends too many tokens to a few experts, those GPUs bottleneck while others idle. So I need to track how often each expert is selected and whether consecutive tokens keep reusing the same experts, because skew creates over-subscription while locality can be useful for caching.

Now the code. The MoE layer holds a router (a bias-free linear to $n$ logits) and $n$ SwiGLU experts; the forward computes top-$K$ gates, softmaxes over the kept logits, and accumulates the weighted expert outputs only for the tokens routed to each expert.

```python
import dataclasses
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclasses.dataclass
class MoeArgs:
    num_experts: int          # n  (e.g. 8)
    num_experts_per_tok: int  # K  (e.g. 2)

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)   # n independent SwiGLU FFNs
        self.gate = gate                        # nn.Linear(dim, n, bias=False) = W_g
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):    # inputs: (num_tokens, dim)
        gate_logits = self.gate(inputs)         # (num_tokens, n)  =  inputs @ W_g
        # top-K experts per token, then softmax over ONLY the kept logits:
        # equivalent to Softmax(TopK(.)) since the dropped logits would be -inf -> 0.
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += (
                weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
            )
        return results                          # only K experts run per token
```

The expert is the existing SwiGLU FFN, and the MoE layer drops into the transformer block in place of the dense feed-forward:

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def build_moe_ffn(args):                        # replaces every FFN sub-block
    experts = [SwiGLU(args.dim, args.hidden_dim) for _ in range(args.num_experts)]
    gate = nn.Linear(args.dim, args.num_experts, bias=False)   # the router W_g
    return MoeLayer(experts, gate, MoeArgs(args.num_experts, args.top_k_experts))
# config: dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
#         hidden_dim=14336, vocab=32000, context_len=32768,
#         num_experts=8, top_k_experts=2   ->  ~47B total / ~13B active
```

So the chain: in a dense Transformer every parameter runs for every token, welding quality (∝ parameters) to per-token cost. Splitting parameters into total vs active and only paying for active breaks that weld; the per-token-independent FFN is the place to do it; replace it with $n$ expert FFNs and a sparse gate so each token uses only $K$ of them; a top-$K$-then-softmax linear router makes the gate sparse and normalized; $K=2$ buys real blending of experts at minimal compute; experts are ordinary SwiGLU FFNs; with $n=8,K=2$ on a Mistral-shaped backbone the total reaches ~47B while active stays ~13B — about 5× fewer active parameters than a 70B dense model. Block-sparse kernels and expert parallelism make it fast, with load balancing the thing to watch.
