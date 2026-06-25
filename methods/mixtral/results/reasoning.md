Let me start from the thing that actually constrains me. I have a dense decoder-only Transformer — the Mistral 7B recipe, grouped-query attention and a SwiGLU feed-forward block per layer — and it's good. The obvious way to make it better is to make it bigger, and the field has shown quality keeps climbing with parameters: LLaMA-2 goes 7B, 13B, 70B and gets steadily stronger. But in a dense model *every* parameter touches *every* token. So when I add parameters to get more quality, I raise the FLOPs per token and push per-token cost and latency up with it. Quality and per-token cost are welded together. A 70B dense model is the quality I want and the serving cost I don't. So the real question isn't "how do I add capacity" — it's "how do I add capacity *without* adding per-token compute."

What would it even mean to add parameters that a token doesn't pay for? It means each token must only *touch* a fraction of the total parameters. Some parameters are used for this token, others for that token, and the union is large but the per-token slice is small. That splits the parameter count into two numbers that I've been conflating: the **total** parameter count — everything in the model, which sets the memory footprint — and the **active** parameter count — what a single token actually runs through, which sets the compute. In a dense model these are equal. The thing I want is to pry them apart: grow total, hold active. The whole rest of this is figuring out a mechanism that does that without wrecking quality or trainability.

Where in the block should the extra capacity live? Attention mixes across tokens and is comparatively lean; the feed-forward sub-block is where most of the per-layer parameters and FLOPs sit, and crucially it's applied *independently per token* — position $t$'s FFN doesn't look at position $t'$. That independence matters: if I want to make a token-by-token decision about which parameters fire, I can only safely do it in a place where one token's computation doesn't depend on another's, and the FFN is exactly that place. Attention is not — its whole job is cross-token mixing. So the FFN is where token-dependent parameter selection can go.

So: instead of one FFN per layer, put $n$ separate FFNs — call them experts $E_0,\dots,E_{n-1}$ — and have each token use only some of them. The classic way to make a layer use a subset of sub-networks per input is a gating network. Let the layer output, for token $x$, be a weighted combination of expert outputs,

  y = Σ_{i=0}^{n-1} G(x)_i · E_i(x),

where $G(x)$ is an $n$-dimensional vector of gate weights. Now look at what happens when $G(x)$ is *sparse* — most of its coordinates exactly zero. For every $i$ with $G(x)_i = 0$ the term $G(x)_i E_i(x)$ is zero, and I never have to compute $E_i(x)$ at all. So the FLOPs are paid only for the experts with nonzero gate. If I keep, say, only $K$ of the $n$ gates nonzero, then each token runs through $K$ experts regardless of how large $n$ is. That's the decoupling I was after, written as a formula: crank $n$ up to add total parameters (capacity, memory), hold $K$ fixed to hold active parameters (compute). Increasing $n$ at fixed $K$ raises capacity at essentially constant per-token cost — but only if I can actually build a $G$ that is genuinely sparse, normalized, and learnable. That "if" is the next thing to nail down.

So $G(x)$ has to be sparse and learnable. The simplest thing that learns *which* experts a token wants is a linear router: project the token to $n$ logits, $x\cdot W_g$, one per expert. To turn logits into a sparse, normalized gate, the natural move is take the top-$K$ logits, discard the rest, and softmax over what remains. Formally, define $\mathrm{TopK}(\ell)_i = \ell_i$ if $\ell_i$ is among the top-$K$ coordinates of $\ell$ and $-\infty$ otherwise; then

  G(x) := Softmax(TopK(x · W_g)).

The claim I'm leaning on is that the $-\infty$ entries exponentiate to 0, so the non-top-$K$ gates are *exactly* zero (I skip those experts) and the surviving $K$ gates form a normalized convex combination summing to 1. I want to be sure of two things before I build on this: that masking-to-$-\infty$-then-softmaxing really does give the same numbers as just softmaxing over the $K$ kept logits (the cheaper thing I'd actually code), and that the dropped entries land at a hard zero rather than a tiny nonzero that would force me to run all $n$ experts anyway. Let me check it on a concrete vector. Take $n=8$ logits $\ell = [1.541, -0.293, -2.179, 0.568, -1.085, -1.399, 0.403, 0.838]$ and $K=2$. Top-2 are indices 0 and 7 (values $1.541, 0.838$). Masking everything else to $-\infty$ and taking the full 8-way softmax gives gate $[0.6688, 0,0,0,0,0,0, 0.3312]$. Softmaxing over just the two kept values $[1.541, 0.838]$ gives $[0.6688, 0.3312]$ — and placing those back at indices 0 and 7 gives exactly the same vector. The max absolute difference between the two is $0$, the six non-kept entries are exactly zero (not small — *zero*), and the two survivors sum to $1.0$. Good: the gate is genuinely sparse, the cheap softmax-over-kept-logits is identical to the masked version, and I can implement it without ever touching the $n-K$ skipped experts.

How many experts should each token get — what's $K$? $K$ is the dial that sets active compute. $K=1$ means each token's hidden state is rebuilt by a *single* expert. That's the cheapest, but it gives the token no mixing — its representation is whatever one expert says, with no blending, and the router has to make an all-or-nothing bet. With $K=1$ the softmax over a single kept logit is just $1.0$, so the gate carries no relative weighting either; the expert's output passes through ungated. $K=2$ lets each token combine two experts' outputs, so the representation is a learned convex blend (the two softmaxed weights) rather than a hard single-expert choice, while still using only a small fraction of $n$. Going higher buys more mixing for linearly more FFN compute. So $K=2$ is the smallest setting that still blends: the two selected logits are softmax-normalized and their two expert outputs are added as a weighted sum. With $n=8$ experts and $K=2$, each token sees 2 of 8 FFNs.

Let me make the expert concrete. The experts don't need to be anything exotic — an expert is just a standard feed-forward block, the same SwiGLU FFN the dense model already uses. So $E_i = \mathrm{SwiGLU}_i$, eight independent copies of the FFN with their own weights, and the rest of the block (attention, norms, residuals) is unchanged. This routing decision happens independently for each token at each layer. Putting it together, the per-token output is

  y = Σ_{i=0}^{n-1} Softmax(Top2(x · W_g))_i · SwiGLU_i(x).

Now I should check whether this decoupling actually buys what I claimed, with real numbers on the Mistral-7B-shaped backbone: dim 4096, 32 layers, FFN hidden dim 14336, 32 heads with 8 KV heads (grouped-query), head_dim 128, vocab 32000, and 8 experts per layer with top-2 routing. Attention and embeddings are shared as before; what multiplies is the FFN. A single SwiGLU FFN has three dim×hidden projections, $3 \cdot 4096 \cdot 14336 \approx 176\text{M}$ params. The attention block (q proj $4096 \cdot 32 \cdot 128$, k/v projs each $4096 \cdot 8 \cdot 128$, out proj $32 \cdot 128 \cdot 4096$) is $\approx 42\text{M}$ per layer, plus a tiny router $4096 \cdot 8$. Embeddings are $32000 \cdot 4096$ for input and again for the output head, $\approx 0.26\text{B}$ together.

For the **total (sparse)** count, every layer carries all 8 experts:
  total ≈ embeds + 32·(attn + 8·SwiGLU + gate) ≈ 0.26B + 32·(42M + 8·176M + 33K) ≈ **46.7B**.
For the **active** count, only $K=2$ experts fire per token:
  active ≈ embeds + 32·(attn + 2·SwiGLU + gate) ≈ 0.26B + 32·(42M + 2·176M + 33K) ≈ **12.9B**.

So I carry ~47B of capacity in memory while spending ~13B of compute per token — and the experts dominate: the FFN/expert weights are $32 \cdot 8 \cdot 176\text{M} \approx 45\text{B}$, about 97% of the total, which is why moving $n$ moves total almost one-for-one. The comparison I care about: LLaMA-2 70B spends 70B active parameters per token; this spends ~12.9B, a ratio of $70/12.9 \approx 5.4\times$ fewer, and the 47B of total capacity is still *smaller* than 70B. So I'm not even matching 70B's memory — I'm under it — while spending roughly a fifth of its per-token compute. Whether that actually reaches 70B-band quality I can't settle from arithmetic; capacity-vs-active is a compute argument, not a quality proof, and I'd want the benchmark numbers to confirm it. But the efficiency side is real and now checked. (Note the asymmetry I'm buying: memory cost scales with the 47B sparse count, compute with the 12.9B active count — so this wins on compute, and is most attractive in batched serving where the routing/expert-gather overhead amortizes.)

There's a question of *where* to put the MoE layers and how to route. GShard, which this resembles, replaces the FFN in only *every other* layer with an MoE layer and uses a more elaborate gating strategy for the second expert. The other direction is to replace *every* FFN sub-block with an MoE layer (so every layer gets the capacity boost) and keep the gate dead simple — plain top-2 softmax for both chosen experts, no special second-expert logic. The arithmetic above already assumed MoE in every layer; that's where the 45B of expert weights came from. Going every-other-layer would roughly halve the expert capacity and waste the per-token-independence of half the FFNs for no compute saving (active is set by $K$, not by how many layers are MoE). So: MoE everywhere, simplest possible gate.

Before trusting any of this I should make sure the implementation I have in mind actually computes the math above and not something subtly different. The efficient way to run it on a GPU is to loop over experts, gather the tokens routed to each, run that expert once on its batch, and scatter-add the gated outputs back — rather than looping over tokens. That gather/scatter is fiddly enough that I want to confirm it equals the plain definition $y = \sum_i G(x)_i E_i(x)$. Let me trace it on a small case: $n=8$, $K=2$, dim 4, three tokens, experts as distinct linear maps so I can tell them apart. The router sends the three tokens to expert pairs $[1,4]$, $[2,6]$, $[7,4]$ — already a useful sign that routing is genuinely token-dependent (every token picks a different pair, and expert 4 is shared between two of them). The scatter-add forward produces, for the three tokens, outputs matching the direct per-token weighted sum $\sum_{k} g_{t,k} E_{\mathrm{idx}(t,k)}(x_t)$ to a max absolute difference of $3\times10^{-8}$ — floating-point noise. So the gather-by-expert loop and the textbook weighted-sum agree; the implementation computes the intended layer.

The remaining concern is efficiency and balance at scale. The math says "skip the zero-gated experts," but the implementation that gathers the tokens routed to each expert is awkward on a GPU because different experts receive *different numbers of tokens* — in my trace expert 4 got two tokens and most experts got zero, and at scale these ragged batches kill throughput. The fix is to cast the whole per-expert FFN computation as large block-sparse matrix multiplications (the Megablocks approach), which handles variable token counts per expert without padding waste and runs fast on one GPU. And when experts are spread across GPUs — Expert Parallelism, where each token is shipped to the GPU holding its chosen expert and the result shipped back — the danger is load imbalance: if the router sends too many tokens to a few experts, those GPUs bottleneck while others idle. So I'd want to track how often each expert is selected and whether consecutive tokens keep reusing the same experts, because skew creates over-subscription while locality can be useful for caching.

Now the code. The MoE layer holds a router (a bias-free linear to $n$ logits) and $n$ SwiGLU experts; the forward computes top-$K$ gates, softmaxes over the kept logits, and accumulates the weighted expert outputs only for the tokens routed to each expert — exactly the scatter-add loop I just traced.

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

So the chain: in a dense Transformer every parameter runs for every token, welding quality (∝ parameters) to per-token cost. Splitting parameters into total vs active and only paying for active breaks that weld; the per-token-independent FFN is the only place where token-by-token parameter selection is safe; replace it with $n$ expert FFNs and a sparse gate so each token uses only $K$ of them; a top-$K$-then-softmax linear router makes the gate sparse and normalized (checked: dropped gates land at exact zero, kept gates sum to 1, and softmax-over-kept matches the masked form to the bit); $K=2$ buys real blending of experts at minimal compute; experts are ordinary SwiGLU FFNs; with $n=8,K=2$ on a Mistral-shaped backbone the arithmetic gives ~46.7B total while active stays ~12.9B — about 5.4× fewer active parameters than a 70B dense model, at less total memory. The scatter-add implementation matches the textbook weighted sum to floating-point precision. Block-sparse kernels and expert parallelism make it fast, with load balancing the thing to watch.
