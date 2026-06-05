# Context

The setting is building a strong open-weights large language model around late 2023, when the dominant recipe is a dense decoder-only Transformer (the LLaMA-2 family up to 70B, Mistral 7B) scaled up in parameters. The hard fact of dense models is that *every* parameter is used for *every* token: increasing capacity to improve quality multiplies the FLOPs and the memory bandwidth per token proportionally, so quality and inference cost are chained together. A 70B dense model is strong but expensive to serve, especially the per-token compute.

## Research question

Can a model's *capacity* (total parameter count, which correlates with quality) be increased without paying the matching increase in *per-token compute and latency*? In a dense Transformer the feed-forward (FFN) sub-block of every layer processes every token with the same full weight matrix. If quality scales with parameters but we don't want compute to scale with parameters, we need an architecture where each token only touches a *fraction* of the parameters. The goal: decouple the **total (sparse) parameter count** from the **active parameter count used per token**, so capacity can grow while the compute per token stays roughly fixed — and do so in a way that trains stably, serves efficiently on GPUs, and matches or beats a much larger dense model.

## Background

- **The Transformer decoder block** (Vaswani et al. 2017): each layer is multi-head self-attention followed by a position-wise feed-forward network applied independently to every token. The FFN holds a large share of the parameters and compute.
- **Mistral 7B base recipe** (Jiang et al. 2023): a decoder-only Transformer with grouped-query attention (separate `n_kv_heads` < `n_heads`), rotary position handling, and a **SwiGLU** FFN (a gated GLU variant, $\mathrm{SwiGLU}(x) = (\mathrm{SiLU}(xW_1)\odot xW_3)W_2$). This is the dense baseline being extended.
- **Conditional computation / Mixture of Experts.** The idea of routing each input to a subset of specialized sub-networks goes back to mixtures of experts (Jacobs et al. 1991). The modern large-scale form is the **sparsely-gated MoE** (Shazeer et al. 2017): replace a layer with $n$ expert networks and a gating network $G(x)$ that, per token, picks a few experts; the layer output is $\sum_i G(x)_i\,E_i(x)$. If $G$ is sparse (most coordinates 0), the zero-gated experts need not be computed, so adding experts adds parameters without adding per-token compute. This introduces a **load-balancing** problem: tokens must be spread evenly across experts or some experts/GPUs are overloaded.
- **GShard** (Lepikhin et al. 2020): scales MoE Transformers by replacing the FFN in *every other* layer with an MoE layer and using a top-2 gate with an auxiliary balancing loss and capacity factors; demonstrates expert and model parallelism for huge sparse models.
- **Switch Transformer / MoE reviews** (Fedus et al.): top-1 routing and a thorough account of MoE training/serving trade-offs; establishes the **sparse vs active** parameter distinction.
- **Efficient MoE kernels and parallelism.** Megablocks (Gale et al. 2022) casts the per-expert FFN computation as large block-sparse matrix multiplications, handling the variable number of tokens per expert efficiently on a single GPU. **Expert Parallelism** distributes experts across GPUs, routing each token to the GPU holding its expert; this needs careful load balancing.
- **Top-K gating mechanics.** A simple, performant gate (Shazeer et al. 2017) takes a linear projection of the token to $n$ logits, keeps the top-$K$, sets the rest to $-\infty$, and applies a softmax, so the kept gate weights are a normalized convex combination and the rest are exactly 0.

## Baselines

- **Dense LLaMA-2 (7B/13B/70B)** and **Mistral 7B**: decoder-only Transformers where every token uses all parameters. The 70B model is the quality target; the gap to close is matching it while using far fewer parameters per token. Limitation: per-token compute is tied to total capacity.
- **GShard-style MoE Transformer**: MoE in alternating layers with a more elaborate second-expert gating strategy. Establishes that sparse routing scales capacity, but uses MoE in only every other block and a more complex gate.
- **Top-1 (Switch) routing**: minimal per-token compute (one expert) but each token's representation is built from a single expert, limiting the per-token mixing.

## Evaluation settings

- **Benchmarks (pre-existing):** commonsense reasoning (HellaSwag, WinoGrande, PIQA, SIQA, OpenBookQA, ARC-easy/challenge, CommonsenseQA, 0-shot); world knowledge (NaturalQuestions, TriviaQA, 5-shot); reading comprehension (BoolQ, QuAC, 0-shot); math (GSM8K 8-shot maj@8, MATH 4-shot maj@4); code (HumanEval 0-shot, MBPP 3-shot); aggregates (MMLU 5-shot, BBH 3-shot, AGIEval). Multilingual ARC-c/HellaSwag/MMLU in French, German, Spanish, Italian. Long-context passkey retrieval and perplexity on proof-pile. Bias: BBQ, BOLD. Instruct evaluation: MT-Bench, LMSys human arena.
- **Metric of efficiency:** report **active parameters** per token (proportional to inference compute) separately from **sparse** (total) parameters (proportional to memory). The natural comparison is quality at a given active-parameter budget against dense models of various total sizes.
- **Architecture knobs (a priori):** model dim, number of layers, heads, KV heads, FFN hidden dim, vocab, context length, plus — for the new layer — number of experts $n$ and experts-per-token $K$.

## Code framework

Pre-method scaffold of a decoder-only Transformer block with the per-token feed-forward slot left open for the conditional-computation contribution.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLU(nn.Module):                       # the dense FFN that already exists
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):                    # grouped-query attention, already exists
    ...

class FeedForwardBlock(nn.Module):
    """The per-token feed-forward sub-block of a transformer layer."""
    def __init__(self, args):
        super().__init__()
        # TODO: how each token's hidden state is transformed here is the slot
        #       the contribution fills. A plain dense block uses all parameters
        #       for every token; we want capacity to grow without per-token
        #       compute growing.  <-- contribution goes here
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = Attention(args)
        self.ffn  = FeedForwardBlock(args)     # TODO body above
        self.attn_norm = nn.RMSNorm(args.dim)
        self.ffn_norm  = nn.RMSNorm(args.dim)
    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        return h + self.ffn(self.ffn_norm(h))
```
