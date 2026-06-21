## Research question

The setting is building a strong open-weights large language model around late 2023, when the dominant recipe is a dense decoder-only Transformer (the LLaMA-2 family up to 70B, Mistral 7B) scaled up in parameters. In a dense Transformer the feed-forward (FFN) sub-block of every layer processes every token with the same full weight matrix, so total parameters and per-token compute scale together.

How can the feed-forward slot in a decoder-only Transformer be designed to provide high model quality while keeping per-token compute and latency manageable, and doing so in a way that trains stably and serves efficiently on GPUs?

## Background

- **The Transformer decoder block** (Vaswani et al. 2017): each layer is multi-head self-attention followed by a position-wise feed-forward network applied independently to every token. The FFN holds a large share of the parameters and compute.
- **Mistral 7B base recipe** (Jiang et al. 2023): a decoder-only Transformer with 32 attention heads, 8 KV heads, and a **SwiGLU** FFN (a gated GLU variant, $\mathrm{SwiGLU}(x) = (\mathrm{SiLU}(xW_1)\odot xW_3)W_2$). This is the dense baseline being extended.
- **Conditional computation / Mixture of Experts.** In the sparsely gated form (Shazeer et al. 2017), a layer contains $n$ expert networks and a gating network $G(x)$ that, per token, picks a few experts; the layer output is $\sum_i G(x)_i\,E_i(x)$. If $G$ is sparse (most coordinates 0), the zero-gated experts need not be computed, so adding experts adds parameters without adding per-token compute. Token routing across experts raises load-balancing considerations: tokens must be spread across experts or some experts/GPUs see uneven utilization.
- **GShard** (Lepikhin et al. 2020): scales MoE Transformers by replacing the FFN in *every other* layer with an MoE layer and using a more elaborate strategy for the second selected expert; demonstrates expert and model parallelism for huge sparse models.
- **MoE reviews** (Fedus et al.): survey sparse expert models and their training/serving trade-offs.
- **Efficient MoE kernels and parallelism.** Megablocks (Gale et al. 2022) casts the per-expert FFN computation as large block-sparse matrix multiplications, handling the variable number of tokens per expert efficiently on a single GPU. **Expert Parallelism** distributes experts across GPUs, routing each token to the GPU holding its expert.
- **Gating networks.** The sparsely gated form (Shazeer et al. 2017) learns the gate $G(x)$ as a small projection of the token to $n$ logits; its design — how logits become a per-token selection of experts and how the chosen experts' outputs are weighted — determines whether and how cheaply the layer is actually sparse.

## Baselines

- **Dense LLaMA-2 (7B/13B/70B)** and **Mistral 7B**: decoder-only Transformers where every token uses all parameters. The 70B model is the quality target; the question is how to approach that quality band with a more inference-efficient architecture.
- **GShard-style MoE Transformer**: MoE in alternating layers with a more elaborate second-expert gating strategy. Establishes that sparse routing scales capacity.
- **Single-expert routing**: the minimal sparse-routing setting, sending each token to one sub-network; the cheapest per-token compute, with the router making an all-or-nothing assignment.

## Evaluation settings

- **Benchmarks (pre-existing):** commonsense reasoning (HellaSwag, WinoGrande, PIQA, SIQA, OpenBookQA, ARC-easy/challenge, CommonsenseQA, 0-shot); world knowledge (NaturalQuestions, TriviaQA, 5-shot); reading comprehension (BoolQ, QuAC, 0-shot); math (GSM8K 8-shot maj@8, MATH 4-shot maj@4); code (HumanEval 0-shot, MBPP 3-shot); aggregates (MMLU 5-shot, BBH 3-shot, AGIEval). Multilingual ARC-c/HellaSwag/MMLU in French, German, Spanish, Italian. Long-context passkey retrieval and perplexity on proof-pile. Bias: BBQ, BOLD. Instruct evaluation: MT-Bench, LMSys human arena.
- **Metric of efficiency:** report inference compute per token (which sets latency and serving cost) separately from total parameter count (which sets the memory footprint). The natural comparison is quality at a given per-token compute budget against dense models of various sizes.
- **Backbone to match:** the target sits on a Mistral-7B-shaped backbone (dim 4096, 32 layers, 32 heads, 8 KV heads, FFN hidden dim 14336, vocab 32000, context 32768), aiming to reach the quality band of the 70B dense model at a far smaller per-token compute budget.
- **Architecture knobs:** model dim, number of layers, heads, KV heads, FFN hidden dim, vocab, context length, and whatever additional knobs the chosen feed-forward slot design exposes.

## Code framework

A decoder-only Transformer block has a feed-forward slot whose body can change while attention, normalization, and residual structure stay fixed.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLU(nn.Module):                       # dense FFN used by the decoder
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):                    # grouped-query attention
    ...

class FeedForwardBlock(nn.Module):
    """The per-token feed-forward sub-block of a transformer layer."""
    def __init__(self, args):
        super().__init__()
        # TODO: choose how this slot transforms each token's hidden state.
        #       The dense baseline sends every token through one shared FFN.
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
