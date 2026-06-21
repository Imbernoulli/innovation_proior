# Context

## Research question

Large language models improve as parameter count, data, and training compute grow, but a dense Transformer ties two of those knobs together: every token touches every feed-forward weight. Adding parameters by widening layers or adding layers therefore adds proportional FLOPs, memory pressure, and communication once the model no longer fits on one accelerator.

The open problem is to make parameter count a more independent scaling axis for Transformer language models.

## Background

Conditional computation is the broad idea of selecting part of a network per example. Its promise is capacity without proportional compute. Accelerators are efficient when work is batched into dense matrix multiplications; a practical sparse layer needs a large enough selected block to amortize routing overhead, and it must batch routed tokens in a shape the compiler can know ahead of time.

Mixture-of-experts layers are the main existing tool. A router computes logits \(h(x)=W_r x\), converts them to probabilities
\[
  p_i(x)=\frac{\exp h_i(x)}{\sum_j \exp h_j(x)},
\]
and selects a small set \(\mathcal T\) of experts from \(\{E_i\}_{i=1}^N\). The layer output is
\[
  y=\sum_{i\in\mathcal T} p_i(x)E_i(x).
\]
With \(|\mathcal T|=k\ll N\), expert parameters grow with \(N\) but per-token expert FLOPs grow mainly with \(k\).

Two engineering constraints shape MoE design. First, the router may send most tokens to a few experts, so previous systems add differentiable auxiliary losses to encourage balanced assignment; the actual token counts come from argmax decisions and are not directly differentiable. Second, the number of tokens routed to an expert is data-dependent, while TPU-style compilation requires fixed tensor sizes. Existing sparse layers handle this by assigning every expert a fixed capacity and dropping overflowed tokens from that expert computation; the surrounding residual path then carries those token representations forward.

## Baselines

The dense Transformer feed-forward sublayer is the local replacement target. For every token \(x\in\mathbb R^{d_{model}}\), it applies
\[
  h=xW_{in},\qquad y=\phi(h)W_{out},
\]
usually with \(d_{ff}\) several times larger than \(d_{model}\). It is simple, well optimized, and naturally batched.

The sparsely gated MoE layer replaces the feed-forward sublayer with top-\(k\) expert routing. In the original neural MoE design, \(k>1\) was argued to be necessary for useful router gradients because learning to route seemed to require comparing multiple experts. Subsequent Transformer MoE systems kept top-2 routing, used fixed expert capacity, and trained successfully at scale.

## Evaluation settings

The clean comparison is FLOP-matched pretraining against dense T5-style Transformer baselines on C4 with the span-corruption objective: corrupt 15% of tokens, replace corrupted spans with sentinels, and report negative log perplexity in nats. FLOP matching isolates whether extra sparse parameters help when the token-level arithmetic budget is held roughly fixed.

The second comparison is wall-clock time, because routing adds all-to-all communication and small router computation not present in the dense model. Capacity factor sweeps test the static-buffer tradeoff: larger capacity drops fewer tokens but wastes more padded slots; smaller capacity saves memory and communication but risks overflow.

For transfer, the expected protocol is pretrain then fine-tune on smaller NLP tasks such as GLUE, SuperGLUE, summarization, extractive and closed-book question answering, and reasoning or commonsense benchmarks. Fine-tuning is important because very high parameter count at fixed FLOPs may overfit small tasks differently from a dense model.

## Code framework

The existing implementation substrate has a Transformer block, a position-wise dense feed-forward layer, a router placeholder, and distributed tensor primitives that can gather tokens into fixed expert buffers and scatter outputs back. The contribution has to fill the router, the capacity rule, overflow behavior, and a differentiable balancing loss.

```python
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Dense per-token Transformer FFN."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.wo(self.act(self.wi(x)))


class Router(nn.Module):
    """Choose expert(s), gate value(s), and fixed buffer slots for each token."""
    def __init__(self, d_model, num_experts):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def balancing_loss(*args, **kwargs):
    """Differentiable auxiliary loss discouraging expert collapse."""
    raise NotImplementedError


class SparseFFN(nn.Module):
    """Drop-in FFN replacement using a bank of independent expert FFNs."""
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.experts = nn.ModuleList(FeedForward(d_model, d_ff)
                                     for _ in range(num_experts))
        self.router = Router(d_model, num_experts)

    def forward(self, x):
        raise NotImplementedError
```
