# Context

## Research question

Large autoregressive Transformer language models have, by the early 2020s, become the dominant tool in NLP, and the headline lesson from the field has been brutally simple: make the model bigger and it gets better, and at sufficient size it starts solving tasks from a handful of examples or a textual instruction. The natural response has been a race in parameter count — 175B, 280B, 540B — under the working assumption that more parameters is the road to better performance.

But there is a cost question hiding underneath that race, and it has two halves that are usually conflated. There is the cost of *training* a model once, and there is the cost of *running* it every time it answers a query in deployment. A model that is trained once but served billions of times pays its inference cost over and over. The precise problem is this: given a target level of quality, which model should one actually build and ship? Not "what is the biggest model I can afford to train," and — as will matter here — not even "what model is cheapest to train to that quality," but "what model is cheapest to *serve* at that quality." A solution has to (1) get a clear-eyed account of the training-vs-inference cost trade-off, (2) decide how many tokens to train on once the objective is inference cost rather than training cost, and (3) assemble a Transformer that trains stably and efficiently for the very long token horizons this implies — using, as an additional self-imposed constraint, only publicly available data so the result can be openly released.

## Background

**Few-shot ability emerges with scale.** Large models trained on massive corpora exhibit in-context learning — performing a new task from a few examples or an instruction with no gradient updates. This behaviour appeared once models were scaled past a certain size, which launched the line of work that kept scaling parameter count.

**Power laws for language models.** Empirical studies established that test loss falls as a power law in model size, dataset size, and training compute. Hestness et al. (2017) and Rosenfeld et al. (2019) observed such power laws across deep learning; Kaplan et al. (2020) derived them specifically for Transformer language models, finding that loss is predictable from compute and that, under their analysis, the compute-optimal allocation puts most of an increased compute budget into a *bigger model* and comparatively little into more data.

**The compute-optimal correction.** Hoffmann et al. (2022) revisited this and found the opposite balance. With the learning-rate schedule properly matched to the number of training tokens, fitting the loss surface over model size $N$ and tokens $D$ at fixed compute $C \approx 6ND$ gives a minimum where model size and data should be scaled *in roughly equal proportion*, $N_{\text{opt}} \propto C^{a}$ and $D_{\text{opt}} \propto C^{b}$ with $a \approx b \approx 0.5$. The practical consequence: many existing large models were substantially *over-parameterized and under-trained* for their compute. Their recommended allocations are smaller-than-fashionable models on many more tokens (e.g. on the order of a ~10B model trained on ~200B tokens for a given budget).

**The gap this leaves.** Every one of those analyses optimizes *training* compute. None of them speaks to inference. But there is a relevant empirical fact about training dynamics, knowable by simply training a small model for a long time: a 7B-parameter model's loss keeps decreasing well past 1T training tokens — far beyond the token count any training-compute-optimal rule would assign to a model that size. The loss has not flattened. So "train a small model far past its training-compute optimum" is on the table as a way to reach a target quality in a small package, and nobody chasing the training-compute frontier has a reason to do it.

**Architectural ingredients available off the shelf.** By this point a number of refinements to the original Transformer have been published and validated in large models:

- *Pre-normalization.* The original Transformer places LayerNorm after each residual addition (post-norm); pre-norm instead normalizes the *input* to each sub-layer, leaving the residual path unnormalized. This makes deep Transformers far easier to optimize and has become standard in large decoder models.
- *RMSNorm* (Zhang & Sennrich, 2019). LayerNorm both re-centers (subtract the mean) and re-scales (divide by standard deviation). Zhang & Sennrich argue the re-centering is dispensable — the benefit is dominated by re-scaling, which gives scale-invariance and an implicit learning-rate adaptation — and propose normalizing by the root-mean-square alone. It is cheaper (no mean, no bias) and matches LayerNorm quality, cutting normalization runtime by a reported 7–64%.
- *Gated linear unit FFNs* (Dauphin et al., 2017; Shazeer, 2020). The standard FFN is $W_2\,\sigma(W_1 x)$ with $\sigma=\text{ReLU}$ and hidden width $4d$. A GLU replaces this with a component-wise product of two projections, one passed through a nonlinearity: $W_2\big(\phi(W_1 x)\odot(W_3 x)\big)$. Shazeer's sweep found that using Swish/SiLU as $\phi$ (the "SwiGLU" variant) gives the best perplexity at constant compute. Because a GLU has three weight matrices rather than two, holding parameters and FLOPs fixed requires shrinking the hidden width to $\tfrac23$ of its original value.
- *Rotary position embeddings* (Su et al., 2021). Absolute position embeddings add a position-dependent vector to the input once, at the bottom of the stack. RoPE instead seeks position maps $f_q(x,m), f_k(x,n)$ whose inner product $\langle f_q(x_m,m), f_k(x_n,n)\rangle = g(x_m,x_n,\,m-n)$ depends only on the *relative* offset $m-n$. The solution rotates each 2D sub-vector of the query and key by an angle proportional to position, with per-pair frequencies $\theta_i = 10000^{-2(i-1)/d}$, applied at every layer. It injects relative position directly into the attention dot product and decays attention smoothly with distance.

**Efficient long-horizon training primitives.** Memory-efficient attention that avoids materializing the full $n\times n$ score matrix (Rabe & Saunders, 2021), with a fused backward (Dao et al., 2022), is available in libraries such as xformers. Activation checkpointing trades recomputation for memory. Tensor/sequence model parallelism (Shoeybi et al., 2019; Korthikanti et al., 2022) splits a model across many accelerators, with collective `all_reduce` communication that can be overlapped with computation.

## Baselines

**GPT-3 (Brown et al., 2020).** 175B-parameter dense decoder Transformer, post-attention the standard architecture with pre-norm, trained on a large mostly-private mixture. Demonstrated strong few-shot in-context learning. The limitation: enormous parameter count makes it expensive to serve, the training data is not public, and — under the later compute-optimal view — it was under-trained for its size.

**Gopher (Rae et al., 2021) and PaLM (Chowdhery et al., 2022).** 280B and 540B dense decoders pushing the parameter-count frontier further. PaLM contributed and validated SwiGLU FFNs and other refinements at scale. Same limitations amplified: gigantic to serve, proprietary data.

**Chinchilla (Hoffmann et al., 2022).** Not a bigger model but a re-allocation: a 70B model trained on ~1.4T tokens, derived from the compute-optimal analysis above, which matches or beats the 280B/540B models. Its contribution is the recipe ($N \propto C^{0.5}$, $D \propto C^{0.5}$). Its gap, for the present purpose: it optimizes the *training* budget. It explicitly stops at the training-compute-optimal token count and says nothing about, and does not exploit, the regime where you keep training a *smaller* model past that optimum to lower its inference cost. Its data is also not public.

**Existing open models — OPT, GPT-NeoX, BLOOM, GLM.** Openly released, but none competitive with PaLM-62B or Chinchilla in quality. So "open" and "state of the art" had not yet coexisted.

## Evaluation settings

The natural yardsticks are the standard zero-/few-shot NLP suites used to compare large LMs: commonsense reasoning (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC easy/challenge, OpenBookQA), closed-book question answering (Natural Questions, TriviaQA), reading comprehension (RACE), massive multitask understanding (MMLU), mathematical reasoning (GSM8k, MATH), and code generation (HumanEval, MBPP), alongside held-out perplexity and tracking the training-loss curve against tokens consumed. Inference cost — model size and decode latency at a target quality — is the operating metric this work cares about, with publicly available datasets only as the training-data constraint.

## Code framework

The primitives that already exist: PyTorch modules, an AdamW optimizer, a cosine learning-rate schedule, BPE tokenization (SentencePiece), and tensor-parallel linear/embedding layers from a model-parallel library. A decoder language model is assembled from a normalization module, a self-attention module, a position-wise feed-forward module, and a way of injecting token position, stacked into residual blocks. The slots the design will fill:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    multiple_of: int = 256       # round FFN hidden width to a multiple of this
    norm_eps: float = 1e-5
    max_seq_len: int = 2048


class Norm(nn.Module):
    # TODO: a normalization layer for each sub-layer's input.
    def __init__(self, dim, eps):
        super().__init__()
        pass
    def forward(self, x):
        pass


def precompute_position(dim, end):
    # TODO: precompute whatever per-position quantities the attention needs.
    pass


def apply_position(xq, xk, pos):
    # TODO: inject position into queries/keys.
    pass


class Attention(nn.Module):
    # TODO: multi-head self-attention with a KV cache for incremental decode.
    def __init__(self, args):
        super().__init__()
        pass
    def forward(self, x, start_pos, pos, mask):
        pass


class FeedForward(nn.Module):
    # TODO: the position-wise feed-forward sub-layer.
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        pass
    def forward(self, x):
        pass


class TransformerBlock(nn.Module):
    # TODO: residual block wiring norm -> attention and norm -> feed-forward.
    def __init__(self, layer_id, args):
        super().__init__()
        pass
    def forward(self, x, start_pos, pos, mask):
        pass


class Transformer(nn.Module):
    # TODO: token embedding -> stack of blocks -> final norm -> output projection.
    def __init__(self, args):
        super().__init__()
        pass
    def forward(self, tokens, start_pos):
        pass
```
