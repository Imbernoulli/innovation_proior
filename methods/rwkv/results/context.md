# Context: the ground a linear-time sequence model stands on

## Research question

The problem is autoregressive sequence modeling — given a sequence of tokens `(x_1, …, x_T)`, model `p(x) = ∏_t p(x_t | x_{<t})` — at a scale of billions of parameters. The central design tension is between two families of architectures: recurrent nets, which carry a fixed-size state and process tokens sequentially, and Transformers, which compare all token pairs in parallel. How can a single architecture support efficient large-scale training while also providing cheap autoregressive generation?

## Background

The field is split between two architecture families.

**Recurrent nets.** Gated cells — LSTM (Hochreiter & Schmidhuber 1997) and GRU (Cho et al. 2014) — are the long-standing recurrent state of the art. An LSTM computes, at each step, gates and a cell update of the form `f_t = σ(W_f x_t + U_f h_{t-1} + b_f)` (and likewise for input/output gates), `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`, `h_t = o_t ⊙ σ(c_t)`. The computation factors into input-only linear blocks (`W`) and recurrent blocks (`U`), where the dependence of step `t` on step `t-1` through `U h_{t-1}` ties together successive time steps. Recurrent nets carry a fixed-size state and cost `O(Td)` time with `O(d)` inference memory.

**Transformers.** Vaswani et al. (2017) replaced recurrence with attention, computing relationships between all positions in parallel. For autoregressive use the per-position output is `Attn(Q,K,V)_t = Σ_{i=1}^{T} e^{q_tᵀk_i} v_i / Σ_{i=1}^{T} e^{q_tᵀk_i}` (multi-head and the `1/√d_k` scaling omitted). This parallelizes across time and captures long-range dependencies; a layer costs `O(T²d)` time and `O(T² + Td)` memory. A large literature produces variants — Reformer's `O(T log T)` LSH attention, Performer's kernel-feature approximation, Linformer's low-rank projection, BigBird's sparsity.

Two threads matter most as the load-bearing concepts here. **Linear Transformers** (Katharopoulos et al. 2020) replace `e^{q_tᵀk_i}` with a kernel feature map `φ(q_t)ᵀφ(k_i)`; because the kernel factorizes, the numerator `Σ_i φ(q_t)ᵀφ(k_i) v_i = φ(q_t)ᵀ Σ_i φ(k_i) v_iᵀ` can be accumulated in a running state, giving `O(Td²)` time and an RNN-like inference recurrence. **Attention Free Transformer** (Zhai et al. 2021) goes further and removes the query–key dot product entirely: it computes `Attn⁺(W,K,V)_t = Σ_{i=1}^{t} e^{w_{t,i}+k_i} ⊙ v_i / Σ_{i=1}^{t} e^{w_{t,i}+k_i}`, where `w_{t,i}` is a *learned scalar pairwise position bias* and the interaction with the value is element-wise rather than through a dot product. AFT shows that attention-like reweighting can be done with a position bias plus a key, no `QKᵀ` matrix; `w_{t,i}` is a full `T×T` matrix indexed by the absolute pair `(t,i)`.

The supporting machinery for training deep stacks is also part of the ground: layer normalization (Ba et al. 2016) for gradient stability, residual connections and identity-mapping initialization (He et al. 2016) for depth, the squared-ReLU activation (So et al. 2021, Primer) as a stronger position-wise nonlinearity, and DeepSpeed (Rasley et al. 2020) for large-model training. The training data is the Pile (Gao et al. 2020).

## Baselines

**LSTM / GRU recurrent language models (Hochreiter & Schmidhuber 1997; Cho et al. 2014).** Fixed-size recurrent state; `O(Td)` time, `O(d)` inference memory; the recurrence is sequential in time and the update at each step depends on the previous hidden state.

**Transformer / decoder-only LM (Vaswani et al. 2017; Radford et al. 2019).** Self-attention over all positions, fully parallel training, strong long-range modeling, excellent scaling. `O(T²d)` time and `O(T² + Td)` memory per layer; autoregressive decoding maintains a KV cache that grows with context length.

**Linear Transformer (Katharopoulos et al. 2020).** Kernel-feature attention `φ(q)ᵀφ(k)` that factorizes into a running state, giving `O(Td²)` time and an RNN-form inference recurrence. Approximates softmax attention through the choice of feature map `φ`.

**Attention Free Transformer (Zhai et al. 2021).** Drops the `QKᵀ` dot product; reweights values element-wise by `e^{w_{t,i}+k_i}` with a learned pairwise position-bias matrix `w_{t,i}`. Shows attention can be done without query–key interaction; `w_{t,i}` is a full `T×T` matrix tied to absolute positions.

## Evaluation settings

The natural yardstick is causal language modeling on a large, diverse corpus — the **Pile** (Gao et al. 2020), ~825GB of text — with the standard cross-entropy / perplexity objective, reported across a sweep of model sizes (from ~169M up to billions of parameters) so that scaling behavior can be read off. Downstream, the accepted protocol is a battery of zero-shot / few-shot NLP benchmarks of the kind used to evaluate GPT-style models (commonsense reasoning and language-understanding tasks). Efficiency is measured directly: inference latency and memory as a function of sequence length, and training throughput. These datasets, metrics, and the size-sweep protocol all predate any particular architecture and are the established measuring stick.

## Code framework

A generic autoregressive language-model harness in PyTorch fixes everything that is not in question — how tokens become vectors, how a stack of residual blocks is assembled, how the final logits and cross-entropy loss are formed — and leaves the body of each block's two sub-layers as empty slots. A block mixes information *across* positions (a token-mixing sub-layer) and then transforms each position *independently* (a position-wise sub-layer), each wrapped as a pre-normalized residual. The two `# TODO` bodies are the slots the work will fill.

```python
import torch, torch.nn as nn

class TokenMixing(nn.Module):
    """Mixes information across time positions. To be designed."""
    def __init__(self, d_model, layer_id, n_layers):
        super().__init__()
        # TODO: the across-time mixing operator
        pass
    def forward(self, x):                 # x: (B, T, d_model)
        # TODO
        raise NotImplementedError

class PositionwiseMixing(nn.Module):
    """Transforms each position independently. To be designed."""
    def __init__(self, d_model, layer_id, n_layers):
        super().__init__()
        # TODO: the per-position transform
        pass
    def forward(self, x):                 # x: (B, T, d_model)
        # TODO
        raise NotImplementedError

class Block(nn.Module):
    def __init__(self, d_model, layer_id, n_layers):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.tmix = TokenMixing(d_model, layer_id, n_layers)
        self.pmix = PositionwiseMixing(d_model, layer_id, n_layers)
    def forward(self, x):
        x = x + self.tmix(self.ln1(x))    # pre-LN residual
        x = x + self.pmix(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab, d_model, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.ln_emb = nn.LayerNorm(d_model)            # extra post-embedding norm
        self.blocks = nn.ModuleList(
            Block(d_model, i, n_layers) for i in range(n_layers))
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
    def forward(self, idx):
        x = self.ln_emb(self.emb(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_out(x))

def loss_fn(logits, targets):
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))
```
