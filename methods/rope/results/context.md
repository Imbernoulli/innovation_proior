# Context: Injecting position into Transformer self-attention

## Research question

Transformer self-attention has no built-in notion of order. For a sequence of token embeddings `{x_1, ..., x_N}`, the attention output at position `m` is

```
o_m = sum_n a_{m,n} v_n,   a_{m,n} = softmax_n( q_m^T k_n / sqrt(d) )
```

and if `q, k, v` are computed from `x` alone, the whole computation is permutation-equivariant: shuffle the tokens and the outputs shuffle identically. "The dog bit the man" and "The man bit the dog" become indistinguishable. Order, which is central to language, is invisible.

So position information has to be added explicitly. The question is *how*. The single place order needs to enter, in order to change which tokens attend to which, is the attention logit `q_m^T k_n`. The goal a good solution should hit:

- The logit should depend on the two contents `x_m, x_n` and on the **relative offset `m - n`**, not on the absolute indices `m` and `n` separately — language dependencies are about how far apart tokens are, not where they sit in the buffer.
- It should impose a sensible inductive bias: tokens that are far apart should, all else equal, interact less.
- It should not cap the sequence length, and should behave gracefully at lengths not seen in training.
- It should ideally inject position as a **per-token transformation** of `q` and `k`, so it survives in efficient/linear attention variants that never materialize the full `N x N` logit matrix.

No existing scheme hits all of these at once.

## Background

**Permutation-equivariance of self-attention.** With `q, k, v` linear in `x`, attention is a function of the *set* of inputs, not the sequence. This is the formal pain point that every position-encoding method exists to fix.

**Sinusoidal absolute encoding (Vaswani et al., 2017).** A fixed (non-learned) vector `p_i` is added to each token embedding before projection, `f_t(x_i, i) = W_t (x_i + p_i)`, with

```
p_{i,2t}   = sin( i / 10000^{2t/d} )
p_{i,2t+1} = cos( i / 10000^{2t/d} )
```

Each pair of dimensions `(2t, 2t+1)` is a sinusoid whose wavelength runs geometrically from `2*pi` up to roughly `10000 * 2*pi`. This is a multi-resolution "clock": fast hands resolve nearby positions, slow hands carry coarse global position. The encoding is added to the embedding, and the relationship between positions is left for the network to discover from the absolute signals.

**Learned absolute encoding (BERT, GPT, ALBERT, ELECTRA).** Replace the fixed `p_i` with a trainable vector per index, up to a maximum length `L`. Flexible, but caps length at `L` and learns nothing for positions beyond it.

**The cross-term problem of additive absolute encoding.** Whatever `p` is, additive absolute encoding feeds `W_q(x_m + p_m)` and `W_k(x_n + p_n)` into the logit. Expanding,

```
q_m^T k_n = x_m^T W_q^T W_k x_n
          + x_m^T W_q^T W_k p_n
          + p_m^T W_q^T W_k x_n
          + p_m^T W_q^T W_k p_n
```

Three of the four terms involve absolute `p_m` or `p_n`. The dependence is on absolute position, not on `m - n`. The relative structure is not expressed; it has to be learned, indirectly, from absolute signals.

**Distance decay as a language prior.** Dependencies in text typically weaken with distance — adjacent and nearby words interact strongly, distant ones less so. A position scheme that *builds in* a decay of the positional contribution with `|m - n|` is encoding a true prior about the data, not a benchmark trick. This is a property of language known independently of any particular encoding.

**Linear / kernelized attention (Katharopoulos et al., 2020; Shen et al., 2021).** Softmax attention costs `O(N^2)` because it forms every `q_m^T k_n`. Linear attention replaces the similarity with a factorized feature map,

```
Attention(Q,K,V)_m = ( sum_n phi(q_m)^T psi(k_n) v_n ) / ( sum_n phi(q_m)^T psi(k_n) )
```

with non-negative `phi, psi` (e.g. `elu(x) + 1`). Using associativity, `sum_n psi(k_n) v_n^T` and `sum_n psi(k_n)` are precomputed once, giving `O(N)`. The defining constraint: position must enter as a transformation of the per-token features `phi(q_m)` and `psi(k_n)`. Anything that lives only inside the `N x N` matrix of pairwise logits is, by construction, incompatible — it destroys the factorization that buys the linear cost.

## Baselines

**Sinusoidal / learned absolute (above).** Core idea: add a position vector, let attention sort out the rest. Gap: absolute and additive — the cross-term expansion shows the logit depends on absolute indices; no clean relative dependence; learned variants cap length.

**Shaw et al. (2018), Self-Attention with Relative Position Representations.** The seminal relative scheme. Position is injected as a learned *relative* embedding added to keys and values:

```
f_q(x_m)    = W_q x_m
f_k(x_n, n) = W_k ( x_n + p~^k_r )
f_v(x_n, n) = W_v ( x_n + p~^v_r ),   r = clip(m - n, r_min, r_max)
```

`p~^k_r, p~^v_r` are trainable vectors indexed by the (clipped) relative distance. First method to make attention explicitly relative. Gaps: clipping discards all distinction beyond a window (`r` saturates), so it cannot represent fine long-range offsets; it adds learned tables (extra parameters, no closed form); it also perturbs the *values*, not just the logit; and the relative term is mixed into the expanded dot product, so it is not a per-token transform — it does not survive linear attention.

**Transformer-XL (Dai et al., 2019) and the decomposition family.** Start from the additive expansion above and surgically rewrite terms: replace absolute `p_n` with a sinusoidal *relative* encoding `p~_{m-n}`; replace the `p_m` factors in the third and fourth terms with two trainable global vectors `u, v` independent of the query position; and split `W_k` into separate content and position projections `W_k, W~_k`:

```
q_m^T k_n = x_m^T W_q^T W_k x_n
          + x_m^T W_q^T W~_k p~_{m-n}
          + u^T W_q^T W_k x_n
          + v^T W_q^T W~_k p~_{m-n}
```

Position is removed from the values. Gap: it is an *edit* of the absolute-additive decomposition — each term is patched by hand toward relative-ness, with extra learned vectors and projections; the relative term still sits inside the pairwise logit.

**T5 (Raffel et al., 2020).** Collapses the whole expansion to a content term plus a single learned scalar **relative bias**, bucketed by distance:

```
q_m^T k_n = x_m^T W_q^T W_k x_n + b_{m,n}
```

Minimal and effective. Gap: `b_{m,n}` is a learned lookup, not a closed form, and it lives squarely in the `N x N` logit matrix — exactly the object linear attention refuses to build.

**DeBERTa (He et al., 2020) and related (Ke et al. 2020; Huang et al. 2020).** Keep the two "content x position" cross terms with relative embeddings `p~_{m-n}`, argue those carry the relative information; drop or restructure the rest. Same shared limitation: position is an additive term in the expanded logit, tied to the `N x N` matrix, parameterized by learned relative tables/biases.

**The pattern across all relative baselines.** Every one of them *starts from "add a position vector, then expand the dot product"* and then edits the resulting terms to look relative. Every one parameterizes the relative signal with a learned table or a learned bias rather than a closed form; and none is compatible with linear attention, because the relative signal always ends up inside the pairwise logit matrix.

## Evaluation settings

The natural yardsticks:

- **Machine translation:** WMT 2014 English-German (~4.5M sentence pairs), BPE vocabulary (~37k joint), BLEU on the test set; standard `fairseq` Transformer-base recipe (Adam, inverse-sqrt schedule with linear warmup, label smoothing 0.1, checkpoint averaging, beam search).
- **Masked language-model pre-training:** BookCorpus + English Wikipedia, BERT-base architecture, MLM loss as the tracked quantity; AdamW.
- **Downstream fine-tuning:** GLUE tasks — MRPC, SST-2, QNLI, STS-B, QQP, MNLI; metrics F1 (MRPC, QQP), Spearman (STS-B), accuracy (others).
- **Linear-attention setting:** Performer-style attention on character-level Enwik8, tracking pre-training loss.
- **Long-document setting:** Chinese long-text matching (e.g. CAIL2019-SCM), where inputs exceed the usual 512-token window, to probe behavior past the trained length.

Tooling that already exists: PyTorch, `fairseq`, the HuggingFace Transformers and Datasets libraries, and standard BERT/Transformer-base architectures.

## Code framework

A bare self-attention harness. Position injection is a single empty slot.

```python
import torch
import torch.nn as nn

class PositionStrategy:
    """How position information enters q/k."""
    def __init__(self, head_dim):
        # TODO: precompute whatever per-position quantities the scheme needs
        pass

    def apply(self, q, k, positions):
        # q, k: [batch, heads, seq, head_dim]; positions: [seq]
        # TODO: transform q and k so that the eventual logit q^T k
        #       carries position information the way we want
        return q, k

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, position: PositionStrategy):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.position = position

    def forward(self, x, positions=None, mask=None):
        B, T, _ = x.shape
        if positions is None:
            positions = torch.arange(T, device=x.device)

        def split(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))

        q, k = self.position.apply(q, k, positions)  # the slot

        logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        attn = logits.softmax(dim=-1)
        o = attn @ v
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.Wo(o)
```

The empty slot should be filled so that the logit `q^T k` depends on contents and relative offset, ideally as a per-token transform that also survives the linear-attention factorization.
