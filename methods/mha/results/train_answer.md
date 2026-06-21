The job I care about is sequence transduction: take a sequence of vectors $(x_1, \dots, x_n)$ and produce another sequence $(z_1, \dots, z_n)$ in which each $z_i$ has been informed by the relevant content anywhere else in the sequence. That non-local mixing is the whole point, because in language the thing that decides a word's representation is often far away — the subject a verb must agree with, the noun a pronoun refers to. The layer I would reach for today is recurrence: thread a hidden state through the sequence, $h_t = f(h_{t-1}, x_t)$, an LSTM or a GRU cell. It works, but it loses on two axes at once. The recursion is sequential by construction — I cannot compute $h_t$ before $h_{t-1}$ exists — so within a single training example there is nothing to parallelize, and the forward pass is a chain of $n$ dependent steps; on hardware where one large batched matmul dwarfs $n$ small dependent ones, this is a throughput wall that worsens exactly as sequences lengthen and memory stops me from batching more examples. And the path a signal must travel between two dependent positions is long: if position $j$ influences position $i$, that influence passes through $|i-j|$ recurrent steps both forward and backward, and long paths are precisely where gradients die. So recurrence pays $O(n)$ sequential operations and $O(|i-j|)$ path length, and both are the enemy. Convolution fixes only the first: stacked 1-D convolutions compute all positions at once, dropping the sequential count to $O(1)$, but a kernel of width $k < n$ only connects positions within its receptive field, so to let two distant positions influence each other I must stack $O(n/k)$ layers (or $O(\log_k n)$ dilated), leaving path length growing with distance, and a convolution costs a factor of $k$ more per step. I want both properties — $O(1)$ sequential and $O(1)$ path length — and no single existing layer gives me both.

The one mechanism that already nails the path-length property, in another setting, is attention. In encoder-decoder translation the decoder no longer reads the source through a single bottleneck vector; at each output step $i$ it forms a context vector as a content-weighted sum over all source annotations, $c_i = \sum_j \alpha_{ij} h_j$, with weights $\alpha_{ij} = \mathrm{softmax}_j(e_{ij})$ from a learned compatibility score $e_{ij} = a(s_{i-1}, h_j)$. Structurally, output position $i$ reaches source position $j$ through one weighted sum regardless of $|i-j|$ — constant path length — and all the weights are computed in parallel. But in every system I know this attention sits between two sequences while the within-sequence mixing is still done by an RNN that carries the sequential cost. I propose to turn the mechanism inward: let a sequence read from itself, so that each position $i$ forms its new representation as a content-weighted sum over all positions $j$ of the same sequence. This is self-attention, and built up across heads it is Multi-Head Attention (MHA). Once the within-sequence mixing happens through the weighted sum, recurrence is no longer the thing carrying information along the sequence, and the cross-positional layer inherits $O(1)$ sequential operations and $O(1)$ path length.

To build it I first need the compatibility score $a(\text{query}, \text{key})$, and here a design choice matters more than it would for a single layer, because I intend to use attention everywhere — many layers, as the only cross-positional mechanism — so the constant factor dominates wall-clock. Bahdanau's additive score runs a one-hidden-layer MLP, $e = v_a^\top \tanh(W_a q + U_a k)$, scoring each query against each key; Luong's multiplicative score is the plain inner product $q^\top k$. The two cost about the same in theory, but the additive score has to form pairwise hidden activations and run a tanh-plus-output for every (query, key) pair, and it never collapses to one score matrix multiplication. The dot product does: pack all queries into $Q$ (one row per position) and all keys into $K$, and the entire matrix of pairwise scores is $QK^\top$, a single dense GEMM — the most heavily optimized routine on the hardware, with no per-pair activations to store. For a model that is nothing but attention layers, "it is one matmul" is decisive, so I take the dot product. I keep the query/key/value roles explicit, reading the layer as a soft dictionary lookup: the query is what position $i$ is looking for, the key is what each position $j$ advertises, the value is what $j$ actually contributes if attended to. I deliberately let keys and values be different objects, because "how well does $j$ match this query" and "what does $j$ pass on" are genuinely different questions — the thing a position is recognized by need not be the thing it contributes. Packing queries, keys, and values into $Q, K, V$, the whole layer is $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top)\,V$: $QK^\top$ is the $n\times n$ matrix of pairwise scores, the row-wise softmax turns row $i$ into a distribution over positions, and the multiply by $V$ reads out the weighted sum of values. The per-layer cost is $O(n^2 d)$, the sequential count is $O(1)$ since none of the three steps waits on a previous position, and the path length is $O(1)$ since every $i$ sees every $j$ inside the single $QK^\top$ — exactly the pair of properties I could not get before, and against recurrence's $O(n d^2)$ this is cheaper whenever $n < d$, the common regime for sentence-level translation with sub-word vocabularies.

A clean expression is not automatically a trainable layer, and the first thing that worries me is the raw magnitude of $QK^\top$. The score $q\cdot k = \sum_{m=1}^{d_k} q_m k_m$ is a sum of $d_k$ products and $d_k$ is not small, so let me compute its scale rather than guess. Model the components of $q$ and $k$ as independent with mean $0$ and variance $1$, a reasonable stand-in for well-normalized activations. The mean is $E[q\cdot k] = \sum_m E[q_m]E[k_m] = 0$. The variance is the load-bearing part: since the mean is zero,
$$\mathrm{Var}(q\cdot k) = E\!\left[\Big(\sum_m q_m k_m\Big)\Big(\sum_l q_l k_l\Big)\right] = \sum_m \sum_l E[q_m k_m q_l k_l].$$
For $m \ne l$ all four factors are independent, so the term factors into a product of zero means and vanishes; for $m = l$ the term is $E[q_m^2]E[k_m^2] = 1$, and there are $d_k$ such diagonal terms, giving $\mathrm{Var}(q\cdot k) = d_k$. The logits therefore have standard deviation $\sqrt{d_k}$. That is the failure mode: as $d_k$ grows, the logits spread with standard deviation $\sqrt{d_k}$, and a softmax over wide-spread logits pushes its mass onto the largest entries, driving the distribution toward one-hot. In that saturated regime its Jacobian $\partial p_i / \partial s_j = p_i(\delta_{ij} - p_j)$ is tiny in every entry, so gradients through the weights nearly vanish — which lines up with the reported fact that dot-product and additive scores behave similarly at small $d_k$ but the raw dot product falls behind as $d_k$ grows (the additive score, with its bounded nonlinearity, is not a bare sum of $d_k$ products). The fix falls straight out of the variance: I want unit-variance logits, the dot product has variance $d_k$, so I divide by its standard deviation,
$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$
The scaling is not a tuning knob; it is the exact correction for a variance I can compute, and it keeps the speed of the single-matmul score while curing its degradation at large head width.

There is a deeper limitation that scaling does not touch. The output for position $i$ is $\sum_j \alpha_{ij} v_j$ with $\sum_j \alpha_{ij} = 1$ — a convex combination, an average. If position $i$ genuinely needs two different things at once, say the subject it must agree with and the adverb that modifies it, a single distribution can put mass on both, but the output is then one blended average of their values and the two signals are smeared into the same vector through one set of weights. A single attention function can only commit to one weighted-average view of the sequence; that is an expressivity ceiling. So I run $h$ attention functions in parallel, each free to attend to a different aspect, and keep their results distinct until the end. To make them genuinely different, each looks at the sequence through its own learned linear lens: project queries, keys, and values into a lower-dimensional subspace with head-specific matrices, run scaled dot-product attention there, and repeat $h$ times with $h$ different projection sets. Then I concatenate the per-head outputs and pass them through one more learned map,
$$\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V), \qquad \mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O,$$
with $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}}\times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}}\times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$. The output projection $W^O$ is not decoration: concatenation merely stacks $h$ independent blocks side by side, and $W^O$ is what linearly recombines them into a single coherent $d_{\text{model}}$-vector — without it the heads never interact. And separate learned $W^Q, W^K, W^V$ are what let each head choose which subspace to match in independently of what content it carries. The remaining question is how big each head should be: if every head kept the full $d_{\text{model}}$ width, the $n\times n$ score-and-value work would multiply by $h$, exactly the blowup I am avoiding. So I set $d_k = d_v = d_{\text{model}}/h$. Then each head does its $QK^\top$ and $A V$ in a $d_{\text{model}}/h$-wide space at cost $O(n^2 d_{\text{model}}/h)$, and $h$ of them together cost $h\cdot O(n^2 d_{\text{model}}/h) = O(n^2 d_{\text{model}})$ — the same pairwise-attention order as one full-width head. The learned projections still cost the usual model-width linear work $O(n d_{\text{model}}^2)$, which a single full-width attention layer also pays; splitting the model dimension across heads is what keeps the part that would have exploded from exploding. With $d_{\text{model}} = 512$ and $h = 8$ this gives $d_k = d_v = 64$: eight independent projected views inside one head's compute budget, with $h$ a width-allocation choice rather than a free lunch, since more heads means thinner subspaces.

One requirement remains, from the autoregressive setting. When self-attention sits inside a decoder predicting the next token, position $i$ must not look at positions $j > i$, or it would peek at the answer. This is trivial to enforce: before the softmax, set the score for every illegal pair $(i,j)$ with $j>i$ to $-\infty$; since the softmax exponentiates and $\exp(-\infty) = 0$, those positions get exactly zero weight and the remaining causal weights renormalize to one. A single masked-fill on the score matrix turns the layer causal with no other change. The whole thing is also light to engineer: the three projections $W^Q, W^K, W^V$ are linear maps that I fuse into one $\mathtt{nn.Linear}$ of width $3 d_{\text{model}}$ — one GEMM instead of three — then split; the per-head split is a reshape into $(h, d_k)$ and a transpose putting the head axis next to the batch axis, so the $n\times n$ attention runs as a batched matmul over heads, and concatenation is the inverse reshape. Nothing needs a hand-written kernel, which was the point of choosing the dot-product score.

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """softmax(Q K^T / sqrt(d_k)) V.  q,k,v: (batch, n_head, seq_len, d_k)."""
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)        # variance of logits back to 1
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # future logits -> -inf -> weight 0
    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v                                            # weighted sum of values


class SequenceMixing(nn.Module):
    """The cross-positional sub-layer implemented as h parallel scaled-dot-product
    heads, concatenated and projected back to d_model by W^O."""

    def __init__(self, config, causal=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head              # d_k = d_v = d_model / h
        self.causal = causal
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # fused Q,K,V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)      # W^O
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)     # project, split into Q, K, V
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, n_head, T, d_k)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            m = self.mask[:, :, :T, :T] if self.causal else None
            y = scaled_dot_product_attention(q, k, v, mask=m, dropout=self.attn_dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C)       # concat heads -> d_model
        y = self.resid_dropout(self.c_proj(y))                 # W^O
        return y
```
