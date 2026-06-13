Let me start from what actually slows me down when I train a sequence model. I want a layer that takes a sequence of vectors `(x_1, ..., x_n)` and produces another sequence `(z_1, ..., z_n)`, where each `z_i` has been informed by the relevant content anywhere else in the sequence — that non-local mixing is the whole job, because in language the thing that decides a word's representation is often far away (the subject that a verb must agree with, the noun a pronoun refers to). The layer I reach for today is recurrence: I thread a hidden state through the sequence, `h_t = f(h_{t-1}, x_t)`, an LSTM or a GRU cell. It works, but two things hurt. First, that recursion is sequential by construction — I cannot compute `h_t` until `h_{t-1}` exists, so within a single training example there is nothing to parallelize; the whole forward pass is a chain of `n` dependent steps. On modern hardware, where one big batched matmul is vastly cheaper than `n` small dependent ones, this is the throughput wall, and it gets worse exactly as sequences get longer and memory stops me from batching more examples to hide the latency. Second, think about how far a signal has to travel between two positions that depend on each other. If position `j` influences position `i`, that influence has to pass through `|i - j|` recurrent steps, forward and backward. Long paths are exactly where gradients die and dependencies fail to be learned. So the recurrent layer pays `O(n)` sequential operations and an `O(|i-j|)` path length between positions, and both of those are the enemy.

What would the ideal layer look like? I want the number of *sequential* steps to not grow with `n` at all — ideally `O(1)`, everything computed in parallel — and I want the maximum path length between any two positions to be short and constant, so a far-away dependency is no harder to learn than a nearby one. Can convolution give me that? A convolutional sequence model computes every position at once, so the sequential count drops to `O(1)`, good. But a kernel of width `k < n` only sees a window of `k` around each position; to let position `i` and position `j` actually influence each other I have to stack layers until their receptive fields overlap — `O(n/k)` layers for plain kernels, `O(log_k n)` for dilated ones. So the path length between two positions still grows with their distance. I've fixed parallelism but not the long-range path. And a convolution costs a factor of `k` more than a recurrent layer per step. Convolution gets me halfway.

So I have two properties I want — `O(1)` sequential, `O(1)` path length — and no single existing layer gives me both. Let me look harder at the one mechanism that already nails the path-length property in another setting: attention. In the encoder-decoder translation models, the decoder doesn't read the source through a single bottleneck vector anymore; at each output step `i` it forms a context vector as a content-weighted sum over *all* source annotations, `c_i = Σ_j α_ij h_j`, with weights `α_ij = softmax_j(e_ij)` coming from a learned compatibility score `e_ij = a(s_{i-1}, h_j)` between the current decoder state and source position `j`. Stare at what this does structurally. Output position `i` is connected to source position `j` through *one* weighted sum, no matter how far apart they are — a path of constant length, independent of `|i - j|`. And the weights for all `j` are computed in parallel from the same `s_{i-1}`. This is precisely the property convolution and recurrence couldn't give me: cross-positional reach at constant path length, computed in parallel across positions.

But in every system I know, this attention sits *between* the decoder and the encoder — it bridges two sequences — while the within-sequence mixing is still done by the RNN. The RNN is still there carrying the sequential cost. If the same content-weighted-sum mechanism reads from the sequence it is updating, each position `i` can form its new representation as a weighted sum over all positions `j` of that same sequence. Then recurrence is no longer the thing that carries information along the sequence; the within-sequence mixing happens through the weighted sum itself. That gives the cross-positional layer `O(1)` sequential operations, because the scores are parallel matmuls and a softmax, and `O(1)` path length, because every position reaches every other in one hop. Now I have to build the mechanism and look for its failure modes.

First I need the compatibility score `a(query, key)`. There are two families on the table. Bahdanau's *additive* score runs a little feed-forward net: `e = v_a^T tanh(W_a q + U_a k)` — a one-hidden-layer MLP scoring each query against each key. Luong's *multiplicative* family is cheaper: the plain `dot` score is just `q^T k`, an inner product, or `q^T W_a k` with a learned matrix in between. Theoretically the two cost about the same. But I'm not building one attention layer; I'm planning to use attention *everywhere*, many layers, as the only cross-positional mechanism in the whole model. So the constant factor is going to dominate my wall-clock, and that changes the calculus. The additive score can batch the query and key projections, but it still has to form pairwise hidden activations and run a tanh-plus-output score for every (query, key) pair; it does not collapse to one score matrix multiplication. The dot-product score does: if I pack all the queries into a matrix `Q` (one row per position) and all the keys into `K`, then the entire matrix of pairwise scores is just `Q K^T`, a single dense multiplication. That hits the most heavily optimized routine on the hardware — dense GEMM — and it's far more space-efficient because there are no per-pair hidden activations to store. For a model that is *nothing but* attention layers, "it's one matmul" is decisive. I'll take the dot product.

Let me make the abstraction clean before I write the matrix form, because the query/key roles want to be explicit. I have a query (what position `i` is looking for), a set of keys (what each position `j` advertises about itself), and a set of values (what each position `j` actually contributes if attended to). The output for position `i` is a weighted sum of the values, where the weight on value `j` is the softmax-normalized compatibility of query `i` with key `j`. Reading it as a soft dictionary lookup: the query is matched against every key, the match scores become a probability distribution, and I read out the expected value under that distribution. Now I want the keys and values to be different objects, not the same vector, because "how well does `j` match this query" and "what does `j` contribute" are genuinely different questions — the thing a position is recognized *by* need not be the thing it passes *on*. So values get their own slot. Pack queries into `Q`, keys into `K`, values into `V`, all with one row per position. Then the whole layer is

  Attention(Q, K, V) = softmax(Q K^T) V.

`Q K^T` is the `n×n` matrix of all pairwise scores; softmax along each row turns row `i` into a distribution over positions; multiplying by `V` reads out, for each `i`, the weighted sum of values. One matmul, one softmax, one matmul. The cost per layer is `O(n^2 · d)` — the `Q K^T` and the `· V` are each `n×n` against a `d`-wide thing — the sequential operation count is `O(1)` because none of these three steps waits on a previous position, and the path length between any two positions is `O(1)` because every `i` sees every `j` directly inside the single `Q K^T`. That is exactly the pair of properties I couldn't get before. And the `O(n^2 · d)` cost is actually competitive: against the recurrent layer's `O(n · d^2)`, attention is cheaper whenever `n < d`, which is the common regime for sentence-level translation with sub-word vocabularies where `d` is several hundred and `n` is a few tens.

Now I have to check where the formula breaks, because a clean expression is not automatically a trainable layer. The first thing I'm uneasy about is the raw magnitude of `Q K^T`. The score `q · k = Σ_{m=1}^{d_k} q_m k_m` is a sum of `d_k` products, and `d_k` — the dimension of each query/key vector — is not small. Let me reason about the scale carefully rather than guess. Suppose, as a working model, that the components of `q` and `k` are independent, each with mean 0 and variance 1 — a reasonable stand-in for well-normalized activations. What's the distribution of `q · k`? The mean is easy: `E[q · k] = Σ_m E[q_m k_m] = Σ_m E[q_m] E[k_m] = 0` by independence and zero mean. The variance is the load-bearing part. Since the mean is zero, `Var(q·k) = E[(q·k)^2] = E[(Σ_m q_m k_m)(Σ_l q_l k_l)] = Σ_m Σ_l E[q_m k_m q_l k_l]`. Split the double sum by whether `m = l`. For `m ≠ l`, all four factors are independent, so `E[q_m k_m q_l k_l] = E[q_m]E[k_m]E[q_l]E[k_l] = 0` — every cross term vanishes because the means are zero. For `m = l`, the term is `E[q_m^2 k_m^2] = E[q_m^2] E[k_m^2] = 1 · 1 = 1`. There are `d_k` such diagonal terms, so `Var(q·k) = d_k`. The dot product has standard deviation `√d_k`.

That gives me the failure mode. As `d_k` grows, the logits going into the softmax don't stay `O(1)` — they spread out with standard deviation `√d_k`. A softmax of logits with large spread pushes most of the mass onto the largest entries, driving some weights toward 1 and the rest toward 0. In that saturated regime its Jacobian becomes very small: if `p = softmax(s)`, then `∂p_i/∂s_j = p_i(δ_ij − p_j)`, and every entry is tiny when the distribution is nearly one-hot. So at large `d_k`, raw dot-product attention can produce weak gradients through the weights. This lines up with the reported behavior: additive and dot-product scores behave similarly at small `d_k`, but the raw dot-product score falls behind as `d_k` grows. The feed-forward additive score is not a bare sum of `d_k` independent products in the same way, and its bounded nonlinearity changes the scale behavior. The diagnosis I can act on is the variance I just computed: the raw dot product's magnitude grows with `d_k`.

The fix falls right out of the variance. I want the logits to have variance `1` going into the softmax, not `d_k`. The dot product has variance `d_k`, so I divide by its standard deviation, `√d_k`:

  Attention(Q, K, V) = softmax( Q K^T / √d_k ) V.

Now the scaled logits have variance `1` under the unit-variance assumption, the softmax stays away from saturation, and I keep the speed of the single-matmul dot product. The division addresses the failure mode that made the raw dot-product score degrade at larger `d_k`, without switching to the slower per-pair MLP score. The scaling isn't a tuning knob; it's the correction for a variance I can compute directly.

Let me push on the next weak spot, because there's something a single softmax-weighted sum fundamentally cannot do. The output for position `i` is `Σ_j α_ij v_j` with `Σ_j α_ij = 1` — a convex combination of the value vectors. That's an *average*, just a content-weighted one. Now suppose position `i` genuinely needs to gather two different things at once — say, the subject it must agree with *and* the adverb that modifies it, which live at different places and matter for different reasons. A single distribution `α_i` over positions can put mass on both, sure, but then the output is one blended average of their values, and the blend smears the two signals together; I can't keep "the agreement information" and "the modifier information" separate, because they've been added into the same vector through one set of weights. With a single attention function, averaging inhibits the layer from attending to several distinct kinds of things at the same time — one head can only commit to one weighted-average view of the sequence. That's a real expressivity ceiling, not a tuning issue.

So I want several attention functions operating at once, each free to attend to a different aspect, and their results kept distinct until the very end. The way to make them genuinely different is to let each one look at the sequence through its *own* learned linear lens: project the queries, keys, and values down into a lower-dimensional subspace with head-specific matrices, run scaled dot-product attention there, and do this `h` times in parallel with `h` different sets of projections. Head 1 might learn projections that surface positional/syntactic relations; head 2 might surface a coreference-like relation; each operates in its own representation subspace and produces its own weighted average there. Then I concatenate the `h` per-head outputs and pass them through one more learned linear map to mix them back into the model dimension:

  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V),
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O,

with `W_i^Q, W_i^K ∈ R^{d_model × d_k}`, `W_i^V ∈ R^{d_model × d_v}`, and `W^O ∈ R^{h d_v × d_model}`. The point of `W^O` is not decoration: the concatenation just stacks `h` independent blocks side by side, and `W^O` is what lets the layer linearly recombine the heads' contributions into a single coherent `d_model`-vector — without it, the heads would never get to interact. And the projections being learned and separate is what lets each head choose *which* subspace to match in (via `W^Q, W^K`) independently of *what content* it passes on (via `W^V`); a head can match on one relation and carry a different signal.

Now, how big should each head be, and how many heads? If I gave every head the full `d_model` dimension I'd multiply the pairwise score-and-value work by `h`, which is exactly the kind of cost blowup I'm trying to avoid. But I have a free parameter: the per-head dimension `d_k = d_v`. If I set `d_k = d_v = d_model / h`, then each head does the `QK^T` and `A V` work in a `d_model/h`-wide space at cost `O(n^2 · d_model/h)`, and `h` of them together cost `h · O(n^2 · d_model/h) = O(n^2 · d_model)` — the same pairwise-attention order as one full-width attention function. The learned projections still cost model-width linear work, `O(n · d_model^2)`, but a single full-width query/key/value/output attention layer pays that same kind of projection cost; the part that would have exploded is the per-head `n×n` interaction if every head stayed full width. So I split the model dimension across the heads rather than stacking full-width heads. With `d_model = 512` and `h = 8`, that gives `d_k = d_v = 64` per head: the dot products are still over a useful subspace, and eight independent projected views fit inside the same pairwise-attention budget. More heads means more simultaneous relations but thinner subspaces, so `h` is a width allocation choice, not a free lunch.

One more requirement before code, from the autoregressive setting. When I use self-attention inside a decoder — a language model predicting the next token — position `i` must not be allowed to look at positions `j > i`, or it would peek at the answer it's supposed to predict. With this attention that's straightforward to enforce: before the softmax, set the score for every illegal pair `(i, j)` with `j > i` to `-inf`. Since `softmax` exponentiates, `exp(-inf) = 0`, so those positions get exactly zero weight and contribute nothing, while the remaining causal weights still renormalize to sum to one. A single masked-fill on the score matrix turns the layer causal, with no change to the rest of the computation.

Let me also notice, while I'm here, how lightweight the whole thing is to engineer. The three projections `W^Q, W^K, W^V` for a head are just linear maps; across all heads I can compute all of `Q`, `K`, `V` for every head with a few large matrix multiplications instead of many small ones — in fact I can fuse the query, key, and value projections for self-attention into a single `nn.Linear` of width `3 · d_model` applied to the input, then split the result, which is one GEMM instead of three. The per-head split is then just a reshape of that output into `(h, d_k)` and a transpose to put the head axis next to the batch axis, so the `n×n` attention runs in a batched matmul over heads. Concatenation back is the inverse reshape. Nothing here needs a hand-written kernel; it's all standard dense ops, which was the point of choosing the dot-product score.

The cross-positional slot can now be filled with packed queries/keys/values, `Q K^T / √d_k`, optional mask, softmax, weighted sum of values, then the wrapper that projects into `h` heads, applies the attention, concatenates, and projects out.

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    # q, k, v: (batch, n_head, seq_len, d_k)
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # Q K^T / sqrt(d_k): variance back to 1
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # causal: future logits -> -inf -> weight 0
    attn = scores.softmax(dim=-1)                          # row-wise distribution over positions
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v                                        # weighted sum of values


class SequenceMixing(nn.Module):
    """The cross-positional sub-layer implemented as h scaled-dot-product
    heads in parallel, concatenated and projected back to d_model."""

    def __init__(self, config, causal=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head          # d_k = d_v = d_model / h
        self.causal = causal
        # fused Q,K,V projection: one GEMM instead of three (self-attention)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)   # W^O
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)        # project, then split into Q,K,V
        # reshape each into heads: (B, T, C) -> (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        if self.flash:
            # fused kernel: same softmax(QK^T/sqrt(d_k))V, with optional causal masking
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
        else:
            m = self.mask[:, :, :T, :T] if self.causal else None
            y = scaled_dot_product_attention(q, k, v, mask=m, dropout=self.attn_dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # concat heads back to d_model
        y = self.resid_dropout(self.c_proj(y))                   # W^O output projection
        return y
```

I started stuck with recurrence: it mixes positions but only through an `O(n)` sequential chain and an `O(|i-j|)` path between dependent positions, which throttles training and makes long-range dependencies hard. Convolution parallelized the sequential axis to `O(1)` but left the path length growing with distance and added a factor of `k` in cost. The content-weighted-sum bridge already connected any two positions at constant path length, in parallel, so I turn it inward and let a sequence read from itself. For the score I choose the dot product over the additive MLP because, used in every layer, one dense score matmul wins on speed and memory; packing it into `softmax(Q K^T) V` gives `O(1)` sequential ops and `O(1)` path length at `O(n^2 d)` pairwise cost. Computing the variance of `q · k` under unit-variance components shows it grows as `d_k`, pushing the softmax toward saturation at large head width, so dividing by `√d_k` restores unit variance under that assumption. A single head produces one weighted average, smearing distinct relations together, so I run `h` heads in parallel through their own learned low-dimensional projections, keep their outputs separate, concatenate, and mix them with `W^O`; setting each head's width to `d_model/h` keeps the total pairwise-attention cost at `O(n^2 d_model)` while the projection work stays at the usual model-width linear cost. Masking future logits to `-inf` before the softmax makes the layer causal for autoregressive decoding because `exp(-inf) = 0`. The cross-positional slot is filled by fused QKV projection, reshape into heads, scaled dot-product attention with optional causal mask, concat, and output projection, all built from dense matmuls and a softmax.
