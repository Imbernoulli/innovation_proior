Let me start from what actually hurts. I have a sequence of vectors `(x_1, ..., x_n)`, each `d`-dimensional, and I want a layer that turns it into a new sequence `(z_1, ..., z_n)` where every `z_i` has been allowed to look at every other `x_j` and pull in whatever it needs, with the amount it pulls decided by the *content* of the two — not by how far apart they sit. The way everyone builds this today is to thread the positions through a recurrence, `h_t = f(h_{t-1}, x_t)`, so each position's representation is a function of all the earlier ones. It works, it's state of the art for translation, but it has two costs baked into its shape, and both are killing me. First, `h_t` cannot be computed until `h_{t-1}` exists, so the positions have to be visited in order — `O(n)` sequential steps, zero parallelism within a single example, and on long sequences I run out of memory before I can batch enough examples to fill the hardware. Second, and this is the one I care about more, the shortest path a signal travels between position `i` and position `j` runs through every position in between, so it's `O(|i - j|)` long. And the longer the path a forward or backward signal has to survive, the harder it is to learn the dependency across it — that's the whole reason long-range dependencies are the perennial pain in sequence models. So I want to collapse both of these: `O(1)` sequential operations, and a path length between any two positions that doesn't grow with how far apart they are.

What are my options for getting rid of the recurrence? Convolution is the obvious one — ConvS2S, ByteNet, the Neural GPU all compute every position in parallel with a width-`k` filter, so the sequential depth drops to `O(1)`. But a single convolution with `k < n` only sees a window of size `k` around each position; to let position `i` talk to position `j` when `|i - j| > k`, I have to stack layers until the receptive fields overlap, which means `O(n/k)` layers for contiguous kernels or `O(log_k n)` for dilated ones. So I've traded the recurrence's `O(n)` sequential cost for a path length that still grows with the sequence, just more slowly — `log_k n` instead of `n`. That's better but it isn't the constant path length I'm after, and each convolutional layer costs `O(k · n · d^2)`, a factor of `k` more than recurrence. Convolution is the wrong primitive: it's local by construction, and locality is exactly what forces the path to grow.

So I should think about the thing that *already* connects arbitrary positions in one step. Attention. Bahdanau's idea, originally invented to relieve the fixed-vector bottleneck in translation: at output step `i`, score every source annotation `h_j` against the current decoder state, `e_ij = a(s_{i-1}, h_j)`, normalize across `j` with a softmax, `alpha_ij = softmax_j(e_ij)`, and take the weighted average `c_i = sum_j alpha_ij h_j`. Look at what that gives me for free — it connects position `i` to *all* `j` in a single, content-based, distance-agnostic operation. The weight on `h_j` depends on how well it matches the query, not on `|i - j|`. That's the constant path length I want; there's no chain to traverse, just one weighted sum. The reason this hasn't already solved my problem is purely that in every system I know of, attention is glued on top of a recurrent encoder: the `h_j` it averages are themselves produced sequentially. The attention is `O(1)` sequential, but the thing it attends over is `O(n)` sequential. So the question almost asks itself — what if I throw away the recurrence entirely and let attention do *all* the position-to-position communication? Let each position attend to all the others directly, with no RNN underneath. The annotations and the queries would both just be the layer's own inputs. A position attending to the other positions of its own sequence — attention turned inward.

Let me try to write that down concretely and see if it holds together. For each position I need a *query* (what am I looking for), and for every position I need a *key* (what do I offer as a match target) and a *value* (what I actually hand over if matched). If I just used the raw `x_i` for all three roles, the space I match in and the space I carry information in would be forced to be the same, which is a needless constraint, so I'll learn three linear maps: `Q = X W^Q`, `K = X W^K`, `V = X W^V`, where `X` stacks the `x_i` as rows. Then the output at position `i` is a weighted average of the value rows, with weights from matching its query against all keys: `z_i = sum_j softmax_j(score(q_i, k_j)) v_j`. Stack that over all `i` and it's a clean matrix expression: weights `= softmax(Q K^T)` row-wise, output `= softmax(Q K^T) V`. Cost: forming `Q K^T` is `n × n` scores each from a `d`-dimensional dot, so `O(n^2 · d)`, and the projections are `O(n · d^2)`. Compare to recurrence's `O(n · d^2)`: the mixing term `n^2 · d` is *smaller* than `n · d^2` exactly when `n < d`, and for the sub-word vocabularies strong translation systems use, sequences really are shorter than the hidden width. So in the regime I live in, this is not just more parallel, it's cheaper per layer. And it's `O(1)` sequential — one batched matmul, a softmax, another matmul — and the path between any two positions is `O(1)`. That's all three of my structural wishes at once.

Now I have to pick the score function, and there are two on the table. Additive, like Bahdanau: feed the query and key into a small feed-forward net with a hidden layer and read off a scalar. Or multiplicative, like Luong: just dot the query and key, `score = q^T k`, maybe with a learned matrix in the middle, `q^T W k`. In raw accuracy they're about even. But I'm going to be doing this `n^2` times per layer and stacking many layers, so the constant factor is not a detail — it's the difference between feasible and not. The dot-product version is a single matrix multiply `Q K^T`, which maps straight onto the most heavily optimized kernel on the hardware, dense matrix multiplication; the additive version is a per-pair little MLP, which doesn't. So multiplicative, for speed and memory. The whole point of dropping recurrence was to lean on big parallel matmuls, and I should keep leaning on them in the scoring too.

So my layer is `softmax(Q K^T) V`. Let me sanity-check it before I get attached, by actually thinking through what the softmax sees. The entry `(Q K^T)_{ij}` is `q_i · k_j = sum_{c=1}^{d_k} q_{i,c} k_{j,c}`, a sum of `d_k` products, where `d_k` is the width of the query/key vectors. I need to be careful, because there's a known failure of unscaled dot-product scoring at large width and I want to understand it, not just patch it. Suppose, as a rough model, that the components of `q` and `k` are independent with mean `0` and variance `1` — which is roughly what normalized activations look like. Then each product `q_{i,c} k_{j,c}` has mean `E[q_{i,c}] E[k_{j,c}] = 0` by independence, and the whole dot product `q_i · k_j` has mean `0`. Its variance: the terms are independent, so the variance of the sum is the sum of the variances, and `Var(q_{i,c} k_{j,c}) = E[q_{i,c}^2] E[k_{j,c}^2] - 0 = 1 · 1 = 1`, so `Var(q_i · k_j) = sum_{c=1}^{d_k} 1 = d_k`. The logits going into the softmax have standard deviation `sqrt(d_k)`. So as I make the matching space wider, the logits get systematically larger, and a softmax fed large logits collapses toward one-hot — almost all the mass on the single largest entry. And a near-one-hot softmax has almost no gradient: the Jacobian of softmax is `diag(p) - p p^T`, which vanishes as `p` goes to a corner of the simplex. So at large `d_k` the scores saturate the softmax and the gradients die. That's the documented degradation of unscaled dot-product attention at large width, and now I can see it's not bad luck, it's the `sqrt(d_k)` growth of the logit scale.

The fix falls right out of the variance computation. I don't want the logits to grow with `d_k`; I want them to stay at a fixed scale regardless of width. The standard deviation is `sqrt(d_k)`, so divide the logits by exactly that: `softmax(Q K^T / sqrt(d_k))`. After dividing, the variance of each logit is `d_k / (sqrt(d_k))^2 = 1`, back to unit scale, independent of width. So the attention becomes

  Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V.

The `1 / sqrt(d_k)` isn't a tuning knob I reached for; under this variance model it is the scale that cancels the width-dependence I just derived. I could imagine learning a temperature instead, but why start unstable and ask optimization to rediscover the scale I can compute directly? Scale by `sqrt(d_k)` and move on.

Now I stare at `softmax(Q K^T / sqrt(d_k)) V` for a while and a different worry surfaces. This produces, for each query, *one* probability distribution over positions, and the output is *one* weighted average of values. But a position usually needs to gather several different *kinds* of information at once: maybe it wants the syntactic head it depends on, and also a semantically related word somewhere else, and also its immediate neighbors. A single softmax has to express all of those as one distribution and the output blends them into one average — and averaging is lossy: if head A wants to look hard at position 3 and relation B wants to look hard at position 17, a single distribution has to compromise between them, and the compromise smears both. One attention function is one relation. I have several relations to capture. The averaging inside a single head actively inhibits keeping them distinct.

So let me run several attention functions in parallel, each in its own learned subspace, and combine them. Project the queries, keys, and values down to a smaller dimension with `h` different learned projections — head `i` gets its own `W_i^Q, W_i^K, W_i^V` — run the scaled-dot-product attention independently in each, and then I have `h` separate weighted averages, each free to focus on a different relation:

  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).

To turn the `h` outputs back into a single `d`-dimensional vector per position, concatenate them and apply one more learned projection `W^O`:

  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O.

Now the question I should always ask when I add capacity: what does this cost me? If each head used the full width `d_model`, then `h` heads would be `h` times the compute, and I'd be paying for the expressivity. But I get to choose the per-head width. Let me set each head to `d_k = d_v = d_model / h`. Then count the operations. The projections: each head's `W_i^Q` is `d_model × (d_model/h)`, and there are `h` of them for the queries, so the query projections together are `h · d_model · (d_model/h) = d_model^2` — same as a single full-width projection. Same for keys and values, and the output projection `W^O` is `h d_v × d_model = d_model × d_model`, also `d_model^2`. The attention scores: each head forms an `n × n` matrix from `d_k`-dimensional dots, `O(n^2 · d_k) = O(n^2 · d_model/h)`, and over `h` heads that's `h · n^2 · d_model/h = n^2 · d_model` — exactly the cost of a single full-width head. So splitting into `h` lower-dimensional heads costs essentially the same as one full-dimensional attention, while letting the model attend to `h` different relations at once. I get separate learned subspaces without changing the leading-order score cost. I'll take `h = 8` heads at `d_model = 512`, so `d_k = d_v = 64` — enough heads to capture several distinct relations, each still wide enough (64) to hold a useful subspace.

Let me look hard at one more thing about this layer before I build the block around it: what kind of function is it, really? Given the attention weights, the output `sum_j alpha_{ij} v_j` is *linear* in the values. The only nonlinearity is the softmax that produces the weights — but that nonlinearity acts on the *mixing coefficients*, not on the content being mixed. So an attention layer is, in effect, a clever data-dependent linear combination across positions. Stacking attention on attention gives me richer and richer mixing, but I never apply a genuine per-position nonlinear transform to the gathered content. That's a hole: I want each position, after it has gathered information from the others, to *process* what it gathered. So after the attention I'll add a small feed-forward network applied to each position separately and identically — the same weights at every position, but a real nonlinearity in the middle:

  FFN(x) = max(0, x W_1 + b_1) W_2 + b_2,

a linear map up to a wider inner dimension, a ReLU, a linear map back down. Two `1×1` convolutions, if you like. How wide should the inner layer be? It's the only place in the block doing per-position nonlinear computation, so it wants enough capacity to be worth having; a four-times expansion, inner dimension `d_ff = 4 · d_model = 2048`, is the balanced point I want, then I contract back to the model width. Narrower and the block is starved of nonlinear capacity; much wider and I'm paying a lot for the per-position map relative to the mixing.

So a block is two sub-layers: gather across positions with multi-head self-attention, then transform per position with the FFN. But I intend to stack a lot of these, and deep stacks have their own failure mode — gradients attenuating as they pass back through many layers, the plain-deep-net degradation problem. The known cure is residual connections: wrap each sub-layer so an identity path runs around it, `y = x + Sublayer(x)`. Then even if a sub-layer's Jacobian is small, the gradient still reaches the layers below through the identity, and the sub-layer only has to learn a *correction* to its input rather than re-encode everything from scratch. So each sub-layer becomes `x + Sublayer(x)`. There's also a happy fit with the FFN-only-nonlinearity worry: the residual means the attention sub-layer is adding gathered information *to* the existing per-position representation rather than replacing it, so nothing is lost when a position doesn't need much from its neighbors.

Residuals alone aren't quite enough for a deep stack, though — the activations passing along the residual stream can drift in scale layer to layer, and that drift is what makes deep training finicky. I want to normalize the activations. Batch normalization is the reflex, but it computes its statistics across the examples in a batch, and that's exactly wrong for me: my "batch" is often a single variable-length sequence, or a small variable-size group, and at batch size one batch-norm has no statistics to compute. I need a normalization that works per example, independent of the batch. That's layer normalization: for a vector of `H` features at one position, compute `mu = (1/H) sum_c a_c` and `sigma = sqrt((1/H) sum_c (a_c - mu)^2)` *over the features of that one position*, then `(g/sigma)(a - mu) + b` with a learned gain and bias. The statistics depend only on that single position's own feature vector, so there's no cross-example coupling at all — it works unchanged at batch size one and for any sequence length. Perfect for a layer that has to operate on variable-length groups.

Now where exactly does the norm go relative to the residual? I have a residual `x + Sublayer(x)` and a normalization, and the order matters. The thing whose scale I most want pinned is the *output* of each sub-layer as it gets handed to the next sub-layer — I want each block to receive a well-conditioned, fixed-scale input regardless of how many blocks came before. So I normalize *after* adding the residual: the output of each sub-layer is

  LayerNorm( x + Sublayer(x) ).

That way the quantity flowing from one block to the next is always normalized, and every sub-layer downstream sees inputs at a controlled scale no matter the depth. So the encoder block is, concretely:

  a = LayerNorm( x + MultiHeadSelfAttention(x) )      # gather across positions, then normalize
  z = LayerNorm( a + FFN(a) )                          # transform per position, then normalize

with self-attention meaning the queries, keys, and values are all projections of the *same* input `x` — every position attending to every position of its own sequence. Stack `N` of these (six is a reasonable depth) and that's an encoder built entirely out of attention and per-position feed-forward maps, no recurrence and no convolution anywhere. Sequential depth `O(1)` per block, path length between any two positions `O(1)`, every operation a big parallel matmul.

Let me also sanity-check the claim that I haven't lost something the convolution or recurrence had. The one thing self-attention gives up is that a single weighted average over many positions can blur fine positional detail — averaging over the whole sequence reduces the effective resolution. But that's precisely what multi-head fixes: with several heads, each can keep a sharp focus on a different small set of positions instead of one head having to average everything, so the multi-head structure I already added for expressivity also recovers the resolution that pure averaging would cost. The two motivations point at the same mechanism, which is a good sign I'm not bolting on parts.

One more property worth pinning down, since it tells me my projections are doing real work and aren't redundant. Suppose I dropped the separate value projection and just averaged the raw inputs, `sum_j alpha_{ij} x_j`. Then the layer could only ever output convex combinations of its inputs — it could route information around but never re-represent it before passing it on. The learned `W^V` lets each position offer a *transformed* view of itself as its value, and `W^O` re-mixes the concatenated heads back into the model dimension; together they make the gathered information something the next layer can build on, not just a reshuffle. So `W^Q, W^K` decide *who talks to whom*, and `W^V, W^O` decide *what gets said and how it's repackaged* — four distinct jobs, which is why the block has four projections per attention sub-layer.

Now let me write the thing as code I'd actually run, grounding each piece in what a clean implementation looks like, filling the empty mixing slot the harness left me. First the scaled dot-product attention as a function, then multi-head as a module that does the projections, runs the attention per head, and re-projects, then the block that wraps each sub-layer in `LayerNorm(x + Sublayer(x))`:

```python
import math
import torch
import torch.nn as nn


def attention(query, key, value, mask=None, dropout=None):
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d_k)) V.
    query/key/value: (..., n, d_k). Returns the gathered values and the weights."""
    d_k = query.size(-1)
    # logits with variance ~ d_k before scaling; / sqrt(d_k) puts them back at unit scale
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)      # custom mask: 1 keeps, 0 blocks
    p_attn = scores.softmax(dim=-1)                        # weights sum to 1 over keys
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn            # weighted average of values


class MultiHeadAttention(nn.Module):
    """h parallel attention heads in d_model/h-dim subspaces, concatenated and re-projected.
    W^Q, W^K, W^V choose who-talks-to-whom; W^O repackages the gathered information."""

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h                            # per-head width keeps total cost = single head
        self.h = h
        # 4 projections: Q, K, V in-projections + the output projection W^O
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)                       # same mask broadcast over heads
        nbatches = query.size(0)
        # project, then split d_model into (h heads x d_k) and move heads to the batch axis
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # scaled dot-product attention in parallel across all heads
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        # concat heads back to d_model, then the output projection
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """Normalize across the feature dimension of each position; batch-size independent."""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))     # gain g
        self.b_2 = nn.Parameter(torch.zeros(features))    # bias b
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)                    # mu over the d features of this position
        var = ((x - mean) ** 2).mean(-1, keepdim=True)      # (1/H) sum_c (a_c - mu)^2
        sigma = torch.sqrt(var + self.eps)
        return self.a_2 * (x - mean) / sigma + self.b_2


class PositionwiseFeedForward(nn.Module):
    """Per-position nonlinearity: max(0, x W_1 + b_1) W_2 + b_2, inner width 4 x d_model."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class EncoderLayer(nn.Module):
    """One block: LayerNorm(x + self-attn(x)) then LayerNorm(x + FFN(x)) (post-norm)."""

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # gather across positions (self-attention: Q=K=V=x), residual, then normalize
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        # transform per position, residual, then normalize
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    """Stack of N identical attention blocks; no recurrence, no convolution."""

    def __init__(self, input_size, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )

    def forward(self, x, mask=None):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, mask)
        return h
```

Let me trace the causal chain one more time to be sure it closes. I started stuck because recurrence makes position-to-position communication `O(n)` sequential and gives a path length that grows with distance, and convolution only fixes the first while leaving the path length growing with the sequence. Attention already connects any two positions in one content-based step, but it had always been bolted onto a recurrent encoder, so I asked what happens if I let attention do all the communication and turn it inward — each position attending to every position of its own sequence. Writing that out gave `softmax(Q K^T) V` with learned query/key/value projections, cheaper than recurrence whenever the sequence is shorter than the width and fully parallel. Computing the variance of the dot products showed the logits grow like `sqrt(d_k)` and saturate the softmax, so I divided by `sqrt(d_k)` to hold them at unit scale. A single softmax can only express one relation and its averaging blurs several, so I ran `h` heads in parallel in `d_model/h`-dimensional subspaces — which the operation count shows costs the same as one full head — and concatenated and re-projected them; the same multi-head structure also restores the resolution that a single averaged head would lose. Because attention is linear in the values, I added a per-position feed-forward network with a four-times-wider ReLU hidden layer to give each position real nonlinear processing of what it gathered. To stack many blocks I wrapped each sub-layer in a residual so gradients flow through the identity, and normalized after the residual with layer normalization — chosen over batch normalization precisely because its statistics are per-position and so batch-size independent, which is what a variable-length group needs. The block is two sub-layers, `LayerNorm(x + SelfAttention(x))` then `LayerNorm(x + FFN(x))`, and an encoder is a stack of these — every operation a big parallel matmul, `O(1)` sequential depth, `O(1)` path between any two positions.
