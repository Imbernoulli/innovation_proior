OK, let me think this through from scratch. I have a Transformer that works — stack of attention-plus-feed-forward layers, residuals, the whole thing — and it's good. The problem is purely operational: I want to train it on really long sequences, $L=64\text{K}$ tokens, deep, and it won't fit on one accelerator. Not because the model is huge in parameters. Let me actually do the arithmetic, because the arithmetic is where the surprise is. The biggest attention layer people report is about $0.5\text{B}$ parameters, which in floats is $2\text{GB}$. The input activations for one such layer at $L=64\text{K}$, width $1024$, batch $8$ are $64\text{K}\times 1\text{K}\times 8 = 0.5\text{B}$ floats, another $2\text{GB}$. So per layer I'm at $4\text{GB}$, which fits fine. The whole BERT corpus is $17\text{GB}$ on disk. By this estimate a big model on a $64\text{K}$ sequence should sit on one device with room to spare. And yet I can't even fine-tune these things on a single machine. So my per-layer estimate is missing the real costs, and I need to find them before I can kill them.

Where does the memory actually go? Three places, and they're multiplicative, which is why the naive estimate is so wrong. First, backprop needs the forward activations of *every* layer to form that layer's gradient, so a depth-$N$ model stores roughly $N$ copies of those per-layer activations, not one. That's a factor of $n_l$ I completely ignored. Second, inside each layer the feed-forward sublayer projects up to an intermediate width $d_{ff}$ that's usually a few times $d_{model}$ — say $d_{ff}=4096$ against $d_{model}=1024$ — so the intermediate tensor is $[b,L,d_{ff}]$, four times fatter than the attention activations I costed. At $d_{ff}=4\text{K}$, $n_l=16$, $L=64\text{K}$ that alone is about $16\text{GB}$. Third, attention forms $QK^\top$, shape $[b,L,L]$; at $L=64\text{K}$ even one head at batch $1$ in fp32 is a $64\text{K}\times 64\text{K}$ matrix $=16\text{GB}$, and it's $O(L^2)$ in both time and memory. So three independent blow-ups: an $n_l$ multiplier from stored activations, a $d_{ff}$ multiplier from the wide FFN, and an $L^2$ term from attention. Each one on its own can exhaust the device. I'll have to take on all three; fixing only one leaves the model still un-trainable on a single machine. Let me take them one at a time, and the order doesn't matter much — start with the one that scares me most, the $L^2$ attention.

So, attention. The output is $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$, and the expensive object is $QK^\top$, every query dotted with every key. Do I actually need all of it? Stare at the softmax. For a single query $q_i$ the attention weights are $\mathrm{softmax}$ over $\{q_i\cdot k_j\}_j$, and a softmax is dominated by its largest arguments — exponentially so. If the top few dot products are even modestly larger than the rest, almost all the weight lands on them and the rest contribute essentially nothing to the output. So for each query I don't need all $L$ keys; I need the handful of keys whose dot product with $q_i$ is largest. If $K$ has length $64\text{K}$, maybe I only need the $32$ or $64$ closest keys for each query. That would turn the per-query cost from $L$ down to a constant, and the whole thing from $O(L^2)$ toward $O(L)$. The catch is in the word "closest": I'd need, for every query, to find its nearest keys among $64\text{K}$ candidates, *fast*, without first computing all the dot products — because computing all the dot products to find the big ones is exactly the $O(L^2)$ I'm trying to avoid.

Nearest neighbors in high dimension, quickly, approximately — that's locality-sensitive hashing. The idea: a hash $h(x)$ is locality-sensitive if vectors that are close get the same hash with high probability and vectors that are far apart usually don't. If I can bucket every vector by such a hash, then "near neighbors of $q_i$" becomes "things in $q_i$'s bucket," which is cheap to gather. The dot product $q_i\cdot k_j$ relates to the angle between them (and the norms), so what I want is an LSH that's sensitive to *angular* distance. There's a clean known scheme for exactly that. Take a random matrix $R$ of shape $[d_k, b/2]$ — $b/2$ random directions. Rotate $x$ by it, $xR$, and form the $b$-vector $[xR;\,-xR]$ by concatenating with its negation. Then set
$$h(x)=\arg\max\big([xR;\,-xR]\big),$$
the index of the largest of those $b$ signed projected coordinates. Geometrically: project $x$ onto a randomly oriented set of axes and assign it to whichever signed axis it points most along. Two vectors that point in nearly the same direction will, with high probability, have their largest signed projection on the same axis — they collide. Two vectors far apart in angle usually pick different axes. That's $b$ buckets out of $b/2$ random directions, it's a few matrix multiplies, it batches trivially over all vectors at once, and it has no learned parameters. Good — this is my near-neighbor finder. I need $b$, the number of buckets, to be even, which it is by construction.

Now there's a wrinkle I have to face before I can use this in attention. I want to bucket *queries* and *keys* together so that a query lands in the same bucket as its near keys. But in a Transformer, $Q$ and $K$ are *different* linear projections of the input — there's no reason $h(q_i)$ relates to $h(k_j)$ at all, even when $q_i$ and $k_j$ would have a big dot product, because they live in differently-transformed spaces. So the hashing notion of "close" and the attention notion of "close" come apart. The fix is uncomfortable but clean: make the queries and the keys the *same*. Use one linear layer to produce a tensor that serves as both $Q$ and $K$ (shared-QK), and a separate layer for $V$. Now $q_j$ and $k_j$ are literally the same vector, so $h(q_i)$ and $h(k_j)$ are computed in the same space and "same bucket" means "similar direction" for both. I worry this hobbles the model — surely $Q$ and $K$ being free to differ is doing something. But I can test it directly on a normal Transformer: tie $Q=K$ and see if quality drops. It doesn't, even if I additionally normalize the keys to unit length. So sharing QK costs nothing measurable, and it's what makes hashing-based attention coherent. I'll take it.

Let me now rewrite attention so I can splice in the bucketing cleanly. Forget the matrix form; write it per query, single position $i$:
$$o_i=\sum_{j\in\mathcal{P}_i}\exp\!\big(q_i\cdot k_j-z(i,\mathcal{P}_i)\big)\,v_j,\qquad \mathcal{P}_i=\{j:i\ge j\},$$
where $\mathcal{P}_i$ is the set of positions $i$ is allowed to attend to (here, causal: everything up to $i$), and $z(i,\mathcal{P}_i)$ is the log-partition $\log\sum_{j\in\mathcal{P}_i}\exp(q_i\cdot k_j)$ — the normalizer pulled out as a subtraction so the exponentials are the softmax weights. (I'm dropping the $1/\sqrt{d_k}$ for clarity; it folds into the scale of $q$.) In practice I won't sum over the exact ragged set $\mathcal{P}_i$; I'll sum over some larger, batch-friendly superset $\widetilde{\mathcal{P}}_i\supseteq\mathcal{P}_i$ and mask out the elements that shouldn't be there:
$$o_i=\sum_{j\in\widetilde{\mathcal{P}}_i}\exp\!\big(q_i\cdot k_j-m(j,\mathcal{P}_i)-z(i,\mathcal{P}_i)\big)\,v_j,\qquad m(j,\mathcal{P}_i)=\begin{cases}\infty & j\notin\mathcal{P}_i\\ 0 & \text{otherwise}.\end{cases}$$
The mask $m=\infty$ sends $\exp(-\infty)=0$, so masked positions drop out. This is just full attention written so the "which positions do I attend to" set is explicit and swappable. Now I swap it: LSH attention is exactly this with the allowed set restricted to "same hash bucket,"
$$\mathcal{P}_i=\{j:h(q_i)=h(k_j)\}.$$
Since the full attention matrix is in practice nearly sparse (a query genuinely attends to few keys), and similar items share a bucket with high probability, attending only within buckets should approximate full attention well.

But now I hit the messiness of actually computing this on a GPU. Buckets are uneven in size — some hash buckets catch many vectors, some few. Worse, because I'm restricting to "$h(q_i)=h(k_j)$" with $q$ and $k$ as separate roles, a bucket can end up with queries but no keys, or vice versa, and then a query has nothing to attend to. The shared-QK choice already half-solves this: if I set $k_j=q_j/\lVert q_j\rVert$ — the key is just the query direction, unit-normalized — then $h(k_j)=h(q_j)$ automatically (the hash only sees direction, and normalizing doesn't change direction), so every position's query and its own key land in the same bucket and no bucket is query-only or key-only. (Why normalize the keys but not the queries? The query's norm is free to act as a per-query temperature on the softmax — a learnable sharpness — so I leave $q$ unnormalized; but the *key* norm would corrupt the correspondence between "hash-close" and "attention-close," so I strip it. That asymmetry is deliberate.)

Even with nonempty buckets I can't batch a ragged collection of variable-size buckets into a matmul. The trick is to *sort*. Sort all positions by their bucket number, and within a bucket by sequence position. That defines a permutation $i\mapsto s_i$. After sorting, positions in the same bucket are contiguous, so in the sorted attention matrix the within-bucket pairs cluster right around the diagonal. Now I can cut the sorted sequence into fixed-size chunks of $m$ consecutive positions and let each chunk attend within itself — fixed-size blocks, a clean batched matmul. The only thing that breaks is when a bucket straddles a chunk boundary: a vector at the start of a chunk might have its bucket-mates at the end of the previous chunk. So I let each chunk also attend to the *one chunk immediately before it*. With "self chunk plus one back," the allowed set becomes, in sorted indices,
$$\widetilde{\mathcal{P}}_i=\Big\{\,j:\Big\lfloor\tfrac{s_i}{m}\Big\rfloor-1\le\Big\lfloor\tfrac{s_j}{m}\Big\rfloor\le\Big\lfloor\tfrac{s_i}{m}\Big\rfloor\Big\}.$$
Read it off: $\lfloor s_j/m\rfloor$ is the chunk index of $j$; it must equal $i$'s chunk or be one less. When is this safe — when does chunking lose nothing relative to true bucket attention? Exactly when no bucket is bigger than a chunk: if $\max_i|\mathcal{P}_i|<m$ then a bucket fits inside "this chunk plus one back" and $\mathcal{P}_i\subseteq\widetilde{\mathcal{P}}_i$. So I want chunk size $m$ to exceed the typical bucket size with margin. The average bucket holds $l/n_{buckets}$ items; I set $m=2l/n_{buckets}$, twice the average, and bet that a bucket rarely doubles past its mean. With $n_{buckets}\propto l$ the chunk size is a constant and the number of chunks scales with $l$, so the attention cost is $l\times(\text{const})$ rather than $l^2$ — the $O(L^2)$ is gone, replaced by something near $O(L\log L)$ once you count the sort. That's the whole point.

Now, two ways this can go wrong that I have to patch. The first is hash misses. LSH is probabilistic: two genuinely close vectors *can* fall in different buckets on any given random rotation, and if they do, that key is invisible to that query and I lose a real attention edge. One rotation is a coin flip I don't fully trust. Do it more than once. Take $n_{rounds}$ *independent* hash functions $\{h^{(1)},h^{(2)},\dots\}$, each with its own random rotation, and let the allowed set be the *union* of the per-round buckets:
$$\mathcal{P}_i=\bigcup_{r=1}^{n_{rounds}}\mathcal{P}_i^{(r)},\qquad \mathcal{P}_i^{(r)}=\{j:h^{(r)}(q_i)=h^{(r)}(q_j)\}.$$
A pair is missed only if it's separated in *every* round; with independent rounds that probability falls off fast, so a handful of rounds recovers almost all the real neighbors. Multi-round is just running the single-round LSH attention $n_{rounds}$ times in parallel — but combining the rounds correctly is fiddly, and I'll come back to it once the single-round version is nailed down.

The second is masking, and there are two distinct masks. Causal first: in a decoder, position $i$ must not see the future, $j>i$. But I've *sorted* the sequence by bucket, so "the future" is scrambled — sorted-adjacent positions can be anywhere in the original order. The clean fix: carry each position's original index along as a payload, permute those indices with the *same* sort I applied to the vectors, and inside a chunk compare the original indices to build the causal mask. So I never need the original layout back during the attention; I just need the original index of each sorted slot, and a comparison `orig_index_query >= orig_index_key`.

The second mask is subtler and is forced on me by the shared-QK choice. Normally a Transformer lets a position attend to itself. But here $k_i=q_i/\lVert q_i\rVert$, so $q_i\cdot k_i=\lVert q_i\rVert$, which is almost always the *largest* dot product in the bucket — a query's self-similarity beats its similarity to anything else. If I allow self-attention, every token just attends to itself and the layer does nothing. So I must forbid a token from attending to its own position. But not absolutely: a token might have *no other* valid target — the very first position in a causal sequence has nothing to its left — and then it has to attend to itself or it attends to nothing and the softmax is degenerate. So the self-mask can't be $-\infty$; it has to be a large but *finite* penalty, big enough to lose to any genuine competitor but still selectable when it's the only option. I'll set the self-logit to a large negative constant like $-5\times10^4$: dominated whenever another key exists, but finite so it survives a softmax that has no alternative (and small enough in magnitude to stay safe in half precision).

Let me now make the multi-round combination exact, because the union has a double-counting trap. I run round $r$ and get, for query $i$, a local attention output $o_i^{(r)}$ computed over that round's chunked set with that round's own partition function $z(i,\mathcal{P}_i^{(r)})$. I want the *global* output $o_i$, which is the single softmax over the union $\mathcal{P}_i$. The problem: if a key $j$ lands in $i$'s bucket in several rounds, it appears in several $\mathcal{P}_i^{(r)}$, and naively summing the per-round outputs counts it multiple times. So I need to (a) reweight each round's local softmax to the global partition, and (b) divide out the multiplicity of each pair. Let
$$N_{i,j}=\big|\{r':j\in\mathcal{P}_i^{(r')}\}\big|$$
be the number of rounds in which $j$ is a neighbor of $i$. Start from the global single-softmax form and split the sum over the union into rounds:
$$o_i=\sum_{j\in\widetilde{\mathcal{P}}_i}\exp\!\big(q_i\cdot k_j-m(j,\mathcal{P}_i)-z(i,\mathcal{P}_i)\big)\,v_j.$$
A key $j$ that appears in $N_{i,j}$ rounds is, if I sum round by round, going to be added $N_{i,j}$ times; to make the total come out as if it were added once, I scale each appearance by $1/N_{i,j}$. And each round's exponentials are normalized by that round's partition $z(i,\mathcal{P}_i^{(r)})$, not the global $z(i,\mathcal{P}_i)$, so I correct with a factor $\exp\!\big(z(i,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i)\big)$ that re-bases each round's local softmax onto the global denominator. Putting both together:
$$o_i=\sum_{r=1}^{n_{rounds}}\exp\!\big(z(i,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i)\big)\sum_{j\in\widetilde{\mathcal{P}}_i^{(r)}}\frac{1}{N_{i,j}}\exp\!\big(q_i\cdot k_j-m(j,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i^{(r)})\big)\,v_j.$$
The inner sum, absent the $1/N_{i,j}$, is exactly the round-$r$ local output $o_i^{(r)}$, so this is $o_i=\sum_r \exp\!\big(z(i,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i)\big)\,o_i^{(r)}$ once the dedup is folded in. And the cleanest place to fold $1/N_{i,j}=\exp(-\log N_{i,j})$ is straight into the per-round mask, since the mask already lives in the exponent. So I let the round-$r$ mask carry three cases:
$$m^{(r)}_{i,j}=\begin{cases}\infty & j\notin\mathcal{P}_i^{(r)}\\[2pt] 5\times10^4 & i=j\\[2pt] \log N_{i,j} & \text{otherwise}.\end{cases}$$
First case kills out-of-bucket pairs as before. Second case is the finite self-penalty I derived. Third case is the dedup: subtracting $\log N_{i,j}$ in the exponent multiplies that pair's weight by $1/N_{i,j}$, so a pair seen $N$ times across rounds contributes its full weight exactly once in total. The outer $\exp(z^{(r)}-z)$ then just sums the rebased rounds. Concretely in code this means: per round, compute the local output and its log-partition (a logsumexp over that round's chunk), then combine the rounds by a softmax over the per-round log-partitions and a weighted sum of the per-round outputs — which is precisely $\sum_r \exp(z^{(r)}-z)\,o^{(r)}$ with $z=\log\sum_r\exp(z^{(r)})$. The two views agree.

That's attention handled: shared-QK, angular-LSH bucketing, sort-and-chunk with one-back, multi-round union with the $\log N$ dedup, causal mask via permuted indices, finite self-mask. Memory and time go from $O(L^2)$ to roughly $O(L\log L)$. But the $L^2$ term was only one of my three blow-ups, and look at the cost table even after this fix: every entry still carries a leading $b\cdot n_h\cdot l\cdot d_k$ (equivalently $b\cdot l\cdot d_{model}$) factor — the activations entering each layer are already $b\cdot l\cdot d_{model}$, unavoidable. The killer is the *multiplier* on it. The whole model's activation memory is at least $b\cdot l\cdot d_{model}\cdot n_l$ because I store activations for every one of the $n_l$ layers for backprop, and inside the feed-forward sublayers it's even worse, $b\cdot l\cdot d_{ff}\cdot n_l$. At $d_{ff}=4\text{K}$, $n_l=16$, $l=64\text{K}$ that's $16\text{GB}$ again. So I have to kill the $n_l$ multiplier and the $d_{ff}$ multiplier. Two separate moves.

The $n_l$ multiplier exists for one reason: backprop needs every layer's forward activations, so I cache all $n_l$ of them. What if I didn't have to cache them — what if I could *recompute* a layer's input from its output during the backward pass, using only the layer's own functions? Then I'd store the activations once (the output of the stack), and reconstruct each layer's activations on demand as the gradient flows back, paying recompute instead of storage. A normal residual layer $y=x+F(x)$ can't do this — given $y$ I can't recover $x$ without knowing $F(x)$, which needs $x$. But reversible residual layers can. Split the activation into two halves $x_1,x_2$ and interleave two residual functions $F$ and $G$:
$$y_1=x_1+F(x_2),\qquad y_2=x_2+G(y_1).$$
This is exactly invertible. Given the outputs $(y_1,y_2)$, recover the inputs by undoing the additions in reverse order:
$$x_2=y_2-G(y_1),\qquad x_1=y_1-F(x_2).$$
Check: $y_1$ is known, so $G(y_1)$ is computable, so $x_2=y_2-G(y_1)$; now $x_2$ is known so $F(x_2)$ is computable, so $x_1=y_1-F(x_2)$. No stored activations needed — the layer regenerates its own inputs from its outputs. So during backprop I march from the network's output back toward its input, and at each layer I invert to get that layer's input activations, then do the local gradient computation, then discard. Activation memory becomes independent of $n_l$. The $n_l$ multiplier is gone.

Mapping this onto the Transformer is natural because a Transformer layer already has exactly two sublayers: let $F$ be the attention sublayer and $G$ be the feed-forward sublayer.
$$Y_1=X_1+\mathrm{Attention}(X_2),\qquad Y_2=X_2+\mathrm{FeedForward}(Y_1).$$
For this to match the normal Transformer's parameter count, I make $X_1$ and $X_2$ each the full width $d_{model}$ — so the reversible block runs on a doubled signal of width $2d_{model}$, with each half feeding one sublayer. (Normalization moves *inside* the residual functions — pre-norm inside $F$ and $G$ — so the residual additions stay clean linear inverses; that's required for the subtraction to undo exactly what the addition did.) One subtlety for correctness of recompute: any randomness in $F$ or $G$ (dropout) must produce the *same* mask on the recompute as on the forward, or the inversion is wrong. So I save the RNG state on the forward pass and restore it before recomputing on the backward pass — then $F(x_2)$ and $G(y_1)$ recompute identically and the algebra holds exactly. I expect this to barely move training, since it computes the same function as a normal residual stack; only the bookkeeping differs. Test it: a reversible Transformer with the same parameter count trains with learning curves essentially identical to the normal one. Good — the memory saving is free.

Now the last multiplier, $d_{ff}$. Even with reversibility removing the depth factor, a single feed-forward sublayer still materializes a $[b,l,d_{ff}]$ intermediate, four-plus times fatter than the layer activations. But look at what the FFN *is*: a position-wise map, applied to each sequence position completely independently of the others. There's no mixing across positions inside the FFN — that's what attention is for. So I never need all $l$ positions' intermediates in memory at once. Split the sequence into $c$ chunks of positions and run the FFN one chunk at a time, concatenating the outputs:
$$Y_2=\big[\,Y_2^{(1)};\dots;Y_2^{(c)}\,\big]=\big[\,X_2^{(1)}+\mathrm{FeedForward}(Y_1^{(1)});\dots;X_2^{(c)}+\mathrm{FeedForward}(Y_1^{(c)})\,\big].$$
This is numerically identical to running the FFN on the whole sequence — chunking changes nothing about the computed function, only the peak memory, since at any instant I hold one chunk's $d_{ff}$-wide intermediate instead of the whole sequence's. The reverse computation and the backward pass get chunked the same way. The $d_{ff}$ multiplier on peak memory drops to $d_{ff}/c$. And the same logic applies at the very output: for a large vocabulary the log-probabilities over the sequence are another big tensor, position-independent, so I chunk the loss computation over the sequence too.

Tally the three fixes against the three blow-ups. Reversible layers remove the $n_l$ activation multiplier — the whole stack stores activations once. Chunking removes the $d_{ff}$ multiplier on peak FFN memory. LSH attention removes the $O(l^2)$ attention term, taking it to $\sim O(l\log l)$. Per layer the activation cost is back down near the irreducible $b\cdot l\cdot d_{model}$ — the very number my naive estimate started from — but now without the $n_l$, $d_{ff}$, and $l^2$ multipliers that made the real model un-trainable. One residual cost remains: the *parameters* still grow with depth (reversibility only fixed activations, not weights). But since chunking and reversibility have made the activation footprint tiny, and the batch-times-length here is enormous, the compute per layer dwarfs the cost of streaming that layer's parameters in from CPU memory — so I can swap parameters to and from the host when a layer isn't active and amortize the transfer. With that, a deep model on a $64\text{K}$ sequence fits on a single accelerator.

Let me write the core code, grounded in the structure above. First the LSH attention — hashing, the sort, the chunking with one-back, the masks, and the multi-round combine.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_SELF_ATTN_VALUE = -5e4  # large but FINITE: self-attn loses to any real key,
                              # yet survives when a token has no other target

def sort_key_val(t1, t2, dim=-1):
    # sort t1, reorder t2 by the same permutation
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

class LSHAttention(nn.Module):
    def __init__(self, bucket_size=64, n_hashes=8, causal=False, dropout=0.):
        super().__init__()
        self.bucket_size = bucket_size      # chunk size m after sorting
        self.n_hashes = n_hashes            # number of independent rounds
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def hash_vectors(self, n_buckets, vecs):
        # angular LSH: h(x) = argmax([xR ; -xR]) over n_buckets signed axes.
        # a fresh random rotation R per round to make rounds independent.
        batch_size, device = vecs.shape[0], vecs.device
        assert n_buckets % 2 == 0
        rotations_shape = (1, vecs.shape[-1], self.n_hashes, n_buckets // 2)
        R = torch.randn(rotations_shape, device=device).expand(batch_size, -1, -1, -1)
        rotated = torch.einsum('btf,bfhi->bhti', vecs, R)        # [b, hashes, t, n_buckets/2]
        rotated = torch.cat([rotated, -rotated], dim=-1)          # [.. , n_buckets]
        buckets = torch.argmax(rotated, dim=-1)                   # bucket per (round, position)
        # offset per round so bucket ids from different rounds don't collide
        offsets = (torch.arange(self.n_hashes, device=device) * n_buckets).reshape(1, -1, 1)
        return torch.reshape(buckets + offsets, (batch_size, -1))

    def forward(self, qk, v, query_len=None, input_mask=None):
        batch_size, seqlen, dim, device = *qk.shape, qk.device
        query_len = query_len or seqlen
        assert seqlen % (self.bucket_size * 2) == 0
        n_buckets = seqlen // self.bucket_size
        total_hashes = self.n_hashes

        buckets = self.hash_vectors(n_buckets, qk)                # [b, n_hashes * seqlen]

        # sort by (bucket, then original position): pack both into one key.
        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = (seqlen * buckets + (ticker % seqlen)).detach()
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)  # the permutation i -> s_i
        _, undo_sort = sticker.sort(dim=-1)                       # to unsort outputs later

        st = sticker % seqlen                                     # original index of each sorted slot
        sqk = batched_index_select(qk, st)                        # gather q/k and v into sorted order
        sv  = batched_index_select(v,  st)

        # split into chunks of bucket_size: chunk_size = n_hashes * n_buckets chunks
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv  = torch.reshape(sv,  (batch_size, chunk_size, -1, dim))

        # shared-QK: query unnormalized (norm = learnable temperature), key unit-normalized
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # each chunk attends to itself AND one chunk back (buckets may straddle a boundary)
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)
        bk    = look_one_back(bk)
        bv    = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # within-chunk dot products
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = -torch.finfo(dots.dtype).max

        if self.causal:
            # causal mask via the carried ORIGINAL indices (sorting scrambled position order)
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            dots.masked_fill_(mask, masked_value)

        # forbid attending to self, except finite so it survives when it's the only option
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)

        # local softmax per round, keeping the log-partition z^{(r)} for cross-round combine
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)   # z(i, P_i^{(r)})
        dots = torch.exp(dots - dots_logsumexp)
        dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dots, bv)                 # round-r local output o_i^{(r)}
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # undo the sort: back to original positions, separated per round
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)
        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        # combine rounds: o_i = sum_r exp(z^{(r)} - z) o^{(r)},  z = logsumexp_r z^{(r)}
        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)
        return out
```

That code carries every step: `hash_vectors` is the angular-LSH $\arg\max([xR;-xR])$ with a fresh rotation per round and per-round offsets so union-by-bucket is well defined; the `buckets_and_t`/`sort_key_val` pair sorts by bucket then position and remembers `undo_sort`; `bk = F.normalize(bqk)` with `bq = bqk` is the shared-QK with normalized keys and the query norm left as temperature; `look_one_back` is the self-chunk-plus-one-back set $\widetilde{\mathcal{P}}_i$; the causal and self masks use the carried original indices `bq_t`/`bkv_t`, the self mask finite per the no-other-target argument; and the final `probs`/`out` is exactly $\sum_r\exp(z^{(r)}-z)\,o^{(r)}$, the multi-round combine I derived (with the $\log N_{i,j}$ dedup folding into the same logsumexp accounting when a pair recurs).

Now the reversible stack, which stores activations once and recomputes per layer on the backward pass.

```python
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

class Deterministic(nn.Module):
    # wraps a sublayer so its randomness (dropout) recomputes identically on backward:
    # save RNG state on forward, restore it before the recompute.
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None; self.cuda_in_fwd = None
        self.gpu_devices = None; self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = self.gpu_devices if self.cuda_in_fwd else []
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

class ReversibleBlock(nn.Module):
    # F = attention, G = feed-forward.  y1 = x1 + F(x2); y2 = x2 + G(y1).
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        with torch.no_grad():                       # forward stores no graph
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        dy1, dy2 = torch.chunk(dy, 2, dim=2)

        # recover x2 = y2 - G(y1), backprop through G
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            dx1 = dy1 + y1.grad
            y1.grad = None

        # recover x1 = y1 - F(x2), backprop through F
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            dx2 = dy2 + x2.grad
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()                          # store activations ONCE, at the top
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y, kwargs = ctx.y, ctx.kwargs
        for block in ctx.blocks[::-1]:              # invert layer-by-layer from the output down
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None
```

`backward_pass` is the inversion read straight off the equations: `x2 = y2 - gy1` is $x_2=y_2-G(y_1)$ and `x1 = y1 - fx2` is $x_1=y_1-F(x_2)$, with the gradients accumulated through the recomputed $G$ and $F$. `_ReversibleFunction` stashes only the final output (`ctx.y`) — that's the "store activations once" — and reconstructs every layer's activations on the way back. `Deterministic` is what makes the recompute exact under dropout.

Finally the model: feed-forward chunked over positions, attention with shared-QK, both wrapped as the $F,G$ of reversible blocks, and the doubled signal $\mathrm{cat}([x,x])$ that supplies $X_1,X_2$.

```python
class Chunk(nn.Module):
    # run a position-wise sublayer on chunks of the sequence to cap peak memory
    def __init__(self, chunks, fn, along_dim=-2):
        super().__init__()
        self.chunks, self.fn, self.dim = chunks, fn, along_dim
    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        cs = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in cs], dim=self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult); self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout); self.w2 = nn.Linear(dim * mult, dim)
    def forward(self, x, **kwargs):
        return self.w2(self.dropout(self.act(self.w1(x))))

class PreNorm(nn.Module):
    # LayerNorm moved INSIDE the residual function (keeps the residual add a clean inverse)
    def __init__(self, dim, fn):
        super().__init__(); self.norm = nn.LayerNorm(dim); self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=8, causal=False):
        super().__init__()
        self.heads = heads
        self.toqk = nn.Linear(dim, dim, bias=False)   # ONE projection for both Q and K
        self.tov  = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.lsh = LSHAttention(bucket_size, n_hashes, causal)
    def forward(self, x, **kwargs):
        b, t, e, h = *x.shape, self.heads
        qk = self.toqk(x); v = self.tov(x)
        merge = lambda z: z.view(b, t, h, -1).transpose(1, 2).reshape(b * h, t, -1)
        qk, v = merge(qk), merge(v)
        out = self.lsh(qk, v, **kwargs)
        out = out.view(b, h, t, -1).transpose(1, 2).reshape(b, t, -1)
        return self.to_out(out)

class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])
    def forward(self, x, **kwargs):
        bk = {'f_args': kwargs, 'g_args': {}}
        return _ReversibleFunction.apply(x, self.blocks, bk)

class Reformer(nn.Module):
    def __init__(self, dim, depth, heads=8, bucket_size=64, n_hashes=8,
                 ff_chunks=100, causal=False):
        super().__init__()
        blocks = []
        for _ in range(depth):
            f = PreNorm(dim, LSHSelfAttention(dim, heads, bucket_size, n_hashes, causal))
            g = PreNorm(dim, Chunk(ff_chunks, FeedForward(dim), along_dim=-2))  # chunked FFN
            blocks.append((f, g))
        self.layers = ReversibleSequence(blocks)
    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)                 # supply (X1, X2) to the reversible block
        x = self.layers(x, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)  # recombine the two halves

class ReformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads=8,
                 bucket_size=64, n_hashes=8, ff_chunks=100, causal=True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.reformer = Reformer(dim, depth, heads, bucket_size, n_hashes, ff_chunks, causal)
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)
    def forward(self, x, **kwargs):
        t = torch.arange(x.shape[1], device=x.device)
        x = self.token_emb(x) + self.pos_emb(t)
        x = self.reformer(x, **kwargs)
        return self.to_logits(self.norm(x))
```

The causal chain, end to end: a Transformer on a long sequence blows up memory three independent ways — $O(L^2)$ attention, activations stored $n_l$ times for backprop, and a feed-forward intermediate that is $d_{ff}$ times wide. Softmax is dominated by the largest dot products, so each query only needs its nearest keys; angular LSH ($h(x)=\arg\max([xR;-xR])$) finds those by bucketing similar directions, made coherent by sharing $Q=K$ and normalizing the keys; sorting by bucket and chunking with one-look-back turns the ragged buckets into fixed batched matmuls, multiple independent rounds (combined by $\sum_r\exp(z^{(r)}-z)\,o^{(r)}$, deduped by $\log N_{i,j}$) recover missed neighbors, and a finite self-mask handles the shared-QK self-similarity — attention is now $\sim O(L\log L)$. Reversible residual layers $y_1=x_1+F(x_2),\,y_2=x_2+G(y_1)$ invert to $x_2=y_2-G(y_1),\,x_1=y_1-F(x_2)$, so activations are stored once and recomputed per layer on the backward pass, erasing the $n_l$ multiplier. And because the feed-forward is position-wise, chunking it over the sequence erases the $d_{ff}$ multiplier on peak memory. The three multipliers are gone; the model fits and trains at $L=64\text{K}$ on a single accelerator.
