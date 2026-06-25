Let me start from what actually hurts. Training these attention models is fine — I run a whole sequence through at once, every position in parallel, the matmuls are big and dense and the hardware loves them. But generation is a different animal. To produce text autoregressively I have to emit one token, feed it back in, emit the next, and so on, strictly one position at a time, because the token I sample at position t becomes the input at position t+1. There's no parallelism over the sequence to hide behind. And at every single one of those steps, the self-attention layer has to look back at everything that came before — the query at the new position attends to the keys and values produced at all earlier positions. So the question I want to answer precisely is: when I generate token by token, what is actually limiting me? Because if I just count flops, incremental decoding does the same total arithmetic as the parallel version, and yet it feels far slower. Flop-counting is lying to me about where the time goes.

The thing flop-counting ignores is that on this hardware arithmetic is cheap and memory is dear. The chip can do on the order of a hundred arithmetic operations in the time it takes to stream one operand in from memory. So the real currency isn't flops, it's the ratio of bytes I have to move to flops I get to do. If that ratio is small — if every byte I load gets reused in many operations — I'm compute-bound and fast, the arithmetic units stay busy. If the ratio creeps up toward one, I'm memory-bandwidth bound: the multiply-add units sit idle waiting for operands to arrive, and no amount of arithmetic peak helps. So I should stop counting flops and start counting the memory-to-arithmetic ratio, separately for the parallel case and the incremental case, and see where it goes bad.

Let me set up the bookkeeping with a few clean assumptions so the algebra stays readable: the memory I attend over has the same length as the query sequence, m = n; the per-head key and value dimensions are k = v = d/h, which is the standard choice that keeps total attention cost near that of a single full-width head; and the sequence is no longer than the model width, n ≤ d. Now the batched, all-at-once attention — the training-style forward where I push all n query positions through together. What arithmetic does it do? Each of the projection einsums and the attention contractions is O(b n d²) under these assumptions, so the total is Θ(b n d²). What memory does it touch? Sum the sizes of every tensor: the activations X, M, Q, K, V, O, Y are each O(b n d); the logits and attention weights are O(b h n²); the projection tensors P_q, P_k, P_v, P_o are O(d²). So memory is O(b n d + b h n² + d²). Divide memory by arithmetic and the ratio comes out O(1/k + 1/(b·n)). That's small — k is in the tens or hundreds and b·n is large. Good. The reason it's small is intuitive: each key and value I load gets dotted against all n query positions, so the cost of fetching it is amortized over n queries. Bandwidth is not the problem here.

Now the incremental case, the one that actually hurts. I generate one position at a time, so there are n separate calls, each handling a single query position, and I keep a cache of all the keys and values produced so far so I don't recompute them. Across all n calls the arithmetic is still Θ(b n d²) — same total work, I just sliced it differently. But the memory access is the whole story now. At decode step t I have to reload the cached K and V, which by then hold all the earlier positions. Summed across the n calls, the K/V reloading costs Θ(b n² d) — at each step I reload an O(b·(step)·d)-sized cache, and summing step from 1 to n gives the n² — and reloading the projection tensors across the n calls costs Θ(n d²). So total memory is Θ(b n² d + n d²). Divide by the arithmetic Θ(b n d²):

  memory / arithmetic = Θ( (b n² d + n d²) / (b n d²) ) = Θ( n/d + 1/b ).

There it is. The ratio for incremental decoding is Θ(n/d + 1/b), and unlike the parallel case it is not automatically small. Let me put the actual model numbers in to see whether this is a real problem or just an asymptotic worry. The translation model I'll be measuring against has d = 1024, h = 8, and the decode protocol generates 128 target tokens, so n runs up to 128. At a single sequence, b = 1: the ratio is 128/1024 + 1/1 = 0.125 + 1.0 = 1.125. Above one. If I imagine pushing the same model out to n = 1024, it's 1.0 + 1.0 = 2.0. So in the small-batch regime this isn't asymptotic hand-waving — the ratio is literally ≥ 1, meaning I'm moving at least as many bytes as I do flops, and on hardware where bytes cost ~100× a flop the multiply-add units are sitting almost completely idle. The diagnosis holds with real constants: incremental decoding is slow because of the bandwidth spent reloading the key/value cache (and the projections) at every step, not because of arithmetic.

One thing those numbers immediately flag, though: at b = 1 the 1/b term is the whole show — it's 1.0 out of the 1.125. The n/d term I'm about to attack is only 0.125 of it. So I should be honest that whatever I do to the cache will barely move the needle unless I also fix 1/b. That's a useful warning to carry forward, not a reason to abandon the cache term: 1/b is trivially fixed by batching, and once it's gone the cache term is what's left.

Two terms to kill, n/d and 1/b, both need to be pushed well below 1. The 1/b term is easy and I shouldn't waste effort on it: it just says use a bigger batch, decode more sequences at once, memory permitting. The reloading of the projection matrices that sits behind it gets amortized over the batch. Fine. The hard term is n/d. Let me trace exactly where it comes from so I attack the right thing. It came from the b n² d memory term, the cost of reloading the K and V caches across all the steps. For one of K or V at full length, standard dense multi-head attention stores b · h · m · k elements. Substituting m = n and k = d/h gives b · h · n · (d/h) = b · n · d, and summing those full-history reloads over n decode steps gives Θ(b n² d). The important part is how that b n d is built: it is h separate d/h-wide key/value slices. If I can make the cache one such slice instead of h slices, h · k = d becomes just k = d/h, and the cumulative reload term loses a factor of h.

So now I see the lever. The bandwidth bottleneck in decoding is the per-head multiplicity of the cached keys and values. If I could remove the head dimension from what I cache and reload for K and V, I'd cut that dominant term by a factor of h. Let me see what the obvious ways to do that cost me.

The most direct knob: just use fewer heads. Drop h from 8 to, say, 1 or 2, widen the feed-forward layers to keep the parameter count fair, and the K/V cache shrinks proportionally. Let me think about whether that's acceptable, because it's the lazy answer and I suspect it's wrong. What do the h heads actually buy me? Each head is an independent query/key/value projection into a d/h-dimensional subspace, an independent read pattern — one head can track subject-verb agreement, another can copy a rare token, another can attend to the previous word, and the output projection mixes all those reads. The representational power of multi-head attention is precisely that it runs many of these read patterns in parallel over the same memory and combines them. If I cut h down to 1 or 2, I'm not just shrinking a cache, I'm throwing away most of the model's ability to attend in multiple ways at once. And it's documented across the design space that reduced-head and reduced-d_k multi-head variants give up quality far out of proportion to the state they save — they degrade markedly relative to the full-head baseline. So "just use fewer heads" trades the cache for the thing I was trying to protect. Wall. Same verdict for shrinking d_k and d_v directly: that cuts the per-head capacity and degrades quality for the same reason. The cache and the quality are entangled if I shrink heads symmetrically.

The other family I know about reduces n instead — restrict each position to a local window of, say, the previous 31 positions, or otherwise compress how many memory positions get attended to. That attacks the n in n/d directly and it's a real, useful idea. But it changes what the model is allowed to look at — its reach — and more to the point for my problem, it's orthogonal: even with a local window, within that window every head still keeps its own K and V, so the heads factor in the cached state is untouched. It composes with whatever I do about heads, it doesn't substitute for it. So neither knob hits the thing I isolated, which is specifically the h-fold replication of the keys and values.

Let me stare at the symmetry between queries and keys/values for a second, because that's where the entanglement might be breakable. Why do I have h of everything? The standard layer gives me h query projections, h key projections, h value projections, and an h-way output projection. The reason fewer heads hurt is that I lose the h different *read patterns*. But which part of a head actually constitutes a "read pattern"? A head asks a question (its query) and combines an answer (its mixed value, projected out). The keys and values are the *content of the memory* being queried — the same underlying sequence, just projected. The h-fold multiplicity on the query side is h different questions; the h-fold multiplicity on the key/value side is h different *projections of the same memory*. Those are not obviously equally important. The questions — the queries, plus the output projection that mixes the per-question answers — are plausibly where the multi-head expressiveness lives. The keys and values are the answers written into memory, and maybe one shared set of answers, read by h different questions, retains most of the power while h separate sets are largely redundant.

If that's right, I don't have to shrink heads symmetrically. I could keep all h query heads — keep the h read patterns, keep the h-way output projection, keep exactly the part I just argued quality depends on — and remove the head dimension *only* from the keys and values. One shared set of keys, one shared set of values, read by all h query heads. The queries would still differ head to head; the output would still mix h head-outputs; only the memory being looked at would be shared. That's an asymmetric design — many query heads, one key head, one value head — but it's a hypothesis, not a proof: I'm betting that the K/V multiplicity is the redundant part. I'll have to check two things before I trust it: that the operation is even well-defined, and that it actually buys back the bandwidth I derived.

First, well-definedness. In the standard layer the per-head attention is, for head i: logits_i = q_i · K_iᵀ, weights_i = softmax(logits_i), o_i = weights_i · V_i, and then y = Σ_i o_i · P_o,i. If I share K and V across heads, I just drop the head index from them: logits_i = q_i · Kᵀ for the single shared K, weights_i = softmax of that, o_i = weights_i · V for the single shared V, and the output mixing y = Σ_i o_i · P_o,i is unchanged. Each query head still produces its own logits (because q_i still differs) and its own attention pattern and its own mixed output — the heads remain genuinely different reads — they just read against a common K and V. In the projection tensors this is: P_q stays [h, d, k], P_o stays [h, d, v], but P_k drops from [h, d, k] to [d, k] and P_v drops from [h, d, v] to [d, v]. In einsum, wherever K, V, P_k, or P_v carried an h index, I delete the h. K = einsum("bmd,dk->bmk", M, P_k) instead of "bmd,hdk->bhmk"; V likewise; the logits contraction "bhnk,bmk->bhnm" pairs the per-head queries against the shared keys; the value mixing "bhnm,bmv->bhnv" against the shared values; the output "bhnv,hdv->bnd" is untouched.

That's a lot of index juggling to do in my head, and the part I'm least sure of is whether the batched einsum form really computes the same thing as h independent heads looping over a shared K and V — and whether the incremental decode loop, with its growing headless cache, reproduces the batched forward. If either of those is off, the whole speed argument is moot. So I'll trace it on a tiny case rather than trust the indices: b=1, n=m=3, d=8, h=2, k=v=4 (so k=d/h holds), random projections, causal mask. I compute three things. (a) the batched MQ forward exactly as written above. (b) an explicit Python loop over the h heads, each doing qi·Kᵀ, softmax, ·V on the *same* shared K, V, accumulating oi·P_o,i — this is the "h different questions, one memory" semantics I claimed. (c) the incremental decode: start with empty headless caches prevK:[b,0,k], prevV:[b,0,v], and for each position append one shared (k,v) row and attend. Running it: the batched output and the per-head loop agree to max-abs-difference 0.0, so the einsum really is h independent reads of a shared memory and I didn't fumble an index. And the incremental output equals the batched output to 0.0 as well — the growing headless cache reproduces the parallel forward position-for-position, with the causal mask falling out automatically because at step t the cache only contains positions ≤ t. Good: the operation is well-defined and the decode path is faithful. Now the part that actually matters — does it move the ratio.

Now I have to check it actually fixes the ratio I derived, or this is all for nothing. Redo the incremental memory accounting with shared K and V. The arithmetic is unchanged, Θ(b n d²). The memory across the n calls: the small per-step activations x, q, o, y contribute Θ(b n d); the cached K and V now have no head dimension, so each is of size b · m · k = b · n · (d/h) per length, and summed over the n decode steps that term is Θ(b n² k) = Θ(b n² d / h); the projection reloads contribute Θ(n d²). So memory is Θ(b n d + b n² k + n d²). Divide by arithmetic Θ(b n d²):

  memory / arithmetic = Θ( 1/d + n/(d h) + 1/b ).

Compare to the multi-head Θ(n/d + 1/b). The offending n/d term has become n/(d·h) — reduced by exactly the factor h I was hunting for. With h = 8 that's an 8× reduction in that term. But I got burned once already by asymptotics, so let me plug the model constants in again instead of declaring victory. d = 1024, h = 8, n = 128.

At b = 1: multi-head is 0.125 + 1.0 = 1.125; multi-query is 1/1024 + 128/(1024·8) + 1.0 = 0.001 + 0.0156 + 1.0 = 1.017. So at a single sequence the improvement is only 1.125 → 1.017, about 1.1×. That's almost nothing — exactly the warning I flagged earlier: the 1/b = 1.0 term dominates both and MQA doesn't touch it. If I'd only ever looked at b = 1 I might have concluded MQA barely helps, which is the trap of attacking the wrong term.

Now do it at a real decode batch, b = 128: multi-head is 0.125 + 0.0078 = 0.1328; multi-query is 0.001 + 0.0156 + 0.0078 = 0.0244. Ratio 0.1328 / 0.0244 ≈ 5.4×, and both are now well below 1. So the picture is: batching kills the 1/b term and drags both variants under the bandwidth ceiling, and *once it's gone* MQA's h-fold shrinkage of the cache term is what separates them — pushing decoding from near the ceiling down to comfortably compute-bound. (At n = 1024, b = 128 the gap widens further, ≈ 7.5×, because the cache term grows with n and that's exactly the term MQA divides by h.) So the order of operations matters and I had it right: batch first to clear 1/b, then MQA to clear the cache term. The cache I reload every step shrank by the number of heads, and after batching that was the binding constraint.

Let me also count it the way I'd think about the memory footprint of the cache directly, as a sanity check from the storage side rather than the bandwidth side — if the two pictures disagree I've made an error somewhere. The bytes I cache per generated token are: 2 (one for K, one for V) times the number of layers times, per layer, the size of the cached key matrix, times the bytes per element. In the standard layer that key matrix is h × d_head, so the per-token cache is 2 · n_layers · h · d_head · bytes. The h sits right there. Share the keys and values across heads and that h × d_head collapses to a single 1 × d_head. With the model constants, per layer per token: multi-head caches 2 · h · d_head = 2 · 8 · 128 = 2048 elements; multi-query caches 2 · 1 · 128 = 256. That's exactly an 8× = h reduction, the same factor the bandwidth derivation gave — the two pictures agree, which is what I wanted to see. So the cost was the heads multiplicity on K and V, and sharing removes it: one head's worth of keys and values, written once and reread by every query head, is enough to keep the read patterns while cutting the cache by h.

I should be clear-eyed about what I have and haven't shown. I've shown the operation is well-defined, that the decode path is faithful, and that the bandwidth and cache-footprint both drop by exactly h. What I have *not* shown here is that quality holds — the argument that the K/V heads are "the redundant part" is a hypothesis backed by intuition about read patterns and by the documented fragility of head-shrinking, but the only thing that can actually confirm it is training the thing and comparing BLEU / perplexity at matched parameters. I expect a small quality cost for a large speed gain; I'd want to verify the size of that cost on the WMT and Billion-Word setups before claiming the trade is worth it.

There are a couple of design details I have to get right so the comparison is honest and the layer is numerically sound. First, parameters. Sharing K and V deletes (h−1) heads' worth of key and value projection parameters. If I just drop them, the multi-query model has fewer parameters than the baseline, and then any quality difference is confounded with capacity. So to compare fairly at equal capacity, I widen the feed-forward hidden layers to put those parameters back — e.g. on the translation model the FFN goes from 4096 up to 5440, on the language model from 8192 up to 9088. That's a fair-comparison control, not part of the attention mechanism; the mechanism is purely the shared K/V. Second, the logit scaling. Scaled dot-product attention divides the logits by √d_k to keep the softmax in a sane range when d_k is large — without it, dot products of d_k-dimensional vectors grow like d_k and push the softmax into saturated, low-gradient regions. In the einsum code I can leave that constant out and fold it into P_q or P_k, since scaling the query or key projection by 1/√d_k is equivalent to scaling the logits; in a framework where I call a fused attention kernel I'll just pass the scale explicitly. Either way it's the same √d_k I already know from standard attention, and sharing K/V doesn't change it.

Now let me write the layer the way I'd actually run it, because I want this droppable into a real codebase, not just einsum sketches. Start from the incremental decode step, since that's the case the whole derivation was about. I form the query for the new position keeping its h heads, q = einsum("bd,hdk->bhk", x, P_q). Then — and this is the only structural change — I project the new position's key and value with the *headless* P_k:[d,k] and P_v:[d,v], giving a single [b,k] and [b,v], and append each to its running cache, which now has shape [b, m+1, k] and [b, m+1, v] with no head dimension. Then the per-head query attends against the shared cache: logits = einsum("bhk,bmk->bhm", q, K), weights = softmax(logits), o = einsum("bhm,bmv->bhv", weights, V), and the output mixes the h head-outputs as before, y = einsum("bhv,hdv->bd", o, P_o). The batched training-time forward is the same edit applied to the parallel version: Q keeps "bhnk", K and V lose their h to become "bmk"/"bmv", the logits contraction is "bhnk,bmk->bhnm", the value mix "bhnm,bmv->bhnv", the output "bhnv,hdv->bnd". One mask add on the logits handles autoregressive masking exactly as in multi-head.

```python
import tensorflow as tf  # einsum: named-index tensor contraction


def MultiqueryAttentionBatched(X, M, mask, P_q, P_k, P_v, P_o):
    """Multi-query attention: h query heads, ONE shared key head and value head.
    X: [b, n, d]   M: [b, m, d]   mask: [b, h, n, m]
    P_q: [h, d, k]   P_k: [d, k]   P_v: [d, v]   P_o: [h, d, v]  ->  Y: [b, n, d]"""
    Q = tf.einsum("bnd,hdk->bhnk", X, P_q)        # queries keep h heads
    K = tf.einsum("bmd,dk->bmk", M, P_k)          # ONE shared key head (no h)
    V = tf.einsum("bmd,dv->bmv", M, P_v)          # ONE shared value head (no h)
    logits = tf.einsum("bhnk,bmk->bhnm", Q, K)    # each query head vs the shared keys
    weights = tf.softmax(logits + mask)
    O = tf.einsum("bhnm,bmv->bhnv", weights, V)   # each query head mixes the shared values
    Y = tf.einsum("bhnv,hdv->bnd", O, P_o)        # h head-outputs projected and summed
    return Y


def MultiquerySelfAttentionIncremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    """One decode step. prev_K/prev_V are headless caches: [b, m, k] / [b, m, v].
    This is the case the whole derivation was about: the reloaded cache has no h."""
    q = tf.einsum("bd,hdk->bhk", x, P_q)          # h query heads for the new position
    K = tf.concat(                                 # append the new shared key (no h)
        [prev_K, tf.expand_dims(tf.einsum("bd,dk->bk", x, P_k), axis=1)], axis=1)
    V = tf.concat(                                 # append the new shared value (no h)
        [prev_V, tf.expand_dims(tf.einsum("bd,dv->bv", x, P_v), axis=1)], axis=1)
    logits = tf.einsum("bhk,bmk->bhm", q, K)       # per-head queries vs shared keys
    weights = tf.softmax(logits)
    o = tf.einsum("bhm,bmv->bhv", weights, V)      # per-head mix of shared values
    y = tf.einsum("bhv,hdv->bd", o, P_o)           # mix h head-outputs and project out
    return y, K, V                                 # caches stay headless -> h-fold less reload
```

That's the method in its native einsum form: keep the h on the queries and the output, delete it everywhere on the keys and values, and the cache you reload every decode step is suddenly h times smaller.

For a contemporary dense-tensor codebase I'd realize the same idea a little differently, because there I don't keep four separate projection tensors — I fuse the query, key, and value projections into one linear and split the result, and I lean on a fused attention kernel. The structural content is identical: the fused projection produces a query of full width n_embd (h heads' worth) and a *single* head of key and value, head_dim each; I reshape the query to (b, h, n, d_head) and the key and value to (b, 1, n, d_head); then, because the kernel wants matching head counts, I broadcast the one shared K/V head up to h heads before the attention call — a logical expansion of one physical head, no extra cached state — and call scaled dot-product attention with causal masking and the √d_k scale applied inside. The thing that's cached and reloaded in decoding is still the single (k, v) head, so the bandwidth win is preserved.

```python
import math
import torch
import torch.nn as nn


def expand_kv_to_q_heads(t, target_heads):
    """Broadcast the shared KV head(s) up to the query-head count for the kernel.
    Logical only: no extra (k, v) is cached, so the decode reload stays h-fold smaller."""
    cur = t.size(1)
    if cur == target_heads:
        return t
    full = target_heads // cur
    rem = target_heads % cur
    parts = []
    if full > 0:
        parts.append(t.repeat_interleave(full, dim=1))
    if rem > 0:
        parts.append(t[:, :rem, :, :])
    return torch.cat(parts, dim=1)


class CausalSelfAttention(nn.Module):
    """Multi-query attention: h query heads, ONE shared (key, value) head."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.n_kv_head = 1                                   # the whole idea: one shared KV head
        # fused projection: full-width query (h heads) + ONE head each of key and value
        kv_dim = 2 * self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)  # P_o (mixes h heads)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        b, n, c = x.size()
        qkv = self.c_attn(x)
        q, kv = qkv.split([self.n_embd, 2 * self.n_kv_head * self.head_dim], dim=2)
        k, v = kv.chunk(2, dim=2)
        q = q.view(b, n, self.n_head, self.head_dim).transpose(1, 2)        # (b, h, n, d_head)
        k = k.view(b, n, self.n_kv_head, self.head_dim).transpose(1, 2)     # (b, 1, n, d_head)
        v = v.view(b, n, self.n_kv_head, self.head_dim).transpose(1, 2)     # (b, 1, n, d_head)
        k = expand_kv_to_q_heads(k, self.n_head)    # broadcast 1 -> h for the kernel (logical)
        v = expand_kv_to_q_heads(v, self.n_head)
        y = torch.nn.functional.scaled_dot_product_attention(               # softmax(QKᵀ/√d_head)V
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0, is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, n, c)                    # concat heads
        y = self.resid_dropout(self.c_proj(y))                             # P_o
        return y
```

The causal chain, start to finish: incremental autoregressive decoding felt slow even though it does the same arithmetic as parallel training, so I stopped counting flops and counted the ratio of memory bytes moved to arithmetic done — the real currency on hardware where compute outruns bandwidth by ~100×. That ratio is small for batched training, O(1/k + 1/(b·n)), because each loaded key/value is reused across all positions; but for incremental decoding it is Θ(n/d + 1/b), close to 1 for long sequences or small batches, so decoding is memory-bandwidth bound. The 1/b term is just "use a bigger batch"; the hard n/d term traces to reloading the cached keys and values every step, whose stored width is h·d_head in dense multi-head attention. Shrinking K/V the obvious way — fewer heads or smaller d_k — also shrinks the cache but wrecks quality, because the h heads are the model's parallel read patterns. The break is that the multiplicity matters asymmetrically: the queries (and the output mixing) are the read patterns where quality lives, while the keys and values are h projections of the same memory and are largely redundant. So keep all h query heads and share a single key head and value head across them — delete the h index from K, V, P_k, P_v only. Re-deriving the incremental ratio with shared K/V gives Θ(1/d + n/(d·h) + 1/b): the offending term drops by exactly h, and the per-token KV cache (hence the per-step reload) drops by h — confirmed both by plugging in the model constants (1.125 → 1.017 at b=1, but 0.133 → 0.024, a 5.4× gap, once batching clears the 1/b term) and by the direct cache count (2048 → 256 elements). The order matters: batch first to kill 1/b, then MQA to kill the cache term and bring decoding under the bandwidth ceiling. Widen the FFN to match parameters for a fair quality comparison, keep the √d_k scale folded into the projections or applied in the kernel, and the layer drops in as a one-index edit to the einsum forward — or, in a fused-projection codebase, as a single shared (k, v) head broadcast up to h heads for the attention kernel while only one head is ever cached.
