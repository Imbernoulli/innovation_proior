OK, let me think this through from scratch. The thing that hurts is serving these big decoder models — generation is slow — so before I reach for any architectural cleverness I want to know *exactly* where the time goes when I generate one token. Not the asymptotics of training a full sequence in parallel; the actual per-step cost of autoregressive decoding, because that's the regime I'm paying for.

So I generate one token. What does the machine actually do? It reads the model weights out of memory, it reads the cached keys and values for every position I've already produced, and it does some arithmetic with all of that to emit a single next-token distribution. Each loaded weight is reused only across the current token batch, and only one token's worth of activations is flowing through. The *reads* are not small: I have to stream the weight matrices and the entire history of keys and values across the memory bus, every step, just to produce one token. That smells memory-bound, not compute-bound. Let me make that precise instead of hand-waving, because "feels memory-bound" isn't an argument.

The clean way to talk about this is arithmetic intensity: operations performed per byte loaded from memory. The hardware has a fixed ceiling — it can do so many FLOPs per second and read so many bytes per second, and the ratio of those two is a property of the chip. If my kernel does fewer operations per byte than that ratio, then the compute units finish early and sit idle waiting on the memory bus; I'm memory-bound, and making the arithmetic cheaper buys me nothing — only moving fewer bytes helps. So the real question is: what is the arithmetic intensity of incremental attention decoding, and which term is dragging it down?

Let me count both sides over a whole generated sequence. Batch $b$, sequence length $n$, model dimension $d$, $h$ heads, head dimension $k = d/h$ so that the heads tile the full width. Arithmetic first. Per token, the work is dominated by the four projections — query, key, value, and the output projection — each a $d \times d$ matmul on one token's vector, so $\Theta(d^2)$ per token. The attention score-and-mix itself is $\Theta(\text{history length} \cdot k)$ per head, which summed over heads is $\Theta(d \cdot \text{history})$ — smaller than $d^2$ until the history gets very long, so the projections dominate. Over $n$ steps and batch $b$: arithmetic $= \Theta(b\,n\,d^2)$.

Now memory. Two things get streamed. The projection matrices — reloaded each step because there's only one token's worth of activation to reuse them on — cost $\Theta(d^2)$ per step, $\Theta(n\,d^2)$ over the sequence. And the cache: at step $i$ the keys and values each have shape $[b, h, i, k]$, so up to the fixed factor of two for separate key and value tensors I'm touching $b\,h\,i\,k$ numbers to compute step $i$. Sum over $i$ from $1$ to $n$: $\sum_i b\,h\,i\,k = b\,h\,k \cdot \tfrac{n(n+1)}{2} = \Theta(b\,h\,n^2 k)$. And $h k = d$, so that's $\Theta(b\,n^2 d)$. Total memory $= \Theta(b\,n^2 d + n\,d^2)$.

So the ratio of bytes moved to operations done is

$$\frac{b\,n^2 d + n\,d^2}{b\,n\,d^2} = \frac{n}{d} + \frac{1}{b}.$$

The $1/b$ term I can beat by batching more requests together — fine, that's a throughput knob. The $n/d$ term is the one I can't batch away, and I want to know whether it's actually a problem or just a term in an inequality, so let me put real numbers in it. Take a large model in the shape I'd deploy: $d = 4096$, $h = 64$ heads, and serve it at a healthy batch $b = 32$. For a short generation, $n = 512$: the ratio is $512/4096 + 1/32 = 0.125 + 0.031 = 0.156$. Comfortably below one — the $1/b$ batching term is even the larger of the two, so at short context I'm not badly memory-bound. Now push to the long-input regime I care about, $n = 4096$ (right at $d$): $4096/4096 + 1/32 = 1.0 + 0.031 = 1.03$. The ratio has crossed one. By the roofline picture that means roughly one full memory load per arithmetic operation — past the machine balance of any real accelerator, so the compute units finish and sit idle waiting on the bus. And the term that moved it there is $n/d$: it went from $0.125$ to $1.0$, an eightfold jump, while the batching term stayed pinned at $0.031$. So the bottleneck at long context is not batching and not the weights; it's whatever the $n/d$ term represents.

What does it represent? Tracing it back, the $n/d$ came entirely from the $b\,n^2 d$ memory term, which was the cost of streaming the cached keys and values. Those carry an $h$ inside ($b\,h\,n^2 k$) — one key and one value per head per position. The cache is big, and grows quadratically in $n$ when summed over steps, because every head keeps its own keys and values.

That's the whole problem localized to one design decision. Loading the *weights* I can't easily avoid — they're the model. Loading the *keys and values* I might be able to shrink, because that $h$ in $b\,h\,n^2 k$ is a choice, not a law.

There are other ways to chase serving speed, and I should separate them before I cut into attention. I can tile attention more carefully so I do not materialize the full score matrix; that is excellent for training and for full-sequence prefill, but during one-token decoding the thing being reloaded is the cache itself, not a giant fresh attention matrix. I can quantize weights and cached activations; that shrinks bytes per number, but the number of cached key/value vectors is unchanged. I can distill into a smaller model; useful, but then I am changing the whole model and probably losing capacity everywhere, not just in the offending cache term. I can remove some cross-attention layers or use a small draft model to propose tokens for a larger one to verify; those are serving tricks around the model. None of them answers the narrow structural question staring at me: why am I storing a distinct key and a distinct value for every attention head?

So what controls the cache size? The number of key heads and value heads. In the standard layer I have $h$ query heads and, to match, $h$ key heads and $h$ value heads — each head a separate learned projection, each contributing its own slice to the cache. Do I actually need $h$ key/value heads? The query I bring to a decoding step is just the current token's query — tiny, one position, not cached. It's the *keys and values* that accumulate across all past positions and get reloaded every step. The cache cost is entirely about how many distinct key/value heads I store, not how many query heads I run.

The query heads are cheap to keep, and they do the representational work of attending in $h$ different ways, so I should not cut them first. The accumulating objects are the key and value heads. The extreme version is a *single* key head and a *single* value head, shared by all $h$ query heads. Every query head still computes its own attention pattern, but they all attend over the same one set of keys and one set of values.

Let me redo the count under that change and see whether it actually moves the offending term, because shrinking the cache is only worth anything if the cache term was the one dominating. Arithmetic is essentially unchanged — still $\Theta(b\,n\,d^2)$, the query/output projections still dominate (the key/value projections got smaller, which only helps). Memory: the cache is now $[b, n, k]$ — no $h$ — so summed over steps it's $\Theta(b\,n^2 k) = \Theta(b\,n^2 d / h)$. Plus the small reloads of activations $\Theta(b\,n\,d)$ and weights $\Theta(n\,d^2)$. New ratio:

$$\frac{b\,n\,d + b\,n^2 k + n\,d^2}{b\,n\,d^2} = \frac{1}{d} + \frac{n}{d\,h} + \frac{1}{b}.$$

The $n/d$ term has become $n/(dh)$ — divided by $h$. Let me re-run the same long-context numbers to see how much that actually buys, since the asymptotic factor only matters if it dominates the constants. Same $d=4096$, $h=64$, $b=32$, $n=4096$: now $1/4096 + 4096/(4096\cdot 64) + 1/32 = 0.0002 + 0.0156 + 0.031 = 0.047$. The full multi-head ratio at this point was $1.03$; the single-head version is $0.047$, about $22\times$ smaller, and crucially back under one — out of the memory-bound regime entirely. The leftover $0.047$ is now mostly the $1/b = 0.031$ batching term, not the cache. So the collapse didn't just nudge the bottleneck, it removed it: the cache term fell from $1.0$ to $0.016$, almost exactly the factor of $h=64$ I cut from the cache, and what's left is the batching floor I always had.

So why isn't that the end of the story? Let me actually think about what I gave up, not just celebrate the speed. I collapsed $h$ key/value subspaces into one. Each query head used to attend through its *own* key/value projection — its own little subspace in which "what matches what" is defined. Now all $h$ query heads are forced to read and write through a single shared subspace. That's a real loss of capacity. The query heads can still differ in *how* they weight a fixed set of keys, but they can no longer disagree about *what the keys and values are*. Intuitively that should hurt quality, and it does — the single-key/value-head model underperforms the full multi-head one.

And there's a second symptom that bothers me more than a quality dip, because it suggests the cut is too sharp rather than just lossy. When I train a model this way from scratch, training is fragile — loss spikes during pre-training, and outright divergence when I fine-tune on long-input tasks, the exact tasks where the cache savings matter most. Squeezing everything through one key/value head seems to make the optimization landscape mean. So the all-or-nothing collapse isn't just a little worse; it's brittle.

Now I have a tension. Full multi-head: best quality, ruinous cache. Single shared key/value head: cheap cache, degraded and unstable. These are the two endpoints of an axis I've been treating as binary — but is it actually binary? The cache cost scaled with the *number of key/value heads*. I went from $h$ to $1$. Nothing forced that to be the only two options. What if I keep some intermediate number?

Let me see what the diagnostic says about an intermediate count, because the whole point is the cache term, and I want to know it varies smoothly. Suppose I keep $G$ key/value heads, $1 \le G \le h$. Then the cache is $[b, G, n, k]$, the cached-memory term is $\Theta(b\,G\,n^2 k) = \Theta(b\,n^2 d \cdot G/h)$, and the memory-to-compute ratio is

$$\frac{1}{d} + \frac{nG}{d\,h} + \frac{1}{b}.$$

At $G = h$ the cache term is $n/d$ — I'm back to full multi-head, no surprise. At $G = 1$ it's $n/(dh)$ — the single-head case. And in between it slides linearly in $G$. Let me check one intermediate point with the same numbers, because "slides linearly" should give a concrete prediction I can read off the formula. Try $G = 8$ at $d=4096$, $h=64$, $b=32$, $n=4096$: the cache term is $nG/(dh) = 4096\cdot 8/(4096\cdot 64) = 0.125$, and the full ratio is $0.0002 + 0.125 + 0.031 = 0.156$. So $G=8$ sits at $0.156$, between the single-head $0.047$ and the multi-head $1.03$, and its cache term $0.125$ is exactly the multi-head cache term $1.0$ divided by $h/G = 64/8 = 8$ — the factor I expected. The per-step memory traffic that dominates decoding really is proportional to $G$; $G$ is a discrete dial, and it dials exactly the quantity I care about. And $0.156$ is still under one — even keeping eight key/value subspaces, I'm out of the worst of the memory-bound regime.

What does an intermediate $G$ mean structurally? I have $h$ query heads and I want them to share $G$ key/value heads. The natural thing: partition the $h$ query heads into $G$ groups of $h/G$ each, and give each group its own single key head and value head, shared within the group. So query heads in the same group attend over a common key/value subspace; query heads in different groups get different ones. $G = 1$ is one big group — all query heads share — which is the single-head collapse. $G = h$ is $h$ groups of one — every query head gets its own — which is full multi-head. The grouping *is* the interpolation, and the two methods I started with are literally its two endpoints.

Why is an intermediate $G$ not just a compromise but a genuinely good trade? A few things line up. First, capacity: instead of one shared subspace I now have $G$ of them, so the query heads aren't all forced through the same bottleneck — the representational damage of the collapse is spread over $G$ buckets rather than concentrated in one. That should recover most of the quality and, I'd bet, fix the brittleness, since the optimization no longer has to cram everything through a single head. Second, and this is the part I like, the trade *scales the right way*. Bigger models use more heads $h$. The single-head collapse is a fixed cut to one head regardless of $h$, so as $h$ grows it becomes a relatively more aggressive amputation of capacity. But if I instead fix the *proportion* — say $G$ a fixed fraction of $h$, or just a fixed modest number like $G=8$ — then as the model grows I keep cutting the cache by the same proportional factor $h/G$ while the per-group capacity grows with the model. The reduction in bandwidth and the reduction in capacity stay matched as I scale.

There's a sharper version of why intermediate $G$ should be cheap on the speed side for the large models I actually deploy. The cache scales at most linearly with the model width, while the model's parameters and FLOPs scale with $d^2$. So as the model gets bigger, the cache term shrinks *relative* to the compute term — large models are less dominated by the cache to begin with. That means moving $G$ up from $1$ toward $h$ should only slowly re-introduce bandwidth cost at first; the expensive region should be closer to $G=h$, where the cache has grown all the way back to the full multi-head size. A small integer such as $G = 8$ is a sensible target to try: eight subspaces instead of one, while the cache is still cut by $h/8$.

And there's a sharding subtlety that quietly favours $G$ over $1$. When I shard a big model across, say, $P$ accelerators, each partition wants its own copy of the key/value heads to work with. With a single shared head and $P$ partitions, that one head gets *replicated* $P$ times across the partitions — so the "single head" isn't really paying off once it drops below the partition count; I'm storing copies. If instead I pick $G$ around the number of partitions, each partition naturally owns a distinct key/value head and there's no wasteful replication. The grouping lines up with the hardware.

This bandwidth pain is specifically a *decoding* problem — it comes from the sequential, one-token-at-a-time loop where each step reloads the cache. The encoder processes its whole input in parallel; it's compute-bound, not bandwidth-bound, and it has no growing cache to stream. So there's no reason to pay the capacity cost of grouping there. Apply the grouping to the decoder's self-attention and to the cross-attention, both of which sit inside the sequential decode loop, but leave the encoder's self-attention as plain multi-head.

So I think the architecture is settled: partition the $h$ query heads into $G$ groups, one key head and one value head per group, shared within the group, cache proportional to $G$, with $G$ a small number that scales sensibly. Now the harder, more practical question, and honestly the one that decides whether any of this gets used.

There are a lot of excellent multi-head checkpoints already trained, at enormous cost, that I'd love to serve cheaply — but they were trained with $h$ key/value heads, and my grouped model has $G$. The naive answer is "train the grouped model from scratch," and that's a non-starter: I'm not going to spend a full pre-training run just to get a faster-decoding variant of a model I already have. So can I *convert* an existing multi-head checkpoint into a grouped one cheaply, re-using everything I can?

What carries over untouched? The query projections — same $h$ heads. The output projection — same. Everything outside attention — embeddings, feed-forward, norms — same. The *only* thing that structurally changes is the key and value projections: I had $h$ of each, I now want $G$ of each. So the entire conversion problem reduces to: given the $h$ trained key projection matrices in a group, produce *one* key projection matrix for that group (and likewise for values). It's a surgical edit to just the key/value projections.

How should I build the one group projection from the $h/G$ trained head projections in that group? Let me think about what each option destroys. I could just *pick* one of the heads in the group and throw the rest away — but then I've discarded the learned content of $h/G - 1$ heads; the surviving head's subspace is arbitrary and the others' information is gone. I could *randomly initialize* a fresh key/value head for each group — but then I've thrown away *all* the trained key/value structure and I'm asking continued training to relearn it; that's the most destructive option, closest to training from scratch. Both of those waste information that's sitting right there in the checkpoint.

What I want is the *single* projection that best stands in for the whole group of trained projections — that preserves as much of their collective behaviour as possible. Each head's key projection is a linear map $W_k^{(j)}$ from the residual stream to that head's key space. If I'm going to replace the group by one map, the least-committal, most-information-preserving choice is the *average* of the maps in the group:

$$W_k^{g} = \frac{1}{|g|} \sum_{j \in g} W_k^{(j)}, \qquad W_v^{g} = \frac{1}{|g|} \sum_{j \in g} W_v^{(j)},$$

i.e. mean-pool the key projection matrices of the heads in the group, and the value projection matrices likewise. Why the mean specifically? Because it keeps the *average* response of the group's heads intact — the shared, common component of what those heads were computing survives the merge, and only the head-to-head *differences* within a group are lost. Picking one head keeps that one head's idiosyncrasies and loses the average; the mean keeps the average and loses the idiosyncrasies. I can't prove from here that the average is the right thing to keep — that's an empirical bet that the shared component carries more useful signal than any one head's idiosyncrasies, and it's the kind of claim I'd want to check by actually uptraining all three and comparing. What I *can* pin down is that the mean is the most conservative of the three: it's the unique choice that's a no-op on the part the heads agree on and only discards their disagreement, whereas select-one discards $h/G-1$ heads outright and random-init discards everything. On information-preserved grounds I'd order them mean-pool, then select-one, then random-init, and uptrain to settle whether that ordering survives. And the grouping makes the mean sensible: I'm only averaging the $h/G$ heads that are about to share a subspace anyway, not all $h$, so within a group the heads I'm merging are the ones I've *chosen* to make share.

Before I trust this recipe I should check it actually does what I think on the two cases I already understand — the endpoints — rather than assume it. The mechanic is: take the trained key projection, a matrix whose rows are stacked head-by-head, reshape it to $[G, h/G, \text{head\_dim}, \cdot]$, and average over the middle axis. Let me trace it on a toy with $h=4$ query heads, head\_dim $=2$, so the weight is $[8, \cdot]$ with rows $\{0,1\}$ for head 0, $\{2,3\}$ for head 1, and so on. At $G = h = 4$: reshape to $[4, 1, 2, \cdot]$, average over an axis of length one — the average of a single element is itself, so every row comes back unchanged and the output equals the input exactly. I ran this: `torch.equal(converted, original)` is `True`. So the $G=h$ conversion is genuinely the identity, a no-op, and I keep the exact multi-head checkpoint — not "should be," is. At $G = 1$: reshape to $[1, 4, 2, \cdot]$, average over the axis of length four — that's the mean over all four heads' rows, and the trace gives back exactly $\tfrac14\sum_j W_k^{(j)}$, the single-head conversion. And the genuinely new case, $G = 2$ with $h=4$: reshape to $[2, 2, 2, \cdot]$, average over the middle pair, and the trace confirms it produces the average of heads $\{0,1\}$ stacked above the average of heads $\{2,3\}$ — adjacent heads grouped, each group's two heads merged, exactly the partition I wanted. So the same mean-pool recipe converts to any $G$, the two methods I started from are its literal endpoints, and the grouped case is the natural in-between — and I've now watched it happen on real tensors rather than asserted it.

After the surgery the model isn't quite itself — the averaged key/value heads don't perfectly match what the query heads and the rest of the network expect, because those were tuned against $h$ distinct heads. But the damage should be much smaller than random reinitialization: most weights are untouched and the key/value heads start from a sensible average rather than from noise. So I don't need to start over; I need to let the model *adapt* to its new attention structure. Continue pre-training — same recipe, same data, same objective as the original — for a small fraction $\alpha$ of the original number of steps. How small? Enough that the network can re-coordinate the averaged heads with the unchanged query and output projections, but not so much that I've paid for another pre-training run. A few percent is the right scale for a surgical edit; something like $\alpha = 0.05$ is a reasonable default, and a larger value such as $0.10$ is a check on whether adaptation is still the bottleneck. And I'd expect the grouped model to need less of this adaptation than the single-head one: the single-head conversion is the violent merge, all $h$ heads into one, whereas the grouped conversion preserves more structure before any adaptation steps at all.

So the full recipe is two moves: convert the checkpoint by mean-pooling each group's key and value projections into one, then continue pre-training briefly to let it adapt. The architecture is the grouping; the cheap-to-obtain part is the conversion. Together they should let me take a trained multi-head model and get a faster-decoding grouped model for a few percent of the original compute, with capacity loss controlled by $G$ instead of forced all the way to one shared subspace.

Let me write the code in the shape of the decoder attention module I actually need. The config carries $H$ query heads and $G$ key/value heads. The query projection still emits $H$ heads. The key and value projections emit only $G$ heads, so the cache stores $G$. Right before the score matmul, after any cache update, I repeat each stored key/value head $H/G$ times so the tensors line up with the $H$ query heads. The repeat is deliberately after the cache: store the small tensor, expand only for the local computation.

The one part of this I should actually verify rather than wave at is the repeat, because the whole correctness of grouping rests on *which* query head ends up reading *which* shared key/value head. The expansion does `x[:, :, None, :, :].expand(...).reshape(b, G*n_rep, ...)` — an interleave that should turn $G$ kv heads into $H$ by adjacent repetition. Trace it with $G=2$ kv heads, $n\_rep = H/G = 2$, kv head 0 carrying the marker $10$ and kv head 1 carrying $20$: the expanded tensor comes out as $[10, 10, 20, 20]$. So query heads 0 and 1 read shared kv head 0, query heads 2 and 3 read shared kv head 1 — adjacent query heads grouped together, which is the same partition the mean-pool used (heads $\{0,1\}$ averaged into group 0, $\{2,3\}$ into group 1). The conversion and the runtime agree on the grouping, which they have to, or a query head would attend through a key it wasn't merged with. And I should sanity-check the degenerate case the asymptotics promised: with $G=H$, `n_rep` is 1, the function returns the cache untouched, and running the full attention with it against a random input gives output bit-for-bit identical to plain multi-head attention (`torch.equal` is `True`). So at $G=H$ the module *is* the baseline, not merely close to it — the structural change is exactly localized to the kv-head count and nothing else shifts.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionConfig:
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_key_value_heads=None,
        attention_dropout=0.0,
        attention_bias=False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_heads = (
            num_heads if num_key_value_heads is None else num_key_value_heads
        )
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias


def repeat_key_value_heads(hidden_states, n_rep):
    # [b, G, s, k] -> [b, G*n_rep, s, k]. Only the [b, G, s, k] tensor
    # is cached; this expansion is local to the attention matmul.
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, seq_len, head_dim
    )


class DecoderAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_key_value_heads(key_states, self.num_key_value_groups)
        value_states = repeat_key_value_heads(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)
```

The conversion is the same surgical idea in code. I reshape the trained key/value projection rows as heads, group them, average over the within-group axis, and write the result into the smaller key/value projection. Query and output projections copy directly. If $G=h$, the group size is one and this is exactly the identity on the key/value weights; if $G=1$, the mean is over all heads.

```python
def convert_attention_checkpoint(pretrained_attention, config):
    converted = DecoderAttention(
        config, layer_idx=getattr(pretrained_attention, "layer_idx", 0)
    )
    source_heads = pretrained_attention.num_heads
    target_heads = config.num_key_value_heads
    head_dim = pretrained_attention.head_dim
    if source_heads % target_heads != 0:
        raise ValueError("source heads must be divisible by target key/value heads")
    heads_per_group = source_heads // target_heads

    def copy_linear(dst, src):
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)

    def mean_pool_weight(weight):
        grouped = weight.view(target_heads, heads_per_group, head_dim, -1)
        return grouped.mean(dim=1).reshape(target_heads * head_dim, -1)

    def mean_pool_bias(bias):
        grouped = bias.view(target_heads, heads_per_group, head_dim)
        return grouped.mean(dim=1).reshape(target_heads * head_dim)

    with torch.no_grad():
        copy_linear(converted.q_proj, pretrained_attention.q_proj)
        copy_linear(converted.o_proj, pretrained_attention.o_proj)
        converted.k_proj.weight.copy_(mean_pool_weight(pretrained_attention.k_proj.weight))
        converted.v_proj.weight.copy_(mean_pool_weight(pretrained_attention.v_proj.weight))
        if converted.k_proj.bias is not None and pretrained_attention.k_proj.bias is not None:
            converted.k_proj.bias.copy_(mean_pool_bias(pretrained_attention.k_proj.bias))
        if converted.v_proj.bias is not None and pretrained_attention.v_proj.bias is not None:
            converted.v_proj.bias.copy_(mean_pool_bias(pretrained_attention.v_proj.bias))
    return converted


def continue_pretraining(model, pretrain_step_fn, original_steps, alpha=0.05):
    # The averaged key/value heads have to re-coordinate with the untouched
    # query/output projections and the rest of the network.
    for _ in range(int(alpha * original_steps)):
        pretrain_step_fn(model)
    return model
```

The causal chain, start to end: autoregressive decoding is memory-bound, and the diagnostic pins the bottleneck to one term — the $n/d$ in the memory-to-compute ratio — which is the cost of streaming a key/value cache that carries one key and one value per head. Collapsing to a single shared key/value head cuts that term by the head count $h$ and makes decoding fast, but forcing all query heads through one subspace costs quality and destabilizes training. Since the cache cost is simply proportional to the number of key/value heads, that number is a discrete dial: partition the query heads into $G$ groups sharing $G$ key/value heads, with the single-head and full multi-head cases as the $G=1$ and $G=h$ endpoints — capacity spread over $G$ subspaces and cache cut by $h/G$. And because only the key/value projections change, an existing multi-head checkpoint converts by mean-pooling each group's trained key/value projections into one — the most conservative merge, discarding only the heads' within-group disagreement, and the one I'd back over select-one or random-init pending an uptraining comparison — followed by a few percent of continued pre-training to adapt, giving a way to lower decode bandwidth without paying for a fresh pre-training run.
