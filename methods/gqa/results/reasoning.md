OK, let me think this through from scratch. The thing that hurts is serving these big decoder models — generation is slow — so before I reach for any architectural cleverness I want to know *exactly* where the time goes when I generate one token. Not the asymptotics of training a full sequence in parallel; the actual per-step cost of autoregressive decoding, because that's the regime I'm paying for.

So I generate one token. What does the machine actually do? It reads the model weights out of memory, it reads the cached keys and values for every position I've already produced, and it does some arithmetic with all of that to emit a single next-token distribution. Each loaded weight is reused only across the current token batch, and only one token's worth of activations is flowing through. The *reads* are not small: I have to stream the weight matrices and the entire history of keys and values across the memory bus, every step, just to produce one token. That smells memory-bound, not compute-bound. Let me make that precise instead of hand-waving, because "feels memory-bound" isn't an argument.

The clean way to talk about this is arithmetic intensity: operations performed per byte loaded from memory. The hardware has a fixed ceiling — it can do so many FLOPs per second and read so many bytes per second, and the ratio of those two is a property of the chip. If my kernel does fewer operations per byte than that ratio, then the compute units finish early and sit idle waiting on the memory bus; I'm memory-bound, and making the arithmetic cheaper buys me nothing — only moving fewer bytes helps. So the real question is: what is the arithmetic intensity of incremental attention decoding, and which term is dragging it down?

Let me count both sides over a whole generated sequence. Batch $b$, sequence length $n$, model dimension $d$, $h$ heads, head dimension $k = d/h$ so that the heads tile the full width. Arithmetic first. Per token, the work is dominated by the four projections — query, key, value, and the output projection — each a $d \times d$ matmul on one token's vector, so $\Theta(d^2)$ per token. The attention score-and-mix itself is $\Theta(\text{history length} \cdot k)$ per head, which summed over heads is $\Theta(d \cdot \text{history})$ — smaller than $d^2$ until the history gets very long, so the projections dominate. Over $n$ steps and batch $b$: arithmetic $= \Theta(b\,n\,d^2)$.

Now memory. Two things get streamed. The projection matrices — reloaded each step because there's only one token's worth of activation to reuse them on — cost $\Theta(d^2)$ per step, $\Theta(n\,d^2)$ over the sequence. And the cache: at step $i$ the keys and values each have shape $[b, h, i, k]$, so up to the fixed factor of two for separate key and value tensors I'm touching $b\,h\,i\,k$ numbers to compute step $i$. Sum over $i$ from $1$ to $n$: $\sum_i b\,h\,i\,k = b\,h\,k \cdot \tfrac{n(n+1)}{2} = \Theta(b\,h\,n^2 k)$. And $h k = d$, so that's $\Theta(b\,n^2 d)$. Total memory $= \Theta(b\,n^2 d + n\,d^2)$.

So the ratio of bytes moved to operations done is

$$\frac{b\,n^2 d + n\,d^2}{b\,n\,d^2} = \frac{n}{d} + \frac{1}{b}.$$

Stare at that. The $1/b$ term I can beat by batching more requests together — fine, that's a throughput knob. But the $n/d$ term is the killer: as the generated sequence length $n$ climbs toward the model dimension $d$ — and for the long-input tasks I care about, $n$ in the thousands is right in the neighbourhood of $d$ — this ratio approaches and then exceeds one. Roughly one full memory load per arithmetic operation. That's deep in the memory-bound regime; the compute units are starving. And I can read off precisely *where* the $n/d$ comes from: it's the $b\,n^2 d$ memory term, which is the cost of streaming the cached keys and values. Those carry an $h$ inside ($b\,h\,n^2 k$) — one key and one value per head per position. The cache is big because every head keeps its own keys and values.

That's the whole problem localized to one design decision. Loading the *weights* I can't easily avoid — they're the model. Loading the *keys and values* I might be able to shrink, because that $h$ in $b\,h\,n^2 k$ is a choice, not a law.

There are other ways to chase serving speed, and I should separate them before I cut into attention. I can tile attention more carefully so I do not materialize the full score matrix; that is excellent for training and for full-sequence prefill, but during one-token decoding the thing being reloaded is the cache itself, not a giant fresh attention matrix. I can quantize weights and cached activations; that shrinks bytes per number, but the number of cached key/value vectors is unchanged. I can distill into a smaller model; useful, but then I am changing the whole model and probably losing capacity everywhere, not just in the offending cache term. I can remove some cross-attention layers or use a small draft model to propose tokens for a larger one to verify; those are serving tricks around the model. None of them answers the narrow structural question staring at me: why am I storing a distinct key and a distinct value for every attention head?

So what controls the cache size? The number of key heads and value heads. In the standard layer I have $h$ query heads and, to match, $h$ key heads and $h$ value heads — each head a separate learned projection, each contributing its own slice to the cache. Do I actually need $h$ key/value heads? The query I bring to a decoding step is just the current token's query — tiny, one position, not cached. It's the *keys and values* that accumulate across all past positions and get reloaded every step. The cache cost is entirely about how many distinct key/value heads I store, not how many query heads I run.

So here's the lever. Keep all $h$ query heads — they're cheap to keep and they do the representational work of attending in $h$ different ways. But collapse the key and value projections down. The extreme version: a *single* key head and a *single* value head, shared by all $h$ query heads. Every query head still computes its own attention pattern, but they all attend over the same one set of keys and one set of values.

Let me redo the count under that change and confirm it actually moves the offending term. Arithmetic is essentially unchanged — still $\Theta(b\,n\,d^2)$, the query/output projections still dominate (the key/value projections got smaller, which only helps). Memory: the cache is now $[b, n, k]$ — no $h$ — so summed over steps it's $\Theta(b\,n^2 k) = \Theta(b\,n^2 d / h)$. Plus the small reloads of activations $\Theta(b\,n\,d)$ and weights $\Theta(n\,d^2)$. New ratio:

$$\frac{b\,n\,d + b\,n^2 k + n\,d^2}{b\,n\,d^2} = \frac{1}{d} + \frac{n}{d\,h} + \frac{1}{b}.$$

The $n/d$ term has become $n/(dh)$ — reduced by a factor of $h$. That's the win, and it's exactly the factor by which I shrank the cache. Decoding gets dramatically faster, and it gets faster *because* the dominant memory term was the cache and I cut the cache by $h$.

So why isn't that the end of the story? Let me actually think about what I gave up, not just celebrate the speed. I collapsed $h$ key/value subspaces into one. Each query head used to attend through its *own* key/value projection — its own little subspace in which "what matches what" is defined. Now all $h$ query heads are forced to read and write through a single shared subspace. That's a real loss of capacity. The query heads can still differ in *how* they weight a fixed set of keys, but they can no longer disagree about *what the keys and values are*. Intuitively that should hurt quality, and it does — the single-key/value-head model underperforms the full multi-head one.

And there's a second symptom that bothers me more than a quality dip, because it suggests the cut is too sharp rather than just lossy. When I train a model this way from scratch, training is fragile — loss spikes during pre-training, and outright divergence when I fine-tune on long-input tasks, the exact tasks where the cache savings matter most. Squeezing everything through one key/value head seems to make the optimization landscape mean. So the all-or-nothing collapse isn't just a little worse; it's brittle.

Now I have a tension. Full multi-head: best quality, ruinous cache. Single shared key/value head: cheap cache, degraded and unstable. These are the two endpoints of an axis I've been treating as binary — but is it actually binary? The cache cost scaled with the *number of key/value heads*. I went from $h$ to $1$. Nothing forced that to be the only two options. What if I keep some intermediate number?

Let me see what the diagnostic says about an intermediate count, because the whole point is the cache term, and I want to know it varies smoothly. Suppose I keep $G$ key/value heads, $1 \le G \le h$. Then the cache is $[b, G, n, k]$, the cached-memory term is $\Theta(b\,G\,n^2 k) = \Theta(b\,n^2 d \cdot G/h)$, and the memory-to-compute ratio is

$$\frac{1}{d} + \frac{nG}{d\,h} + \frac{1}{b}.$$

At $G = h$ the cache term is $n/d$ — I'm back to full multi-head, no surprise. At $G = 1$ it's $n/(dh)$ — the single-head case. And in between it slides linearly: the cache, and the per-step memory traffic that dominates decoding, is simply *proportional to $G$*. So $G$ is a continuous dial between the two regimes, and it dials exactly the quantity I care about.

What does an intermediate $G$ mean structurally? I have $h$ query heads and I want them to share $G$ key/value heads. The natural thing: partition the $h$ query heads into $G$ groups of $h/G$ each, and give each group its own single key head and value head, shared within the group. So query heads in the same group attend over a common key/value subspace; query heads in different groups get different ones. $G = 1$ is one big group — all query heads share — which is the single-head collapse. $G = h$ is $h$ groups of one — every query head gets its own — which is full multi-head. The grouping *is* the interpolation, and the two methods I started with are literally its two endpoints.

Why is an intermediate $G$ not just a compromise but a genuinely good trade? A few things line up. First, capacity: instead of one shared subspace I now have $G$ of them, so the query heads aren't all forced through the same bottleneck — the representational damage of the collapse is spread over $G$ buckets rather than concentrated in one. That should recover most of the quality and, I'd bet, fix the brittleness, since the optimization no longer has to cram everything through a single head. Second, and this is the part I like, the trade *scales the right way*. Bigger models use more heads $h$. The single-head collapse is a fixed cut to one head regardless of $h$, so as $h$ grows it becomes a relatively more aggressive amputation of capacity. But if I instead fix the *proportion* — say $G$ a fixed fraction of $h$, or just a fixed modest number like $G=8$ — then as the model grows I keep cutting the cache by the same proportional factor $h/G$ while the per-group capacity grows with the model. The reduction in bandwidth and the reduction in capacity stay matched as I scale.

There's a sharper version of why intermediate $G$ is nearly free on the speed side for the large models I actually deploy. The cache scales with $d$ (it's $b\,G\,n\,k$ and $G k \le d$), but the model's parameters and FLOPs scale with $d^2$. So as the model gets bigger, the cache term shrinks *relative* to the compute term — large models are less dominated by the cache to begin with. Which means moving $G$ up from $1$ toward $h$ only slowly re-introduces the bandwidth cost: near $G=1$ the cache is a small slice of the total time, so a few extra key/value heads barely move the wall clock; it's only as $G$ approaches $h$ that the cache term grows back to dominate. So the speed-versus-$G$ curve is flat near the cheap end and steep near the expensive end — I can buy a lot of capacity back from $G=1$ for almost no time, which is exactly the shape I want. Something like $G = 8$ sits in the flat region: near the single-head speed, but with eight subspaces instead of one.

And there's a sharding subtlety that quietly favours $G$ over $1$. When I shard a big model across, say, $P$ accelerators, each partition wants its own copy of the key/value heads to work with. With a single shared head and $P$ partitions, that one head gets *replicated* $P$ times across the partitions — so the "single head" isn't really paying off once it drops below the partition count; I'm storing copies. If instead I pick $G$ around the number of partitions, each partition naturally owns a distinct key/value head and there's no wasteful replication. The grouping lines up with the hardware.

One scope check. This bandwidth pain is specifically a *decoding* problem — it comes from the sequential, one-token-at-a-time loop where each step reloads the cache. The encoder processes its whole input in parallel; it's compute-bound, not bandwidth-bound, and it has no growing cache to stream. So there's no reason to pay the capacity cost of grouping there. Apply the grouping to the decoder's self-attention and to the cross-attention (both run inside the sequential decode loop), but leave the encoder's self-attention as plain multi-head. Free quality, no speed cost.

So I think the architecture is settled: partition the $h$ query heads into $G$ groups, one key head and one value head per group, shared within the group, cache proportional to $G$, with $G$ a small number that scales sensibly. Now the harder, more practical question, and honestly the one that decides whether any of this gets used.

There are a lot of excellent multi-head checkpoints already trained, at enormous cost, that I'd love to serve cheaply — but they were trained with $h$ key/value heads, and my grouped model has $G$. The naive answer is "train the grouped model from scratch," and that's a non-starter: I'm not going to spend a full pre-training run just to get a faster-decoding variant of a model I already have. So can I *convert* an existing multi-head checkpoint into a grouped one cheaply, re-using everything I can?

What carries over untouched? The query projections — same $h$ heads. The output projection — same. Everything outside attention — embeddings, feed-forward, norms — same. The *only* thing that structurally changes is the key and value projections: I had $h$ of each, I now want $G$ of each. So the entire conversion problem reduces to: given the $h$ trained key projection matrices in a group, produce *one* key projection matrix for that group (and likewise for values). It's a surgical edit to just the key/value projections.

How should I build the one group projection from the $h/G$ trained head projections in that group? Let me think about what each option destroys. I could just *pick* one of the heads in the group and throw the rest away — but then I've discarded the learned content of $h/G - 1$ heads; the surviving head's subspace is arbitrary and the others' information is gone. I could *randomly initialize* a fresh key/value head for each group — but then I've thrown away *all* the trained key/value structure and I'm asking continued training to relearn it; that's the most destructive option, closest to training from scratch. Both of those waste information that's sitting right there in the checkpoint.

What I want is the *single* projection that best stands in for the whole group of trained projections — that preserves as much of their collective behaviour as possible. Each head's key projection is a linear map $W_k^{(j)}$ from the residual stream to that head's key space. If I'm going to replace the group by one map, the least-committal, most-information-preserving choice is the *average* of the maps in the group:

$$W_k^{g} = \frac{1}{|g|} \sum_{j \in g} W_k^{(j)}, \qquad W_v^{g} = \frac{1}{|g|} \sum_{j \in g} W_v^{(j)},$$

i.e. mean-pool the key projection matrices of the heads in the group, and the value projection matrices likewise. Why the mean specifically? Because it keeps the *average* response of the group's heads intact — the shared, common component of what those heads were computing survives the merge, and only the head-to-head *differences* within a group are lost. Picking one head keeps that one head's idiosyncrasies and loses the average; the mean keeps the average and loses the idiosyncrasies — and the average is the better thing to keep, because it's the part all the grouped query heads can agree on. I'd order the three by how much trained information they preserve: mean-pool best, select-one-head next, random-init worst. And the grouping makes the mean sensible: I'm only averaging the $h/G$ heads that are about to share a subspace anyway, not all $h$, so within a group the heads I'm merging are the ones I've *chosen* to make share.

Notice the endpoints fall out cleanly here too. At $G = h$ each group is a single head, the "mean" is over one element — it's the identity, the conversion is a no-op, and I keep the exact multi-head checkpoint. At $G = 1$ the mean is over all $h$ heads — the single-head conversion. So the same mean-pool recipe converts to any $G$, and the grouped case is just the natural in-between.

After the surgery the model isn't quite itself — the averaged key/value heads don't perfectly match what the query heads and the rest of the network expect, because those were tuned against $h$ distinct heads. But the damage should be much smaller than random reinitialization: most weights are untouched and the key/value heads start from a sensible average rather than from noise. So I don't need to start over; I need to let the model *adapt* to its new attention structure. Continue pre-training — same recipe, same data, same objective as the original — for a small fraction $\alpha$ of the original number of steps. How small? Enough that the network can re-coordinate the averaged heads with the unchanged query and output projections, but not so much that I've paid for another pre-training run. A few percent is the right scale for a surgical edit; something like $\alpha = 0.05$ is a reasonable default, and a larger value such as $0.10$ is a check on whether adaptation is still the bottleneck. And I'd expect the grouped model to need less of this adaptation than the single-head one: the single-head conversion is the violent merge, all $h$ heads into one, whereas the grouped conversion preserves more structure before any adaptation steps at all.

So the full recipe is two moves: convert the checkpoint by mean-pooling each group's key and value projections into one, then continue pre-training briefly to let it adapt. The architecture is the grouping; the cheap-to-obtain part is the conversion. Together they should let me take a trained multi-head model and get a faster-decoding grouped model for a few percent of the original compute, with capacity loss controlled by $G$ instead of forced all the way to one shared subspace.

Let me write the code, grounding it in how a real decoder attention layer is structured. The architecture first. I have $H$ query heads and a smaller number of key/value heads; call it $G$ groups, with group size $H/G$. The query projection outputs $H$ heads' worth; the key and value projections output only $G$ heads' worth — that's the whole structural change, and it's why the cache is $G$ times smaller. Then, to form the attention scores, each of the $G$ key/value heads has to serve its $H/G$ query heads, so I expand the key/value heads back up to $H$ by repeating each one $H/G$ times — repeat, not re-project, so I store $G$ in the cache but compute against $H$:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [b, G, s, k]  ->  [b, G*n_rep, s, k]
    # Each of the G key/value heads is repeated n_rep = H/G times so it
    # serves its group of query heads. Only G heads live in the cache;
    # the expansion happens after loading, just before the score matmul.
    b, G, s, k = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(b, G, n_rep, s, k)
    return x.reshape(b, G * n_rep, s, k)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads          # H query heads
        self.num_kv_heads = num_kv_heads     # G key/value heads (groups)
        self.num_kv_groups = num_heads // num_kv_heads   # group size = H/G = n_rep
        self.head_dim = hidden_size // num_heads

        # Queries: full H heads. Keys/Values: only G heads -> cache is G/H the size.
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x, kv_cache=None, mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:                  # the thing we stream each decode step
            k, v = kv_cache.append(k, v)          # cache holds only G heads

        # Expand G kv-heads up to H to match the query heads (no extra cache cost).
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)
```

And the conversion — the surgical edit that takes a trained multi-head layer and produces the grouped one by mean-pooling within each group. The query and output projections copy straight over; only key and value change. I reshape the $H$ trained heads into $G$ groups of $H/G$, average over the within-group head axis, and that average is the group's single key/value head:

```python
def convert_mha_to_gqa(mha, num_kv_heads):
    H = mha.num_heads
    G = num_kv_heads
    rep = H // G                              # heads per group
    Dh = mha.head_dim

    gqa = GroupedQueryAttention(mha.q_proj.in_features, H, G)
    # Unchanged structure: copy directly.
    gqa.q_proj.weight.data.copy_(mha.q_proj.weight.data)
    gqa.o_proj.weight.data.copy_(mha.o_proj.weight.data)

    # Key/value: mean-pool the rep trained head projections in each group
    # into one. weight rows are [head, head_dim, in_features]; average over
    # the head axis within each group.
    def mean_pool(weight):                    # weight: [H*Dh, in]
        w = weight.view(G, rep, Dh, -1)        # [G, rep, Dh, in]
        return w.mean(dim=1).reshape(G * Dh, -1)  # [G*Dh, in]: one head per group

    gqa.k_proj.weight.data.copy_(mean_pool(mha.k_proj.weight.data))
    gqa.v_proj.weight.data.copy_(mean_pool(mha.v_proj.weight.data))
    return gqa
    # G == H: rep == 1, mean over one element -> identity (still MHA).
    # G == 1: mean over all H heads -> single shared kv head.
```

```python
def uptrain(model, pretrain_step_fn, original_steps, alpha=0.05):
    # After conversion the averaged kv-heads don't perfectly fit the rest of
    # the network; run the ORIGINAL pre-training recipe/data for a small
    # fraction alpha of the original steps to let the model adapt. Cheap:
    # ~5% of a pre-training run, vs. a full run from scratch.
    for step in range(int(alpha * original_steps)):
        pretrain_step_fn(model)
    return model
```

The causal chain, start to end: autoregressive decoding is memory-bound, and the diagnostic pins the bottleneck to one term — the $n/d$ in the memory-to-compute ratio — which is the cost of streaming a key/value cache that carries one key and one value per head. Collapsing to a single shared key/value head cuts that term by the head count $h$ and makes decoding fast, but forcing all query heads through one subspace costs quality and destabilizes training. Since the cache cost is simply proportional to the number of key/value heads, that number is a continuous dial: partition the query heads into $G$ groups sharing $G$ key/value heads, with the single-head and full multi-head cases as the $G=1$ and $G=h$ endpoints — capacity spread over $G$ subspaces, cache cut by $h/G$, and the speed curve flat near the cheap end so a modest $G$ buys back most of the quality for almost no time. And because only the key/value projections change, an existing multi-head checkpoint converts by mean-pooling each group's trained key/value projections into one — the most information-preserving merge — followed by a few percent of continued pre-training to adapt, yielding a fast model at near-multi-head quality without paying for a fresh pre-training run.
