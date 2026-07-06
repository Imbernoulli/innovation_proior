Let me take stock of where the single-GPU ladder has landed and what wall every rung has been pushing
against. Compile + static cache got the device running flat-out at ~1397 GB/s, ~70% of one A100's HBM
peak. int8 and int4 then shrank the bytes per token — 1397 down through 1069 to 862 GB/s achieved, with
tokens/s rising because bytes/token fell — and speculative decoding amortized the verifier's weight read
over several tokens, pushing a 70B-int4 verifier drafted by a 7B-int4 to 48.4 tok/s, a rate a 70B could
never reach un-speculated. Every one of those wins is a different way of saying the same thing: the
bottleneck is *one GPU's memory bandwidth*, and I've been fighting it by reducing demand on it — fewer
bytes per token, or fewer passes per token. But there is a complementary move I haven't made — instead of
asking less of one HBM, use *more HBM in parallel*. The 70B model can't even fit on a single 80 GB card in
bf16 (that's ~140 GB, an instant OOM; only int4 at ~35 GB squeezes it on), and for the 7B I'm
bandwidth-bound on a single device by definition. A second A100 doubles the aggregate memory bandwidth
available; eight of them give eight times. If I can spread one token's weight read across `N` cards so each
reads `1/N` of the weights *simultaneously*, the effective time to stream the model drops by `N`, and a
bandwidth-bound decode should speed up close to linearly in `N` — exactly the regime where throwing
hardware at the wall actually works, because the wall is bandwidth and bandwidth is what more cards buy.

Let me be quantitative about the target before designing the mechanism. For the 7B at bf16, one GPU streams
13.5 GB per token; the achieved-bandwidth story says a token takes ~9.5 ms at ~1397 GB/s. Split across `N`
cards each reading `13.5/N` GB simultaneously, the per-token read time should fall toward `9.5/N` ms if the
cards truly work in parallel — so the ceiling scaling is linear in `N`, and the only thing that can spoil
it is time the cards spend *not* streaming weights: communication and synchronization. So the whole design
question reduces to: how do I split the model so each GPU does `1/N` of the work on every token, while
keeping the inter-card communication small enough that it doesn't eat the parallel-bandwidth win?

I walk the three ways to use multiple GPUs, because only one of them helps this particular workload. Data
parallelism — replicate the whole model on each card and send different requests to each — raises aggregate
*throughput* across many requests but does nothing for a *single* request at batch 1: one card still
streams the whole 13.5 GB for my one token, at the same latency as before. Off the table, since the metric
is single-stream tok/s. Pipeline parallelism — put layers 0-15 on GPU 0 and 16-31 on GPU 1 — is the naive
"split the model" and it's wrong here: at batch 1 the token flows through the layers serially, so GPU 1
sits idle while GPU 0 works and vice versa. That doesn't add parallel bandwidth, it just relays; the
per-token latency is unchanged (still one full pass through all layers, now with a hop in the middle), and
the aggregate bandwidth is wasted because only one card streams at a time. Pipelining only helps when you
can fill the pipe with many in-flight tokens, which batch 1 forbids. The one that works is splitting
*within* each layer — **tensor parallelism** — so all `N` cards work the same layer at the same time, each
on its own slice of the matrices, and every card is streaming its shard of the weights simultaneously on
every layer of every token. That's the only decomposition that turns `N` cards into `N×` the bandwidth for
a single token.

Now derive the split from the structure of a transformer layer, because the layer's algebra dictates which
dimension to shard and where the communication is unavoidable. Take the feed-forward first, the simpler
case. It's `w2(silu(w1(x)) * w3(x))`: project up from `dim=4096` to `intermediate=11008` with `w1` and
`w3`, gate elementwise, project back down to 4096 with `w2`. The intermediate dimension is the one to
split — it's the big internal axis and the natural place to cut. If I shard `w1` and `w3` **column-wise**
(each GPU holds a contiguous slice of the output/intermediate columns, so `11008/N` columns each), then
each GPU computes its own slice of the intermediate activation `silu(w1(x)) * w3(x)` entirely locally — the
gating is elementwise, so column `j` of the gated intermediate depends only on column `j` of `w1(x)` and
column `j` of `w3(x)`, no cross-GPU dependence. Then `w2` projects the `11008`-wide intermediate back to
4096; shard it **row-wise** (each GPU holds the input rows matching its intermediate slice, `11008/N` rows
each), and each GPU produces a *partial sum* of the 4096-wide output over its slice of the intermediate.
The full output is the sum of the `N` partials, so the layer ends in one **all-reduce** across the GPUs.

Let me verify the colwise-then-rowwise pairing actually reconstructs the right answer, by writing the sum
out, because if the partials don't add to `w2 · (gated intermediate)` the whole scheme is wrong.
Column-sharding `w1`/`w3` gives GPU `r` the intermediate slice `h_r = silu(w1_r x) * w3_r x`, which is
columns `r·(11008/N) : (r+1)·(11008/N)` of the full intermediate `h`. Row-sharding `w2` gives GPU `r` the
rows of `w2` matching those same columns, i.e. `w2_r` is the `4096 × (11008/N)` block that multiplies
exactly `h_r`. So GPU `r` computes `w2_r · h_r`, a full-width `4096` vector that is the partial contribution
of its intermediate slice. Summing over `r`: `Σ_r w2_r · h_r = w2 · h` — because concatenating the column
blocks of `w2` and stacking the slices of `h` is exactly the block decomposition of the matrix-vector
product. The partials sum to the true output. That's why column-wise *then* row-wise is precisely the
pairing that works: it pushes the one unavoidable sync to the very end and keeps everything before it local.
Any other pairing (row-then-row, or col-then-col) would either need a sync *in the middle* of the FFN or
fail to reconstruct the product.

```python
def _apply_tp_ffn(mlp):
    _apply_tp_linear(mlp.w1, "colwise")
    _apply_tp_linear(mlp.w3, "colwise")
    _apply_tp_linear(mlp.w2, "rowwise")
    world_size = _get_world_size()
    mlp.register_forward_hook(lambda _m, _i, output:
        funcol.all_reduce(output, "sum", list(range(world_size))))
```

Attention shards along the same seam, with the heads as the natural unit. The QKV projection produces
per-head queries, keys, values; split it column-wise so each GPU owns a disjoint subset of the attention
heads and computes their attention entirely on its own — each head is independent (its softmax is over its
own query-key dot products, no cross-head mixing inside attention), so `32/N` heads per card attend
locally. The output projection `wo` mixes the concatenated head outputs back to the model dimension; shard
it row-wise so each GPU contributes the partial output of its heads, and again the layer closes with one
all-reduce — the identical column-then-row structure, with "head" playing the role "intermediate column"
played in the FFN. The QKV split has one wrinkle: it has to keep query/key/value heads grouped correctly
when sharding. Llama-2-70B uses grouped-query attention with only 8 key/value heads (against 64 query
heads), so I can't just cut the stacked QKV matrix into `N` even pieces — I have to split Q, K, V
*separately* by their own head counts and recombine, or a shard would end up with query heads whose
matching key/value heads live on another card. And the per-GPU head/dim bookkeeping (`n_head`, `dim`,
`head_dim`, `n_local_heads`) is divided by the world size so the KV-cache and rotary embeddings are sized
for the local shard, not the global model:

```python
def _apply_tp_attn(attn):
    kv_size = attn.n_local_heads * attn.head_dim
    _apply_tp_linear(attn.wqkv, "colwise", [attn.dim, kv_size, kv_size])
    _apply_tp_linear(attn.wo, "rowwise")
    world_size = _get_world_size()
    attn.n_head = attn.n_head // world_size
    attn.dim = attn.dim // world_size
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = attn.n_local_heads // world_size
    attn.register_forward_hook(lambda _m, _i, output:
        funcol.all_reduce(output[0], "sum", list(range(world_size))))
```

The sharding of a single linear is the core primitive underneath both. For a column-wise shard I split the
weight's output dimension across ranks; for row-wise, the input dimension. The weight matrix is stored
transposed as `(out_features, in_features)`, so column-wise is `dim 0` and row-wise is `dim 1`, and each
rank keeps only its `tensor_split(...)[rank]` slice — so each GPU literally loads and reads `1/N` of that
matrix's bytes, which is the whole point: the per-GPU weight read shrinks by `N`, and shrinking the weight
read is what every rung of this ladder has been about, now achieved by division across cards rather than by
quantization within one:

```python
def _apply_tp_linear(linear, style, weight_splits=[]):
    rank, world_size = _get_rank(), _get_world_size()
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    shard_dim, size_attr = dim_lookup[style]
    def shard(x, dim):
        return torch.tensor_split(x, world_size, dim=dim)[rank]
    # (qkv is sharded per-component when weight_splits is given; int4 weights shard their
    #  packed dim and scales_and_zeros correspondingly)
    sharded_weight = shard(linear.weight, shard_dim)
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)
```

Let me run the shapes through concretely for the 7B at `N=8` to make sure the divisions land on integers
and the per-shard model is coherent, because a ragged split would break the matmuls. Global `dim=4096`,
`n_head=32`, `head_dim=128`, `intermediate=11008`. Per shard: `dim` stays 4096 as the *activation* width
(each card still produces a full 4096-wide partial), but the sharded `w1`/`w3` output `11008/8 = 1376`
intermediate columns each, `w2` takes `1376` input rows each; attention gets `32/8 = 4` query heads per
card, `4 × 128 = 512`-wide local QKV output. Every one of those divides evenly — 11008/8 = 1376, 32/8 = 4 —
which isn't an accident: the model dims are chosen as multiples of small powers of two precisely so they
shard cleanly at 2, 4, 8 cards, and the `assert getattr(linear, size_attr) % world_size == 0` in the shard
primitive is there to catch the case where they don't. The one dimension I must *not* naively divide is the
70B's KV heads: with grouped-query attention it has only 8 KV heads, so at `N=8` each card gets exactly 1
KV head, and at `N>8` the even split would fail — which is the hard reason the QKV shard goes per-component
(`weight_splits=[dim, kv_size, kv_size]`) rather than slicing the stacked matrix, and the reason 8 cards is
a natural cap for the 70B beyond just the NVLink-node boundary.

That per-shard bookkeeping is also what makes `setup_caches` allocate correctly after sharding. Because
`_apply_tp_Transformer` divides `config.n_head` and `config.n_local_heads` by the world size *before* the
caches are built, each card's `KVCache` is sized for its local `n_local_heads` — so the KV-cache is itself
sharded across cards (each holds the keys/values for its own heads), not replicated, which keeps the cache
memory per card at `1/N` and lets the 70B's cache fit alongside its `1/8` weight shard. The rotary table is
sized from the local `head_dim` for the same reason. So sharding isn't only about the weight matrices; the
whole decode state — weights, KV-cache, rotary — is divided so each card's footprint and each card's
per-token read are both `1/N` of the global model.

And the top-level application walks the model, fixes the config dims so `setup_caches` allocates per-shard
KV-caches, and shards every FFN and attention block:

```python
def _apply_tp_Transformer(model):
    world_size = _get_world_size()
    model.config.n_head = model.config.n_head // world_size
    model.config.dim = model.config.dim // world_size
    model.config.n_local_heads = model.config.n_local_heads // world_size

def apply_tp(model):
    _apply_tp_Transformer(model)
    for block in model.layers:
        _apply_tp_ffn(block.feed_forward)
        _apply_tp_attn(block.attention)
```

The all-reduce lives in a `register_forward_hook` on the FFN and attention modules rather than being
written into their `forward`, and that placement is deliberate: it fires *after* the sub-layer's own
forward returns its partial output, sums across ranks, and hands back the reduced tensor, without my having
to edit the frozen module bodies — it slots the collective in at exactly the seam the algebra says it
belongs (the end of the column-then-row pair) while leaving the module's internal computation untouched.
For attention the hook reduces `output[0]` because the attention module returns a tuple; for the FFN it
reduces the bare output. Registering it as a hook also means it composes with the compiled decode graph:
the collective is a captured op in the CUDA graph like any other, so I keep the scheduling win from the
compile rung even with communication in the loop, as long as the collective is graph-capturable (which the
functional-collectives `funcol.all_reduce` is).

The composition with int4 weights is worth spelling out because it's where the shard primitive earns its
complexity. A quantized linear doesn't just have a `weight` to split — it has `scales_and_zeros` (int4) or
`scales` (int8) that must be split *consistently* with the weight, or the dequant would pair the wrong
scales with the wrong weight rows. For int4, the packed weight is split along its packed dimension while
`scales_and_zeros` is split along `1 - shard_dim` (the complementary axis), and when the QKV split is
per-component the packed weight splits use `[i // 8 for i in weight_splits]` because the int4 packing folds
8 values into each stored unit, so the split boundaries in packed space are the head-boundary sizes divided
by the packing factor. Getting that index arithmetic right is what lets a card load `1/N` of the *int4*
weight and its matching scales and dequant correctly — so a sharded int4 70B reads `~35/8 ≈ 4.4 GB` per
card per token, and the tensor-parallel and quantization byte reductions multiply. That's the fuller shard
primitive; the simplified version above shows the bf16 path, and the packed-weight branches handle the
quantized shards on top of it.

Now cost the communication I'm trading against the parallel bandwidth, because that's the one thing that
can spoil the linear scaling and I want to know how big it is. There are two all-reduces per layer (one
after attention, one after the FFN), so 64 syncs per token across 32 layers. What's the *volume* of each?
At batch 1 in decode, the activation being reduced is the `dim=4096` output vector — `4096 × 2 B ≈ 8 KB`
per all-reduce. A ring all-reduce moves about `2(N−1)/N` times the message size per rank, so on the order
of `~16 KB` of traffic per rank per all-reduce, times 64 per token ≈ ~1 MB of communication per token.
Compare that to what it buys: at `N=8` the per-GPU weight read for the 7B drops from 13.5 GB to `~1.7 GB`,
which at 2 TB/s is `~0.85 ms`. Moving ~1 MB over NVLink at hundreds of GB/s is single-digit microseconds
plus latency — so the communication is a small fraction of the parallelized weight read, which is exactly
why this scales well *within* a node where NVLink's inter-GPU bandwidth is high. It's also why the
benchmark caps at 8×A100 on one hybrid-cube-mesh box: cross-node interconnect is far slower than NVLink, so
the all-reduces would stop being cheap and the scaling would collapse the moment I left the node.

But I should be honest that the scaling is strong, not perfect, and the derivation shows exactly why. As
`N` grows the per-GPU weight slice shrinks like `1/N`, but the all-reduce cost does *not* shrink — the
message is the fixed `dim`-wide activation regardless of `N`, and the ring-reduce term `2(N−1)/N` actually
*grows* slightly toward `2` as `N` increases, and latency per collective is roughly constant. So the
communication becomes a larger *fraction* of the (shrinking) per-GPU compute as `N` grows, and the marginal
card buys less than the one before it. Concretely I expect near-linear scaling at small `N` bending
sub-linear as `N` climbs — the second card should nearly double throughput, but the eighth adds
proportionally less. This composes cleanly with the quantization rungs, too: each shard can itself be int8
or int4 (the shard logic already handles the packed weights and their scales, splitting the packed
dimension and `scales_and_zeros` correspondingly), so tensor parallelism multiplies whatever single-GPU
byte reduction I've already banked. And crucially, tensor parallelism changes *nothing* about the numerics
— it's an exact re-expression of the same matmuls split across cards, the partials provably summing to the
unsharded product as I verified above — so like compile and unlike the quantization rungs it preserves the
output distribution exactly, no accuracy check needed.

This is the closing move, so let me set the bar it has to clear and why I believe it can. The standing
single-A100 record for the 7B is the compile + static-KV-cache "Base" at 104.9 tok/s — that's the
apples-to-apples bf16 number a second card has to beat, and I hold the comparison at bf16 so the
parallelism win is isolated from the quantization wins already measured. The bet is that because decode is
bandwidth-bound, splitting each token's weight read across `N` NVLinked A100s multiplies the aggregate
bandwidth and drives throughput up nearly linearly: I'd expect 2 GPUs to reach roughly 1.6× the single-GPU
104.9 and 8 GPUs to push it several-fold higher, with the all-reduce overhead bending the curve sub-linear
as `N` grows exactly as the communication accounting predicts. The two hedges are real and are precisely
the ones the derivation exposed: the 64 all-reduces per token are pure overhead that doesn't exist on one
GPU, so on a slow interconnect this could fail to scale or even regress — it *relies* on NVLink; and the
per-GPU work shrinks while the sync stays fixed, so the marginal card buys less each time. The other thing
tensor parallelism unlocks that no single-GPU rung can is the 70B model *at all* in bf16: it doesn't fit in
80 GB, so without sharding its single-GPU bf16 throughput is undefined (OOM), and only by splitting it
across 8 cards does it become servable — the sharded weights fit because each card holds `1/8` of them.
What I'd validate is the realized tokens/s at 2, 4, and 8 GPUs against the 104.9 single-GPU baseline for
the 7B, and the 8-GPU throughput for the 70B that otherwise can't run at all. If splitting the weight read
across NVLinked cards scales the way a bandwidth-bound workload should, tensor parallelism is the lever
that takes this past every single-device record and makes the largest models servable, and that is where
the ladder ends.
