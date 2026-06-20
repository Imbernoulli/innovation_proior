Let me take stock of where the single-GPU ladder has landed and what wall every rung has been pushing
against. Compile + static cache got the device running flat-out at ~1397 GB/s, ~70% of one A100's HBM
peak. int8 and int4 then shrank the bytes per token, and speculative decoding amortized the verifier's
weight read over several tokens. Every one of those wins is a different way of saying the same thing: the
bottleneck is *one GPU's memory bandwidth*, and I've been fighting it by reducing demand on it. But there
is a complementary move I haven't made — instead of asking less of one HBM, use *more HBM in parallel*.
The 70B model can't even fit on a single 80 GB card (it OOMs in bf16), and for the 7B I'm bandwidth-bound
on a single device by definition. A second A100 doubles the aggregate memory bandwidth available; eight of
them give eight times. If I can spread one token's weight read across N cards so each reads 1/N of the
weights *simultaneously*, the effective time to stream the model drops by N, and a bandwidth-bound decode
should speed up close to linearly in N — exactly the regime where throwing hardware at the wall actually
works, because the wall is bandwidth and bandwidth is what more cards buy.

The question is how to split the model so each GPU does 1/N of the work on every token without forcing the
cards to wait on each other. The naive split — put layers 0-15 on GPU 0 and 16-31 on GPU 1 (pipeline
parallelism) — is wrong here: at batch 1 the token flows through the layers serially, so GPU 1 sits idle
while GPU 0 works and vice versa. That doesn't add parallel bandwidth, it just relays. I need every GPU
busy on *every* layer of *every* token. That means splitting *within* each layer — **tensor
parallelism** — so all N cards work the same layer at the same time, each on its own slice of the
matrices.

Now derive the split from the structure of a transformer layer, because the layer's algebra dictates which
dimension to shard and where the communication is unavoidable. Take the feed-forward first, the simpler
case. It's `w2(silu(w1(x)) * w3(x))`: project up to the intermediate dimension with `w1` and `w3`, gate,
project back down with `w2`. The intermediate dimension is the one to split. If I shard `w1` and `w3`
**column-wise** (each GPU holds a contiguous slice of the output/intermediate columns), then each GPU
computes its own slice of the intermediate activation `silu(w1(x)) * w3(x)` entirely locally — the gating
is elementwise, no cross-GPU dependence. Then `w2` projects the intermediate back to the model dimension;
shard it **row-wise** (each GPU holds the input rows matching its intermediate slice), and each GPU
produces a *partial sum* of the output over its slice of the intermediate. The full output is the sum of
the partials, so the layer ends in one **all-reduce** across the GPUs. That all-reduce is the only
communication the FFN needs — column-wise then row-wise is precisely the pairing that pushes the sync to
the very end and keeps everything before it local:

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
heads and computes their attention entirely on its own (each head is independent — no cross-head
dependence inside attention). The output projection `wo` mixes the concatenated head outputs back to the
model dimension; shard it row-wise so each GPU contributes the partial output of its heads, and again the
layer closes with one all-reduce. The QKV split has to keep query/key/value heads grouped correctly when
sharding — Llama-2-70B uses grouped-query attention with only 8 key/value heads, so the shard logic splits
Q, K, V separately by their own head counts and recombines — and the per-GPU head/dim bookkeeping
(`n_head`, `dim`, `head_dim`, `n_local_heads`) is divided by the world size so the KV-cache and rotary
embeddings are sized for the local shard:

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

The sharding of a single linear is the core primitive. For a column-wise shard I split the weight's
output dimension across ranks; for row-wise, the input dimension. The weight matrix is stored transposed
as `(out_features, in_features)`, so column-wise is `dim 0` and row-wise is `dim 1`, and each rank keeps
only its `tensor_split(...)[rank]` slice — so each GPU literally loads and reads 1/N of that matrix's
bytes, which is the whole point: the per-GPU weight read shrinks by N:

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

And the top-level application walks the model, fixes the config dims so `setup_caches` allocates
per-shard KV-caches, and shards every FFN and attention block:

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

The cost I'm trading against the parallel bandwidth is the two all-reduces per layer (one after attention,
one after the FFN), 64 syncs per token across 32 layers. On a single node with NVLink the inter-GPU
bandwidth is high enough that these are cheap relative to the weight read they let me parallelize, which is
why this scales well *within* a node and the benchmark caps at 8×A100 on one hybrid-cube-mesh box. As N
grows the all-reduce volume grows and the per-GPU weight slice shrinks, so the communication eventually
eats into the parallel-bandwidth win — the scaling is strong but sub-linear, and it composes cleanly with
the quantization rungs (each shard can itself be int8 or int4, the shard logic already handles the packed
weights and scales). Crucially, tensor parallelism changes *nothing* about the numerics — it's an exact
re-expression of the same matmuls split across cards, so like compile and unlike the quantization rungs it
preserves the output distribution exactly.

This is the closing move, so let me set the bar it has to clear and why I believe it can. The standing
single-A100 record for the 7B is the compile + static-KV-cache "Base" at 104.9 tok/s — that's the
apples-to-apples bf16 number a second card has to beat. The bet is that because decode is bandwidth-bound,
splitting each token's weight read across N NVLinked A100s multiplies the aggregate bandwidth and drives
throughput up nearly linearly: I'd expect 2 GPUs to roughly 1.6× the single-GPU 104.9 and 8 GPUs to push
it several-fold higher, with the all-reduce overhead bending the curve sub-linear as N grows. The two
hedges are real and exactly the ones the derivation exposed: the 64 all-reduces per token are pure
overhead that doesn't exist on one GPU, so on a slow interconnect this could fail to scale or even regress
— it relies on NVLink; and the per-GPU work shrinks while the sync grows, so the marginal card buys less
each time. The other thing tensor parallelism unlocks that no single-GPU rung can is the 70B model at all:
it doesn't fit in 80 GB, so without sharding its single-GPU throughput is undefined (OOM), and only by
splitting it across 8 cards does it become servable. What I'd validate is the realized tokens/s at 2, 4,
and 8 GPUs against the 104.9 single-GPU baseline for the 7B, and the 8-GPU throughput for the 70B that
otherwise can't run at all. If splitting the weight read across NVLinked cards scales the way a
bandwidth-bound workload should, tensor parallelism is the lever that takes this past every single-device
record and makes the largest models servable, and that is where the ladder ends.
