Every rung of the single-GPU ladder has been a different way of fighting the same wall. Compile plus the static cache got the device running flat-out at ~1397 GB/s, ~70% of one A100's HBM peak; int8 and int4 shrank the bytes per token; speculative decoding amortized the verifier's weight read over several tokens. Each win reduces *demand* on one GPU's memory bandwidth. The complementary move I have not made is to add *more* HBM in parallel. The 70B model cannot even fit on a single 80 GB card — it OOMs in bf16 — and for the 7B I am bandwidth-bound on one device by definition. A second A100 doubles the aggregate memory bandwidth; eight give eight times. If one token's weight read is spread across $N$ cards so each reads $1/N$ of the weights *simultaneously*, the time to stream the model drops by $N$, and a bandwidth-bound decode should speed up close to linearly in $N$ — the one regime where throwing hardware at the wall works, because the wall is bandwidth and bandwidth is what more cards buy.

The question is how to split the model so each GPU does $1/N$ of every token's work without the cards waiting on each other. The naive split — layers 0–15 on GPU 0, 16–31 on GPU 1 (pipeline parallelism) — is wrong at batch 1: the token flows through layers serially, so each card sits idle while the other works. That relays the token; it does not add parallel bandwidth. I need every GPU busy on *every* layer of *every* token, which means splitting *within* each layer. I propose **tensor parallelism**: shard each layer's matrices across ranks so all $N$ cards work the same layer at the same time, each on its own slice, and each reads only $1/N$ of the weights per token.

The split is dictated by the layer's algebra. Take the feed-forward, $w_2(\operatorname{silu}(w_1(x)) * w_3(x))$ — project up to the intermediate dimension, gate, project back down. Shard $w_1$ and $w_3$ **column-wise**: each GPU holds a contiguous slice of the intermediate columns and computes its own slice of $\operatorname{silu}(w_1(x)) * w_3(x)$ entirely locally, because the gating is elementwise with no cross-GPU dependence. Then shard $w_2$ **row-wise**: each GPU holds the input rows matching its intermediate slice and produces a *partial sum* of the output. The full output is the sum of the partials, so the sub-layer ends in one **all-reduce**. The column-then-row pairing is precisely what pushes the only synchronization to the very end and keeps everything before it local. Attention shards along the same seam with the heads as the natural unit: shard the QKV projection column-wise so each GPU owns a disjoint subset of attention heads and computes their attention on its own (heads are independent), and shard the output projection $w_o$ row-wise so each GPU contributes the partial output of its heads, closing again with one all-reduce. The QKV split must keep query/key/value heads grouped correctly — Llama-2-70B uses grouped-query attention with only 8 KV heads, so $Q$, $K$, $V$ are split separately by their own head counts and recombined — and the per-GPU bookkeeping (`n_head`, `dim`, `head_dim`, `n_local_heads`) is divided by the world size so the KV-cache and rotary embeddings are sized for the local shard. The core primitive underneath both is sharding a single linear: the weight is stored transposed as `(out_features, in_features)`, so column-wise is `dim 0` and row-wise is `dim 1`, and each rank keeps only its `tensor_split(...)[rank]` slice — literally loading and reading $1/N$ of that matrix's bytes, which is the whole point.

The cost traded against the parallel bandwidth is two all-reduces per layer — one after attention, one after the FFN — 64 syncs per token across 32 layers. On a single node with NVLink the inter-GPU bandwidth is high enough that these are cheap relative to the weight read they let me parallelize, which is why this scales well *within* a node and the benchmark caps at 8×A100 on one box. As $N$ grows the all-reduce volume grows while the per-GPU weight slice shrinks, so communication eventually eats into the win and the scaling is strong but sub-linear. It composes cleanly with the quantization rungs — each shard can itself be int8 or int4, and the shard logic already handles the packed weights and scales. Crucially, tensor parallelism changes *nothing* about the numerics: it is an exact re-expression of the same matmuls split across cards, so like compile and unlike the quantization rungs it preserves the output distribution exactly.

This is the closing move, so the bar is set against the standing single-A100 record for the 7B — the compile + static-KV-cache "Base" at 104.9 tok/s, the apples-to-apples bf16 number a second card must beat. The bet is that splitting each token's weight read across $N$ NVLinked A100s multiplies aggregate bandwidth and drives throughput up nearly linearly, with the all-reduce overhead bending the curve sub-linear: roughly $1.6\times$ at 2 GPUs and several-fold by 8. The hedges are exactly the ones the derivation exposes — the 64 all-reduces per token are pure overhead that does not exist on one GPU, so on a slow interconnect this could fail to scale or regress; it relies on NVLink, and the marginal card buys less each time. What tensor parallelism unlocks that no single-GPU rung can is the 70B *at all*: it does not fit in 80 GB, so its single-GPU throughput is undefined (OOM), and only by splitting it across 8 cards does it become servable. Validating the realized tokens/s at 2, 4, and 8 GPUs against the 104.9 baseline for the 7B, and the 8-GPU throughput for the 70B, is where the ladder ends — the lever that takes serving past every single-device record and makes the largest models servable at all.

```python
def _apply_tp_linear(linear, style, weight_splits=[]):
    rank, world_size = _get_rank(), _get_world_size()
    # weight is (out_features, in_features): colwise -> dim 0, rowwise -> dim 1
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    shard_dim, size_attr = dim_lookup[style]
    assert getattr(linear, size_attr) % world_size == 0
    def shard(x, dim):
        return torch.tensor_split(x, world_size, dim=dim)[rank]
    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        return torch.cat((shard(q, dim), shard(k, dim), shard(v, dim)), dim=dim)
    if weight_splits:                       # attention QKV: split per Q/K/V (grouped-query safe)
        if isinstance(linear, WeightOnlyInt4Linear):
            sharded_weight = shard_qkv(linear.weight, shard_dim, [i // 8 for i in weight_splits])
            linear.scales_and_zeros = shard_qkv(linear.scales_and_zeros, 1 - shard_dim, weight_splits)
        else:
            sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if isinstance(linear, WeightOnlyInt4Linear):
            linear.scales_and_zeros = shard(linear.scales_and_zeros, 1 - shard_dim)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)
    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

def _apply_tp_ffn(mlp):
    _apply_tp_linear(mlp.w1, "colwise")
    _apply_tp_linear(mlp.w3, "colwise")
    _apply_tp_linear(mlp.w2, "rowwise")
    world_size = _get_world_size()
    mlp.register_forward_hook(lambda _m, _i, output:
        funcol.all_reduce(output, "sum", list(range(world_size))))

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

# run: ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=8 generate.py --compile \
#        --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```
