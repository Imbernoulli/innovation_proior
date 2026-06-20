**Problem (from baseline).** Single-GPU decode is bandwidth-bound on one A100's HBM; every prior rung
reduced *demand* on that one bus (fewer bytes per token, fewer passes per token). The complementary move
is to add *more* HBM in parallel: N NVLinked A100s give N× aggregate bandwidth, and a bandwidth-bound
decode should scale near-linearly if each token's weight read is split N ways. Pipeline parallelism fails
at batch 1 (cards relay, idle in turn); the split must be *within* each layer so all N cards work every
layer of every token.

**Key idea.** **Tensor parallelism.** Shard each layer's matrices across ranks so each GPU reads 1/N of
the weights per token. FFN: `w1`/`w3` column-wise (each rank computes its own slice of the intermediate;
the SiLU-gate is elementwise and local), `w2` row-wise (each rank yields a partial output) → one
**all-reduce** to sum. Attention: shard QKV column-wise by heads (each head independent), `wo` row-wise →
one all-reduce; per-rank `n_head`/`dim`/`head_dim`/`n_local_heads` divided by world size so caches and
rotary sizing match the shard (QKV split per-component to respect grouped-query attention). Two all-reduces
per layer are the only communication.

**Why it works.** Decode is bandwidth-bound, so splitting the weight read across N cards multiplies
aggregate bandwidth and drives throughput up nearly linearly; the column-then-row pairing pushes the sync
to each sub-layer's end and keeps everything before it local. On a single node with NVLink the all-reduces
are cheap relative to the weight read they parallelize, so it scales well within a node (benchmark caps at
8×A100) and bends sub-linear as N grows. It is numerically exact (an identity re-expression of the same
matmuls), so it preserves the output distribution, and it composes with int8/int4 — and it is the *only*
way the 70B (which OOMs on one 80 GB card) becomes servable at all.

**Change / code.** `apply_tp` in `tp.py`: shard every FFN and attention block, fix config dims, register
the all-reduce hooks. Launch with `torchrun --nproc_per_node=N`.

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
