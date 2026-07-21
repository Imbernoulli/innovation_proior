Every improvement so far has fought the same wall: one GPU's memory bandwidth. Compile plus static cache got
the device running flat-out at ~1397 GB/s; int8 and int4 shrank the bytes per token; speculative decoding
amortized the verifier's weight read over several tokens. All of them reduce *demand* on a single HBM bus.
There is a complementary move I haven't made — instead of asking less of one HBM, use *more HBM in parallel*.
The 70B can't even fit on a single 80 GB card in bf16 (~140 GB, instant OOM; only int4 at ~35 GB squeezes
on), and for the 7B I'm bandwidth-bound on one device by definition. A second A100 doubles the aggregate
bandwidth; eight give eight times. If I spread one token's weight read across `N` cards so each reads `1/N`
simultaneously, the per-token read time drops by `N`, and a bandwidth-bound decode should speed up close to
linearly in `N` — exactly the regime where throwing hardware at the wall works, because the wall is bandwidth
and bandwidth is what more cards buy.

Quantitatively: the 7B at bf16 streams 13.5 GB per token, ~9.5 ms at ~1397 GB/s. Split across `N` cards each
reading `13.5/N` GB at once, the per-token read falls toward `9.5/N` ms — linear ceiling scaling, and the only
thing that can spoil it is time the cards spend *not* streaming weights: communication and synchronization. So
the design question is: how do I split the model so each GPU does `1/N` of the work on every token, while
keeping inter-card communication small enough that it doesn't eat the parallel-bandwidth win?

Only one of the three ways to use multiple GPUs helps this workload. Data parallelism — replicate the model,
send different requests to each card — raises aggregate *throughput* across many requests but does nothing for
a *single* request at batch 1: one card still streams the whole 13.5 GB. Pipeline parallelism — layers 0-15 on
GPU 0, 16-31 on GPU 1 — is the naive "split the model," and it's wrong here: at batch 1 the token flows through
layers serially, so GPU 1 idles while GPU 0 works and vice versa. It relays rather than parallelizes; per-token
latency is unchanged (one full pass, now with a hop) and only one card streams at a time. Pipelining only helps
when many tokens are in flight, which batch 1 forbids. The one that works is splitting *within* each layer —
**tensor parallelism** — so all `N` cards work the same layer at once, each on its own slice of the matrices,
every card streaming its shard of the weights simultaneously on every layer of every token. That's the only
decomposition that turns `N` cards into `N×` the bandwidth for a single token.

The layer's algebra dictates which dimension to shard and where communication is unavoidable. Take the
feed-forward, `w2(silu(w1(x)) * w3(x))`: project up from `dim=4096` to `intermediate=11008` with `w1` and `w3`,
gate elementwise, project back down with `w2`. Shard `w1` and `w3` **column-wise** — each GPU holds `11008/N`
output columns — and each GPU computes its own slice of the intermediate `silu(w1(x)) * w3(x)` entirely
locally, since the gate is elementwise (column `j` of the gated intermediate depends only on column `j` of
each). Then shard `w2` **row-wise** — each GPU holds the `11008/N` input rows matching its intermediate slice —
so each produces a partial sum of the 4096-wide output, and the layer closes with one **all-reduce** to sum the
partials. Writing the sum out confirms the pairing reconstructs the right answer: GPU `r` holds intermediate
slice `h_r` (columns `r·(11008/N):(r+1)·(11008/N)` of `h`) and the matching `w2` block `w2_r`, computes `w2_r ·
h_r`, and `Σ_r w2_r · h_r = w2 · h` — exactly the block decomposition of the matrix-vector product. Column-wise
*then* row-wise is precisely the pairing that pushes the one unavoidable sync to the layer's end and keeps
everything before it local; any other pairing would need a sync mid-FFN or fail to reconstruct the product.

Attention shards along the same seam with heads as the unit. Split the QKV projection column-wise so each GPU
owns a disjoint subset of heads and attends entirely on its own — each head is independent, its softmax over its
own query-key dots, no cross-head mixing — so `32/N` heads per card locally. The output projection `wo` mixes
the concatenated head outputs back to the model dimension; shard it row-wise, and the layer again closes with
one all-reduce — identical column-then-row structure, "head" playing the role "intermediate column" played in
the FFN. The QKV split has a wrinkle: Llama-2-70B uses grouped-query attention with only 8 key/value heads
against 64 query heads, so I can't cut the stacked QKV matrix into `N` even pieces — I split Q, K, V separately
by their own head counts and recombine, or a shard would end up with query heads whose matching KV heads live on
another card. The per-GPU `n_head`/`dim`/`head_dim`/`n_local_heads` are divided by the world size so the
KV-cache and rotary are sized for the local shard.

The core primitive underneath both shards a single linear: for column-wise I split the weight's output
dimension (`dim 0`, since weights are stored `(out_features, in_features)`), for row-wise the input dimension
(`dim 1`), each rank keeping only its `tensor_split(...)[rank]` slice — so each GPU literally loads and reads
`1/N` of that matrix's bytes, which is the whole point. An `assert size % world_size == 0` catches a ragged
split; the model dims are multiples of small powers of two precisely so they shard cleanly at 2, 4, 8 cards
(11008/8 = 1376, 32/8 = 4). The one dimension I must *not* naively divide is the 70B's 8 KV heads: at `N=8`
each card gets exactly 1, and at `N>8` the even split fails — the hard reason the QKV shard goes per-component
and the reason 8 cards is a natural cap for the 70B beyond just the NVLink-node boundary. Because the config
dims are divided *before* the caches are built, each card's `KVCache` is sized for its local `n_local_heads`,
so the KV-cache is itself sharded (each card holds keys/values for its own heads), not replicated — keeping
cache memory per card at `1/N` and letting the 70B's cache fit alongside its `1/8` weight shard. The whole
decode state — weights, cache, rotary — is divided so each card's footprint and per-token read are both `1/N`.
Full sharding code is in the answer.

The all-reduce lives in a `register_forward_hook` on the FFN and attention modules rather than in their
`forward`: it fires after the sub-layer returns its partial output, sums across ranks, and hands back the
reduced tensor, slotting the collective in at exactly the seam the algebra says it belongs without editing the
frozen module bodies. Registering it as a hook also means it composes with the compiled decode graph — the
collective is a captured op like any other, so I keep the scheduling win even with communication in the loop,
as long as it's graph-capturable (which `funcol.all_reduce` is). Composition with int4 is where the primitive
earns its complexity: a quantized linear has `scales_and_zeros` (int4) or `scales` (int8) that must split
*consistently* with the weight, or dequant pairs the wrong scales with the wrong rows. For int4 the packed
weight splits along its packed dimension while `scales_and_zeros` splits along the complementary axis, and a
per-component QKV split uses `[i // 8 for i in weight_splits]` because the packing folds 8 values into each
stored unit. Getting that index arithmetic right lets a card load `1/N` of the *int4* weight and its matching
scales, so a sharded int4 70B reads `~35/8 ≈ 4.4 GB` per card per token — the tensor-parallel and quantization
byte reductions multiply.

Now cost the communication I'm trading against the parallel bandwidth, since that's the one thing that can
spoil linear scaling. Two all-reduces per layer, 64 per token across 32 layers. At batch 1 in decode the
reduced activation is the `dim=4096` output vector, `4096 × 2 B ≈ 8 KB`; a ring all-reduce moves `~2(N−1)/N`
times that per rank, on the order of ~16 KB per rank per collective, times 64 ≈ ~1 MB of communication per
token. Against what it buys — at `N=8` the per-GPU 7B weight read drops from 13.5 GB to ~1.7 GB, ~0.85 ms at
2 TB/s — moving ~1 MB over NVLink at hundreds of GB/s is single-digit microseconds plus latency, a small
fraction of the parallelized read. That's exactly why this scales well *within* a node where NVLink bandwidth
is high, and why the benchmark caps at 8×A100 on one hybrid-cube-mesh box: cross-node interconnect is far
slower, so the all-reduces would stop being cheap and scaling would collapse the moment I left the node. But
the scaling is strong, not perfect, and the derivation shows why: the per-GPU weight slice shrinks like `1/N`
while the all-reduce cost doesn't — the message is the fixed `dim`-wide activation regardless of `N`, the ring
term `2(N−1)/N` actually grows toward 2, and latency per collective is roughly constant. So communication
becomes a larger *fraction* of the shrinking per-GPU compute as `N` grows, and the marginal card buys less than
the one before it. Tensor parallelism changes *nothing* about the numerics — it's an exact re-expression of the
same matmuls, the partials provably summing to the unsharded product — so like compile and unlike quantization
it preserves the output distribution, no accuracy check needed, and it composes with int8/int4 to multiply
whatever single-GPU byte reduction I've banked.

The bar to clear is the single-A100 record for the 7B, the compile + static-KV-cache "Base" at 104.9 tok/s —
the apples-to-apples bf16 number a second card has to beat, held at bf16 so the parallelism win is isolated
from the quantization wins. The bet: because decode is bandwidth-bound, splitting each token's weight read
across `N` NVLinked A100s multiplies aggregate bandwidth and drives throughput up nearly linearly — I'd expect
2 GPUs near 1.6× the 104.9 and 8 GPUs several-fold higher, the curve bending sub-linear as `N` grows exactly as
the communication accounting predicts. The two hedges are the ones the derivation exposed: the 64 all-reduces
per token are pure overhead that doesn't exist on one GPU, so on a slow interconnect this could fail to scale or
regress — it *relies* on NVLink; and the per-GPU work shrinks while the sync stays fixed, so the marginal card
buys less each time. The thing tensor parallelism unlocks that no single-GPU change can is the 70B *at all* in
bf16: it doesn't fit in 80 GB, so its single-GPU bf16 throughput is undefined (OOM), and only by splitting it
across 8 cards does it become servable. What I'd validate is realized tokens/s at 2, 4, and 8 GPUs against the
104.9 single-GPU baseline for the 7B, and the 8-GPU throughput for the 70B that otherwise can't run — and if
splitting the weight read across NVLinked cards scales the way a bandwidth-bound workload should, this is the
lever that takes throughput past every single-device record and makes the largest models servable.
