**Problem (from step 5).** Even with cuBLAS batched GEMMs and a one-pass online softmax, attention still
*materializes* the (B,NH,T,T) score matrix in HBM: the QKᵀ GEMM writes it, the softmax reads/writes it, the
·V GEMM reads it — two O(T²) round-trips per layer, and O(T²) activation capacity that grows quadratically with
context.

**Key idea.** The online softmax is a streaming algorithm — it carries a running max `m` and denominator `d`
across tiles — so tile the K/V along the sequence: for each block of queries, loop over K/V blocks, compute each
block's scores *on-chip*, fold them into the running `(m, d)` and the running output with the online rescale,
and discard them. The score matrix never reaches HBM. This is **flash attention**, and rather than hand-roll it
(a minimal hand-rolled attempt was ~3× *slower* than naive), express it as a **cuDNN `sdpa` graph** — the
vendor's fused, architecture-tuned flash-attention primitive, with `set_attn_scale` and `set_causal_mask`
carrying the scale and causality.

**Why it works.** Attention turns from HBM-bandwidth/capacity-bound into compute-bound: only O(T·hs) running
state lives on-chip, never the O(T²) scores. Backward also avoids the materialization — it recomputes scores
tile-by-tile from Q,K,V and the small saved softmax `stats` (B,NH,T). BF16 I/O with FP32 intermediate/compute
keeps the precision discipline; the math is exact scaled-dot-product causal attention, holding 3.29. QKV is
consumed as strided views of the existing (B,T,3,NH,HS) tensor (no permute), and the built graph is cached by
shape since building is slow. Cost: a heavy cuDNN dependency, so it is a compile-time option, not the default.
Still open: GELU-after-matmul round-trips, and the (B,T,V) classifier.

**Change / code.** The cuDNN flash-attention graph (BF16 I/O, FP32 compute, causal, scale fused) and its
execution (`llmc/cudnn_att.cpp`):

```cpp
namespace fe = cudnn_frontend;
#if defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#else // default to bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

auto lookup_cache_or_build_graph_fwd(int B,int H,int T,int HS, int is_inference_only) {
    // ... shape-keyed cache lookup; build only on first use (building is the VERY SLOW part) ...
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS): consume Q,K,V as strided views, no external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim({B, H, T, HS})
                               .set_uid(Q_UID).set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim({B, H, T, HS})
                               .set_uid(K_UID).set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim({B, H, T, HS})
                               .set_uid(V_UID).set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_uid(Attn_scale_UID)
                               .set_is_pass_by_value(true).set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);   // 1/sqrt(HS), fused
    sdpa_options.set_causal_mask(true);        // causal masking, no T x T materialization
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);   // fused QK^T -> softmax -> ��V

    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(O_UID);
    if (is_inference_only == false) {   // softmax stats (B,NH,T) FP32, needed for backward recompute
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
              .set_dim({B, H, T, 1}).set_stride({H * T, T, 1, 1}).set_uid(Stats_UID);
    }
    // ... validate / build_operation_graph / create_execution_plans / build_plans; size workspace; cache ...
}

void attention_forward_cudnn(floatX* out, float* stats, floatX* inp, int B, int T, int NH, int C, cudaStream_t stream) {
    int HS = C / NH;
    bool is_inference_only = (stats == nullptr);
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    auto graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);
    void* devPtrQ = inp; void* devPtrK = (inp + C); void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};
    if (is_inference_only == false) { variant_pack[Stats_UID] = stats; }
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));  // run the cached graph
}
```
