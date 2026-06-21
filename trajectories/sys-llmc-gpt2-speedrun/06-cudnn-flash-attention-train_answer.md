The attention math is now exact and fast — cuBLAS batched GEMMs and a one-pass causal online softmax — but it still *materializes* the score matrix. The $QK^\top$ GEMM writes the $(B, NH, T, T)$ scores to HBM, the online-softmax kernel reads them back and writes the normalized weights, and the $\cdot V$ GEMM reads those. Even confined to the causal lower triangle, that is an $O(T^2)$ tensor making two full HBM round-trips per layer, and it is $O(T^2)$ in *capacity* too — a large chunk of the activation memory that grows quadratically with context. The online-softmax trick removed the *extra passes within* the softmax but not the fundamental materialization, because I still compute the entire score matrix in one GEMM before the softmax runs.

What has been staring at me since I wrote the online softmax is that its running-max-and-denominator bookkeeping does not need the whole row at once. It is a streaming algorithm: it processes elements one tile at a time, carrying $(m, d)$ across tiles, and the rescale $d \leftarrow d\cdot\exp(m_{\text{old}} - m_{\text{new}})$ corrects everything seen so far whenever the max grows. So I can tile the *keys and values* along the sequence: for a block of query rows, loop over blocks of K/V; for each K-block compute that block's scores into on-chip memory (shared memory / registers), update the running $(m, d)$ and a running output accumulator with the online rescale, and *throw the block's scores away*. The $(T, T)$ score matrix is never assembled in HBM — only $O(T\cdot h_s)$ of running state per query block lives on-chip — and $QK^\top$, the softmax, and $\cdot V$ all happen *inside one kernel*, tile by tile, fused. That is exactly flash attention: the same exact softmax (no approximation — it is the online algorithm carried across tiles), but the $O(T^2)$ intermediate stays on the chip, turning attention from HBM-bandwidth- and capacity-bound into compute-bound.

Do I write it by hand? I have a strong prior against it, and not just the general "use the library" rule — I have *direct evidence from this very path*: a minimal hand-rolled flash attention I tried earlier came out about 3× *slower* than the naive port. A flash attention that is both correct and *fast* is brutal: careful tiling of Q/K/V into shared memory, the running-softmax rescale interleaved with tensor-core mma instructions on the score and output tiles, double-buffering to hide the K/V loads behind compute, register-pressure management so the accumulators and running state fit, and per-architecture tuning of tile shapes — the kind of kernel where a naive implementation is not merely suboptimal but *negative*, slower than not fusing at all. Getting it to beat the materialized-score path takes the level of engineering that goes into a vendor kernel.

And there is now a vendor kernel for exactly this. So I propose to express attention as a **cuDNN Flash Attention graph**. cuDNN ships a fused scaled-dot-product attention primitive — `sdpa` in the frontend — that does the whole $QK^\top \to$ causal-softmax $\to \cdot V$ fused, on the tensor cores, without materializing the score matrix, tuned per architecture by NVIDIA. The same argument that sent the matmuls to cuBLAS sends attention to cuDNN. The frontend is graph-based: I describe Q, K, V as tensors, set an SDPA options object, and ask the graph for the fused op. The options carry exactly the three things my hand-written kernel handled by hand — `set_attn_scale` is the $1/\sqrt{h_s}$ I was fusing into the `exp`; `set_causal_mask(true)` is the causal masking I was doing by only touching the lower triangle; and `set_is_inference` controls whether the softmax statistics needed for the backward pass are emitted. The graph returns `[O, stats]`, where `stats` is the small per-row $(m, d)$ softmax state of shape $(B, NH, T)$ that the *backward* pass needs: flash attention does not keep the score matrix, so for backprop it recomputes the scores tile-by-tile from Q, K, V and the saved `stats` rather than reading a stored $(B,NH,T,T)$ tensor. That is the second place the $O(T^2)$ materialization is avoided — not just forward, but backward too.

A few engineering points make it clean. The data type matches `floatX`: the graph is set to BF16 I/O with FP32 intermediate and compute (`set_io_data_type(CUDNN_16BIT).set_intermediate_data_type(FLOAT).set_compute_data_type(FLOAT)`), so the precision discipline from the mixed-precision rung carries through (BF16 I/O, FP32 accumulation). The QKV layout costs nothing extra: my activation is already $(B, T, 3, NH, HS)$ from the projection, and cuDNN can consume Q, K, V as *strided views* into that single tensor by setting their strides, avoiding the explicit permute/unpermute the previous batched-GEMM path needed — one less pass over the data. And building the cuDNN graph (validate, build the operation graph, create execution plans) is *slow* — it can take a noticeable fraction of a second — so I cache the built graph keyed by the problem shape $(B, NH, T, HS, \text{is\_inference})$ and build only once; every step after the first executes the cached graph with a variant pack of pointers.

The honest cost: cuDNN is a heavy third-party dependency (it bloats compile time from seconds to ~a minute and must be installed), so it cannot be the always-on default — it is a compile-time option. But where it is available it removes the last big structural memory cost in attention: the score matrix never exists in HBM, forward or backward; attention stops being $O(T^2)$ in activation memory and stops paying the round-trip bandwidth; and the fused tensor-core kernel runs at vendor-tuned speed instead of my 3×-slower hand-rolled attempt. Correctness is the same exact scaled-dot-product causal attention, matching the reference to mixed-precision tolerance, so 3.29 holds.

What is left after this is back on the matmul side: the MLP's up-projection is followed by a GELU as a *separate* kernel that re-reads the matmul output from HBM, even though the matmul had that output on-chip in its epilogue — and I set up the fix two rungs ago by routing the matmuls through cuBLASLt, whose epilogue can do exactly that. The classifier still materializes the full $(B,T,V)$ logits. The cuDNN flash-attention graph (BF16 I/O, FP32 compute, causal, scale fused) and its execution:

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
