**Problem (from step 4).** Attention is the textbook three-pass path and the worst kernel in the model: its two
batched matmuls (QKᵀ, ·V) are hand-rolled, and the softmax between them round-trips the entire O(T²)
(B,NH,T,T) score matrix through HBM across a max pass, an exp-and-sum pass, and a normalize pass — on the
single largest activation in the attention path.

**Key idea.** Move both attention matmuls to **cuBLAS strided-batched GEMMs** (one call does all B·NH heads,
on the tensor cores), and replace the softmax with a single fused kernel that (a) only ever touches the
**lower-triangular** part, since attention is causal; (b) fuses the `1/√hs` **scale** into the exp argument; and
(c) uses the **online softmax** — one streaming pass keeping a running max `m` and running denominator `d`,
rescaling `d ← d·exp(m_old − m_new)` whenever a larger element arrives — so the max and the normalizer come out
of one read of each row.

**Why it works.** The cuBLAS batched GEMMs are most of the ~20× over the naive attention port. Online softmax is
algebraically exact (the running rescale is not an approximation), so one pass replaces two; causal masking
halves the score work and skips the masked writes entirely; the fused scale deletes a whole score-matrix-wide
kernel. Recomputing `exp(x−max)` in the normalize loop is cheaper than a memory round-trip of the exponentials.
FP32-internal math holds the 3.29 target. Still open: the (lower-triangular) score matrix is *still*
materialized in HBM between the two GEMMs — only flash attention removes that.

**Change / code.** The fused online-softmax kernel: scale-fused, directly autoregressive (causal), one
streaming pass with running max/denominator and a warp reduction (`dev/cuda/attention_forward.cu`,
`softmax_forward_kernel5`):

```c
__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), N = B*NH. fuses the scale; directly autoregressive (lower triangle only);
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) { return; }
    int own_pos = idx % T;           // this row's token position t
    int pos_by_4 = own_pos / 4;
    const float* x = inp + idx * T;  // one row of scores

    float maxval = -FLT_MAX;
    float sumval = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {   // only up to own_pos (causal)
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) { maxval = fmaxf(maxval, vec_at(v, k)); }
        sumval *= expf(inv_temperature * (old_maxval - maxval));         // online rescale of the running sum
        for(int k = 0; k < 4; ++k) { sumval += expf(inv_temperature * (vec_at(v, k) - maxval)); }
    }
    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {                     // ragged tail up to the diagonal
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));          // rescale each lane to the global max
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}
```
