With the projection matmuls on cuBLAS, the elementwise layers vectorized and the residual+LayerNorm seams fused, attention is now the worst thing in the model. It is still the textbook three-pass path, and it stacks two separate problems: the matmuls inside it are hand-rolled, and the softmax in the middle round-trips the entire $O(T^2)$ score matrix through HBM. I take them in order.

The two big multiplies are $QK^\top$ — for each head, the $(T, h_s)$ query matrix times the $(h_s, T)$ key matrix, giving the $(T, T)$ scores — and the weighted sum, the $(T, T)$ attention weights times the $(T, h_s)$ value matrix. These are *batched* matmuls, one per (batch, head) pair, $B{\cdot}NH$ of them, and my naive attention does them with hand-written kernels. I already learned on the projection matmuls that hand-rolling a GEMM is a mistake when cuBLAS exists, and cuBLAS has exactly the primitive: a strided-batched GEMM (`cublasGemmStridedBatched`) that does all $B{\cdot}NH$ independent products in one call by striding through Q, K, V. So $QK^\top$ becomes one batched GEMM and $\cdot V$ another, both on the tensor cores — which is most of the ~20× over the naive port that a cuBLAS-plus-custom-softmax attention buys.

The softmax left in the middle is the second, subtler cost. The scores $S = QK^\top/\sqrt{h_s}$ form a $(B, NH, T, T)$ tensor — for $T=1024$ across $B{\cdot}NH$ heads, the largest activation in the attention path. The textbook softmax over each row does pass one to find the row max (for numerical stability: subtract the max before exponentiating so `exp` does not overflow), pass two to compute $\exp(S - \max)$ and the row sum, then a divide. As separate kernels the score matrix is written, read for the max, read again for the exp-and-sum, the normalized weights written, and finally read by the $\cdot V$ GEMM — the $O(T^2)$ tensor crossing HBM several times. After the GEMMs are fixed, that is the dominant attention cost.

So the method is **cuBLAS batched-GEMM attention with a fused causal online-softmax**. Two things make the softmax much cheaper. First, attention is *causal* — token $t$ attends only to tokens $\le t$ — so row $t$ has only $t{+}1$ valid entries and the upper triangle is masked to $-\infty$, contributing nothing. The textbook code computes the full $T\times T$ matrix and then masks; instead I only ever touch the lower triangle: for row $t$ I loop $i = 0..t$, never $0..T$. That halves the score work on average and means I never write the masked entries at all.

Second, the *online* softmax replaces the two passes with one. The two-pass version reads each row twice; the online version walks the row keeping a running max $m$ and a running denominator $d = \sum \exp(x_i - m)$. When a new element $x$ exceeds the current max, the old denominator was computed relative to a smaller max and must be rescaled: $d \leftarrow d\cdot\exp(m_{\text{old}} - m_{\text{new}})$, then $d \mathrel{+}= \exp(x - m_{\text{new}})$. The rescale factor $\exp(m_{\text{old}} - m_{\text{new}}) \le 1$ corrects every previously-accumulated term to the new max in one multiply, so after one pass I have both the true row max and the correct normalizer, having read each element exactly once. I fold the $1/\sqrt{h_s}$ scale directly into this loop as `inv_temperature` — there is no reason to run a separate kernel multiplying the whole score matrix by the scale when I am already touching every element; the scale rides along for free inside the `exp` argument. A warp cooperates on one row: each lane streams a strip maintaining its own $(\text{maxval}, \text{sumval})$, processing four elements at a time as a `float4` for coalescing; then a single warp-reduction gives the global max, each lane rescales its `sumval` to that global max, and a warp-sum gives the true normalizer. One pass, one warp-reduction, the scale fused, only the lower triangle touched.

There is a deliberate judgment call in the final normalize loop. To write the normalized weights I need $\exp(x - \max)$ again, which I computed during the streaming pass but did not keep (storing all $t$ per row would cost shared memory). I can stash them or recompute them; recomputing means a second `exp` per element but avoids a round-trip of the exponentials through memory, and on this hardware an `exp` is cheaper than a global-memory round-trip for a bandwidth-bound kernel — recalculation is faster than the round-trip. So I recompute, with `__ldcs`/`__stcs` streaming hints because the score row is read once and the normalized weights written once, neither wanting to linger in cache.

The attention path therefore becomes: a permute kernel laying Q, K, V out as the contiguous $(B, NH, T, h_s)$ tensors the batched GEMM wants; a batched cuBLAS GEMM for $QK^\top$; this fused online-softmax kernel that scales, masks causally, and normalizes each row in one streaming pass over its lower triangle; a batched cuBLAS GEMM for $\cdot V$; an unpermute back to $(B, T, C)$. The online softmax is algebraically identical to the two-pass softmax — the running rescale is exact, not an approximation — the causal masking is what the model defines, and the math is FP32 internally, so this matches the reference to tolerance and the 3.29 bar holds. The win is the ~20× over naive attention from the cuBLAS GEMMs plus the single-pass, causal-only, scale-fused softmax.

What still nags: the score matrix, even lower-triangular, is *still* written to HBM by the $QK^\top$ GEMM and read back by the softmax and the $\cdot V$ GEMM. I made the softmax one-pass and causal but did not remove the $O(T^2)$ materialization itself — the $(B,NH,T,T)$ tensor still exists in global memory between the GEMMs. The online-softmax trick is precisely the key that *would* let me avoid materializing it: if I tile K/V along the sequence and carry the running max and denominator across tiles, I could compute attention block-by-block and never write the full score matrix. That is the flash-attention idea, and it is where the next rung goes. The fused online-softmax kernel — scale-fused, directly autoregressive, one streaming pass with running max/denominator and a warp reduction:

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
