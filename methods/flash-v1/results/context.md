# Context: exact self-attention on memory-bound GPUs (circa 2021-2022)

## Research question

The self-attention layer at the heart of the Transformer (Vaswani et al. 2017) computes, for
query/key/value matrices `Q, K, V` in `R^{N x d}` (sequence length `N`, head dimension `d`),

```
S = Q K^T  in R^{N x N},   P = softmax(S)  (row-wise),   O = P V  in R^{N x d}.
```

The cost grows quadratically in `N`: the score matrix `S` and the probability matrix `P` are
each `N x N`. People increasingly want long sequences — language models at 1K-16K tokens,
document classification, the long-range-arena tasks — and the question is how to compute attention
**exactly** (the same mathematical function, within floating-point tolerance, with no
approximation) on real GPU hardware as `N` grows. Two facts shape the setting: softmax couples an
entire row of `S` (the denominator sums over all `N` keys), and the backward pass conventionally
uses the `N x N` matrices `S, P` to form gradients.

## Background

**The standard implementation.** The textbook attention kernel materializes `S = Q K^T` to GPU
high-bandwidth memory (HBM), reads it back to compute `P = softmax(S)`, writes `P` to HBM, then
reads `P` and `V` to compute `O = P V`. Often `N >> d` (GPT-2: `N = 1024`, `d = 64`), so the
`N x N` tensors are large relative to the inputs. The softmax, masking, and dropout that sit on
`S`/`P` are elementwise and reduction operations, doing little arithmetic per byte moved.

**The GPU memory hierarchy and the roofline.** A modern GPU (A100) has 40-80 GB of HBM at
1.5-2.0 TB/s and only about 192 KB of on-chip SRAM per streaming multiprocessor (108 of them),
but that SRAM runs at roughly 19 TB/s — about an order of magnitude faster bandwidth than HBM and
many orders of magnitude smaller in size. A kernel loads from HBM into registers/SRAM, computes,
and writes back to HBM. The roofline / arithmetic-intensity view (Williams et al. 2009) classifies
an operation as **compute-bound** (limited by FLOPs — large matrix multiplies, big convolutions)
or **memory-bound** (limited by HBM traffic — elementwise activation/dropout, reductions like
sum/softmax/layernorm). Across GPU generations compute throughput has outpaced memory bandwidth, so
more and more operations are memory-bound. Profiling the standard attention kernel shows the time
goes into moving the `N x N` matrices across the HBM bus, dominated by its softmax and its `N x N`
reads/writes.

**Kernel fusion** is the standard cure for memory-bound work: if several operations touch the same
data, load it once into SRAM, do all the operations there, and write once — instead of round-trips
to HBM between each op. Compilers fuse chains of elementwise ops automatically. In *training*, the
intermediate values are also needed again in the backward pass, so naive fusion writes them to HBM.

**The IO-complexity lens.** A classical way to reason about memory-bound algorithms is to count
reads and writes between fast and slow memory rather than FLOPs (Aggarwal & Vitter 1988, the
input/output complexity model; also the working-set, data-locality, and roofline traditions). This
model has guided database joins, image-processing pipelines, and dense linear algebra (BLAS). High-
level frameworks (PyTorch, TensorFlow) expose no fine-grained control over memory movement.

**Numerically stable and online softmax.** For a vector `x in R^B`, softmax is computed in the
"safe" form that subtracts the row maximum before exponentiating, because `e^{x}` overflows to
`inf` in floating point once `x` exceeds ~89:

```
m(x) = max_i x_i,   f(x)_i = e^{x_i - m(x)},   l(x) = sum_i f(x)_i,   softmax(x) = f(x) / l(x).
```

The safe form naively needs three passes over `x` (max, normalizer, output). Milakov & Gimelshein
(2018) showed that the max `m` and the normalizer `l` can be accumulated in a **single pass** with
a rescaling recurrence — when a new element raises the running max from `m_{j-1}` to `m_j`, the
running normalizer is corrected by the factor `e^{m_{j-1} - m_j}` before the new term is added:

```
m_j = max(m_{j-1}, x_j),   l_j = l_{j-1} * e^{m_{j-1} - m_j} + e^{x_j - m_j}.
```

They prove by induction that this yields exactly `m_V = max_k x_k` and `l_V = sum_k e^{x_k - m_V}`.
More generally, for two sub-vectors `x = [x^(1)  x^(2)]` the statistics of the whole combine from
the statistics of the parts: `m(x) = max(m(x^(1)), m(x^(2)))`, and the normalizers add after being
rescaled to the common max,
`l(x) = e^{m(x^(1)) - m(x)} l(x^(1)) + e^{m(x^(2)) - m(x)} l(x^(2))`. This is "algebraic
aggregation" — a reduction that can be computed block by block as long as a little summary state
(`m`, `l`) is carried along.

**Selective recomputation (gradient checkpointing).** Instead of storing every activation for the
backward pass, one can store a subset and recompute the rest during backprop (Griewank & Walther
2008; Chen et al. 2016). It is the standard way to trade extra compute for reduced peak memory.

## Baselines

These are the prior approaches a new attention method is measured against.

**Standard (dense, exact) attention.** The three-step `S = QK^T`, `P = softmax(S)`, `O = PV`
pipeline above, implemented as separate kernels (often with masking fused into softmax, e.g.
Megatron-LM, Shoeybi et al. 2019). It is exact and uses heavily optimized dense matmuls. It
materializes the `N x N` matrices `S` and `P` in HBM, costing `O(N^2)` memory and
`Theta(Nd + N^2)` HBM accesses.

**Approximate attention (sparse and low-rank).** A large family trades exactness for asymptotically
cheaper compute: hashing/sparsity (Reformer, Kitaev et al. 2020; routing/Smyrf), low-rank or
kernel-feature approximations (Linformer, Wang et al. 2020; Performer, Choromanski et al. 2020;
linear attention, Katharopoulos et al. 2020), and hybrids (Longformer, Beltagy et al. 2020;
BigBird, Zaheer et al. 2020; Scatterbrain). These reduce FLOPs to near-linear in `N` by changing
the attention function.

**Lazy-softmax / chunked exact attention with reduced memory footprint (Rabe & Staats 2021;
Jang et al. 2019).** This line keeps attention *exact* and reduces **memory footprint**. The move
is to defer the `1/sum` division to the very end (distributive law) and process keys/values
incrementally: maintain a running unnormalized output `v*` and running normalizer `s*`, plus a
running max `m*` for stability, renormalizing `v*` and `s*` by `e^{m* - m_i}` whenever a new score
raises the max. This needs only `O(1)` memory per query (`O(log n)` / `O(sqrt n)` in practice on a
TPU) and is exact. It summarizes each block into a temporary output and combines them at the end,
and for the backward pass uses gradient checkpointing, recomputing the attention matrix and each
block's temporary output.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Attention microbenchmarks**: forward, and forward+backward, runtime and memory of a single
  attention layer as a function of sequence length (e.g. `N` from 128 to 64K), head dimension `d`
  (64-128), number of heads, and batch size, on an A100. Reporting FLOPs, HBM read/write volume,
  and wall-clock latency side by side, since the question is precisely whether FLOPs or HBM
  accesses determine runtime. Also sweeping the block size to see its effect on HBM accesses and
  runtime, and comparing causal vs non-causal.
- **End-to-end Transformer training speed**: BERT-large (seq. length 512), measured against the
  MLPerf 1.1 training-speed records; GPT-2 (seq. length 1K) against HuggingFace and Megatron-LM
  implementations; the Long-Range Arena suite (seq. length 1K-4K).
- **Model quality from longer context**: language-model perplexity (GPT-2), long-document
  classification, and the long-sequence Long-Range Arena tasks including Path-X (16K) and
  Path-256 (64K), where the question is whether a longer context is even feasible to run.
- **Correctness**: the computed output must match a reference exact-attention implementation to
  within floating-point tolerance (the method must be exact, not approximate).
- Hardware: NVIDIA A100-class GPUs, FP16/BF16 inputs with FP32 accumulation, contiguous
  `(batch, heads, seqlen, headdim)` tensors. The benchmark FLOP convention for causal attention is
  `4 * batch * heads * seqlen^2 * headdim / 2`.

## Code framework

The kernel plugs into the standard GPU attention-kernel harness. A high-level framework (PyTorch)
supplies the tensors and the reference, but the fast path is a hand-written GPU kernel that has
fine-grained control over memory movement — here expressed in a GPU kernel DSL (Triton) so the
on-chip vs off-chip data flow is explicit. Nothing about *how* the attention is organized inside
the kernel is settled yet — that organization is exactly what is to be designed — so the substrate
is only the generic pieces that already exist: a launch wrapper that takes `Q, K, V`, allocates the
output and an auxiliary buffer for anything a training kernel may need to save, computes a grid,
and launches a kernel; and a kernel that is handed pointers and strides and is expected to write
the correct output `softmax(QK^T) V`. The single empty slot is the body of the kernel.

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out, Aux,       # pointers to HBM tensors
    sm_scale,                # softmax scale, typically 1/sqrt(headdim)
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    seqlen,
    BLOCK_M: tl.constexpr,   # how many query rows this program handles
    BLOCK_N: tl.constexpr,   # how many key/value rows per step
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """GPU attention forward kernel.
    One program is responsible for some rows of the output.
    """
    # TODO: implement the kernel body that produces
    #       O = softmax(Q K^T * sm_scale) V (causal-masked if IS_CAUSAL).
    pass


def attention_forward(q, k, v, causal=True, sm_scale=None):
    """Launch wrapper: lay out the grid and call the kernel."""
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    aux = torch.empty((batch * nheads, seqlen), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 64, 64                       # working-set tile sizes
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
    _attn_fwd[grid](
        q, k, v, o, aux, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o, aux
```

The wrapper fixes the launch and the tensor layout; the kernel body is where the attention
algorithm will live.
