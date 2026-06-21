# Context: Fast and memory-efficient exact attention

## Research question

On long sequences, the self-attention module of a Transformer has time and memory that both scale
quadratically in the sequence length N. Given queries, keys, and values Q, K, V ∈ ℝ^{N×d},
attention forms the score matrix S = QK^T ∈ ℝ^{N×N}, applies a row-wise softmax P = softmax(S), and
produces O = PV ∈ ℝ^{N×d}. The N×N matrices S and P account for the bulk of the memory footprint
and the runtime. The question: how to compute attention faster and with a smaller memory footprint
on real hardware, so that Transformers can be trained on longer contexts within wall-clock and
memory budgets.

## Background

**GPU memory hierarchy.** A modern GPU (e.g., A100) has a large but relatively slow high-bandwidth
memory (HBM): 40–80 GB at 1.5–2.0 TB/s. It also has a small, very fast on-chip SRAM: about 192 KB
per streaming multiprocessor across 108 SMs, at an estimated ~19 TB/s. SRAM bandwidth is roughly an
order of magnitude higher than HBM, but SRAM is orders of magnitude smaller. A GPU kernel loads
inputs from HBM into registers and SRAM, computes, and writes outputs back to HBM.

**Compute has outpaced memory.** Over successive GPU generations, arithmetic throughput has grown
faster than memory bandwidth. Consequently many operations are limited not by how much arithmetic
they do but by how much data they move to and from HBM. Operations are classified by their
*arithmetic intensity* (FLOPs per byte of memory access, per the Roofline model of Williams et al.
2009): *compute-bound* operations (large matrix multiplies, convolutions with many channels) are
limited by arithmetic; *memory-bound* operations (elementwise ops such as activation and dropout,
and reductions such as sum, softmax, batch/layer norm) are limited by memory traffic.

**Attention is largely memory-bound.** In standard attention the softmax, masking, and dropout
applied to the N×N matrices are elementwise/reduction operations — memory-bound. The matrix
multiplies QK^T and PV move the N×N matrices through HBM. The empirical consequence is that a large
fraction of attention's wall-clock time is spent moving S and P to and from HBM, not doing useful
arithmetic.

**Kernel fusion.** A standard remedy for memory-bound operations is kernel fusion: if several
operations apply to the same data, load it once, do all of them, write once. Compilers fuse many
elementwise operations automatically. During training the intermediate values are written to HBM so
the backward pass can use them to compute gradients.

**Online (single-pass) softmax.** Milakov and Gimelshein (2018) showed that the softmax of a
vector can be computed in a single pass that never holds the whole vector at once. Maintain a
running maximum m and a running denominator d; on reading a new entry x, update
m_new = max(m, x) and d_new = d·e^{m − m_new} + e^{x − m_new}. The rescaling factor e^{m − m_new}
corrects the partial denominator whenever the running maximum changes. This "algebraic aggregation"
decouples the entries of a softmax so the reduction can be split across pieces and recombined.

**Linear extra memory for attention.** Rabe and Staats (2021) applied the online-softmax idea to
attention and showed that the *extra* memory needed is only linear in N, not quadratic — the N×N
matrix need not be fully materialized to compute the result. Their target was the peak memory
footprint; their method summarizes each block by a temporary output plus softmax statistics and
combines all block outputs at the end, and it uses generic gradient checkpointing (recompute via
the forward) for the backward pass.

**IO complexity / external-memory model.** Aggarwal and Vitter (1988) formalized analyzing
algorithms by the number of transfers between a small fast memory and a large slow memory. Lower
bounds over a subrange of the fast-memory size are standard in the streaming-algorithms literature
(e.g. Woodruff 2004).

**Gradient checkpointing.** Griewank and Walther (2008) and Chen et al. (2016) reduce training
memory by not storing some activations and recomputing them during the backward pass — trading
extra computation for lower peak memory.

**Block-sparse / structured patterns.** A long line of work reduces attention cost by restricting
which (query, key) pairs interact: sparse patterns (e.g. Child et al. 2019 with sparsity ~N^{-1/2};
BigBird, Longformer with ~N^{-1} log N), and structured matrices such as butterfly factorizations
(Dao et al. 2019; Dao et al. 2020 kaleidoscope; Dao et al. 2021 pixelated butterfly), which can
express arbitrary structured matrices with near-optimal parameter count and runtime and are more
hardware-friendly than arbitrary sparsity.

## Baselines

**Standard attention.** The standard dense implementation computes, in sequence:
S = QK^T (load Q, K; write the N×N matrix S to HBM); P = softmax(S) row-wise (read S, write P to
HBM); O = PV (read P, V; write O). It materializes both S and P in HBM, costing O(N²) memory and a
number of HBM accesses that grows quadratically in N. The backward pass loads the stored P and dO
to get dV = P^T dO, dP = dO V^T, then forms dS via the softmax Jacobian and dQ = dS K, dK = dS^T Q
— again touching the N×N matrices in HBM.

**Approximate-attention methods (sparse / low-rank).** To cut the quadratic cost, sparse methods
(e.g. Reformer with locality-sensitive hashing; sparse-pattern transformers) compute only a subset
of the entries of S; low-rank methods (e.g. Performer, Linformer) approximate softmax(QK^T) by a
low-rank factorization, giving linear or near-linear FLOPs; combinations (Longformer, BigBird,
Scatterbrain) mix the two. These trade exactness for fewer FLOPs.

**Linear-extra-memory attention (Rabe & Staats 2021).** Uses online softmax to compute exact
attention with only linear extra memory and gradient checkpointing for the backward. It keeps one
temporary output per block and combines at the end; its backward recomputes via the forward.

## Evaluation settings

The natural yardsticks at the time, for measuring an attention implementation:

- **Speed / IO micro-benchmarks**: forward and forward+backward wall-clock runtime on a single GPU
  (e.g. A100) for representative configurations (sequence lengths from ~128 up to tens of
  thousands, head dimension d = 64–128, multiple heads, large batch), plus measured HBM read/write
  volume and FLOP counts, and runtime as a function of block size.
- **Language-model training**: BERT-large (sequence length 512), GPT-2 (sequence length 1K), with
  wall-clock training time and validation perplexity; MLPerf training-time references.
- **Long-sequence benchmarks**: the Long-Range Arena suite (sequence lengths ~1K–4K) and its
  hardest pixel-level tasks Path-X (16K) and Path-256 (64K); long-document classification.
- **Metrics**: wall-clock time, peak memory, HBM-access volume, perplexity, and downstream accuracy.

## Code framework

The primitives that already exist: a tensor library with optimized dense matrix multiply and
softmax, an autograd engine, and the ability to write a custom GPU kernel for fine-grained control
of memory (CUDA, or a kernel DSL such as Triton). Standard attention sits on these as:

```python
import torch

def standard_attention(Q, K, V, scale):
    # Q, K, V: (N, d). Materializes the N x N matrices in HBM.
    S = (Q @ K.transpose(-1, -2)) * scale      # (N, N) written to HBM
    P = torch.softmax(S, dim=-1)               # (N, N) read + written
    O = P @ V                                   # (N, d)
    return O


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale):
        # TODO: replace the dense attention body with the implementation
        #       strategy being designed.
        O = None      # TODO
        saved = ()    # TODO
        ctx.save_for_backward(*saved)
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        # TODO: produce gradients matching standard attention.
        dQ = dK = dV = None   # TODO
        return dQ, dK, dV, None


def attention_kernel(Q, K, V, scale, config):
    # TODO: fill in the custom kernel body.
    pass
```

The empty slots are where a new attention implementation would define the forward computation,
the saved state for gradients, the backward computation, and any custom kernel body.
