## Research question

Progress in deep learning is paced by the availability of efficient GPU compute kernels. A
handful of vendor libraries — cuBLAS for dense linear algebra, cuDNN for convolutions —
cover a fixed catalogue of primitives extremely well, and as long as a model is built out of
exactly those primitives it runs near the hardware's peak. The moment a researcher steps
off that catalogue — a fused activation, a block-sparse weight matrix, a shifted
convolution, a custom attention pattern — there is no library entry, and the operation runs
at a small fraction of peak unless an expert hand-writes a CUDA or PTX kernel for it. That
hand-written route is brutal: it takes months, demands intimate knowledge of one specific
GPU's memory hierarchy and warp scheduling, and the result does not survive the next
architecture or the arrival of new arithmetic units. So novel ideas are either slow (poor
device utilization) or expensive (a specialist writes and re-writes low-level code).

The precise goal is a programming abstraction, plus a compiler behind it, that lets someone
who is *not* a GPU-microarchitecture specialist write a custom kernel at a high level and
have it compiled into code competitive with the hand-tuned vendor libraries — and to do so
*portably*, so the same source survives across architectures. It must additionally be able
to express the operations the vendor libraries omit, including the non-rectangular indexing
of structured-sparse computations; and it must make it natural to *fuse* a chain of
operations into a single kernel, because for the memory-bandwidth-bound operations that
dominate the omitted catalogue (elementwise activations, normalizations), the only way to go
faster is to move fewer bytes to and from DRAM. Closing that gap — high-level and portable,
yet on par with hand-tuned code, and fusion-friendly — is the problem.

## Background

The hardware sets the rules. A modern NVIDIA GPU is a set of streaming multiprocessors
(cores, on the order of 64), sharing a large L2 data cache (~10 MB) and several DRAM memory
controllers; each core has a register file, an addressable L1 / shared memory (~128 kB), and
issues instructions over groups of 32 threads called warps. Increasingly the cores also
carry tensor cores — specialized units whose matrix-multiply-accumulate throughput is an
order of magnitude above ordinary fused-multiply-adds, but which are correspondingly harder
to program.

The dominant cost is the *memory wall*: a DRAM load or store is several orders of magnitude
slower than an arithmetic instruction. The Roofline model (Williams, Waterman & Patterson
2009) makes this quantitative — achievable performance is bounded by
`min(peak_FLOPs, arithmetic_intensity x bandwidth)`, where arithmetic intensity is FLOPs per
byte moved. An operation with low arithmetic intensity is *bandwidth-bound*: its run time is
essentially `bytes_moved / bandwidth`, and the only lever that helps is moving fewer bytes.
This is exactly the regime of elementwise and reduction operations. When such an operation is
expressed as a composition of separate library calls, every intermediate tensor is written
out to DRAM and read back in — so a multilayer-perceptron block that does `linear ->
activation -> linear` round-trips its (large) hidden activation through DRAM multiple times,
paying the memory wall once per op boundary.

The known levers for getting near peak, all established before any of this, are: (1) *memory
coalescing* — because DRAM is read in bursts, adjacent threads in a warp must access adjacent
addresses, or effective bandwidth collapses; (2) *shared memory reuse* — stage a tile of data
in the fast addressable L1 so it is read from DRAM once and reused, while arranging it to
avoid *bank conflicts*, since accesses to the same one of the ~32 banks serialize; (3)
*latency hiding* — overlap loads with computation through instruction-level parallelism and
prefetching; and (4) *occupancy* — launch the right amount of work, which depends both on
fixed architecture details (number of SMs) and on quantities known only at run time (the
actual tensor shapes). It is an observed pathology that good block-shape choices depend on
all of these at once, and that a choice that fits cache and saturates bandwidth on one input
shape or one GPU is wrong on another.

Two further pieces of background matter for a concrete MLP kernel. First, the Gaussian
Error Linear Unit (Hendrycks & Gimpel 2016) is the smooth activation `GELU(x) = x * Phi(x)`,
where `Phi` is the standard-normal CDF, so `GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`; in
practice it is evaluated with the tanh approximation
`0.5 * x * (1 + tanh[ sqrt(2/pi) * (x + 0.044715 * x^3) ])`. In a low-precision kernel, the
polynomial and tanh are evaluated in float32 before casting back: fp16 can overflow in the
cubic term, and bf16/half precision is too coarse for the approximation. Second, the standard
way convolutions reach peak hardware is *implicit GEMM*
(im2col): each input patch is flattened so the convolution becomes a matrix multiplication
consumable by the tensor cores — established since AlexNet.

## Baselines

These are the routes a new system would be measured against and would react to.

**Hand-written CUDA / PTX micro-kernels (cuBLAS, cuDNN, CUTLASS, BLISlab).** The execution
model is the standard *per-thread* SPMD of CUDA: the programmer writes the body of a single
scalar thread, identifies its position with `blockIdx`/`threadIdx`, and is responsible for
partitioning a data tile across the threads of a block, staging operands into shared memory,
inserting `__syncthreads()` barriers, and laying out accesses so they coalesce. The core
algorithmic content is blocked/tiled GEMM with explicit shared-memory staging and a
carefully tuned thread-to-data mapping. This achieves peak performance — it *is* the vendor
libraries — and is the gold standard for code quality. **Limitation:** the effort is enormous
and architecture-specific. A single high-performance kernel can take months of expert work
in PTX; the result is tuned for one GPU's register file, shared-memory size and warp count,
and does not carry over to the next architecture or to new arithmetic units (tensor cores)
without a rewrite; and the catalogue stays fixed, so any operation the experts did not write
is simply absent. Every kernel pays, again, for the programmer to reason explicitly about
intra-tile thread layout, coalescing and barriers.

**Polyhedral compilers (Tensor Comprehensions, Diesel, Tiramisu, the MLIR affine dialect).**
The idea is to lift a loop nest into an integer-polyhedron representation: the set of loop
iterations is a polyhedron `P = { x : A x + b >= 0 }` cut out by affine inequalities, and a
*schedule* is an affine map `Theta_S(x) = T_S [x; g; 1]` of loop indices `x` and global
parameters `g` that fixes the slow-to-fast traversal order of statement `S`. Loop
transformations — fusion, fission, interchange, tiling, parallelization — become algebraic
transformations of these maps, and the compiler verifies that each preserves semantics. This
is fully automatic from C-like source and reaches library-class performance on dense matrix
multiplication. **Limitation:** two of them. The space of legal schedules grows with the
number of statements and the size of the iteration domain, and checking legality requires
solving integer linear programs, so compilation and the attendant auto-tuning are
expensive — made worse because the right schedule also depends on cache sizes, SM count and
runtime tensor shapes. And the model only applies to *Static Control Parts*: loop bounds and
array subscripts must be affine functions of the loop indices, which holds for regular dense
computations but not for the non-affine indexing of structured-sparse operations, which the
framework therefore cannot express; in practice it also remains slower than the vendor
libraries on many real workloads.

**Scheduling languages (Halide, TVM).** These enforce a separation of concerns at the
grammatical level: the *algorithm* (what is computed, e.g. `C(i,j) += A(i,k) * B(k,j)`) is
written separately from the *schedule* (how to tile, vectorize, unroll, parallelize, stage in
shared memory). The schedule is a tractable, searchable specification, and auto-tuners
(AutoTVM) search over schedules for a fast one. **Limitation:** writing a good schedule is
itself expert work, and the schedules are not performance-portable — a schedule tuned for one
GPU underperforms on another and lags the introduction of new tensor intrinsics, which are
awkward to target and lack portability. The iteration spaces the language imposes are still
rectangular/affine, so many structured-sparse patterns remain inexpressible. And even after
auto-tuning, the generated code is measurably below the hardware's peak, noticeably so for
smaller matrices.

Across all three, the unit over which parallelism is reasoned about is the scalar thread —
either the programmer maps a tile onto threads by hand (CUDA), or the compiler must search/
verify schedules and is confined to affine iteration spaces (polyhedral, scheduling
languages). The cost each pays — manual coalescing and barriers, or expensive legality
checks, or non-portable hand-written schedules, or an inability to express sparsity — is
where the prior art stalls.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Hardware**: NVIDIA GeForce GTX 1070 / GTX 1070 Ti (Pascal) and Tesla V100 (Volta, with
  tensor cores), against vendor libraries cuBLAS (9.2 / 10.0) and cuDNN (7.0) and the DSLs
  AutoTVM, Tensor Comprehensions and PlaidML, each auto-tuned per problem size per its own
  documentation.
- **Matrix multiplication** workloads: square `C = A B^T` for sizes from 128 to ~3072;
  tall-skinny covariance `C = A B^T` with `A, B in R^{M x K}`, `M = N = 64`, large `K`
  (deep reductions); batched MLP inference `C = A B^T` with `A in R^{N x N}`, `B in R^{16 x N}`.
  These come from recurrent (DeepSpeech2) and transformer architectures.
- **Convolution** workloads: dense conv tasks drawn from ResNet and DeepSpeech2 (varying
  `H, W, C, B, K, R, S`), benchmarked via the implicit-GEMM formulation; and shifted
  convolutions as a structured-sparse stress test.
- **Metrics**: throughput in TFLOPS read against the Roofline of the device. In the
  downstream language-model pretraining setting where such a custom kernel is dropped into an
  MLP block, the relevant yardsticks instead are validation cross-entropy loss (lower is
  better) and end-to-end training throughput / elapsed time.

## Code framework

The kernel plugs into an existing model's MLP block. The surrounding machinery already
exists: PyTorch supplies the tensors and the two weight matrices, the autograd engine, and
the two dense matrix multiplications that bracket a pointwise activation; the accelerator
toolchain supplies a way to call a custom GPU routine over a flat buffer. What does *not*
exist yet is the body of that pointwise routine, how it is launched over the hidden tensor,
and how the MLP backward is organized around the pointwise derivative — those are the slots
to be filled.

```python
import torch


def elementwise_kernel(in_ptr, out_ptr, n_elements, BLOCK):
    # TODO: fill in a custom pointwise GPU routine over the flattened buffer.
    pass


class CustomActivationMLP(torch.autograd.Function):
    """The MLP step x -> (linear) -> activation -> (linear) -> out.
    The two linears are ordinary matmuls; the pointwise slot and its derivative
    are open."""

    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        h = x @ w_fc.t()              # first linear (exists)
        # TODO: fill the pointwise step over h.
        act = h
        out = act @ w_proj.t()        # second linear (exists)
        ctx.save_for_backward(x, w_fc, w_proj, h, act)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, h, act = ctx.saved_tensors
        # gradients of the two linears are plain matmuls (exist);
        # TODO: fill the pointwise derivative and combine it with those matmuls.
        pass


def fused_mlp_forward(x, w_fc, w_proj):
    # signature fixed by the harness: x (B*T, n_embd),
    # w_fc (4*n_embd, n_embd), w_proj (n_embd, 4*n_embd) -> (B*T, n_embd)
    return CustomActivationMLP.apply(x, w_fc, w_proj)
```

The single open slot is the elementwise computation and the way the MLP forward/backward is
built around it.
