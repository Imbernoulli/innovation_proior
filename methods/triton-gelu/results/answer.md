# Triton, distilled

Triton is a language and compiler for writing custom GPU kernels in which the **tile** — a
statically-shaped multi-dimensional sub-array — is a first-class value. A kernel is written
*single-threaded but blocked*: one program instance owns a whole tile of work, identified by a
program id, and the compiler automatically parallelizes that tile across the GPU's SIMD lanes
and tensor cores. This removes the per-thread bookkeeping (manual coalescing, shared-memory
staging, barriers) that hand-written CUDA forces, without confining the user to the
affine-only iteration spaces of polyhedral compilers or the non-portable hand-written
schedules of Halide/TVM. Because a program instance holds an entire tile, **fusing** a chain
of operations into one kernel is natural — the key win for bandwidth-bound operations such as
elementwise activations, which otherwise pay the memory wall by round-tripping intermediates
through DRAM.

## Problem it solves

Operations outside the fixed vendor-library catalogue (cuBLAS/cuDNN) get poor GPU utilization
unless an expert hand-writes a CUDA/PTX kernel — months of architecture-specific, non-portable
work. The goal is a high-level, portable way to write a custom kernel that compiles to code on
par with hand-tuned libraries, can express structured-sparse indexing, and makes operator
fusion easy.

## Key idea

Lift the atom of parallelism from the scalar thread to the tile.

- **Tile as first-class value.** Programs operate on whole multi-dimensional sub-arrays
  (`float A[16,16]`, `dot(A, trans(B))`), not on scalars distributed across threads.
- **Single-threaded, blocked SPMD.** Each kernel instance is single-threaded and owns one tile,
  indexed by `tl.program_id(axis)` in modern kernels; instances run in parallel on a launch grid. No `threadIdx`,
  no `__syncthreads`, no manual coalescing in the source.
- **Block-level IR + compiler do the rest.** An LLVM-based IR with tile types (`f32<M,N>`), SSA,
  numpy broadcasting (`reshape`/`broadcast`), `dot`/`trans`, and Predicated SSA with `psi`
  merges for intra-tile control flow. Modern Python kernels expose the same bounds-guard idea
  as masks on `tl.load`, `tl.store`, and selection. On that IR, standard data-flow analysis
  recovers everything the programmer used to hand-author.

## Why these design choices

- **Block, not thread**: removes the intra-tile thread-layout problem that forces manual
  coalescing/barriers in CUDA, and avoids the affine-iteration-space restriction that blocks
  polyhedral/scheduling approaches from sparse operations.
- **Predicated tile control flow, surfaced as masks**: tile ops are atomic, so per-element
  branching is impossible; the IR needs predicate streams and `psi` merges, while the Python
  kernel expresses ordinary bounds guards with compact `mask=` arguments on loads and stores.
- **LLVM-based IR with tile SSA**: makes block-level data-flow analysis possible, which is what
  recovers parallelism, coalescing, shared-memory placement and barriers automatically.
- **Tunable tile sizes + auto-tuner**: the one irreducibly empirical knob (best tile shape
  depends on SM count, cache sizes, register pressure, and runtime tensor shapes) is isolated
  into a search whose space the IR's passes expose.

## The compiler passes (all from block-level data-flow)

- **Simplification (peephole on tile algebra)**: `(X^T)^T = X`; collapse chained reductions into
  one reduction over a reshaped array.
- **Parallelizing computation (hierarchical sub-blocking)**: split tile -> micro-tile ->
  nano-tile to fit the core; choose fragment shape per op — vectorize elementwise (e.g. 1x4),
  *tensorize* FP16 matmul (e.g. 2x2x2) onto tensor cores.
- **Coalescing (contiguity analysis)**: statically detect the contiguous axis of a tile of
  addresses and order SIMD lanes along it, so adjacent lanes hit adjacent addresses.
- **Shared-memory allocation**: stage a tile to shared memory iff it feeds an
  arithmetically-intense op (`alpha(v) = comp(v) / sum_{p in pred(v)} mem(p)` high); compute
  live ranges by liveness data-flow, then a linear-time static storage allocator assigns offsets.
- **Shared-memory synchronization (barriers)**: forward data-flow over pending read-after-write
  and write-after-read buffers,

  ```
  IN_RAW(s)  = union_{p in pred(s)} OUT_RAW(p)
  OUT_RAW(s) = {}                       if IN_RAW(s) ∩ read(s)  != {}   (emit barrier)
             = IN_RAW(s) ∪ write(s)     otherwise
  IN_WAR(s)  = union_{p in pred(s)} OUT_WAR(p)
  OUT_WAR(s) = {}                       if IN_WAR(s) ∩ write(s) != {}   (emit barrier)
             = IN_WAR(s) ∪ read(s)      otherwise
  ```

  inserting exactly the barriers the hazards require.

## Worked example: a fused GELU activation

Fusion is the payoff for bandwidth-bound ops. In an MLP block `x -> linear -> GELU -> linear`,
the two linears stay ordinary dense matmuls, but the activation slot is still a wide
pointwise pass. The concrete kernel fuses the tanh-approximation arithmetic for GELU into
one elementwise launch: read `h` once, form the cubic, tanh, and final multiply in fp32, and
write the activated tensor once.

GELU is `GELU(x) = x*Phi(x) = x*0.5*(1+erf(x/sqrt(2)))`; the kernel uses the
tanh approximation
`0.5*x*(1 + tanh[ sqrt(2/pi) * (x + 0.044715*x^3) ])`, with `sqrt(2/pi) = 0.7978845608028654`.
The tanh form is easy to implement with device `tanh` and has a closed-form derivative; the
polynomial/tanh path is computed in float32 for stable low-precision execution and then cast back.

Analytic derivative (used for the backward), with `inner = c(x + a*x^3)`, `c = sqrt(2/pi)`,
`a = 0.044715`:

```
d/dx GELU = 0.5*(1 + tanh(inner)) + 0.5*x*sech^2(inner)*d_inner,
            d_inner = c*(1 + 3*a*x^2),   sech^2 = 1 - tanh^2(inner)
```

Limit checks: `x -> +inf` gives slope `-> 1` (identity); `x -> -inf` gives `-> 0` (saturates);
`x = 0` gives `0.5`.

For the MLP, with `x : (M, K)`, `w_fc : (N, K)`, `w_proj : (D, N)`, and
`g = dL/dout : (M, D)`:

```
h           = x @ w_fc.T
act         = GELU_tanh(h)
out         = act @ w_proj.T
d_act       = g @ w_proj
grad_w_proj = g.T @ act
d_h         = d_act * GELU_tanh'(h)
grad_x      = d_h @ w_fc
grad_w_fc   = d_h.T @ x
```

## Working code

A Triton elementwise GELU kernel inside the MLP forward, with an analytic backward. The two
linears are ordinary matmuls (their gradients are matmuls); the kernel handles the pointwise
activation. Filling the open slot of the MLP harness:

```python
import torch

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fused_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                               # which program instance / tile chunk
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                          # guard the tail
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # compute the polynomial and tanh in fp32, then cast back
    xf = x.to(tl.float32)
    c = 0.7978845608028654                               # sqrt(2/pi)
    inner = c * (xf + 0.044715 * xf * xf * xf)
    tanh_val = libdevice.tanh(inner)
    out = xf * 0.5 * (1.0 + tanh_val)                    # 0.5 x (1 + tanh(inner))
    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)


class _TritonGELUMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        h = x @ w_fc.t()                                 # first linear
        act = torch.empty_like(h)
        n = h.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)                 # one instance per chunk
        _fused_gelu_kernel[grid](h, act, n, BLOCK_SIZE=BLOCK)
        out = act @ w_proj.t()                           # second linear
        ctx.save_for_backward(x, w_fc, w_proj, h, act)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, h, act = ctx.saved_tensors
        dtype = grad_output.dtype
        g = grad_output.reshape(-1, grad_output.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        act_flat = act.reshape(-1, act.shape[-1])
        d_act = grad_output @ w_proj.to(dtype)             # g @ W_proj
        grad_w_proj = g.t() @ act_flat.to(dtype)
        # analytic tanh-GELU derivative (matches the forward), in fp32
        h_f = h.float()
        c = 0.7978845608028654
        inner = c * (h_f + 0.044715 * h_f * h_f * h_f)
        tanh_inner = torch.tanh(inner)
        sech2 = 1.0 - tanh_inner * tanh_inner
        d_inner = c * (1.0 + 3.0 * 0.044715 * h_f * h_f)
        gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * h_f * sech2 * d_inner
        d_h = (d_act.float() * gelu_grad).to(dtype)
        grad_x = d_h @ w_fc.to(dtype)
        grad_w_fc = d_h.reshape(-1, d_h.shape[-1]).t() @ x_flat.to(dtype)
        return grad_x, grad_w_fc, grad_w_proj


def fused_mlp_forward(x, w_fc, w_proj):
    """MLP forward with a Triton tanh-GELU elementwise kernel between the two linears."""
    return _TritonGELUMLP.apply(x, w_fc, w_proj)
```

A bare elementwise Triton kernel (the canonical skeleton this builds on) has the same shape —
`program_id` -> `offsets` -> `mask` -> `tl.load` -> compute -> `tl.store` — and is launched with
`grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`; tensors are passed as pointers to their first
element, and the jitted function indexed by the grid yields the launchable kernel.
