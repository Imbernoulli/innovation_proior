The activation is what I ultimately want to design, but it never runs in isolation — it sits between two dense matmuls inside an FFN that dominates the per-step compute of a GPT-2-class language model, and before I change *what* that activation computes I need an honest reference for how the two-matrix slot runs on the GPU and what it costs. The scaffold ships GELU through plain PyTorch ops, `h = F.gelu(x @ w_fc.t()); return h @ w_proj.t()`, which is the right floor to start from precisely because GELU is the activation everyone already trusts in BERT/GPT-style models: if I run it correctly and read off its number, any later improvement is attributable to a change I made and not to a baseline I crippled. So this first rung keeps GELU unchanged and only tightens its *execution*, to fix the reference on both axes — quality per step and wall-clock per step.

What I propose for rung 1 is a **Triton fused-GELU activation kernel** between two ordinary torch matmuls, with an analytic tanh-GELU backward in a custom autograd Function. The activation I keep is the cheap tanh approximation the field actually uses, $\mathrm{GELU}(x) \approx 0.5\,x\,(1 + \tanh[\sqrt{2/\pi}\,(x + 0.044715\,x^3)])$ with $\sqrt{2/\pi} = 0.7978845608\ldots$, rather than the exact $x\,\Phi(x) = 0.5\,x\,(1+\mathrm{erf}(x/\sqrt2))$. The tanh form is the right one for a kernel for two reasons: it is trivial to implement with a device `tanh`, and its derivative is closed-form *in terms of `tanh`* — since $\frac{d}{du}\tanh(u) = 1-\tanh^2(u) = \mathrm{sech}^2(u)$ — so when I hand-write the backward I never touch `erf`'s derivative either. Matching forward and backward to the *same* approximation is the difference between a correct gradient and a subtly wrong one, so I commit to the tanh form on both sides.

The real content of this rung is the execution, and the reason it matters is a roofline argument. Run as three PyTorch ops, the two matmuls go to cuBLAS and are compute-bound — near peak, nothing to improve without changing the math. The activation is the opposite animal: it reads all $M\cdot N$ elements of the wide hidden $h$ (with $N = 4\cdot\texttt{n\_embd}$) from HBM, does a handful of trivial flops per element, and writes all $M\cdot N$ back, after which the second matmul reads them yet again. Its arithmetic intensity is at the floor, so its runtime is essentially the time to stream the block's largest tensor through HBM. Worse, the default expresses that one pass as a *chain* of generic elementwise launches over temporaries — one tensor for $x^3$, one for the inner polynomial, one for the `tanh`, one for the product. So at minimum I want a *single* custom kernel that loads a chunk of $h$ once, forms the cubic, the `tanh`, and the final multiply in registers, and stores the activated tensor once, collapsing four kernels and their temporaries into one launch.

Triton is the tool for that: I write the body of one *program instance* that owns a contiguous `BLOCK`-sized chunk of the flat tensor, and the compiler recovers threading, coalescing, and vectorization underneath. For a pure elementwise op this is the simplest possible use of the abstraction — flatten the tensor, ask which chunk with `program_id(0)`, build the offset tile `pid*BLOCK + arange(0, BLOCK)`, mask the tail against `n_elements` so the last partial block does not fault, do a masked load, compute the tanh-GELU, do a masked store. No `threadIdx`, no shared memory, no hand-written barriers. There is one numerical trap I must respect: under the loop's bf16 autocast $h$ arrives in bf16, and the cubic $x^3$ can overflow at moderate magnitudes in low precision and is needlessly coarse in bf16. So I upcast the loaded chunk to fp32, form the polynomial and the `tanh` in fp32, and cast back to the input dtype before the store — compute in fp32, store in the original dtype, the same discipline the surrounding loop already follows for its reductions. I choose `BLOCK_SIZE = 1024`: a power of two, enough work per instance to amortize the launch and saturate the bus, small enough to keep occupancy.

There is a deliberate scope decision here, and it is the one place I hold back from the most aggressive thing the tile abstraction allows. I *could* fuse the up-projection matmul and the activation into one tiled kernel, running tanh-GELU on the matmul's fp32 accumulator in registers before it ever touches HBM, deleting the activation's round-trip entirely. But that means hand-writing the matmul, and the matmul is exactly what cuBLAS — and `torch.compile`'s fused path here — does extremely well. At this first rung I am not trying to beat cuBLAS at its own game; I am establishing the GELU floor with a *correct, clean* activation kernel and leaving the two matmuls as the well-tuned `x @ w_fc.t()` and `@ w_proj.t()` torch ops. So the kernel fuses *the activation's own elementwise chain* into one launch over $h$ and stops there. Whether folding the activation into the matmul epilogue actually pays is a question for a later rung, once I have a number to compare against.

The backward I want analytic rather than traced, partly because autograd cannot differentiate the Triton kernel at all, and partly because I just wrote the activation explicitly and its gradient is just as local. The two linears are matmuls: with `out = act @ w_proj.t()`, the gradient into `act` is `d_act = grad_out @ w_proj` and `grad_w_proj = grad_out.t() @ act`; after the activation gradient gives `d_h`, then `grad_x = d_h @ w_fc` and `grad_w_fc = d_h.t() @ x`. The one piece needing care is the activation's own derivative. Differentiating $\mathrm{gelu}(x) = 0.5\,x\,(1+\tanh(\text{inner}))$ with $\text{inner} = c\,(x + a\,x^3)$, $c=\sqrt{2/\pi}$, $a=0.044715$, by the product rule gives

$$\frac{d}{dx}\mathrm{gelu} = 0.5\,(1+\tanh(\text{inner})) + 0.5\,x\,\mathrm{sech}^2(\text{inner})\cdot c\,(1 + 3a\,x^2),$$

using $\mathrm{sech}^2 = 1-\tanh^2$ and the chain factor $d(\text{inner})/dx = c\,(1+3a\,x^2)$. The limits confirm the constants: as $x\to+\infty$, $\tanh\to1$ so the first term $\to1$ while $\mathrm{sech}^2\to0$ kills the second, giving derivative $\to1$ (GELU is the identity for large positive $x$); as $x\to-\infty$ both terms $\to0$, giving $0$ (left saturation); at $x=0$, the first term is $0.5\,(1+0)=0.5$ and the second vanishes, the expected slope at the origin. So in the backward I save the pre-activation $h$ from the forward (cheaper than recomputing the first matmul), recompute `inner`, `tanh(inner)`, `sech2`, `d_inner` in fp32 on $h$, assemble `gelu_grad`, and form `d_h = (d_act · gelu_grad)` cast back to the gradient dtype. I also save `act` itself so `grad_w_proj = grad_out.t() @ act` need not recompute the activation. One dtype discipline runs through it: I cast the saved tensors and weights to `grad_output`'s dtype consistently in the matmuls, so a bf16 weight never meets an fp32 grad — a small thing that otherwise silently slows or NaNs the path.

This fill is, by design, the conservative one — the scaffold's GELU run correctly, only its execution tightened — and that is the point. Quality should land at the unremarkable GELU level: GELU is asymptotically linear, so a strongly-firing and a barely-firing unit pass through at nearly the same slope, the activation adds no super-linear shaping, and on LM perplexity GELU is famously only on par with plain ReLU. This is the *quality* floor by construction, and I expect a later activation change — not a kernel change — to move it. Throughput should be respectable but not extraordinary: collapsing the elementwise chain into one launch makes the activation cheaper than the multi-kernel default, but the dominant cost is still the two cuBLAS matmuls plus the single HBM round-trip of $h$ that I deliberately did *not* fuse away. So the falsifiable read is concrete — this rung should be the worst on quality of anything I try and likely the best on throughput — and whatever numbers it posts become the reference against which both axes get read.

```python
# EDITABLE region of custom_pretrain.py (lines 33-48) — step 1: Triton fused GELU
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _fused_gelu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute entirely in float32 to avoid bfloat16 overflow in x^3
    xf = x.to(tl.float32)
    # tanh-approximation GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (xf + 0.044715 * xf * xf * xf)
    tanh_val = libdevice.tanh(inner)
    out = xf * 0.5 * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)

class _TritonGELUMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        h = x @ w_fc.t()
        act = torch.empty_like(h)
        n = h.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
        _fused_gelu_kernel[grid](h, act, n, BLOCK_SIZE=BLOCK)
        out = act @ w_proj.t()
        ctx.save_for_backward(x, w_fc, w_proj, h, act)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, h, act = ctx.saved_tensors
        dtype = grad_output.dtype
        d_act = grad_output @ w_proj.to(dtype)
        grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ act.to(dtype).reshape(-1, act.shape[-1])
        # Analytical gradient of tanh-approximation GELU (matches the Triton forward)
        # gelu(x) = 0.5 * x * (1 + tanh(inner)), inner = c * (x + 0.044715 * x^3)
        # d_gelu/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner/dx
        # d_inner/dx = c * (1 + 3 * 0.044715 * x^2)
        h_f = h.float()
        c = 0.7978845608028654
        inner = c * (h_f + 0.044715 * h_f * h_f * h_f)
        tanh_inner = torch.tanh(inner)
        sech2 = 1.0 - tanh_inner * tanh_inner
        d_inner = c * (1.0 + 3.0 * 0.044715 * h_f * h_f)
        gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * h_f * sech2 * d_inner
        d_h = (d_act.float() * gelu_grad).to(dtype)
        grad_x = d_h @ w_fc.to(dtype)
        grad_w_fc = d_h.reshape(-1, d_h.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])
        return grad_x, grad_w_fc, grad_w_proj

def fused_mlp_forward(x, w_fc, w_proj):
    """MLP forward with Triton fused GELU kernel."""
    return _TritonGELUMLP.apply(x, w_fc, w_proj)
```
