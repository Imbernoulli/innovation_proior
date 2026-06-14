The activation is the whole point, but it executes inside an FFN, and with the stock activation in
stock ops that FFN is the floor — so the place to start is the cheapest honest baseline: keep the
default activation the scaffold ships, GELU, and just establish how this two-matrix FFN actually runs
on the GPU and what it costs. The default fill is `h = F.gelu(x @ w_fc.t()); return h @ w_proj.t()` —
three operations the way PyTorch sees them: an up-projection matmul, an elementwise GELU on the wide
hidden, a down-projection matmul. Before I change *what* the activation computes, I want to fix *how*
it executes and read off the number it lands, because that number is the bar everything else has to
beat, and because the execution question turns out to be where the first real lever is.

Let me look at the GELU the scaffold actually uses. GELU is the smooth nonlinearity `GELU(x) = x·Φ(x)`
with `Φ` the standard-normal CDF, so exactly `GELU(x) = x · 0.5 · (1 + erf(x/√2))`. The reading is
nice: instead of gating an input by a hard 0/1 step like ReLU, GELU weights `x` by the probability
that a standard Gaussian falls below it — a smooth, stochastic-feeling gate made deterministic. It is
the de-facto choice in BERT/GPT-style LMs, which is exactly why it is the scaffold's default, and so
the right thing to start from: I want the floor to be the activation everyone already trusts, run
correctly, so that any later win is attributable to a *change I made* and not to a baseline I crippled.
In practice I will not call `erf` at all; the cheap form used in these models is the tanh
approximation `GELU(x) ≈ 0.5·x·(1 + tanh[√(2/π)·(x + 0.044715·x³)])`, with `√(2/π) = 0.7978845608…`.
Two reasons it is the right form for a kernel and not just for convenience: it is trivial to implement
with a device `tanh`, and — this matters once I hand-write the backward — its derivative is closed-form
in terms of `tanh`, since `d/dx tanh(u) = 1 − tanh²(u) = sech²(u)`, so I never need `erf`'s derivative
either. Matching the forward and backward to the *same* approximation is the difference between a
correct gradient and a subtly wrong one, so I will commit to the tanh form on both sides.

Now the execution, which is the real content of this rung. Run as three PyTorch ops, the two matmuls
go to cuBLAS and are compute-bound — lots of FLOPs per byte, near peak, nothing to improve there
without changing the math. The activation is the opposite animal. It reads all `M·N` elements of the
wide hidden `h` (with `N = 4·n_embd`) from HBM, computes one `erf`/`tanh`-flavored expression per
element, and writes all `M·N` elements back to HBM, after which the second matmul reads them yet again.
Let me count the bytes the way I would for a bandwidth-bound op, because that exposes the waste. The
activation does `O(M·N)` trivial flops for `M·N` reads plus `M·N` writes; its arithmetic intensity is
at the floor; by the roofline picture its runtime is essentially the time to stream the block's
largest tensor through HBM twice. And that traffic accomplishes nothing the matmul could not have done
in passing — except that here, with the matmuls handed to cuBLAS, the value of every `h` element was
already written out and must be read back. So at minimum I want the activation itself to stop being a
chain of generic elementwise launches over temporaries (one tensor for `x³`, one for the inner
polynomial, one for the `tanh`, one for the product). I want a *single* custom kernel that loads a
chunk of `h` once, forms the cubic, the tanh and the final multiply in registers, and stores the
activated tensor once.

The tool for that is a tile-level kernel — Triton — where I write the body of one *program instance*
that owns a whole contiguous block of the flat tensor, indexed by a program id, and the compiler
recovers the threading, coalescing and vectorization underneath. For a pure elementwise op this is the
simplest possible use of the abstraction: flatten the tensor, give each instance one `BLOCK`-sized
chunk, ask which chunk with `program_id(0)`, build the tile of offsets `pid·BLOCK + arange(0, BLOCK)`,
mask the tail against `n_elements` so the last partial block does not fault, do a masked load, compute
the tanh-GELU, and do a masked store. No `threadIdx`, no shared memory, no hand-written barriers — for
an elementwise op the compiler's passes do the rest. There is one numerical trap I have to respect:
the cubic and the tanh must *not* be evaluated in the input's low-precision dtype. Under the loop's
bf16 autocast, `h` arrives in bf16; the cubic `x³` can overflow at moderate magnitudes in fp16 and is
needlessly coarse in bf16, so I upcast the loaded chunk to fp32, form the polynomial and the tanh in
fp32, and cast the result back to the input dtype before the store. Compute in fp32, store in the
original dtype — the same discipline the surrounding loop already follows for its reductions.

A scope decision falls out here, and it is the one place this rung diverges from the most aggressive
thing the tile abstraction allows. I *could* fuse the up-projection matmul and the activation into one
tiled kernel, running `tanh`-GELU on the matmul's fp32 accumulator in registers before it ever touches
HBM, which would delete the activation's round-trip entirely. But that means hand-writing the matmul,
and the matmul is exactly what cuBLAS (and, here, `torch.compile`'s fused path) does extremely well. At
this first rung I am not trying to beat cuBLAS at its own game; I am establishing the GELU floor with a
*correct, clean* activation kernel and leaving the two matmuls as the well-tuned `x @ w_fc.t()` and
`@ w_proj.t()` torch ops. So the kernel I write fuses *the activation's own elementwise chain* into one
launch over `h` — collapsing four generic kernels and their temporaries into one — and stops there. The
matmuls stay in torch. That is the literal edit: the function does the first matmul, runs the GELU
kernel over `h` as one custom pass, does the second matmul. Whether folding the activation *into* the
matmul epilogue actually pays is a question for a later rung, once I have a number to compare against.

Now the backward, and I want it analytic rather than letting autograd trace through my elementwise
kernel — partly because autograd cannot differentiate the Triton kernel at all, and partly because I
just wrote the activation explicitly and its gradient is just as local. The two linears are easy:
their gradients are matmuls. With `out = act @ w_proj.t()`, the gradient flowing into `act` is
`d_act = grad_out @ w_proj`, and `grad_w_proj = grad_out.t() @ act`; after the activation gradient I
have `d_h`, and then `grad_x = d_h @ w_fc` and `grad_w_fc = d_h.t() @ x`. Let me check the shapes so I
trust them. `out` is `(N_rows, n_embd)`, `act` is `(N_rows, 4·n_embd)`, `w_proj` is
`(n_embd, 4·n_embd)`; `grad_out (N_rows, n_embd) @ w_proj (n_embd, 4·n_embd) → (N_rows, 4·n_embd)`
matches `act`; `grad_out.t() (n_embd, N_rows) @ act (N_rows, 4·n_embd) → (n_embd, 4·n_embd)` matches
`w_proj`. Good. The one piece that needs care is the activation's own derivative. Differentiate
`gelu(x) = 0.5·x·(1 + tanh(inner))` with `inner = c·(x + a·x³)`, `c = √(2/π)`, `a = 0.044715`, by the
product rule:

`d/dx gelu = 0.5·(1 + tanh(inner)) + 0.5·x·sech²(inner)·d(inner)/dx`,

with the chain `d(inner)/dx = c·(1 + 3a·x²)` and `sech² = 1 − tanh²`. Let me sanity-check the limits so
I trust the signs and constants, because a dropped factor here is a silent training bug. As `x → +∞`:
`inner → +∞`, `tanh → 1`, so the first term `→ 0.5·(1+1) = 1`; the second term has `sech² → 0` killing
it, so the derivative `→ 1` — correct, GELU acts like the identity for large positive `x`. As
`x → −∞`: `tanh → −1`, first term `→ 0`, second term `→ 0`, derivative `→ 0` — correct, GELU saturates
to zero on the left. At `x = 0`: `inner = 0`, `tanh 0 = 0`, first term `0.5·(1+0) = 0.5`, second term
`0.5·0·1·c = 0`, so the derivative is `0.5` — the expected slope of GELU at the origin. Signs and
constants check out. So in the backward I save the pre-activation `h` from the forward (cheaper than
recomputing the first matmul), recompute `inner`, `tanh(inner)`, `sech²`, `d_inner` in fp32 on `h`,
assemble `gelu_grad`, and form `d_h = (d_act · gelu_grad)` cast back to the gradient dtype. I also save
`act` itself so the down-projection weight gradient `grad_out.t() @ act` does not have to recompute the
activation. I wrap the whole thing in a `torch.autograd.Function` so forward stashes
`(x, w_fc, w_proj, h, act)` and backward runs exactly this derivation. One dtype discipline: I cast the
saved tensors and the weights to `grad_output`'s dtype consistently in the matmuls, so I never mix a
bf16 weight with an fp32 grad — a small thing that otherwise silently slows the path or NaNs it.

So the step-1 fill is settled and it is, deliberately, the conservative one: the scaffold's GELU,
expressed as one fused-elementwise Triton activation kernel between two ordinary torch matmuls, with an
analytic tanh-GELU backward in a custom autograd Function (the full module is in the answer). No change
to *what* the FFN computes — same activation everyone trusts — only a tighter execution of it and an
explicit backward.

Now reason about what this floor must do, because reading its number is the entire point of running it.
Quality first: I have changed nothing about the function the model computes — it is still GELU to the
precision of the tanh approximation — so I expect `val_loss`, `wikitext2_ppl` and `lambada_ppl` to land
at the unremarkable GELU level. This is by construction the *quality* floor: GELU is asymptotically
linear, so a strongly-firing and a barely-firing unit are pushed through at nearly the same slope, and
the activation adds no super-linear shaping; on LM perplexity GELU is famously only on par with plain
ReLU. So whatever number it posts, I should read it as "this is what the default activation buys, with
no shaping beyond the knee," and expect a later activation change — not a kernel change — to move it.
Throughput second: collapsing the activation's elementwise chain into one fused launch over `h` should
make the activation cheaper than the multi-kernel default, but the dominant cost is still the two
cuBLAS matmuls plus the one HBM round-trip of `h` that I did *not* fuse away (I left the matmuls in
torch), so `elapsed` should be respectable but not extraordinary — the activation round-trip is still
there, just executed in one pass instead of four. The falsifiable expectations, then, are concrete: I
expect this rung to be the *worst on quality* of anything I try (it is unshaped GELU) and *competitive,
likely the best, on throughput* (it leaves the matmuls to cuBLAS and only tidies the activation). If a
later rung lowers `val_loss` while holding throughput near this, that isolates the activation as the
quality lever; if a later rung lowers `val_loss` but *raises* `elapsed`, it tells me fusing the matmul
myself cost more than it saved, and that the cleanest win keeps the matmuls exactly where they are here.
Either way, this GELU number is the reference against which both axes get read.
