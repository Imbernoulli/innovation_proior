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
correct gradient and a subtly wrong one, so I commit to the tanh form on both sides. The tanh form
disagrees with exact `erf` by only a few parts in ten thousand across the range `x` occupies — well
below bf16's own ~`4×10⁻³` ulp on an `O(1)` pre-activation — so it is free on quality while buying a
device-native forward and a clean analytic gradient.

Now the execution, which is the real content of this rung. Run as three PyTorch ops, the two matmuls
go to cuBLAS and are compute-bound — lots of FLOPs per byte, near peak, nothing to improve there
without changing the math. The activation is the opposite animal. It reads all `M·N` elements of the
wide hidden `h` (with `N = 4·n_embd`) from HBM, computes one `tanh`-flavored expression per element,
and writes all `M·N` elements back to HBM, after which the second matmul reads them yet again. Put real
sizes on it: with `n_embd = 1024` the inner width is `N = 4096`, and a single forward micro-batch of 64
sequences of block-size 1024 flattens to `M = 65,536` rows, so `h` holds `M·N ≈ 2.68×10⁸` elements —
about 537 MB in bf16 for *one* FFN's hidden in *one* micro-batch. The activation reads that 537 MB and
writes another 537 MB, ~1.07 GB of HBM traffic per FFN per micro-batch; across the 24 layers that is
~25.8 GB, ~12.9 ms of pure bandwidth time on a ~2 TB/s bus spent doing nothing but pushing `h` in and
out of memory. Its arithmetic intensity is at the floor — a handful of flops per element against a
read-plus-write — so by the roofline picture its runtime is essentially the time to stream the block's
largest tensor through HBM twice. And the default makes that worse than it needs to be by expressing
the one pass as a *chain* of generic elementwise launches over temporaries — one tensor for `x³`, one
for the inner polynomial, one for the `tanh`, one for the product — each restreaming the full wide
hidden again. So at minimum I want the activation itself to stop being a chain of generic launches. I
want a *single* custom kernel that loads a chunk of `h` once, forms the cubic, the tanh and the final
multiply in registers, and stores the activated tensor once.

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
needlessly coarse in bf16 (cubing an already-rounded value compounds the error into the polynomial and
the tanh), so I upcast the loaded chunk to fp32, form the polynomial and the tanh in fp32, and cast the
result back to the input dtype before the store. Compute in fp32, store in the original dtype — the
same discipline the surrounding loop already follows for its reductions.

`BLOCK = 1024` is the unremarkable sweet spot: a power of two for clean masking and coalesced 128-bit
loads, with `n ≈ 2.68×10⁸` giving a grid of ~262,144 instances — vastly more than the ~108 SMs, so the
scheduler has many waves to hide launch latency and memory stalls, exactly what a bandwidth-bound
kernel wants — yet few enough elements per instance to keep the fp32 register footprint cheap. For this
exact shape `n` divides evenly by 1024 so the tail mask never fires, but I keep the
`mask = offsets < n_elements` guard anyway: the function must stay correct for any `M` where the last
block is partial, and a masked load costs nothing when the mask is all-true.

Before I commit to writing this kernel at all, I owe myself one honest reckoning, because the loop runs
under `torch.compile` and collapsing a pointwise chain into a single kernel is precisely what its
inductor backend does unprompted. The four-launch picture I drew is the *eager* decomposition; hand
that same expression to inductor and it fuses the pointwise ops into one Triton kernel over `h`, the
fp32 upcast and downcast included, without my writing a line. So I must not tell myself the hand kernel
buys a four-to-one traffic collapse over the *compiled* baseline: against eager it would, but against
the default the loop actually runs, the activation is already close to a single fused pass. Its
throughput edge over the compiled GELU is therefore near zero, and pretending otherwise would poison
every later comparison. What the kernel is really for is the two things inductor's automatic fusion
does *not* hand me. First, it pins the throughput reference to an execution I control and can reason
about tile by tile, so a later rung's delta is read against a fixed, understood floor rather than
whatever schedule the compiler happens to emit. Second — the load-bearing reason to write it now rather
than leave `F.gelu` untouched — it forces the explicit `torch.autograd.Function` with a hand-written
backward into existence, and that harness is what every later activation swap plugs into: the moment I
want an analytic backward for a custom activation, or to run the activation on the matmul's fp32
accumulator before it ever reaches HBM, I am outside what the compiler will do for me and I need this
machinery already standing. So I write the kernel for the backward and the reference, not for a speedup
I would only be imagining.

A scope decision falls out here, and it is the one place this rung diverges from the most aggressive
thing the tile abstraction allows. I *could* fuse the up-projection matmul and the activation into one
tiled kernel, running `tanh`-GELU on the matmul's fp32 accumulator in registers before it ever touches
HBM, which would delete the activation's round-trip entirely — that ~12.9 ms of per-micro-batch
bandwidth time. But that means hand-writing the matmul, and the matmul is exactly what cuBLAS (and
`torch.compile`'s fused path) does extremely well: compute-bound, autotuned, tensor-core-saturating. At
this first rung I am not trying to beat cuBLAS at its own game; I am establishing the GELU floor with a
*correct, clean* activation kernel and leaving the two matmuls as the well-tuned `x @ w_fc.t()` and
`@ w_proj.t()` torch ops. So the kernel fuses *the activation's own elementwise chain* into one launch
over `h` and stops there. That is the literal edit: the function does the first matmul, runs the GELU
kernel over `h` as one custom pass, does the second matmul. Whether folding the activation *into* the
matmul epilogue actually pays — whether the round-trip I leave on the table is worth more than the
cuBLAS-versus-hand-rolled matmul gap I would take on — is a real question, but one I can only answer
once I have this floor's number to compare against, so I defer it deliberately rather than guess.

Now the backward, and I want it analytic rather than letting autograd trace through my elementwise
kernel — partly because autograd cannot differentiate the Triton kernel at all, and partly because I
just wrote the activation explicitly and its gradient is just as local. The two linears are easy:
their gradients are matmuls. With `out = act @ w_proj.t()`, the gradient flowing into `act` is
`d_act = grad_out @ w_proj`, and `grad_w_proj = grad_out.t() @ act`; after the activation gradient I
have `d_h`, and then `grad_x = d_h @ w_fc` and `grad_w_fc = d_h.t() @ x`. Checking shapes: `out` is
`(N_rows, n_embd)`, `act` is `(N_rows, 4·n_embd)`, `w_proj` is `(n_embd, 4·n_embd)`;
`grad_out (N_rows, n_embd) @ w_proj (n_embd, 4·n_embd) → (N_rows, 4·n_embd)` matches `act`, and
`grad_out.t() @ act → (n_embd, 4·n_embd)` matches `w_proj`. The one piece that needs care is the
activation's own derivative. Differentiate `gelu(x) = 0.5·x·(1 + tanh(inner))` with
`inner = c·(x + a·x³)`, `c = √(2/π)`, `a = 0.044715`, by the product rule:

`d/dx gelu = 0.5·(1 + tanh(inner)) + 0.5·x·sech²(inner)·d(inner)/dx`,

with the chain `d(inner)/dx = c·(1 + 3a·x²)` and `sech² = 1 − tanh²`. Sanity-check the limits, since a
dropped factor here is a silent training bug. As `x → +∞`: `tanh → 1`, first term `→ 1`, `sech² → 0`
kills the second, derivative `→ 1` — GELU acts like the identity for large positive `x`. As `x → −∞`:
`tanh → −1`, both terms `→ 0`, derivative `→ 0` — GELU saturates to zero on the left. At `x = 0`:
`inner = 0`, first term `0.5`, second term `0`, derivative `0.5` — the expected slope at the origin.
Signs and constants check out. So in the backward I save the pre-activation `h` from the forward
(cheaper than recomputing the first matmul), recompute `inner`, `tanh(inner)`, `sech²`, `d_inner` in
fp32 on `h`, assemble `gelu_grad`, and form `d_h = (d_act · gelu_grad)` cast back to the gradient
dtype. I also save `act` itself so the down-projection weight gradient `grad_out.t() @ act` does not
have to recompute the activation. Stashing both `h` and `act` — each `M·N` bf16, ~537 MB per layer per
micro-batch — costs ~1.07 GB of live activation memory per FFN held across the step; I take that here
because tanh is a transcendental and the point of the floor is a clean, unhurried reference, not a
memory-minimized one. The day the activation is cheap enough that recomputing it costs less than
holding it, saving only the pre-activation becomes the right call — but that is a trade for when the
activation *is* cheap, not now. I wrap the whole thing in a `torch.autograd.Function` so forward
stashes `(x, w_fc, w_proj, h, act)` and backward runs exactly this derivation, casting saved tensors
and weights to `grad_output`'s dtype consistently in the matmuls so I never mix a bf16 weight with an
fp32 grad.

So the step-1 fill is settled and it is, deliberately, the conservative one: the scaffold's GELU,
expressed as one fused-elementwise Triton activation kernel between two ordinary torch matmuls, with an
analytic tanh-GELU backward in a custom autograd Function (the full module is in the answer). No change
to *what* the FFN computes — same activation everyone trusts — only a tighter execution of it and an
explicit backward.

Now reason about what this floor must do, because reading its number is the entire point of running it.
Quality first: I have changed nothing about the function the model computes — it is still GELU to the
precision of the tanh approximation — so I expect `val_loss`, `wikitext2_ppl` and `lambada_ppl` to land
at the unremarkable GELU level, and the downstream zero-shot numbers at baseline GELU accuracies. This
is by construction the *quality* floor: GELU is asymptotically linear, so a strongly-firing and a
barely-firing unit are pushed through at nearly the same slope, and the activation adds no super-linear
shaping; on LM perplexity GELU is famously only on par with plain ReLU. So whatever number it posts, I
read it as "this is what the default activation buys, with no shaping beyond the knee," and expect a
later *activation* change — not a kernel change — to move it. I deliberately do not have a target value
in hand; the point of the rung is to manufacture the reference, not to hit a preconceived one.
Throughput second: collapsing the activation's chain into one fused launch should make the activation
cheaper than the multi-kernel default, but the dominant cost is still the two cuBLAS matmuls plus the
one HBM round-trip of `h` I did *not* fuse away, so `elapsed` should be respectable but not
extraordinary. The falsifiable expectations are concrete: this rung should be the *worst on quality* of
anything I try (unshaped GELU) and *competitive, likely the best, on throughput* (it leaves the matmuls
to cuBLAS and only tidies the activation). If a later rung lowers `val_loss` while holding throughput
near this, that isolates the activation as the quality lever; if a later rung lowers `val_loss` but
*raises* `elapsed`, it tells me fusing the matmul myself cost more than it saved, and that the cleanest
win keeps the matmuls exactly where they are here.
