The activation is the whole point, but it executes inside an FFN, and with the stock activation in
stock ops that FFN is the floor — so the place to start is the cheapest honest baseline: keep the
default activation the scaffold ships, GELU, and just establish how this two-matrix FFN actually runs
on the GPU and what it costs. The default fill is `h = F.gelu(x @ w_fc.t()); return h @ w_proj.t()` —
three operations the way PyTorch sees them: an up-projection matmul, an elementwise GELU on the wide
hidden, a down-projection matmul. Before I change *what* the activation computes, I want to fix *how*
it executes and read off the number it lands, because that number is the bar everything else has to
beat, and because the execution question turns out to be where the first real lever is. The bill I am
ultimately paying is `steps × step-time`, and the steps are fixed at 13,535, so the only thing I can
move at this rung is step-time, and the only part of step-time I am allowed to touch is what happens
inside `fused_mlp_forward` — the two matmuls and the pointwise operation wedged between them.

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
correct gradient and a subtly wrong one, so I will commit to the tanh form on both sides. The exact
`erf` form and the tanh form disagree by only a few parts in ten thousand across the range where `x`
actually lives — concretely, at `x = 2` the exact `2·Φ(2) = 1.9545` sits against the tanh form's
`1.9546`, a gap of ~`1×10⁻⁴` that widens to only a few parts in ten thousand out near `x ≈ 2.5` where
the two forms part most. That gap is *below bf16's own rounding* of `h`: bf16 carries ~8 mantissa bits,
so one ulp on an `O(1)` pre-activation is ~`4×10⁻³`, forty-odd times the approximation error — the
tanh-versus-`erf` choice is buried under the precision the loop already spends, genuinely free on
quality, while buying me a device-native forward and a clean analytic gradient. An unambiguous trade.

There is a prior question I should settle before writing any kernel at all: do I even need one for a
floor? The literal minimum is to leave `F.gelu` exactly as PyTorch ships it and read its number. But
the floor's job is to fix *two* references — quality per step and wall-clock per step — and those two
have very different sensitivities to what I do here. Quality is pinned by the *math*: GELU is GELU
regardless of how I schedule the arithmetic, so for the quality axis a plain `F.gelu` and a hand-written
kernel land at the same place, and I gain nothing by writing a kernel. Throughput is the axis that is
genuinely open, and it is open in a way I want to *measure and isolate*: I want to know how much I can
tighten the activation's execution *without touching the two matmuls*, because that number cleanly
separates "how much does executing the activation better buy" from "how much does changing the
activation buy," and the later rungs need exactly that separation to attribute their wins. And there is
a second, structural reason to build the kernel scaffolding now: I want an explicit
`torch.autograd.Function` with a hand-written backward, because that harness is what every later
activation swap plugs into — once the forward/backward plumbing exists, changing the activation is a
few lines in one place. So the kernel at this rung is not overkill dressed up as a baseline; it is a
deliberate investment that fixes the throughput reference tightly and stands up the machinery I will
reuse. I keep the *math* untouched and invest only in the *execution* and the backward.

Now the execution, which is the real content of this rung. Run as three PyTorch ops, the two matmuls
go to cuBLAS and are compute-bound — lots of FLOPs per byte, near peak, nothing to improve there
without changing the math. Let me make "compute-bound" quantitative so I trust the decision to leave
them alone. The up-projection does `2·M·N·K ≈ 5.5×10¹¹` FLOPs while touching `x` (`M·K`), `w_fc`
(`N·K`) and `h` (`M·N`) — in bf16 that is `2·(M·K + N·K + M·N) ≈ 6.8×10⁸` bytes, an arithmetic
intensity of ~810 flops/byte. The machine's roofline ridge sits near ~156 flops/byte, so the matmul
lives a factor of ~5 *past* the ridge, deep in the compute-bound regime where cuBLAS already runs near
peak and there is no memory slack for me to reclaim. That is the concrete reason the matmuls are not my
lever at this rung: hand-rolling them could only match, not beat, a vendor kernel that is already
saturating the tensor cores. The activation is the opposite animal. It reads all `M·N` elements of the
wide hidden `h` (with `N = 4·n_embd`) from HBM, computes one `erf`/`tanh`-flavored expression per
element, and writes all `M·N` elements back to HBM, after which the second matmul reads them yet again.
Let me put real sizes on this, because the ratios are what expose the waste. With `n_embd = 1024` the
inner width is `N = 4096`; a single forward micro-batch of 64 sequences of block-size 1024 flattens to
`M = 64·1024 = 65,536` rows. So `h` holds `M·N = 65,536 × 4096 ≈ 2.68×10⁸` elements — about 537 MB in
bf16 for *one* FFN's hidden, in *one* micro-batch. The activation reads that 537 MB and writes another
537 MB, roughly 1.07 GB of HBM traffic per FFN per micro-batch; across the model's 24 layers that is
~25.8 GB of streaming, which on a ~2 TB/s bus is ~12.9 ms of pure bandwidth time per micro-batch spent
doing nothing but pushing `h` in and out of memory. Set that against the arithmetic: the up-projection
alone is `2·M·N·K ≈ 5.5×10¹¹` FLOPs (`K = n_embd = 1024`), and the activation is on the order of
`10·M·N ≈ 2.7×10⁹` FLOPs — the matmul does roughly 200× the arithmetic of the activation, but the
activation costs a comparable slice of *memory* time because its arithmetic intensity is at the floor:
about 10 flops per element against 4 bytes of read-plus-write, ~2.5 flops/byte, when the machine's
roofline ridge sits above a hundred flops/byte. So the activation lives far out on the bandwidth-bound
side of the roofline; its runtime is essentially the time to stream the block's largest tensor through
HBM. And the default makes that worse than it needs to be by expressing the one pass as a *chain* of
generic elementwise launches over temporaries — one tensor for `x³`, one for the inner polynomial, one
for the `tanh`, one for the product — each of which reads and writes the full wide hidden again. So at
minimum I want the activation itself to stop being a chain of generic launches. I want a *single*
custom kernel that loads a chunk of `h` once, forms the cubic, the tanh and the final multiply in
registers, and stores the activated tensor once — collapsing four kernels and their temporaries into
one.

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
needlessly coarse in bf16 (bf16 carries only ~8 bits of mantissa, so cubing an already-rounded value
compounds the error into the polynomial and the tanh), so I upcast the loaded chunk to fp32, form the
polynomial and the tanh in fp32, and cast the result back to the input dtype before the store. Compute
in fp32, store in the original dtype — the same discipline the surrounding loop already follows for its
reductions.

I have to pick `BLOCK`, and the choice is an occupancy-versus-overhead argument I can make concretely.
Each instance loads `BLOCK` bf16 elements, upcasts them to fp32 for the compute, so the live register
footprint scales with `BLOCK`; the number of instances is `grid = cdiv(n_elements, BLOCK)`. With
`BLOCK = 1024` and `n = 2.68×10⁸`, the grid is ~262,144 program instances — vastly more than the
GPU's ~108 SMs, so the scheduler has many waves to hide launch latency and memory stalls, which is
exactly what a bandwidth-bound kernel wants. Push `BLOCK` far smaller (say 128) and I multiply the
instance count and the per-instance launch/index overhead for no bandwidth gain; push it far larger
(say 8192) and the fp32 register footprint per instance climbs and occupancy drops, starving the bus
of in-flight requests. `BLOCK = 1024` is a power of two — clean masking, natural alignment for
coalesced 128-bit loads — with enough elements per instance to amortize the launch and saturate the
bus and few enough to keep registers cheap. It is the unremarkable sweet spot, and for an elementwise
op there is little to gain from agonizing past that. Worth noting that for this exact shape the tail
mask never actually fires — `n = 2.68×10⁸` is `262,144 × 1024` on the nose, so the grid divides evenly
— but I keep the `mask = offsets < n_elements` guard anyway, because the function must stay correct for
any `M` (a short final micro-batch, or a different width) where the last block *is* partial, and a
masked load costs nothing when the mask is all-true. Correctness for free is worth the one comparison.

Before I commit to writing this kernel at all, I owe myself one honest reckoning, because the loop runs
under `torch.compile` and collapsing a pointwise chain into a single kernel is precisely what its
inductor backend does unprompted. The four-launch picture I drew — `x³`, the inner polynomial, the
`tanh`, the final product, each a generic elementwise kernel restreaming the full `h` — is the *eager*
decomposition; hand that same expression to inductor and it fuses the pointwise ops into one Triton
kernel over `h`, the fp32 upcast and downcast included, without my writing a line. So I must not tell
myself the hand kernel buys a four-to-one traffic collapse over the *compiled* baseline: against eager
it would, but against the default the loop actually runs, the activation is already close to a single
fused pass. That reframes, honestly, what this rung's kernel is for. Its throughput edge over the
compiled GELU is near zero — one fused activation pass either way — and pretending otherwise would
poison every later comparison. Its value is the two things inductor's automatic fusion does *not* hand
me. First, it pins the throughput reference to an execution I control and can reason about tile by tile,
rather than to whatever schedule the compiler happens to emit, so a later rung's delta is read against a
fixed, understood floor. Second — and this is the load-bearing reason to write it now rather than leave
`F.gelu` untouched — it forces the explicit `torch.autograd.Function` with a hand-written backward into
existence, and that is the harness every later activation swap plugs into. inductor will fuse a
pointwise forward for free, but the moment I want an analytic backward for a custom activation, or to run
the activation on the matmul's fp32 accumulator before it ever reaches HBM, I am outside what the
compiler will do for me and I need this machinery already standing. So I write the kernel knowing its
forward is a wash against the compiler, and I write it for the backward and the reference, not for a
speedup I would only be imagining.

A scope decision falls out here, and it is the one place this rung diverges from the most aggressive
thing the tile abstraction allows. I *could* fuse the up-projection matmul and the activation into one
tiled kernel, running `tanh`-GELU on the matmul's fp32 accumulator in registers before it ever touches
HBM, which would delete the activation's round-trip entirely — that ~12.9 ms of per-micro-batch
bandwidth time I counted above. But that means hand-writing the matmul, and the matmul is exactly what
cuBLAS (and, here, `torch.compile`'s fused path) does extremely well: it is the compute-bound half
running near peak, autotuned tile shapes, software-pipelined, tensor-core-saturating. At this first
rung I am not trying to beat cuBLAS at its own game; I am establishing the GELU floor with a *correct,
clean* activation kernel and leaving the two matmuls as the well-tuned `x @ w_fc.t()` and
`@ w_proj.t()` torch ops. So the kernel I write fuses *the activation's own elementwise chain* into one
launch over `h` — collapsing four generic kernels and their temporaries into one — and stops there. The
matmuls stay in torch. That is the literal edit: the function does the first matmul, runs the GELU
kernel over `h` as one custom pass, does the second matmul. Whether folding the activation *into* the
matmul epilogue actually pays — whether the round-trip I leave on the table is worth more or less than
the cuBLAS-versus-hand-rolled matmul gap I would take on — is a real question, but it is one I can only
answer once I have this floor's number to compare against, so I defer it deliberately rather than guess.

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
constants check out at the three asymptotic anchors, but those are the easy points; I want one check
in the interior where all the terms are live and a dropped factor would actually show. Take `x = 1`.
The inner argument is `c·(1 + a) = 0.7978846·1.044715 = 0.83356`; `tanh(0.83356) = 0.68219`; so the
forward is `gelu(1) = 0.5·(1 + 0.68219) = 0.84109`. The analytic derivative: first term
`0.5·1.68219 = 0.84109`; `sech² = 1 − 0.68219² = 0.53462`; `d_inner = c·(1 + 3a) = 0.7978846·1.134145
= 0.90493`; second term `0.5·1·0.53462·0.90493 = 0.24191`; sum `= 1.08300`. Now check it against the
forward by finite difference, which knows nothing about my product-rule algebra: `gelu(1.05) = 0.89552`,
`gelu(0.95) = 0.78729`, so the central slope is `(0.89552 − 0.78729)/0.10 = 1.08266`. Analytic `1.0830`
against numerical `1.0827` — agreement to three significant figures, which is exactly the residual I
would expect from an `O(h²)` central difference and tells me the closed form has no dropped factor.
There is a nice tell in that number too: the derivative at `x = 1` is *greater than 1*. GELU's
derivative is not monotone — it rises above unity, peaking around `1.13` near `x ≈ 1.3` before settling
back to `1` — so a slope of `1.083` at `x = 1` is the expansive shoulder of that bump, not a bug. A
correct GELU gradient *should* exceed 1 there, and mine does; had I dropped the `sech²` term or the
chain factor I would have gotten a slope clamped at or below the first term's `0.84`, and the finite
difference would have caught me. So in the backward I save the pre-activation `h` from the forward
(cheaper than recomputing the first matmul), recompute `inner`, `tanh(inner)`, `sech²`, `d_inner` in
fp32 on `h`, assemble `gelu_grad`, and form `d_h = (d_act · gelu_grad)` cast back to the gradient
dtype. I also save `act` itself so the down-projection weight gradient `grad_out.t() @ act` does not
have to recompute the activation. There is a memory-versus-recompute choice buried in that decision
worth naming: `h` and `act` are each `M·N` bf16, ~537 MB per layer per micro-batch, so stashing both
costs ~1.07 GB of live activation memory held across the step for every FFN — summed over the model's 24
layers, ~25.8 GB of saved hidden resident at the moment a micro-batch's backward begins, a real slice of
an 80 GB card spent on activations alone, which is the concrete pressure that would eventually make
saving only the pre-activation the right call. I could instead save only
`h` and recompute `act` in the backward from the saved pre-activation, trading a wide tensor for one
extra tanh-GELU pass. At this floor I choose to save both, because the activation recompute is not
free — tanh is a transcendental, not a `relu` — and the point of the rung is a clean, unhurried
reference, not a memory-minimized one; the day the activation is cheap enough that recomputing it costs
less than holding it, saving only the pre-activation becomes the right call, but that is a trade I make
when the activation *is* cheap, not now. I wrap the whole thing in a `torch.autograd.Function` so
forward stashes `(x, w_fc, w_proj, h, act)` and backward runs exactly this derivation. One dtype
discipline: I
cast the saved tensors and the weights to `grad_output`'s dtype consistently in the matmuls, so I never
mix a bf16 weight with an fp32 grad — a small thing that otherwise silently slows the path or NaNs it.

So the step-1 fill is settled and it is, deliberately, the conservative one: the scaffold's GELU,
expressed as one fused-elementwise Triton activation kernel between two ordinary torch matmuls, with an
analytic tanh-GELU backward in a custom autograd Function (the full module is in the answer). No change
to *what* the FFN computes — same activation everyone trusts — only a tighter execution of it and an
explicit backward.

Now reason about what this floor must do, because reading its number is the entire point of running it.
Quality first: I have changed nothing about the function the model computes — it is still GELU to the
precision of the tanh approximation — so I expect `val_loss`, `wikitext2_ppl` and `lambada_ppl` to land
at the unremarkable GELU level, whatever that turns out to be, and the downstream zero-shot numbers to
sit at the baseline GELU accuracies. This is by construction the *quality* floor: GELU is
asymptotically linear, so a strongly-firing and a barely-firing unit are pushed through at nearly the
same slope, and the activation adds no super-linear shaping; on LM perplexity GELU is famously only on
par with plain ReLU. So whatever number it posts, I should read it as "this is what the default
activation buys, with no shaping beyond the knee," and expect a later activation change — not a kernel
change — to move it. I deliberately do *not* have a target value in hand; the point of the rung is to
manufacture the reference, not to hit a preconceived one. Throughput second: collapsing the
activation's elementwise chain into one fused launch over `h` should make the activation cheaper than
the multi-kernel default, but the dominant cost is still the two cuBLAS matmuls plus the one HBM
round-trip of `h` that I did *not* fuse away (I left the matmuls in torch), so `elapsed` should be
respectable but not extraordinary — the activation round-trip is still there, just executed in one pass
instead of four. The falsifiable expectations, then, are concrete: I expect this rung to be the *worst
on quality* of anything I try (it is unshaped GELU) and *competitive, likely the best, on throughput*
(it leaves the matmuls to cuBLAS and only tidies the activation). If a later rung lowers `val_loss`
while holding throughput near this, that isolates the activation as the quality lever; if a later rung
lowers `val_loss` but *raises* `elapsed`, it tells me fusing the matmul myself cost more than it saved,
and that the cleanest win keeps the matmuls exactly where they are here. Either way, this GELU number is
the reference against which both axes get read.
