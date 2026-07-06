The fused rung resolved the bet I made, and it split clean: the activation half won and the kernel half
lost. Quality dropped exactly as the multiplicative-interaction story predicted — `val_loss` 2.2749
under the GELU floor's 2.2868, `wikitext2_ppl` 43.52 under 43.94, and `lambada_ppl` 66.88 well under
68.2. Let me read those moves as sizes rather than just directions, because the magnitudes are the
mechanism check. The `val_loss` drop is 0.0119 in cross-entropy, which in perplexity is
`exp(2.2868)/exp(2.2749) = 9.843/9.727`, about 1.2% lower held-out perplexity; `wikitext2` fell 0.42
points, ~0.96%; `lambada` fell 1.32 points, ~1.94% — the largest relative move, and it being largest is
consistent with the story, since `lambada` rewards confident completion of a long-range dependency and
squared ReLU's gift is exactly sharpening confident, strongly-firing units. The downstream reads
corroborate softly: `arc_easy` 54.67→54.8, `hellaswag` 33.61→33.77, `piqa` 63.44→63.49, all a hair up;
`winogrande` 51.7→48.46 went *down* 3.24, but that sits right at the chance band for a 355M model on a
single seed, so I read it as eval noise rather than a signal, exactly as I said I would when I wrote the
floor's reference down. So the quality verdict is unambiguous in the direction and modest in size: squared
ReLU is real. Changing the activation from asymptotically-linear GELU to the self-gating `relu(z)²`
bought a genuine quality gain in this exact two-matrix slot, confirming that the lever is the activation,
not the kernel.

But the throughput half went the wrong way, and badly: `elapsed` 30344 against the GELU rung's 20035 —
that is +10,309 seconds, a factor of 1.51, roughly 50% *slower*. That is the falsifiable expectation I
wrote down coming true on the pessimistic side of the asymmetric bet: the bounded round-trip the fusion
deleted was worth far less than the unbounded gap I took on by replacing cuBLAS's up-projection with a
hand-rolled tiled matmul at fixed `BLOCK_M=BLOCK_N=64`, `BLOCK_K=32`. Let me quantify the trade, because
the isolation I built into the fused rung makes the number clean: the *only* execution change from the
floor was that single forward up-projection kernel — every gradient matmul and the forward down-matmul
stayed plain torch/cuBLAS in both rungs — so the entire +10,309 s is attributable to that one kernel and
nothing else. Spread it out: 10,309 s over 13,535 iterations is ~0.76 s added per iteration; each
iteration runs 8 gradient-accumulation micro-batches over 24 layers, so `8·24 = 192` forward
up-projection calls per iteration, which puts ~4.0 ms of added time on *each* forward up-projection. Now
set that against what the fusion was supposed to buy: the activation round-trip it deleted is ~1.07 GB of
HBM traffic per FFN per micro-batch, ~0.5 ms on a ~2 TB/s bus. So per up-projection I spent ~4 ms of
extra matmul time to buy back ~0.5 ms of memory traffic — an 8:1 losing trade, and that ratio *is* the
"naive tiled GEMM can't touch cuBLAS on these FFN shapes" gap made quantitative. Under `torch.compile`
the GELU rung's matmuls were already heavily optimized; my fixed-tile kernel, however clean its epilogue,
carries none of the autotuned tiling, software pipelining, or shape-specialized scheduling a vendor GEMM
brings. cuBLAS picks tile shapes matched to the exact `(M, N, K)` and the tensor-core MMA geometry,
overlaps global-memory loads with compute through multi-stage async pipelines, and blocks the operands
through register and shared-memory tiers tuned per architecture; my kernel does a single fixed
`64×64/32` tiling with one accumulator and no prefetch, so it leaves the memory pipe stalling between K
steps. That is a structural gap, not a constant factor I could shave, and the saved round-trip is a
rounding error next to it. The diagnosis writes itself, and it is
the exact next move I named in advance: keep the activation that won, throw away the kernel that lost.
This rung is that move — squared ReLU in *pure torch*, so the two matmuls go back to cuBLAS, and only
the activation and its hand-written backward stay custom.

"Drop the kernel" has a few flavors, though, and I want to pick the right one rather than reflexively.
One tempting middle option is to *keep* the fused up-matmul but autotune its tile shapes — sweep
`BLOCK_M/BLOCK_N/BLOCK_K`, add pipelining — to close the gap to cuBLAS. But the 8:1 measurement kills
this: the fusion's entire upside is bounded at ~0.5 ms of round-trip per up-projection, so even a
*perfect* hand-rolled GEMM that exactly matched cuBLAS would win only that half-millisecond, and getting
there is an open-ended tuning effort against a vendor kernel tuned for years. Chasing a bounded, tiny
prize at unbounded cost is the wrong trade; the rational move is to abandon the fusion, not perfect it.
A second middle option is the analog of what the GELU floor did: keep the matmuls in cuBLAS but write a
custom Triton *elementwise* kernel for `relu(h)²` between them, fusing the activation's own chain. The
GELU floor did exactly that and it was worth it — tanh-GELU is a multi-op transcendental (cubic, tanh,
products) worth collapsing into one launch. But squared ReLU is two flops, a `max` and a multiply; a
plain torch `relu(h) * relu(h)` under autocast is something `torch.compile` can already fuse into the
epilogue of the up-matmul or the prologue of the down-matmul, so a hand-written elementwise kernel would
buy essentially nothing over the compiler's own fusion while adding launch and maintenance. So the
forward activation goes in plain torch and I let the compiler place it.

Let me be precise about what "let the compiler place it" actually yields, because it decides the
throughput prediction. Inductor keeps a large bf16 GEMM as an extern cuBLAS call — it does *not* fold a
pointwise op into cuBLAS's fixed epilogue, which is exactly the fusion my last rung had to hand-write and
exactly what cost it. What inductor *will* do is emit one fused pointwise kernel for `relu(h)²` sitting
between the two extern matmuls: read `h` once, do the `max` and the multiply, write `post` once. So the
compiled execution of this rung is cuBLAS up-matmul → one fused activation pass over `h` → cuBLAS
down-matmul — structurally identical to the GELU floor's execution graph, two cuBLAS matmuls with a
single activation pass between them, the only difference being the pointwise op is `relu²` rather than
tanh-GELU. That identity is what lets me predict throughput sharply instead of hoping: I am running the
GELU floor's exact execution with a cheaper kernel in the middle, so `elapsed` should land at the floor's
20035, plausibly a hair under it since a `max` and a multiply is lighter than a `tanh`. The one
activation round-trip the floor still paid is still here — I am not deleting it; I tried deleting it last
rung by folding it into a hand-rolled matmul and it cost ~50% — I am simply refusing to pay for that
deletion with a matmul cuBLAS does better. The one thing I *do* keep custom
is the backward, in a `torch.autograd.Function` — and here I should be honest that, unlike the Triton
rungs where autograd literally could not trace the kernel, autograd *would* work here, since
`relu(h)**2` and two matmuls are all differentiable torch ops. I keep the explicit backward anyway for
the same save-tensor and dtype control the earlier rungs used and because it costs nothing to write, not
because autograd would be wrong. That is the precise shape of "pure torch": custom `autograd.Function`,
plain-torch matmuls and activation inside it.

Let me re-derive the activation cleanly, because I am now committing to it as the design, not as half of
a kernel experiment. The bill is training compute; the FFN dominates it; the activation between its two
matmuls is the one parameter-free knob — and the fused rung's 2.2749 proves pushing it pays. Why squared
ReLU specifically, restated so the reasoning stands on its own. The defaults ReLU, GELU, Swish are all
asymptotically linear — out in the tail a strongly-firing and a barely-firing unit get pushed through at
nearly the same slope ≈1, so they differ only near the origin, the soft knee. That is why GELU barely
beat ReLU historically and why the GELU floor here sat at the unshaped level: reshaping the knee is
deck-chairs, the bulk of the function stays linear. The real lever is the *asymptotics*. The gated FFNs
(ReGLU/GEGLU/SwiGLU, building on Dauphin's GLU) change the asymptotics by going multiplicative —
`(act(xW) ⊙ xV) W2`, a product of two linear projections — and that multiplicative interaction is what
the literature credits for their perplexity wins over GELU. But they need a third weight matrix `V` and
a `2/3` inner-width shrink to stay parameter-matched, and the edit surface hands me only `w_fc` and
`w_proj` — no `V`. So the gated FFN is structurally out of reach in this slot.

The escape is the same one the fused rung found, and it is worth re-tracing because it is the crux of
why this works with two matrices. Take ReGLU's hidden unit, `max(uᵀx, 0)·(vᵀx)`, and tie the two weight
vectors, `u = v`: it becomes `max(uᵀx, 0)·(uᵀx)`. Let `z = uᵀx` be the pre-activation the up-projection
already produces. If `z > 0`, `max(z,0) = z` and the unit is `z·z = z²`; if `z ≤ 0`, `max(z,0) = 0` and
the product is `0`. So tying the weights turns ReGLU's unit into `relu(z)²` — and the second
multiplicative factor was never something I needed a new matrix for, it was sitting right there in the
up-projection's own output `xW`. Squared ReLU captures the GLU-family multiplicative benefit with one
matrix instead of two, no width shrink, a literal drop-in into the existing two-matrix FFN. That is the
whole reason it fits where GEGLU/SwiGLU cannot. And the "multiplicative" is not a slogan: expanding
`z² = (Σⱼ wⱼ xⱼ)² = Σⱼ,ₖ wⱼ wₖ xⱼ xₖ` shows a single unit computes an explicit sum of pairwise input
products — a bilinear form — from the one projection, which no univariate reshaping of GELU could ever
produce.

Now the choices that pin it down, each derivable rather than assumed. Why power 2 and not 3 or 4?
Rectified polynomials `F(z) = zⁿ` for `z ≥ 0` are sharper for higher `n` (Krotov–Hopfield's
associative-memory capacity scales like `Nⁿ⁻¹`), but sharper cuts both ways for an activation I have to
train in bf16: the general rectified power has derivative `n·relu(z)ⁿ⁻¹`, which for `n = 3` is `3z²` —
quadratic gradient growth, much easier to destabilize — and the forward `zⁿ` widens the dynamic range
of the hidden viciously, throwing away bf16 precision when the wide tensor is down-cast. `n = 2` is the
*minimal* super-linear rectified polynomial: the smallest step away from linear that still gives the
multiplicative second-order term and a magnitude-aware gradient, with the gradient growing only linearly
(`2z`). The safe rung. Why rectify, rather than plain `x²`? `x²` is even, `x² = (−x)²`, so it fires
identically for `+z` and `−z` — it throws away the sign of the pre-activation entirely, makes every
neuron blind to which side of zero it is on, and is non-monotonic, dropping then rising through 0; two
inputs the up-projection deliberately separated by sign would collapse to the same output. `relu(z)²` is
monotone non-decreasing — flat-zero on the negative half, rising on the positive — so it keeps ReLU's
"off" behavior for negatives and only curves the active half upward. The rectification is what preserves
the gate while letting the active half go super-linear.

There is also a real inductive-bias story in the gradient, and it is worth naming because it is *why*
2.2749 happened. The derivative of `relu(z)²` is `2·relu(z)`: 0 for dead units, and `2·relu(z)` for
live ones — growing linearly with the pre-activation, unbounded above. Contrast GELU and Swish, whose
derivatives saturate near 1 as `z` grows, so a strongly-firing and a mildly-firing unit send back nearly
the same gradient. With squared ReLU a unit that fires hard gets a proportionally *larger* gradient: the
activation tells the optimizer "the confident, strongly-active units matter more, push them harder," and
it does this for free because the magnitude information is already in `z`. That is a genuine inductive
bias, not a cosmetic change — and it is the mechanism behind the perplexity drop the fused rung measured
and that this rung inherits, since the activation is *identical*. The flip side is the danger I keep in
mind: a quadratic with an unbounded-above gradient could in principle run away in low precision, which is
exactly why power 2 and not higher — the gradient grows only linearly. I do not add a clamp inside the
activation, since that would change the function I just derived; I keep it clean and let the loop's
existing grad-clip (1.0) and AdamW handle scale. Let me spot-check the `2·relu(h)` gradient against the
forward by finite difference so I trust the no-case-split form. At `h = 1.5`: forward `relu(1.5)² = 2.25`,
analytic derivative `2·1.5 = 3.0`; numerically `(1.51² − 1.49²)/0.02 = (2.2801 − 2.2201)/0.02 = 3.0` —
exact, as it must be for a quadratic under a central difference. At `h = −0.3`: forward `0`, analytic
`2·relu(−0.3) = 0`, and the forward is flat there, so the numerical slope is `0` too. And the ramp is
continuous through the origin: approaching from the right the slope `2h → 0`, from the left it is
identically `0`, they meet — the `C¹` join I claimed. No dropped indicator, no jump; the single
expression `2·relu(h)` is the whole derivative.

The forward is settled: `h = x @ w_fc.t()` gives the pre-activation, `a = relu(h)²`, `out = a @
w_proj.t()`. The single change from the fused rung is *execution*: no Triton, no hand-rolled matmul.
Both matmuls are plain torch `@`, so they go to cuBLAS under `torch.compile` — the exact path the GELU
rung used to post 20035 — and only the activation and its backward are custom. This is the direct
answer to the 30344 regression: I am betting almost all the fused rung's quality gain came from the
activation and almost none from the fusion, so removing the fusion should recover throughput to the
GELU level while keeping the quality. The activation itself, `relu` then square, is cheaper per element
than GELU's `erf`/`tanh` — a max and a multiply against a transcendental — so even as a separate
elementwise pass it is no worse than the GELU floor's activation and arguably a touch lighter.

The backward I want hand-written rather than left to autograd, because the gradient is local and I can
save exactly what I need. Forward with the matmuls: `h = x W_fcᵀ`, `r = relu(h)`, `a = r·r = r²`,
`y = a W_projᵀ`; upstream `g = ∂L/∂y`. Walk it back. Through the second matmul `y = a W_projᵀ`:
`∂L/∂a = g @ W_proj` — shapes `g (N, n_embd) @ W_proj (n_embd, 4·n_embd) → (N, 4·n_embd)`, matching `a`.
Call it `d_a`. Through the square-of-relu `a = relu(h)²`: `∂L/∂h = d_a · d(relu(h)²)/dh`, and
`d(relu(h)²)/dh = 2·relu(h)·1[h>0]`, but the indicator is redundant against `relu(h)` — where `h ≤ 0`,
`relu(h) = 0` already — so it is exactly `2·relu(h)`. Thus `∂L/∂h = 2·r·d_a`; call it `d_h`. That is the
one place I need `r = relu(h)`; everything else is matmuls. Weight gradients: `∂L/∂W_proj = gᵀ @ a`
(shapes `gᵀ (n_embd, N) @ a (N, 4·n_embd) → (n_embd, 4·n_embd)`, matching `W_proj`), and
`∂L/∂W_fc = d_hᵀ @ x` (`(4·n_embd, N) @ (N, n_embd) → (4·n_embd, n_embd)`, matching `W_fc`). Input
gradient `∂L/∂x = d_h @ W_fc` (`(N, 4·n_embd) @ (4·n_embd, n_embd) → (N, n_embd)`, matching `x`). So the
backward is two cheap elementwise ops — the `2·r·d_a` step — plus three matmuls, and the only
nonlinear-specific value needed is `r = relu(h)`.

What to stash. I save the pre-activation `h` and `relu_h = relu(h)`: `relu_h` supplies the `2·relu(h)`
factor and, squared, the activation value `a` for the `W_proj` gradient. This is a small but deliberate
difference from the fused rung, which saved only `pre` and recomputed `relu(pre)` and `relu(pre)²` in
backward to hold one fewer wide tensor across the step — there the kernel was already paying for a
custom path, so minimizing saved tensors mattered; here, in pure torch with cuBLAS matmuls, stashing
`relu_h` directly is the cleaner mirror of the derivation and avoids recomputing `relu(h)` twice. I should name the price of that choice rather than wave it
through: `h` and `relu_h` are each `M·N` bf16, ~537 MB per layer per micro-batch, so saving both holds
~1.07 GB per FFN — ~25.8 GB across the 24 layers resident when a micro-batch's backward begins, twice
the ~12.9 GB the fused rung held by stashing only `pre`. On the 80 GB card that fits with room, and what
the extra tensor buys — never recomputing `relu(h)` — is exactly the trade the fused kernel had to make
and this one does not, since here the three backward matmuls dwarf a second `relu`. So I am choosing the
extra wide tensor knowingly, for a cleaner backward, not by oversight. Both
are correct; this rung saves `(x, w_fc, w_proj, h, relu_h)` and reads the analytic factors straight out.
One precision note: under bf16 autocast the pre-activations and outputs are low precision, but I cast
the saved tensors and weights to `grad_output`'s dtype in the backward matmuls and the `2·relu(h)`
multiply, so I never mix a bf16 weight with an fp32 grad. I wrap it in a `torch.autograd.Function` with
the fixed signature `fused_mlp_forward(x, w_fc, w_proj)` — forward stashes the five tensors, backward
runs the derivation above (the full module is in the answer). No `CONFIG_OVERRIDES`: the activation is
parameter-matched and tested on the loop's existing AdamW / cosine schedule, exactly as the GELU floor
was, so any difference is the activation, not a retune.

Before I read the expectations I want to pin down the one place quality could legitimately differ from
the fused rung's 2.2749, because "the activation is identical" is not quite the whole story at the bit
level. The fused kernel accumulated the up-projection in an fp32 register accumulator and computed
`relu(acc)²` in fp32 before down-casting to bf16. In pure torch, `h = x @ w_fc.t()` — but cuBLAS's bf16
GEMM *also* accumulates internally in fp32 and down-casts its result, so the matmul precision is
identical between the two paths; the difference is not in the matmul at all. The one genuine difference
is where the square happens: the fused kernel squared `acc` while it was still fp32, whereas pure torch
gets `h` already rounded to bf16 out of cuBLAS's epilogue and computes `relu(h)²` in bf16. So the pure-
torch activation squares a bf16-rounded pre-activation. For `O(1)` pre-activations that is a tiny effect,
and I can bound it: bf16 has 8 mantissa bits, so one ulp is ~`2⁻⁸ ≈ 0.4%` relative, and squaring roughly
doubles that to ~0.8% relative error per element. But those rounding errors are zero-mean and
uncorrelated across the wide hidden, and the down-projection sums `4·n_embd = 4096` of them per output,
so the error on each output element averages down by ~`1/√4096 = 1/64`, to ~0.01% — far below anything
that moves a cross-entropy at the third decimal. So I expect it to wash out over training. But I hold it as the honest
caveat: if quality comes in *worse* than 2.2749, it means the fused kernel's fp32 square was actually
buying accuracy and I gave something up; the derivation says the activation, not that last fp32 square,
is what moved perplexity, and the `val_loss` table will tell me which.

It helps to lay the two axes out as coordinates, because "Pareto" is a claim about a picture. Plot
`(val_loss, elapsed)`, both lower-better. The GELU floor sits at `(2.2868, 20035)`: fast, but the worst
quality. The fused rung sits at `(2.2749, 30344)`: the best quality so far, but the worst throughput.
Neither dominates the other — GELU trades quality for speed, the fused rung trades speed for quality, so
the two of them define a little frontier with a gap between them. What I am claiming for this rung is a
point at roughly `(2.2749, ~20035)`: the fused rung's quality at the GELU rung's speed. If it lands
there, it dominates the fused rung outright — equal quality, ~10,000 s less wall-clock — and it also
weakly dominates the GELU floor, since it holds the same throughput while improving quality by the
activation's ~1.2% perplexity. In other words, if the diagnosis is right, this single point collapses
the frontier: both prior rungs become dominated, and the reason is exactly the decomposition the fused
rung's split revealed — quality lives in the activation, throughput lives in leaving the matmuls to
cuBLAS, and this rung is the first to take both.

Now the falsifiable expectations against both prior rungs, because this rung is precisely a *Pareto*
claim. On quality I expect to *match or slightly beat* the fused rung's 2.2749 `val_loss`: the
activation is mathematically identical, the matmuls accumulate the same way, and the only residual is
the bf16-versus-fp32 square I just bounded as negligible — so I expect to land at or within a hair of
2.2749 on `val_loss`, `wikitext2_ppl` near 43, `lambada_ppl` near 67, and the downstream reads to hold
around the fused rung's `arc_easy` mid-54s and `hellaswag` low-33s. On throughput the bet is the sharp
one: I expect `elapsed` to collapse from the fused rung's 30344 back toward the GELU floor's 20035 —
both matmuls are now the same cuBLAS path the GELU rung used, and the only overhead beyond GELU is that
squared ReLU runs as a separate (but cheaper-than-GELU) elementwise pass, so I would not be surprised to
land a touch above 20035 for that one extra pass, but nowhere near 30344. If `elapsed` lands near or
just above 20035 while `val_loss` holds at ~2.2749, this rung Pareto-dominates the fused one — better or
equal quality *and* far better throughput — which is exactly the outcome the 30344-vs-20035 split
predicted: the win was the activation all along, and the kernel was a detour.
