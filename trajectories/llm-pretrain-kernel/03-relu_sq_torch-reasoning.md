The fused rung resolved the bet I made, and it split clean: the activation half won and the kernel half
lost. Quality dropped exactly as the multiplicative-interaction story predicted — `val_loss` 2.2749
under the GELU floor's 2.2868, `wikitext2_ppl` 43.52 under 43.94, and `lambada_ppl` 66.88 well under
68.2, the largest relative move. So squared ReLU is real: changing the activation from asymptotically-
linear GELU to the self-gating `relu(z)²` bought a genuine quality gain in this exact two-matrix slot,
confirming that the lever is the activation, not the kernel. But the throughput half went the wrong way,
and badly: `elapsed` 30344 against the GELU rung's 20035 — roughly 50% *slower*. That is the falsifiable
expectation I wrote down coming true on the pessimistic side: the round-trip the fusion deleted was
worth less than the gap I took on by replacing cuBLAS's up-projection with a hand-rolled tiled matmul at
fixed `BLOCK_M=BLOCK_N=64`, `BLOCK_K=32`. Under `torch.compile`, the GELU rung's matmuls were already
heavily optimized; my naive tiled kernel, however clean its epilogue, cannot match a vendor GEMM on a
355M-param model's FFN shapes, and the activation round-trip I saved is a rounding error next to that
matmul gap. The diagnosis writes itself: keep the activation that won, throw away the kernel that lost.
This rung is that move — squared ReLU in *pure torch*, so the two matmuls go back to cuBLAS, and only
the activation and its hand-written backward stay custom.

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
whole reason it fits where GEGLU/SwiGLU cannot.

Now the choices that pin it down, each derivable rather than assumed. Why power 2 and not 3 or 4?
Rectified polynomials `F(z) = zⁿ` for `z ≥ 0` are sharper for higher `n` (Krotov–Hopfield's
associative-memory capacity scales like `Nⁿ⁻¹`), but sharper cuts both ways for an activation I have to
train in bf16: the general rectified power has derivative `n·relu(z)ⁿ⁻¹`, which for `n = 3` is `3z²` —
quadratic gradient growth, much easier to overflow — and the forward `zⁿ` overflows sooner. `n = 2` is
the *minimal* super-linear rectified polynomial: the smallest step away from linear that still gives the
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
existing grad-clip (1.0) and AdamW handle scale.

The forward is settled: `h = x @ w_fc.t()` gives the pre-activation, `a = relu(h)²`, `out = a @
w_proj.t()`. The single change from the fused rung is *execution*: no Triton, no hand-rolled matmul.
Both matmuls are plain torch `@`, so they go to cuBLAS under `torch.compile` — the exact path the GELU
rung used to post 20035 — and only the activation and its backward are custom. This is the direct
answer to the 30344 regression: I am betting almost all the fused rung's quality gain came from the
activation and almost none from the fusion, so removing the fusion should recover throughput to the
GELU level while keeping the quality. The activation itself, `relu` then square, is cheaper per element
than GELU's `erf`/`tanh`, so even as a separate elementwise pass it is no worse than the GELU floor's
activation and arguably a touch lighter.

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
`relu_h` directly is the cleaner mirror of the derivation and avoids recomputing `relu(h)` twice. Both
are correct; this rung saves `(x, w_fc, w_proj, h, relu_h)` and reads the analytic factors straight out.
One precision note: under bf16 autocast the pre-activations and outputs are low precision, but I cast
the saved tensors and weights to `grad_output`'s dtype in the backward matmuls and the `2·relu(h)`
multiply, so I never mix a bf16 weight with an fp32 grad. I wrap it in a `torch.autograd.Function` with
the fixed signature `fused_mlp_forward(x, w_fc, w_proj)` — forward stashes the five tensors, backward
runs the derivation above (the full module is in the answer). No `CONFIG_OVERRIDES`: the activation is
parameter-matched and tested on the loop's existing AdamW / cosine schedule, exactly as the GELU floor
was, so any difference is the activation, not a retune.

Now the falsifiable expectations against both prior rungs, because this rung is precisely a *Pareto*
claim. On quality I expect to *match or slightly beat* the fused rung's 2.2749 `val_loss`: the
activation is mathematically identical to the fused rung's, so the only sources of difference are
floating-point — the fused kernel computed the up-projection in an fp32 accumulator and down-cast in its
epilogue, while pure torch under autocast follows cuBLAS's bf16 path — and I expect that to be a wash or
a hair in my favor, landing at or just under 2.2749 on `val_loss`, with `wikitext2_ppl` near 43 and
`lambada_ppl` near 67. If quality came in *worse* than the fused rung, it would mean the fp32 epilogue
accumulation was actually buying accuracy and I would have to reconsider giving it up — but the
derivation says the activation, not the accumulation, is what moved perplexity. On throughput the bet is
the sharp one: I expect `elapsed` to collapse from the fused rung's 30344 back toward the GELU floor's
20035 — both matmuls are now the same cuBLAS path the GELU rung used, and the only overhead beyond GELU
is that squared ReLU runs as a separate (but cheaper-than-GELU) elementwise pass. If `elapsed` lands
near or below 20035 while `val_loss` holds at ~2.2749, this rung Pareto-dominates the fused one — better
or equal quality *and* far better throughput — which is exactly the outcome the 30344-vs-20035 split
predicted: the win was the activation all along, and the kernel was a detour.
