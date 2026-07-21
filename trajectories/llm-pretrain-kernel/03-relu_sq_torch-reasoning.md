The fused rung resolved the bet I made, and it split clean: the activation half won and the kernel half
lost. Quality dropped exactly as the multiplicative-interaction story predicted — `val_loss` 2.2749
under the floor's 2.2868, `wikitext2_ppl` 43.52 under 43.94, `lambada_ppl` 66.88 under 68.2. Read as
sizes rather than directions: the `val_loss` drop is 0.0119 in cross-entropy, ~1.2% lower held-out
perplexity; `wikitext2` fell ~0.96%; `lambada` fell ~1.94%, the largest relative move — consistent with
the story, since `lambada` rewards confident completion of a long-range dependency and squared ReLU's
gift is exactly sharpening confident, strongly-firing units. Downstream corroborates softly: `arc_easy`
54.67→54.8, `hellaswag` 33.61→33.77, `piqa` 63.44→63.49 all a hair up; `winogrande` 51.7→48.46 fell
3.24, but that sits in the chance band for a 355M model on one seed, eval noise as I said I would read
it. So the quality verdict is unambiguous in direction and modest in size: changing the activation from
asymptotically-linear GELU to the self-gating `relu(z)²` bought a genuine gain, confirming the lever is
the activation, not the kernel.

But the throughput half went the wrong way, and badly: `elapsed` 30344 against 20035 — +10,309 s, a
factor of 1.51, ~50% slower. That is the pessimistic side of the asymmetric bet coming true. The
isolation I built in makes the number clean: the *only* execution change from the floor was that single
forward up-projection kernel — every gradient matmul and the forward down-matmul stayed plain
torch/cuBLAS in both rungs — so the entire +10,309 s is attributable to it. Spread out, 10,309 s over
13,535 iterations is ~0.76 s per iteration; each iteration runs 8 grad-accumulation micro-batches over
24 layers, `8·24 = 192` forward up-projection calls, so ~4.0 ms of added time on *each*. Set that
against what the fusion bought — the deleted round-trip is ~1.07 GB, ~0.5 ms per FFN per micro-batch.
So per up-projection I spent ~4 ms of extra matmul time to buy back ~0.5 ms of memory traffic: an 8:1
losing trade, and that ratio *is* the "naive tiled GEMM can't touch cuBLAS on these FFN shapes" gap made
quantitative. My fixed `64×64/32` tiling with one accumulator and no prefetch leaves the memory pipe
stalling between K steps; cuBLAS picks tile shapes matched to the exact `(M, N, K)`, overlaps loads
with compute through multi-stage async pipelines, and blocks the operands through register and
shared-memory tiers tuned per architecture. That is a structural gap, not a constant factor I could
shave, and the saved round-trip is a rounding error next to it. The diagnosis is the exact next move I
named in advance: keep the activation that won, throw away the kernel that lost. This rung is that move
— squared ReLU in *pure torch*, both matmuls back to cuBLAS, only the activation and its hand-written
backward custom.

"Drop the kernel" has a few flavors, and I want the right one rather than reflexively. One middle
option is to *keep* the fused up-matmul but autotune its tile shapes to close the gap to cuBLAS. The
8:1 measurement kills it: the fusion's entire upside is bounded at ~0.5 ms of round-trip per
up-projection, so even a *perfect* hand-rolled GEMM that exactly matched cuBLAS would win only that
half-millisecond, and getting there is an open-ended tuning effort against a vendor kernel tuned for
years. Chasing a bounded, tiny prize at unbounded cost is the wrong trade — abandon the fusion, don't
perfect it. A second option is the analog of the GELU floor: keep the matmuls in cuBLAS but write a
custom Triton *elementwise* kernel for `relu(h)²` between them. There it was worth it — tanh-GELU is a
multi-op transcendental (cubic, tanh, products) worth collapsing into one launch. But squared ReLU is
two flops, a `max` and a multiply; `torch.compile` can already fuse a plain torch `relu(h) * relu(h)`
into a pointwise kernel between the matmuls, so a hand-written elementwise kernel would buy essentially
nothing over the compiler's own fusion while adding launch and maintenance. So the forward activation
goes in plain torch and I let the compiler place it.

Let me be precise about what "let the compiler place it" yields, because it decides the throughput
prediction. Inductor keeps a large bf16 GEMM as an extern cuBLAS call — it does *not* fold a pointwise
op into cuBLAS's fixed epilogue, which is exactly the fusion my last rung had to hand-write and exactly
what cost it. What inductor *will* do is emit one fused pointwise kernel for `relu(h)²` sitting between
the two extern matmuls: read `h` once, `max` and multiply, write `post` once. So the compiled execution
is cuBLAS up-matmul → one fused activation pass over `h` → cuBLAS down-matmul — structurally identical
to the GELU floor's execution graph, the only difference being the pointwise op is `relu²` rather than
tanh-GELU, and lighter (a `max` and a multiply against a transcendental). That identity lets me predict
throughput sharply instead of hoping: I am running the floor's exact execution with a cheaper kernel in
the middle, so `elapsed` should land at ~20035, plausibly a hair under. The one activation round-trip
the floor still paid is still here — I tried deleting it last rung by folding it into a hand-rolled
matmul and it cost ~50% — I am simply refusing to pay for that deletion with a matmul cuBLAS does
better.

The activation is unchanged from the fused rung: squared ReLU, the tied-weight ReGLU self-gate I
derived there — `max(0, xW)·(xW) = relu(xW)²` captures the GLU multiplicative interaction with the two
matrices the edit surface gives, no third `V`, power 2 the minimal super-linear rectified polynomial
whose gradient `2·relu(z)` grows only linearly and is magnitude-aware where GELU/Swish saturate near 1.
I am now committing to it as the design rather than as half of a kernel experiment, so the one thing I
*do* keep custom is the backward, in a `torch.autograd.Function`. Here I should be honest that, unlike
the Triton rungs where autograd literally could not trace the kernel, autograd *would* work now, since
`relu(h)**2` and two matmuls are all differentiable torch ops. I keep the explicit backward anyway for
the same save-tensor and dtype control the earlier rungs used and because it costs nothing to write, not
because autograd would be wrong. That is the precise shape of "pure torch": custom `autograd.Function`,
plain-torch matmuls and activation inside it.

The backward is local, so I save exactly what I need. `h = x W_fcᵀ`, `r = relu(h)`, `a = r²`,
`y = a W_projᵀ`; upstream `g = ∂L/∂y`. Back through the second matmul, `d_a = g @ W_proj` (shapes
`(N, n_embd) @ (n_embd, 4·n_embd) → (N, 4·n_embd)`, matching `a`); through `a = relu(h)²`,
`d_h = 2·relu(h)·d_a` (the `1[h>0]` indicator is redundant against `relu(h)`, which is already zero
where `h ≤ 0`); weight grads `∂L/∂W_proj = gᵀ @ a` and `∂L/∂W_fc = d_hᵀ @ x`, input grad
`∂L/∂x = d_h @ W_fc`. The only nonlinear-specific value is `r = relu(h)`, so I save `h` and
`relu_h = relu(h)`: `relu_h` supplies the `2·relu(h)` factor and, squared, the activation value `a` for
the `W_proj` gradient. This is a deliberate difference from the fused rung, which saved only `pre` and
recomputed `relu(pre)` in backward to hold one fewer wide tensor — there the kernel was already paying
for a custom path, so minimizing saved tensors mattered; here, in pure torch, stashing `relu_h`
directly is the cleaner mirror of the derivation and avoids recomputing `relu(h)` twice. The price is
honest: `h` and `relu_h` are each `M·N` bf16, ~537 MB per layer per micro-batch, so saving both holds
~1.07 GB per FFN — twice what the fused rung held by stashing only `pre` — but on the 80 GB card that
fits with room, and the three backward matmuls dwarf a second `relu`, so I take the extra tensor
knowingly for a cleaner backward. Under bf16 autocast I cast the saved tensors and weights to
`grad_output`'s dtype in the backward matmuls and the `2·relu(h)` multiply, so I never mix a bf16
weight with an fp32 grad. Forward stashes `(x, w_fc, w_proj, h, relu_h)` and backward reads the
analytic factors straight out (the full module is in the answer). No `CONFIG_OVERRIDES`: the activation
is parameter-matched and runs on the loop's existing AdamW / cosine schedule, so any difference is the
activation, not a retune.

There is one place quality could legitimately differ from the fused rung's 2.2749, because "the
activation is identical" is not quite the whole story at the bit level. The fused kernel squared `acc`
while it was still in the fp32 accumulator, whereas pure torch gets `h` already rounded to bf16 out of
cuBLAS's epilogue and computes `relu(h)²` in bf16 — the matmul precision itself is identical (cuBLAS
also accumulates in fp32 and down-casts), the only difference is where the square happens. Bound the
effect: bf16's ~8 mantissa bits give ~0.4% relative error per element, squaring roughly doubles it to
~0.8%, but those errors are zero-mean and uncorrelated across the wide hidden, and the down-projection
sums `4·n_embd = 4096` of them per output, so each output element's error averages down by
~`1/√4096 = 1/64` to ~0.01% — far below anything that moves a cross-entropy at the third decimal. So I
expect it to wash out over training. But I hold it as the honest caveat: if quality comes in *worse*
than 2.2749, it means the fused kernel's fp32 square was buying accuracy I just gave up.

It helps to lay the two axes out as coordinates, because "Pareto" is a claim about a picture. Plot
`(val_loss, elapsed)`, both lower-better. The GELU floor sits at `(2.2868, 20035)`: fast, worst
quality. The fused rung sits at `(2.2749, 30344)`: best quality so far, worst throughput. Neither
dominates the other. What I claim for this rung is a point at roughly `(2.2749, ~20035)` — the fused
rung's quality at the floor's speed. If it lands there it dominates the fused rung outright (equal
quality, ~10,000 s less wall-clock) and weakly dominates the floor (same throughput, better quality by
the activation's ~1.2% perplexity). A single point collapsing the frontier, for exactly the
decomposition the fused rung's split revealed: quality lives in the activation, throughput lives in
leaving the matmuls to cuBLAS, and this is the first rung to take both.

So the falsifiable expectations, because this rung is precisely a *Pareto* claim. On quality I expect
to *match or slightly beat* 2.2749: the activation is mathematically identical and the only residual is
the bf16-versus-fp32 square I just bounded as negligible, so `val_loss` at or within a hair of 2.2749,
`wikitext2_ppl` near 43, `lambada_ppl` near 67, and the downstream reads around the fused rung's
`arc_easy` mid-54s and `hellaswag` low-33s. On throughput the bet is the sharp one: `elapsed` should
collapse from 30344 back toward 20035 — both matmuls are now the same cuBLAS path the floor used, and
the only overhead beyond GELU is that squared ReLU runs as a separate (cheaper) elementwise pass, so I
would not be surprised to land a touch above 20035 for that one pass, but nowhere near 30344. If
`elapsed` lands near or just above 20035 while `val_loss` holds at ~2.2749, this rung Pareto-dominates
the fused one — better-or-equal quality *and* far better throughput — which is exactly the outcome the
30344-vs-20035 split predicted: the win was the activation all along, and the kernel was a detour.
