Softmax attention trains beautifully in parallel — form `QKᵀ`, softmax it, multiply by `V`, and because
the whole sequence is known at training time every output row comes out of two big matmuls that saturate
the GPU. But that same `L×L` score object is the problem: compute and memory are `O(L²)`, and at decode I
have to keep every past key and value around, a cache that grows linearly with how far I have generated.
At block size 1024 the score matrix is a million entries per head per layer, and across 16 heads and 24
layers the attention arithmetic and its activation memory dominate the loop — and the cost lands exactly
where language models are headed, toward long context. So I want a mixer with the mirror-image profile of
an RNN — fixed-size state, `O(1)` per step at inference, constant memory, no growing cache — without
giving up quality against a strong softmax Transformer trained on the same data. That second clause is the
whole problem: cheapness is trivial, cheapness that holds quality is the research question. So I start at
the cheapest *credible* mixer, the one whose failure will diagnose what the next rung has to fix.

Lay the credible floors side by side. Plain
linear attention — replace `exp(qₜ·kᵢ)` with a feature-map dot product so `φ(qₜ)` factors out of the
causal sum and the layer becomes `Sₜ = S_{t−1} + kₜᵀvₜ`, `oₜ = qₜ Sₜ` — is the obvious floor, but its
disease is already known in the abstract: the additive write never forgets, the state just accumulates
every key it has ever seen, which is precisely why it loses to softmax on language modeling. Opening there
means my first rung is a known-broken baseline and I learn nothing I do not already suspect. A state-space
model (S4-style) is the other tempting floor — `sₜ = A s_{t−1} + B xₜ`, `oₜ = C sₜ`, unrolled into an
FFT-trainable convolution, strong at long range — but `A, B, C` are *fixed parameters* that do not read
the token; the model cannot condition *what it stores* on *what it is seeing*. Language modeling is
content-addressed retrieval at its core — fetch the value I filed under a key because I have now seen that
key again — and a content-independent recurrence structurally cannot do that. And jumping straight to a
data-dependent forget gate — which the 1-D RNN literature says carries most of a gated cell's capacity —
is exactly where the ladder is heading, but a gate that depends on the previous hidden state serializes
the recurrence and destroys the parallel matmul training form. Opening at that rung leaps into the hardest
engineering wall before I have established that a decay helps at all, and its residual failure would
confound "is the decay expressive enough" with "can I train this thing."

That triangulates the floor exactly: linear attention plus the cheapest forget mechanism that does not
break parallel training — a single fixed scalar `γ ∈ (0,1)`, `Sₜ = γ S_{t−1} + kₜᵀvₜ`. Using *one fixed
scalar* is not laziness: a scalar pulls cleanly out of the cumulative product, keeping the attention-style
parallel/chunkwise machinery alive where a matrix or state-dependent transition would not. So when this
falls short I know the shortfall is about expressivity of the forgetting, not trainability.

Build it from the recurrence, since that is where the `O(1)` inference lives, then prove it has a parallel
face I can train with. Start as general as I can, with a state *matrix*: `sₙ = A s_{n−1} + kₙᵀvₙ`,
`oₙ = qₙ sₙ`. Unroll it: `sₙ = Σ_{m≤n} A^{n−m} kₘᵀvₘ`, so `oₙ = Σ_{m≤n} qₙ A^{n−m} kₘᵀvₘ`. A linear
recurrence unrolled is *already* a causal weighted sum over the whole past — every term `m ≤ n` weighted
by `qₙ A^{n−m} kₘᵀ`, exactly the shape of causal attention. The recurrence gives `O(1)` inference; the
unrolled sum is a candidate parallel training form. The whole game is now the matrix power `A^{n−m}`: I
need that weight content-aware and cheap to compute in parallel.

Content-awareness first, because that is what separates me from the state-space line. Make `Q = XW_Q`,
`K = XW_K` depend on the input, so `qₙ A^{n−m} kₘᵀ` is a genuine content-based score modulated by
`A^{n−m}`. Now the matrix power: a general `A^{n−m}` is the expensive, opaque part, so diagonalize
`A = Λ diag(γ e^{iθ}) Λ⁻¹`, complex eigenvalues in polar form as a per-dimension magnitude `γ` and phase
`θ`. Then `A^{n−m} = Λ diag(γ e^{iθ})^{n−m} Λ⁻¹`, and `Λ, Λ⁻¹` sit on the outside multiplying `qₙ` and
`kₘᵀ` — but those are learnable projections anyway, so absorb `Λ` into `W_Q` and `Λ⁻¹` into `W_K` for
free. What is left is a diagonal power. Split it across positions,
`(γ e^{iθ})^{n−m} = (γ e^{iθ})^n (γ e^{iθ})^{−m}`: `qₙ (γ e^{iθ})^n` is the query scaled by `γⁿ` and
rotated by `e^{inθ}`, `kₘ (γ e^{iθ})^{−m}` the key with the conjugate. The phase part is exactly rotary
position embedding — `q` gets `e^{inθ}`, `k` gets `e^{−imθ}`, their product depends on `n−m` — and with
the magnitude layered on it is the xPos form: a relative position encoding *with a decay*. The position
encoding I would normally bolt on by hand falls out of the recurrence's state matrix; I did not choose to
encode position, the linear recurrence forced a relative-position kernel on me, and all I get to pick is
its magnitude profile.

A per-dimension magnitude is more bookkeeping than I want, and worse, `γ^{−m}` on the key grows unbounded
as `m` shrinks — for `γ = 0.96` and `m` in the hundreds it is astronomically large, a live overflow risk.
Collapse `γ` to a single scalar per head. Then `γ^{n−m}` pulls out of the per-coordinate structure and
only the phase rotation stays inside the query/key factors:
`oₙ = Σ_{m≤n} γ^{n−m} (qₙ e^{inθ})(kₘ e^{imθ})† vₘ`. The scalar `γ^{n−m}` is a clean per-distance decay
multiplying a rotary-encoded content score, fully parallel — every term an independent product, no softmax
coupling positions. I started from a recurrence and landed on a parallel, position-aware, content-based
weighted sum. Call this operator **retention**. Its **parallel** face packs the rotation into the
projections and the decay-and-causality into one matrix `D_{nm} = γ^{n−m}` for `n ≥ m`, `0` otherwise —
the causal mask (zero above the diagonal) and the exponential decay at once — giving
`Retention(X) = (QKᵀ ⊙ D) V`, softmax deleted. Its **recurrent** face is `Sₙ = γ S_{n−1} + KₙᵀVₙ`,
`oₙ = Qₙ Sₙ`, a fixed `d_k×d_v` state, `O(1)` per step. They compute the same function: `Sₙ` unrolls to
`Σ_{m≤n} γ^{n−m} KₘᵀVₘ`, so `Qₙ Sₙ` is exactly row `n` of `(QKᵀ ⊙ D) V` — the causal mask in `D` is the
same statement as "the state only accumulates the past." A third, **chunkwise** face runs the parallel
form inside chunks of length `C` and carries the state recurrently across them, `O(LCd + Ld²)` total —
linear in `L` against softmax's `O(L²d)`; at `L = 1024` with `C = 64` the intra-chunk term is 16× cheaper
than the full score, and this is the mode the FLA Triton kernels implement.

Is a single scalar `γ` expressive enough? One scalar fixes one decay rate, one memory timescale, but
different parts of language want different horizons — some heads long-tailed, some sharply local. With
softmax that diversity comes from heads in different subspaces; here I have a *second* axis, the decay rate
itself. So give `h` heads each its own `γ`, geometrically spanning the range, `γ = 1 − 2^{−5−arange(h)}`,
and concatenate — multi-scale retention. Compute what the schedule buys at this scale, because the numbers
decide whether the floor is even well-covered. The half-life of a memory is roughly
`ln 2 / (1−γ) ≈ 0.693 · 2^{5+i}` tokens: head 0 ≈ 22 tokens, head 5 ≈ 710, head 6 ≈ 1419 (already past
the block size), head 15 ≈ 727,000. So of 16 heads only the first six have half-lives that fit *inside* a
1024-token block; the other ten decay so slowly they effectively never forget within a sequence. The fixed
schedule densely covers "22 to 710 tokens" and then jumps straight to "essentially persistent" — a coarse,
lopsided partition, chosen *a priori*, identical for every channel, context, and word through a given
head. Nothing in it can say "hold *this* named entity for the 340 tokens until it is referenced, then drop
it": the rate is a property of the head, not of the content. That is the seed of what the next rung must
fix, and for now it is what makes this a floor rather than a finale. The multi-scale choice also creates a
magnitude wrinkle I can size: with roughly unit random keys and values the readout norm scales like
`√(1/(2(1−γ)))`, spanning `√16 = 4` for head 0 up to `√1024 ≈ 32` for the persistent heads — an 8× range.
Concatenated raw, the persistent heads dominate the mix by that factor while the fast heads contribute
almost nothing to the gradient, so normalize each head *separately* (group norm, one group per head)
before mixing. And deleting softmax cost me a nonlinearity — softmax was normalizing *and* injecting a
nonlinearity — so restore it with a content-dependent **output gate**, `MSR(X) = (swish(XW_G) ⊙ Y) W_O`,
the missing expressiveness back without the `O(n)` softmax.

Now make it concrete in the two editable regions, since the loop is fixed. I do not hand-roll the chunkwise
kernel — FLA ships `MultiScaleRetention` implementing exactly the three faces above, so my edit imports it
and wires it in, but I derive the configuration against the harness rather than the generic recipe. The
layer takes `hidden_size = n_embd = 1024`, `num_heads = n_head = 16`; the live design choices are the
expansion ratios `expand_k` (`d_k = expand_k·d`) and `expand_v` (`d_v = expand_v·d`), which set the state
`d_k×d_v` and thus the memory capacity. To choose them I count the budget I am matching: a softmax mixer is
`c_attn` `3d²` plus `c_proj` `d²` = `4d²` per layer, the MLP is `8d²`, so a block is `12d²`; the mixer is
`4d²` of that, and that is the number to respect. The canonical width-matched retention widens the value to
`2d` and shrinks the FFN to `2d` to stay balanced — but the scaffold's `Block` and `MLP` are fixed at the
`4·d` GELU FFN and I am not rewriting them, so I cannot do the FFN-shrink. Widening the value to `2d` on
top of an unshrunk `4d` FFN takes the mixer from `4d²` toward `8d²`, as large as the whole MLP — past the
softmax budget. The honest, parameter-conservative choice is the symmetric one, `expand_k = expand_v =
1.0`, so `d_k = d_v = d` and q/k/v/o reproduce softmax's `4d²`, with the swish output gate adding about one
more `d²` — a shade above softmax, and that shade is the restored nonlinearity, not a widened state. So
this rung is the *unwidened* variant, which isolates "does multi-scale decayed retention match softmax at
matched width" without confounding it with a larger state. I keep `use_output_gate = True`,
`gate_fn = 'swish'`, and the per-head norm internal to FLA, and set `self.use_pos_emb = False`: retention's
`γ^{n−m}` decay plus the rotary phase *is* the relative position signal, so the loop must skip its learned
`wpe`, which would otherwise double-encode position and fight the decay. The `Block` stays the scaffold
default — swapping only the mixer keeps the comparison fair, so any quality difference is the mixer, not
the wrapper. (The literal edit is in the answer.)

Now what this floor must do. Retention's decay is fixed
and data-independent — one `γ` per head, chosen a priori — and as the half-life arithmetic made vivid it
cannot look at content and decide "this is a key fact, hold it" versus "this is filler, forget it fast." It
is the cheapest credible forget gate, better than no decay (which is why it is a real contender, not a
strawman), but exactly the data-independent gate the RNN literature warns against. So I expect retention to
train stably and land in the credible range — a uniform predictor over the 50257-token vocabulary sits at
`ln 50257 ≈ 10.82` nats, a competent GPT-2-Medium on web data around `2.5–3.0` — but to be the **weakest**
rung, the least information-adaptive memory I can build. The shortfall should show most on the perplexity
benchmarks and the recall-flavored downstream tasks, where holding the *right* facts, not a fixed-rate
exponential average of all of them, is what matters. If that is the shape, the direction is clear: the
decay itself must stop being fixed — the model must choose its forgetting rate from content — and the hard
part will be doing that without destroying the matmul chunkwise form the fixed scalar `γ` protected.
