The thing that actually hurts is concrete, and I want to stare at it before I reach for any
architecture. Softmax attention trains beautifully in parallel — form `QKᵀ`, an `L×L` block of scores,
softmax it, multiply by `V`, and because the whole sequence is known at training time every output row
comes out of two big matmuls that saturate the GPU. But that same `L×L` object is the problem: compute
and memory are `O(L²)`, and at decode I have to keep every past key and value around — a cache that
grows linearly with how far I have generated, with per-step cost that grows with context. At block size
1024 the score matrix is a million entries per head per layer, and with 16 heads and 24 layers the
attention arithmetic and its activation memory are the dominant term in the loop. The deployment story
is bad exactly where language models are headed, toward long context. So I want a sequence mixer with
the mirror-image profile of an RNN — a fixed-size state, `O(1)` per step at inference, constant memory,
no growing cache — but I cannot give up language-model quality against a strong softmax Transformer
trained on the same data. That second clause is the whole problem. Cheapness is trivial; cheapness that
holds quality is the entire research question, and where I have to start the ladder is the *cheapest
credible* mixer, the one whose failure will teach me what the next rung must fix. I want the first rung
to be the simplest thing that is plausibly competitive, so that everything above it is a diagnosed
response.

Before I commit to one, let me lay the credible starting points side by side and kill the ones that
teach me nothing. Plain linear attention itself — replace the softmax kernel `exp(qₜ·kᵢ)` with a
feature-map dot product so `φ(qₜ)` factors out of the causal sum and the layer becomes `Sₜ = S_{t−1} +
kₜᵀvₜ`, `oₜ = qₜ Sₜ` — is the obvious floor, but I already know its disease in the abstract: the
additive write never forgets, the state just accumulates every key it has ever seen, and that is
precisely why it loses to softmax on language modeling. Starting the ladder there would mean my very
first rung is a known-broken baseline, and I would learn nothing I do not already suspect. A
state-space model (S4-style) is the other tempting floor — a linear recurrence `sₜ = A s_{t−1} + B xₜ`,
`oₜ = C sₜ`, unrolled into a convolution trainable by FFT, strong at long range — but look at where
`A, B, C` come from: they are *fixed parameters*, they do not read the token. The mixing is a fixed
convolution kernel; the model cannot condition *what it stores* on *what it is seeing*. Language
modeling is content-addressed retrieval at its core — I need to fetch the value I filed under a key
because I have now seen that key again — and a content-independent recurrence structurally cannot do
that, so an SSM floor would fail for a reason I would then have to fix by bolting content-awareness
back on. I would rather start from something already content-based. And the third candidate, jumping
straight to a data-dependent forget gate — the thing the 1-D RNN literature says is the real source of a
gated cell's capacity — is exactly where I suspect the ladder is *heading*, but a gate that depends on
the previous hidden state serializes the recurrence and destroys the parallel matmul training form. If I
open at that rung I have leapt into the hardest engineering wall before I have even established that a
decay helps at all, and its residual failure would confound "is the decay expressive enough" with "can
I train this thing." I want the floor to be the cheapest decay that *keeps the whole attention-style
training machinery intact*, precisely so that when it falls short I know the shortfall is about
expressivity of the forgetting, not about trainability.

That triangulates the floor exactly: linear attention plus the cheapest forget mechanism that does not
break parallel training. The minimal credible decay is a single fixed scalar `γ ∈ (0,1)`: `Sₜ = γ
S_{t−1} + kₜᵀvₜ`. The reason to use *one fixed scalar* is not laziness — it is that a scalar pulls
cleanly out of the cumulative product, which keeps the attention-style parallel/chunkwise machinery
alive where a matrix or state-dependent transition would not. So the first rung is settled in principle,
and now I want to derive why that operator is the right floor and exactly what it looks like in this
task's edit surface.

Let me build it from the recurrence, because the recurrence is where the `O(1)` inference lives, and
then prove it has a parallel face I can train with. Start as general as I can: a linear recurrence with
a state *matrix*, not a scalar, project the input to a value `vₙ`, carry a state `sₙ`, read out with a
query. `sₙ = A s_{n−1} + kₙᵀvₙ`, `oₙ = qₙ sₙ`. Each step accumulates the outer-product term `kₙᵀvₙ`
transformed by `A`. Unroll it: `s₁ = k₁ᵀv₁`, `s₂ = A k₁ᵀv₁ + k₂ᵀv₂`, and in general
`sₙ = Σ_{m≤n} A^{n−m} kₘᵀvₘ`. Read it out: `oₙ = qₙ sₙ = Σ_{m≤n} qₙ A^{n−m} kₘᵀvₘ`. That is the bridge.
A linear recurrence, unrolled, is *already* a causal weighted sum over the whole past — every term
`m ≤ n` contributes, weighted by `qₙ A^{n−m} kₘᵀ` — which is exactly the shape of causal attention.
The recurrence gives me `O(1)` inference for free, and the unrolled sum is a candidate parallel form to
train with. The entire game is now in the matrix power `A^{n−m}`: I need that weight to be content-aware
and cheap to compute in parallel.

Content-awareness first, because that is what separates me from the state-space line where the mixing
never sees the tokens. Make the query and key projections depend on the input — `Q = XW_Q`, `K = XW_K`
— so `qₙ A^{n−m} kₘᵀ` is a genuine content-based score modulated by `A^{n−m}`. Now the matrix power. A
general `A^{n−m}` is the expensive, opaque part, so diagonalize: `A = Λ diag(γ e^{iθ}) Λ⁻¹`, allowing
complex eigenvalues written in polar form as a per-dimension magnitude `γ` and phase `θ`. Then
`A^{n−m} = Λ diag(γ e^{iθ})^{n−m} Λ⁻¹`, and the `Λ, Λ⁻¹` sit on the outside, left-multiplying `qₙ` and
right-multiplying `kₘᵀ`. They are learnable matrices anyway, so absorb `Λ` into `W_Q` and `Λ⁻¹` into
`W_K` — the change of basis is free, it costs nothing because those projections were going to be learned
regardless. What is left is a diagonal power. Split the relative exponent across the two positions,
`(γ e^{iθ})^{n−m} = (γ e^{iθ})^n (γ e^{iθ})^{−m}`, and attach each piece to its own factor:
`oₙ = Σ_{m≤n} (qₙ (γ e^{iθ})^n)(kₘ (γ e^{iθ})^{−m})ᵀ vₘ`. Look at what each factor is.
`qₙ (γ e^{iθ})^n` is the query scaled by `γⁿ` and rotated by `e^{inθ}`; `kₘ (γ e^{iθ})^{−m}` the key
with `γ^{−m}` and `e^{−imθ}`. The phase part is exactly rotary position embedding — `q` gets `e^{inθ}`,
`k` gets `e^{−imθ}`, their product depends on `n−m` — and with the magnitude `γ^{±}` layered on it is
precisely the xPos form: a relative position embedding *with a decay*. The position encoding I would
normally bolt on by hand falls out of the recurrence's state matrix. That is a good sign, and worth
pausing on: the decay-and-rotation is not an arbitrary add-on, it is what `A^{n−m}` *is* once
diagonalized. I did not choose to encode position; the linear recurrence forced a relative-position
kernel on me, and all I get to pick is its magnitude profile.

A per-dimension magnitude `γᵢ` is more bookkeeping than I want, and worse, `γ^{−m}` on the key grows
unbounded as `m` shrinks — for `γ = 0.96` and `m` in the hundreds, `γ^{−m}` is astronomically large,
numerically ugly and a live overflow risk. Simplify `γ` from a per-dimension vector to a single scalar
per head. Then `γ^{n−m}` pulls out of the per-coordinate structure entirely and I keep only the phase
rotation inside the query/key factors: `oₙ = Σ_{m≤n} γ^{n−m} (qₙ e^{inθ})(kₘ e^{imθ})† vₘ`, with `†`
the conjugate transpose. The scalar `γ^{n−m}` is now a clean per-distance decay multiplying a
rotary-encoded content score, fully parallelizable: every term is an independent product, no softmax
coupling positions. I started from a recurrence and landed on a parallel, position-aware, content-based
weighted sum. Call this operator **retention**, since it is literally a state that retains a decaying
summary of the past, and pin down its faces. The **parallel** face packs the rotation into the
projections and the decay-and-causality into one matrix `D_{nm} = γ^{n−m}` for `n ≥ m`, `0` otherwise —
which does the causal mask (zero above the diagonal) and the exponential decay (`γ^{n−m}` on/below) at
once — giving `Retention(X) = (QKᵀ ⊙ D) V`, the GPU-friendly shape with softmax deleted. The
**recurrent** face is `Sₙ = γ S_{n−1} + KₙᵀVₙ`, `Retention(Xₙ) = Qₙ Sₙ`, fixed-size `d_k×d_v` state,
`O(1)` per step. They compute the same function: `Sₙ` unrolls to `Σ_{m≤n} γ^{n−m} KₘᵀVₘ`, so
`Qₙ Sₙ = Σ_{m≤n} γ^{n−m}(Qₙ Kₘᵀ) Vₘ`, which is exactly row `n` of `(QKᵀ ⊙ D) V` — the causal mask in
`D` is the same statement as "the state only accumulates the past."

Let me not take that equivalence on faith; it is the load-bearing claim, so I trace two steps by hand.
Take `S₀ = 0`. Step 1: `S₁ = γ·0 + k₁ᵀv₁ = k₁ᵀv₁`, so `o₁ = q₁ S₁ = q₁ k₁ᵀ v₁ = (q₁·k₁) v₁`. The
parallel form: row 1 has only `D₁₁ = γ⁰ = 1`, so `o₁ = (q₁·k₁) v₁`. Identical. Step 2:
`S₂ = γ S₁ + k₂ᵀv₂ = γ k₁ᵀv₁ + k₂ᵀv₂`, so `o₂ = q₂ S₂ = γ (q₂·k₁) v₁ + (q₂·k₂) v₂`. The parallel form:
row 2 has `D₂₁ = γ¹ = γ` and `D₂₂ = γ⁰ = 1`, so `o₂ = γ(q₂·k₁) v₁ + (q₂·k₂) v₂`. Identical again, and I
can see *why*: the exponent `n−m` in `D` and the number of times `γ` multiplies through the recurrence
before term `m` is read at step `n` are the same count. So I train with the matmul, infer with the
recurrence, no approximation between them. And a third, **chunkwise** face runs the parallel form inside
chunks of length `C` and carries the state recurrently across them: within-chunk it is the masked
`(QKᵀ⊙D)V` in `O((L/C)·C²·d) = O(LCd)` work, across chunks it decays and updates the `d_k×d_v` state in
`O((L/C)·C·d²/... ) = O(Ld²)` work, so `O(LCd + Ld²)` total — linear in `L`, against softmax's `O(L²d)`.
At `L = 1024` with a chunk of 64 the intra-chunk term is 16× cheaper than the full quadratic score, and
this is the mode the FLA Triton kernels actually implement.

Is a single scalar `γ` expressive enough? One scalar fixes one decay rate, one timescale of memory, but
different parts of language want different horizons — some heads should keep a long tail of context,
some should be sharply local. With softmax I would get that diversity from heads in different subspaces;
here I have a *second* axis to vary, the decay rate itself. So use `h` heads, each with its own `γ`,
geometrically spanning the range — `γ = 1 − 2^{−5−arange(h)}`, from fast forgetting to almost none —
and concatenate. Multi-scale retention. Let me actually compute what that schedule buys at this task's
scale, because the numbers decide whether the floor is even well-covered. The per-step decay deficit is
`1−γ_i = 2^{−5−i}`, and the half-life of a memory is roughly `ln 2 / (1−γ) ≈ 0.693 · 2^{5+i}` tokens.
Head 0: `0.693 · 32 ≈ 22` tokens. Head 5: `0.693 · 1024 ≈ 710` tokens. Head 6: `≈ 1419` tokens, already
past the block size of 1024. Head 15: `0.693 · 2^{20} ≈ 727,000` tokens. So of my 16 heads, only the
first six have half-lives that fit *inside* a 1024-token block; the other ten decay so slowly they
effectively *never forget within a sequence*. The fixed schedule densely covers "22 to 710 tokens" with
six heads and then jumps straight to "essentially persistent." That is a coarse and lopsided partition
of memory horizons, and — this is the important part — it is chosen *a priori*, identical for every
channel, context, and word that passes through a given head. Nothing in it can say "hold *this* named
entity for the 340 tokens until it is referenced, then drop it": the rate is a property of the head, not
of the content. That observation is exactly the seed of what the next rung will have to fix, but for now
it is what makes this a floor rather than a finale. The multi-scale choice also creates a wrinkle, and I
can size it. Heads with different `γ` produce outputs of different magnitude: with roughly unit, roughly
random keys and values the readout norm scales like `√(Σ_m γ^{2(n−m)}) ≈ √(1/(2(1−γ)))`. For head 0
(`1−γ = 1/32`) that is `√16 = 4`; for the slowest head whose horizon still fits the block (head 5,
`1−γ = 1/1024`) it is `√512 ≈ 23`; the fully persistent heads saturate near `√1024 ≈ 32`. So the per-head
output scales span roughly an 8× range, and if I concatenate them raw the persistent heads dominate the
mix by that factor while the fast heads contribute almost nothing to the gradient. Normalize each head
*separately* (group norm, one group per head) before mixing, or the high-variance heads swamp the rest. And deleting softmax cost me a nonlinearity — softmax
was doing double duty, normalizing *and* injecting a nonlinearity. Restore it with a content-dependent
**output gate**: `MSR(X) = (swish(XW_G) ⊙ Y) W_O`, a multiplicative data-dependent gate on the
normalized retention output, the missing expressiveness back without the `O(n)` softmax.

Now make it concrete in *this task's edit surface*, because the loop is fixed and I only get to fill in
`CausalSelfAttention` and `Block`. I do not hand-roll the chunkwise kernel — FLA ships
`MultiScaleRetention` with the chunk Triton kernel that implements exactly the three faces above, so my
edit imports it and wires it into the scaffold. Here is where I have to be careful and *not* import the
canonical recipe wholesale, because the harness exposes a specific configuration and I should
derive against it, not against the generic version. The FLA layer takes `hidden_size = n_embd = 1024`
and `num_heads = n_head = 16`. The two expansion ratios are the live design choices: `expand_k` sets the
key/query width `d_k = expand_k·d` and `expand_v` the value width `d_v = expand_v·d`, and the recurrent
state is `d_k×d_v`, so these set the memory capacity. To choose them I have to know the budget I am
matching, so let me count it. A softmax mixer is `c_attn` (q,k,v) `= 3d²` plus `c_proj = d²`, i.e. `4d²`
per layer; the MLP is `4d²` up plus `4d²` down `= 8d²`; so a block is `12d²`, and with `d = 1024` that
is `12 · 1.05M ≈ 12.6M` params, times 24 layers `≈ 302M`, plus a tied `50257 × 1024 ≈ 51M` embedding —
`≈ 353M`, which is the ~355M the context quotes. The mixer is `4d²` of that per-block `12d²`, and that is
the number I have to respect. The full width-matched retention design widens the value to `2d` and shrinks the
FFN to `2d` to keep the count balanced — but the scaffold's `Block` and `MLP` are fixed at the standard
`4·d` GELU FFN and I am not rewriting them, so I cannot do the FFN-shrink trick. And widening the value
to `2d` on top of an unshrunk `4d` FFN is not free: it multiplies the value, gate, and output
projections, taking the mixer from `4d²` toward `8d²` — as large as the entire MLP, doubling the layer
past the softmax budget. The honest, parameter-conservative choice given a fixed `4d` MLP is the
symmetric one: `expand_k = 1.0`, `expand_v = 1.0`, so `d_k = d_v = d` and the q/k/v/o projections
reproduce softmax's `4d²`, with the swish output gate adding about one more `d²` — a shade above
softmax, and that shade is the price of the restored nonlinearity, not a widened state. So this rung's
retention is the *unwidened* variant — same value width as softmax — which is the right floor: it
isolates "does multi-scale decayed retention match softmax at matched width" without confounding it with
a larger state. I keep `use_output_gate = True` and `gate_fn = 'swish'` (the gate is the nonlinearity I
argued for), and the per-head normalization is the RMSNorm/group-norm FLA applies internally. And I set
`self.use_pos_emb = False`: retention's `γ^{n−m}` decay plus the rotary phase *is* the relative position
signal, so the loop must skip its learned `wpe`, which would otherwise double-encode position and fight
the decay.

For the `Block`, I leave the standard pre-norm structure exactly as the scaffold has it —
`x = x + attn(ln_1(x))`, `x = x + mlp(ln_2(x))` — because the only thing I am swapping is the mixer, and
keeping the block identical to the softmax baseline is what makes the comparison fair: any quality
difference is the *mixer*, not the wrapper. (The literal scaffold edit is in the answer.) The forward is
the one-liner `o, _, _ = self.attn(x); return o` — FLA's layer returns `(output, attn_weights,
past_kv)` and I take the output.

So at rung one the mixer is settled: multi-scale retention at matched width, a fixed per-head scalar
decay, rotary phase, per-head norm, swish output gate, no absolute position embeddings, dropped into the
two editable regions and nothing else. Now reason about what this floor must do, because that is the
point of running it. Retention's decay is *fixed and data-independent*: every head has one `γ`, chosen a
priori, and — as the half-life arithmetic made vivid — it cannot look at the content and decide "this
token is a key fact, hold it" versus "this is filler, forget it fast." It is the cheapest credible
forget gate — better than no decay, which is the whole reason it is a real contender and not a strawman
— but it is exactly the data-independent gate the RNN literature warns against. So my falsifiable
expectation is this: retention should *train stably* and land in the credible range — clearly better
than gateless linear attention would, a real language model, not a degenerate one — but it should be the
**weakest** rung on the ladder, because its single fixed decay rate per head is the least
information-adaptive memory I can build. Let me pin what "credible range" even means so the prediction is
falsifiable and not a mood: a model emitting a uniform distribution over the 50257-token vocabulary has
cross-entropy `ln 50257 ≈ 10.82` nats, and a competent GPT-2-Medium on this kind of web data lands
somewhere around `2.5–3.0` nats (validation perplexity `exp(2.5) ≈ 12` to `exp(3.0) ≈ 20`). So credible
means a validation loss clearly under 3 and downstream accuracies clearly above chance — 25% for the
4-way tasks (ARC-Easy, HellaSwag), 50% for the binary ones (PIQA, WinoGrande) — and degenerate would be
a loss stuck near the uniform 10.8 or accuracies pinned at chance. That is the band I am betting
retention lands inside while still trailing a data-dependent mixer. I expect its validation loss to sit
*above* (worse than) any mixer that makes the decay data-dependent, and I expect the gap to show up most sharply on the
perplexity benchmarks and on the recall-flavored downstream tasks, where holding the *right* facts — not
a fixed-rate exponential average of all of them — is what matters. WikiText-2 and LAMBADA perplexity
should be the loosest, and the downstream accuracies should trail, precisely because a fixed decay
throws away the specific long-range tokens that a content-chosen gate would have kept, and because the
six-heads-inside-a-block / ten-heads-persistent partition has no way to place a horizon exactly where a
given fact lives. If that is what the numbers show, then the direction the shortfall points me in is
clear: the decay itself would have to stop being fixed — the model would have to choose its forgetting
rate from the content — and the hard part would be doing that without destroying the matmul chunkwise
form that the fixed scalar `γ` protected.
