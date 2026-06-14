The thing that actually hurts is concrete, and I want to stare at it before I reach for any
architecture. Softmax attention trains beautifully in parallel — form `QKᵀ`, an `L×L` block of scores,
softmax it, multiply by `V`, and because the whole sequence is known at training time every output row
comes out of two big matmuls that saturate the GPU. But that same `L×L` object is the problem: compute
and memory are `O(L²)`, and at decode I have to keep every past key and value around — a cache that
grows linearly with how far I have generated, with per-step cost that grows with context. The
deployment story is bad exactly where language models are headed, toward long context. So I want a
sequence mixer with the mirror-image profile of an RNN — a fixed-size state, `O(1)` per step at
inference, constant memory, no growing cache — but I cannot give up language-model quality against a
strong softmax Transformer trained on the same data. That second clause is the whole problem. Cheapness
is trivial; cheapness that holds quality is the entire research question, and where I have to start the
ladder is the *cheapest credible* mixer, the one whose failure will teach me what the next rung must
fix. I want the first rung to be the simplest thing that is plausibly competitive, so that everything
above it is a diagnosed response.

Why not start from plain linear attention itself? Because I already know its disease in the abstract:
the additive write `Sₜ = S_{t−1} + kₜᵀvₜ` never forgets, the state just accumulates every key it has
ever seen, and that is precisely why it loses to softmax on language modeling. Starting the ladder
there would mean my very first rung is a known-broken baseline, and I would learn nothing I do not
already suspect. I want the first rung to already carry the *one* fix that the 1-D RNN literature says
is non-negotiable — a decay, a forget mechanism — in its simplest possible form, so that the rung is a
real contender and its residual failure points somewhere specific. The minimal credible decay is a
single fixed scalar `γ ∈ (0,1)`: `Sₜ = γ S_{t−1} + kₜᵀvₜ`. This is linear attention with a recency
bias, and the reason to use *one fixed scalar* is not laziness — it is that a scalar pulls cleanly out
of the cumulative product, which keeps the entire attention-style parallel/chunkwise training machinery
intact. So the first rung is "linear attention plus the cheapest forget gate that does not break
training," and I want to derive why that operator is the right floor and exactly what it looks like in
this task's edit surface.

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
`W_K` — the change of basis is free. What is left is a diagonal power. Split the relative exponent
across the two positions, `(γ e^{iθ})^{n−m} = (γ e^{iθ})^n (γ e^{iθ})^{−m}`, and attach each piece to
its own factor: `oₙ = Σ_{m≤n} (qₙ (γ e^{iθ})^n)(kₘ (γ e^{iθ})^{−m})ᵀ vₘ`. Look at what each factor is.
`qₙ (γ e^{iθ})^n` is the query scaled by `γⁿ` and rotated by `e^{inθ}`; `kₘ (γ e^{iθ})^{−m}` the key
with `γ^{−m}` and `e^{−imθ}`. The phase part is exactly rotary position embedding — `q` gets `e^{inθ}`,
`k` gets `e^{−imθ}`, their product depends on `n−m` — and with the magnitude `γ^{±}` layered on it is
precisely the xPos form: a relative position embedding *with a decay*. The position encoding I would
normally bolt on by hand falls out of the recurrence's state matrix. That is a good sign: the
decay-and-rotation is not an arbitrary add-on, it is what `A^{n−m}` *is* once diagonalized.

A per-dimension magnitude `γᵢ` is more bookkeeping than I want, and `γ^{−m}` on the key grows unbounded
as `m` shrinks — numerically ugly. Simplify `γ` from a per-dimension vector to a single scalar per
head. Then `γ^{n−m}` pulls out of the per-coordinate structure entirely and I keep only the phase
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
`D` is the same statement as "the state only accumulates the past." Train with the matmul, infer with
the recurrence, no approximation. And a third, **chunkwise** face runs the parallel form inside chunks
and carries the state recurrently across them in linear time — the within-chunk decay-to-boundary and
boundary-to-query factors reconstitute the true relative decay — which is the long-sequence training
mode and the one the FLA kernels actually implement.

Is a single scalar `γ` expressive enough? One scalar fixes one decay rate, one timescale of memory, but
different parts of language want different horizons — some heads should keep a long tail of context,
some should be sharply local. With softmax I would get that diversity from heads in different subspaces;
here I have a *second* axis to vary, the decay rate itself. So use `h` heads, each with its own `γ`,
geometrically spanning the range — `γ = 1 − 2^{−5−arange(h)}`, from fast forgetting to almost none —
and concatenate. Multi-scale retention. But the multi-scale choice creates a wrinkle: heads with
different `γ` produce outputs of different magnitude — a near-1 `γ` sums many terms and grows large, a
small `γ` sums few. Normalize each head *separately* (group norm, one group per head) before mixing, or
the high-variance heads swamp the rest. And deleting softmax cost me a nonlinearity — softmax was doing
double duty, normalizing *and* injecting a nonlinearity. Restore it with a content-dependent **output
gate**: `MSR(X) = (swish(XW_G) ⊙ Y) W_O`, a multiplicative data-dependent gate on the normalized
retention output, the missing expressiveness back without the `O(n)` softmax.

Now make it concrete in *this task's edit surface*, because the loop is fixed and I only get to fill in
`CausalSelfAttention` and `Block`. I do not hand-roll the chunkwise kernel — FLA ships
`MultiScaleRetention` with the chunk Triton kernel that implements exactly the three faces above, so my
edit imports it and wires it into the scaffold. Here is where I have to be careful and *not* import the
paper-faithful recipe wholesale, because the harness exposes a specific configuration and I should
derive against it, not against the generic version. The FLA layer takes `hidden_size = n_embd = 1024`
and `num_heads = n_head = 16`. The two expansion ratios are the live design choices: `expand_k` sets the
key/query width `d_k = expand_k·d` and `expand_v` the value width `d_v = expand_v·d`, and the recurrent
state is `d_k×d_v`, so these set the memory capacity. The full paper-matched retention widens the value
to `2d` (a bigger state means more memory) and shrinks the FFN to `2d` to keep the parameter count
matched to a Transformer. But the scaffold's `Block` and `MLP` are fixed — the MLP is the standard
`4·d` GELU FFN and I am not rewriting it — so I cannot do the FFN-shrink trick, and widening the value
to `2d` on top of an unshrunk `4d` FFN would *inflate* the layer past the softmax budget rather than
match it. The honest, parameter-conservative choice given a fixed `4d` MLP is the symmetric one:
`expand_k = 1.0`, `expand_v = 1.0`, so `d_k = d_v = d` and the attention block lands at roughly the
softmax `4d²` allocation. So this rung's retention is the *unwidened* variant — same value width as
softmax — which is the right floor: it isolates "does multi-scale decayed retention match softmax at
matched width" without confounding it with a larger state. I keep `use_output_gate = True` and
`gate_fn = 'swish'` (the gate is the nonlinearity I argued for), and the per-head normalization is the
RMSNorm/group-norm FLA applies internally. And I set `self.use_pos_emb = False`: retention's `γ^{n−m}`
decay plus the rotary phase *is* the relative position signal, so the loop must skip its learned `wpe`,
which would otherwise double-encode position and fight the decay.

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
priori, and it cannot look at the content and decide "this token is a key fact, hold it" versus "this is
filler, forget it fast." It is the cheapest credible forget gate — better than no decay, which is the
whole reason it is a real contender and not a strawman — but it is exactly the data-independent gate the
RNN literature warns against. So my falsifiable expectation is this: retention should *train stably* and
land in the credible range — clearly better than gateless linear attention would, a real language model,
not a degenerate one — but it should be the **weakest** rung on the ladder, because its single fixed
decay rate per head is the least information-adaptive memory I can build. I expect its validation loss
to sit *above* (worse than) any mixer that makes the decay data-dependent, and I expect the gap to show
up most sharply on the perplexity benchmarks and on the recall-flavored downstream tasks, where holding
the *right* facts — not a fixed-rate exponential average of all of them — is what matters. WikiText-2
and LAMBADA perplexity should be the loosest, and the downstream accuracies should trail, precisely
because a fixed decay throws away the specific long-range tokens that a content-chosen gate would have
kept. If that is what the numbers show, the diagnosis for rung two is already written: the decay must
become *data-dependent* — the model has to choose its forgetting rate from the content — and the
challenge will be doing that without destroying the matmul chunkwise form that the fixed scalar `γ`
protected.
