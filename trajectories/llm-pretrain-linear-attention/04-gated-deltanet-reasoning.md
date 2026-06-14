DeltaNet's numbers confirmed the recall diagnosis almost exactly, and that very success exposes the
one thing it threw away. The error-correcting write did what the diagonal gate could not: LAMBADA
perplexity fell hard, from GLA's 84.73 and RetNet's 82.36 down to 70.48; WikiText-2 tightened from 64.32
to 49.88; validation loss dropped from 2.4482 to 2.3481; HellaSwag, stuck at 31.1 across both prior
rungs, finally moved to 32.77, and ARC-Easy rose to 53.58. The clean-retrieval story held on every
metric that rewards fetching a specific earlier value. But look at the one number that did *not* improve:
WinoGrande sat at 49.17, fractionally *below* GLA's 49.88 and well below RetNet's 52.01. WinoGrande is
the most coreference-heavy of the suite — it turns on tracking which of two candidate antecedents a
pronoun binds to, and doing that *over a span* where the binding may need to survive intervening
content. That is not "retrieve once and read out"; it is "hold the right binding while a lot of other
material streams past, and let go of the stale ones." And that is precisely the capability DeltaNet does
not have, because in the move from GLA to DeltaNet I deliberately removed the decay. Recall the rung-3
edit: I took `use_gate = False` and, more fundamentally, DeltaNet's recurrence
`Sₜ = S_{t−1}(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ` has **no `α` term at all**. The Householder transition erases
exactly the one direction being overwritten and leaves *everything else perfectly intact, forever*. So
DeltaNet can do surgical, targeted removal of a colliding key, but it has lost the ability GLA had to
*globally* let memory fade — to say "this whole context is getting old, decay it." It is all scalpel and
no eraser. WinoGrande stalling is that missing eraser made visible.

So now I can name the two capabilities the ladder has separated, and see that they were never in
conflict — I just never had both at once. GLA (rung 2) gave me a **data-dependent decay**: a content-chosen
forgetting rate, an eraser, but a *blunt* one — multiplicative and diagonal, it scales the whole state
and cannot localize the removal to the specific association a new key collides with. DeltaNet (rung 3)
gave me a **content-addressed write**: the error-correcting delta rule, a scalpel that removes exactly
the stale direction, but with *no* global fading — once written and not directly overwritten, an
association persists indefinitely. RetNet's failure was no data-dependence; GLA fixed the decay axis but
left the write; DeltaNet fixed the write axis but dropped the decay. The two fixes live on *orthogonal*
axes of the same recurrence — one is the multiplicative pre-factor on `S_{t−1}`, the other is the
additive transition structure — and there is no reason I cannot apply both. That is the move: a
recurrence that *gates and deltas at once*. The literature names the intuition precisely — gating enables
rapid, global memory erasure while the delta rule facilitates targeted, content-addressed updates, and
the two mechanisms are complementary rather than redundant.

Let me derive the combined rule and check that it is well-posed, not just a hopeful concatenation. I want
the previous state to first be *globally decayed* by a data-dependent scalar `αₜ ∈ (0,1)`, then have the
delta-rule's targeted erase-and-write applied. Write the gated delta rule as
`Sₜ = αₜ (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ`. Read the limits to make sure it is a genuine
generalization of both rungs and not some third thing. Set `αₜ = 1` for all `t`: the pre-factor
vanishes and I get back `Sₜ = (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ` — *exactly* DeltaNet, rung 3. Set
`βₜ = 0`: the delta write vanishes and the Householder collapses to the identity, leaving
`Sₜ = αₜ S_{t−1}` — the pure scalar-gated *decay skeleton* that the Mamba2 / scalar-gated-linear-attention
family is built on (that family then adds its own additive write back on top; what the `βₜ = 0` limit
isolates is the gating mechanism GLA also lives by — GLA's gate is per-channel, this one per-head scalar,
but the same idea). So the combined rule sits *above* both rung 2 and rung 3 as special cases — it is strictly
more general, which is exactly what I want from a finale: it cannot be worse than the better of the two
unless the extra freedom is mis-trained, and it can be better wherever a task needs *both* a global
eraser and a local scalpel — which is the WinoGrande profile.

Now the design choices, each of which I have to justify rather than inherit, because a combined rule has
new failure modes. First, what *shape* is `αₜ`? GLA used a per-key-channel diagonal gate; here I make
`αₜ` a single **scalar per head**, the cheapest possible global decay. Why scalar and not diagonal? Two
reasons. The delta rule already supplies the fine-grained, per-direction control — the Householder picks
out the exact subspace to erase — so I do not need the gate to *also* be fine-grained; I need it to do
the one thing the delta rule cannot, which is uniform global fading. A scalar `αₜ` is precisely that and
nothing more, which keeps the two mechanisms doing non-overlapping jobs. And a scalar pulls cleanly out
of the chunkwise cumulative product — exactly the property that made RetNet's fixed `γ` parallelizable —
so the combined rule keeps the same matmul chunkwise training form as DeltaNet, with the decay folded in
as a per-position log-cumsum. (If the gate were diagonal *and* combined with the Householder, the
telescoping that made each of GLA and DeltaNet parallelizable would no longer compose cleanly; the
scalar is what keeps the UT-transform chunk algorithm intact.) Second, how do I parameterize `αₜ` so it
trains well? A fresh sigmoid sits near 0.5 — halving the state every step, killing long memory before
training starts (the same trap I worried about for GLA). So borrow the Mamba2 discretization, which is
built to give a long-memory prior. Compute a positive timescale `Δₜ = softplus(a_proj(xₜ) + dt_bias)`
with `dt_bias` initialized so `Δₜ` starts small, and a per-head positive rate `A = exp(A_log)` with
`A_log` initialized broadly; the log-decay is `gₜ = −A · Δₜ ≤ 0`, so `αₜ = exp(gₜ) ∈ (0,1]`, sitting
near 1 (slow forgetting) at init and only decaying when the content asks for it. This is exactly the
well-conditioned, near-1 initialization that the gate needs, and it makes `αₜ` data-dependent through
`a_proj(xₜ)`. Third, stability: the Householder still needs L2-normalized keys so its non-unit
eigenvalue is `1 − βₜ`, and now the *combined* contractive factor along the key direction is
`αₜ(1 − βₜ) ∈ [0,1]` while orthogonal directions decay by `αₜ ∈ (0,1]` — both inside the unit disk, so
the recurrence is stable by the same argument as DeltaNet plus a strictly-contractive scalar. I keep
DeltaNet's SiLU-then-L2 on q/k, the short convolution, and the learned `βₜ`.

One more decision the combined rule reopens that DeltaNet had closed: the **output gate**. At rung 3 I
set `use_gate = False` deliberately, to isolate the write-rule change and not confound it with the swish
output gate that RetNet and GLA carried. But the finale is no longer isolating one axis — it is the full
combination — and now there is a clean reason to bring the output gate *back*: with a data-dependent
decay re-introduced, the per-head outputs again have content-dependent, head-varying scale (the thing the
output gate and per-head norm were for in GLA and RetNet), and the swish gate is the cheap restoration of
the nonlinearity that deleting softmax removed. So I set `use_gate = True`, which routes the output
through a fused gated RMSNorm with a `g_proj` swish gate — matching the GLA/RetNet output recipe — rather
than DeltaNet's bare RMSNorm. This is a deliberate departure from the rung-3 edit and I am naming it: the
finale re-adds the output gate that DeltaNet dropped, because the finale is the union of both lineages,
not the write-rule axis in isolation.

Make it concrete in this task's edit surface. FLA ships `GatedDeltaNet` with the `chunk_gated_delta_rule`
kernel that implements the combined recurrence above — its forward computes the per-head log-decay
`g = −exp(A_log)·softplus(a_proj(x) + dt_bias)`, the learned `β = sigmoid(b_proj(x))`, applies the
SiLU+short-conv to q/k/v and L2-normalizes q/k inside the kernel, and runs the UT-transform chunk
algorithm with the scalar decay folded in as a chunk-local cumsum. The edit imports it into
`CausalSelfAttention`. The one harness-specific subtlety is head shaping: this layer follows the Mamba2
convention `num_heads · head_dim = 0.75 · hidden_size` when `use_gate = True` (so the layer lands at the
documented ~`6·d²` parameter budget, comparable to the other rungs' attention-plus-budget), *not*
`n_head` directly — so for GPT-2 Medium (`hidden_size = 1024`) I pass `head_dim = 128` and
`num_heads = 6` (`6·128 = 768 = 0.75·1024`), with `expand_v = 2.0` (value head dim `256`, value width
`1536`), which is the configuration that satisfies the layer's integer-dimension assertions. I keep
`use_short_conv = True`, `conv_size = 4`, and `use_gate = True` (the re-added output gate). I set
`self.use_pos_emb = False` for the same reason as every rung since RetNet — the decay and recurrence
carry relative position, so the loop skips its `wpe`. The `Block` stays the scaffold default; only the
mixer is swapped. (The literal scaffold edit is in the answer.)

So the finale is the natural close of the ladder: RetNet established that a decay must be data-dependent;
GLA made the decay data-dependent but left the additive write; DeltaNet fixed the write but dropped the
decay; and the gated delta rule applies *both fixes on their orthogonal axes at once* —
`Sₜ = αₜ(I − βₜ kₜ kₜᵀ)S_{t−1} + βₜ vₜ kₜᵀ` — a data-dependent global eraser composed with a
content-addressed scalpel, reducing to DeltaNet at `αₜ = 1` and to scalar-gated linear attention at
`βₜ = 0`. Here is the bar it has to clear, stated against the real numbers, since this rung gets no
feedback of its own. The strongest baseline is DeltaNet at validation loss **2.3481**, WikiText-2
**49.88**, LAMBADA **70.48**, ARC-Easy **53.58**, HellaSwag **32.77**, and the open wound WinoGrande
**49.17**. For the finale to earn the endpoint it must (1) hold or improve DeltaNet's clean-retrieval
gains — validation loss at or below 2.3481, LAMBADA at or below 70.48, since at `αₜ ≈ 1` it *is*
DeltaNet and the added decay should only help where global fading matters; and (2) *recover the eraser
deficit* — WinoGrande should rise back above DeltaNet's 49.17 and ideally toward RetNet's 52.01, because
WinoGrande was exactly the "hold the right binding while content streams past, let the stale ones fade"
profile that the missing global decay was costing. The thing I would most want to validate is that the
combined rule does not simply collapse to one of its parents: if it trains to `αₜ ≈ 1` everywhere it is
just DeltaNet again (no WinoGrande recovery), and if it trains to `βₜ ≈ 0` everywhere it is just gated
linear attention (LAMBADA regresses back toward 84). The endpoint claim is that with the Mamba2-style
near-1 decay initialization both mechanisms stay alive, the validation loss drops below 2.3481, and
WinoGrande recovers — the one metric three rungs of single-axis fixes never simultaneously held with the
recall gains.
