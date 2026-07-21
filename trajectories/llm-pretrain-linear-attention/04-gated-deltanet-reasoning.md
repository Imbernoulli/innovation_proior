DeltaNet's numbers confirmed the recall diagnosis almost exactly, and that very success exposes the one
thing it threw away — so before reaching for a finale I read the whole three-rung table as arithmetic,
because the *shape* of the improvements is what tells me where the last leverage is. Validation loss went
2.4795 → 2.4482 → 2.3481: steps of −0.0313 (RetNet→GLA, the decay fix) and −0.1001 (GLA→DeltaNet, the
write-rule fix), the write fix about 3.2× the drop the decay fix bought; in perplexity, `11.93 → 11.57 →
10.47`, so the decay fix shaved 3.0% and the write fix 9.5%. WikiText-2 says it louder: 66.67 → 64.32 →
49.88, steps of −2.35 then −14.44. So whatever I do next, I have to respect that the error-correcting
write, not the gate, is where the real quality lived — I must not regress it. Now the metrics I told myself
to watch, with signs. LAMBADA: 82.36 → 84.73 (a 2.9% regression under GLA) → 70.48 (DeltaNet), an
11.88-point fall from RetNet, the recall metric collapsing exactly when I fixed the write. HellaSwag, stuck
at ~31.1 across two rungs, finally broke to 32.77; ARC-Easy climbed monotonically 51.47 → 53.11 → 53.58;
PIQA barely moved. Everything that rewards fetching a specific earlier value moved right, and on LAMBADA it
moved hard. But one number did not: WinoGrande, 52.01 → 49.88 → 49.17 — falling at *every* rung, the only
metric now below its rung-1 value, and *below 50 twice*, below chance for a binary task. A two-rung
monotone decline ending below the coin-flip floor is a structural signal, pointing at something the recall
story alone does not explain.

Here is the hinge of the finale. LAMBADA and WinoGrande are *both* recall-flavored — I lumped them together
at rung 2 as "the tasks that reward tracking a specific earlier token" — yet under DeltaNet they
*dissociated*: LAMBADA recovered spectacularly and WinoGrande kept sinking, so they cannot be testing the
same capability. LAMBADA asks for a passage's final word, recoverable by *retrieving a specific named
entity once* — a single clean fetch, exactly what the delta rule's error-corrected write makes clean.
WinoGrande is different in kind: it turns on which of two candidate antecedents a pronoun binds to, and the
binding must *survive a span* of intervening content while the *other* candidate, and everything else that
streamed past, fades out of the way. That is not "retrieve once and read out"; it is "hold the right
binding while a lot of other material streams past, and let go of the stale ones." The first half DeltaNet
does superbly (why LAMBADA fell); the second half — let the stale ones fade globally — it cannot do at all
(why WinoGrande did not follow). The two "recall" tasks split along exactly the seam between a
content-addressed *write* and a global *fade*, and DeltaNet has the former and none of the latter.

I can read the "none of the latter" straight off the spectrum. DeltaNet is
`Sₜ = S_{t−1}(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ` — **no `α` term**, no multiplicative pre-factor on `S_{t−1}`. With
L2-normalized keys the transition `Mₜ = I − βₜ kₜ kₜᵀ` has eigenvalue `1 − βₜ` along `kₜ` and eigenvalue
*exactly 1* on the entire `d_k − 1`-dim subspace orthogonal to `kₜ`. Eigenvalue one means those directions
*never decay*: whatever was written into them persists undamped forever, until some future key happens to
point along them and overwrites it. The Householder is a scalpel that removes exactly the one overwritten
direction and leaves everything else intact — precisely wrong for WinoGrande, where the stale candidate
binding must *fade on its own* though no later key explicitly overwrites it. Put the three rungs on that
axis and the monotone decline reads itself: RetNet had a clean uniform global fade (a fixed `γ^{n−m}`
shrinking *every* direction every step) and the *highest* WinoGrande, 52.01; GLA replaced the scalar fade
with a diagonal data-dependent gate, a blunter eraser scaling channels rather than fading the whole state,
and WinoGrande dropped to 49.88; DeltaNet removed the fade entirely, pinning `d_k − 1` eigenvalues at
exactly 1, and it dropped to 49.17. The trace 52.01 → 49.88 → 49.17 is the progressive loss of a clean
global fade, metric-ized — DeltaNet all scalpel and no eraser.

So the ladder has separated two capabilities that were never actually in conflict; I just never had both at
once. GLA's **data-dependent decay** and DeltaNet's **content-addressed write** live on *orthogonal*
structural slots of the same recurrence — one the multiplicative pre-factor in front of `S_{t−1}`, the other
the additive-plus-Householder transition — and nothing about them competes for the same slot. So the finale
is a recurrence that gates *and* deltas at once, the gate doing rapid global erasure and the delta rule
doing targeted content-addressed updates, complementary because each does exactly the job the other
structurally cannot.

Write the combined rule and read its limits to confirm it degenerates to each parent exactly, not to some
third thing: `Sₜ = αₜ (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ`, `oₜ = Sₜ qₜ`, with `αₜ ∈ (0,1]` a
data-dependent scalar decay. Set `αₜ = 1`: the pre-factor is the identity and I get back
`Sₜ = (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ`, *exactly* DeltaNet. Set `βₜ = 0`: the delta write vanishes and
the Householder collapses to the identity, leaving `Sₜ = αₜ S_{t−1}` — the pure scalar-gated decay skeleton
the whole scalar-gated-linear-attention family (and Mamba2) is built on before it bolts its own additive
write back, the gating mechanism GLA also lives by (GLA's per-channel diagonal, this one per-head scalar).
So the combined rule sits strictly *above* both rung 2 and rung 3 as special cases — a real guarantee: a
strict generalization cannot be worse than the better of its two parents unless the extra freedom is
mis-trained, and it can be strictly better wherever a task needs both a global eraser and a local scalpel,
exactly the WinoGrande profile.

Now the design decisions, and a combined rule opens failure modes neither parent had. The load-bearing one:
what *shape* is `αₜ`? The tempting default is to reuse GLA's per-key-channel diagonal `Diag(αₜ)` on the
theory that more expressive is better and I have the machinery. Two checks kill it. First, composition into
the chunk kernel: DeltaNet is trainable only because the product of its transitions has a WY/UT-transform
representation `I − Σ wₜ kₜᵀ` — *identity minus a sum of rank-one terms sharing the factors `kₜᵀ`* — which
lets forward substitution turn the recurrence into matmuls. Multiply out a diagonal-gated factor:
`Diag(αₜ)(I − βₜ kₜ kₜᵀ) = Diag(αₜ) − βₜ (αₜ ⊙ kₜ) kₜᵀ`, a *diagonal minus rank-one* — neither identity
minus rank-one nor diagonal — and a product of such factors does not collapse into the WY form. The UT
transform breaks. Second, a commutation check: a *scalar* `αₜ` commutes with everything, so "decay then
delta" and "delta then decay" are the same transition, no ordering to choose; a diagonal `Diag(αₜ)` does
*not* commute with the Householder (`Diag(αₜ) kₜ kₜᵀ` scales rows, `kₜ kₜᵀ Diag(αₜ)` scales columns), so it
would force an arbitrary ordering on top of breaking the kernel. And a third reason it is not even wanted:
the delta rule *already* supplies fine-grained per-direction control (the Householder picks the exact
subspace to erase), so a diagonal gate is redundant with that; what the delta rule cannot do is uniform
*global* fading, and a scalar `αₜ` is precisely that one missing thing and nothing more. So the scalar is
not the lazy choice, it is the *unique* choice: the only gate shape that does the one job the delta rule
lacks, is redundant with nothing it already does, and pulls cleanly out of the chunkwise cumulative product
(RetNet's parallelizability property), keeping DeltaNet's exact matmul UT-transform training form with the
decay folded in as a per-position scalar log-cumsum. `αₜ` is a **per-head scalar**.

One more structural check, because the placement of `αₜ` is not free. I wrote `αₜ` multiplying *only the
carried state*, not the fresh write. Had I written `αₜ [(I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ]`, the current
token's own write would be attenuated by `αₜ` the instant it is written — forgetting a fact in the same
step I record it, nonsense for a memory. Fade the *past*, then add the *fresh* write undecayed, so `αₜ`
sits outside the Householder-times-state product and outside the new write; because it is scalar it
commutes with the Householder, so decaying-then-erasing equals erasing-then-decaying, unambiguous. `αₜ` and
`βₜ` are both functions of `xₜ` alone, so the recurrence stays linear in the state and parallelizable.

How to parameterize `αₜ` so it trains well, because a naive gate collapses long memory before training. A
fresh sigmoid near 0.5 halves the state every step — a half-life of one token, long-range capacity dead at
init, the exact trap GLA defused with its temperature. Borrow the Mamba2 discretization: a positive
per-token timescale `Δₜ = softplus(a_proj(xₜ) + dt_bias)` with `dt_bias` initialized so `Δₜ` starts small,
a per-head positive rate `A = exp(A_log)`, log-decay `gₜ = −A · Δₜ ≤ 0`, `αₜ = exp(gₜ) ∈ (0,1]`. Init
numbers: with `Δₜ ≈ 0.01` and `A ≈ 1`, `αₜ ≈ e^{−0.01} ≈ 0.99`, half-life `≈ 69` tokens — deliberately
*slower* than GLA's aggressive 16 and inside RetNet's in-block range, starting near "barely forgetting" and
letting content pull `αₜ` down only when it asks. And the number that matters most: at init `αₜ ≈ 0.99`, so
the finale starts life as *almost exactly DeltaNet* (`αₜ = 1` is DeltaNet), the right place to start given
DeltaNet is the strongest rung — it begins from the best parent and *learns* to fade.

Stability I derive, because a combined transition can leave the unit disk in ways neither parent did.
`Mₜ = αₜ (I − βₜ kₜ kₜᵀ)` with `‖kₜ‖₂ = 1`: on the `d_k − 1` directions orthogonal to `kₜ` it acts as
`αₜ · I` (eigenvalue `αₜ`); along `kₜ`, eigenvalue `αₜ(1 − βₜ)`. So the spectrum is `{αₜ` (mult. `d_k − 1`)`,
αₜ(1 − βₜ)}`, and with `αₜ ∈ (0,1]`, `βₜ ∈ [0,1]`, both satisfy `0 ≤ αₜ(1 − βₜ) ≤ αₜ ≤ 1` — inside the
closed unit disk; the gate can only shrink the already-stable spectrum. And this spectrum *is* the eraser
made literal: DeltaNet pinned `d_k − 1` eigenvalues at exactly 1, and the gate pulls those from 1 down to
`αₜ < 1` — that single change, the orthogonal subspace's eigenvalue coming off 1, is the global fade
WinoGrande has asked for across three rungs. So I keep DeltaNet's SiLU-then-L2 on `q`/`k` (which forces
`‖kₜ‖ = 1` and makes the Householder an exact projection at `βₜ = 1`), the depthwise short conv, and the
learned `βₜ = σ(W_β xₜ)`.

The combined rule reopens one decision DeltaNet deliberately closed: the **output gate**. At rung 3 I set
`use_gate = False` to isolate the write-rule change from the swish output gate RetNet and GLA carried. But
the finale is no longer isolating a single axis, so that reason has expired, and there is a positive
argument to bring the gate back: re-introducing a data-dependent decay means the per-head outputs again
carry content-dependent, head-varying scale — the same variance-across-heads problem the per-head norm and
swish gate tamed in GLA and RetNet — and the swish gate is the cheap restoration of the nonlinearity
deleting softmax cost. Both reasons are live again the moment `αₜ` returns. So `use_gate = True`, routing
the output through a fused gated RMSNorm with a `g_proj` swish gate rather than DeltaNet's bare per-head
RMSNorm — a deliberate reversal of the rung-3 edit, since the finale is the *union* of both lineages.

Now the edit surface. FLA ships `GatedDeltaNet` with the `chunk_gated_delta_rule` kernel implementing the
combined recurrence: it computes `g = −exp(A_log)·softplus(a_proj(x) + dt_bias)`, `β = sigmoid(b_proj(x))`,
applies SiLU and the short conv to `q`/`k`/`v`, L2-normalizes `q`/`k` in-kernel, and runs the UT-transform
chunk algorithm with the scalar decay folded in as the chunk-local cumsum. So my job is configuration. The
subtlety: with `use_gate = True` this layer does *not* honor `n_head` directly but follows the Mamba2
convention `num_heads · head_dim = 0.75 · hidden_size`. For `hidden_size = 1024`, `0.75 · 1024 = 768`, so
`head_dim = 128`, `num_heads = 6` (`6 · 128 = 768`), with `expand_v = 2.0` giving value head dim 256 and
value width 1536. Count the budget, because it is a genuine departure from the `~4d²` I held for three
rungs and I want to be honest about what I am buying: `q,k` each `d → 768 = 0.75d`, so `0.75d²` each; `v`,
the output projection, and the `g_proj` gate each touch the 1536 width, `1.5d²` each; total
`0.75 + 0.75 + 1.5 + 1.5 + 1.5 ≈ 6d²`, about `1.5×` softmax's mixer. Over 24 layers the extra `2d²` is
`≈ 50M` parameters, taking the ~353M base to ~404M, about +14%. I carry that caveat into how I read the
result: because the finale is *not* parameter-matched to the prior rungs, a validation-loss improvement is
partly confounded with the extra capacity. But the *WinoGrande* test is clean of that confound — more
capacity applied to DeltaNet's mechanism means a bigger store holding *more* associations undamped, which
if anything makes the "stale bindings never fade" problem *worse*, so any WinoGrande recovery cannot come
from parameters, only from the gate re-introducing a global fade. What the bigger budget buys on the memory
side is a state of `128 × 256 = 32,768` per head times 6 heads `= 196,608`, about `3×` DeltaNet's `65,536`,
which should also help the bulk retrieval metrics. I keep `use_short_conv = True`, `conv_size = 4`,
`use_gate = True`, and `self.use_pos_emb = False` (the scalar decay and delta recurrence carry relative
position). The `Block` stays the scaffold default. (The literal edit is in the answer.)

Against DeltaNet's numbers (val_loss 2.3481, WikiText-2 49.88, LAMBADA 70.48, WinoGrande 49.17), the finale
must do two things *jointly*, and the jointness is the whole falsifiable content. First, hold or improve
DeltaNet's clean-retrieval gains — val_loss at or below 2.3481 and LAMBADA at or below 70.48 — because at
`αₜ ≈ 1` it *is* DeltaNet, and the near-1 init plus ~50M extra parameters both push val_loss down; a
regression here would mean the added freedom is mis-trained. Second, and the metric the finale exists for,
*recover the eraser deficit*: WinoGrande must climb back above 49.17, out of the below-chance band, toward
RetNet's 52.01, because the gate's job is to pull the DeltaNet orthogonal-subspace eigenvalues off 1 so
stale bindings can fade. The one thing I most want to rule out is a collapse to one parent: trained to
`αₜ ≈ 1` everywhere it is just DeltaNet and WinoGrande does not recover; trained to `βₜ ≈ 0` everywhere it
is just gated linear attention and LAMBADA regresses toward GLA's 84.73. Both mechanisms have to stay alive
for the two-sided claim — one clean fetch and one clean fade at once — to hold.
