DeltaNet's numbers confirmed the recall diagnosis almost exactly, and that very success exposes the
one thing it threw away — so before I reach for a finale I want to read the whole three-rung table as
arithmetic, because the *shape* of the improvements, not their mere existence, is what tells me where
the last piece of leverage is. Line the primary metric up first. Validation loss went 2.4795 → 2.4482
→ 2.3481. The two steps are −0.0313 (RetNet to GLA, the decay fix) and −0.1001 (GLA to DeltaNet, the
write-rule fix): the write fix bought about 3.2× the drop the decay fix did. Convert to model
perplexity to feel it — `exp(2.4795) ≈ 11.93`, `exp(2.4482) ≈ 11.57`, `exp(2.3481) ≈ 10.47` — so the
decay fix shaved 3.0% off perplexity and the write fix shaved 9.5%, three times as much again.
WikiText-2 tells the same story louder: 66.67 → 64.32 → 49.88, steps of −2.35 then −14.44, the write
fix worth six times the decay fix. So whatever I do next, I have to respect that the error-correcting
write, not the gate, was where the real quality lived — I must not regress it.

Now the metrics I told myself to watch, read with signs. LAMBADA: 82.36 (RetNet) → 84.73 (GLA, a 2.9%
*regression*) → 70.48 (DeltaNet), a 14.25-point fall from GLA and 11.88 from RetNet — the recall
metric collapsed downward exactly when I fixed the write, precisely as the rung-3 construction
predicted. HellaSwag, stuck at 31.12 then 31.10 across two rungs (six points over its four-way chance
floor of 25, essentially not moving), finally broke to 32.77. ARC-Easy climbed monotonically 51.47 →
53.11 → 53.58. PIQA barely moved, 62.40 → 62.40 → 62.95. Everything that rewards fetching a specific
earlier value moved the right way, and on LAMBADA it moved *hard*. But one number did not: WinoGrande,
52.01 → 49.88 → 49.17. It has fallen at *every single rung*, it is the only metric now below its
rung-1 value, and at 49.88 and 49.17 it is *below 50 twice* — below chance for a binary task. That is
not noise; a two-rung monotone decline ending below the coin-flip floor is a structural signal, and it
is pointing at something the recall story alone does not explain.

Here is the part that took me a second read to see, and it is the whole hinge of the finale. LAMBADA
and WinoGrande are *both* recall-flavored — I lumped them together at rung 2 as "the tasks that reward
tracking a specific earlier token" — yet under DeltaNet they *dissociated*: LAMBADA recovered
spectacularly and WinoGrande kept sinking. So they cannot be testing the same capability. LAMBADA
hands you a passage and asks for its final word, recoverable by *retrieving a specific named entity
once* — a single clean fetch against the store, which is exactly what the delta rule's error-corrected,
content-addressed write is built to make clean. WinoGrande is different in kind: it turns on tracking
which of two candidate antecedents a pronoun binds to, and the binding must *survive a span* of
intervening content while the *other* candidate, and everything else that streamed past, has to fade
out of the way. That is not "retrieve once and read out"; it is "hold the right binding while a lot of
other material streams past, and let go of the stale ones." The first half of that — hold the right
binding cleanly — DeltaNet does superbly, which is why LAMBADA fell. The second half — let the stale
ones fade globally — DeltaNet cannot do at all, and that is why WinoGrande did not follow LAMBADA down.
The two "recall" tasks split along exactly the seam between a content-addressed *write* and a global
*fade*, and DeltaNet has the former and none of the latter.

Let me make the "none of the latter" precise, because I can read it straight off the recurrence and
even off its spectrum. DeltaNet is `Sₜ = S_{t−1}(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ` — there is **no `α`
term at all**, no multiplicative pre-factor on `S_{t−1}`. With L2-normalized keys the transition
`Mₜ = I − βₜ kₜ kₜᵀ` has eigenvalue `1 − βₜ` along the single direction `kₜ` and eigenvalue *exactly 1*
on the entire `d_k − 1`-dimensional subspace orthogonal to `kₜ` (for `u ⊥ kₜ`, `Mₜ u = u`). Eigenvalue
one means those directions of the state *never decay*: whatever was written into them persists,
undamped, forever, until some future key happens to point along them and overwrites it. The Householder
is a scalpel that removes exactly the one overwritten direction and leaves everything else perfectly
intact — which is precisely the wrong behavior for WinoGrande, where I need the stale candidate binding
to *fade on its own* even though no later key ever explicitly overwrites it. Now put the three rungs on
that axis and the monotone decline reads itself. RetNet had a clean, uniform, global fade — a fixed
per-head `γ^{n−m}` that shrinks *every* direction of the state every step — and it had the *highest*
WinoGrande, 52.01. GLA replaced the clean scalar fade with a diagonal data-dependent gate: a blunter
eraser that scales channels rather than fading the whole state uniformly, and WinoGrande dropped to
49.88. DeltaNet removed the fade entirely, pinning `d_k − 1` eigenvalues at exactly 1, and WinoGrande
dropped again to 49.17. The trace 52.01 → 49.88 → 49.17 is the progressive loss of a clean global fade,
metric-ized. DeltaNet is all scalpel and no eraser, and that below-chance WinoGrande is the missing
eraser made visible.

So the ladder has separated two capabilities that, I now see, were never actually in conflict — I just
never had both at once. GLA gave a **data-dependent decay**: a content-chosen forgetting rate, an
eraser, but a *blunt* one, multiplicative and diagonal, scaling the whole state without localizing the
removal to the specific association a new key collides with. DeltaNet gave a **content-addressed
write**: the error-correcting delta rule, a scalpel that removes exactly the stale direction, but with
*no* global fading, so an association persists indefinitely once written and not directly overwritten.
RetNet's failure was that it had no data-dependence anywhere; GLA fixed the decay axis and left the
write; DeltaNet fixed the write axis and dropped the decay. Crucially, these two fixes live on
*orthogonal* structural slots of the same recurrence — one is the multiplicative pre-factor sitting in
front of `S_{t−1}`, the other is the additive-plus-Householder transition structure — and nothing about
them competes for the same slot. There is no reason I cannot apply both. That is the finale: a
recurrence that gates *and* deltas at once, the gate doing rapid global erasure and the delta rule doing
targeted content-addressed updates, the two mechanisms complementary rather than redundant because each
does exactly the job the other structurally cannot.

Write the combined rule down and check it is well-posed, not a hopeful concatenation. I want the
previous state first globally decayed by a data-dependent scalar `αₜ ∈ (0,1]`, then have the delta
rule's targeted erase-and-write applied: `Sₜ = αₜ (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ`, `oₜ = Sₜ qₜ`.
The way to trust a "combination" is to read its limits and confirm it degenerates to each parent
exactly, not to some third thing. Set `αₜ = 1` for all `t`: the pre-factor is the identity and I get
back `Sₜ = (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ`, *exactly* DeltaNet, rung 3, character for character.
Set `βₜ = 0`: the delta write `βₜ vₜ kₜᵀ` vanishes and the Householder `I − 0·kₜkₜᵀ` collapses to the
identity, leaving `Sₜ = αₜ S_{t−1}` — the pure scalar-gated decay skeleton that the whole
scalar-gated-linear-attention family (and Mamba2) is built on; that family then bolts its own additive
write back on top, and what the `βₜ = 0` limit isolates is exactly the gating mechanism GLA also lives
by (GLA's is per-channel diagonal, this one per-head scalar, but the same idea). So the combined rule
sits strictly *above* both rung 2 and rung 3 as special cases. That is precisely what I want from a
finale and it is a real guarantee, not decoration: a strict generalization cannot be worse than the
better of its two parents *unless the extra freedom is mis-trained*, and it can be strictly better
wherever a task needs both a global eraser and a local scalpel — which is exactly the WinoGrande
profile the monotone decline flagged.

Now the design decisions, and I have to earn each one rather than inherit it, because a combined rule
opens failure modes neither parent had. The first and load-bearing one: what *shape* is `αₜ`? The
tempting default is to reuse GLA's gate wholesale — a per-key-channel diagonal `Diag(αₜ)` — on the
theory that more expressive is better and I already have the machinery. Let me actually walk that path a
few steps before taking it, because it is the obvious move and I want to see it fail concretely rather
than wave it off. The combined transition would be `Diag(αₜ)(I − βₜ kₜ kₜᵀ)`. Two checks kill it. First,
composition into the chunk kernel. DeltaNet is trainable only because the product of its transitions,
`∏(I − βₜ kₜ kₜᵀ)`, has a WY / UT-transform representation `I − Σ wₜ kₜᵀ` — an *identity minus a sum of
rank-one terms sharing the right factors `kₜᵀ`* — which is what lets forward substitution turn the
sequential recurrence into matmuls. Multiply out a single diagonal-gated factor:
`Diag(αₜ)(I − βₜ kₜ kₜᵀ) = Diag(αₜ) − βₜ (αₜ ⊙ kₜ) kₜᵀ`. That is a *diagonal minus rank-one* — neither
"identity minus rank-one" nor diagonal — and a product of such factors does not collapse into the clean
WY form, because the diagonal parts do not pull through the rank-one parts. The UT transform that made
DeltaNet trainable breaks. Second, a commutation check that also exposes an ambiguity. A *scalar* `αₜ`
commutes with everything: `αₜ (I − βₜ kₜ kₜᵀ) = (I − βₜ kₜ kₜᵀ) αₜ`, so "decay then delta" and "delta
then decay" are the *same* transition, no ordering to choose. A diagonal `Diag(αₜ)` does *not* commute
with the Householder — `Diag(αₜ) kₜ kₜᵀ = (αₜ ⊙ kₜ) kₜᵀ` scales the rows while `kₜ kₜᵀ Diag(αₜ) =
kₜ (αₜ ⊙ kₜ)ᵀ` scales the columns, and those differ unless `αₜ` is uniform — so a diagonal gate would
force an arbitrary decay-before-or-after-delta choice on top of breaking the kernel. And there is a
third reason it is not even wanted: the delta rule *already* supplies fine-grained, per-direction
control, since the Householder picks out the exact subspace to erase. A diagonal gate would be redundant
with that; what the delta rule cannot do is uniform *global* fading, and a scalar `αₜ` is precisely that
one missing thing and nothing more, keeping the two mechanisms on strictly non-overlapping jobs. So the
scalar is not the lazy choice, it is the *unique* choice: it is the only gate shape that (a) does the one
job the delta rule lacks, (b) is redundant with nothing the delta rule already does, and (c) pulls
cleanly out of the chunkwise cumulative product — the same property that made RetNet's fixed `γ`
parallelizable — so the combined rule keeps DeltaNet's exact matmul UT-transform training form, with the
decay folded in as a per-position scalar log-cumsum riding on top. The chunk algorithm becomes
"RetNet-style scalar decay cumsum" × "DeltaNet UT-transform," each of which I already know composes into
matmuls. `αₜ` is a **per-head scalar**.

One more structural check on the recurrence itself, because the placement of `αₜ` is not free. I wrote
`Sₜ = αₜ (I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ`, with `αₜ` multiplying *only the carried state*, not the
fresh write. Suppose instead I had written `αₜ [(I − βₜ kₜ kₜᵀ) S_{t−1} + βₜ vₜ kₜᵀ]`, factoring the
decay over everything. Then the current token's own write `βₜ vₜ kₜᵀ` would be attenuated by `αₜ` at the
very instant it is written — I would be forgetting a fact in the same step I record it, which is
nonsense for a memory. The correct operation is: fade the *past*, then add the *fresh* write undecayed.
So `αₜ` belongs outside the Householder-times-state product and outside the new write. And because `αₜ`
is scalar it commutes with the Householder, so this is unambiguous — decaying then erasing equals
erasing then decaying — which is a small but real dividend of having chosen the scalar over the diagonal.
`βₜ` and `αₜ` are both functions of `xₜ` alone (input-only gating), so the recurrence stays linear in
the state and parallelizable, the same discipline GLA needed to keep its chunk form.

Next, how do I parameterize `αₜ` so it trains well, because a naive gate collapses long memory before
training even starts. A fresh sigmoid sits near 0.5 — that would halve the state every step, a half-life
of one token, and long-range capacity dies at initialization; this is the exact trap GLA had to defuse
with its temperature. Borrow the Mamba2 discretization, which is built to give a long-memory prior by
construction. Compute a positive per-token timescale `Δₜ = softplus(a_proj(xₜ) + dt_bias)` with `dt_bias`
initialized so `Δₜ` starts small, and a per-head positive rate `A = exp(A_log)`; the log-decay is
`gₜ = −A · Δₜ ≤ 0` and `αₜ = exp(gₜ) ∈ (0,1]`. Put init numbers on it. With `Δₜ ≈ 0.01` and `A ≈ 1`,
`gₜ ≈ −0.01`, so `αₜ ≈ e^{−0.01} ≈ 0.99` and the half-life is `ln 0.5 / ln 0.99 ≈ 0.693/0.01 ≈ 69`
tokens. Sanity-check that against the neighbors on the ladder: GLA's temperature-16 gate initialized at
`0.5^{1/16} = 2^{−1/16} ≈ 0.9576`, a half-life of `0.693/0.0424 ≈ 16` tokens; RetNet's fastest head had
a half-life around 22 tokens and its slower in-block heads out to ~710. So the finale's default fade,
~70 tokens, sits deliberately *slower* than GLA's aggressive 16 and inside RetNet's in-block range,
starting close to "barely forgetting" and letting the content pull `αₜ` down only when it asks. And the
number that matters most: at init `αₜ ≈ 0.99`, i.e. the finale starts life as *almost exactly DeltaNet*
(`αₜ = 1` is DeltaNet), which is the right place to start given DeltaNet is the strongest rung — it
begins from the best parent and *learns* to fade, rather than starting somewhere it has to climb back
from. This is my hedge against one of the two collapse modes I will state at the end.

Stability I derive rather than assume, because a combined transition can leave the unit disk in ways
neither parent did. `Mₜ = αₜ (I − βₜ kₜ kₜᵀ)` with `‖kₜ‖₂ = 1`. On the `d_k − 1` directions orthogonal
to `kₜ`, `Mₜ` acts as `αₜ · I`, eigenvalue `αₜ`; along `kₜ`, `Mₜ kₜ = αₜ(1 − βₜ) kₜ`, eigenvalue
`αₜ(1 − βₜ)`. So the spectrum is `{αₜ` (multiplicity `d_k − 1`)`, αₜ(1 − βₜ)` (multiplicity 1)`}`. With
`αₜ ∈ (0,1]` and `βₜ ∈ [0,1]`, both families satisfy `0 ≤ αₜ(1 − βₜ) ≤ αₜ ≤ 1` — inside the closed unit
disk, stable by DeltaNet's argument plus a strictly-contractive scalar; the gate can only *shrink* the
already-stable spectrum, never push it out. And this spectrum is the eraser made literal: DeltaNet pinned
`d_k − 1` eigenvalues at exactly 1 (the directions that never fade), and the gate is exactly the move
that pulls those from 1 down to `αₜ < 1`. That single spectral change — the orthogonal subspace's
eigenvalue coming off 1 — *is* the global fade WinoGrande has been asking for across three rungs. So I
keep DeltaNet's SiLU-then-L2 on `q`/`k` (which is what forces `‖kₜ‖ = 1` and makes the Householder an
exact projection at `βₜ = 1`), the depthwise short convolution, and the learned `βₜ = σ(W_β xₜ)`.

The combined rule also reopens one decision DeltaNet had deliberately closed: the **output gate**. At
rung 3 I set `use_gate = False` on purpose, to isolate the write-rule change and not confound it with
the swish output gate that RetNet and GLA both carried — I wanted DeltaNet's gain to be attributable to
the write and nothing else. But the finale is no longer isolating a single axis; it is the full
combination, so that reason has expired, and there is now a positive argument to bring the gate *back*.
Re-introducing a data-dependent decay means the per-head outputs again carry content-dependent,
head-varying scale — the same variance-across-heads problem the per-head norm and the swish gate were
built to tame in GLA and RetNet, where I sized the roughly-8× spread of per-head output norms. And the
swish gate is the cheap restoration of the nonlinearity that deleting softmax cost me in the first place.
Both reasons that justified the gate for the gated-linear-attention lineage are live again the moment
`αₜ` returns. So `use_gate = True`, routing the output through a fused gated RMSNorm with a `g_proj`
swish gate — the GLA/RetNet output recipe — rather than DeltaNet's bare per-head RMSNorm. I name this as
a deliberate reversal of the rung-3 edit: the finale is the *union* of both lineages, not the write-rule
axis in isolation, so the output-gate machinery of the decay lineage rejoins it.

Now make it concrete in this task's edit surface, and this is where a harness-specific subtlety forces
an arithmetic I have to get exactly right or the layer will not even instantiate. FLA ships
`GatedDeltaNet` with the `chunk_gated_delta_rule` kernel that implements the combined recurrence above:
its forward computes the per-head log-decay `g = −exp(A_log)·softplus(a_proj(x) + dt_bias)`, the learned
`β = sigmoid(b_proj(x))`, applies SiLU and the short conv to `q`/`k`/`v`, L2-normalizes `q`/`k` inside
the kernel, and runs the UT-transform chunk algorithm with the scalar decay folded in as the chunk-local
cumsum I argued for. So my job is configuration, not re-derivation. The subtlety is head shaping: with
`use_gate = True` this layer does *not* honor `n_head` directly but follows the Mamba2 convention
`num_heads · head_dim = 0.75 · hidden_size`. For GPT-2 Medium, `hidden_size = 1024`, so
`0.75 · 1024 = 768`, and I need integer heads times head-dim to hit 768 — `head_dim = 128`,
`num_heads = 6`, since `6 · 128 = 768`, satisfies it, with `expand_v = 2.0` giving a value head dim of
`2 · 128 = 256` and a value width of `6 · 256 = 1536`. Let me count the parameter budget this lands at,
because it is a genuine departure from the `~4d²` I held for the previous three rungs and I want to be
honest about what I am buying. The `q` and `k` projections each map `d → 768 = 0.75d`, so `0.75d²` each;
the `v` projection, the output projection, and the `g_proj` swish gate each touch the 1536 value width,
`1.5d²` each; total `0.75 + 0.75 + 1.5 + 1.5 + 1.5 ≈ 6d²`, plus the tiny `a_proj`, `b_proj`, `dt_bias`,
`A_log`, and short-conv parameters. That is *larger* than softmax's `4d²` and larger than the earlier
rungs — about `1.5×` the mixer. Sized against the whole model: the extra `2d²` per layer over 24 layers
is `48d² ≈ 48 · 1.05M ≈ 50M` parameters, taking the ~353M base to roughly ~404M, about +14%. I have to
carry that caveat into how I read the finale's result: because it is *not* parameter-matched to the
prior rungs, a validation-loss improvement is partly confounded with the extra capacity and cannot be
attributed to the mechanism alone. But — and this is why the caveat does not undermine the central claim
— the *WinoGrande* test is clean of that confound. More capacity, applied to DeltaNet's mechanism, means
a bigger store holding *more* associations undamped, which if anything makes the "stale bindings never
fade" problem *worse*, not better; so any recovery of WinoGrande cannot come from the extra parameters —
it can only come from the gate re-introducing a global fade. Capacity confounds val_loss; it does not
confound the eraser test. What the bigger budget does buy on the memory side is a state of
`128 × 256 = 32,768` entries per head times 6 heads `= 196,608`, about `3×` DeltaNet's
`64 × 64 × 16 = 65,536` — a materially larger associative store, which should also help the bulk
retrieval metrics. I keep `use_short_conv = True`, `conv_size = 4` (the local-comparison short conv, for
the same reason DeltaNet needed it — pure content-addressing is blind to adjacency), `use_gate = True`,
and set `self.use_pos_emb = False` for the reason every rung since RetNet has: the scalar decay and the
delta recurrence together carry relative position, so the loop must skip its learned `wpe` or it would
double-encode. The `Block` stays the scaffold default; only the mixer is swapped, so the comparison to
the prior rungs stays as clean as the changed budget allows.

So the finale is the natural close of the ladder, and I can state exactly what it inherits from each
rung: RetNet established that a fixed decay is not enough — the fade must be data-dependent; GLA made the
decay data-dependent but left the additive Hebbian write and paid for it on LAMBADA; DeltaNet fixed the
write with an error-correcting content-addressed swap but dropped the decay and paid for it on
WinoGrande; and the gated delta rule applies *both fixes on their orthogonal axes at once* —
`Sₜ = αₜ(I − βₜ kₜ kₜᵀ)S_{t−1} + βₜ vₜ kₜᵀ` — a data-dependent global eraser composed with a
content-addressed scalpel, reducing to DeltaNet at `αₜ = 1` and to scalar-gated linear attention at
`βₜ = 0`. Here is the bar it has to clear, stated against the real numbers, since this rung gets no
feedback of its own and I have to reason the prediction all the way out. The strongest baseline is
DeltaNet: val_loss **2.3481**, WikiText-2 **49.88**, LAMBADA **70.48**, ARC-Easy **53.58**, HellaSwag
**32.77**, PIQA **62.95**, and the open wound WinoGrande **49.17**. For the finale to earn the endpoint
it must do two things *jointly*, and the jointness is the whole falsifiable content. First, hold or
improve DeltaNet's clean-retrieval gains — validation loss at or below 2.3481 and LAMBADA at or below
70.48 — because at `αₜ ≈ 1` it *is* DeltaNet, and the near-1 init plus the ~50M extra parameters both
push val_loss down, so any *regression* here would mean the added freedom is being mis-trained rather
than used. The bulk downstream metrics (HellaSwag ~32.77, ARC-Easy ~53.58, PIQA ~62.95, WikiText-2
~49.88) I expect to hold or edge up for the same "it's DeltaNet-plus-capacity at init" reason; I would
be surprised to see any of them regress. Second, and this is the metric the finale actually exists for,
*recover the eraser deficit*: WinoGrande must climb back above 49.17, out of the below-chance band, and
ideally toward RetNet's 52.01 — because WinoGrande is exactly the "hold the right binding while content
streams past, let the stale ones fade" profile, and the gate's job is precisely to pull the DeltaNet
orthogonal-subspace eigenvalues off 1 so the stale bindings can fade. The single thing I most want to
rule out is that the combined rule collapses to one of its parents rather than using both. If it trains
to `αₜ ≈ 1` everywhere, it is just DeltaNet again and WinoGrande does *not* recover; the Mamba2-style
near-1 initialization is a hedge that keeps `αₜ` alive and slightly below 1 without pinning it, but it is
only a hedge. If it trains to `βₜ ≈ 0` everywhere, it is just gated linear attention and LAMBADA
regresses back toward GLA's 84.73; the learned `βₜ` inherited from the rung that already drove LAMBADA to
70.48 is the hedge against that. So the falsifiable endpoint claim is specific and two-sided: with both
mechanisms kept alive, validation loss drops below 2.3481 *and* WinoGrande recovers above 49.17 — the two
things three rungs of single-axis fixes could never hold together, one clean fetch and one clean fade at
the same time.
