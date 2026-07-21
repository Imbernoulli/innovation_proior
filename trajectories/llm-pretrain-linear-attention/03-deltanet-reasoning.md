GLA's numbers landed where I bet on the bulk metric and failed where I flagged the risk, and the split
between the two is the whole story, so I read it as arithmetic before mechanism. On the bulk-and-local axis
everything moved right: validation loss dropped from RetNet's 2.4795 to 2.4482 (perplexity `11.9 → 11.6`),
WikiText-2 tightened from 66.67 to 64.32 (−3.5%), ARC-Easy rose 51.47 → 53.11. The data-dependent diagonal
gate bought exactly the local gain I predicted. But now the two metrics I said to watch, read with signs.
LAMBADA got *worse* — 84.73 versus RetNet's 82.36, a 2.9% regression — and WinoGrande *fell* 52.01 → 49.88,
not just a drop but a drop *below 50*, below chance for a binary task, so the model now reads the
coreference structure worse than a coin. HellaSwag sat flat (31.10 vs 31.12), PIQA at exactly 62.40 both
times. Line the signs up: val_loss, WikiText-2, ARC-Easy improved; LAMBADA and WinoGrande regressed;
HellaSwag and PIQA did nothing. That is not noise around a uniform improvement — a uniform gain moves
everything the same direction — it is a *clean dissociation*, two metrics going the opposite way from the
three that improved. And the two that regressed are exactly the recall-flavored ones: LAMBADA asks for the
final word of a passage, recoverable only by *retrieving* a specific entity named earlier; WinoGrande turns
on resolving a reference to the right antecedent. The metrics that improved reward smooth local language
modeling; the metrics that stalled or regressed reward *associative recall*. So the diagnosis is exactly
the doubt I closed the GLA derivation with, now confirmed by the sign structure: GLA's gate, however
data-dependent, is *diagonal and multiplicative*. It can decide how fast each channel forgets but cannot
reach into the state and remove the *specific stored association* that collides with the new key. The
failure is not in the decay anymore. It is in the **write rule**.

Look at why a multiplicative gate cannot do targeted removal, and put a number on the damage. RetNet, GLA,
and the whole gated-linear-attention family share one template: `Sₜ = S_{t−1} ⊙ Mₜ + kₜᵀvₜ`, a decay times
the old state plus an additive outer-product write. The write is *Hebbian* — token `t` stamps `kₜᵀvₜ` in
and the decay is the only thing that ever shrinks anything. Write `S = Σᵢ vᵢ kᵢᵀ` and probe with a stored
key `k_j`: `S k_j = vⱼ (kⱼᵀkⱼ) + Σ_{i≠j} (kᵢᵀk_j) vᵢ`. The first term is what I wanted; the second is
cross-talk from every key not orthogonal to `k_j`. Now the arithmetic that makes it fatal at *this* scale:
the head dimension is `n_embd/n_head = 1024/16 = 64`, so each head's key lives in a 64-dim space, but the
block is 1024 tokens — a 64-dim store asked to hold up to 1024 associations, sixteen times overcapacity. In
64 dimensions there are at most 64 mutually orthogonal vectors, so past position 64 the keys *cannot* be
orthogonal and cross-talk is guaranteed. Size it: for random unit keys each `kᵢᵀk_j` is about `N(0, 1/64)`,
std `1/8`; summing `L−1 ≈ 1023` of them in random value directions gives interference norm about
`(1/8)·√1023 ≈ 4` times `‖v‖`, while the signal is just `‖vⱼ‖`. The retrieval signal is buried four-to-one
under interference and it only worsens as I write more. The multiplicative gate scales the *whole* state
before each write; it cannot say "the association I stored for *this* key now collides — erase *that one*."
Targeted removal would require the previous state `S_{t−1}` to enter the *write itself*, interacting with
the incoming key, not merely be scaled by a decay. An elementwise gate is structurally the wrong tool for
the collision problem, and LAMBADA going *backward* under GLA is that wrongness made visible: GLA's
content-chosen decay actually made the interference worse on the one task that most needs clean retrieval.

Before changing the write, ask whether the cheaper moves that leave the additive Hebbian rule alone could
rescue it — because if widening the state fixes recall I do not need a new rule. Two obvious ones. First,
make the key dimension bigger so the store is not overcapacity: the cross-talk scaled like `√(L/d_k)`, so
bringing it to the signal needs `d_k ≈ L = 1024`, a `1024×d_v` state and a `q,k` projection blown up well
past the `4d²` budget — and even then, the moment the sequence runs a little past `d_k` the keys are
non-orthogonal again. Widening *delays* the collision linearly and never removes it, at quadratic parameter
cost. Second, lift the keys through a nonlinear feature map `φ(k)` to raise the effective store dimension —
but the write is still `S += vᵢ φ(kᵢ)ᵀ`, still additive, still Hebbian, so the cross-talk just moves into
the lifted space, same disease in a bigger room. Both treat the symptom (too many collisions in too small a
space) and leave the cause (a write that only ever *adds*, never *removes*). So the fix has to be in the
write rule, and I want the *minimal* change that lets the old contents participate in the removal.

What write rule lets the old contents shape the removal? The classical adaptive-filtering idea — the delta
rule, Widrow-Hoff least-mean-squares. Treat `S` as a little regressor supposed to map `kₜ → vₜ`, and
instead of blindly adding, take one gradient step on the squared prediction error `Lₜ(S) = ½‖S kₜ − vₜ‖²`.
Its gradient is `(S kₜ − vₜ) kₜᵀ`, the outer product of the *residual* with the key, so one SGD step with
rate `βₜ` is `Sₜ = S_{t−1} − βₜ (S_{t−1} kₜ − vₜ) kₜᵀ`. Read it as a value swap: retrieve the old value
`vₜ^old = S_{t−1} kₜ`, blend `vₜ^new = βₜ vₜ + (1−βₜ) vₜ^old`, replace. The write is proportional to the
error `vₜ − vₜ^old`: if `S_{t−1}` already maps `kₜ` close to `vₜ`, almost nothing happens; if it maps `kₜ`
to a stale value, the correction is strong. Contrast the additive rule's implicit *linear* loss
`−⟨S kₜ, vₜ⟩`, whose gradient `−vₜ kₜᵀ` is constant regardless of how wrong the prediction is — the
no-error-correction behavior that drove the cross-talk and GLA's LAMBADA regression. The quadratic loss
self-corrects, and the delta-rule fast weight has been known for decades to have higher capacity than the
Hebbian one. The scalar `βₜ = σ(W_β xₜ) ∈ (0,1)` is a learned *writing strength*: at 1 the old value is
fully overwritten, at 0 the memory is untouched. This is the write rule I want.

The catch is the training wall. For additive linear
attention and GLA the value written at step `t` is `vₜ` (or `vₜ`-after-decay), independent of the running
state, so the whole output is one masked matmul `O = (Q Kᵀ ⊙ M) V` over precomputed `V` — matmul-rich,
tensor-core-bound. The delta rule breaks that: `vₜ^new` is tangled up with `vₜ^old = S_{t−1} kₜ`, which
depends on the *running state*. I cannot stack the writes ahead of time, because each needs `S_{t−1}`, the
output of all previous writes. The naive computation rolls the recurrence forward, materializing the
`d_k×d_v` state at every step — and size it, because it is why this rule stayed off the ladder: `64×64 =
4096` floats per head per step, times 16 heads, 1024 steps, micro-batch 32, is `2.1×10⁹` floats, about
8.6 GB in fp32 for the per-step states of a *single* layer before the other 23. Strictly sequential,
elementwise, never touching a tensor core, memory-infeasible if materialized. So I get the better write
rule only if I can break the state-dependence and recover a matmul form.

Look at the update as a transition on `S`. Substitute the blend into the swap:
`Sₜ = S_{t−1}(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ` — the previous state *multiplied* by an identity-plus-rank-one
matrix (a generalized Householder) plus a rank-one additive term. That structure is the lever. First, keep
`S` in the additive shape of vanilla linear attention: claim `Sₜ = Σ_{i≤t} uᵢ kᵢᵀ` for "pseudo-values"
`uᵢ`. Induct — `S₁ = β₁ v₁ k₁ᵀ` gives `u₁ = β₁ v₁`, and expanding the recurrence and collecting the `kₜᵀ`
term gives `uₜ = βₜ(vₜ − Σ_{i<t} uᵢ (kᵢᵀkₜ))`, which matches the value-blend interpretation
(`uₜ = vₜ^new − vₜ^old`). So the delta layer is vanilla linear attention with `vᵢ` replaced by the
error-corrected pseudo-value `uᵢ`, the per-token matrix state never materialized — the problem reduces to
*computing the `uᵢ`*, and every downstream matmul from vanilla linear attention applies. Second, the
chunkwise form also needs the product of transitions `Pₙ = ∏_{t≤n}(I − βₜ kₜ kₜᵀ)`, and the same induction
gives the WY representation `Pₙ = I − Σ_{t≤n} wₜ kₜᵀ` with `wₜ = βₜ(kₜ − Σ_{i<n} wᵢ(kᵢᵀkₙ))` — the *exact
same recurrence* as `uₜ` with `kₜ` in place of `vₜ`. Both `uₜ` and `wₜ` are still sequential within a
chunk, so I have moved the bottleneck, not removed it.

Remove it: read the recurrence as the linear system it is. The rows of `W` satisfy a strictly-lower-triangular
dependency `(I + L) W = B K` with `B = diag(β)` and `L = tril(diag(β) K Kᵀ, −1)`, so `W = (I + L)⁻¹ B K =
T K` and identically `U = T V`, with `T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)` — the *same* `T` for
both. The sequential dependency is absorbed into one matrix `T`, and `I + L` is unit lower-triangular so its
inverse is cheap and matmul-friendly via forward substitution (the UT transform for accumulating Householder
products). Now every part is a matmul — `T` by forward substitution, `W = T K`, `U = T V`, the masked
intra-chunk products, the chunk-state update — with only `L/C` sequential steps between chunks. Cost
`O(LCd + Ld²)`, recomputing chunk states in the backward, exactly the asymptotics and hardware profile of
the chunkwise GLA I ran at rung two. The delta rule, which looked irreducibly sequential and
memory-infeasible, is now matmul-rich and materializes `L/C` chunk states instead of `L` per-step states.
That is the whole reason it can sit on *this* ladder.

Stability has to be derived, because `Mₜ = I − βₜ kₜ kₜᵀ` will blow up if its eigenvalues leave the unit
disk. `Mₜ` acts as the identity on the `d_k−1`-dim subspace orthogonal to `kₜ` (eigenvalue 1) and scales
the `kₜ` direction by `Mₜ kₜ = (1 − βₜ‖kₜ‖²) kₜ`. So the one non-unit eigenvalue is `1 − βₜ‖kₜ‖²`, and
stability wants `0 ≤ βₜ‖kₜ‖² ≤ 2`. If `‖kₜ‖` is left unbounded a large key pushes it below `−1` and the
recurrence diverges. L2-normalizing the keys fixes it exactly: with `‖kₜ‖₂ = 1` the contractive eigenvalue
is `1 − βₜ ∈ [0,1]`, always stable. And at the boundary `βₜ = 1`, `Mₜ = I − kₜ kₜᵀ` has that eigenvalue 0
— an orthogonal *projection* that annihilates exactly the one-dimensional subspace spanned by `kₜ` and
leaves the other `d_k−1` dimensions untouched. That is targeted forgetting made literal: a full write
erases exactly the direction being overwritten, the content-addressed deallocation the diagonal gate could
not localize and that LAMBADA punished GLA for missing. So L2 normalization is what makes the erase
surgical. I L2-normalize `q` and `k`, apply SiLU before the normalization (keeps sign, smooth, no hard
zeroing), use `βₜ = σ(W_β xₜ)` (one sigmoid scalar per head, negligible parameters), and add a lightweight
depthwise **short convolution** (kernel 4) on the `q`, `k`, `v` projections. It earns its place: the delta
rule is pure content-addressing — it retrieves by matching the current key against stored keys — and that
is exactly the operation *blind to adjacency*. "The token immediately before," "the two-word phrase I just
saw" are relations of position, not content, and a dot-product against the state cannot express them unless
position was baked into the representation. A depthwise conv of kernel 4 mixes each channel over the last
four tokens before the projections, so the key filed and the query probed already carry a little local
n-gram context — the layer can compare short spans, not just single tokens — at only `4·d` parameters per
projection. An output RMSNorm per head before the projection rounds it out.

Concrete in the edit surface: FLA ships `DeltaNet` with the chunk kernel that implements the UT-transform
path, so the edit imports it — `hidden_size = 1024`, `num_heads = 16`, `use_beta = True` (the learned
writing strength), `use_short_conv = True`, `conv_size = 4`, `qk_activation = 'silu'`, `qk_norm = 'l2'`
(the SiLU-then-L2 that makes the transition an exact projection at `βₜ = 1`). I take the default
`expand_k = expand_v = 1.0` — symmetric width, state `d×d` — and the budget checks out: `q,k,v` at `d²`
each is `3d²`, `o_proj` is `d²`, the short conv and the `βₜ` projection are negligible, so the layer is
`~4d²`, matched to softmax and RetNet. And `use_gate = False`: DeltaNet does *not* add the swish output
gate here, only the per-head RMSNorm, because the error-correcting write is already the expressivity I am
buying and I want this rung to isolate the *write rule* change, not confound it with the output gate GLA and
RetNet carried — which also makes this mixer a touch *leaner* than GLA's. I set `self.use_pos_emb = False`
— the recurrence and short conv handle ordering. The `Block` stays the scaffold default. (The literal edit
is in the answer.)

The delta from rung two: GLA fixed the *decay* but left the additive Hebbian *write*, the source of the
four-to-one cross-talk and the LAMBADA regression; this rung fixes the write itself, replacing the additive
stamp with an error-correcting content-addressed swap, and recovers a matmul form via the pseudo-value
reduction and the UT transform. The falsifiable claim: validation loss should drop
below 2.4482, because higher capacity and cleaner retrieval help the bulk objective too. But the *real*
test is the recall metrics GLA stalled or regressed on: **LAMBADA must come down hard** — 84.73 under GLA,
82.36 under RetNet, and it should fall well below both, because LAMBADA is exactly "retrieve the entity
named earlier" and the projection-style erase is exactly clean retrieval. WikiText-2 should tighten below
64.32, and HellaSwag (stuck at 31.1 across both prior rungs) and ARC-Easy should rise. If LAMBADA does
*not* improve over GLA, the recall diagnosis was wrong and the failure lives somewhere I have not named;
but the whole construction predicts it will, and that this is the strongest of the three precisely on the
recall axis the diagonal gate could never address.
