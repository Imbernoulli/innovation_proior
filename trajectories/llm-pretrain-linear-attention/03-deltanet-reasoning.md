GLA's numbers landed where I bet they would on the bulk metric, and they failed where I flagged the
risk — and the split between the two is the whole story for this rung, so I want to read it as arithmetic
before I read it as a mechanism. On the bulk-and-local axis everything moved the right way: validation
loss dropped from RetNet's 2.4795 to 2.4482 (a perplexity of `exp(2.4795) ≈ 11.9` down to
`exp(2.4482) ≈ 11.6`), WikiText-2 tightened from 66.67 to 64.32 (−3.5%), and ARC-Easy rose from 51.47 to
53.11 (+1.6 points). So the data-dependent diagonal gate bought exactly the local language-modeling gain
I predicted: holding the right facts at a content-chosen rate beats an exponential average at a fixed
rate, and the bulk cross-entropy reflects it. But now read the two metrics I explicitly said to watch,
and read them with signs. LAMBADA perplexity got *worse* — 84.73 versus RetNet's 82.36, a 2.9% regression
— and WinoGrande *fell* from 52.01 to 49.88, which is not just a two-point drop, it is a drop *below 50*,
below chance for a binary task, so the gated model is now reading the coreference structure worse than a
coin. HellaSwag sat flat at 31.10 against 31.12, and PIQA at exactly 62.40 both times. Line the signs up:
val_loss, WikiText-2, ARC-Easy all improved; LAMBADA and WinoGrande both regressed; HellaSwag and PIQA
did nothing. That is not noise around a uniform improvement — a uniform gain would move everything the
same direction by roughly the same relative amount — it is a *clean dissociation*, two metrics going the
opposite way from the three that improved. And the two that regressed are exactly the recall-flavored
ones: LAMBADA gives you a passage and asks for the final word, recoverable only if the model is
*retrieving* a specific entity named earlier, sometimes far back; WinoGrande turns on resolving a
reference to the right antecedent. The metrics that improved reward smooth local language modeling; the
metrics that stalled or regressed reward *associative recall* — fetch the particular value you stored
earlier, keyed on a token you have now seen again. So the diagnosis is exactly the doubt I closed the
GLA derivation with, now confirmed by the sign structure of the numbers: GLA's gate, however
data-dependent, is *diagonal and multiplicative*. It can decide how fast each channel forgets, but it
cannot reach into the state and remove the *specific stored association* that collides with the new key.
The failure is not in the decay anymore. It is in the **write rule**.

Let me look hard at why a multiplicative gate cannot do targeted removal, and put a number on the
damage, because the number tells me how urgent the fix is. Both RetNet and GLA, and the whole
gated-linear-attention family, share one template: `Sₜ = S_{t−1} ⊙ Mₜ + kₜᵀvₜ`, a decay times the old
state plus an additive outer-product write. The write is *Hebbian* — token `t` stamps `kₜᵀvₜ` into the
state and the decay is the only thing that ever shrinks anything. Read back what that store holds. Write
`S = Σᵢ vᵢ kᵢᵀ` and probe with a stored key `k_j`: `S k_j = vⱼ (kⱼᵀkⱼ) + Σ_{i≠j} (kᵢᵀk_j) vᵢ`. The
first term is what I wanted; the second is cross-talk from every key not orthogonal to `k_j`. Now the
arithmetic that makes this fatal at *this* scale: the head dimension here is `n_embd / n_head = 1024/16 =
64`, so each head's key lives in a 64-dimensional space, but the block is 1024 tokens long — I am asking
a 64-dimensional store to hold up to 1024 associations, sixteen times overcapacity. In `d_k = 64`
dimensions there are at most 64 mutually orthogonal vectors, so past position 64 the keys *cannot* be
orthogonal and the cross-talk is guaranteed. Size it: for roughly random unit keys, each cross-term
coefficient `kᵢᵀk_j` is about `N(0, 1/d_k)`, standard deviation `1/8`; summing `L−1 ≈ 1023` of them in
random value directions gives an interference norm of about `(1/8)·√1023 ≈ 4` times `‖v‖`, while the
signal I wanted is just `‖vⱼ‖`. The retrieval signal is buried roughly four-to-one under interference,
and it only gets worse as I write more. The multiplicative gate scales the *whole* state (per channel)
before each write; it cannot say "the association I previously stored for *this* key now collides with
the new key — erase *that one* and leave everything else." Targeted, content-addressed removal would
require the previous state `S_{t−1}` to enter the *write itself*, interacting with the incoming key — not
merely to be scaled by a decay. An elementwise gate is structurally the wrong tool for the collision
problem, and LAMBADA going *backward* under GLA is that wrongness made visible: GLA's content-chosen
decay actually made the interference pattern worse on the one task that most needs clean retrieval.

Before I change the write, I should ask whether the cheaper moves — the ones that leave the additive
Hebbian rule alone — could rescue it, because if widening the state fixes recall I do not need a new rule
at all. Two obvious ones. First, just make the key dimension bigger so the store is not overcapacity: the
cross-talk norm scaled like `√(L/d_k)`, so to bring it down to the signal I would need `d_k ≈ L = 1024`,
which at one head is a `1024×d_v` state and a `q,k` projection blown up accordingly, well past the `4d²`
budget — and even then, the moment the sequence runs a little past `d_k` the keys are non-orthogonal
again and the interference returns. Widening *delays* the collision linearly in `d_k` but never removes
it; it buys margin at quadratic parameter cost. Second, lift the keys through a nonlinear feature map
`φ(k)` (the performer/elu-style trick) to raise the *effective* dimension of the store. But the write is
still `S += vᵢ φ(kᵢ)ᵀ`, still additive, still Hebbian — the cross-talk just moves into the lifted space,
same disease in a bigger room, and I have paid a feature-map that does not touch the mechanism doing the
damage. Both alternatives treat the symptom (too many collisions in too small a space) and leave the
cause (a write that only ever *adds*, never *removes*). So the fix has to be in the write rule itself, and
I want the *minimal* change that lets the old contents participate in the removal.

What write rule lets the old contents shape the removal? Go back to the classical adaptive-filtering
idea — the delta rule, Widrow and Hoff's least-mean-squares. Treat `S` as a little regressor that is
supposed to map `kₜ` to `vₜ`, and instead of blindly adding, take one gradient step on the squared
prediction error. The loss is `Lₜ(S) = ½‖S kₜ − vₜ‖²`, its gradient w.r.t. `S` is `(S kₜ − vₜ) kₜᵀ`,
the outer product of the *residual* with the key, so one SGD step with rate `βₜ` is
`Sₜ = S_{t−1} − βₜ (S_{t−1} kₜ − vₜ) kₜᵀ`. This is exactly the targeted removal the gate could not do.
Read it as a value swap: retrieve the old value `vₜ^old = S_{t−1} kₜ`, blend `vₜ^new = βₜ vₜ +
(1−βₜ) vₜ^old`, and replace — `Sₜ = S_{t−1} − vₜ^old kₜᵀ + vₜ^new kₜᵀ` — removing the old association for
*this key* and writing the new one. The write is proportional to the error `vₜ − vₜ^old`: if `S_{t−1}`
already maps `kₜ` close to `vₜ`, almost nothing happens; if it maps `kₜ` to a stale value, the
correction is strong. Contrast the additive rule's implicit loss, the *linear* `−⟨S kₜ, vₜ⟩`, whose
gradient `−vₜ kₜᵀ` is constant regardless of how wrong the prediction is — the no-error-correction
behavior that drove the cross-talk and, I now believe, GLA's LAMBADA regression. The quadratic loss
gives gradients that scale with the error, so the rule self-corrects, and the delta-rule fast weight has
been known for decades to have higher capacity than the Hebbian one. The scalar `βₜ = σ(W_β xₜ) ∈ (0,1)`
is a learned *writing strength*: at `βₜ = 1` the old value is fully overwritten, at `βₜ = 0` the memory
is untouched. This is the write rule I want.

So why has nobody just trained this at scale, and what is the catch I have to pay? The training wall.
For additive linear attention — and for GLA — the value written at step `t` is just `vₜ` (or
`vₜ`-after-decay), independent of the running state, so the whole output is one big masked matmul
`O = (Q Kᵀ ⊙ M) V` over precomputed `V`; that is what made it trainable, matmul-rich, tensor-core-bound.
The delta rule breaks that: the value written, `vₜ^new`, is tangled up with `vₜ^old = S_{t−1} kₜ`, which
depends on the *running state*. I cannot stack the writes into a matrix ahead of time, because each one
needs `S_{t−1}`, the output of all the previous writes. The naive computation rolls the recurrence
forward, materializing the `d_k×d_v` state at every step — and let me size that too, because it is the
reason this rule stayed off the ladder: `64×64 = 4096` floats per head per step, times 16 heads, times
1024 steps, times a micro-batch of 32, is `2.1×10⁹` floats — about 8.6 GB in fp32 for the per-step
states of a *single* layer, before I have touched the other 23. Strictly sequential, elementwise, never
touching a tensor core, and memory-infeasible if materialized. That is exactly the hardware-inefficiency
that kept the delta-rule layer off the scaling ladder. So I get the better write rule only if I can break
the state-dependence and recover a matmul form.

Look at the update as a transition on `S`. Substitute the blend into the swap:
`Sₜ = S_{t−1}(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ`. The delta update is the previous state *multiplied* by an
identity-plus-rank-one matrix — a generalized Householder transformation — plus a rank-one additive
term. That structure is the lever. Two moves break the sequential dependence. First, keep `S` in the
same additive shape as vanilla linear attention so I can reuse all its machinery: claim
`Sₜ = Σ_{i≤t} uᵢ kᵢᵀ` for "pseudo-values" `uᵢ`. Induct: `S₁ = β₁ v₁ k₁ᵀ` so `u₁ = β₁ v₁`; the inductive
step expands `Sₜ = S_{t−1}(I − βₜ kₜ kₜᵀ) + βₜ vₜ kₜᵀ` and collects the `kₜᵀ` term, giving
`uₜ = βₜ(vₜ − Σ_{i<t} uᵢ (kᵢᵀkₜ))`, which sanity-checks against the value-blend interpretation
(`uₜ = vₜ^new − vₜ^old`). So the delta layer is vanilla linear attention with `vᵢ` replaced by the
pseudo-value `uᵢ`, and the per-token matrix state never has to be materialized — the problem reduces to
*computing the `uᵢ`*. Let me verify the reduction on two steps rather than trust the induction blindly.
Directly: `S₁ = β₁ v₁ k₁ᵀ`, and `S₂ = S₁(I − β₂ k₂ k₂ᵀ) + β₂ v₂ k₂ᵀ = β₁ v₁ k₁ᵀ − β₁β₂ v₁(k₁·k₂) k₂ᵀ +
β₂ v₂ k₂ᵀ = β₁ v₁ k₁ᵀ + [β₂ v₂ − β₁β₂(k₁·k₂) v₁] k₂ᵀ`. So the coefficient of `k₁ᵀ` is `u₁ = β₁ v₁` and
the coefficient of `k₂ᵀ` is `u₂ = β₂ v₂ − β₁β₂(k₁·k₂) v₁ = β₂(v₂ − (k₁·k₂) u₁)`, which is exactly
`uₜ = βₜ(vₜ − Σ_{i<t} uᵢ(kᵢᵀkₜ))` at `t = 2`. The pseudo-value recurrence reproduces the delta update, and
`S₂ = u₁ k₁ᵀ + u₂ k₂ᵀ` is back in the plain additive shape — the state is a sum of outer products again,
just with error-corrected values, so every downstream matmul from vanilla linear attention applies. Second, the chunkwise form also needs the product of transition matrices
`Pₙ = ∏_{t≤n}(I − βₜ kₜ kₜᵀ)`, and the same induction gives the WY representation
`Pₙ = I − Σ_{t≤n} wₜ kₜᵀ` with `wₜ = βₜ(kₜ − Σ_{i<n} wᵢ(kᵢᵀkₙ))` — the *exact same recurrence* as `uₜ`
with `kₜ` in place of `vₜ`. Both `uₜ` and `wₜ` are still sequential within a chunk, though, so I have
moved the bottleneck, not removed it.

Remove it: read the recurrence as a linear system, because it is one. The rows of `W` (the stacked
`wₜ`) satisfy a strictly-lower-triangular dependency, `(I + L) W = B K` with `B = diag(β)` and
`L = tril(diag(β) K Kᵀ, −1)`, so `W = (I + L)⁻¹ B K = T K` and identically `U = T V`, with
`T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)` — the *same* `T` for both. The sequential dependency is
absorbed into one matrix `T`, and `I + L` is unit lower-triangular so its inverse is cheap and
matmul-friendly via forward substitution (the UT transform for accumulating Householder products). Now
every part is a matmul — `T` by forward substitution, `W = T K`, `U = T V`, the masked intra-chunk
products, the chunk-state update — with only `L/C` sequential steps between chunks. Cost `O(LCd + Ld²)`,
recomputing chunk states in the backward to save memory, exactly the same asymptotics and the same
hardware profile as the chunkwise GLA I ran at rung two. The delta rule, which looked irreducibly
sequential and memory-infeasible, is now matmul-rich and materializes only `L/C` chunk states instead of
`L` per-step states. That is the whole reason it can sit on *this* ladder at all.

Stability has to be derived, not assumed, because the transition `Mₜ = I − βₜ kₜ kₜᵀ` will blow up or
vanish if its eigenvalues leave the unit disk, so I compute them. `Mₜ` acts as the identity on the
`d_k−1`-dimensional subspace orthogonal to `kₜ` (for `u ⊥ kₜ`, `Mₜ u = u − βₜ kₜ(kₜ·u) = u`, eigenvalue
1) and scales the `kₜ` direction by `Mₜ kₜ = kₜ − βₜ kₜ‖kₜ‖² = (1 − βₜ‖kₜ‖²) kₜ`. So the one non-unit
eigenvalue is `1 − βₜ‖kₜ‖²`, and stability wants `0 ≤ βₜ‖kₜ‖² ≤ 2`. If `‖kₜ‖` is left unbounded a large
key can push `βₜ‖kₜ‖² > 2`, the eigenvalue drops below `−1`, and the recurrence diverges. L2-normalizing
the keys fixes it exactly: with `‖kₜ‖₂ = 1`, the contractive eigenvalue is `1 − βₜ ∈ [0,1]` for
`βₜ ∈ (0,1)`, always stable. And at the boundary `βₜ = 1`, `Mₜ = I − kₜ kₜᵀ` with unit `kₜ` has that
eigenvalue equal to 0 — it is an orthogonal *projection*: it annihilates exactly the one-dimensional
subspace spanned by `kₜ` and leaves the other `d_k−1` dimensions untouched. That is targeted forgetting
made literal — a full write erases exactly the direction being overwritten and preserves everything
else, the content-addressed deallocation the diagonal gate could not localize and that LAMBADA was
punishing GLA for missing. So L2 normalization is not a hack; it is what makes the erase surgical. I
L2-normalize `q` and `k`, apply SiLU before the normalization (keeps sign, smooth, no hard zeroing), use
`βₜ = σ(W_β xₜ)` (one sigmoid scalar per head, negligible parameters), and add a lightweight depthwise
**short convolution** (kernel 4) on the `q`, `k`, `v` projections before the recurrence. It is worth
saying why this earns its place rather than being decoration. The delta rule is pure content-addressing:
it retrieves by matching the current key against stored keys, and that is exactly the operation that is
*blind to adjacency* — "the token immediately before the current one," "the two-word phrase I just saw"
are relations of position, not of content, and a dot-product against the state cannot express them
without the positions having been baked into the representation. A depthwise convolution with kernel 4
mixes each channel over the last four tokens before the `q`/`k`/`v` projections, so the key the layer
files under, and the query it probes with, already carry a little local n-gram context — the layer can
compare short spans, not just single tokens. It generalizes the shift operator this way, costs only
`4·d` parameters per projection (a rounding error against `4d²`), and does the local work that pure
content-addressing is structurally bad at. An output RMSNorm per head before the projection rounds it
out.

Make it concrete in this task's edit surface. FLA ships `DeltaNet` with the chunk kernel that implements
the UT-transform training path, so the edit imports it into `CausalSelfAttention`: `hidden_size = 1024`,
`num_heads = 16`, `use_beta = True` (the learned writing strength `βₜ`, the heart of the rule),
`use_short_conv = True`, `conv_size = 4` (the local-mixing short conv), `qk_activation = 'silu'` and
`qk_norm = 'l2'` (the SiLU-then-L2 that makes the transition an exact projection at `βₜ = 1`). I take
the default `expand_k = 1.0`, `expand_v = 1.0` — symmetric width, state `d×d`, and the parameter budget
is worth checking: `q,k,v` at `d²` each is `3d²`, `o_proj` is `d²`, the short conv is `4·d` per
projection (a few thousand params, negligible) and the `βₜ` projection is `d·16 ≈ 16k`, so the layer is
`~4d²`, matched to softmax and to the RetNet rung. And the default `use_gate = False`: DeltaNet does
*not* add a swish output gate here, just the per-head output RMSNorm, because the error-correcting write
is already the expressivity I am buying and I want this rung to isolate the *write rule* change, not
confound it with the output gate that GLA and RetNet carried — which also means this mixer is a touch
*leaner* than GLA's, since it skips the gate's `d²` projection. (That is a real difference from those
rungs worth naming: no output gate.) I set `self.use_pos_emb = False` — DeltaNet handles sequence
ordering through the recurrence and the short conv, so the loop skips its learned `wpe`. The `Block`
stays the scaffold default; only the mixer is swapped. (The literal scaffold edit is in the answer.)

So the delta from rung two is precise: where GLA fixed the *decay* (made forgetting data-dependent and
diagonal) but left the additive Hebbian *write* — the thing that produced the four-to-one cross-talk and
the LAMBADA regression — this rung fixes the write itself, replacing the additive stamp with an
error-correcting, content-addressed swap, and recovers a matmul training form via the pseudo-value
reduction and the UT transform. Reading GLA's shape, here is the falsifiable claim. The validation loss
should drop below 2.4482, because higher memory capacity and cleaner retrieval help the bulk objective
too. But the *real* test, the one this rung is for, is the recall metrics that GLA stalled or regressed
on: **LAMBADA must come down hard** — it was 84.73 under GLA and 82.36 under RetNet, and if the
error-correcting write does what I derived, it should fall well below both, because LAMBADA is exactly
"retrieve the entity named earlier" and the projection-style erase is exactly clean retrieval. WikiText-2
should also tighten below 64.32, and the downstream suite — HellaSwag, which was stuck at 31.1 across
both prior rungs, only six points over its four-way chance floor, and ARC-Easy — should rise, since both
reward holding specific earlier content. If LAMBADA does *not* improve over GLA, then the recall
diagnosis was wrong and the failure lives somewhere I have not named; but the whole construction predicts
it will, and that this rung will be the strongest of the three precisely on the recall axis the diagonal
gate could never address.
