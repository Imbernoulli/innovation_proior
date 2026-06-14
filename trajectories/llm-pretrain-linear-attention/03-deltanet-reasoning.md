GLA's numbers landed where I bet they would on the bulk metric, and they failed where I flagged the
risk — and the split between the two is the whole story for this rung. The validation loss dropped from
RetNet's 2.4795 to 2.4482, WikiText-2 perplexity tightened from 66.67 to 64.32, and ARC-Easy rose from
51.47 to 53.11. So the data-dependent diagonal gate bought exactly the local language-modeling gain I
predicted: holding the right facts at a content-chosen rate beats an exponential average at a fixed
rate, and the bulk cross-entropy reflects it. But now read the two metrics I explicitly said to watch.
LAMBADA perplexity got *worse* — 84.73 versus RetNet's 82.36 — and WinoGrande *fell*, from 52.01 to
49.88, while HellaSwag sat flat at 31.10 against 31.12. That is not noise around a uniform improvement;
it is a clean dissociation. LAMBADA is the canonical long-range completion task: it gives you a passage
and asks for the final word, which is recoverable only if the model is *retrieving* a specific entity or
fact named earlier, sometimes far back. WinoGrande likewise turns on resolving a reference to the right
antecedent. The metrics that improved are the ones that reward smooth local language modeling; the
metrics that stalled or regressed are the ones that reward *associative recall* — fetch the particular
value you stored earlier, keyed on a token you have now seen again. So the diagnosis is exactly the
doubt I closed the GLA derivation with, now confirmed by numbers: GLA's gate, however data-dependent, is
*diagonal and multiplicative*. It can decide how fast each channel forgets, but it cannot reach into the
state and remove the *specific stored association* that collides with the new key. The failure is not in
the decay anymore. It is in the **write rule**.

Let me look hard at why a multiplicative gate cannot do targeted removal, because that is what tells me
what to replace it with. Both RetNet and GLA, and the whole gated-linear-attention family, share one
template: `Sₜ = S_{t−1} ⊙ Mₜ + kₜᵀvₜ`, a decay times the old state plus an additive outer-product
write. The write is *Hebbian* — token `t` stamps `kₜᵀvₜ` into the state and the decay is the only thing
that ever shrinks anything. Read back what that store holds. Write `S = Σᵢ vᵢ kᵢᵀ` and probe with a
stored key `k_j`: `S k_j = vⱼ (kⱼᵀkⱼ) + Σ_{i≠j} (kᵢᵀk_j) vᵢ`. The first term is what I wanted; the
second is cross-talk from every key not orthogonal to `k_j`. In `d` dimensions there are at most `d`
mutually orthogonal vectors, so the moment a sequence is longer than the key dimension — which, at block
size 1024 and head dimensions in the dozens, is *always* — the keys cannot all be orthogonal, the store
is overcapacity, and retrieval is contaminated by interference that only grows as I write more. The
multiplicative gate scales the *whole* state (per channel) before each write; it cannot say "the
association I previously stored for *this* key now collides with the new key — erase *that one* and
leave everything else." Targeted, content-addressed removal would require the previous state `S_{t−1}`
to enter the *write itself*, interacting with the incoming key — not merely to be scaled by a decay. An
elementwise gate is structurally the wrong tool for the collision problem, and LAMBADA going *backward*
under GLA is that wrongness made visible: GLA's content-chosen decay actually made the interference
pattern worse on the one task that most needs clean retrieval.

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
forward, materializing the `d×d` state at every step — `O(d²)` memory per step, strictly sequential,
elementwise, never touching a tensor core. That is exactly the hardware-inefficiency that kept the
delta-rule layer off the scaling ladder. So I get the better write rule only if I can break the
state-dependence and recover a matmul form.

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
*computing the `uᵢ`*. Second, the chunkwise form also needs the product of transition matrices
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
sequential, is now matmul-rich. That is the whole reason DeltaNet can sit on *this* ladder at all.

Stability has to be derived, not assumed, because the transition `Mₜ = I − βₜ kₜ kₜᵀ` will blow up or
vanish if its eigenvalues leave the unit disk. `Mₜ` is the identity on everything orthogonal to `kₜ`
(eigenvalue 1) and scales the `kₜ` direction by `1 − βₜ‖kₜ‖²`. So I need `0 ≤ βₜ‖kₜ‖² ≤ 2`, which I get
exactly by L2-normalizing the keys: with `‖kₜ‖₂ = 1`, the contractive eigenvalue is `1 − βₜ ∈ [0,1]` for
`βₜ ∈ (0,1)`, always stable. And at the boundary `βₜ = 1`, `Mₜ = I − kₜ kₜᵀ` with unit `kₜ` is an
orthogonal *projection*: it annihilates exactly the one-dimensional subspace spanned by `kₜ` and leaves
the other `d−1` dimensions untouched. That is targeted forgetting made literal — a full write erases
exactly the direction being overwritten and preserves everything else, the content-addressed
deallocation the diagonal gate could not localize and that LAMBADA was punishing GLA for missing. So L2
normalization is not a hack; it is what makes the erase surgical. I L2-normalize `q` and `k`, apply SiLU
before the normalization (keeps sign, smooth, no hard zeroing), use `βₜ = σ(W_β xₜ)` (one sigmoid scalar
per head, negligible parameters), and add a lightweight depthwise **short convolution** (kernel 4) on
the `q`, `k`, `v` projections before the recurrence — it generalizes the shift operator and lets the
layer do precise local token comparisons that pure content-addressing is bad at, cheap and empirically
important. An output RMSNorm per head before the projection rounds it out.

Make it concrete in this task's edit surface. FLA ships `DeltaNet` with the chunk kernel that implements
the UT-transform training path, so the edit imports it into `CausalSelfAttention`: `hidden_size = 1024`,
`num_heads = 16`, `use_beta = True` (the learned writing strength `βₜ`, the heart of the rule),
`use_short_conv = True`, `conv_size = 4` (the local-mixing short conv), `qk_activation = 'silu'` and
`qk_norm = 'l2'` (the SiLU-then-L2 that makes the transition an exact projection at `βₜ = 1`). I take
the default `expand_k = 1.0`, `expand_v = 1.0` — symmetric width, state `d×d`, parameter budget matched
to softmax and to the RetNet rung — and the default `use_gate = False`: DeltaNet does *not* add a swish
output gate here, just the per-head output RMSNorm, because the error-correcting write is already the
expressivity I am buying and I want this rung to isolate the *write rule* change, not confound it with
the output gate that GLA and RetNet carried. (That is a real difference from those rungs worth naming:
no output gate.) I set `self.use_pos_emb = False` — DeltaNet handles sequence ordering through the
recurrence and the short conv, so the loop skips its learned `wpe`. The `Block` stays the scaffold
default; only the mixer is swapped. (The literal scaffold edit is in the answer.)

So the delta from rung two is precise: where GLA fixed the *decay* (made forgetting data-dependent and
diagonal) but left the additive Hebbian *write* — the thing that produced the cross-talk and the LAMBADA
regression — DeltaNet fixes the write itself, replacing the additive stamp with an error-correcting,
content-addressed swap, and recovers a matmul training form via the pseudo-value reduction and the UT
transform. Reading GLA's shape, here is the falsifiable claim. The validation loss should drop below
2.4482, because higher memory capacity and cleaner retrieval help the bulk objective too. But the *real*
test, the one this rung is for, is the recall metrics that GLA stalled or regressed on: **LAMBADA must
come down hard** — it was 84.73 under GLA and 82.36 under RetNet, and if the error-correcting write does
what I derived, it should fall well below both, because LAMBADA is exactly "retrieve the entity named
earlier" and the projection-style erase is exactly clean retrieval. WikiText-2 should also tighten below
64.32, and the downstream suite — HellaSwag, which was stuck at 31.1 across both prior rungs, and
ARC-Easy — should rise, since both reward holding specific earlier content. If LAMBADA does *not*
improve over GLA, then the recall diagnosis was wrong and the failure lives somewhere I have not named;
but the whole construction predicts it will, and that DeltaNet will be the strongest of the three rungs
precisely on the recall axis the diagonal gate could never address.
