RetNet's numbers told me what is missing, and in the place I predicted. Validation loss came in at 2.4795
— a real language model, trained stably, no degenerate collapse (a uniform predictor over the 50257-token
vocabulary would sit at `ln 50257 ≈ 10.82` nats, and 2.4795 is a perplexity of `exp(2.4795) ≈ 11.9`,
squarely in the competent-GPT-2 band). So the floor was a genuine contender. But it is *high*, and I want
to read the downstream suite as margins over chance rather than bare percentages, because that tells me
whether the model actually holds context. HellaSwag is four-way, chance 25; at 31.12 it clears chance by
only 6 points, and a four-way completion that barely beats a coin over four options is not holding the
mid-range context the task needs. WinoGrande is binary, chance 50; at 52.01 it is two points above a coin,
essentially reading nothing from the coreference structure. PIQA (binary, 62.40) clears chance by a
healthier 12 and ARC-Easy (four-way, 51.47) by 26, so the model does carry real world-knowledge — but the
two tasks that most reward *tracking a specific earlier token* are exactly the two closest to chance, and
the perplexities echo it (WikiText-2 66.67, LAMBADA 82.36, both loose). This is the failure I expected from
the construction, and it is sharp: not an optimization failure — the loss is in range — but a *memory*
failure. RetNet forgets at a rate it chose before it ever saw a token: every head a single fixed `γ`, the
same decay for every channel, context, and word, six heads with in-block horizons and ten that never
forget within a sequence. It cannot look at a token and decide "key fact, hold it" versus "filler, let it
decay." The 1-D RNN lesson is unambiguous — the forget gate carries most of a gated cell's capacity and it
has to be *data-dependent* — and a fixed scalar `γ` is exactly a data-independent gate. So the diagnosis
points at one move: the decay must become a function of the input. And the whole risk, the thing RetNet's
fixed scalar was *protecting*, is that a data-dependent gate normally destroys the matmul chunkwise
training form. That is the wall I have to get through without re-breaking what made retention trainable.

Restate the operator, stripped to the skeleton I must preserve. Plain linear attention is
`Sₜ = S_{t−1} + kₜᵀvₜ`, `oₜ = qₜ Sₜ` — a matrix-valued linear RNN that never forgets, which is why it
loses. RetNet added the cheapest forget gate, a global scalar `Sₜ = γ S_{t−1} + kₜᵀvₜ`, and 2.4795 says
that helped but not enough. `γ` is a *scalar* precisely so it pulls out of the cumulative product cleanly
and the parallel/chunkwise machinery survives — the property I cannot lose. So the question is precise:
what is the *most* expressive gate that still keeps the chunkwise matmul form alive?

The moment I say "data-dependent gate" a tension bites. A classic RNN forget gate depends on the previous
hidden state, and that dependence is what serializes the recurrence and kills parallel training. The way
out is to make the gate depend *only on the current input*, not the previous state. Then the recurrence
stays linear in the state — the gate is a sequence of input-determined coefficients — still
parallelizable, and still carrying the cumulative-product structure I can exploit. Input-only gating is
known to work at moderate scale, so the constraint is fixed: the gate is a function of `xₜ` alone.

What *shape* should the gate be? This is the real decision, a three-way trade between parameter
efficiency, state expressivity, and training efficiency, and the parameter counts settle it. The most
general gate is a full matrix `Gₜ ∈ (0,1)^{d_k×d_v}` applied Hadamard, `Sₜ = Gₜ ⊙ S_{t−1} + kₜᵀvₜ` —
maximally expressive — but `xₜ → Gₜ` needs a projection of size `d·d_k·d_v`. With the widths I will end up
wanting (`d_k = 512`, `d_v = 1024`) that is `1024·512·1024 ≈ 537M` per *layer*, larger than the whole
355M model. Dead on arrival. The other extreme is RetNet's scalar: one number for the whole state, and
2.4795 is what it costs. The middle is a low-rank outer-product gate `Gₜ = αₜᵀβₜ`, costing
`d·d_k + d·d_v ≈ 1.57M` per layer — a genuine 2-D gate with per-key-channel and per-value-channel recency
control, four hundred times smaller than the full matrix. Why not just take it? Because of training, the
axis parameter-counting hides. Prior matrix-gate algorithms materialize the full matrix-valued state for
*every* step in slow memory — `L·d_k·d_v` to HBM — and the gated update does not reduce to tensor-core
matmuls, so it is slow both from I/O and from leaving the tensor cores idle. A parallel cautionary tale is
the fully input-dependent full-rank selective transition: more expressive still, but *precisely because*
it cannot be written as a matmul it must materialize per-step states, capping the state expansion (~16) to
keep them in SRAM — which shows up as weak recall. The pattern I have to break is expressivity that
destroys the matmul/chunkwise structure and buys a slow, memory-bound trainer. I need a gate expressive
enough to fix the fixed-decay failure but structured enough that the chunkwise matmul form survives.

So try the outer-product gate `Gₜ = αₜᵀβₜ` and *check* whether the chunkwise structure survives. Unroll:
`Sₜ = Σ_{i≤t} ((∏_{j=i+1}^t G_j) ⊙ kᵢᵀvᵢ)`. The cumulative product is the thing I fear — but it is an
outer product of cumulative products. Define `bₜ = ∏_{j≤t} α_j`, `dₜ = ∏_{j≤t} β_j`; then by the
mixed-product property `∏_{j=i+1}^t α_jᵀβ_j = (bₜ/bᵢ)ᵀ(dₜ/dᵢ)`, so the product-of-outer-products collapses
into a single outer product of ratios, and `(∏ G_j) ⊙ (kᵢᵀvᵢ) = ((bₜ/bᵢ) ⊙ kᵢ)ᵀ((dₜ/dᵢ) ⊙ vᵢ)` — still
one outer product. Pushing it through the readout with `⟨a, b⊙c⟩ = ⟨a⊙b, c⟩`, the whole thing is exactly a
linear-attention parallel form on *preconditioned* tensors: `Q̃ = Q ⊙ B`, `K̃ = K / B`, `Ṽ = V / D`,
`Õ = ((Q̃ K̃ᵀ) ⊙ M) Ṽ`, `O = Õ ⊙ D`. The data-dependent matrix gate did **not** destroy the matmul
structure — it just preconditions Q, K, V by cumulative-product factors. That is the wall broken.

Do I actually need `β` too? The value-side `dₜ` are extra cumulative products, extra log-space bookkeeping,
extra failure modes. Fix `β = 1`: then `Gₜ = αₜᵀ1` — every row of the state shares the same per-key-channel
decay `αₜ`, broadcast across the value dimension — the value-side transforms vanish (`Ṽ = V`, `O = Õ`), and
the model is `Sₜ = Diag(αₜ) S_{t−1} + kₜᵀvₜ`, a **per-key-channel data-dependent forget gate**, exactly the
fine-grained content-chosen forgetting RetNet's single `γ` could not express, at half the gating
machinery. The value-side gate would add a second decay axis but it is the expensive part of the
bookkeeping and it overlaps with the value projection and the output gate I will add anyway. And keeping
the *key* axis rather than the value axis is not arbitrary once `β = 1` forces the choice: the readout
`qₜ Sₜ` contracts `qₜ` against the *key* side of each stored outer product `kᵢᵀvᵢ` — the key is the index a
value is filed under. So a per-key-channel decay `Diag(αₜ)` is a decay on the *indexing directions*: it
controls how fast each retrieval axis fades, exactly the knob that decides whether a fact is still
findable. A per-value-channel decay would instead fade the stored contents uniformly across all keys, a
blunter, less retrieval-shaped control. So `Gₜ = αₜᵀ1`.

Now the numerics, because the clean parallel form is dead on arrival and I should see how it dies. With
`bₜ = ∏_{j≤t} α_j` and each `α_j < 1`, `bₜ` is a cumulative product of sub-one numbers. A fresh sigmoid
gate sits at `α ≈ 0.5`, so `bₜ ≈ 2^{−t}`; fp32's smallest positive normal is `≈ 2^{−126}`, so `bₜ`
underflows to zero around position 126 — and then `K / B` is a division by zero, `inf`, for the back
three-quarters of a 1024-token block. In fp16 (max `≈ 2^{16}`) it is worse the other way, `K / bₜ = K·2^t`
overflowing by position 16. The fix is to compute the scores in log space: the cumulative product becomes a
cumulative *sum* of log-gates `log bₜ = Σ_{j≤t} log α_j` (a sum of negatives, no underflow), and
`log B_{ik} − log B_{jk}` is the accumulated log-decay between key `j` and query `i` — a data-dependent
relative-position factor, a learned content-dependent ALiBi, successor to RetNet's fixed `γ^{n−m}`. Stable
— but the `exp` of a difference that depends on both `i` and `k` sits between `Q` and `K`, so the score is
no longer a single matmul; I have thrown away the tensor cores. The escape: underflow is a *long-range*
phenomenon. Within a chunk of length `C` the cumulative product runs over at most `C` steps and stays
bounded, so redo the chunking with the gate, measuring cumulative gates *relative to chunk boundaries*.
Split each query position into "inside its own chunk" (the masked parallel form with a chunk-local
cumulative gate `Λ`, bounded over ≤ `C` steps) and "from all earlier chunks" (read the carried state `S`,
weight it by `Λ`); the state updates as `S_new = γ_chunk ⊙ S_old + (Γ ⊙ K)ᵀ V`, with `Γ` decaying each
in-chunk key to the chunk's end and `γ_chunk` the whole-chunk product. Every cumulative product then spans
at most one chunk — with `C ≈ 64` the log-magnitudes stay near `64 · (−0.043) ≈ −2.8`, comfortably inside
fp16 — and the true long-range decay is reconstructed exactly by composing the per-chunk factors through
the inter-chunk recurrence. A second level of tiling pushes the full-precision log-space work down to the
small diagonal sub-blocks while every off-diagonal block and the inter-chunk recurrence run as
half-precision matmuls, and the FlashLinearAttention I/O tricks (materialization for sequence-parallelism
at small batch, backward recomputation, a closed-form log-space gate gradient with no per-step states) make
it fast in wall-clock, not just FLOPs.

The remaining design choices each earn their place. The gate `αₜ ∈ (0,1)^{d_k}` from `xₜ` wants a linear
map plus sigmoid, but a full `d×d_k` gate projection adds `1024·512 ≈ 0.5M` params and I am holding the
`~4d²` budget (same as RetNet, so the comparison stays clean). Make it **low-rank**: `xₜ → W_α¹ → W_α² →
sigmoid` with a rank-16 bottleneck, `d·16 + 16·d_k ≈ 25k` params — nearly free, and enough to choose a
per-channel forget rate from content. And a subtlety the underflow arithmetic already flagged: a fresh
sigmoid near 0.5 halves the state every step — a half-life of one token, long-range capacity dead before
training. Bias it toward 1 with a temperature, `αₜ = σ(logits)^{1/τ}`, `τ = 16`: since `σ < 1` the `1/16`
power pushes toward 1, at init `0.5^{1/16} ≈ 0.9576`, a half-life of `≈ 16` tokens, so slow forgetting is
the default and the model must actively decide to forget. In log space `log αₜ = (1/16) logsigmoid(logits)
≈ −0.043` per step at init, so `log b_{1024} ≈ −44` — a small, well-conditioned cumulative sum, exactly
what the stable form wants. I scale queries by `1/√d_k`. And the dimensions: here I depart from RetNet's
symmetric width on purpose. The state is `d_k×d_v` and its size is memory capacity, so I want the value
full-width `d_v = d` (`expand_v = 1.0`) but do not need the key as large, `d_k = d/2` (`expand_k = 0.5`).
Budget: `q,k` at `0.5d²` each, `v` and the output gate `g` and `o` at `d²` each, plus the `25k` gate —
`0.5 + 0.5 + 1 + 1 + 1 ≈ 4d²`. Halving `d_k` buys back the `d²` the swish output gate spends, so the layer
lands right back at softmax's `~4d²` with a `512×1024` state. Multi-head split, per-head RMSNorm (a
linear-attention layer has no softmax normalizing the heads, the same 8×-scale-spread taming argument as
RetNet), then a Swish output gate `r = swish(xW_r)` applied before the projection — the same
gated-nonlinearity recipe RetNet used, which matters for closing the gap.

Concrete in the edit surface: FLA ships `GatedLinearAttention` with the chunk kernel, so the edit imports
it — `hidden_size = 1024`, `num_heads = 16`, `mode = 'chunk'`, `expand_k = 0.5`, `expand_v = 1.0`,
`use_output_gate = True`, `gate_fn = 'swish'`. The log-space low-rank gate, the rank-16 bottleneck, the
per-head RMSNorm, and the secondary-tiling chunk kernel are internal to the layer — I select the
configuration, not re-derive it in code. One harness detail: `torch.compile` is disabled for this task, and
to be safe against any residual compile interaction with the Triton kernel I wrap the FLA call in a
`@torch.compiler.disable` helper so the chunk kernel is never traced. I set `self.use_pos_emb = False` for
the same reason as RetNet — the cumulative data-dependent decay *is* the relative position signal. The
`Block` stays the scaffold default; the only difference from the RetNet rung is the gate. (The literal edit
is in the answer.)

The delta from rung one: RetNet's decay was a single fixed scalar per head; here the decay becomes a
per-key-channel function of content, kept in the same chunkwise matmul form via the outer-product
telescoping and log-space stability. Reading RetNet's shape, the falsifiable bet: the
validation loss should drop below 2.4795 (the model can hold the *right* facts at the *right* rate instead
of a fixed-rate exponential average, exactly what cross-entropy on natural text rewards), the perplexities
should tighten — WikiText-2 66.67 and LAMBADA 82.36 were the loosest, and content-chosen forgetting is most
valuable where a fixed rate threw away the salient token — and the downstream accuracies should rise,
HellaSwag especially, since 31.12 read as "not holding mid-range context." But I hold one doubt that the
next rung may have to answer: this gate is still a *diagonal, multiplicative* decay — it forgets globally
or per-channel, it does not remove the *specific stored association* a new key collides with. If the
residual failure is recall — fetching a particular value stored earlier keyed on a token seen again — then
the gate, however data-dependent, is structurally the wrong tool, and the fix would point at the *write
rule* rather than the decay. LAMBADA, precisely "retrieve the entity named earlier," is where that would
show, and I would not be shocked to see the bulk metrics improve while it fails to follow.
