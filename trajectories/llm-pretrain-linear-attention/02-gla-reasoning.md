RetNet's numbers told me exactly what is missing, and they told me in the place I predicted. The
validation loss came in at 2.4795 — a real language model, trained stably, clearly not a degenerate
collapse (a uniform predictor over the 50257-token vocabulary would sit at `ln 50257 ≈ 10.82` nats, and
2.4795 is a validation perplexity of `exp(2.4795) ≈ 11.9`, squarely in the competent-GPT-2 band). So the
floor was a genuine contender and not a strawman. But it is *high*, and I want to read the downstream
suite as margins over chance rather than as bare percentages, because that is what tells me whether the
model actually holds context. HellaSwag is a four-way task, so chance is 25; at 31.12 the model clears
chance by only 6 points, and a four-way sentence-completion that barely beats a coin over four options is
not holding the mid-range context the task needs. WinoGrande is binary, chance 50; at 52.01 it is two
points above a coin flip — essentially reading nothing from the coreference structure. PIQA (binary,
62.40) clears chance by a healthier 12, and ARC-Easy (four-way, 51.47) by 26, which says the model does
carry real world-knowledge; but the two tasks that most reward *tracking a specific earlier token* are
exactly the two closest to chance. The perplexities echo it: WikiText-2 66.67 and LAMBADA 82.36 are both
loose. This is the failure I expected from the construction, and it is sharp: it is not an optimization
failure — the loss curve was clean and the model is in range — it is a *memory* failure. RetNet forgets
at a rate it chose before it ever saw a token. Every head has a single fixed `γ`, geometrically spaced
across heads, and within a head that decay is the same for every channel, every context, every word.
The half-life arithmetic from the last rung made this concrete — six heads with in-block horizons and
ten heads that never forget within a sequence, a fixed and lopsided partition — and it cannot look at a
token and decide "this is a key fact, hold it" versus "this is filler, let it decay." The lesson from
1-D RNNs is unambiguous: the forget gate carries most of a gated cell's capacity and it has to be
*data-dependent* to do its job, and a fixed scalar `γ` is exactly a data-independent gate. So the
diagnosis points at one move: the decay must become a function of the input. The whole risk, the thing
RetNet's fixed scalar was *protecting*, is that a data-dependent gate normally destroys the matmul
chunkwise training form. That is the wall I have to get through, and getting through it without
re-breaking what made retention trainable is rung two.

Let me restate the operator I am improving, stripped to its skeleton, because the skeleton is what I
must preserve. Plain linear attention is `Sₜ = S_{t−1} + kₜᵀvₜ`, `oₜ = qₜ Sₜ` — a matrix-valued linear
RNN, `O(1)` inference, but it never forgets and that is why it loses. RetNet added the cheapest forget
gate, a global scalar, `Sₜ = γ S_{t−1} + kₜᵀvₜ`, and the 2.4795 says that helped but not enough. The
reason `γ` is a *scalar* is that a scalar pulls out of the cumulative product cleanly, so the whole
attention-style parallel and chunkwise machinery survives — that is the property I cannot lose. So the
question is precise: what is the *most* expressive gate I can use that still keeps the chunkwise matmul
form alive?

The moment I say "data-dependent gate" a tension bites. A classic RNN forget gate depends on both the
previous hidden state and the current input, and the dependence on the previous state is exactly what
serializes the recurrence and kills parallel training. The way out is to make the gate depend *only on
the current input*, not on the previous state. Then the recurrence stays linear in the state — the gate
is a sequence of input-determined coefficients — and a linear recurrence with input-determined
coefficients is still parallelizable, and crucially it still has the cumulative-product structure I can
exploit. Input-only gating is known to work at moderate scale, so the constraint is fixed: the gate is a
function of `xₜ` alone.

What *shape* should the gate be? This is the real design decision, and it is a three-way trade between
parameter efficiency, state expressivity, and training efficiency — so let me not argue it in the
abstract, let me count parameters, because the counts settle it. The most general gate is a full matrix
`Gₜ ∈ (0,1)^{d_k×d_v}` applied Hadamard to the state, `Sₜ = Gₜ ⊙ S_{t−1} + kₜᵀvₜ` — every entry forgets
at its own data-dependent rate, maximally expressive — but mapping `xₜ → Gₜ` needs a projection of size
`d·d_k·d_v`. With the widths I will end up wanting (`d_k = d/2 = 512`, `d_v = d = 1024`) that is
`1024·512·1024 ≈ 537M` parameters *per layer*, larger than the entire 355M model. Dead on arrival. The
other extreme is RetNet's scalar, the rung I am leaving behind: cheap, trivially parallel, but one number
for the whole state, and 2.4795 is what that costs. There is a middle, a low-rank outer-product gate
`Gₜ = αₜᵀβₜ`, costing `d·d_k + d·d_v = 1024·512 + 1024·1024 ≈ 1.57M` per layer — a genuine 2-D gate with
per-key-channel and per-value-channel recency control, four hundred times smaller than the full matrix.
So why not just take that? Because of training, and this is the axis parameter-counting hides. The prior
matrix-gate algorithms, to handle a data-dependent matrix gate, materialize the full matrix-valued
hidden state for *every* time step in slow memory — `L·d_k·d_v` written to HBM — and the gated update
they implement does not reduce to tensor-core matmuls. So it is slow at scale, both from the I/O and
from leaving the tensor cores idle. There is a parallel cautionary tale in a fully input-dependent,
full-rank selective state transition: it is even more expressive, but *precisely because* the full-rank
selective update cannot be written as a matrix multiply, it cannot touch tensor cores and must
materialize per-step states, capping the state expansion (around 16) to keep them in SRAM — which shows
up as weak recall. The pattern is the one I have to break: expressivity that destroys the
matmul/chunkwise structure buys a slow, memory-bound trainer. I need a gate expressive enough to fix
RetNet's fixed-decay failure but structured enough that the chunkwise matmul form survives.

So let me try the outer-product gate `Gₜ = αₜᵀβₜ` and actually *check* whether the chunkwise structure
survives, rather than assume. Unroll the recurrence. With `Gₜ = αₜᵀβₜ`, the state is
`Sₜ = Σ_{i≤t} ((∏_{j=i+1}^t G_j) ⊙ kᵢᵀvᵢ)`. The cumulative product `∏ G_j` is the thing I am afraid of
— but it is an outer product of cumulative products: define `bₜ = ∏_{j≤t} α_j` and `dₜ = ∏_{j≤t} β_j`
(elementwise products of the gate vectors). Then `∏_{j=i+1}^t α_jᵀβ_j = (bₜ/bᵢ)ᵀ(dₜ/dᵢ)` — the
telescoping makes the product-of-outer-products collapse into a single outer product of ratios, by the
mixed-product property. So `(∏ G_j) ⊙ (kᵢᵀvᵢ) = ((bₜ/bᵢ) ⊙ kᵢ)ᵀ((dₜ/dᵢ) ⊙ vᵢ)`, still one outer
product. Plug into the output and use `⟨a, b⊙c⟩ = ⟨a⊙b, c⟩`, and the whole thing is exactly a
linear-attention parallel form on *preconditioned* tensors: `Q̃ = Q ⊙ B`, `K̃ = K / B`, `Ṽ = V / D`,
compute `Õ = ((Q̃ K̃ᵀ) ⊙ M) Ṽ`, read off `O = Õ ⊙ D`. The data-dependent matrix gate did **not** destroy
the matmul structure — it just preconditions Q, K, V by cumulative-product factors. That is the wall
broken: an outer-product gate keeps the chunkwise matmul form alive where a full-rank transition does
not.

I do not want to trust that on the algebra alone, so I trace two steps with the value side dropped
(`β = 1`, which I am about to argue for anyway). Take `S₀ = 0`, `b₁ = α₁`, `b₂ = α₁⊙α₂`. From the
recurrence, `S₂ = Diag(α₂) S₁ + k₂ᵀv₂ = Diag(α₂) k₁ᵀv₁ + k₂ᵀv₂`, so the coefficient of `v₁` in
`o₂ = q₂ S₂` is `q₂ Diag(α₂) k₁ᵀ = Σ_c q₂ᶜ α₂ᶜ k₁ᶜ`. Now the preconditioned parallel form: the row-2
query is `q̃₂ = q₂ ⊙ b₂ = q₂ ⊙ (α₁⊙α₂)`, the key-1 vector is `k̃₁ = k₁ / b₁ = k₁ / α₁`, and their score
is `Σ_c q₂ᶜ α₁ᶜ α₂ᶜ k₁ᶜ / α₁ᶜ = Σ_c q₂ᶜ α₂ᶜ k₁ᶜ`. Identical — and I can see *why* the `α₁` cancels:
both the query at position 2 and the key at position 1 are measured against the same cumulative-product
origin, so only the decay *between* them, `α₂`, survives. The preconditioning is doing exactly the
relative-decay bookkeeping the recurrence does, one matmul at a time.

Do I actually need `β` too? The `dₜ`/`1/dᵢ` on the value side are extra cumulative products, extra
log-space bookkeeping, extra failure modes. Fix `β = 1` and check what I lose. Then `Gₜ = αₜᵀ1` — every
row of the state shares the same per-key-channel decay `αₜ`, broadcast across the value dimension — and
the value-side transforms vanish (`Ṽ = V`, `O = Õ`). The model is `Sₜ = Diag(αₜ) S_{t−1} + kₜᵀvₜ`, a
**per-key-channel data-dependent forget gate** — exactly the fine-grained, content-chosen forgetting
that RetNet's single `γ` could not express — at half the gating machinery. The value-side gate would
add a second independent decay axis, but that is the expensive part of the bookkeeping and it overlaps
with the value projection and the per-head output gate I will add anyway. So I take the middle ground:
`Gₜ = αₜᵀ1`. Strictly more expressive than RetNet's scalar (per-channel *and* data-dependent), far
cheaper than the full matrix, keeps the chunkwise form, avoids the value-side cumulative products that
make stability harder.

And it is worth being explicit that keeping the *key* axis rather than the value axis is not arbitrary
once `β = 1` forces a choice of which single axis to gate. The state is `Sₜ = Σ (decay) kᵢᵀvᵢ`, and the
readout `qₜ Sₜ` contracts `qₜ` against the *key* side of each stored outer product — the key is the index
under which a value is filed and the query is what probes it. So a per-key-channel decay `Diag(αₜ)`
acting from the left is a decay on the *indexing directions*: it controls how fast each retrieval axis
fades, which is exactly the knob that decides whether a fact stored under some key is still findable.
A per-value-channel decay would instead fade the stored *contents* uniformly across all keys, a blunter
and less retrieval-shaped control. Given one axis, the key axis is the one that matches what the layer is
for, so `Gₜ = αₜᵀ1` is the right half to keep.

Now the numerics, because the clean parallel form is dead on arrival and I should see exactly how it
dies. The recurrent form is `Sₜ = Diag(αₜ) S_{t−1} + kₜᵀvₜ`, `oₜ = qₜ Sₜ`, with `αₜ ∈ (0,1)^{d_k}` from
`xₜ`, and the parallel form uses `bₜ = ∏_{j≤t} α_j`: `P = (Q ⊙ B)(K / B)ᵀ`, `O = (P ⊙ M) V`. Each
`α_j < 1`, so `bₜ` is a cumulative product of sub-one numbers. Put numbers on it. A fresh sigmoid gate
sits at `α ≈ 0.5`, so `bₜ ≈ 0.5^t = 2^{−t}`; fp32's smallest positive normal is `≈ 2^{−126}`, so `bₜ`
underflows to exactly zero around position 126 — and then `K / B` is a division by zero, `inf`, for the
back three-quarters of a 1024-token block. In fp16, where the max representable is `65504 ≈ 2^{16}`, it
is worse the other way: `K / bₜ = K · 2^{t}` overflows by roughly position 16. Either precision blows up
within a fraction of the sequence. The fix is to compute the scores in log space: `P_{ij} = Σ_k Q_{ik}
K_{jk} exp(log B_{ik} − log B_{jk})` for `i ≥ j`, where the cumulative product becomes a cumulative
*sum* of log-gates `log bₜ = Σ_{j≤t} log α_j` — a sum of negative numbers, no underflow — and
`log B_{ik} − log B_{jk}` is the accumulated log-decay between key `j` and query `i`, i.e. a
data-dependent relative-position factor (a learned, content-dependent ALiBi, the spiritual successor to
RetNet's fixed `γ^{n−m}`). Stable — but `exp` of a difference that depends on both `i` and `k` sits
between `Q` and `K`, so the score is *not* a single matmul, and I have thrown away the tensor cores. The
escape is that the underflow is a *long-range* phenomenon: within a chunk of length `C`, the cumulative
product runs over at most `C` steps and stays bounded. So redo the chunking with the gate, measuring
cumulative gates *relative to chunk boundaries* — `Λ` the decay from chunk start, `Γ` the decay to chunk
end, `γ_chunk` the whole-chunk decay — so every cumulative product spans at most one chunk. Concretely,
split each query position into "inside its own chunk" and "from all earlier chunks." The inside-chunk
contribution is the masked parallel form with the local cumulative gate `Λ` running from the chunk's own
start, bounded over at most `C` steps; the cross-chunk contribution reads the state `S` carried in at the
boundary and weights it by the query's decay-from-chunk-start `Λ`. The carried state updates as
`S_new = γ_chunk ⊙ S_old + (Γ ⊙ K)ᵀ V`, where `Γ` decays each in-chunk key forward to the chunk's end and
`γ_chunk` is the whole-chunk product that decays the incoming state across the chunk. So the only
cumulative products that ever appear are `Λ`, `Γ`, `γ_chunk`, each spanning a single chunk of length `C`
— with `C` on the order of 64 the log-magnitudes stay near `64 · (−0.043) ≈ −2.8`, comfortably inside
fp16 range — and the true long-range decay is reconstructed exactly by composing the per-chunk `γ_chunk`
factors through the inter-chunk recurrence. The intra-chunk output is thus the same masked parallel form
with chunk-local preconditioning. A second level of tiling pushes the full-precision log-space work down
to the small diagonal sub-blocks while
every off-diagonal sub-block and the inter-chunk recurrence run as half-precision matmuls. The same
FlashLinearAttention I/O tricks — materialization for sequence-level parallelism at small batch,
recomputation in the backward to claw back memory, and a closed-form log-space gate gradient `d log αₜ`
(no per-step states) — make it fast in wall-clock, not just FLOPs.

The remaining design choices each have to earn their place. The gate `αₜ ∈ (0,1)^{d_k}` is
data-dependent on `xₜ`, so a linear map plus sigmoid is natural — but a full `d×d_k` gate projection
adds `1024·512 ≈ 0.5M` real parameters, and I am trying to hold the same `~4d²` budget as softmax (and
as the RetNet rung, so the comparison stays clean). Make it **low-rank**: `xₜ → W_α¹ → W_α² → sigmoid`
with a rank-16 bottleneck, `d·16 + 16·d_k = 16384 + 8192 ≈ 25k` parameters — nearly free, and more than
enough to choose a per-channel forget rate from content. And a subtlety that matters and that the
underflow arithmetic already flagged: a fresh sigmoid gate sits near 0.5, meaning the state *halves*
every step — a half-life of one token, far too aggressive, the model cannot hold anything for more than
a few tokens, long-range capacity dead before training starts. Bias it toward 1 with a temperature:
`αₜ = σ(logits)^{1/τ}` with `τ = 16`. Since `σ < 1`, the `1/16` power pushes toward 1: at init
`0.5^{1/16} = 2^{−1/16} ≈ 0.9576`, a half-life of `0.693/(1−0.9576) ≈ 16` tokens instead of one, so slow
forgetting is the default and the model must actively decide to forget. And in log space this is
`log αₜ = (1/16) logsigmoid(logits)`, at init `≈ −0.043` per step, so `log b_{1024} ≈ −44` — a small,
well-conditioned cumulative-sum quantity, exactly what the stable form wants, instead of the catastrophic
`log 2^{−1024}` of the raw gate. The temperature does double duty: long-memory prior *and* a small
log-decay. I scale queries by `1/√d_k` (the standard attention logit scaling). And the dimensions: the
FLA layer exposes `expand_k` and `expand_v`, and here I depart from the symmetric width I used for RetNet
on purpose. The state is `d_k×d_v` and its size is memory capacity, so I want the value full-width,
`d_v = d` (`expand_v = 1.0`), but I do not need the key as large, so `d_k = d/2` (`expand_k = 0.5`).
Count the budget this lands at: `q,k` are `d·d_k = 0.5d²` each, `v` and the output gate `g` and `o` are
`d·d_v = d²` each, plus the `25k` low-rank gate — total `0.5 + 0.5 + 1 + 1 + 1 ≈ 4d²`. Halving `d_k`
buys back the `d²` the swish output gate spends, so the layer lands right back at the softmax `~4d²`,
with a `512×1024` state (half RetNet's key dimension, full value width). Multi-head split, per-head
RMSNorm on each head's output (a linear-attention layer has no softmax normalizing the heads to a
simplex, so they need taming — the same 8×-scale-spread argument as RetNet), then a Swish output gate
`r = swish(xW_r)` applied multiplicatively before the projection — the same gated-nonlinearity recipe I
used for RetNet, which matters for closing the gap.

Make it concrete in this task's edit surface. FLA ships `GatedLinearAttention` with the chunk kernel, so
the edit imports it into `CausalSelfAttention`: `hidden_size = 1024`, `num_heads = 16`, `mode = 'chunk'`,
`expand_k = 0.5`, `expand_v = 1.0`, `use_output_gate = True`, `gate_fn = 'swish'`. The log-space
low-rank gate (`logsigmoid(low-rank logits) / gate_logit_normalizer`), the rank-16 bottleneck, the
per-head RMSNorm, and the secondary-tiling chunk kernel are all internal to the FLA layer — I am not
re-deriving them in code, only selecting the configuration. One harness-specific detail: `torch.compile`
is disabled for this task, and to be safe against any residual compile interaction with the Triton
kernel I wrap the FLA call in a `@torch.compiler.disable`-decorated helper so the chunk kernel is never
traced. I set `self.use_pos_emb = False` for the same reason as RetNet — the cumulative data-dependent
decay *is* the relative position signal, so the loop must skip its learned `wpe`. The `Block` stays
exactly the scaffold default, only the mixer swapped, so the only difference from the RetNet rung is the
gate. (The literal scaffold edit is in the answer.)

So the delta from rung one is precise: where RetNet's decay was a single fixed scalar per head — the
data-independent gate that cost it 2.4795 — I now make the decay a per-key-channel function of the
content, while keeping the same chunkwise matmul training form via the outer-product telescoping and
log-space stability. Reading RetNet's shape, here is what I expect and where I am unsure. The validation
loss should drop below 2.4795: the model can now hold the *right* facts at the *right* rate instead of an
exponential average at a fixed rate, and that is exactly what cross-entropy on natural text rewards. The
perplexities should tighten — WikiText-2 66.67 and LAMBADA 82.36 were the loosest under RetNet, and
content-chosen forgetting is most valuable precisely where a fixed rate threw away the salient token. The
downstream accuracies should rise, HellaSwag especially, since 31.12 (six points over chance) read as
"not holding mid-range context" and a data-dependent gate is the cure for that. But I hold one doubt
that the next rung may have to answer: this gate is still a *diagonal, multiplicative* decay — it forgets
globally or per-channel, it does not remove the *specific stored association* that a new key collides
with. If the residual failure after this rung is recall — fetching a particular value I stored earlier
keyed on a token I have now seen again — then the gate, however data-dependent, is structurally the wrong
tool, and the diagnosis would point at changing the *write rule* itself rather than the decay. LAMBADA is
the metric that would expose it, since it is precisely "retrieve the entity named earlier," and I would
not be shocked to see the bulk metrics improve while LAMBADA fails to follow. For now, the falsifiable
claim is simple: this gate should beat RetNet on val_loss, on both perplexities, and on the downstream
suite, and if it does not beat it on perplexity then the data-dependence is not buying what the RNN
literature promised.
