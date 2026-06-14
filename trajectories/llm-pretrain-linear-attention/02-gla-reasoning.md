RetNet's numbers told me exactly what is missing, and they told me in the place I predicted. The
validation loss came in at 2.4795 — a real language model, trained stably, clearly not a degenerate
collapse, which confirms the floor was a genuine contender and not a strawman. But it is *high*: the
perplexities are loose (WikiText-2 66.67, LAMBADA 82.36) and the downstream accuracies are modest
(ARC-Easy 51.47, HellaSwag 31.12, WinoGrande 52.01, PIQA 62.40). HellaSwag at 31.12 is barely above
chance for a four-way task and tells me the model is not holding the kind of mid-range context that
sentence-completion needs. This is the failure I expected from the construction, and it is sharp: it is
not an optimization failure — the loss curve was clean and the model is in range — it is a *memory*
failure. RetNet forgets at a rate it chose before it ever saw a token. Every head has a single fixed
`γ`, geometrically spaced across heads, and within a head that decay is the same for every channel,
every context, every word. It cannot look at a token and decide "this is a key fact, hold it" versus
"this is filler, let it decay." The lesson from 1-D RNNs is unambiguous — the forget gate carries most
of a gated cell's capacity and it has to be *data-dependent* to do its job — and a fixed scalar `γ` is
exactly a data-independent gate. So the diagnosis points at one move: the decay must become a function
of the input. The whole risk, the thing RetNet's fixed scalar was *protecting*, is that a
data-dependent gate normally destroys the matmul chunkwise training form. That is the wall I have to get
through, and getting through it without re-breaking what made retention trainable is rung two.

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
serializes the recurrence and kills parallel training. Martin and Cundy's fix is the way out: make the
gate depend *only on the current input*, not on the previous state. Then the recurrence stays linear in
the state — the gate is a sequence of input-determined coefficients — and a linear recurrence with
input-determined coefficients is still parallelizable, and crucially it still has the cumulative-product
structure I can exploit. HGRN showed input-only gating works at moderate scale. So the constraint is
fixed: the gate is a function of `xₜ` alone.

What *shape* should the gate be? This is the real design decision, and it is a three-way trade between
parameter efficiency, state expressivity, and training efficiency. The extremes bracket it. The most
general gate is a full matrix `Gₜ ∈ (0,1)^{d_k×d_v}` applied Hadamard to the state,
`Sₜ = Gₜ ⊙ S_{t−1} + kₜᵀvₜ` — every entry forgets at its own data-dependent rate, maximally expressive
— but mapping `xₜ → Gₜ` directly needs a projection of size `d·d_k·d_v`, absurd in parameters. The
other extreme is RetNet's scalar, the rung I am leaving behind: cheap, trivially parallel, but one
number for the whole state, and 2.4795 is what that costs. There is a middle, a low-rank outer-product
gate `Gₜ = αₜᵀβₜ`, only `d·d_k + d·d_v` parameters, a genuine 2-D gate with per-key-channel and
per-value-channel recency control. So why not just take that? Because of training. The prior
matrix-gate algorithms, to handle a data-dependent matrix gate, materialize the full matrix-valued
hidden state for *every* time step in slow memory — `L·d_k·d_v` written to HBM — and the gated update
they implement does not reduce to tensor-core matmuls. So it is slow at scale, both from the I/O and
from leaving the tensor cores idle. There is a parallel cautionary tale in Mamba: it makes the state
transition fully input-dependent and full-rank, even more expressive, but *precisely because* the
full-rank selective update cannot be written as a matrix multiply, it cannot touch tensor cores and must
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

Now the numerics, because the clean parallel form is dead on arrival. The recurrent form is
`Sₜ = Diag(αₜ) S_{t−1} + kₜᵀvₜ`, `oₜ = qₜ Sₜ`, with `αₜ ∈ (0,1)^{d_k}` from `xₜ`, and the parallel
form uses `bₜ = ∏_{j≤t} α_j`: `P = (Q ⊙ B)(K / B)ᵀ`, `O = (P ⊙ M) V`. But each `α_j < 1`, so `bₜ` is a
cumulative product of sub-one numbers, decaying toward zero fast over a long sequence, and `K / B`
divides by something tiny and *explodes* — overflow in fp16 or even fp32. The fix is to compute the
scores in log space: `P_{ij} = Σ_k Q_{ik} K_{jk} exp(log B_{ik} − log B_{jk})` for `i ≥ j`, where the
cumulative product becomes a cumulative *sum* of log-gates `log bₜ = Σ_{j≤t} log α_j` — a sum of
negative numbers, no underflow — and `log B_{ik} − log B_{jk}` is the accumulated log-decay between key
`j` and query `i`, i.e. a data-dependent relative-position factor (a learned, content-dependent ALiBi,
which is the spiritual successor to RetNet's fixed `γ^{n−m}`). Stable — but `exp` of a difference that
depends on both `i` and `k` sits between `Q` and `K`, so the score is *not* a single matmul, and I have
thrown away the tensor cores. The escape is that the underflow is a *long-range* phenomenon: within a
chunk of length `C`, the cumulative product runs over at most `C` steps and stays bounded. So redo the
chunking with the gate, measuring cumulative gates *relative to chunk boundaries* — `Λ` the decay from
chunk start, `Γ` the decay to chunk end, `γ_chunk` the whole-chunk decay — so every cumulative product
spans at most one chunk. The inter-chunk recurrence decays the carried state by the whole-chunk gate and
adds the boundary-weighted keys; the intra-chunk output is the same masked parallel form with chunk-local
preconditioning. A second level of tiling pushes the full-precision log-space work down to the small
diagonal sub-blocks while every off-diagonal sub-block and the inter-chunk recurrence run as
half-precision matmuls. The same FlashLinearAttention I/O tricks — materialization for sequence-level
parallelism at small batch, recomputation in the backward to claw back memory, and a closed-form
log-space gate gradient `d log αₜ` (no per-step states) — make it fast in wall-clock, not just FLOPs.

The remaining design choices each have to earn their place. The gate `αₜ ∈ (0,1)^{d_k}` is data-dependent
on `xₜ`, so a linear map plus sigmoid is natural — but a full `d×d_k` gate projection adds real
parameters, and I am trying to hold the same `4d²` budget as softmax (and as the RetNet rung, so the
comparison stays clean). Make it **low-rank**: `xₜ → W_α¹ → W_α² → sigmoid` with a rank-16 bottleneck,
nearly free in parameters and more than enough to choose a per-channel forget rate from content. And a
subtlety that matters: a fresh sigmoid gate sits near 0.5, meaning the state *halves* every step — far
too aggressive, the model cannot hold anything for more than a few tokens, long-range capacity dead
before training starts. Bias it toward 1 with a temperature: `αₜ = σ(logits)^{1/τ}` with `τ = 16`. Since
`σ < 1`, the `1/16` power pushes toward 1 (`0.5^{1/16} ≈ 0.96`), so slow forgetting is the default and
the model must actively decide to forget — and in log space this is `log αₜ = (1/16) logsigmoid(logits)`,
exactly the small, well-conditioned cumulative-sum quantity the stable form wants. The temperature does
double duty: long-memory prior *and* a small log-decay. I scale queries by `1/sqrt(d_k)` (the standard
attention logit scaling). And the dimensions: the FLA layer exposes `expand_k` and `expand_v`, and here
I depart from the symmetric width I used for RetNet on purpose. The state is `d_k×d_v` and its size is
memory capacity, so I want the value full-width, `d_v = d` (`expand_v = 1.0`), but I do not need the key
as large, so `d_k = d/2` (`expand_k = 0.5`) — controlling the parameter count and state size while
still leaving a sizeable `d/2 × d` state and, with the low-rank gate, landing the layer back at ~`4d²`.
Multi-head split, per-head RMSNorm on each head's output (a linear-attention layer has no softmax
normalizing the heads to a simplex, so they need taming), then a Swish output gate `r = swish(xW_r)`
applied multiplicatively before the projection — the same gated-nonlinearity recipe I used for RetNet,
which matters for closing the gap.

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
perplexities should tighten — WikiText-2 and LAMBADA were the loosest under RetNet, and content-chosen
forgetting is most valuable precisely where a fixed rate threw away the salient token. The downstream
accuracies should rise, HellaSwag especially, since 31.12 read as "not holding mid-range context" and a
data-dependent gate is the cure for that. But I hold one doubt that the next rung may have to answer:
GLA's gate is still a *diagonal, multiplicative* decay — it forgets globally or per-channel, it does not
remove the *specific stored association* that a new key collides with. If the residual failure after GLA
is recall — fetching a particular value I stored earlier keyed on a token I have now seen again — then
the gate, however data-dependent, is structurally the wrong tool, and the diagnosis would point at
changing the *write rule* itself rather than the decay. For now, the falsifiable claim is simple: GLA
should beat RetNet on val_loss, on both perplexities, and on the downstream suite, and if it does not
beat it on perplexity then the data-dependence is not buying what the RNN literature promised.
