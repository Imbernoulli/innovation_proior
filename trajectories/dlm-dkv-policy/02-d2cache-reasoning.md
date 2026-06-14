The uncached control told me exactly what is missing, and it told me in numbers. On all three workloads
the quality gate is clear — MATH 38.4, HumanEval 43.9, ARC 84.0 — and on all three the reuse_ratio is a
flat 0.0 with refresh_ratio 1.0. That is the floor behaving precisely as constructed: it recomputes every
position every step, so the dominant efficiency term (reuse, weight 0.75) is zero everywhere, and the
throughput is the slowest of the field (19.5 / 11.5 / 35.7 tokens/s) because it does the most work per
step. The diagnosis is not a quality problem at all; it is that the policy spends compute it does not have
to. The redundancy is real — across a rollout there are at most `L` distinct conditioning states, most
steps flip only a handful of masked positions and leave the rest of the input identical — and the floor
captures none of it. So the move is forced: start reusing the keys and values of positions whose context
did not meaningfully change this step, and recompute only the positions that did. The whole question is
*which positions*, and at *what granularity* I am allowed to decide it.

The crude answer is a segment-and-schedule cache: split the sequence into a static part (prompt) and a
dynamic part (response), refresh the static part on a long clock and the dynamic part on a short one. But a
segment is not a unit of KV dynamics. Inside one response segment at one step, some tokens are dead stable
and some are about to change a lot, and a single interval has to treat them the same — so it inevitably
does both wrong things at once: refreshes tokens whose KV already settled (wasted compute, exactly the
floor's sin in miniature) and reuses tokens whose KV is actively moving (lost quality, which on a gated
metric like MATH is dangerous). To beat the floor on reuse *without* sliding under the quality gate I need
the dynamics at the granularity they actually live at, which is per token, per step.

What does a single masked token's KV trajectory look like across the rollout? It is not a smooth drift and
not a constant; it is three phases. Early on the key and value barely move — a gradual-change phase. Then,
in the handful of steps right before this token itself gets decoded, the state lurches — a rapid-change
phase. Then, once decoded, it goes essentially still — a stable phase. And the lurch has a cause I can name:
it happens exactly when the token's local context is filling in, because when a neighbor flips from a
`[MASK]` embedding to a concrete token, that is a big change to the context this token attends to, and the
closer the neighbor the bigger the change. So the KV of a masked token moves most when its neighborhood is
resolving, and only then. That is the lever the floor ignored: I only need to recompute a masked token's KV
during its rapid-change phase; in the gradual and stable phases I can reuse it and lose nothing, because I
am not skipping an informative update.

But this walks straight into a wall. To refresh a token *during* its rapid-change phase I have to identify
it *before* it is decoded — and "about to be decoded" is exactly what the decoding process is in the middle
of figuring out. Circular. So I need a predictor of imminence that does not require having already decoded
the token, something structural and readable from the current state. The decoding order supplies it: the
model overwhelmingly decodes the next token *near* the one it just decoded — the decoding front is spatially
local — which fits the lurch, since a masked token surrounded by already-known tokens has a tightly
constrained context and resolves sooner. So imminence is the *density of known tokens in the local
neighborhood* of a masked position, which depends only on which positions are currently known, not on
having decoded the masked one. The circularity breaks.

Turn "density of known tokens nearby" into a number, and make it distance-aware, because closeness is the
whole point — a known token right next to my masked position should count for far more than one ten
positions away. A Gaussian in the separation distance does that with no hard cutoff: for masked position
`i`, `D(i) = Σ_j exp(−|i−j|²/2σ²) · 1{j known}`, summing over prompt and already-decoded positions. The
single knob `σ` is the neighborhood width; given that the observed decoding locality sits at a scale of
about ten positions, `σ = 10` makes the Gaussian's reach match the empirical reach of the front. Density
tells me a token is *structurally* imminent; but I also have, for free after each forward, the model's own
prediction confidence `s^i`. Structural and predictive certainty are different — a token can sit in a dense
neighborhood yet be predicted unsurely, or vice versa — and I want both, so I multiply: the calibrated
score is `D(i)·s^i`, a soft logical AND, and I refresh the top-`current_k` masked tokens by it. The front
is narrow, so a small fixed budget suffices; `current_k = 32`.

That criterion handles masked tokens, but it is blind to the prompt and already-decoded tokens — they have
no prediction confidence `s^i` to multiply in. Their KV moves little, which is why caching them helps, but
"little" is not "nothing," and never refreshing them accumulates error that would eventually push a gated
score under threshold. They need a *different* reason to refresh, and the right one is that *other* tokens
depend on them: a token many queries attend to strongly is one whose KV I cannot let drift. So a second
stage selects by **attention importance**, and to measure importance faithfully through a deep stack I
cannot just read the last layer — I compose attention across all layers with the residual path folded in
(`W = normalize(E + I)`, `C ← W·C`), then take the column sum `c_j = Σ_i C_ij` as how much influence flows
into token `j`. I select these by nucleus mass with `rollout_p = 0.1` — a tenth of the cumulative influence
already captures the genuinely salient tokens, because dLLM attention is concentrated, and the adjacent-step
rollout maps are stable enough that this step's importance predicts next step's recompute set. The recompute
set is the union: the imminent masked tokens by `D·s`, the salient known tokens by rollout, plus any token
freshly transferred this step (its embedding just changed fundamentally, so it must be refreshed once).
Everything else reuses cache.

Now make this concrete *in this task's hooks*, because I do not write the per-layer block forward or the
rollout matrix — the harness owns those — I write the plan that drives them, and the plan has to match this
task's exact configuration, which differs from the generic method in two places I have to get right. First,
the block schedule. The generic method preserves a frame/delta outer loop with semi-autoregressive blocks;
*this task runs d2Cache as pure diffusion* — `block_schedule` sets `block_length = gen_length` and
`num_steps = gen_length`, so the whole response is one block decoded one token per step, no block
boundaries. That matters because `D·s` is doubling as a decoding criterion: it already prefers tokens next
to known tokens, and known tokens grow outward from the prompt, so decoding by `D·s` produces a
quasi-left-to-right order on its own and curbs the premature-EOS pathology that block decoding existed to
fix — so the harness can drop the explicit blocks here. Second, the query plan: at step 0 I forward the
`full_sequence` to fill the cache, and on every later step I narrow to `active_query_rows`, handing the
harness the `active_q_mask` from `cache_state` and the masked window `(block_start, block_end)`; the
`cache_refresh_plan` declares `row_selector = "certainty_density_attention_rollout"` and `kv_update =
"active_q_mask"` (inactive rows reuse cached K/V), with the feature cache off because the saving here is
narrowing the *query rows*, not interval-skipping features. The `attention_probe_plan` must request
`need_attention_weights = True` — the rollout needs the explicit `softmax(QKᵀ)` matrix, which the fused
kernels never materialize, so the harness routes attention through eager mode — and carries `rollout_p =
0.1`, `current_k = 32`, `sigma = 10.0`. The one knob I deliberately set against the generic default is
`inflate_w = 0`: the method offers a gap-inflation that fills small holes between selected rows to keep the
attention kernel contiguous, but this task keeps the active set strictly minimal (no inflation), trading
some kernel efficiency for the smallest possible recompute set and the highest possible reuse_ratio.
Transfer stays low-confidence over the current block, forcing one; `after_step` is where the harness
updates the active query mask, attention rollout, tracked positions, and density scores from the stats it
just produced (the full policy is in the answer).

So the delta from the floor is concrete: where the uncached control forwarded `full_sequence` and refreshed
every row every step, d2Cache forwards only the union of the imminent masked tokens (`D·s`, top-32), the
salient known tokens (rollout nucleus, `p=0.1`), and the freshly transferred tokens, reusing cached K/V for
everyone else, with `inflate_w=0` so the active set is as small as the selection allows. Reading the floor's
numbers, here is what I expect and where I am uneasy. Reuse_ratio should jump hard off the floor's 0.0 — the
active set is a small fraction of the sequence, so the bulk of generated-token cache work is reused; I would
expect the highest reuse of any policy I have tried, well above 0.8 on the long workloads, because pure
diffusion over a long response leaves most tokens inactive most steps. Throughput, though, should *not*
jump much: I am still doing a full bidirectional forward in the sense that the active queries attend over
the whole sequence, the eager-attention requirement for the rollout adds real overhead, and computing `D·s`
plus the rollout matrix every step is not free — so I expect tokens/s only modestly above the floor's, maybe
into the low-20s where the floor sat at ~20, not a multiplicative gain. The risk I can already feel is the
quality gate, specifically on MATH. Aggressively narrowing the query rows on a pure-diffusion rollout means
reusing the KV of any token my `D·s` selector underrates, and MATH answers are long, sequential, and
unforgiving — a single stale feature that flips a committed digit is unrecoverable, and MATH's gate is the
tightest relative to the floor (35.4 of headroom below the 38.4 native score, gate at 35). If the active-row
narrowing costs even a few accuracy points on MATH, the score falls *under* the gate and the gate penalty
multiplies the whole math workload down, so a policy with the best reuse on the board could still rank below
a blander one that keeps quality. Concretely against the floor: I expect reuse_ratio to go from 0.0 to
~0.81–0.90 across workloads and throughput to inch up from the ~20/11/36 floor rather than leap; HumanEval
and ARC quality should hold near the floor's 43.9 and 84.0, but MATH is the one to watch — if it slides
from 38.4 below 35 the math gate craters that workload, and since the task score is a geometric mean across
all three, one gated workload drags the whole policy down. If that is what happens, the next step's job is
already written: keep this kind of selective reuse but stop bleeding MATH accuracy — refresh on a signal
tied to whether the *prediction* actually changed, not just on a structural density proxy, so the policy
buys reuse without paying the quality the gate punishes.
