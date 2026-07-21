The uncached control told me what is missing, in numbers, so read them before moving. Reuse is a flat 0.0
with refresh 1.0 on all three — no other value the floor can take. Throughput came in 19.54 on MATH, 11.46
on HumanEval, 35.72 on ARC: ARC fastest, HumanEval slowest, MATH between, exactly the ordering the schedule
arithmetic predicted from `N` and step count. That matters — it means the floor's per-step cost really is
dominated by the full forward over the whole sequence, so a policy that shrinks the per-step forward
attacks the right thing, and one that does not will not move throughput no matter how much reuse it books.

The quality column is where the constraint lives. MATH 38.4 clears its 35 gate by 3.4; HumanEval 43.9
clears 40 by 3.9; ARC is 83.959 against a gate of 84, so it lands 0.041 *under* — right on the line, which
the near-lossless soft gate absorbs with a negligible penalty but which says something sharp: the ARC gate
is calibrated essentially at native accuracy, so ARC has *no headroom to give*. Any policy that degrades ARC
quality at all starts its fall from the wrong side of the line. MATH has 3.4 points of nominal headroom but
is the most fragile place to spend them — a MATH answer is a long exact-match string and one flipped
committed digit is a zero. So the two long-tail risks are opposite in shape: ARC tolerates no drop from a
starting point at the gate, MATH tolerates a small drop in principle but punishes the *kind* of error
caching produces. HumanEval, with 3.9 points of slack and execution-checked local structure, is the most
forgiving.

So the diagnosis is not quality; the policy spends compute it does not have to. The redundancy is the one I
sized at the floor — most per-position recomputations reproduce a value already computed — and the floor
books none of it. The move is forced in direction: reuse the KV of positions whose context did not
meaningfully change this step, recompute only those that did. The whole question is *which positions*, and
at *what granularity* I am allowed to decide it.

The crude answer is the segment-and-clock cache: split into static prompt and dynamic response, a long
clock on one, a short clock on the other. But a segment is not a unit of KV dynamics — inside one response
segment some tokens are dead stable and some are about to change a lot, and a single interval treats them
the same, so it refreshes settled tokens (wasted, the floor's sin in miniature) and reuses moving ones
(lost quality, dangerous on MATH). The alternative that keeps every prediction computed from a complete
input — run all queries, reuse only internal features with a per-token top-up — is safe but leaves the
per-step forward full-cost, which I just read is exactly what pins throughput; I set it aside as the
conservative option if aggression fails. The aggressive move attacks the granularity head-on: decide reuse
*per token, per step*, and narrow the forwarded query rows to only the tokens that changed. That is the one
that can win the reuse term hard, so I follow it — eyes open to the MATH fragility I just measured.

A masked token's KV trajectory across the rollout is not a smooth drift and not a constant; it is three
phases. Early on the key and value barely move — a gradual-change phase, a far-off `[MASK]` whose
neighborhood is still mostly masked. Then, in the handful of steps right before the token is decoded, the
state lurches — a rapid-change phase. Then, once decoded, it goes essentially still. The lurch has a cause:
it happens exactly when the local context fills in, because when a neighbor flips from a `[MASK]` embedding
to a concrete token that is a large change to the context this token attends to, and the closer the neighbor
the bigger the change — the same mechanism from the floor, where revealing position 2 swung position 1's
read-out. So a masked token's KV moves most when its neighborhood resolves, and only then: I need to
recompute it during its rapid-change phase, and in the gradual and stable phases I can reuse and lose
nothing.

But to refresh a token *during* its rapid-change phase I have to identify it *before* it is decoded — which
is exactly what the decoding process is figuring out. Circular. I need a predictor of imminence readable
from the current state without having decoded the token. The decoding order supplies it: the model
overwhelmingly decodes near the token it just decoded — the front is spatially local — so a masked token
surrounded by known tokens has a tightly constrained context and resolves sooner. Imminence is the *density
of known tokens in the local neighborhood*, which depends only on which positions are currently known. The
circularity breaks.

Make the density distance-aware, since closeness is the point: for masked position `i`, `D(i) = Σ_j
exp(−|i−j|²/2σ²)·1{j known}` over prompt and decoded positions, no hard cutoff. The knob `σ` is the
neighborhood width, and I want its reach to match the decoding front's, which sits at order ten positions —
so `σ = 10`. That puts the half-weight radius at `√(200·ln2) ≈ 11.8`: a known token contributes
substantially within about twelve positions, a tenth of its weight by twenty, effectively nothing past
thirty — a soft window with support ~±20, matching a front of order ten. `σ = 2` would see only immediate
neighbors and miss the front's width; `σ = 40` would blur distant irrelevant tokens in. Density is
*structural* imminence; I also have, free after each forward, the prediction confidence `s^i`. Structural
and predictive certainty differ — a token can sit in a dense neighborhood yet be predicted unsurely — so I
multiply, `D(i)·s^i`, a soft AND, and refresh the top-`current_k` masked tokens by it. The rapid-change
front is about one block wide, so `current_k = 32` sizes the budget to `block_length`: smaller refuses
genuinely-lurching tokens, larger spends refreshes on tokens still gradual.

That criterion is blind to prompt and decoded tokens — they have no `s^i`. Their KV moves little, but never
refreshing them accumulates the second-order prompt drift I flagged at the floor, which on ARC — sitting at
its gate — I cannot let creep. They need a different reason to refresh: that *other* tokens depend on them.
So a second stage selects by **attention importance**, folding attention across all 32 layers with the
residual path in (`W = normalize(E+I)`, `C ← W·C`) and taking the column sum `c_j = Σ_i C_ij`, then
selecting by nucleus mass `rollout_p = 0.1`. A tenth of the cumulative influence captures the genuinely
salient tokens because dLLM attention is concentrated, and adjacent-step rollout maps are stable enough that
this step's importance predicts next step's set. 0.1 not 0.5 because this stage is a top-up: the `D·s` stage
already covers the imminent front, and this only needs the handful of heavily-attended anchors whose drift
would poison many readers; half the mass would drag in a long mildly-attended tail. The recompute set is the
union — imminent masked tokens by `D·s`, salient known tokens by rollout, plus any freshly transferred token
(its embedding just changed, refresh once) — and everything else reuses cache.

Land it in the task's hooks, noting two places this differs from the generic method. First the block
schedule: the generic method keeps semi-autoregressive blocks, but *this runs as pure diffusion* —
`block_length = gen_length`, `num_steps = gen_length`, the whole response one block decoded one token per
step. That works because `D·s` doubles as a decoding criterion: it prefers tokens next to known tokens,
known tokens grow outward from the prompt, so decoding by `D·s` is quasi-left-to-right and curbs the
premature-EOS pathology that blocking existed to fix. Second the query plan: `full_sequence` at step 0 to
fill the cache, then `active_query_rows` on every later step, handing the harness `active_q_mask` and the
masked window; `row_selector = "certainty_density_attention_rollout"`, `kv_update = "active_q_mask"`
(inactive rows reuse cached K/V), feature cache off because the saving here is narrowing query rows, not
interval-skipping features. `attention_probe_plan` sets `need_attention_weights = True` — the rollout needs
the explicit `softmax(QKᵀ)` the fused kernels never materialize, so the harness routes attention eager —
carrying `rollout_p = 0.1`, `current_k = 32`, `sigma = 10.0`. Against the generic default I set `inflate_w =
0`: the method offers a gap-inflation that fills holes between selected rows to keep the kernel contiguous,
but I keep the active set strictly minimal, trading kernel efficiency for the smallest recompute set and
highest reuse. Transfer stays low-confidence over the current block, force one; `after_step` is where the
harness updates the active mask, rollout, tracked positions, and density scores (the full policy is in the
answer).

So d2Cache forwards only the union — imminent masked (`D·s` top-32), salient known (rollout `p=0.1`),
freshly transferred — reusing cached K/V for everyone else. Predict the reuse from the active-set
arithmetic: on MATH the generated region is 256 wide and ~32 rows are active, about 0.125 refreshed, reuse
near 0.875; on HumanEval 512 wide, ~32/512 is lower still but late-rollout anchors grow, so high 0.8s; on
ARC only 64 wide with `current_k = 32` half of it, and though the masked count falls fast, the average
active fraction stays highest, so ARC lands lowest — I expect roughly 0.81–0.90, highest on the long
workloads, lowest on ARC. At step 0 the whole sequence is masked so `D = 0` and `D·s` degenerates to pure
confidence, but the query plan already forces `full_sequence` there to fill the cache, so the design never
depends on `D` being informative before any token is known.

Throughput is the term I am not optimistic about. The active queries still attend over the whole sequence —
narrowing the query rows cuts the number of rows I compute, but each surviving row still reads keys and
values across all positions, so the quadratic term is not cut in proportion. On top of that eager attention
materializes the full `softmax(QKᵀ)`, and folding the rollout matrix through 32 layers every step is real
overhead. So I expect tokens/s only modestly above the floor on the long workloads, and I am uneasy about
ARC specifically: its floor throughput is the field-high 35.72 over a tiny forward, and the
eager-attention-plus-rollout tax is a fixed per-step cost a 64-token generation may not out-save — it would
not shock me if ARC throughput *fell* below its floor. A falsifiable call: high reuse but ARC tokens/s under
35.72 confirms this scheme buys reuse and not speed.

The risk I can feel is the MATH gate. `D·s` is a *structural* imminence proxy — it scores a token by
whether its neighborhood looks resolved, not by whether its feature actually moved. On a long sequential
MATH answer a token can have a sparse-looking neighborhood (low `D`) yet a feature that shifted because of a
*distant* commit that bidirectional attention propagated, and the selector will leave it on stale cache;
that stale feature flips a digit, the commit freezes, MATH loses a point it cannot recover. MATH has only
3.4 of headroom and the error mode caching produces is most lethal there, so if the narrowing costs a few
points MATH slides toward or under 35, the gate multiplies that workload down, and under the geometric mean
one gated workload drags the whole policy below a blander one that kept every gate. HumanEval's slack and
ARC's 64 tokens should tolerate the same staleness and hold near 43.9 and 84. So MATH is the one to watch —
if it falls under 35 the policy loses on the geometric mean despite the best reuse on the board, and the
next step's job is already written: keep selective reuse but drive the refresh off whether the *prediction*
actually changed, not a structural proxy.
