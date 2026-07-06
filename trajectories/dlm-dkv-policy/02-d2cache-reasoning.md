The uncached control told me exactly what is missing, and it told me in numbers, so let me read the numbers
carefully before I move, because they both confirm the cost model I built at the floor and set the exact bar
I have to clear. On all three workloads the reuse_ratio is a flat 0.0 with refresh_ratio 1.0 — no surprise,
there is no other value the floor can take — and the throughput came in at 19.54 tokens/s on MATH, 11.46 on
HumanEval, 35.72 on ARC. That ordering, ARC fastest and HumanEval slowest with MATH between, is exactly the
ordering my schedule arithmetic predicted from `N` and step count, which is worth a beat: it means the floor's
per-step cost really is dominated by the full forward over the whole sequence, so a policy that shrinks the
per-step forward is attacking the right thing, and one that does not will not move throughput no matter how
much reuse it books. Now the quality column, which is where the real constraint lives. MATH 38.4, HumanEval
43.9, ARC 83.959. Subtract the gates: MATH clears `35` by `3.4`, HumanEval clears `40` by `3.9`, and ARC —
the gate is `84` and the score is `83.959`, so ARC does not clear it; it lands `0.041` *under*, right on the
line, which the near-lossless soft gate absorbs with a negligible penalty but which tells me something sharp.
The ARC gate is calibrated essentially at native accuracy, so ARC has *no headroom to give* — any policy that
degrades ARC quality at all starts its fall from the wrong side of the line. MATH, by contrast, has 3.4 points
of nominal headroom but is the most fragile place to spend them, because a MATH answer is a long exact-match
string and one flipped committed digit is a zero. So the two long-tail risks are opposite in shape: ARC
tolerates no drop from a starting point at the gate, MATH tolerates a small drop in principle but punishes the
*kind* of error caching produces. HumanEval, with 3.9 points of slack and execution-checked local structure,
is the most forgiving.

So the diagnosis is not a quality problem at all; it is that the policy spends compute it does not have to.
The redundancy is real and I sized it at the floor — across a rollout there are on the order of `num_steps`
distinct global inputs, and at any single position the local conditioning changes only during the handful of
steps its neighborhood is resolving, so the great majority of per-position recomputations reproduce a value
already computed — and the floor captures none of it, booking reuse 0. The move is therefore forced in
direction: start reusing the keys and values of positions whose context did not meaningfully change this step,
and recompute only the positions that did. The entire question is *which positions*, and at *what granularity*
I am allowed to decide it, and that granularity choice is where the design space forks.

Three options are genuinely on the table, and I want to walk them rather than reflex to one. The first is the
segment-and-clock cache I already sketched at the floor: split the sequence into a static part (prompt) and a
dynamic part (response), refresh the static part on a long clock and the dynamic part on a short one. Walk it
one step further than before. A segment is not a unit of KV dynamics: inside one response segment at one step,
some tokens are dead stable and some are about to change a lot, and a single interval has to treat them the
same — so it inevitably does both wrong things at once, refreshing tokens whose KV already settled (wasted
compute, exactly the floor's sin in miniature) and reusing tokens whose KV is actively moving (lost quality,
which on a gated metric like MATH is dangerous). The second option is an interval *feature* cache with a
per-token top-up: run all queries but reuse cached internal features for most tokens and recompute a chosen
fraction. That keeps every prediction computed from a complete input, which is safe, but "run all queries"
means the per-step forward stays full-cost, and I just read off the floor that per-step forward cost is what
pins throughput — so this option, whatever it buys in reuse, cannot move the throughput term, and it books
less reuse per unit risk because it never narrows the forward. I set it aside as the *conservative* fallback
if aggression fails, not the first move. The third option attacks the granularity problem head-on: decide
reuse *per token, per step*, and narrow the actual forwarded query rows to only the tokens that changed. This
is the aggressive option, and it is the one that could win the reuse term hard, so I follow it — with my eyes
open to the MATH fragility I just measured.

To decide reuse per token I have to know what a single masked token's KV trajectory looks like across the
rollout, so let me trace it. It is not a smooth drift and not a constant; it is three phases. Early on the key
and value barely move — a gradual-change phase, the token is a far-off `[MASK]` whose neighborhood is still
mostly masked, and its representation is a bland prior. Then, in the handful of steps right before this token
itself gets decoded, the state lurches — a rapid-change phase. Then, once decoded, it goes essentially still —
a stable phase, it is a committed real token and its own embedding is fixed. And the lurch has a cause I can
name: it happens exactly when the token's local context is filling in, because when a neighbor flips from a
`[MASK]` embedding to a concrete token, that is a large change to the context this token attends to, and the
closer the neighbor the bigger the change — precisely the mechanism I traced at the floor, where revealing
position 2 swung position 1's read-out. So the KV of a masked token moves most when its neighborhood is
resolving, and only then. That is the lever the floor ignored: I only need to recompute a masked token's KV
during its rapid-change phase; in the gradual and stable phases I can reuse it and lose nothing, because I am
not skipping an informative update.

But this walks straight into a wall. To refresh a token *during* its rapid-change phase I have to identify it
*before* it is decoded — and "about to be decoded" is exactly what the decoding process is in the middle of
figuring out. Circular. So I need a predictor of imminence that does not require having already decoded the
token, something structural and readable from the current state. The decoding order supplies it: the model
overwhelmingly decodes the next token *near* the one it just decoded — the decoding front is spatially local —
which fits the lurch, since a masked token surrounded by already-known tokens has a tightly constrained context
and resolves sooner. So imminence is the *density of known tokens in the local neighborhood* of a masked
position, which depends only on which positions are currently known, not on having decoded the masked one. The
circularity breaks.

Turn "density of known tokens nearby" into a number, and make it distance-aware, because closeness is the whole
point — a known token right next to my masked position should count for far more than one ten positions away. A
Gaussian in the separation distance does that with no hard cutoff: for masked position `i`, `D(i) = Σ_j
exp(−|i−j|²/2σ²) · 1{j known}`, summing over prompt and already-decoded positions. The single knob `σ` is the
neighborhood width, and I want its reach to match the empirical reach of the decoding front, which sits at a
scale of about ten positions. So set `σ = 10` and check what that actually weights, because I do not want to
guess the shape. At separation 0 the weight is 1; at 5 it is `exp(−25/200) = 0.88`; at 10 it is `exp(−0.5) =
0.61`; at 15, `exp(−1.125) = 0.32`; at 20, `exp(−2) = 0.14`; at 30, `exp(−4.5) = 0.011`. Solving
`exp(−d²/200) = 0.5` puts the half-weight radius at `d = √(200·ln2) ≈ 11.8`. So a known token contributes
substantially within about twelve positions, is down to a tenth of its weight by twenty, and is effectively
invisible past thirty — a soft window whose meaningful support is roughly `±20`, matching a decoding front of
order ten. That is the reason `σ = 10` and not `σ = 2` (which would see only immediate neighbors and miss the
front's real width) or `σ = 40` (which would blur distant, irrelevant tokens into the score). Density tells me
a token is *structurally* imminent; but I also have, for free after each forward, the model's own prediction
confidence `s^i`. Structural and predictive certainty are different — a token can sit in a dense neighborhood
yet be predicted unsurely, or vice versa — and I want both, so I multiply: the calibrated score is `D(i)·s^i`,
a soft logical AND, and I refresh the top-`current_k` masked tokens by it.

How big should `current_k` be? The rapid-change front is narrow — it is the neighborhood that is resolving
right now — so a small fixed budget suffices, and there is a natural scale to pin it to: the block. LLaDA
decodes in blocks of `block_length = 32`, so the set of positions actively resolving at any moment is about one
block wide, and `current_k = 32` sizes the imminent-refresh budget to exactly that. It is not a coincidence I
want to hide behind; it is the reason the number is 32 and not 8 or 128 — smaller would refuse to refresh some
genuinely-lurching tokens (quality risk on the fragile workloads), larger would spend refreshes on tokens still
in their gradual phase (wasted compute, and lost reuse).

That criterion handles masked tokens, but it is blind to the prompt and already-decoded tokens — they have no
prediction confidence `s^i` to multiply in. Their KV moves little, which is why caching them helps, but
"little" is not "nothing," and never refreshing them accumulates the second-order prompt drift I flagged at the
floor, which on ARC — sitting exactly at its gate — I cannot afford to let creep. They need a *different* reason
to refresh, and the right one is that *other* tokens depend on them: a token many queries attend to strongly is
one whose KV I cannot let drift. So a second stage selects by **attention importance**, and to measure
importance faithfully through a deep stack I cannot just read the last layer — attention composes across the 32
layers — so I fold the whole stack with the residual path included (`W = normalize(E + I)`, `C ← W·C`), then
take the column sum `c_j = Σ_i C_ij` as how much influence flows into token `j`. I select these by nucleus mass
with `rollout_p = 0.1` — a tenth of the cumulative influence already captures the genuinely salient tokens,
because dLLM attention is concentrated on a few positions, and the adjacent-step rollout maps are stable enough
that this step's importance predicts next step's recompute set. Why 0.1 and not 0.5? Because the nucleus is a
*top-up*, not the main course: the masked-token stage already covers the imminent front, and this stage only
needs to catch the handful of heavily-attended anchors whose drift would poison many readers; a tenth of the
influence mass is a few tokens, which is the right size for "the anchors," whereas half the mass would drag in
a long tail of mildly-attended tokens and inflate the forward for little quality return. The recompute set is
the union: the imminent masked tokens by `D·s`, the salient known tokens by rollout, plus any token freshly
transferred this step (its embedding just changed fundamentally, so it must be refreshed once). Everything else
reuses cache.

Now make this concrete *in this task's hooks*, because I do not write the per-layer block forward or the
rollout matrix — the harness owns those — I write the plan that drives them, and the plan has to match this
task's exact configuration, which differs from the generic method in two places I have to get right. First,
the block schedule. The generic method preserves a frame/delta outer loop with semi-autoregressive blocks;
*this task runs d2Cache as pure diffusion* — `block_schedule` sets `block_length = gen_length` and
`num_steps = gen_length`, so the whole response is one block decoded one token per step, no block boundaries.
That matters because `D·s` is doubling as a decoding criterion: it already prefers tokens next to known tokens,
and known tokens grow outward from the prompt, so decoding by `D·s` produces a quasi-left-to-right order on its
own and curbs the premature-EOS pathology that block decoding existed to fix — so the harness can drop the
explicit blocks here. Second, the query plan: at step 0 I forward the `full_sequence` to fill the cache, and on
every later step I narrow to `active_query_rows`, handing the harness the `active_q_mask` from `cache_state`
and the masked window `(block_start, block_end)`; the `cache_refresh_plan` declares `row_selector =
"certainty_density_attention_rollout"` and `kv_update = "active_q_mask"` (inactive rows reuse cached K/V), with
the feature cache off because the saving here is narrowing the *query rows*, not interval-skipping features. The
`attention_probe_plan` must request `need_attention_weights = True` — the rollout needs the explicit
`softmax(QKᵀ)` matrix, which the fused kernels never materialize, so the harness routes attention through eager
mode — and carries `rollout_p = 0.1`, `current_k = 32`, `sigma = 10.0`. The one knob I deliberately set against
the generic default is `inflate_w = 0`: the method offers a gap-inflation that fills small holes between
selected rows to keep the attention kernel contiguous, but this task keeps the active set strictly minimal (no
inflation), trading some kernel efficiency for the smallest possible recompute set and the highest possible
reuse_ratio. Transfer stays low-confidence over the current block, forcing one; `after_step` is where the
harness updates the active query mask, attention rollout, tracked positions, and density scores from the stats
it just produced (the full policy is in the answer).

Before I trust this, let me trace the active set on one concrete mid-rollout MATH state and check the two
degenerate ends, because a selector that misbehaves at the boundaries is a selector that silently breaks the
gate. Take the step where half the response is committed: `prompt ≈ 128`, generated region 256 with 128 already
decoded and 128 still masked, so `N ≈ 384`. The `D·s` stage scores the 128 masked tokens; the decoded tokens
cluster toward the left (quasi-left-to-right by `D·s`), so the masked tokens just to the right of the decoding
front have several known neighbors within `σ = 10` and score high, while masked tokens deep in the still-blank
tail have `D ≈ 0` and score near zero — the top-32 lands right on the front, which is exactly the rapid-change
band I wanted. The nucleus stage adds a few heavily-attended anchors (a tenth of the influence mass on a
concentrated attention pattern is a handful of tokens, mostly early prompt and recently-decoded anchors), and
the transfer adds the one token committed last step. So the active set is roughly `32 + ~4 + 1 ≈ 37` rows out
of 384, and the other ~347 reuse cache — a refresh fraction around `37/256 ≈ 0.14` on the generated region,
i.e. reuse near `0.86`, consistent with the arithmetic I am about to lean on. Now the two ends. At step 0 the
whole sequence is masked, so `D(i) = 0` for every masked token (no known neighbors yet) and `D·s` degenerates
to pure confidence `s^i`; the query plan already forces `full_sequence` at step 0 to fill the cache, so this
degeneracy never actually drives a narrowed forward — good, the design does not depend on `D` being informative
before any token is known. At the other end, when fewer than 32 masked tokens remain, `current_k = 32` simply
selects all of them, the active set shrinks to the true remaining front, and reuse rises toward 1 as the
rollout finishes — no pathology. The selector behaves at both boundaries, and the mid-rollout trace matches the
reuse I expect.

So the delta from the floor is concrete: where the uncached control forwarded `full_sequence` and refreshed
every row every step, d2Cache forwards only the union of the imminent masked tokens (`D·s`, top-32), the
salient known tokens (rollout nucleus, `p=0.1`), and the freshly transferred tokens, reusing cached K/V for
everyone else, with `inflate_w=0` so the active set is as small as the selection allows. Now let me predict the
reuse from the arithmetic rather than hand-wave it. Per step the active set is roughly `current_k = 32` imminent
masked tokens plus the freshly transferred (about one on a pure-diffusion rollout) plus the nucleus anchors,
and reuse_ratio measures the fraction of *generated-token* cache work reused. On MATH the generated region is
256 wide, so a ~32-of-256 refresh is about `0.125`, putting reuse near `0.875`; on HumanEval the region is 512,
so ~32-of-512 is `0.0625` and reuse could reach the high `0.9`s except that late in the rollout most of the
sequence is decoded and the anchor set grows, so I temper that to the high `0.8`s; on ARC the region is only 64
wide and `current_k = 32` is half of it, so early steps can have nearly half the region active, but the masked
count falls fast as tokens commit, dragging the average active fraction well under a half — I expect ARC to
land at the *low* end, in the low-to-mid `0.8`s, the least reuse of the three precisely because its generation
is shortest relative to the fixed 32-budget. So my reuse prediction is roughly `0.81–0.90` across workloads,
highest on the long ones, lowest on ARC — the reuse_ratio column will tell me if the active-set model is right.

Throughput is the term I am *not* optimistic about, and the floor already told me why. The active queries still
attend over the whole sequence — narrowing the query rows shrinks the number of rows I compute, but each
surviving row still reads keys and values across all positions, so the quadratic attention term is not cut in
proportion to the row count. On top of that the rollout forces eager attention, which materializes the full
`softmax(QKᵀ)` the fused kernels skip, and computing `D·s` plus folding the rollout matrix through 32 layers
every step is real overhead. So I expect tokens/s only modestly above the floor's — into the low-20s on the
long workloads where the floor sat around 20 and 11 — and I am genuinely uneasy about ARC: its floor throughput
is the field-high `35.72`, its per-step forward is already tiny, and the eager-attention plus rollout overhead
is a fixed per-step tax that a 64-token generation may not out-save. It would not shock me if ARC throughput
*fell* below its floor. That is a falsifiable call: if d2Cache books high reuse but ARC's tokens/s drops under
35.72, the diagnosis is confirmed that this scheme buys reuse and not speed, and the throughput term stays
unwon.

The risk I can already feel, though, is the quality gate, and the floor's headroom arithmetic points it
straight at MATH. Aggressively narrowing the query rows on a pure-diffusion rollout means reusing the KV of any
token my `D·s` selector underrates, and `D·s` is a *structural* imminence proxy — it scores a token by whether
its neighborhood looks resolved, not by whether its feature actually moved. On a long, sequential MATH answer a
token can have a sparse-looking neighborhood (low `D`) and yet have a feature that shifted because of a
*distant* commit that bidirectional attention propagated — exactly the everyone-moves-when-anyone-reveals
mechanism from the floor — and the density selector will happily leave it on stale cache. That stale feature
flips a digit, the commit freezes, and MATH loses a point it cannot recover. MATH has only `3.4` of nominal
headroom above its gate, and it is the workload where the error mode caching produces is most lethal, so if the
active-row narrowing costs even a few accuracy points, MATH slides from 38.4 toward or under 35, the gate
penalty multiplies that whole workload down, and since the task score is a geometric mean across all three
workloads, one gated workload drags the entire policy below a blander one that kept every gate. HumanEval, with
its slack and its shorter execution-checked spans, and ARC, whose 64 tokens leave almost nothing to get stale,
should tolerate the same staleness and hold near their floor scores of 43.9 and 84.0. So concretely against the
floor I expect: reuse from 0.0 up to ~0.81–0.90, highest on the long workloads; throughput inching up on MATH
and HumanEval and possibly falling on ARC; HumanEval and ARC quality holding; and MATH the one to watch — if it
falls below 35 the math workload craters and the policy loses on the geometric mean despite the best reuse on
the board. If that is what happens, the next step's job is already written: keep this kind of selective reuse
but stop bleeding MATH accuracy — refresh on a signal tied to whether the *prediction* actually changed, not
just on a structural density proxy, and back off the query narrowing that threw away the exact rollout's
accuracy, so the policy buys reuse without paying the quality the gate punishes.
