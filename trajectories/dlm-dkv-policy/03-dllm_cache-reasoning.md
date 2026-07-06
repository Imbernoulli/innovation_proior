The d2Cache run did what I most worried it would, and the numbers say it sharply, so let me read all three
columns before I react, because the failure is more instructive than "MATH broke." The reuse jumped exactly as
predicted — 0.8655 on MATH, 0.8976 on HumanEval, 0.8102 on ARC, highest reuse I have seen, lowest on ARC just
as the active-set arithmetic said (32 imminent rows are half of ARC's 64-token generation, so it books the
least). That confirms the per-token, per-step selection genuinely finds the inactive bulk and narrows the
active query rows hard on a pure-diffusion rollout. Throughput behaved as I feared and the split is telling:
20.97 on MATH (up a token from the floor's 19.54), 19.84 on HumanEval (up sharply from 11.46, a 73% rise
because the long HumanEval forward is where narrowing rows finally out-saves the eager-attention tax), but
22.66 on ARC — *down* from the floor's 35.72, a 37% loss, exactly the outcome I flagged: on a 64-token
generation the eager-attention and rollout overhead is a fixed per-step tax that the tiny forward cannot
out-save. So the active-row scheme buys reuse but not speed, and on the shortest workload it costs speed.

But the verdict is the quality column, and it is worse than a single-workload failure. MATH collapsed from the
floor's 38.4 to 25.0 — a drop of 13.4 points, 35% of the score gone, landing 10 full points *under* the 35
gate. And HumanEval, which I expected to "hold near 43.9," did not hold: it fell to 40.2439, a 3.66-point drop
that left it clearing its 40 gate by only 0.24. That is the part I under-weighted. The density proxy did not
bleed one fragile workload; it bled *two*, and only the tightest of margins kept HumanEval from gating as well.
ARC alone was untouched — 84.215, actually up 0.26 from the floor's 83.959, which at ARC's single-letter,
64-token scale is noise, not a real gain. So the pattern is not "MATH is special"; it is "the longer and more
sequential the committed answer, the more the structural proxy bleeds," with MATH (256-token exact-match chains)
worst, HumanEval (512-token but execution-checked, locally structured code) badly dented, and ARC (a single
committed letter) immune. And the scoring consequence of MATH gating is exactly the trap I described at the
floor: the near-lossless gate multiplies a workload 10 points under threshold down toward zero, and under a
geometric mean across three workloads one near-zero factor drags the whole cube-root product down, so d2Cache —
with the best reuse on the board — will score *below* a policy that books less reuse but keeps every gate. Top
reuse with a broken gate loses to moderate reuse with all gates intact. That sentence is now written in the
data on two workloads, not one.

So I have to reread *why* the long answers broke while ARC survived, because the fix has to attack that cause
and nothing else. A committed answer on MATH is long, strictly sequential, and unforgiving: a single wrong
committed digit ruins the exact-match, and committed tokens are frozen for the rest of the rollout. The
active-row scheme reuses the cached KV of any token its `D·s` selector underrates — and `D·s` is a *structural*
proxy for imminence, a measure of whether the neighborhood *looks* resolved, not a measurement of whether the
token's representation actually moved. On a long sequential answer, a token whose neighborhood looks sparse (low
`D`) can still have a feature that shifted because of a *distant* commit that the bidirectional attention
propagated — the same everyone-moves-when-anyone-reveals mechanism I traced at the floor, now biting from far
away — and the density selector will happily leave it on stale cache. That stale feature flips a digit, the
commit freezes, and MATH falls under the gate. HumanEval sees a weaker version of the same thing (longer
sequences than ARC, so more distant-commit drift, but execution tolerates local reformattings) and dents
without quite gating; ARC has 64 tokens and almost nothing to get stale, so it is immune. So the failure is not
"reuse is bad"; it is "reuse driven by a structural imminence proxy misses drift from distant commits, and that
drift is lethal on exactly the long-answer workloads." Two things have to change together: drive the refresh
decision off a signal tied to whether the *feature actually changed*, and stop the aggressive query narrowing
that threw away the exact rollout's accuracy in the first place.

Let me walk the options rather than jump. The first is to keep d2Cache's active-row narrowing and bolt on a
quality guard — refresh more where the gate is tight, e.g. a bigger `current_k` on MATH. I kill this on two
counts. The task pins one predeclared policy across all workloads with no per-benchmark search, so "refresh more
on MATH" is not a legal knob; and even if it were, it treats the symptom — the root cause is that the *selector*
is structural, so a bigger budget just refreshes more structurally-chosen tokens and still misses the
distant-commit drift, which by construction lives in low-density tokens the budget deprioritizes. The second
option is to run all queries (protecting every prediction) and reuse only internal features on a pure *clock* —
a segment-and-interval cache with no per-token top-up. This fixes the query-narrowing quality loss but reintroduces
the granularity sin: one interval per segment refreshes tokens that already settled and reuses tokens that are
actively moving, so between clock ticks the moving tokens go stale — the same failure mode, milder. It needs a
per-token signal for the between-refresh top-up, and the whole lesson of d2Cache is that the signal must be a
*measurement*, not a proxy. The third option keeps the all-queries safety of the second and replaces the proxy
with a measured-change top-up. That is the one that attacks both causes at once, so I build it.

Take the safety piece first, because it is the direct undo of d2Cache's aggression. Keep the **full top-level
rollout** — forward the whole sequence each step rather than narrowing to active query rows — so the model's
prediction at every position is computed from a complete, current input, and the only thing I cache is the
*internal features* I am entitled to skip. This is the structural difference from d2Cache: d2Cache narrows
*which queries run*; here I run all queries but reuse cached intermediate features (keys, values, attention
output, feed-forward output) for the positions whose features did not move. Running all queries is strictly more
compute than d2Cache's active-row scheme, so I already know the arithmetic cost: I will book *less* raw reuse
than d2Cache's 0.81–0.90, because reuse now comes only from skipped features and not from skipped query rows.
That is a deliberate trade — I am spending reuse to buy back the gate-critical accuracy — and I should expect
the reuse_ratio column to fall visibly from d2Cache's, which is the *point*, not a regression.

Now the redundancy I exploit is the same two-part structure as before but read at the segment-plus-row level the
diagnostic actually supports. The prompt region is quasi-static: prompt tokens are never masked, never change
their ids, and their internal features drift only as the response fills in around them — the second-order drift I
sized at the floor, where adjacent-step feature similarity in the prompt sits near one across many steps. So
cache the prompt and refresh it on a *long* interval `K_p`. How long can `K_p` be? The prompt's staleness after
`K_p` steps is set by how much of its context changed, and on a one-token-per-step rollout `K_p` steps commit
about `K_p` new response tokens, so `K_p = 50` means the prompt features are at most "50 committed response
tokens out of date" before I refresh them — on MATH that is about a fifth of the 256-token response, a tolerable
drift for a block whose own self-attention dominates its representation. The payoff is stark: I recompute the
prompt features about `256/50 ≈ 5` times over a MATH rollout instead of 256 times, roughly a 98% cut in
prompt-forward work, for a second-order error I have argued is small. The same `K_p = 50` scales sensibly across
the other two schedules without any per-workload tuning: HumanEval's 512 steps give about 10 prompt refreshes,
and ARC's 64 steps fall to about 1–2 (step 0 plus one near step 50), so on the short workload the prompt cache is
nearly free-standing — which is fine, because ARC's prompt is a short multiple-choice stem where prompt-forward
savings were never the main prize; there the response interval `K_r = 3` is doing the real work. The response is
not uniform: most response
tokens are also nearly frozen step-to-step, but a small minority genuinely move. So refresh the *whole* response
cache on a *short* interval `K_r` to bound how stale anything gets, and between those full refreshes recompute
only the small set of response tokens that actually changed. The short interval is workload-specific because the
workloads differ in how fast their responses churn — `gen_refresh_interval` of 7 for MATH, 8 for HumanEval, 3 for
ARC — a shorter interval where the sequence is shortest and a full refresh is cheapest (ARC's 64 tokens), longer
where there is more stable bulk to coast on (HumanEval's 512).

Third — and this is what should rescue the long workloads — the between-refresh partial update has to decide
*which* response tokens moved, and unlike d2Cache's structural density proxy I want a signal tied to the feature
itself. The trap is the same circularity as before: to know a token's expensive attention/feed-forward output
changed I would have to compute it, which is the thing I am trying to skip. So I need a *cheap* feature whose
adjacent-step change predicts the expensive downstream change. The Value projection is exactly that. It is cheap
— a single matmul `W_V · h`, no attention, no feed-forward — so computing it for the response costs `O(N d²)`
against the full block's `O(N² d + N d_ffn d)`, skipping both the quadratic attention term and the large MLP
term. Put the 8B stack's dimensions on it: a block does roughly four projection matmuls of size `d² ≈ 4096²`
(Q, K, V, O) plus a feed-forward of order `3·d·d_ffn` with `d_ffn` several times `d`, so on the order of a dozen
`d²`-scale matmuls per token, of which the V projection is one. Measuring the V-similarity for the whole response
therefore costs on the order of a twelfth of a full block per token, and it decides which quarter of the response
gets the other eleven-twelfths — so the accounting closes: I spend a small, fixed probe to avoid a large,
unnecessary recompute on three-quarters of the response. And it is *what the attention reads out*: a position's attention output is a weighted sum of Values, so if
a token's own Value barely changed and its neighborhood's Values barely changed, its attention output and the
feed-forward on top of it cannot have changed much. So compute the response Values cheaply, take the *cosine*
similarity of each token's current Value to its cached Value, and recompute the `transfer_ratio = 0.25` fraction
with the *lowest* similarity — the most-changed tokens — scattering their fresh keys, attention, and feed-forward
outputs back into the cache and reusing the rest. Cosine and not L2 because I care about directional/semantic
change: a Value can grow or shrink in norm as a token's confidence sharpens without its meaning moving, and I do
not want to spend a refresh on a magnitude wiggle; the scale-free cosine also makes the similarity comparable
across tokens with very different Value norms, so a single `transfer_ratio` threshold is meaningful across the
whole response. This is the `lowest_value_feature_similarity` row selector with `scatter_refresh` KV update. The
crucial contrast with d2Cache: its `D·s` asks "is this token structurally about to be decoded?", a proxy that
misses distant-commit drift; the V-verify asks "did this token's Value actually move?", which catches drift from
*any* cause, including the distant commits that flipped MATH digits. The refresh is driven by measured change,
not predicted imminence, and that is the quality safeguard.

Let me pin the reuse this produces from arithmetic, because it doubles as the sharpest falsifiable prediction I
have. Over a cycle of `K_r` steps, one step is a full response refresh (refresh fraction 1.0) and the other
`K_r − 1` steps recompute `transfer_ratio = 0.25` of the response, so the average response-refresh fraction is
`[1·1.0 + (K_r − 1)·0.25] / K_r` and reuse is one minus that. For MATH, `K_r = 7`: `[1 + 6·0.25]/7 = 2.5/7 =
0.357`, so reuse ≈ `0.643`. For HumanEval, `K_r = 8`: `[1 + 7·0.25]/8 = 2.75/8 = 0.344`, reuse ≈ `0.656`. For
ARC, `K_r = 3`: `[1 + 2·0.25]/3 = 1.5/3 = 0.500`, reuse ≈ `0.500`. So I predict reuse_ratio around `0.64 / 0.66
/ 0.50` — clearly above the floor's 0 and clearly *below* d2Cache's `0.81 / 0.90 / 0.81`, with the smallest
value on ARC because its short interval `K_r = 3` forces the most frequent full refreshes. The reuse_ratio column
will tell me whether the interval-plus-quarter model is the right description of what the harness actually reuses;
if the measured reuse comes in far from these, my mental model of the refresh accounting is wrong and I need to
reread the substrate before trusting the quality story that rides on it.

Now land it in this task's hooks, noting what the harness owns and what differs from the generic method. I do
not write the per-layer four-case block forward, the RoPE indexing for scattered rows, or the prompt/response
feature split — the harness implements all of that; my `DLMRefreshPolicy` declares the plan. The block schedule
keeps the workload defaults unchanged (`block_length = 32`, the workload's `num_steps`), because here I am *not*
narrowing queries and I keep the standard semi-autoregressive block decoding — the interval feature cache rides
on top of the normal block-wise rollout, it does not replace it, and I already argued at the floor that the
native blocking is part of the quality reference I must not disturb. The query plan stays `full_sequence` every
step — that is the deliberate conservatism, the direct undo of d2Cache. The cache-refresh plan turns the feature
cache on (`use_feature_cache = True`), sets `prompt_refresh_interval = 50` and the per-workload
`gen_refresh_interval`, `transfer_ratio = 0.25`, `row_selector = "lowest_value_feature_similarity"`, `kv_update =
"scatter_refresh"`, no layer reset. No attention probes are needed (`need_attention_weights = False`): unlike
d2Cache, the selection here is a cheap Value-similarity test, not an attention-rollout importance, so I avoid the
eager-attention overhead that cost d2Cache its ARC throughput — which is a second, independent reason this should
be faster per step than d2Cache despite running all queries, and specifically why ARC should not crater the way
it did. Transfer stays low-confidence over the current block; `after_step` is identity because the harness
maintains the feature cache internally across steps (the full policy is in the answer). One thing this task
exposes that the generic method tunes per dataset is the interval choice: the generic method lists different
`(K_p, K_r)` for GSM8K vs HumanEval vs MMLU, but the task pins one predeclared policy used across all workloads,
so I fix `K_p = 50` and bake the per-workload `K_r` into presets (7/8/3) rather than searching per benchmark —
the harness reads the workload from `step_meta` to pick the preset.

Let me sanity-check the design at its limits before committing, because `transfer_ratio = 0.25` is a chosen point
between two rejected extremes. Push `transfer_ratio → 1.0`: every non-refresh step recomputes the entire
response, so the policy degenerates to a full refresh every step with reuse coming only from the prompt interval
— essentially the floor plus prompt caching, safe but slow, no better than option two at its worst. Push
`transfer_ratio → 0`: between clock ticks nothing is recomputed, so the policy degenerates to the pure
segment-clock cache I rejected, where moving tokens go stale until the next tick — the granularity sin returns.
So `0.25` sits between "recompute everything (safe, no reuse)" and "recompute nothing (reuse, stale)," and the
V-similarity is what makes the middle *safe*: it spends the quarter-budget on precisely the tokens that moved, so
the stale-token failure of `transfer_ratio → 0` is avoided without paying the full recompute of
`transfer_ratio → 1`. That is the whole point of measuring rather than clocking the top-up, and the limits
confirm the middle is not arbitrary.

So the delta from d2Cache is concrete and aimed straight at its measured failure: stop narrowing the query rows
(run all queries so every prediction is computed from a complete input, protecting the gate-critical accuracy),
turn on an interval-based feature cache (long prompt interval, short per-workload response interval), and drive
the between-refresh recompute off the *measured* Value-similarity drift rather than a structural imminence proxy,
so the tokens whose features actually moved — including from distant commits — get refreshed. Reading d2Cache's
numbers, here is what I expect. Quality should recover across the board, and MATH specifically should climb back
*over* its 35 gate — that is the entire reason to trade away d2Cache's query narrowing; I expect MATH back into
the mid-30s, comfortably over 35, HumanEval back up off its knife-edge 40.24 toward or above the floor's 43.9,
and ARC holding near 84. Reuse_ratio should be the `0.64 / 0.66 / 0.50` my interval arithmetic predicts — lower
than d2Cache because I run all queries and only skip features, which is the price of safety. Throughput should be
similar to d2Cache or a touch better per step on the long workloads (no eager-attention overhead), and on ARC it
should recover above d2Cache's 22.66 — the eager tax is gone — though because I still run all queries the per-step
forward stays full-cost, so I do *not* expect it to reclaim the floor's 35.72; it should sit between. The net: I
expect this policy to *outrank* d2Cache on the task score not by winning reuse — it loses reuse — but by keeping
every quality gate intact, so no workload is multiplied down and the geometric mean stays whole. Make that
mechanical, because "gates matter" should be arithmetic, not a slogan. d2Cache's MATH is 10 points under the
gate, so its math workload eats a near-lossless-gate penalty that drives that factor close to zero; under the
geometric mean the whole score is multiplied by the cube root of that penalty, and the cube root of a number
near, say, a few hundredths is still only about a third — a three-fold haircut on the entire policy score that
d2Cache cannot escape while MATH gates. dLLM-Cache pays no such haircut: all three factors are intact. The
question is whether my lower reuse gives it all back, and it does not — the reuse gap is d2Cache's ~0.85 against
my predicted ~0.6, a ratio of about 0.7 inside the 0.75-weighted efficiency term, and once cube-rooted across
three workloads that is a mild discount, nowhere near the three-fold haircut of a gated workload. So an intact
gmean at reuse ~0.6 should beat a cratered gmean at reuse ~0.85, which is the whole thesis of this step. The thing I will
be watching is whether the modest reuse and still-modest throughput leave a large gap to the ceiling: if all
three workloads pass their gates but the efficiency terms are only middling — reuse in the 0.5–0.66 band,
throughput still near the floor everywhere — the diagnosis for the next step is already forming, because I have
been refreshing on a *clock* and refreshing *every layer* of every refreshed region, and running all queries. The
real headroom none of these policies has touched is the throughput term, and it lives in refreshing only *when*
the state actually moved and only the *layers* that moved, and in not forwarding the distant masked tokens at all.
