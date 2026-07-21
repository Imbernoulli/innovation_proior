The d2Cache run did what I most worried it would. Reuse jumped exactly as predicted — 0.8655 MATH, 0.8976
HumanEval, 0.8102 ARC, the highest I have seen, lowest on ARC just as the active-set arithmetic said (32
rows are half of ARC's 64-token generation). That confirms the per-token narrowing genuinely finds the
inactive bulk. Throughput behaved as I feared, and the split is telling: 20.97 MATH (up a token from
19.54), 19.84 HumanEval (up sharply from 11.46 — the long forward is where narrowing rows finally out-saves
the eager-attention tax), but 22.66 ARC, *down* from 35.72, a 37% loss, exactly the outcome I flagged — on a
64-token generation the eager-attention overhead is a fixed tax the tiny forward cannot out-save. So the
scheme buys reuse but not speed, and on the shortest workload it costs speed.

The verdict is the quality column, and it is worse than one workload failing. MATH collapsed from 38.4 to
25.0 — 10 points under the 35 gate. And HumanEval, which I expected to hold near 43.9, did not: it fell to
40.24, clearing its 40 gate by 0.24. That is what I under-weighted — the density proxy bled *two* workloads,
and only the tightest margin kept HumanEval from gating. ARC alone was untouched at 84.22 (up 0.26, noise at
its single-letter scale). So the pattern is not "MATH is special"; it is "the longer and more sequential the
committed answer, the more the structural proxy bleeds": MATH (256-token exact-match chains) worst,
HumanEval (512-token but execution-checked, locally structured) dented, ARC (a single letter) immune. And
the scoring consequence is the trap I described at the floor — the near-lossless gate multiplies a workload
10 points under threshold toward zero, and under a geometric mean one near-zero factor drags the whole
product down, so d2Cache with the best reuse on the board scores *below* a policy that books less reuse but
keeps every gate.

Reread *why* the long answers broke while ARC survived, because the fix must attack that cause. `D·s` is a
*structural* proxy — a measure of whether a neighborhood *looks* resolved, not of whether the representation
actually moved. On a long sequential answer a token whose neighborhood looks sparse (low `D`) can still have
a feature shifted by a *distant* commit that bidirectional attention propagated, and the density selector
leaves it on stale cache; the stale feature flips a digit, the commit freezes, MATH falls under the gate.
HumanEval sees a weaker version (longer than ARC, more distant-commit drift, but execution tolerates local
reformatting); ARC's 64 tokens have almost nothing to get stale. So the failure is not "reuse is bad," it is
"reuse driven by a structural imminence proxy misses drift from distant commits, and that drift is lethal on
the long-answer workloads." Two things must change together: drive the refresh off a signal tied to whether
the *feature actually changed*, and stop the aggressive query narrowing that threw away the exact rollout's
accuracy.

Walk the options. Keeping d2Cache's narrowing and bolting on a quality guard — a bigger `current_k` on
MATH — is dead twice over: the task pins one predeclared policy with no per-benchmark search, and even if
allowed, a bigger budget refreshes more *structurally-chosen* tokens and still misses the distant-commit
drift, which by construction lives in the low-density tokens the budget deprioritizes. Running all queries
and reusing only internal features on a pure clock fixes the query-narrowing loss but reintroduces the
granularity sin — one interval per segment refreshes settled tokens and reuses moving ones, so between ticks
the moving tokens go stale, a milder version of the same failure. It needs a per-token top-up, and the
lesson of d2Cache is that the top-up signal must be a *measurement*, not a proxy. The third option keeps the
all-queries safety and replaces the proxy with a measured-change top-up; that attacks both causes, so I
build it.

Take the safety piece first — the direct undo of d2Cache's aggression. Keep the **full top-level rollout**:
forward the whole sequence each step so every prediction is computed from a complete, current input, and
cache only the *internal features* I am entitled to skip. This is the structural difference from d2Cache,
which narrowed *which queries run*; here I run all queries but reuse cached intermediate features (keys,
values, attention output, feed-forward output) for positions whose features did not move. Running all
queries is strictly more compute than d2Cache's active-row scheme, so I already know I will book *less* raw
reuse than d2Cache's 0.81–0.90 — reuse now comes only from skipped features, not skipped rows. That is the
deliberate trade: I spend reuse to buy back the gate-critical accuracy, and I should expect the reuse column
to fall visibly from d2Cache's, which is the point, not a regression.

The redundancy I exploit is the same two-part structure, read at the segment-plus-row level. The prompt
region is quasi-static — prompt tokens are never masked, never change ids, and their features drift only as
the response fills, the second-order drift I sized at the floor. So cache the prompt and refresh it on a
*long* interval `K_p`. On a one-token-per-step rollout `K_p` steps commit about `K_p` new response tokens,
so `K_p = 50` means the prompt features are at most "50 committed tokens out of date" before refresh — on
MATH about a fifth of the 256-token response, tolerable for a block whose self-attention dominates. The
payoff: I recompute the prompt features about `256/50 ≈ 5` times over a MATH rollout instead of 256, roughly
a 98% cut in prompt-forward work for a second-order error. The same `K_p = 50` scales with no per-workload
tuning: HumanEval's 512 steps give ~10 refreshes, ARC's 64 give ~1–2, so on the short workload the prompt
cache is nearly free-standing — fine, because ARC's prompt is a short multiple-choice stem where prompt
savings were never the prize; there the response interval does the work. The response is not uniform: most
response tokens are also nearly frozen step-to-step, but a small minority move. So refresh the *whole*
response cache on a *short* interval `K_r` to bound staleness, and between full refreshes recompute only the
response tokens that actually changed. `K_r` is workload-specific because the responses churn at different
rates — 7 for MATH, 8 for HumanEval, 3 for ARC — shorter where the sequence is shortest and a full refresh
is cheapest (ARC), longer where there is more stable bulk to coast on (HumanEval).

Third, and this is what rescues the long workloads: the between-refresh partial update has to decide *which*
response tokens moved, and unlike d2Cache's structural proxy I want a signal tied to the feature itself. The
trap is the same circularity — to know a token's expensive attention/feed-forward output changed I would
have to compute it, the thing I am skipping. So I need a *cheap* feature whose adjacent-step change predicts
the expensive downstream change. The Value projection is exactly that. It is cheap — a single matmul
`W_V·h`, no attention, no feed-forward — costing `O(N d²)` against the full block's `O(N² d + N d_ffn d)`,
skipping both the quadratic attention term and the large MLP term. On the 8B stack a block is roughly a
dozen `d²`-scale matmuls per token (Q, K, V, O plus a feed-forward several times `d`), of which the V
projection is one, so measuring V-similarity for the whole response costs about a twelfth of a block per
token and decides which quarter gets the other eleven-twelfths — the accounting closes. And the Value is
*what attention reads out*: a position's attention output is a weighted sum of Values, so if a token's own
Value and its neighborhood's Values barely changed, its attention output and the feed-forward on top cannot
have changed much. So compute the response Values cheaply, take the *cosine* similarity of each token's
current Value to its cached Value, and recompute the `transfer_ratio = 0.25` fraction with the *lowest*
similarity — the most-changed tokens — scattering their fresh keys, attention, and feed-forward back into
the cache and reusing the rest. Cosine not L2 because I care about directional change: a Value can grow or
shrink in norm as confidence sharpens without its meaning moving, and the scale-free cosine also makes a
single `transfer_ratio` threshold meaningful across tokens with very different Value norms. This is the
`lowest_value_feature_similarity` selector with `scatter_refresh` update. The contrast with d2Cache is the
whole point: its `D·s` asks "is this token structurally about to be decoded?", a proxy that misses
distant-commit drift; the V-verify asks "did this token's Value actually move?", catching drift from any
cause, including the distant commits that flipped MATH digits.

Pin the reuse from arithmetic, because it doubles as the sharpest falsifiable prediction. Over a `K_r`-step
cycle one step is a full response refresh (fraction 1.0) and `K_r − 1` steps recompute 0.25, so the average
response-refresh fraction is `[1 + (K_r−1)·0.25]/K_r`. MATH `K_r = 7`: `2.5/7 = 0.357`, reuse ≈ 0.643.
HumanEval `K_r = 8`: `2.75/8 = 0.344`, reuse ≈ 0.656. ARC `K_r = 3`: `1.5/3 = 0.5`, reuse ≈ 0.5. So I
predict reuse around 0.64 / 0.66 / 0.50 — clearly above the floor's 0 and clearly *below* d2Cache's 0.81 /
0.90 / 0.81, smallest on ARC because its short interval forces the most frequent full refreshes. If the
measured reuse comes in far from these, my model of the refresh accounting is wrong and I need to reread the
substrate before trusting the quality story that rides on it.

Land it in the hooks, against d2Cache's active-row plan. The block schedule keeps the workload defaults
(`block_length = 32`, native `num_steps`), because here I am *not* narrowing queries and I keep the standard
semi-autoregressive blocking — the interval feature cache rides on top of the normal rollout, and I argued
at the floor the native blocking is part of the quality reference. The query plan stays `full_sequence`
every step, the deliberate undo of d2Cache. The cache-refresh plan turns the feature cache on
(`use_feature_cache = True`), `prompt_refresh_interval = 50` and the per-workload `gen_refresh_interval`,
`transfer_ratio = 0.25`, `row_selector = "lowest_value_feature_similarity"`, `kv_update = "scatter_refresh"`,
no layer reset. No attention probes (`need_attention_weights = False`): the selection is a cheap
Value-similarity test, not an attention rollout, so I avoid the eager-attention overhead that cost d2Cache
its ARC throughput — a second, independent reason this should be faster per step than d2Cache despite
running all queries, and specifically why ARC should not crater. Transfer stays low-confidence over the
current block; `after_step` is identity because the harness maintains the feature cache internally (the full
policy is in the answer). The generic method tunes `(K_p, K_r)` per dataset; the task pins one predeclared
policy, so `K_p = 50` is fixed and the per-workload `K_r` (7/8/3) is baked into presets the harness reads
from `step_meta`. `transfer_ratio = 0.25` sits between two rejected extremes: at `→1` every non-refresh step
recomputes the whole response (safe, no reuse, the floor plus prompt caching), at `→0` nothing is recomputed
between ticks (the pure segment-clock cache, moving tokens go stale) — and the V-similarity is what makes the
middle safe, spending the quarter-budget on precisely the tokens that moved.

So the delta from d2Cache is aimed at its measured failure: stop narrowing queries (protect the
gate-critical accuracy), turn on an interval feature cache, and drive the between-refresh recompute off
*measured* Value drift rather than a structural proxy. Reading d2Cache's numbers, I expect quality to
recover across the board — MATH specifically climbing back *over* its 35 gate into the mid-30s (the entire
reason to trade away the query narrowing), HumanEval off its knife-edge 40.24 toward or above 43.9, ARC
holding near 84 — and reuse at the 0.64 / 0.66 / 0.50 the interval arithmetic predicts, lower than d2Cache
because I run all queries and only skip features, the price of safety. Throughput should be similar to
d2Cache or a touch better per step on the long workloads (no eager tax), and on ARC recover above 22.66,
though since I still run all queries the per-step forward stays full-cost, so I do not expect it to reclaim
the floor's 35.72 — it should sit between. The net: this should *outrank* d2Cache not by winning reuse (it
loses reuse) but by keeping every gate intact, so no workload is multiplied down and the geometric mean
stays whole. The open worry is that the modest reuse and still-modest throughput leave a large gap to the
ceiling: if all three gates pass but the efficiency terms are only middling — reuse in the 0.5–0.66 band,
throughput near the floor everywhere — the next step's target is already forming, because I have been
refreshing on a *clock*, refreshing *every layer* of every refreshed region, and running all queries. The
untouched headroom is the throughput term, and it lives in refreshing only *when* the state moved and only
the *layers* that moved, and in not forwarding the distant masked tokens at all.
