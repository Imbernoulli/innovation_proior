The d2Cache run did what I most worried it would, and the numbers say it sharply. The reuse jumped exactly
as predicted — 0.8655 on MATH, 0.8976 on HumanEval, 0.8102 on ARC — the highest reuse I have seen, which
confirms the per-token, per-step selection is genuinely finding the inactive bulk of the sequence and
narrowing the active query rows hard on a pure-diffusion rollout. And throughput behaved as I feared: 20.97
/ 19.84 / 22.66 tokens/s, barely above the floor's 19.5 / 11.5 / 35.7 (and on ARC it actually *fell*, from
35.7 to 22.66, because the eager-attention requirement for the rollout plus the per-step `D·s` and rollout
matrix overhead cost more than the short 64-token generation saves). So the active-row scheme buys reuse but
not speed. But the real verdict is the one I flagged as the risk: MATH quality collapsed from the floor's
38.4 to **25.0**, far *under* the 35 gate. HumanEval held at 40.2 (just over its 40 gate) and ARC held at
84.2, so the bleed is MATH-specific — and that is the worst possible place for it, because the gate penalty
multiplies the whole MATH workload down toward zero, and since the task score is a geometric mean across all
three workloads, one gated workload drags the entire policy below a blander one. That is the lesson written
in the data: top reuse with a broken gate loses to moderate reuse with all gates intact.

So I have to reread *why* MATH broke while HumanEval and ARC survived. MATH answers are long, strictly
sequential, and unforgiving: a single wrong committed digit ruins the exact-match answer, and committed
tokens are frozen for the rest of the rollout. The active-row scheme reuses the cached KV of any token its
`D·s` density-times-confidence selector underrates — and density is a *structural* proxy for imminence, not a
measurement of whether the token's representation actually moved. On a long sequential answer, a token whose
neighborhood looks sparse (low density) can still have a feature that shifted because of a *distant* commit
that the bidirectional attention propagated, and the density selector will happily leave it on stale cache.
That stale feature flips a digit, the commit freezes, and MATH falls under the gate. HumanEval (code,
shorter committed spans that are checked by execution, more local structure) and ARC (single-letter answer,
64 tokens, almost nothing to get stale) tolerate the same staleness. So the failure is not "reuse is bad";
it is "reuse driven by a structural imminence proxy bleeds quality on exactly the workload with the tightest
gate." The fix is to keep aggressive reuse but drive the refresh decision off a signal tied to whether the
*feature actually changed*, and to be far more conservative about query narrowing so I do not throw away the
exact rollout's accuracy.

Let me design that. First, back off the aggression that cost MATH: keep the **full top-level rollout** —
forward the whole sequence each step rather than narrowing to active query rows — so the model's prediction
at every position is computed from a complete, current input, and the only thing I cache is the *internal
features* I am entitled to skip. This is the structural difference from d2Cache: d2Cache narrows *which
queries run*; here I run all queries but reuse cached intermediate features (keys, values, attention output,
feed-forward output) for the positions whose features did not move. Running all queries is more compute than
d2Cache's active-row scheme, so I expect *less* raw reuse, but it protects the prediction quality the gate
punishes.

Second, the redundancy I exploit is the same two-part structure as before but read at the segment-plus-row
level that the diagnostic actually supports. The prompt region is quasi-static: prompt tokens are never
masked, never change, and their internal features drift only as the response fills in around them — a slow,
second-order effect — so adjacent-step feature similarity in the prompt sits near one across many steps. So
cache the prompt and refresh it on a *long* interval `K_p`; because it is genuinely stable, `K_p` can be
large — I use `prompt_refresh_interval = 50`, recomputing the prompt features once and reusing them for
~50 steps, paying the prompt forward `K/50` times instead of `K` times, which on a long prompt is a large
saving for negligible error. The response is not uniform: most response tokens are also nearly frozen
step-to-step, but a small minority genuinely move. So refresh the *whole* response cache on a *short*
interval `K_r` to bound how stale anything gets, and between those full refreshes recompute only the small
set of response tokens that actually changed. The short interval is workload-specific because the workloads
differ in how fast their responses churn — `gen_refresh_interval` of 7 for MATH, 8 for HumanEval, 3 for ARC
— a shorter interval where the gate is tighter or the sequence shorter, longer where there is more stable
bulk to coast on.

Third — and this is what should rescue MATH — the between-refresh partial update has to decide *which*
response tokens moved, and unlike d2Cache's structural density proxy I want a signal tied to the feature
itself. The trap is circular: to know a token's expensive attention/feed-forward output changed I would have
to compute it, which is the thing I am trying to skip. So I need a *cheap* feature whose adjacent-step change
predicts the expensive downstream change. The Value projection is exactly that: it is cheap (a single
matmul, no attention, no feed-forward), and it is *what the attention reads out* — a position's attention
output is a weighted sum of Values, so if a token's own Value barely changed and its neighborhood barely
changed, its attention output and the feed-forward on top of it cannot have changed much. So compute the
response Values cheaply, take the *cosine* similarity of each token's current Value to its cached Value
(cosine, not L2, because I care about directional/semantic change, not magnitude), and recompute the
`transfer_ratio = 0.25` fraction with the *lowest* similarity — the most-changed tokens — scattering their
fresh keys, attention, and feed-forward outputs back into the cache and reusing the rest. This is the
`lowest_value_feature_similarity` row selector with `scatter_refresh` KV update. The crucial contrast with
d2Cache: its `D·s` asks "is this token structurally about to be decoded?", a proxy that misses
distant-commit-induced drift; the V-verify asks "did this token's value actually move?", which catches drift
from any cause, including the distant commits that flipped MATH digits. The refresh is driven by measured
change, not predicted imminence, and that is the quality safeguard.

Now land it in this task's hooks, noting what the harness owns and what differs from the generic method. I
do not write the per-layer four-case block forward, the RoPE indexing for scattered rows, or the
prompt/response feature split — the harness implements all of that; my `DLMRefreshPolicy` declares the plan.
The block schedule keeps the workload defaults unchanged (`block_length = 32`, the workload's `num_steps`),
because here I am *not* narrowing queries and I keep the standard semi-autoregressive block decoding — the
generic method's prompt/response interval cache rides on top of the normal block-wise rollout, it does not
replace it. The query plan stays `full_sequence` every step — that is the deliberate conservatism. The
cache-refresh plan turns the feature cache on (`use_feature_cache = True`), sets `prompt_refresh_interval =
50` and the per-workload `gen_refresh_interval`, `transfer_ratio = 0.25`, `row_selector =
"lowest_value_feature_similarity"`, `kv_update = "scatter_refresh"`, no layer reset. No attention probes are
needed (`need_attention_weights = False`): unlike d2Cache, the selection here is a cheap Value-similarity
test, not an attention-rollout importance, so I avoid the eager-attention overhead that hurt d2Cache's
throughput — which is a second reason this should be faster per step than d2Cache despite running all
queries. Transfer stays low-confidence over the current block; `after_step` is identity because the harness
maintains the feature cache internally across steps (the full policy is in the answer). One thing this task
exposes that the generic method tunes per dataset is the interval choice: the published method lists
different `(K_p, K_r)` for GSM8K vs HumanEval vs MMLU, but the task pins one predeclared policy used across
all workloads, so I fix `K_p = 50` and bake the per-workload `K_r` into the presets (7/8/3) rather than
searching per benchmark — the harness reads the workload from `step_meta` to pick the preset.

So the delta from d2Cache is concrete and aimed straight at its measured failure: stop narrowing the query
rows (run all queries so every prediction is computed from a complete input, protecting the gate-critical
accuracy), turn on an interval-based feature cache (long prompt interval, short per-workload response
interval), and drive the between-refresh recompute off the *measured* Value-similarity drift rather than a
structural imminence proxy, so the tokens whose features actually moved — including from distant commits —
get refreshed. Reading d2Cache's numbers, here is what I expect. Quality should recover across the board,
and MATH specifically should climb back *over* its 35 gate — that is the entire reason to trade away
d2Cache's query narrowing; I would expect MATH back into the mid-30s, comfortably above 35, with HumanEval
and ARC holding near the floor. Reuse_ratio should be *lower* than d2Cache's 0.81–0.90 because I run all
queries and only skip features — I would expect it in the 0.49–0.66 range, clearly above the floor's 0.0 but
visibly below d2Cache. Throughput should be similar to d2Cache or a touch better per step on the long
workloads (no eager-attention overhead), but on ARC it should not crater the way d2Cache's did, since the
short interval-3 cache still skips real work without the rollout cost. The net: I expect this policy to
*outrank* d2Cache on the task score not by winning reuse — it loses reuse — but by keeping every quality
gate intact, so no workload is multiplied down and the geometric mean stays whole. The thing I will be
watching is whether the modest reuse and still-modest throughput leave a large gap to the ceiling: if all
three workloads pass their gates but the efficiency terms are only middling, the diagnosis for the next step
is that I have been refreshing on a *clock* and refreshing *every layer*, and the real headroom — especially
the throughput term that none of these policies has touched — is in refreshing only *when* the state moved
and only the *layers* that moved.
