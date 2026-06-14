The dLLM-Cache run validated the trade I made and then exposed the next ceiling, both in the numbers. The
trade worked: MATH recovered from d2Cache's 25.0 to 35.2 — back over the 35 gate, by a hair — and HumanEval
climbed to 46.3 and ARC held at 84.0, so all three gates are intact and no workload is multiplied down. That
is exactly why this should and does outrank d2Cache on the task score: keeping the gates whole beats winning
reuse with one gate broken. But look at what it cost and where it leaves me. Reuse fell to 0.6328 / 0.6562 /
0.4922 — clearly above the floor's 0.0 but well below d2Cache's 0.81–0.90 — because I ran *all* queries every
step and only skipped internal features. And the throughput, the term no policy on this ladder has yet moved,
barely budged: 20.84 / 15.58 / 25.95 tokens/s, on HumanEval actually *slower* than the floor at the seq level
and nowhere near a multiplicative gain. So dLLM-Cache bought safety and moderate reuse but essentially no
speed. The geometric mean is middling because two of the three efficiency components — the throughput term
everywhere, and reuse on ARC — are leaving most of the headroom on the table.

Reread *why* the speed never came. I have been refreshing on a **clock** and refreshing **every layer**.
dLLM-Cache's prompt interval (50) and response interval (7/8/3) recompute on a fixed schedule whether or not
anything actually changed that step, and when an interval fires it recomputes *all layers* of the refreshed
region, paying full depth for layers that settled long ago. The V-verify is adaptive about *which response
tokens*, which is good, but it still runs every step and the refresh it triggers is full-depth. And running
all queries means the per-step forward is still essentially full-cost — that is why throughput never moved.
To finally win throughput I need the opposite on two axes at once: refresh *when the state actually moved*,
not on a clock, and refresh only the *layers* that moved, not the whole stack — and stop forwarding the
distant masked tokens that contribute nothing. Each of those is a multiplicative cut to the per-step
compute, which is what throughput rewards.

Take the depth axis first, because it is the largest lever. Watch the model denoise and the step-to-step KV
change is not uniform across depth: it is *larger in deeper layers than shallow ones*. That matches old
probing folklore — early layers lock onto local lexical structure and settle fast, deep layers carry global
semantics that keep shifting — but I can also see why the architecture forces it. In a residual block the
hidden-state change between adjacent steps satisfies a recursion `Δ̄^{ℓ+1} ≤ λ_ℓ Δ̄^ℓ` with `λ_ℓ > 1` for an
expressive layer, so the *bound* on drift grows multiplicatively with depth; combined with the empirical fact
that shallow decoded-token drift vanishes while deep drift stays bounded away from zero, there is a boundary
layer `ℓ*` separating settled shallow layers from active deep ones. The consequence for caching is direct:
when I do refresh, I should *not* recompute layer 1 — its cache has converged — I should reuse the shallow
layers and recompute only from `ℓ*` onward. That is a depth-aware partial refresh that none of the prior
policies did; dLLM-Cache and d2Cache treat every layer the same, paying full depth on every refresh. This is
where the throughput hides: skipping the shallow half of the stack on most refreshes is a near-2x cut to the
refresh cost.

But I will not hand-pick `ℓ*` — hand-set constants are the clock problem in another guise, and `ℓ*` should
move (shallow early in decoding when much is changing, deep late when things have settled). So `ℓ*` has to
fall out of the model's own signal at runtime, which forces the *when* axis: I need a cheap, reliable test
that the cached state has drifted enough to refresh, and ideally the same test produces the boundary. My
first instinct, comparing hidden states `H^ℓ` to last step's, is circular and noisy — while I am reusing
cache, the `H` I carry is itself built from stale KV, so the difference conflates genuine change with my own
caching error, which compounds with depth. I need a signal *upstream* of the KV change and not polluted by my
caching. The source of the KV change is the bidirectional attention itself: when a newly decoded token
suddenly receives real attention, it rewrites the attention output earlier tokens computed when it was still
masked, and *that* shifts their hidden state, hence their K and V. So the change in the **attention weights**
is the *cause*, not the symptom, and it tracks the KV change closely — measure that.

Measuring every token's attention change every step would defeat the purpose, so I need one cheap probe whose
movement lower-bounds everyone else's. The observation that gives it: among already-decoded tokens, the
*most-attended* one has the *smallest* drift, and a short argument shows why — for the most-attended token to
remain most-attended its lead over the runner-up must absorb its own attention swing, which bounds its excess
drift by `O(√d_k/(R_ℓ√N))`, vanishing for long sequences. So the most-attended token is a *conservative*
trigger: if even the most stable token's attention pattern has broken, everything less stable has moved at
least as much. Track it (`track_num = 1` — the theorem gives the conservative signal from top-1), and at each
layer in reuse mode recompute just the tracked token's attention row and compare it to last step's by
**cosine** similarity (scale-free, the pattern is what matters). If the similarity at some layer drops below a
threshold `γ`, the pattern at that layer broke: restore the cached full-sequence hidden at the *next* layer
and full-recompute from there to the last layer; if no layer ever drops below `γ`, keep reusing the whole
cache and pay almost nothing. The same per-layer test answers both axes — the first layer to break *is* `ℓ*`,
adapting per step and per input, and the refresh runs from `ℓ*+1` to `L`. The single knob is `γ = 0.9`:
higher is stricter (triggers more often), lower is laxer.

There is one more axis the dynamics hand me, the distant masked tokens. They are barely attended — a length-
bias prior holding the sequence's shape, not informing the current prediction — so I should not be forwarding
them at all. dLLM-Cache forwarded every query including these; a rigid block cache would freeze masked tokens
right at a block edge that the current prediction still leans on. The fix is a **sliding window** of the
leftmost masked positions, `window_length = 16`, that moves through the response as tokens commit: the nearby
masked tokens that genuinely attend to each other stay live and current, only the truly distant ones get
block-cached. So the set I forward each step is the window plus the tracked-token rows I need for the trigger
— a small fraction of the sequence, which is the third multiplicative cut and the one that finally moves
throughput, because the per-step forward is now over ~16 positions plus a probe, not the whole response.

Now land it in this task's hooks, against dLLM-Cache's full-query plan. The block schedule keeps
`block_length = 32` but adds `window_length = 16` — the sliding live region inside the block. The query plan
forwards `full_sequence` at step 0 to fill the cache, then `tracked_window` on every later step, handing the
harness the `track_positions` from `cache_state` and the masked window. The cache-refresh plan declares
`row_selector = "tracked_tokens_and_masked_window"` (forward the tracked tokens, the newly decoded tokens,
and the window), `kv_update = "tracked_window_layer_reset"`, and crucially `layer_reset =
"attention_similarity"` — this is the depth-aware reset the harness drives off the per-layer cosine trigger,
the mechanism no prior policy used. The attention-probe plan must set `need_attention_weights = True` (the
trigger needs the tracked token's attention row, so eager attention again) and carries `gamma = 0.9`,
`track_num = 1`. The token-transfer plan switches from low-confidence top-k to `mode =
"confidence_threshold"` with `threshold = 0.9`, scope the masked window, `num_transfer_tokens = 1`,
force-one: confidence-aware parallel decoding commits every windowed position whose top probability clears
0.9 (and at least one), which pairs naturally with `γ` — confident, stable predictions keep attention
patterns stable, uncertain revisions make the trigger fire. `after_step` is identity; the harness maintains
the per-layer cache, the tracked positions, and the per-step layer-reset boundary internally (the full policy
is in the answer). One thing this task keeps faithful to the method rather than tuning: the same single
predeclared `γ = 0.9` and `window_length = 16` across all three workloads — no per-benchmark search.

So the delta from dLLM-Cache is three multiplicative compute cuts aimed exactly at the throughput term it
left untouched: forward only a sliding window of live masked tokens plus a probe instead of all queries;
trigger refresh on a measured attention-drift signal instead of a clock; and on refresh, recompute only the
layers from the adaptively-found boundary `ℓ*` onward instead of the whole stack. Reading dLLM-Cache's
numbers, here is what I expect and the bar I have to clear. The headline should be **throughput**: this is the
first policy that cuts the per-step forward itself, so tokens/s should jump *multiplicatively*, not
incrementally — I would expect it well past the ~20/16/26 of dLLM-Cache, plausibly several-fold, and on the
short ARC workload where each forward is tiny relative to the per-step overhead the gain could be very large.
Reuse should also rise back toward d2Cache's neighborhood on the long workloads (the windowed forward plus
shallow-layer reuse skips a lot), so I expect reuse around 0.86–0.88 on MATH and HumanEval, recovering the
ground dLLM-Cache gave up — though on ARC, with only 64 tokens and a 16-window, the windowing buys less reuse
and it may land nearer 0.5. The quality gates are the bar I must not break, having watched d2Cache fall under
the MATH gate: the conservative trigger and the confidence-0.9 threshold exist precisely to keep accuracy at
the native level, so I expect MATH back near 36–37 (over its gate), HumanEval near 47, ARC near 84 — all
passing. If that holds, this policy wins the task not on any single term but by being the only one to move
*all three* — reuse high on the long workloads, throughput multiplied everywhere, gates intact — so its
per-workload efficiency scores are high and the geometric mean clears every prior rung. The number to beat is
dLLM-Cache's profile: gates passed, reuse ~0.49–0.66, throughput stuck near the floor; I expect to match the
gates, match or exceed the reuse on the long workloads, and finally break the throughput term wide open.
