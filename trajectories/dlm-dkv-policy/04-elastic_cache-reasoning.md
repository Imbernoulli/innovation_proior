The dLLM-Cache run validated the trade and then exposed the next ceiling. The trade worked as argued: MATH
recovered from d2Cache's 25.0 to 35.2 — back over the 35 gate by two tenths — HumanEval climbed from the
knife-edge 40.24 to 46.34, and ARC held at 84.04 (a 0.17 change, noise at its scale). All three gates are
intact and no workload is multiplied down, which is precisely why this outranks d2Cache: keeping the gates
whole beats winning reuse with one gate broken.

But read what it cost. Reuse fell to 0.6328 / 0.6562 / 0.4922 — above the floor's zero but well below
d2Cache's 0.8655 / 0.8976 / 0.8102, giving back the most on the short workload. That was the deliberate
price of running *all* queries and skipping only internal features. The interval arithmetic from last step
predicted it almost to the digit — `[1+6·0.25]/7 = 0.357`, `[1+7·0.25]/8 = 0.344`, `[1+2·0.25]/3 = 0.5`,
reuse ≈ 0.643 / 0.656 / 0.500 against the measured 0.6328 / 0.6562 / 0.4922. That match matters beyond
bookkeeping: it confirms my model of *how the harness accounts refresh work* is the right description of the
substrate, and the throughput story I am about to build rides on the same accounting.

The column I have to sit with is throughput, because no policy has broken it open. Against the floor's 19.54
/ 11.46 / 35.72, dLLM-Cache posts 20.84 / 15.58 / 25.95. On MATH that is +6.7%, essentially flat. On
HumanEval it is a real +36% — the long forward is where reusing internal features finally out-saves its own
overhead — yet still *below* d2Cache's 19.84, so even where dLLM sped up it did not lead. And on ARC it went
the wrong way: 25.95 against 35.72, a 27% loss, because the added machinery is pure tax on a 64-token
forward. So throughput inched up on the long workloads, *regressed* on the short one, and nowhere multiplied
— and the highest bar on the board, ARC's 35.72, was set by the uncached floor and then undercut by both
reuse policies. That is where the headroom is, and it will not come from a better clock.

Reread *why* the speed never came. dLLM-Cache refreshes on a **clock**, refreshes **every layer**, and runs
**all queries**. The intervals recompute on a fixed schedule whether or not anything changed; a fired
refresh recomputes *all layers* including ones that settled long ago; and all queries keep the per-step
forward full-cost regardless. Three factors sit at "full" — query count (all N), depth (all L on any
refresh), schedule (fire on a tick, not on need) — each a multiplicative floor on per-step compute, and
throughput is `tokens / (steps · per-step compute)`, so I have to cut one of these multiplicatively, not
shave it.

Walk the options. Tightening `K_r` to book more reuse is dead on arithmetic: the response-refresh fraction
`[1 + (K_r−1)·0.25]/K_r` *rises* as `K_r` shrinks (0.5 at K_r=3, 0.344 at K_r=8), which is exactly why ARC
already has the lowest reuse, and it does not touch throughput at all. Going back to narrowing queries the
way d2Cache did *does* cut the per-step forward — the right axis — but d2Cache already ran that experiment to
25.0 on MATH because its `D·s` selector is a structural proxy that leaves tokens stale. So query narrowing
per se is not the sin; the *selector* is. If I narrow queries I need a set defined by the dynamics — which
tokens still genuinely interact — and I must not freeze anything the current prediction leans on. The third
option attacks all three full-cost factors at once: refresh only the *layers* that moved (depth), only
*when* the state moved (schedule), and forward only the tokens that still interact (queries). That hits
throughput multiplicatively three times over, so I build it, taking depth first because it is the largest
single lever.

Watch the model denoise and the step-to-step KV change is not uniform across depth — it is *larger in deeper
layers than shallow*. That matches the folklore (early layers lock onto local lexical structure and settle
fast, deep layers carry global semantics that keep shifting), and the architecture forces it: in a residual
block the hidden-state change between adjacent steps satisfies a recursion `Δ̄^{ℓ+1} ≤ λ_ℓ Δ̄^ℓ` with `λ_ℓ >
1` for an expressive layer, so the drift *bound* grows multiplicatively with depth. If each layer's factor
sits at even 1.1, then across ~32 layers the bound compounds as `1.1^32 ≈ 21`, so deep-layer KV can move an
order of magnitude more than shallow. Combined with the empirical fact that shallow decoded-token drift
vanishes while deep drift stays bounded away from zero, there is a boundary layer `ℓ*` separating settled
shallow from active deep. So when I refresh I should *not* recompute layer 1 — its cache has converged — I
should reuse the shallow layers and recompute only from `ℓ*` onward. If `ℓ*` sits near mid-stack that alone
is a near-2× cut to the refresh cost, one no prior policy made.

But I will not hand-pick `ℓ*` — a hand-set constant is the clock problem in a new hat, and `ℓ*` should
*move* (shallow early when much is changing, deep late), so it must fall out of the model's own signal at
runtime, which forces the *when* axis. I need a cheap, reliable test that the cached state has drifted
enough to refresh, and ideally the same test hands me the boundary. Comparing hidden states `H^ℓ` to last
step's is circular and noisy: while I reuse cache the `H` I carry is itself built from stale KV, so the
difference conflates genuine change with my own caching error, and that error compounds with depth exactly
where I need a clean read. I need a signal *upstream* of the KV change and unpolluted by my caching. The
source of the KV change is the bidirectional attention itself — when a newly decoded token suddenly receives
real attention it rewrites the attention output earlier tokens computed while it was masked, and that shifts
their hidden state, hence K and V. So the change in the **attention weights** is the *cause*, not the
symptom, and it tracks the KV change closely. Measure that.

Measuring every token's attention change every step would defeat the purpose, so I want one cheap probe
whose movement lower-bounds everyone else's. The observation: among decoded tokens, the *most-attended* one
has the *smallest* drift — for it to remain most-attended its lead over the runner-up must absorb its own
attention swing, which bounds its excess drift by `O(√d_k/(R_ℓ √N))`. Put the 8B numbers on it: `d = 4096`
over `H = 32` heads gives `d_k = 128`, `√d_k ≈ 11.3`; on a MATH sequence of several hundred positions `√N ≈
22`, so the excess-drift ceiling is roughly `0.5 / R_ℓ` — small once the attention lead `R_ℓ` is any healthy
`O(1)`. So the most-attended token is a genuinely conservative trigger: if even the most stable token's
pattern has broken, everything less stable has moved at least as much. The same arithmetic warns me where
the guarantee weakens — on ARC, `√N ≈ 10`, so the ceiling is about `1.1 / R_ℓ`, twice as loose; the probe
triggers more readily and books less reuse there, which is fine because ARC has almost nothing to coast on
anyway. So I track the single most-attended decoded token (`track_num = 1`), and at each layer in reuse mode
recompute just its attention *row* and compare to last step's by **cosine** similarity, scale-free because
the pattern matters, not its magnitude. If similarity at some layer drops below `γ`, the pattern there
broke: restore the cached full-sequence hidden at the *next* layer and full-recompute to the last layer; if
no layer drops below `γ`, keep reusing the whole cache. The same per-layer test answers both axes — the
first layer to break *is* `ℓ*`, adapting per step and per input. The single knob is `γ = 0.9`: higher
stricter, lower laxer.

Here is the piece that decides whether this is affordable, and it is exactly where d2Cache died on ARC. The
trigger needs the tracked token's attention row, i.e. eager attention weights — the same
`need_attention_weights = True` that cost d2Cache. But d2Cache computed the *full* `N×N` matrix every step,
an `O(N² d)` tax; I need only *one row*, the tracked token's query against all keys, `O(N d)` per layer,
`O(L N d)` total — a factor `N` cheaper than d2Cache's tax and a factor `d` cheaper than a full block's MLP
term. On ARC's 64-token generation the probe is on the order of a sixty-fourth of the attention work it
decides whether to skip, so the accounting closes with room, and the overhead that dragged ARC under the
floor is replaced by a probe I can afford every step even on the shortest workload. That is the mechanical
reason to expect ARC throughput to finally clear the floor.

One more axis the dynamics hand me: the distant masked tokens. They are barely attended — a length-bias
prior holding the sequence's shape, not informing the current prediction — so I should not forward them at
all. dLLM-Cache forwarded every query including these; a rigid block cache would freeze masked tokens at a
block edge the current prediction still leans on, the freezing sin I already refused. The fix that avoids
both is a **sliding window** of the leftmost masked positions, `window_length = 16`, moving through the
response as tokens commit: the nearby masks that genuinely attend to each other stay live, only the truly
distant ones get block-cached. Half of `block_length = 32` is a natural width — wide enough to hold the live
front, narrow enough to be a real cut. The response query rows I forward drop from all of them to `16/256 =
6.25%` on MATH (16×), `16/512 = 3.1%` on HumanEval (32×), and `16/64 = 25%` on ARC (4×). So the query lever
is enormous on the long workloads and modest on ARC — the opposite of where I expect the throughput gain
largest, which is worth holding: on the long workloads the win comes from the query cut, on the short one it
comes from the cheap probe removing the per-step overhead that dragged ARC under the floor. The set I
forward each step is this window plus the tracked-token rows — a small fraction of the sequence, the third
multiplicative cut.

Land it in the hooks, against dLLM-Cache's full-query feature-cache plan. The block schedule keeps
`block_length = 32` and adds `window_length = 16`. The query plan forwards `full_sequence` at step 0 to fill
the cache, then `tracked_window` on every later step, handing the harness the `track_positions` and the
masked window. The cache-refresh plan is where the departure is sharpest: `use_feature_cache = False`,
turning *off* dLLM-Cache's interval feature cache entirely — my reuse now comes from reusing KV for the
shallow layers and un-refreshed positions under the drift trigger, an orthogonal mechanism I do not want to
stack on the old one. With the feature cache off the clock knobs are inert (I leave `prompt_refresh_interval`
and `gen_refresh_interval` at 1, `transfer_ratio = 0.0` — the clock and fractional scatter are precisely
what I am replacing). Instead `row_selector = "tracked_tokens_and_masked_window"`, `kv_update =
"tracked_window_layer_reset"`, and crucially `layer_reset = "attention_similarity"` — the depth-aware reset
the harness drives off the per-layer cosine trigger. The attention-probe plan sets `need_attention_weights =
True` with `gamma = 0.9`, `track_num = 1`; the `True` is affordable only because `track_num = 1` makes it a
one-row probe. The token-transfer plan switches from low-confidence top-k to `mode = "confidence_threshold"`,
`threshold = 0.9`, scope the masked window, `num_transfer_tokens = 1`, force-one. This is the quality
counterpart to `γ`: low-confidence top-k commits the k most-confident masked tokens *regardless of absolute
confidence*, so it can freeze a token the model is only 40% sure of — exactly how a wrong committed digit
ruins MATH — whereas the 0.9 threshold refuses to commit under 90% while force-one keeps the rollout
progressing. It pairs with `γ` by construction: a confident stable commit barely moves the attention pattern
so the trigger does not fire and the cache is reused; an uncertain revision shifts the pattern, the trigger
fires, the cache refreshes — the same "is this settled?" question at the same 0.9 drives both commit and
refresh. `after_step` is identity; the harness maintains the per-layer cache, tracked positions, and
layer-reset boundary internally (the full policy is in the answer). One predeclared `γ = 0.9` and
`window_length = 16` across all workloads, no per-benchmark search. `γ = 0.9` sits between two rejected
extremes: `→1` fires at the first layer every step so `ℓ*` collapses to 1 and I full-recompute the whole
stack over the window (safe, no depth lever, slow); `→0` never fires so the KV goes arbitrarily stale (the
staleness that flipped MATH digits) — and the measured cosine is what makes the middle safe.

So the delta from dLLM-Cache is three multiplicative compute cuts aimed at the untouched throughput term:
forward only a sliding window of live masks plus a one-row probe; trigger refresh on measured attention
drift instead of a clock; and on refresh recompute only from the adaptively-found `ℓ*` onward. The scored
throughput term normalizes against the best baseline per workload, so the bars are MATH's d2Cache 20.97,
HumanEval's d2Cache 19.84, and ARC's *uncached floor* 35.72 — the one both reuse policies fell under. This
is the first policy to cut the per-step forward itself, so I expect tokens/s to jump multiplicatively and
clear all three, largest on ARC: it is the most overhead-bound (a tiny forward dominated by fixed per-step
cost) and least-churning (a single letter), so removing the per-step full refresh and replacing the full
eager matrix with a one-row probe should let most steps cost almost nothing, and a 64-token generation's
throughput is limited mostly by that per-step floor — so the multiple could be very large there, while the
long workloads still forward the window against a long key set and refresh more often, capping their
multiple lower. Reuse should recover toward d2Cache's high-reuse neighborhood on the long workloads (the
windowed forward plus shallow-layer reuse skips a lot), and on ARC — 64 tokens, a 16-window covering a
quarter of the response, the top-1 bound loosened by the short sequence — the windowing buys less and the
trigger fires more, so it may land nearer 0.5. I am honest that these reuse figures are softer than last
step's interval arithmetic: the exact number depends on how often the trigger fires and where `ℓ*` lands,
which I cannot compute a priori — the reuse column will tell me. The gates are the bar I must not break,
having watched d2Cache fall under MATH's: the conservative top-1 trigger and the 0.9 threshold exist
precisely to keep accuracy native, so I expect MATH back over its 35 gate, HumanEval comfortably over 40
near its dLLM-Cache level, and ARC near 84 — where I note ARC hovers right at its gate across the board
(83.96 / 84.22 / 84.04), so "near 84" means it must clear 84, and the single-letter answer's robustness is
what I am counting on. If that holds, this policy wins not on any single term but by being the only one to
move *all three* — reuse high on the long workloads, throughput multiplied everywhere and finally over the
ARC floor, gates intact — so its per-workload efficiency scores are high and the geometric mean clears every
prior rung.
