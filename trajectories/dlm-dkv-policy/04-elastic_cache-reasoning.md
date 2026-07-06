The dLLM-Cache run validated the trade I made and then exposed the next ceiling, both in the numbers, so
let me read all three columns slowly before I react. The trade worked exactly as I argued it would. MATH
recovered from d2Cache's 25.0 to 35.2 — a +10.2 swing, back over the 35 gate by two tenths of a point — and
HumanEval climbed from d2Cache's knife-edge 40.24 to 46.34 (+6.1), while ARC held at 84.04 against d2Cache's
84.22, a change of 0.17 that at ARC's single-letter, 64-token scale is noise, not motion. So all three gates
are intact and no workload is multiplied down, which is precisely why this outranks d2Cache on the task
score: keeping the gates whole beats winning reuse with one gate broken. That sentence, which I wrote as a
thesis two rungs ago, is now written in the data.

But read what it cost. Reuse fell to 0.6328 / 0.6562 / 0.4922, clearly above the floor's zero but well below
d2Cache's 0.8655 / 0.8976 / 0.8102 — I gave back 0.233 on MATH, 0.241 on HumanEval, and 0.318 on ARC, the
most on the short workload. That was the deliberate price: I ran *all* queries every step and only skipped
internal features, so reuse now comes from skipped features and not from skipped query rows, and the giveback
is the cost of buying the gate-critical accuracy. The interval arithmetic I did last step predicted this
almost to the digit — over a `K_r`-step cycle one full refresh plus `K_r−1` quarter-refreshes gives refresh
fractions `[1+6·0.25]/7 = 0.357`, `[1+7·0.25]/8 = 0.344`, `[1+2·0.25]/3 = 0.500`, i.e. reuse ≈ 0.643 / 0.656
/ 0.500 — against the measured 0.6328 / 0.6562 / 0.4922. That match matters beyond bookkeeping: it confirms my
mental model of *how the harness accounts refresh work* is the right description of the substrate, and the
whole throughput story I am about to build rides on the same accounting, so I now trust it.

The verdict I actually have to sit with is the throughput column, because that is the term no policy on this
ladder has broken open, and the honest reading is sharper than "it barely budged." Against the floor's 19.54
/ 11.46 / 35.72, dLLM-Cache posts 20.84 / 15.58 / 25.95. On MATH that is +6.7%, essentially flat. On HumanEval
it is 15.58 versus 11.46, a real +36% — the long forward is where reusing internal features finally out-saves
its own overhead — yet it is still *below* d2Cache's 19.84 on the same workload, so even the one place dLLM
sped up, it did not lead. And on ARC it went the wrong way: 25.95 against the floor's 35.72, a 27% *loss*. The
short workload is where the added machinery is pure tax — a 64-token forward is tiny next to the fixed
per-step overhead of the cache logic — so paying that overhead every step to save on a forward that was
already cheap loses. (The `refresh_ratio` says the same thing from the other side: 0.5078 on ARC means more
than half the generated-token cache work is recomputed every rollout, the `K_r = 3` short interval forcing
the most frequent full refreshes.) So the true picture is not "throughput never moved"; it is that throughput
inched up on the long workloads, *regressed* on the short one, and nowhere multiplied — and the highest bar on
the board, ARC's 35.72, is not just unbeaten but was set by the uncached floor and then *undercut* by both
reuse policies. That is where the headroom is, and it is not going to come from a better clock.

Reread *why* the speed never came, because the fix has to attack that cause and nothing else. dLLM-Cache
refreshes on a **clock** and refreshes **every layer**, and runs **all queries**. The prompt interval (50)
and response interval (7/8/3) recompute on a fixed schedule whether or not anything actually changed that
step; when an interval fires it recomputes *all layers* of the refreshed region, paying full depth for layers
that settled long ago; and running all queries keeps the per-step forward essentially full-cost regardless.
Three factors sit at "full": the query count (all N positions), the depth (all L layers on any refresh), and
the schedule (fire on a tick, not on need). Each is a multiplicative floor on per-step compute, and throughput
is exactly `tokens / (steps · per-step compute)`, so to move it I have to cut one of those factors
multiplicatively, not shave it. Before I commit to which, let me walk the options I actually have at this
point rather than jump to one.

The first option is to keep dLLM-Cache's interval feature cache and simply tighten `K_r` to book more reuse —
smaller intervals, more skipping. I kill this on arithmetic. A shorter `K_r` books *less* reuse, not more:
the response-refresh fraction is `[1 + (K_r−1)·0.25]/K_r`, which *rises* as `K_r` shrinks (at `K_r = 3` it is
0.5, at `K_r = 8` it is 0.344), which is exactly why ARC with `K_r = 3` already has the lowest reuse on the
board. And even if the sign went the other way, it would not touch throughput at all — the per-step forward is
still full over all queries and any fired refresh is still full depth. It treats the symptom (reuse magnitude)
and leaves the cause (per-step cost) untouched. Dead.

The second option is to go back to narrowing queries the way d2Cache did — that *does* cut the per-step
forward, which is the right axis. But d2Cache already ran this experiment and the result is 25.0 on MATH, ten
points under the gate, because its `D·s` selector is a *structural imminence proxy* that leaves a token on
stale cache whenever its neighborhood merely *looks* resolved, missing drift propagated from distant commits.
So query narrowing per se is not the sin; the *selector* is. If I narrow queries I need a set defined by the
dynamics — which tokens genuinely still interact — not by a structural resolvedness score, and I need to not
freeze anything the current prediction still leans on. That reframes the second option into a live design, but
only if paired with a principled window and a measured refresh trigger, so it folds into the third.

The third option attacks all three full-cost factors at once: refresh only the *layers* that moved (depth),
refresh only *when* the state moved (schedule), and forward only the tokens that still interact (queries).
This is the one that hits throughput multiplicatively three times over instead of shaving it, so I build it,
taking the depth axis first because it is the largest single lever.

Watch the model denoise and the step-to-step KV change is not uniform across depth: it is *larger in deeper
layers than shallow ones*. That matches the old probing folklore — early layers lock onto local lexical
structure and settle fast, deep layers carry global semantics that keep shifting — but I can also see why the
architecture forces it. In a residual block the hidden-state change between adjacent denoising steps satisfies
a recursion `Δ̄^{ℓ+1} ≤ λ_ℓ Δ̄^ℓ` with `λ_ℓ > 1` for an expressive layer, so the *bound* on drift grows
multiplicatively with depth. Put a number on that to feel the size: if each layer's factor sits even a little
above one, say 1.1, then across the 8B stack's roughly 32 layers the bound compounds as `1.1^{32} ≈ 21`, so
the deep-layer KV can move an order of magnitude more than the shallow. Combined with the empirical fact that
shallow decoded-token drift vanishes while deep drift stays bounded away from zero, there is a boundary layer
`ℓ*` separating settled shallow layers from active deep ones. The consequence for caching is direct: when I
do refresh, I should *not* recompute layer 1 — its cache has converged — I should reuse the shallow layers and
recompute only from `ℓ*` onward. If `ℓ*` sits near mid-stack on a typical refresh, that alone is a near-2×
cut to the refresh cost, and it is a cut none of the prior policies made: dLLM-Cache and d2Cache treat every
layer the same and pay full depth on every refresh. This is where a large slice of the throughput hides.

But I will not hand-pick `ℓ*`. A hand-set constant is the clock problem wearing a different hat, and `ℓ*`
should *move* — shallow early in decoding when much is changing, deep late when things have settled — so it
has to fall out of the model's own signal at runtime, which forces the *when* axis. I need a cheap, reliable
test that the cached state has drifted enough to refresh, and ideally the same test hands me the boundary. My
first instinct, comparing hidden states `H^ℓ` to last step's, is circular and noisy: while I am reusing cache,
the `H` I carry is itself built from stale KV, so the difference conflates genuine change with my own caching
error, and that error compounds with depth exactly where I most need a clean read. I need a signal *upstream*
of the KV change and not polluted by my caching. The source of the KV change is the bidirectional attention
itself — when a newly decoded token suddenly receives real attention, it rewrites the attention output that
earlier tokens computed while it was still masked, and *that* shifts their hidden state, hence their K and V.
So the change in the **attention weights** is the *cause*, not the symptom, and it tracks the KV change
closely. Measure that.

Measuring every token's attention change every step would defeat the purpose, so I want one cheap probe whose
movement lower-bounds everyone else's. The observation that gives it: among already-decoded tokens, the
*most-attended* one has the *smallest* drift. The short argument is that for the most-attended token to remain
most-attended, its lead over the runner-up must absorb its own attention swing, which bounds its excess drift
by `O(√d_k / (R_ℓ √N))`. Put the 8B numbers on that bound to see how conservative it actually is: with `d =
4096` over `H = 32` heads the head dimension is `d_k = 128`, so `√d_k ≈ 11.3`; on a MATH sequence of several
hundred positions `√N ≈ 22`, giving an excess-drift ceiling of roughly `0.5 / R_ℓ` — a small fraction once
the attention lead `R_ℓ` is any healthy `O(1)`. So the most-attended token is a genuinely *conservative*
trigger: if even the most stable token's attention pattern has broken, everything less stable has moved at
least as much. The same arithmetic warns me where the guarantee weakens — on ARC, with a total sequence
around a hundred positions, `√N ≈ 10`, so the ceiling is about `1.1 / R_ℓ`, roughly twice as loose as on
MATH. On the short workload the top-1 probe is a less conservative bound, so it will trigger more readily and
book less reuse there, which I should expect to show up in the ARC `reuse_ratio` and which is fine because
ARC has almost nothing to coast on anyway. So I track the single most-attended decoded token (`track_num = 1`
— the theorem gives the conservative signal from top-1 alone), and at each layer in reuse mode recompute just
the tracked token's attention *row* and compare it to last step's by **cosine** similarity, scale-free
because the pattern is what matters, not its magnitude. If the similarity at some layer drops below a
threshold `γ`, the pattern there broke: restore the cached full-sequence hidden at the *next* layer and
full-recompute from there to the last layer; if no layer ever drops below `γ`, keep reusing the whole cache
and pay almost nothing. The same per-layer test answers both axes — the first layer to break *is* `ℓ*`,
adapting per step and per input, and the refresh runs from `ℓ*+1` to `L`. The single knob is `γ = 0.9`:
higher is stricter and triggers more often, lower is laxer.

Here is the piece that decides whether this is even affordable, and it is exactly where d2Cache died on ARC.
The trigger needs the tracked token's attention row, which means eager attention weights — the same
`need_attention_weights = True` that cost d2Cache its ARC throughput. But d2Cache computed the *full* eager
attention, the whole `N×N` matrix, every step, an `O(N² d)` tax on top of the forward; I only need *one row*,
the tracked token's query against all keys, which is `O(N d_k)` per head, `O(N d)` per layer, `O(L N d)`
total. That is a factor `N` cheaper than d2Cache's attention tax and a factor `d` cheaper than the MLP term of
a full block. On ARC's 64-token generation the probe is on the order of a sixty-fourth of the attention work
it decides whether to skip — so the accounting closes with room to spare, and the very overhead that made
d2Cache and dLLM-Cache slower than the uncached floor on ARC is replaced by a probe I can afford to pay every
step even on the shortest workload. That is the mechanical reason to expect ARC throughput to finally clear
the floor rather than fall under it.

There is one more axis the dynamics hand me, the distant masked tokens. They are barely attended — a
length-bias prior holding the sequence's shape, not informing the current prediction — so I should not be
forwarding them at all. dLLM-Cache forwarded every query including these; a rigid block cache would freeze
masked tokens right at a block edge that the current prediction still leans on, which is the freezing sin I
already refused for the committed tokens. The fix that avoids both is a **sliding window** of the leftmost
masked positions, `window_length = 16`, moving through the response as tokens commit: the nearby masks that
genuinely attend to each other stay live and current, only the truly distant ones get block-cached. Half of
the `block_length = 32` is a natural width — wide enough that the live front and the tokens it interacts with
stay inside it, narrow enough to be a real cut. Size that cut per workload: the response query rows I forward
drop from all of them to `16/256 = 6.25%` on MATH (a 16× reduction), `16/512 = 3.1%` on HumanEval (32×), and
`16/64 = 25%` on ARC (only 4×). So the query lever is enormous on the long workloads and modest on ARC — the
exact opposite of where I expect the *throughput* gain to be largest, which is worth holding onto: on the long
workloads the win comes from the query cut, on the short workload it comes from the cheap-probe-plus-skipped-
refresh removing the per-step overhead that was dragging ARC under the floor. The set I forward each step is
this window plus the tracked-token rows the trigger needs — a small fraction of the sequence, the third
multiplicative cut and the one that, together with skipping distant masks, finally moves the per-step forward
off "full."

Now land it in this task's hooks, against dLLM-Cache's full-query, feature-cache plan. The block schedule
keeps `block_length = 32` but adds `window_length = 16`, the sliding live region inside the block. The query
plan forwards `full_sequence` at step 0 to fill the cache, then `tracked_window` on every later step, handing
the harness the `track_positions` from `cache_state` and the masked window. The cache-refresh plan is where
the departure is sharpest: I set `use_feature_cache = False`, turning *off* dLLM-Cache's interval feature
cache entirely — my reuse no longer comes from skipping internal features on a schedule, it comes from
reusing KV for the shallow layers and the un-refreshed positions under the drift trigger, an orthogonal
mechanism I do not want to stack on the old one. With the feature cache off, the `prompt_refresh_interval` and
`gen_refresh_interval` clock knobs are inert (I leave them at 1) and `transfer_ratio = 0.0` — the clock and
the fractional scatter are precisely what I am *replacing*, so they go quiet. Instead I declare `row_selector
= "tracked_tokens_and_masked_window"` (forward the tracked tokens, the newly decoded tokens, and the window),
`kv_update = "tracked_window_layer_reset"`, and crucially `layer_reset = "attention_similarity"` — the
depth-aware reset the harness drives off the per-layer cosine trigger, the mechanism no prior policy used. The
attention-probe plan sets `need_attention_weights = True` and carries `gamma = 0.9`, `track_num = 1`; the
`True` is affordable only because `track_num = 1` makes it a one-row probe, as I sized above. The
token-transfer plan switches from the floor's low-confidence top-k to `mode = "confidence_threshold"` with
`threshold = 0.9`, scope the masked window, `num_transfer_tokens = 1`, force-one. This is the quality
counterpart to `γ`. Low-confidence top-k commits the k most-confident masked tokens *regardless of absolute
confidence* — it can freeze a token the model is only 40% sure of, which is exactly how a wrong committed
digit ruins MATH's exact-match; the 0.9 threshold refuses to commit anything under 90% while force-one keeps
the rollout progressing, so it never freezes a shaky token and commits confident spans in parallel. And it
pairs with `γ` by construction: a confident, stable commit barely moves the attention pattern, so the trigger
does not fire and the cache is reused; an uncertain revision shifts the pattern, the trigger fires, and the
cache refreshes. The same "is this settled?" question, at the same 0.9, drives both the commit and the
refresh. `after_step` is identity; the harness maintains the per-layer cache, the tracked positions, and the
per-step layer-reset boundary internally (the full policy is in the answer). Faithful to the method, I keep
one predeclared `γ = 0.9` and `window_length = 16` across all three workloads — no per-benchmark search.

Before I commit, sanity-check `γ = 0.9` at its limits, because a threshold is only trustworthy if I know what
it degenerates to. Push `γ → 1` (strictest): the cosine trigger fires at the very first layer almost every
step, so `ℓ*` collapses to 1 and I full-recompute the whole stack over the window every step — safe, but I
have thrown away the depth lever and kept only the query window, roughly dLLM-Cache's safety confined to 16
positions, slow. Push `γ → 0` (laxest): the trigger never fires, `ℓ*` is never set, and I reuse the whole
cache forever — maximum reuse, but the KV goes arbitrarily stale as distant commits accumulate, and that is
the exact staleness that flipped MATH digits two rungs back, so quality craters. So `0.9` sits between
"refresh-always (safe, no speed)" and "refresh-never (fast, stale)," and the measured cosine is what makes the
middle *safe*: it spends the refresh precisely on the layers and steps where the attention actually broke,
avoiding the staleness of `γ → 0` without paying the full-depth every-step cost of `γ → 1`. The limits confirm
the middle is not arbitrary, which is the same shape of argument that justified `transfer_ratio = 0.25` last
step.

So the delta from dLLM-Cache is three multiplicative compute cuts aimed exactly at the throughput term it left
untouched: forward only a sliding window of live masks plus a one-row probe instead of all queries; trigger
refresh on a measured attention-drift signal instead of a clock; and on refresh recompute only the layers from
the adaptively-found `ℓ*` onward instead of the whole stack. Reading dLLM-Cache's numbers, here is what I
expect and, more usefully, the exact bars I have to clear. The scored throughput term normalizes against the
best baseline on each workload, so the bar is per-workload: MATH's best baseline throughput is d2Cache's
20.97, HumanEval's is d2Cache's 19.84, and ARC's is the *uncached floor's* 35.72 — the one both reuse policies
fell under. This is the first policy that cuts the per-step forward itself, so I expect tokens/s to jump
*multiplicatively*, clearing all three bars, and largest on ARC: ARC is the most overhead-bound workload (a
tiny forward dominated by fixed per-step cost) and the least-churning (a single committed letter), so removing
the per-step full refresh and replacing the full eager-attention matrix with a one-row probe should let most
steps cost almost nothing, and throughput on a 64-token generation is limited mostly by that per-step floor —
so the multiple could be very large there, whereas the long workloads still forward the window against a long
key set and refresh more often, capping their multiple lower. Reuse should rise back toward d2Cache's
neighborhood on the long workloads — the windowed forward plus shallow-layer reuse skips a lot — so I expect
roughly 0.86–0.88 on MATH and HumanEval, recovering the ground dLLM-Cache gave up; on ARC, with only 64 tokens,
a 16-window that covers a quarter of the response, and the top-1 bound loosened by the short sequence, the
windowing buys less and the trigger fires more, so it may land nearer 0.5. I will be honest that these reuse
figures are softer than last step's interval arithmetic gave me: the exact number depends on how often the
trigger fires and where `ℓ*` lands, which I cannot compute a priori — the `reuse_ratio` column will tell me,
and if it comes in far off, my picture of the trigger dynamics is wrong. The gates are the bar I must not
break, having watched d2Cache fall under MATH's: the conservative top-1 trigger and the confidence-0.9
threshold exist precisely to keep accuracy at the native level, so I expect MATH back near 36–37 (over its 35
gate), HumanEval near 47 (well over 40), and ARC near 84 — where I note ARC hovers right at its gate across the
board (83.96 uncached, 84.22 d2Cache, 84.04 dLLM), so "near 84" means it must clear 84, and the single-letter
answer's robustness is what I am counting on. If that holds, this policy wins the task not on any single term
but by being the only one to move *all three* — reuse high on the long workloads, throughput multiplied
everywhere and finally over the ARC floor, gates intact — so its per-workload efficiency scores are high and
the geometric mean clears every prior rung. The number to beat is dLLM-Cache's profile: gates passed, reuse
0.49–0.66, throughput inching up on the long workloads and *under the floor* on ARC. I expect to match the
gates, match or exceed the reuse on the long workloads, and finally break the throughput term wide open,
ARC first.
