LagKV did what I hoped on retrieval and confirmed the worry I named on reasoning. Let me read its row
against StreamingLLM's and the anchor, delta by delta, because the two gaps it leaves are what set this
rung's target. Passage retrieval climbed from StreamingLLM's 53.1 to 60.4 — up 7.3, back to 97% of the
anchor's 62.4 — exactly because the lag-relative score keeps the informative middle tokens that
StreamingLLM blindly evicted, and on a needle-in-30-paragraphs task the per-partition quota that
guarantees every paragraph a share of the budget is the whole game. LongBench v2 held at 29.0, neither
rule having purchase on a chance-floored format. Hotpotqa recovered from 25.6 to 31.6 — up 6.0, but still
5.5 short of the anchor's 37.1, only 85% of the ceiling. But two numbers are more sobering than the
recovery. Repobench actually *fell*, 43.2 to 40.9, down 2.3 — now below both StreamingLLM's 43.2 and the
anchor's 47.6. And gsm8k moved 1.7 to 2.0, up 0.3, which is statistically nothing: no recovery at all.

Those two numbers are the tell, and they point the same way. Repobench falling is the clearest: next-line
code completion is a workload where the most-recent tokens overwhelmingly predict the next line, and
StreamingLLM's recent-window keep was already well-suited to it — recall it dropped only 9% from the
anchor, the smallest of the four. LagKV's per-partition rank spends budget on "incoherent" middle tokens
that, for code completion, are not the ones that matter, so it pays for the retrieval recovery with a
code-similarity loss. And gsm8k staying at 2.0 is exactly the failure I flagged at step 3: LagKV scores
the *prompt's* KV statistics, but gsm8k's load-bearing tokens are in the model's own generated reasoning,
so a coherence-with-the-next-chunk score on the prompt has no way to protect a chain-of-thought it cannot
see. So LagKV beats StreamingLLM on the geometric mean — the retrieval recovery outweighs the repobench
slip — which clears the bar I set, but it left two structural gaps: it has no principled tie to the
model's *output*, and its signal is a geometric proxy (distributional coherence) rather than a measure of
how much a token will actually be attended to. Both gaps point the same direction — score a token by its
true effect on what the model produces. The runtime row, incidentally, confirmed the small prediction I
made at step 3: dropping StreamingLLM's re-rotation trimmed the short-decode workloads a touch (356.8 and
328.5 against step 2's 360.9 and 330.9), and the added reductions were cheap enough not to swamp that. So
the scoring budget on this hook is real but small, which tells me I have room to add a heavier
computation at this rung without blowing the runtime term — as long as I watch it.

Let me make "true effect on the output" precise, because vague importance is what got LagKV to a proxy.
Write down what a single cached pair actually does. In one attention head, the query at step `t` attends
over all cached keys and the head writes back a weighted sum of values into the residual stream:
`h_t^out = h_t + sum_{i<=t} a_ti W_o v_i`, where `a_ti` is the normalized attention weight from query `t`
to key `i`, `v_i` the cached value, `W_o` the output projection. The thing to stare at is that this is
purely *additive* — the residual stream is updated by adding, and pair `i` contributes exactly one term,
`Δh_ti = a_ti W_o v_i`. So if I drop pair `i`, the damage at step `t` is precisely removing `Δh_ti`, of
size `||Δh_ti|| = a_ti · ||W_o v_i||`. Let me check the structure of that expression before I trust it.
`a_ti` is a dimensionless weight in `[0,1]`; `||W_o v_i||` is a length in residual-stream space (dimension
2048 here), the same units as `||h||` itself; so `||Δh_ti||` is a genuine perturbation magnitude of the
stream, and it is a *product*, which means it correctly vanishes if *either* factor vanishes. A key some
query attends to heavily but whose value is near zero (`a` large, `||v||≈0`) perturbs nothing; a value
vector that is large and distinctive but that no query ever looks at (`||v||` large, `a≈0`) perturbs
nothing. Only the product is tied to the residual stream the model actually reads downstream — and that
is exactly what LagKV's coherence proxy and StreamingLLM's position rule both miss, because a distinctive
value with no attention scores high on coherence and low on truth. I want to evict the pairs with the
smallest product, because those perturb the stream least, and a small perturbation to the stream is a
small perturbation to everything after it.

Half of this is free. `||W_o v_i||` is computable from the cache right now — I have `v_i`, I have `W_o`.
The trouble is entirely `a_ti`. And here is the wall LagKV walked around instead of through:
`a_ti = z_ti / sum_j z_tj` with `z_ti = exp(q_t^T k_i / sqrt(d))`, and the weights I care about are for the
steps that *matter* — the *future* decode steps, the ones I am deciding now which keys to keep for. But
those future `q_t` do not exist yet; I cannot compute attention from a query I have not generated. This is
the same obstruction the whole task is built around, and it is exactly why H2O and SnapKV read *past*
attention as a stand-in — but past attention is the wrong signal (a key that mattered to tokens already
seen need not be what the next thousand tokens need), and besides, this hook never materializes the
attention matrix at all, so even the past scores are gone. So I cannot observe future `a_ti` and I cannot
read it off a kernel.

Can I *predict* it? I do not need the attention from one specific future query; I need a *typical* future
query's attention, in expectation. That reframes the obstruction from "I do not have `q_t`" to "what is
the distribution of future `q_t`, and what is `E[z_ti]` under it?" If I can get
`E[exp(q^T k_i / sqrt(d))]` over plausible future queries, I have an expected unnormalized score per key,
and I can normalize and rank. For that I need to know how queries are distributed — and here a property of
these models I would otherwise never use becomes load-bearing. The hidden states feeding the attention
block are empirically zero-mean, unimodal, and close to Gaussian, `h ~ N(mu, Sigma)`, across
architectures (the activations *inside* the blocks are heavier-tailed, but it is the pre-block hidden
states I care about, and those are the Gaussian ones). I do not have to explain why; I get to use it.
Because the query is a linear map of the hidden state, `q = R W_Q h` with `W_Q` the query projection and
`R` the RoPE rotation, a Gaussian `h` pushes forward to a Gaussian query: a linear map `q = A h` of
`h ~ N(mu, Sigma)` is `N(A mu, A Sigma A^T)`, so `q_t ~ N(R_t W_Q mu, R_t W_Q Sigma W_Q^T R_t^T)`. I can
estimate `mu` and `Sigma` by running the hidden states I already have at compression time — the prompt's
hidden states — through `W_Q` and taking the sample mean and covariance. Critically, this addresses
LagKV's gsm8k blind spot honestly: I am not scoring on a coherence proxy of the cache, I am estimating the
distribution of the queries that will actually do the reading, which is a strictly more output-relevant
signal than "does this token look like its neighbor."

There is a snag in the subscript: `R_t` depends on position `t`, and I am averaging over many future
positions, each with its own rotation. I want a single tractable query distribution standing for
"attention over the next stretch of generation," not one per position. RoPE in the standard implementation
acts as `R_t x = x * cos_t + rotate_half(x) * sin_t`, which I can write as a matrix
`R_t = diag(cos_t) Id + diag(sin_t) P`, with `P` the signed permutation implementing `rotate_half`. Each
`R_t` is an honest orthonormal rotation. The future query at `t+j` carries `R_{t+j}`; rather than commit to
one representative offset, average the rotation itself over the next `T` positions:
`R̄ = (1/T) sum_{j=1}^T R_{t+j}`, and push the Gaussian through `R̄ W_Q` to get the position-averaged query
`q̄ ~ N(μ̄_q, Σ̄_q)`, `μ̄_q = R̄ W_Q mu`, `Σ̄_q = R̄ W_Q Sigma W_Q^T R̄^T`. I should be exact that `R̄` is
*not* a rotation: each `R_{t+j}` is orthonormal, but the average of rotations is a contraction. Let me
verify that claim frequency by frequency, because whether `R̄` shrinks the right directions decides whether
this whole averaging trick is sound. RoPE's per-frequency angle is `theta_j = base^{-2j/d}` with base
10000 and `d = 128`, so the highest frequency is `theta_0 = 1` and the lowest is
`theta_63 = 10000^{-126/128} ≈ 1.07e-4`. Average `cos(t theta)` over `T = 512` future positions. For the
top frequency `theta_0 = 1`, the phase sweeps `512` radians, which is `512/2π ≈ 81` full cycles, and the
mean of cosine over ~81 whole periods is essentially `0` — that direction is contracted to nothing. For
the bottom frequency `theta_63 ≈ 1.07e-4`, the phase sweeps only `512 * 1.07e-4 ≈ 0.055` radians, over
which the cosine barely moves, so its average stays near its value — that direction is preserved. So `R̄`
is *frequency-selective*: it washes out the fast-churning directions, on which no two future positions
agree, and keeps the slow ones. That is not a bug, it is the right behavior — the high-frequency phases
carry no consistent signal across a 512-token future — so I keep `R̄` as-is and do not re-orthonormalize.

Now the payoff. I have `q̄ ~ N(μ̄_q, Σ̄_q)` and a fixed key `k_i`, and I want
`ẑ_i = E_{q̄}[exp(q̄^T k_i / sqrt(d))]`. This is exactly a moment-generating-function evaluation: for
`X ~ N(m, C)` and fixed `s`, `E[exp(s^T X)] = exp(s^T m + (1/2) s^T C s)`. Let me confirm the constants on
the scalar case rather than copy them, since if I get the temperature or the `1/2` wrong every score is
biased. For scalar `X ~ N(m, σ²)`, the MGF is `E[exp(sX)] = exp(sm + s²σ²/2)`. Put `s = k/sqrt(d)`: the
first term is `mk/sqrt(d)`, and the second is `(k/sqrt(d))² σ²/2 = k² σ² /(2d)`, because the two
`1/sqrt(d)` factors give `1/d` and the MGF's own `1/2` gives the `/(2d)`. Promote to vectors:
`ẑ_i = exp( μ̄_q^T k_i / sqrt(d) + k_i^T Σ̄_q k_i / (2d) )`. The constants are forced by the algebra, not
chosen — the first term is the ordinary attention logit with the query replaced by its mean (temperature
`1/sqrt(d)` as in normal attention); the second is the variance correction with exactly the `1/(2d)` the
scalar check produced. And the direction of the covariance term is a check worth doing too: by Jensen
`E[exp(.)] ≥ exp(E[.])` always, so the correction can only ever *boost* a key, never penalize it — which
is consistent with `k_i^T Σ̄_q k_i ≥ 0` because `Σ̄_q`, a covariance, is positive semidefinite, so that
quadratic form is non-negative for every `k_i`. The boost is largest for keys aligned with a high-variance
direction of the future-query distribution — a key *some* future queries will love even if the average
query is lukewarm. Dropping the covariance term gives the cheaper mean-only estimate; keeping it (the
default here) distinguishes "consistently moderate" from "occasionally strong," which are different
eviction decisions. I take a softmax of the log expected scores over the key dimension to turn expected
unnormalized scores into expected attention *weights* `â_i` summing to one, comparable across keys — the
`a_ti` slot of the importance formula — and substitute back: `||Δĥ_i|| ≈ (â_i + ε) · ||v_i||`. Two
harness-specific notes here. First, strictly the formula wants `||W_o v_i||`, but materializing `W_o v_i`
per value is expensive, so this fill uses `||v_i||` as the value-magnitude proxy — it keeps the
value-size factor without the projection cost. Second, `ε` is a floor (this fill sets it to 0): when a
key's expected attention is essentially zero, `ε` would let the value norm still break ties among
near-ignored keys; with `ε = 0` the attention estimate dominates completely.

That covariance term is also the one place I have a real cost-versus-signal decision, so let me price it
rather than take it for free. Mean-only scoring is a single dot product per key, `μ̄_q^T k_i` — cheap. The
covariance term is a quadratic form `k_i^T Σ̄_q k_i`, an `O(d²)` einsum per key: with `d = 128` that is
~16k multiply-adds per key per query head, and over thousands of prompt keys, 16 query heads, and 36
layers it is not nothing. Is it worth it? Picture two keys with the *same* mean-query logit, so mean-only
scoring calls them a tie. Key A points along a high-variance direction of the future-query distribution —
some future queries will attend to it strongly, others weakly — while key B points along a low-variance
direction where every future query is uniformly lukewarm. The importance formula says A is the one to
keep: it is the key a diverse future *sometimes* loves, exactly the retrieval-needle pattern where one
late query locks onto one span. The covariance term is what breaks the tie in A's favor, because
`k_A^T Σ̄_q k_A` is large where `k_B^T Σ̄_q k_B` is small, and mean-only scoring is blind to the
difference. So the second-order term earns its cost precisely on the workloads I most want to recover —
the retrieval and QA ones with a needle some query will fixate on — and I keep it on by default while
exposing the flag; if the runtime row shows this rung blowing past the others, the covariance einsum is
the first thing I cut back to mean-only.

The future window `T` is the one real hyperparameter in the averaging, and the contraction analysis pins
it. Too small — say `T = 8` — and I am predicting attention for only the immediate next few decode steps,
which is close to SnapKV's "observation window" logic and does not represent a *typical* future query over
a long generation; the whole point was to average over the future, not peek at its first token. Too large
— say `T = 8192` — and the frequency sweep I just computed washes out not only the top frequencies but the
middle ones too (a mid-frequency `theta` churns through many cycles over 8192 positions), so `R̄` collapses
toward its low-frequency-only projection and the query distribution loses most of its positional shape,
drifting toward a single position-independent mean direction. `T = 512` sits in between: it spans a
realistic generation length — gsm8k's chains run to hundreds of tokens — so it stands for a genuine future
query rather than the next step, while staying short enough that the mid and low frequencies survive `R̄`
and the query distribution keeps positional structure. So `n_future_positions = 512`.

Two corrections, both tied to the first tokens, and they point opposite ways — which is the same `n_sink`
duality I have carried since step 2. The initial tokens carry the massive-activation outliers and soak up
attention regardless of content; if I let them into the sample mean and covariance of the queries they
wreck the Gaussian estimate. Concretely: a handful of outlier tokens drag `mu` off the bulk and inflate
`Σ`, and because the score carries `exp(k^T Σ̄_q k / (2d))`, an inflated `Σ` makes that term blow up for
any key aligned with the outlier direction, corrupting the entire ranking. Excluding four sinks from a
sample of thousands is negligible for estimating `mu` and `Σ`, but it removes the heavy-tail contamination
that would otherwise dominate the covariance term. So I exclude the first `n_sink` tokens from the
*statistics* — and from the keys/values/hidden-states I score. But because those sinks are load-bearing
for the model regardless of content (the StreamingLLM lesson, which cost 30 gsm8k points to learn), I must
not *evict* them either: after scoring the body, I pad the sinks back with a score guaranteed to top the
list (this fill uses the running max), so the eviction always keeps them. Drop from the stats, force-keep
in the cache. I compute the query statistics on the *pre-RoPE* queries (`W_Q h`, before rotation, via a
projection helper that handles both `q_proj` and fused `qkv_proj` and applies `q_norm` if present) because
I apply the rotation analytically through `R̄` afterward — keeping all position handling in the averaged
rotation rather than baking a specific position into the sampled queries.

One more shape I have to get right on this model is grouped-query attention, because the score has to come
out per KV head while the expected attention is a per-query-head quantity. This model has 16 query heads
and 2 KV heads, so `num_key_value_groups = 16 // 2 = 8`: each KV head is read by 8 query heads. Eviction
is per-KV-head — the cache is stored and returned per KV head, and `score_tokens` must produce a
`(batch, num_kv_heads, k_len)` tensor — but a key's expected attention differs across the 8 query heads
that share it, each with its own query projection and therefore its own `μ̄_q` and `Σ̄_q`. So I repeat the 2
KV heads up to 16 (`repeat_kv` with `n_rep = 8`), form the log expected scores for all 16 query heads,
softmax each over the keys, then average the 8 query heads that share each KV head back down to one score
per KV head — the reshape to `(bsz, num_kv_heads, num_groups, q_len)` followed by a mean over the groups
axis. That averaging is the honest aggregation: a token is worth keeping for a KV head to the extent the
query heads reading that KV head attend to it, on average.

Now I map it onto the hook, respecting what this harness actually exposes. `retention_plan` declares the
sinks, the future window `n_future_positions`, the covariance and value-norm flags, `epsilon`, and the
budget (force-overridden by the harness). `score_tokens` reads only hidden states, keys, values — never an
attention tensor, which is the entire reason this method *can* run here while H2O/SnapKV cannot. It drops
the sinks, gets the pre-RoPE query mean and (optionally) covariance from the hidden states, averages RoPE
over the future window via the explicit `cos/sin` construction of `R̄` from `module.rotary_emb`, repeats
the KV heads up to the query-head count so grouped-query attention is handled, forms the log expected
scores (mean term plus, if enabled, the covariance term), softmaxes over keys, averages the query heads
sharing each KV head back down, multiplies by the value norm, and pads the sink scores at the front with
`scores.max()`. `select_cache` is the plain top-k gather of keys and values —
`rerotate_selected_keys = False`, because like LagKV and unlike StreamingLLM this method keeps tokens at
their original positions (it ranks by expected future attention, it does not roll a contiguous window), so
the true relative geometry is preserved and no re-rotation is needed; decode continues from the true
sequence length. The full scaffold module is in the answer.

Let me close on the falsifiable bar, against LagKV's measured row and the anchor. Retained stays ~0.20 and
runtime should sit near the prior rungs, maybe a hair higher because of the covariance einsum and the RoPE
averaging per layer — I saw at step 3 that the scoring cost on this hook is real but small, so the
covariance term should be affordable, but if runtime blows past the others the term is too expensive and I
should fall back to mean-only. The whole claim is that scoring by *expected future attention to the
model's output* beats LagKV's coherence proxy. Concretely: I expect repobench to recover above LagKV's
40.9 toward the anchor's 47.6 — because the value-weighted expected-attention score should restore the
recent, heavily-attended code tokens LagKV's incoherence rank de-prioritized — and that recovery is the
sharpest single test, since repobench is where LagKV regressed below StreamingLLM. Hotpotqa should hold or
beat LagKV's 31.6. The honest place I might *lose* is passage retrieval: LagKV's per-partition rank
guarantees coverage across all 30 paragraphs, whereas a global expected-attention top-K could concentrate
budget and miss the paragraph holding the needle, so I would not be shocked to see passage retrieval come
in *below* LagKV's 60.4 — that would be the known weakness of a non-quota'd content score. And gsm8k: the
expected-attention estimate is built from prompt-query statistics too, so I do not expect a miracle, but
if it nudges above LagKV's 2.0 at all it confirms the output-tied signal is at least better than coherence
on reasoning. The bar to be the strongest budgeted rung is the geometric mean across the five workloads:
it must beat LagKV's, and the bet is that recovering repobench and holding hotpotqa outweighs any
passage-retrieval slip — that is the falsifiable line that decides whether expected-attention scoring is
genuinely the better shape.
