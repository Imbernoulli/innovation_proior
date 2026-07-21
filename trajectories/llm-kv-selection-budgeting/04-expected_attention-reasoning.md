LagKV did what I hoped on retrieval and confirmed the worry I named on reasoning. Read delta by delta:
passage retrieval climbed from StreamingLLM's 53.1 to 60.4 — up 7.3, back to 97% of the anchor's 62.4 —
because the lag-relative score keeps the informative middle tokens StreamingLLM evicted, and on a
needle-in-30-paragraphs task the per-partition quota that guarantees every paragraph a share is the whole
game. LongBench v2 held at 29.0. Hotpotqa recovered 25.6 to 31.6 — up 6.0, but still 5.5 short of the
anchor's 37.1. Two numbers are more sobering. Repobench actually *fell*, 43.2 to 40.9, now below both
StreamingLLM and the anchor. And gsm8k moved 1.7 to 2.0 — statistically nothing.

Those two point the same way. Repobench falling is clearest: next-line code completion is a workload
where the most-recent tokens overwhelmingly predict the next line, and StreamingLLM's recent-window keep
already suited it (it dropped only 9% from the anchor); LagKV's per-partition rank spends budget on
"incoherent" middle tokens that, for code completion, are not the ones that matter, paying for the
retrieval recovery with a code-similarity loss. gsm8k staying at 2.0 is the failure I flagged at step 3:
LagKV scores the *prompt's* KV statistics, but gsm8k's load-bearing tokens are in the model's own
generated reasoning, which a coherence-with-the-next-chunk score on the prompt has no way to protect. So
LagKV beats StreamingLLM on the geometric mean — clearing the bar I set — but leaves two structural gaps:
no principled tie to the model's *output*, and a signal that is a geometric proxy (distributional
coherence) rather than a measure of how much a token will actually be attended to. Both point the same
direction: score a token by its true effect on what the model produces. The runtime row also confirmed my
step-3 prediction — dropping the re-rotation trimmed the short-decode workloads a touch and the added
reductions were cheap — so the scoring budget on this hook is real but small, which tells me I have room
to add a heavier computation here without blowing the runtime term, as long as I watch it.

Let me make "true effect on the output" precise, because vague importance is what got LagKV to a proxy.
In one attention head, the query at step `t` attends over all cached keys and writes a weighted sum of
values into the residual stream: `h_t^out = h_t + sum_{i<=t} a_ti W_o v_i`, with `a_ti` the normalized
weight from query `t` to key `i`. This is purely *additive* — pair `i` contributes exactly one term,
`Δh_ti = a_ti W_o v_i` — so dropping pair `i` removes precisely `Δh_ti`, of size `||Δh_ti|| = a_ti ·
||W_o v_i||`. The structure of that expression is what makes it the right signal: `a_ti` is a
dimensionless weight in `[0,1]`, `||W_o v_i||` a length in residual-stream space, so `||Δh_ti||` is a
genuine perturbation magnitude, and being a *product* it vanishes if *either* factor vanishes. A key
attended heavily but with near-zero value perturbs nothing; a large distinctive value no query looks at
perturbs nothing. Only the product is tied to the stream the model reads downstream — exactly what
LagKV's coherence proxy misses, because a distinctive value with no attention scores high on coherence
and low on truth. I want to evict the pairs with the smallest product.

Half of this is free: `||W_o v_i||` is computable from the cache right now. The trouble is entirely
`a_ti = z_ti / sum_j z_tj` with `z_ti = exp(q_t^T k_i / sqrt(d))`, and the weights I care about are for
the *future* decode steps — the ones I am deciding which keys to keep for. Those `q_t` do not exist yet.
This is the same obstruction the whole task is built around, and it is why H2O and SnapKV read *past*
attention as a stand-in — but past attention is the wrong signal (a key that mattered to tokens already
seen need not be what the next thousand need), and this hook never materializes the matrix at all.

Can I *predict* it? I do not need the attention from one specific future query; I need a *typical*
future query's attention in expectation, which reframes the obstruction to "what is the distribution of
future `q_t`, and what is `E[z_ti]` under it?" For that I use a property of these models: the pre-block
hidden states feeding attention are empirically zero-mean, unimodal, and close to Gaussian, `h ~ N(mu,
Sigma)` (the activations *inside* the blocks are heavier-tailed, but it is the pre-block states I care
about). Since the query is a linear map `q = R W_Q h`, a Gaussian `h` pushes forward to a Gaussian query:
`q_t ~ N(R_t W_Q mu, R_t W_Q Sigma W_Q^T R_t^T)`. I estimate `mu` and `Sigma` by running the prompt's
hidden states — which I already have at compression time — through `W_Q`. This addresses LagKV's gsm8k
blind spot honestly: I am not scoring a coherence proxy of the cache, I am estimating the distribution of
the queries that will do the reading, a strictly more output-relevant signal.

There is a snag: `R_t` depends on position `t`, and I average over many future positions, each with its
own rotation. I want one tractable query distribution standing for "attention over the next stretch," not
one per position. RoPE acts as `R_t x = x*cos_t + rotate_half(x)*sin_t`, which I write as `R_t =
diag(cos_t) Id + diag(sin_t) P` with `P` the signed permutation for `rotate_half`. Rather than commit to
one representative offset, average the rotation itself over the next `T` positions: `R̄ = (1/T) sum_{j=1}^T
R_{t+j}`, and push the Gaussian through `R̄ W_Q` to get `q̄ ~ N(μ̄_q, Σ̄_q)`. `R̄` is *not* a rotation —
the average of orthonormal rotations is a contraction — and which directions it shrinks decides whether
the averaging is sound. RoPE's per-frequency angle is `theta_j = base^{-2j/d}`, base 10000, `d = 128`, so
the top frequency is `theta_0 = 1` and the bottom `theta_63 ≈ 1.07e-4`. Averaging `cos(t theta)` over `T
= 512` positions: for `theta_0 = 1` the phase sweeps 512 radians (~81 full cycles), so the mean is
essentially 0 — that direction is contracted to nothing; for `theta_63` the phase sweeps only ~0.055
radians, so its average stays near its value — preserved. So `R̄` is frequency-selective: it washes out
the fast-churning directions on which no two future positions agree and keeps the slow ones. That is the
right behavior, so I keep `R̄` as-is and do not re-orthonormalize.

Now the payoff. I have `q̄ ~ N(μ̄_q, Σ̄_q)` and a fixed `k_i`, and I want `ẑ_i = E_{q̄}[exp(q̄^T k_i /
sqrt(d))]`. This is a moment-generating-function evaluation: for `X ~ N(m, C)`, `E[exp(s^T X)] = exp(s^T
m + (1/2) s^T C s)`. With `s = k_i / sqrt(d)` the two `1/sqrt(d)` factors give `1/d` and the MGF's own
`1/2` gives the `/(2d)`, so `ẑ_i = exp( μ̄_q^T k_i / sqrt(d) + k_i^T Σ̄_q k_i / (2d) )`. The first term is
the ordinary attention logit with the query replaced by its mean; the second is a variance correction. By
Jensen `E[exp(.)] ≥ exp(E[.])`, so the correction can only *boost* a key, never penalize — consistent
with `Σ̄_q` positive semidefinite making `k_i^T Σ̄_q k_i ≥ 0` — and the boost is largest for keys aligned
with a high-variance direction of the future-query distribution: a key *some* future queries will love
even if the average query is lukewarm. I softmax the log expected scores over the key dimension to turn
them into expected attention weights `â_i` summing to one — the `a_ti` slot of the importance formula —
and substitute back: `||Δĥ_i|| ≈ (â_i + ε) · ||v_i||`. Two harness notes: materializing `W_o v_i` per
value is expensive, so I use `||v_i||` as the value-magnitude proxy, keeping the value-size factor without
the projection cost; and `ε` is a floor I set to 0, so the attention estimate dominates completely rather
than value norm breaking ties among near-ignored keys.

The covariance term is the one real cost-versus-signal decision. Mean-only scoring is a single dot
product per key, `μ̄_q^T k_i` — cheap. The covariance term is a quadratic form `k_i^T Σ̄_q k_i`, an
`O(d²)` einsum per key; with `d = 128`, ~16k multiply-adds per key per query head, over thousands of
keys, 16 query heads, 36 layers — not nothing. Is it worth it? Picture two keys with the same mean-query
logit, so mean-only scoring calls them a tie. Key A points along a high-variance direction — some future
queries attend strongly, others weakly — while B points along a low-variance direction where every future
query is uniformly lukewarm. The importance formula says keep A: the key a diverse future *sometimes*
loves, exactly the retrieval-needle pattern where one late query locks onto one span. `k_A^T Σ̄_q k_A` is
large where `k_B^T Σ̄_q k_B` is small, and mean-only scoring is blind to it. So the second-order term
earns its cost precisely on the workloads I most want to recover, and I keep it on by default while
exposing the flag; if runtime blows past the others, mean-only is the first fallback.

The future window `T` is the one real hyperparameter in the averaging, pinned by the contraction
analysis. Too small — `T = 8` — predicts attention for only the immediate next few steps, close to
SnapKV's observation-window logic, not a *typical* future query. Too large — `T = 8192` — washes out not
only the top frequencies but the middle ones (a mid-frequency churns through many cycles over 8192
positions), so `R̄` collapses toward its low-frequency projection and the query distribution loses its
positional shape. `T = 512` sits between: it spans a realistic generation length (gsm8k's chains run to
hundreds of tokens) while staying short enough that mid and low frequencies survive `R̄`. So
`n_future_positions = 512`.

Two corrections tied to the first tokens, pointing opposite ways — the same `n_sink` duality I have
carried since step 2. The initial tokens carry massive-activation outliers and soak up attention
regardless of content; if I let them into the sample mean and covariance they wreck the Gaussian
estimate. A handful of outlier tokens drag `mu` off the bulk and inflate `Σ`, and because the score
carries `exp(k^T Σ̄_q k / (2d))`, an inflated `Σ` makes that term blow up for any key aligned with the
outlier direction, corrupting the ranking. Excluding four sinks from a sample of thousands is negligible
for estimating `mu` and `Σ` but removes the heavy-tail contamination. So I drop the first `n_sink` tokens
from the statistics and from the keys/values/hidden-states I score — but because those sinks are
load-bearing regardless of content (the StreamingLLM lesson, 30 gsm8k points), I must not evict them
either, so after scoring the body I pad the sinks back with a score guaranteed to top the list (the
running max). Drop from the stats, force-keep in the cache. I compute the query statistics on the
*pre-RoPE* queries (`W_Q h`, before rotation, via a projection helper that handles both `q_proj` and
fused `qkv_proj` and applies `q_norm` if present) because I apply the rotation analytically through `R̄`
afterward — keeping all position handling in the averaged rotation rather than baking a specific position
into the sampled queries.

One shape I have to get right on this model is grouped-query attention: the score comes out per KV head
while expected attention is per-query-head. This model has 16 query heads and 2 KV heads, so
`num_key_value_groups = 8`. Eviction is per-KV-head — `score_tokens` must return `(batch, num_kv_heads,
k_len)` — but a key's expected attention differs across the 8 query heads sharing it, each with its own
`μ̄_q` and `Σ̄_q`. So I repeat the 2 KV heads up to 16, form the log expected scores for all 16 query
heads, softmax each over the keys, then average the 8 query heads sharing each KV head back to one score.
That averaging is the honest aggregation: a token is worth keeping for a KV head to the extent the query
heads reading it attend to it on average.

On the hook: `score_tokens` reads only hidden states, keys, values — never an attention tensor, the
entire reason this method *can* run here while H2O/SnapKV cannot. It drops the sinks, gets the pre-RoPE
query mean and (optionally) covariance, averages RoPE over the future window via the explicit cos/sin
construction of `R̄` from `module.rotary_emb`, repeats the KV heads up to the query-head count, forms the
log expected scores, softmaxes over keys, averages the sharing query heads down, multiplies by the value
norm, and pads the sink scores at the front. `select_cache` is a plain top-k gather with
`rerotate_selected_keys = False`: like LagKV and unlike StreamingLLM this method keeps tokens at their
original positions (it ranks by expected future attention, it does not roll a contiguous window), so the
true relative geometry is preserved and decode continues from the true sequence length. The full scaffold
module is in the answer.

The falsifiable bar, against LagKV's row. Retained stays ~0.20 and runtime should sit near the prior
rungs, a hair higher from the covariance einsum and RoPE averaging — affordable given step 3's small
scoring cost, but if it blows up I fall back to mean-only. The claim is that scoring by expected future
attention beats LagKV's coherence proxy. I expect repobench to recover above 40.9 toward the anchor's
47.6 — the sharpest test, since the value-weighted expected-attention score should restore the recent,
heavily-attended code tokens LagKV's incoherence rank de-prioritized, and repobench is where LagKV
regressed below StreamingLLM. Hotpotqa should hold or beat 31.6. The honest place I might *lose* is
passage retrieval: LagKV's per-partition rank guarantees coverage across all 30 paragraphs, while a
global expected-attention top-K could concentrate budget and miss the paragraph holding the needle, so I
would not be shocked to see it come in below 60.4 — the known weakness of a non-quota'd content score.
And gsm8k is built from prompt-query statistics too, so no miracle, but a nudge above 2.0 would confirm
the output-tied signal is at least better than coherence on reasoning. The bar to be the strongest
budgeted rung is the geometric mean: it must beat LagKV's, and the bet is that recovering repobench and
holding hotpotqa outweighs any passage-retrieval slip.
