LagKV did what I hoped on retrieval and confirmed the worry I named on reasoning. Passage retrieval
climbed from StreamingLLM's 53.1 to 60.4 — nearly back to the anchor's 62.4 — exactly because the
lag-relative score keeps the informative middle tokens that StreamingLLM blindly evicted, and on a
needle-in-30-paragraphs task that is the whole game. LongBench v2 held at 29.0, neither rule having much
purchase there. But the rest of the row is more sobering than I expected. Hotpotqa came back only to 31.6
(below the anchor's 37.1, though above StreamingLLM's 25.6), and repobench actually *fell* to 40.9 —
below both StreamingLLM's 43.2 and the anchor's 47.6. That repobench drop is the tell: next-line code
completion is a workload where the most-recent tokens are overwhelmingly what predicts the next line, and
StreamingLLM's recent-window keep was already well-suited to it; LagKV's per-partition rank spends budget
on "incoherent" middle tokens that, for code completion, are not the ones that matter, so it pays for the
retrieval recovery with a code-similarity loss. And gsm8k stayed at 2.0 — essentially StreamingLLM's 1.7,
no recovery at all. That is exactly the failure I flagged: LagKV scores the *prompt's* KV statistics, but
gsm8k's load-bearing tokens are in the model's own generated reasoning, so a coherence-with-the-next-chunk
score on the prompt has no way to protect a chain-of-thought it cannot see. So LagKV beats StreamingLLM on
the geometric mean (the retrieval recovery outweighs the repobench slip), which clears the bar I set, but
it left two structural gaps: it has no principled tie to the model's *output*, and its signal is a
geometric proxy (distributional coherence) rather than a measure of how much a token will actually be
attended to. Both gaps point the same direction — score a token by its true effect on what the model
produces.

Let me make "true effect on the output" precise, because vague importance is what got LagKV to a proxy.
Write down what a single cached pair actually does. In one attention head, the query at step `t` attends
over all cached keys and the head writes back a weighted sum of values into the residual stream:
`h_t^out = h_t + sum_{i<=t} a_ti W_o v_i`, where `a_ti` is the normalized attention weight from query `t`
to key `i`, `v_i` the cached value, `W_o` the output projection. The thing to stare at is that this is
purely *additive* — the residual stream is updated by adding, and pair `i` contributes exactly one term,
`Δh_ti = a_ti W_o v_i`. So if I drop pair `i`, the damage at step `t` is precisely removing `Δh_ti`, of
size `||Δh_ti|| = a_ti · ||W_o v_i||`. There is the exact importance of a cached pair, and it factors into
two pieces: how strongly the query attends to that key, `a_ti`, and how big a kick that value gives the
output, `||W_o v_i||`. This already says what LagKV's coherence proxy and StreamingLLM's position rule
both miss — a pair can have a large, distinctive value vector but be ignored by every query (small `a`),
or be heavily attended but carry a near-zero value; only the *product* matters, and only this product is
tied to the residual stream the model actually reads downstream. I want to evict the pairs with the
smallest product, because those perturb the stream least, and a small perturbation to the stream is a
small perturbation to everything after it.

Half of this is free. `||W_o v_i||` is computable from the cache right now — I have `v_i`, I have `W_o`.
The trouble is entirely `a_ti`. And here is the wall LagKV walked around instead of through: `a_ti = z_ti /
sum_j z_tj` with `z_ti = exp(q_t^T k_i / sqrt(d))`, and the weights I care about are for the steps that
*matter* — the *future* decode steps, the ones I am deciding now which keys to keep for. But those future
`q_t` do not exist yet; I cannot compute attention from a query I have not generated. This is the same
obstruction the whole task is built around, and it is exactly why H2O and SnapKV read *past* attention as
a stand-in — but past attention is the wrong signal (a key that mattered to tokens already seen need not be
what the next thousand tokens need), and besides, this hook never materializes the attention matrix at all,
so even the past scores are gone. So I cannot observe future `a_ti` and I cannot read it off a kernel.

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
*not* a rotation: each `R_{t+j}` is orthonormal, but the average of rotations is a contraction — as the
offsets spread, the per-frequency cos/sin entries average toward smaller magnitudes, so `R̄` shrinks the
high-frequency directions more than the low. That is not a bug, it is the right behavior: directions whose
phase churns fast across the future window get washed out (no future position agrees on them), slow
directions survive. So I keep `R̄` as-is and do not re-orthonormalize.

Now the payoff. I have `q̄ ~ N(μ̄_q, Σ̄_q)` and a fixed key `k_i`, and I want
`ẑ_i = E_{q̄}[exp(q̄^T k_i / sqrt(d))]`. This is exactly a moment-generating-function evaluation: for
`X ~ N(m, C)` and fixed `s`, `E[exp(s^T X)] = exp(s^T m + (1/2) s^T C s)`. With `s = k_i / sqrt(d)`,
`ẑ_i = exp( μ̄_q^T k_i / sqrt(d) + k_i^T Σ̄_q k_i / (2d) )`. The Gaussian assumption is exactly what bought
this — the expectation of the exponential, which would otherwise need sampling, is closed form. The
constants are forced by the algebra, not chosen: the first term is the ordinary attention logit with the
query replaced by its mean (temperature `1/sqrt(d)` as in normal attention); the second came from
`(1/2) s^T C s` with `s = k/sqrt(d)`, the two `1/sqrt(d)` factors giving `1/d` and the MGF's `1/2` giving
`/(2d)`. The covariance term is doing real work, not decoration: by Jensen `E[exp(.)]` exceeds `exp(E[.])`
precisely by the spread, so a key aligned with a high-variance direction of the future-query distribution
gets boosted over its mean-logit — a key *some* future queries will love even if the average query is
lukewarm. Dropping it gives the cheaper mean-only estimate; keeping it (the default here) distinguishes
"consistently moderate" from "occasionally strong," which are different eviction decisions. I take a
softmax of the log expected scores over the key dimension to turn expected unnormalized scores into
expected attention *weights* `â_i` summing to one, comparable across keys — the `a_ti` slot of the
importance formula — and substitute back: `||Δĥ_i|| ≈ (â_i + ε) · ||v_i||`. Two harness-specific notes
here. First, strictly the formula wants `||W_o v_i||`, but materializing `W_o v_i` per value is expensive,
so this fill uses `||v_i||` as the value-magnitude proxy — it keeps the value-size factor without the
projection cost. Second, `ε` is a floor (this fill sets it to 0): when a key's expected attention is
essentially zero, `ε` would let the value norm still break ties among near-ignored keys; with `ε = 0` the
attention estimate dominates completely.

Two corrections, both tied to the first tokens, and they point opposite ways — which is the same `n_sink`
duality I have carried since step 2. The initial tokens carry the massive-activation outliers and soak up
attention regardless of content; if I let them into the sample mean and covariance of the queries they
wreck the Gaussian estimate (a few outliers drag `mu`, inflate `Σ`, and the closed form degrades). So I
exclude the first `n_sink` tokens from the *statistics* — and from the keys/values/hidden-states I score.
But because those sinks are load-bearing for the model regardless of content (the StreamingLLM lesson), I
must not *evict* them either: after scoring the body, I pad the sinks back with a score guaranteed to top
the list (this fill uses the running max), so the eviction always keeps them. Drop from the stats,
force-keep in the cache. I compute the query statistics on the *pre-RoPE* queries (`W_Q h`, before
rotation, via a projection helper that handles both `q_proj` and fused `qkv_proj` and applies `q_norm` if
present) because I apply the rotation analytically through `R̄` afterward — keeping all position handling
in the averaged rotation rather than baking a specific position into the sampled queries.

Now I map it onto the hook, respecting what this harness actually exposes. `retention_plan` declares the
sinks, the future window `n_future_positions`, the covariance and value-norm flags, `epsilon`, and the
budget (force-overridden by the harness). `score_tokens` reads only hidden states, keys, values — never an
attention tensor, which is the entire reason this method *can* run here while H2O/SnapKV cannot. It drops
the sinks, gets the pre-RoPE query mean and (optionally) covariance from the hidden states, averages RoPE
over the future window via the explicit `cos/sin` construction of `R̄` from `module.rotary_emb`, repeats
the KV heads up to the query-head count (`num_attention_heads // num_key_value_heads`) so grouped-query
attention is handled, forms the log expected scores (mean term plus, if enabled, the covariance term),
softmaxes over keys, averages the query heads sharing each KV head back down, multiplies by the value norm,
and pads the sink scores at the front with `scores.max()`. `select_cache` is the plain top-k gather of keys
and values — `rerotate_selected_keys = False`, because unlike StreamingLLM this method keeps tokens at
their original positions (it ranks by expected future attention, it does not roll a contiguous window), so
no re-rotation is needed and decode continues from the true sequence length. The full scaffold module is in
the answer.

Let me close on the falsifiable bar, against LagKV's measured row and the anchor. Retained stays ~0.20 and
runtime should sit near the prior rungs, maybe a hair higher because of the covariance einsum and the RoPE
averaging per layer — if it blows past the others on runtime, the covariance term is too expensive and I
should fall back to mean-only. The whole claim is that scoring by *expected future attention to the model's
output* beats LagKV's coherence proxy. Concretely: I expect repobench to recover above LagKV's 40.9 toward
the anchor's 47.6 — because the value-weighted expected-attention score should restore the recent,
heavily-attended code tokens LagKV's incoherence rank de-prioritized — and that recovery is the sharpest
single test, since repobench is where LagKV regressed below StreamingLLM. Hotpotqa should hold or beat
LagKV's 31.6. The honest place I might *lose* is passage retrieval: LagKV's per-partition rank guarantees
coverage across all 30 paragraphs, whereas a global expected-attention top-K could concentrate budget and
miss the paragraph holding the needle, so I would not be shocked to see passage retrieval come in *below*
LagKV's 60.4 — that would be the known weakness of a non-quota'd content score. And gsm8k: the expected-
attention estimate is built from prompt-query statistics too, so I do not expect a miracle, but if it
nudges above LagKV's 2.0 at all it confirms the output-tied signal is at least better than coherence on
reasoning. The bar to be the strongest budgeted rung is the geometric mean across the five workloads: it
must beat LagKV's, and the bet is that recovering repobench and holding hotpotqa outweighs any passage-
retrieval slip — that is the falsifiable line that decides whether expected-attention scoring is genuinely
the better shape.
