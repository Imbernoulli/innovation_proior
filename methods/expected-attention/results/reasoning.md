Let me start from what actually hurts. I have a pre-trained transformer and a long context, and the KV cache is eating my memory: every token I process leaves behind a key and a value vector in every layer, and that pile grows linearly with sequence length until it dwarfs the model weights themselves — hundreds of gigabytes for a million-token context on a model I can't shrink. I can't retrain it, I can't swap the architecture, I just have the frozen weights and a cache that's too big. So the only lever I have is to throw some of the cached `(k_i, v_i)` pairs away and keep a small fixed-budget subset. The whole problem collapses to one question: which pairs are safe to drop?

The honest answer to "safe to drop" has to be measured against the model's output, not against some convenient proxy. So let me write down what a single cached pair actually does. In one attention head, the query at step `t` attends over all cached keys and the head writes back a weighted sum of values into the hidden state:

  h_t^out = h_t + sum_{i<=t} a_ti W_o v_i,

where `a_ti` is the normalized attention weight from query `t` to key `i`, `v_i` the cached value, `W_o` the output projection. The thing I want to stare at is that this is purely *additive* — the hidden state is a residual stream that each block updates by adding to it, and pair `i` contributes exactly one additive term,

  Δh_ti = a_ti W_o v_i.

That's clean. If I drop pair `i`, the damage I do to the stream at step `t` is precisely removing `Δh_ti`, and the size of that damage is

  ||Δh_ti|| = a_ti · ||W_o v_i||.

So the *exact* importance of a cached pair, the quantity I'd kill to be able to compute, factors into two pieces: how strongly the query attends to that key, `a_ti`, and how big a kick that value gives the output, `||W_o v_i||`. This already tells me something the norm heuristics get wrong: a pair can have a large value vector but be ignored by every query (small `a`), or be heavily attended but carry a near-zero value — either way its real contribution is small. Only the product matters. And I want to evict the pairs whose product is smallest, because those are the ones whose removal barely perturbs the residual stream, and a small perturbation to the stream is a small perturbation to everything downstream.

Now, half of this is free. `||W_o v_i||` is computable from the cache right now — I have `v_i`, I have `W_o`. The trouble is entirely in `a_ti`. Write it out:

  a_ti = z_ti / sum_{j<=t} z_tj,   z_ti = exp(q_t^T k_i / sqrt(d)).

The `a_ti` I care about are the ones for the steps that *matter*, and for compression that means the *future* steps — I'm deciding now which keys to keep so that queries I haven't generated yet can still attend to what they need. But those future `q_t` don't exist. I cannot compute `a_ti` for a query that hasn't been produced. That's the wall the whole problem is built around.

Let me see how the existing methods try to climb it, because their failures will tell me what shape my answer has to be. The attention-score methods — heavy-hitter eviction, the recent-window pooling ones, the last-token ones — all read entries of the realized attention matrix `a_ti` from queries that *have* already happened and keep the keys that got the most. Two problems. First, past attention is the wrong signal: a key that mattered to the tokens already seen is not necessarily the key the next thousand tokens will need, and some of these methods quietly assume a specific question follows the context, which tilts retention toward that question and falls apart when there's no such query. Second, and this one is fatal at deployment: they need to *read* `a_ti`, and the attention kernels everyone actually runs compute the softmax-weighted sum on the fly and never materialize the `t x t` matrix at all. So even the past scores are gone. Any method that wants to be usable has to reconstruct importance from cached keys, values, and hidden states — never from a materialized attention matrix.

The position heuristics dodge `a_ti` entirely by keeping the first few "sink" tokens plus a recent window. That's robust and cheap, but it's blind to content — it discards a distant key holding the answer just because it's old. The norm and embedding heuristics — keep the smallest-norm keys, key-distance metrics, SVD projections — are at least content-aware and Flash-Attention compatible, but they score a key by a geometric proxy with no principled tie to `||Δh_ti||`, and empirically they swing wildly across model families. So the landscape is: one camp has the right target (`a_ti`) but can't compute it and can't read it; the other camp can compute its proxy but the proxy isn't tied to the output. I want the right target *and* computability.

So I can't observe future `a_ti`. Can I predict it? I don't need the attention from one specific future query; I need a *typical* future query's attention, in expectation. That reframes the obstruction from "I don't have `q_t`" to "what is the distribution of future `q_t`, and what is `E[z_ti]` under it?" If I can get `E[exp(q^T k_i / sqrt(d))]` over plausible future queries, I have an expected unnormalized score per key, and I can normalize and rank.

For that I need to know how queries are distributed, and here's where a property of these models that I'd otherwise never use becomes the load-bearing fact. The hidden states feeding the attention block in modern LLMs are empirically zero-mean, unimodal, and close to Gaussian — `h ~ N(mu, Sigma)` is a decent fit, across architectures, even under QK-normalization. (The activations *inside* the blocks are heavier-tailed, Laplacian-ish, but it's the pre-block hidden states I care about, and those are the Gaussian ones.) I don't have to *explain* why they're Gaussian; I just get to use it. Because the query is a linear map of the hidden state — `q = R W_Q h`, with `W_Q` the query projection and `R` the RoPE rotation — a Gaussian `h` pushes forward to a Gaussian query. A linear transform `q = A h + b` of `h ~ N(mu, Sigma)` is `N(A mu + b, A Sigma A^T)`, so with `A = R W_Q` and no bias,

  q_t ~ N(R_t W_Q mu, R_t W_Q Sigma W_Q^T R_t^T).

Good — so the query *is* Gaussian, with a mean and covariance I can estimate by running a handful of hidden states through `W_Q` and computing their sample mean and covariance. But there's a snag in the subscript: `R_t` depends on the position `t`, and I'm averaging over many future positions, each with its own rotation. I don't want one query distribution per future position; I want a single tractable distribution that represents "attention over the next stretch of generation."

Let me think about what RoPE does first, because I have to handle the rotation honestly. RoPE makes `q_t^T k_i` depend on the relative offset through `R_t`: in the standard implementation `R_t x = x ⊙ cos_t + rotate_half(x) ⊙ sin_t`, which I can write as a matrix `R_t = diag(cos_t) Id + diag(sin_t) P` where `P` is the permutation-with-signs that implements `rotate_half` (it maps the second half of the coordinates up to the first with a sign flip, and the first half down to the second). Each `R_t` is an honest orthonormal rotation. The future query at position `t+j` carries `R_{t+j}`. My first instinct is to just pick one representative position, but that arbitrarily commits to a single offset, and attention over a window of future steps averages over a *range* of offsets. So let me average the rotation itself over the next `T` positions:

  R̄ = (1/T) sum_{j=1}^T R_{t+j}.

Then I define a single position-averaged query distribution by pushing the Gaussian through `R̄ W_Q`:

  q̄ ~ N(μ̄_q, Σ̄_q),   μ̄_q = R̄ W_Q mu,   Σ̄_q = R̄ W_Q Sigma W_Q^T R̄^T.

I should pause on what `R̄` is, because it's tempting to assume it's still a rotation and it is not. Each `R_{t+j}` is orthonormal, but the average of rotations is not orthonormal — as the offsets spread, the per-frequency `cos`/`sin` entries average toward smaller magnitudes, so `R̄` is a contraction, shrinking the high-frequency directions more than the low-frequency ones. That's not a bug, it's exactly the right behavior: directions whose phase churns fast across the future window get washed out in the average (no single future position agrees on them), while slow directions survive. `R̄` is the linear operator that says "this is what a query looks like on average over the next `T` steps," and the contraction is the averaging doing its job. So I keep it as-is and don't try to re-orthonormalize.

Now the payoff step. I have `q̄ ~ N(μ̄_q, Σ̄_q)` and a fixed key `k_i`, and I want

  ẑ_i = E_{q̄}[ exp(q̄^T k_i / sqrt(d)) ].

This is exactly a moment-generating function evaluation. For a Gaussian `X ~ N(m, C)` and a fixed vector `s`, `E[exp(s^T X)] = exp(s^T m + (1/2) s^T C s)` — that's the MGF of a multivariate Gaussian, and it's why the Gaussian assumption was worth so much: the expectation of the exponential, which would otherwise need sampling, is a closed form. Set `s = k_i / sqrt(d)`. Then

  ẑ_i = exp( (k_i/sqrt(d))^T μ̄_q + (1/2)(k_i/sqrt(d))^T Σ̄_q (k_i/sqrt(d)) )
      = exp( μ̄_q^T k_i / sqrt(d) + k_i^T Σ̄_q k_i / (2d) ).

Let me sanity-check the constants, because a stray factor here would silently corrupt every score. The first term is the ordinary attention logit with the query replaced by its mean — temperature `1/sqrt(d)` exactly as in normal attention, good. The second term came from `(1/2) s^T C s` with `s = k/sqrt(d)`: the two `1/sqrt(d)` factors multiply to `1/d`, and the MGF's `1/2` divides it, giving `k^T Σ̄_q k / (2d)`. So `/sqrt(d)` on the mean term and `/(2d)` on the covariance term — that's forced by the algebra, not chosen.

This covariance term is doing real work and I want to be clear it's not optional decoration. By Jensen, `E[exp(.)]` exceeds `exp(E[.])` precisely by the spread, so a key that lies along a high-variance direction of the future-query distribution gets an upward boost over what its mean-logit alone would say — it's a key that *some* future queries will love even if the average query is lukewarm, and the covariance term is what records that. If I drop the covariance (use only the mean), I get a cheaper, cruder estimate `ẑ_i ≈ exp(μ̄_q^T k_i / sqrt(d))` — fine as a fallback, but it throws away the second-order information that distinguishes "consistently moderately attended" from "occasionally strongly attended," and those are different eviction decisions. So I keep the covariance term as the default and treat mean-only as the budget option.

With the log of `ẑ_i` per key, the implementation can mirror the real attention normalization by applying a softmax over the key dimension; equivalently,

  â_i = ẑ_i / sum_j ẑ_j.

This turns the expected unnormalized scores into expected attention *weights* that sum to one, comparable across keys, which is what the importance formula wanted in the `a_ti` slot. And now I just substitute back into the contribution magnitude I derived at the very start:

  ||Δĥ_i|| = (â_i + ε) · ||W_o v_i||.

There's the whole score. The `||W_o v_i||` factor I always had; `â_i` is my estimate of the missing attention weight; their product approximates `||Δh_ti||`, the residual-stream contribution. The small `ε` I add to `â_i` before multiplying is a floor: when a key's expected attention is essentially zero, without `ε` the value-norm factor gets annihilated and two near-ignored keys become indistinguishable, but with a small additive `ε` the value norm still breaks ties among the low-attention keys. The knob can be small, and I can set it to zero when I want the attention estimate to dominate completely. One practical economy: strictly the formula wants `||W_o v_i||`, but materializing `W_o v_i` for every cached value is expensive, so the implementation uses `||v_i||` as the value-magnitude proxy. That is no longer the exact projected residual norm, but it preserves the value-size factor without the projection cost.

Two corrections I have to make for the formula to behave, both tied to the first tokens of the sequence. First, those initial tokens carry the "massive activation" outliers and soak up disproportionate attention regardless of content — the attention-sink phenomenon. If I let them into the sample mean and covariance of the queries, they wreck the Gaussian estimate: a handful of outliers drags `mu` and inflates `Σ` and my closed form degrades. So I exclude the first few tokens (call it `n_sink`, four is enough) from the *statistics*. Second — and this is the flip side — because those sink tokens are load-bearing for the model regardless of their content, I must not *evict* them either. So after I've scored the body of the cache, I add the sinks back with a score guaranteed to top the list (the running max plus a hair), so the eviction step always keeps them. Drop them from the stats, force-keep them in the cache — the two roles of `n_sink` point opposite ways and both are right.

Let me also pin down where the statistics come from, because it's what makes this work in both inference phases. The mean and covariance are computed from the queries of the hidden states *I already have at compression time* — for one-shot prefill compression that's the prompt's hidden states; for trimming during long generation it's a small rolling buffer of recent hidden states. Either way I never touch a materialized attention matrix and I never assume a particular question follows, so it's Flash-Attention compatible and phase-agnostic. And one subtlety: I compute the query statistics on the *pre-RoPE* queries — `W_Q h`, before any rotation — because I'm going to apply the rotation analytically through `R̄` afterward. That keeps the position handling in one place (the averaged rotation) instead of baking a specific position into the sampled queries. For models with QK-normalization the query projection isn't strictly linear, but the normalized queries are still close enough to Gaussian in practice that the same machinery holds; I just apply the query norm when extracting the queries.

One more knob falls out naturally rather than being imposed: different heads tolerate different amounts of compression. The scoring rule should not bake in a budget policy; it should return comparable per-pair scores, and then a simple scorer can keep the same top fraction per KV head while an adaptive wrapper can spend the global budget unevenly across heads. That separation matters: the contribution score is the signal, and the head-wise budget allocation is a layer on top of that signal.

So now I can fill the one empty slot — the per-pair `score` rule — with everything above, and the rest of the eviction harness can keep the highest scores and drop the rest. Let me write it in the per-layer scorer shape: hidden states `(batch, seq, hidden)`, keys/values `(batch, n_kv_heads, seq, head_dim)`, scores out at `(batch, n_kv_heads, seq)`, with the attention tensor accepted by the hook but deliberately unused.

```python
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states

@dataclass
class ExpectedAttentionPress(ScorerPress):
    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True
    epsilon: float = 0.0

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        q_len = hidden_states.shape[1]
        h = hidden_states[:, self.n_sink:]             # drop sink outliers from statistics
        query_states = get_prerope_query_states(module, h)

        mu = query_states.mean(dim=2, keepdim=True)
        cov = None
        if self.use_covariance:
            centered = query_states - mu
            cov = torch.einsum("bnsi,bnsj->bnij", centered, centered) / h.shape[1]
        mu = mu.squeeze(2)
        return self.apply_avg_rope(module, mu, cov, q_len)

    def apply_avg_rope(self, module: nn.Module, mu: torch.Tensor, cov: torch.Tensor, q_len: int):
        position_ids = torch.arange(q_len, q_len + self.n_future_positions, device=mu.device).unsqueeze(0)
        head_dim = module.head_dim
        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]

        Id = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
        half = head_dim // 2
        eye_half = torch.eye(half, device=cos.device, dtype=cos.dtype)
        P[half:, :half] = eye_half                     # first half moves to second half
        P[:half, half:] = -eye_half                    # second half moves to first with sign flip

        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P
        R = R.mean(dim=0).to(mu.device)                # average of rotations, hence a contraction
        mu = torch.matmul(mu, R.T)
        if cov is not None:
            cov = torch.matmul(R, torch.matmul(cov, R.T))
        return mu, cov

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        assert keys.size(2) > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        keys = keys[:, :, self.n_sink:]
        values = values[:, :, self.n_sink:]

        mean_query, cov_query = self.get_query_statistics(module, hidden_states)

        bsz, num_key_value_heads, q_len, d = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)

        log_scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
        if self.use_covariance:
            log_scores += torch.einsum("bhin,bhij,bhjn->bhn", keys, cov_query, keys) / d / 2
        scores = F.softmax(log_scores, dim=-1)         # softmax over keys

        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        scores = scores.mean(dim=2)                    # average query heads sharing each KV head

        if self.use_vnorm:
            scores = (scores + self.epsilon) * values.norm(dim=-1)

        return F.pad(scores, (self.n_sink, 0), value=scores.max().item() + 1)
```

Let me trace the causal chain back to be sure it holds together. I started memory-bound, able only to evict cached KV pairs, and I refused to score them by a proxy, insisting on their true effect on the output — which the residual-stream decomposition pins down exactly as `a_i · ||W_o v_i||`. The value-norm half is free; the attention half needs future queries that don't exist and can't be read off a Flash-Attention kernel. I turned "I don't have the future query" into "what's the distribution of future queries," used the empirical fact that hidden states are Gaussian to make queries Gaussian, averaged the RoPE rotation over the future window into a single contracting `R̄` so one query distribution stands in for the whole horizon, and then the Gaussian MGF handed me the expected unnormalized score in closed form, `exp(μ̄_q^T k/sqrt(d) + k^T Σ̄_q k/(2d))` — with the covariance term being the Jensen boost for keys aligned with high-variance future-query directions. Softmax over the log expected scores gives the expected attention weight, the value norm restores the contribution magnitude, `ε` floors the near-ignored keys, and the sink tokens get pulled out of the statistics but forced back into the cache with `max + 1`. The result reads only cached keys, values, and hidden states, works whether I compress the whole prompt once or trim the cache as I generate, and drops straight into the per-layer eviction hook as the one missing scoring rule.
