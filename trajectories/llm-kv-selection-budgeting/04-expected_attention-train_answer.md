LagKV did what I hoped on retrieval and confirmed the worry I had named on reasoning. Passage retrieval climbed from StreamingLLM's 53.1 to 60.4 — nearly back to the anchor's 62.4 — because the lag-relative score keeps the informative middle tokens StreamingLLM blindly evicted. But the rest of the row was sobering: hotpotqa came back only to 31.6 (below the anchor's 37.1), repobench actually *fell* to 40.9 (below both StreamingLLM's 43.2 and the anchor's 47.6) because next-line code completion is overwhelmingly predicted by the most-recent tokens and LagKV's per-partition rank spends budget on "incoherent" middle tokens that, for code, are not the ones that matter, and gsm8k stayed at 2.0 — no recovery, because LagKV scores the *prompt's* KV statistics while gsm8k's load-bearing tokens live in the model's own generated reasoning. LagKV beat StreamingLLM on the geometric mean, but it left two structural gaps: no principled tie to the model's *output*, and a signal that is a geometric proxy (distributional coherence) rather than a measure of how much a token will actually be attended to. Both point the same way — score a token by its true effect on what the model produces.

Let me make "true effect on the output" precise. In one attention head the query at step $t$ writes a weighted sum of values back into the residual stream: $h_t^{\text{out}} = h_t + \sum_{i \le t} a_{ti}\, W_o v_i$, where $a_{ti}$ is the normalized attention weight from query $t$ to key $i$, $v_i$ the cached value, $W_o$ the output projection. The update is purely *additive* — pair $i$ contributes exactly one term, $\Delta h_{ti} = a_{ti} W_o v_i$ — so dropping pair $i$ removes precisely $\Delta h_{ti}$, of size $\|\Delta h_{ti}\| = a_{ti}\,\|W_o v_i\|$. There is the exact importance of a cached pair, and it factors into two pieces: how strongly the query attends to that key, $a_{ti}$, and how big a kick that value gives the output, $\|W_o v_i\|$. This is what both StreamingLLM's position rule and LagKV's coherence proxy miss — a pair can have a large, distinctive value but be ignored by every query, or be heavily attended yet carry a near-zero value; only the *product* matters, and only this product is tied to the residual stream the model actually reads downstream. I want to evict the pairs with the smallest product, because a small perturbation to the stream is a small perturbation to everything after it.

Half of this is free: $\|W_o v_i\|$ is computable from the cache right now. The trouble is entirely $a_{ti}$, with $a_{ti} = z_{ti}/\sum_j z_{tj}$ and $z_{ti} = \exp(q_t^\top k_i/\sqrt{d})$ — and the weights I care about belong to the *future* decode steps, whose queries $q_t$ do not exist yet. This is the same obstruction the whole task is built around, and it is why H2O and SnapKV read *past* attention as a stand-in; but past attention is the wrong signal (a key that mattered to tokens already seen need not be what the next thousand tokens need), and this hook never materializes the attention matrix anyway. So I cannot observe future $a_{ti}$ and cannot read it off a kernel. But I can *predict* it. I do not need the attention from one specific future query — I need a *typical* future query's attention, in expectation. That reframes the obstruction from "I do not have $q_t$" to "what is the distribution of future $q_t$, and what is $\mathbb{E}[z_{ti}]$ under it?"

I propose *Expected Attention*: estimate the distribution of future queries and score each key by its expected unnormalized attention, in closed form. The estimate rests on a property of these models that is otherwise idle: the pre-block hidden states feeding the attention layer are empirically zero-mean, unimodal, and close to Gaussian, $h \sim \mathcal{N}(\mu, \Sigma)$. Because the query is a linear map of the hidden state, $q = R W_Q h$ with $W_Q$ the query projection and $R$ the RoPE rotation, a Gaussian $h$ pushes forward to a Gaussian query — a linear map $q = A h$ of $h \sim \mathcal{N}(\mu, \Sigma)$ is $\mathcal{N}(A\mu, A\Sigma A^\top)$ — so $q_t \sim \mathcal{N}(R_t W_Q \mu,\; R_t W_Q \Sigma W_Q^\top R_t^\top)$. I estimate $\mu$ and $\Sigma$ by running the prompt's hidden states (which I already have at compression time) through $W_Q$ and taking the sample mean and covariance. This is the honest answer to LagKV's gsm8k blind spot: I am not scoring a coherence proxy of the cache, I am estimating the distribution of the queries that will actually do the reading.

There is a snag in the subscript: $R_t$ depends on position $t$, and I am averaging over many future positions, each with its own rotation. RoPE acts as $R_t x = x\cos_t + \text{rotate\_half}(x)\sin_t$, which I can write as a matrix $R_t = \text{diag}(\cos_t)\,\text{Id} + \text{diag}(\sin_t)\,P$ with $P$ the signed permutation implementing $\text{rotate\_half}$. Rather than commit to one representative offset, I average the rotation itself over the next $T$ positions, $\bar R = \tfrac{1}{T}\sum_{j=1}^{T} R_{t+j}$, and push the Gaussian through $\bar R W_Q$ to get a single position-averaged query $\bar q \sim \mathcal{N}(\bar\mu_q, \bar\Sigma_q)$ with $\bar\mu_q = \bar R W_Q \mu$ and $\bar\Sigma_q = \bar R W_Q \Sigma W_Q^\top \bar R^\top$. Note $\bar R$ is *not* a rotation: each $R_{t+j}$ is orthonormal but the average of rotations is a contraction — as the offsets spread, the per-frequency cos/sin entries average toward smaller magnitudes, so $\bar R$ shrinks high-frequency directions more than low ones. That is the right behavior, not a bug: directions whose phase churns fast across the future window get washed out because no future position agrees on them, while slow directions survive. So I keep $\bar R$ as-is and do not re-orthonormalize.

Now the payoff. With $\bar q \sim \mathcal{N}(\bar\mu_q, \bar\Sigma_q)$ and a fixed key $k_i$, I want $\hat z_i = \mathbb{E}_{\bar q}[\exp(\bar q^\top k_i/\sqrt{d})]$, which is exactly a moment-generating-function evaluation: for $X \sim \mathcal{N}(m, C)$ and fixed $s$, $\mathbb{E}[\exp(s^\top X)] = \exp(s^\top m + \tfrac{1}{2}s^\top C s)$. With $s = k_i/\sqrt{d}$,
$$\hat z_i = \exp\!\left( \frac{\bar\mu_q^\top k_i}{\sqrt{d}} + \frac{k_i^\top \bar\Sigma_q k_i}{2d} \right).$$
The Gaussian assumption is exactly what bought the closed form — the expectation of an exponential that would otherwise need sampling. The constants are forced by the algebra, not chosen: the first term is the ordinary attention logit with the query replaced by its mean (temperature $1/\sqrt{d}$), and the second comes from $\tfrac{1}{2}s^\top C s$ with $s = k/\sqrt{d}$, the two $1/\sqrt{d}$ factors giving $1/d$ and the MGF's $\tfrac{1}{2}$ giving the $/(2d)$. The covariance term does real work: by Jensen, $\mathbb{E}[\exp(\cdot)]$ exceeds $\exp(\mathbb{E}[\cdot])$ precisely by the spread, so a key aligned with a high-variance direction of the future-query distribution is boosted over its mean logit — a key *some* future queries will love even if the average query is lukewarm. Dropping the term gives the cheaper mean-only estimate; keeping it (the default) distinguishes "consistently moderate" from "occasionally strong," which are different eviction decisions. I softmax the log expected scores over the key dimension to turn them into expected attention *weights* $\hat a_i$ summing to one — the $a_{ti}$ slot of the importance formula — and substitute back: $\|\Delta\hat h_i\| \approx (\hat a_i + \varepsilon)\,\|v_i\|$. Two harness-specific notes: strictly the formula wants $\|W_o v_i\|$, but materializing $W_o v_i$ per value is expensive, so this fill uses $\|v_i\|$ as the value-magnitude proxy, keeping the value-size factor without the projection cost; and $\varepsilon$ is a floor (set to $0$ here) that would, if positive, let the value norm break ties among near-ignored keys — with $\varepsilon = 0$ the attention estimate dominates completely.

Two corrections, both tied to the first tokens, pointing opposite ways — the same $n_{\text{sink}}$ duality I have carried since StreamingLLM. The initial tokens carry massive-activation outliers and soak up attention regardless of content; if I let them into the sample mean and covariance they wreck the Gaussian estimate, dragging $\mu$ and inflating $\Sigma$. So I exclude the first $n_{\text{sink}}$ tokens from the *statistics*, and from the keys/values/hidden-states I score. But because the sinks are load-bearing for the model regardless of content, I must not *evict* them either: after scoring the body I pad the sinks back with a score guaranteed to top the list (the running max), so eviction always keeps them. Drop from the stats, force-keep in the cache. I compute the query statistics on the *pre-RoPE* queries ($W_Q h$, before rotation, via a projection helper that handles both `q_proj` and fused `qkv_proj` and applies `q_norm` if present), because I apply the rotation analytically through $\bar R$ afterward — keeping all position handling in the averaged rotation rather than baking a specific position into the sampled queries.

Mapping onto the hook, `score_tokens` reads only hidden states, keys, and values — never an attention tensor, which is the entire reason this method can run here while H2O/SnapKV cannot. It drops the sinks, gets the pre-RoPE query mean and (optionally) covariance, averages RoPE over the future window via the explicit cos/sin construction of $\bar R$ from `module.rotary_emb`, repeats the KV heads up to the query-head count ($\text{num\_attention\_heads} // \text{num\_key\_value\_heads}$) so grouped-query attention is handled, forms the log expected scores (mean term plus, if enabled, the covariance term), softmaxes over keys, averages the query heads sharing each KV head back down, multiplies by the value norm, and pads the sink scores at the front with `scores.max()`. `select_cache` is the plain top-k gather, and `rerotate_selected_keys = False` because this method keeps tokens at their original positions — it ranks by expected future attention, it does not roll a contiguous window — so decode continues from the true sequence length. The compression ratio is read from the plan and force-overridden by the harness.

I expect retained to stay $\sim 0.20$ and runtime near the prior rungs, perhaps a hair higher from the covariance einsum and the RoPE averaging — if it blows past the others, the covariance term is too expensive and I fall back to mean-only. The claim is that scoring by expected future attention to the model's output beats LagKV's coherence proxy. Concretely, repobench should recover above LagKV's 40.9 toward the anchor's 47.6, since the value-weighted expected-attention score restores the recent, heavily-attended code tokens LagKV's incoherence rank de-prioritized — the sharpest single test, because repobench is where LagKV regressed below StreamingLLM. Hotpotqa should hold or beat 31.6. The place I might *lose* is passage retrieval: LagKV's per-partition rank guarantees coverage across all 30 paragraphs, whereas a global expected-attention top-K could concentrate budget and miss the paragraph holding the needle, so I would not be shocked to see it come in below 60.4 — the known weakness of a non-quota'd content score. On gsm8k I expect no miracle, since the estimate is still built from prompt-query statistics, but any nudge above 2.0 confirms the output-tied signal is at least better than coherence on reasoning. The bar is the geometric mean across the five workloads: it must beat LagKV's, and the bet is that recovering repobench and holding hotpotqa outweighs any passage-retrieval slip.

```python
# EDITABLE region of custom_selection_eval.py (lines 40-101) — step 4: Expected Attention
class SelectionPolicy:
    """Expected Attention: estimate future-query attention before pruning."""

    method_name = "expected_attention"
    rerotate_selected_keys = False

    def repeat_kv(self, hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            bsz, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)

    def get_prerope_query_states(self, module, hidden_states):
        bsz, q_len, _ = hidden_states.shape
        num_heads = int(module.config.num_attention_heads)
        head_dim = int(module.head_dim)
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states)
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states)
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"Query projection not implemented for {module.__class__}.")
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        if hasattr(module, "q_norm"):
            query_states = module.q_norm(query_states)
        return query_states

    def avg_rope(self, module, mu, cov, q_len, n_future_positions):
        position_ids = torch.arange(q_len, q_len + n_future_positions, device=mu.device).unsqueeze(0)
        head_dim = int(module.head_dim)
        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]
        identity = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        perm = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
        half = head_dim // 2
        perm[half:, :half] = torch.eye(half, device=cos.device, dtype=cos.dtype)
        perm[:half, half:] = -torch.eye(half, device=cos.device, dtype=cos.dtype)
        rotation = (cos.unsqueeze(1) * identity + sin.unsqueeze(1) * perm).mean(dim=0).to(mu.device)
        mu = torch.matmul(mu, rotation.T)
        if cov is not None:
            cov = torch.matmul(rotation, torch.matmul(cov, rotation.T))
        return mu, cov

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,
            "n_future_positions": 512,
            "use_covariance": True,
            "use_value_norm": True,
            "epsilon": 0.0,
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        n_sink = int(plan.get("sink_tokens", 4))
        n_future = int(plan.get("n_future_positions", 512))
        use_covariance = bool(plan.get("use_covariance", True))
        use_vnorm = bool(plan.get("use_value_norm", True))
        epsilon = float(plan.get("epsilon", 0.0))
        assert keys.size(2) > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
        keys_body = keys[:, :, n_sink:]
        values_body = values[:, :, n_sink:]
        h = hidden_states[:, n_sink:]
        query_states = self.get_prerope_query_states(module, h)
        mean_query = query_states.mean(dim=2, keepdim=True)
        cov_query = None
        if use_covariance:
            centered_states = query_states - mean_query
            cov_query = torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states) / max(h.shape[1], 1)
        mean_query = mean_query.squeeze(2)
        mean_query, cov_query = self.avg_rope(module, mean_query, cov_query, hidden_states.shape[1], n_future)
        bsz, num_key_value_heads, q_len, dim = keys_body.shape
        num_key_value_groups = int(module.config.num_attention_heads) // num_key_value_heads
        repeated_keys = self.repeat_kv(keys_body, num_key_value_groups).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), repeated_keys).squeeze(2) / math.sqrt(dim)
        if use_covariance:
            scores += torch.einsum("bhin,bhij,bhjn->bhn", repeated_keys, cov_query, repeated_keys) / dim / 2
        scores = F.softmax(scores, dim=-1)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len).mean(dim=2)
        if use_vnorm:
            scores = (scores + epsilon) * values_body.norm(dim=-1)
        return F.pad(scores, (n_sink, 0), value=scores.max().item())

    def select_cache(self, module, keys, values, scores, n_kept):
        indices = scores.topk(n_kept, dim=-1).indices
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        selected_keys = keys.gather(2, gather_idx).contiguous()
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        selected_values = values.gather(2, gather_idx).contiguous()
        return selected_keys, selected_values
```
