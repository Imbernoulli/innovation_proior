**Problem.** LagKV recovered retrieval (passage retrieval 60.4) but regressed on code completion
(repobench 40.9, below StreamingLLM) and never touched reasoning (gsm8k 2.0), because its score is a
geometric *coherence proxy* with no tie to the model's output. I want to score a cached pair by its
*true effect on the residual stream* — still attention-free and query-free, since the hook passes no
attention matrix.

**Key idea.** With `h_out = h + sum_i a_i W_o v_i`, pair `i` adds `Δh_i = a_i W_o v_i`, so its exact
importance is `||Δh_i|| = a_i · ||W_o v_i||` — attention weight times value kick. The value norm is free;
the attention factor needs future queries that do not exist. Predict it in expectation: hidden states are
approximately Gaussian, so `q = R W_Q h` is Gaussian; average RoPE over the future window into one
contracting `R̄` to get a single query distribution `q̄ ~ N(μ̄_q, Σ̄_q)`; then the Gaussian MGF gives the
expected unnormalized score in closed form, `ẑ_i = exp(μ̄_q^T k_i/sqrt(d) + k_i^T Σ̄_q k_i/(2d))`. Softmax
over keys → expected attention weight `â_i`; multiply by the value norm → expected contribution. Keep the
top budget.

**Why it works.** Unlike LagKV's coherence proxy, this estimates the distribution of the queries that
will actually read the cache, so it is tied to the model output; the covariance term (a Jensen boost)
keeps keys some future queries will attend strongly even if the average query is lukewarm. Reads only
hidden states, keys, values — so it runs under SDPA where H2O/SnapKV cannot.

**Harness fit.** Statistics computed on pre-RoPE queries (`W_Q h`, before rotation; `q_norm` applied if
present); RoPE averaged analytically via `module.rotary_emb`. Grouped-query attention handled by repeating
KV heads to the query-head count and averaging back down. Value-magnitude uses `||v_i||` as a proxy for
`||W_o v_i||`. Sinks dropped from the *statistics* (outliers wreck the Gaussian fit) but force-kept in the
cache (padded with `scores.max()`). `rerotate_selected_keys = False` — tokens keep original positions, so
no re-rotation. `compression_ratio` read from the plan, force-overridden by the harness.

**Hyperparameters.** `sink_tokens = 4`; `n_future_positions = 512` (the future window `T`);
`use_covariance = True` (the Equation-7 second-order term); `use_value_norm = True`; `epsilon = 0.0` (let
the attention estimate dominate).

**What to watch.** Retained ~0.20; runtime near the prior rungs (a touch higher from the covariance einsum
+ RoPE averaging — if it blows up, fall back to mean-only). Expect repobench to recover above 40.9 toward
the anchor's 47.6 (the sharpest test — value-weighted attention restores recent code tokens); hotpotqa to
hold/beat 31.6; passage retrieval possibly *below* LagKV's 60.4 (a global top-K lacks LagKV's
per-paragraph quota); gsm8k maybe a nudge above 2.0. The bar: beat LagKV on the geometric mean.

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
