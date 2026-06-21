A diffusion-based large language model generates text by denoising a fully masked response over many steps. At each step a bidirectional Transformer re-reads the entire prompt-plus-response sequence, predicts clean tokens at every masked position, commits the most confident ones, and re-masks the rest. Because every step is a full forward pass, generating length L costs roughly O(L^3), against the O(L^2) of an autoregressive model that simply appends one token at a time. The obvious remedy, the autoregressive Key-Value cache, cannot be reused here: without a causal mask, a token that was already decoded still changes its Key and Value as neighboring positions get unmasked, and the decoding order is not fixed, so there is no predetermined next position to refresh.

The failure of exact reuse points toward approximate reuse, but approximate reuse has to be heterogeneous to be safe. Measurements of adjacent-step feature similarity show two clear patterns. First, the prompt region is almost perfectly static across steps: the prompt tokens never change, so their internal representations drift only very slowly. Second, the response region is a mixture: most response tokens are highly similar to their previous step, while a small minority move sharply. A single global cache interval therefore fails. If the interval is long, the few moving tokens are corrupted by stale features; if it is short, the stable prompt and the stable majority of the response are recomputed for no gain. The right structure is a split cache with different schedules for the static prompt and the dynamic response, plus a way to identify the response tokens that actually changed without paying the full cost of recomputing them.

The method is dLLM-Cache. It is a training-free adaptive caching framework that maintains per-layer caches for the prompt and the response, each holding the four heavy intermediate features: Key, Value, attention output, and FFN output. The prompt cache is refreshed only on a long interval K_p, on the order of tens to a hundred steps, because the prompt is quasi-static. The response cache is fully refreshed on a short interval K_r, on the order of a few steps, to bound how stale any response feature can become. Between those full refreshes, only the most-changed response tokens are recomputed, selected by a cheap proxy called V-verify.

V-verify works because the adjacent-step similarity of a token's Value vector strongly correlates with the similarity of its downstream attention output and FFN output. The Value projection is inexpensive: just a layer normalization and a linear matmul, with no quadratic attention and no large FFN. At each in-between step, all response Values are projected and compared to the cached Values by cosine similarity. The floor of rho times the response length tokens with the lowest similarity are the movers; their Key, attention output, and FFN output are recomputed and scattered back into the cache, while the remaining tokens reuse their cached features unchanged. Because the fresh Values for all response tokens were already computed for scoring, the entire cached response Value tensor is overwritten at no extra cost, giving the next step a fully up-to-date baseline. Cosine similarity is used rather than Euclidean distance so that selection depends on directional semantic change, not on differences in vector magnitude across tokens or dimensions.

The per-step, per-layer logic therefore has four cases. On the initial step everything is computed from scratch and split into prompt and response caches. After that, at each layer, two booleans are tested: whether this is a prompt-refresh step and whether this is a response-refresh step. If both are true, both caches are fully recomputed, which acts as a periodic error reset. If only the response interval fires, the whole response is recomputed against the cached prompt. If only the prompt interval fires, the prompt is recomputed and the response receives a V-verify partial update. If neither fires, the prompt is reused and the response receives the same partial update, which is the common case. The first layer is always fully refreshed as a cheap way to keep the inputs to deeper cached layers honest.

The error stays controlled for two reasons. Without any refresh, the denoising map is expansive, so approximation error would grow geometrically across steps. The short response refresh interval K_r resets this error to near zero periodically, bounding the worst-case drift to a finite window. V-verify further tightens the bound inside each window, because the rho most erroneous tokens are recomputed each step, so only the cached majority can carry error forward. Storage is also modest: one version of four tensors per layer, flat in the number of denoising steps, adding only a few percent of extra memory on an 8B model.

```python
import torch
import torch.nn.functional as F


def refresh_index(v_new: torch.Tensor, v_cached: torch.Tensor, transfer_ratio: float):
    """V-verify: per-token cosine similarity of current vs cached Value;
    return indices of the `transfer_ratio` fraction with LOWEST similarity (most changed)."""
    num_replace = int(v_new.size(1) * transfer_ratio)
    cos_sim = F.cosine_similarity(v_new, v_cached, dim=-1)              # [B, gen_len]
    return torch.topk(cos_sim, k=num_replace, largest=False).indices    # [B, num_replace]


def dllm_cache_block_forward(self, x, attention_bias=None):
    """One bidirectional Transformer block under dLLM-Cache.
    `dLLMCache` exposes: layer_id, prompt_length, current_step,
    prompt_interval_steps (K_p), gen_interval_steps (K_r), transfer_ratio (rho);
    refresh_prompt()/refresh_gen(); set_cache/get_cache over
    cache_type in {'prompt','gen'} and feature_name in {'kv_cache','attn','mlp'}."""
    fc = dLLMCache()
    fc.update_step(self.layer_id)
    p = fc.prompt_length
    x_prompt, x_gen = x[:, :p, :], x[:, p:, :]
    B, seq_len, dim = x.shape

    refresh_prompt = fc.refresh_prompt(self.layer_id) or self.layer_id == 0
    refresh_gen = fc.refresh_gen(self.layer_id) or self.layer_id == 0
    transfer = 0.0 < fc.transfer_ratio <= 1.0

    def project(z):
        h = self.attn_norm(z)
        return self.q_proj(h), self.k_proj(h), self.v_proj(h)

    def attention(q, k, v, q_index=None):
        att, _ = self.attention(q, k, v, attention_bias, q_index=q_index)
        return att

    def compute_mlp(z):
        h = self.ff_norm(z)
        return self.ff_out(self.act(self.ff_proj(h)) * self.up_proj(h))   # SwiGLU

    # ---------- attention sublayer ----------
    if refresh_gen and refresh_prompt:                                    # Case 1: full refresh / init
        q, k, v = project(x)
        fc.set_cache(self.layer_id, "kv_cache",
                     {"k": k[:, :p], "v": v[:, :p]}, cache_type="prompt")
        fc.set_cache(self.layer_id, "kv_cache",
                     {"k": k[:, p:], "v": v[:, p:]}, cache_type="gen")
        att = attention(q, k, v)
        fc.set_cache(self.layer_id, "attn", att[:, :p], cache_type="prompt")
        fc.set_cache(self.layer_id, "attn", att[:, p:], cache_type="gen")

    elif refresh_gen and not refresh_prompt:                             # Case 2: response-only refresh
        q, k_gen, v_gen = project(x_gen)
        fc.set_cache(self.layer_id, "kv_cache", {"k": k_gen, "v": v_gen}, cache_type="gen")
        kv_p = fc.get_cache(self.layer_id, "kv_cache", cache_type="prompt")
        k = torch.cat([kv_p["k"], k_gen], dim=1)
        v = torch.cat([kv_p["v"], v_gen], dim=1)
        att_gen = attention(q, k, v)
        fc.set_cache(self.layer_id, "attn", att_gen, cache_type="gen")
        att = torch.cat([fc.get_cache(self.layer_id, "attn", cache_type="prompt"), att_gen], dim=1)

    elif refresh_prompt and not refresh_gen:                             # Case 3: prompt refresh + adaptive
        q_p, k_p, v_p = project(x_prompt)
        fc.set_cache(self.layer_id, "kv_cache", {"k": k_p, "v": v_p}, cache_type="prompt")
        kv_gen = fc.get_cache(self.layer_id, "kv_cache", cache_type="gen")
        att_gen_c = fc.get_cache(self.layer_id, "attn", cache_type="gen")
        if transfer:
            x_gen_n = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_n)                                  # V-verify: all response Values
            idx = refresh_index(v_gen, kv_gen["v"], fc.transfer_ratio)
            idx_e = idx.unsqueeze(-1).expand(-1, -1, dim)
            sel = torch.gather(x_gen_n, 1, idx_e)
            q_sel, k_sel = self.q_proj(sel), self.k_proj(sel)
            kv_gen["v"] = v_gen                                           # overwrite ALL Values (free)
            kv_gen["k"].scatter_(1, idx_e, k_sel)                         # scatter selected K
            fc.set_cache(self.layer_id, "kv_cache", kv_gen, cache_type="gen")
        k = torch.cat([k_p, kv_gen["k"]], dim=1)
        v = torch.cat([v_p, kv_gen["v"]], dim=1)
        if transfer:
            prompt_pos = torch.arange(p, device=x.device).unsqueeze(0).expand(B, -1)
            q_all = torch.cat([q_p, q_sel], dim=1)
            pos = torch.cat([prompt_pos, idx + p], dim=1)                 # true positions for RoPE
            att_all = attention(q_all, k, v, q_index=pos)
            att_p = att_all[:, :p]
            att_gen_c.scatter_(1, idx_e, att_all[:, p:])
            fc.set_cache(self.layer_id, "attn", att_gen_c, cache_type="gen")
        else:
            att_p = attention(q_p, k, v,
                              q_index=torch.arange(p, device=x.device).unsqueeze(0).expand(B, -1))
        fc.set_cache(self.layer_id, "attn", att_p, cache_type="prompt")
        att = torch.cat([att_p, att_gen_c], dim=1)

    else:                                                                # Case 4: reuse prompt + adaptive
        att_gen_c = fc.get_cache(self.layer_id, "attn", cache_type="gen")
        if transfer:
            x_gen_n = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_n)                                  # V-verify
            kv_gen = fc.get_cache(self.layer_id, "kv_cache", cache_type="gen")
            kv_p = fc.get_cache(self.layer_id, "kv_cache", cache_type="prompt")
            idx = refresh_index(v_gen, kv_gen["v"], fc.transfer_ratio)
            idx_e = idx.unsqueeze(-1).expand(-1, -1, dim)
            sel = torch.gather(x_gen_n, 1, idx_e)
            q_sel, k_sel = self.q_proj(sel), self.k_proj(sel)
            kv_gen["v"] = v_gen                                           # full Value overwrite
            kv_gen["k"].scatter_(1, idx_e, k_sel)                         # selected K scatter
            fc.set_cache(self.layer_id, "kv_cache", kv_gen, cache_type="gen")
            k = torch.cat([kv_p["k"], kv_gen["k"]], dim=1)
            v = torch.cat([kv_p["v"], kv_gen["v"]], dim=1)
            att_sel = attention(q_sel, k, v, q_index=idx + p)            # selected queries over full seq
            att_gen_c.scatter_(1, idx_e, att_sel)
            fc.set_cache(self.layer_id, "attn", att_gen_c, cache_type="gen")
        att = torch.cat([fc.get_cache(self.layer_id, "attn", cache_type="prompt"), att_gen_c], dim=1)

    x = x + self.dropout(att)
    og_x = x
    x_prompt, x_gen = x[:, :p, :], x[:, p:, :]

    # ---------- FFN sublayer (same four cases) ----------
    if refresh_gen and refresh_prompt:
        x = compute_mlp(x)
        fc.set_cache(self.layer_id, "mlp", x[:, p:], cache_type="gen")
        fc.set_cache(self.layer_id, "mlp", x[:, :p], cache_type="prompt")
    elif refresh_gen and not refresh_prompt:
        x_gen = compute_mlp(x_gen)
        fc.set_cache(self.layer_id, "mlp", x_gen, cache_type="gen")
        x = torch.cat([fc.get_cache(self.layer_id, "mlp", cache_type="prompt"), x_gen], dim=1)
    elif refresh_prompt and not refresh_gen:
        x_gen_c = fc.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            sel = torch.gather(x_gen, 1, idx_e)
            both = compute_mlp(torch.cat([x_prompt, sel], dim=1))
            x_prompt = both[:, :p]
            x_gen_c.scatter_(1, idx_e, both[:, p:])
            fc.set_cache(self.layer_id, "mlp", x_gen_c, cache_type="gen")
        else:
            x_prompt = compute_mlp(x_prompt)
        fc.set_cache(self.layer_id, "mlp", x_prompt, cache_type="prompt")
        x = torch.cat([x_prompt, x_gen_c], dim=1)
    else:
        x_gen_c = fc.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            sel = torch.gather(x_gen, 1, idx_e)
            x_gen_c.scatter_(1, idx_e, compute_mlp(sel))                  # only movers through FFN
            fc.set_cache(self.layer_id, "mlp", x_gen_c, cache_type="gen")
        x = torch.cat([fc.get_cache(self.layer_id, "mlp", cache_type="prompt"), x_gen_c], dim=1)

    x = self.dropout(x)
    return og_x + x, None
```
