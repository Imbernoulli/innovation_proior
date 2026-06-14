# dLLM-Cache, distilled

dLLM-Cache is a **training-free adaptive caching framework** for diffusion language-model
(dLLM) inference. It accelerates the denoising rollout by reusing intermediate Transformer
features across adjacent steps, exploiting the asymmetric dynamics of prompt versus response:
the static prompt is cached on a **long interval**, the dynamic response is fully refreshed on
a **short interval** and, in between, only its **most-changed tokens** are recomputed — those
selected cheaply by the cosine similarity of their Value vectors (**V-verify**).

## Problem it solves

A dLLM (e.g. LLaDA-8B, Dream-7B) generates by denoising a fully masked response with a
**bidirectional** Transformer over `K` steps, each a full forward pass over the whole
prompt-plus-response sequence — roughly `O(N^3)` to generate length `N`, versus `O(N^2)` for
autoregressive models with a KV cache. The AR KV cache does not transfer: with no causal mask,
a committed token's Key/Value keep changing as other positions unmask (timestep-variant K/V),
and decoding order is non-sequential. The goal is to cut the redundant per-step recomputation
without retraining or changing the host model.

## Key idea

Two diagnostic facts about an existing dLLM's rollout drive the design:

1. **Prompt vs response redundancy.** Adjacent-step cosine similarity of `K`, `V`, attention
   output (`AttnOut`), and FFN output (`FFNOut`) is uniformly very high in the **prompt** region
   (it never changes), and mixed in the **response** region — only a *small fraction* of response
   tokens are significantly dissimilar each step. A single uniform reuse interval therefore
   fails: it either staleness-corrupts the few movers or refreshes the stable majority for
   nothing.
2. **V predicts the downstream change.** For response tokens, the adjacent-step similarity of
   the cheap **Value** projection strongly correlates with the similarity of the expensive
   `AttnOut` and `FFNOut`. So the Value change is an early, cheap proxy for which tokens'
   downstream features moved.

From these: maintain per layer `l` a **Prompt Cache** `C_p` and a **Response Cache** `C_r`,
each holding `{K, V, AttnOut, FFNOut}`. Three hyperparameters: prompt interval `K_p`, response
interval `K_r` (with `K_p ≫ K_r`), and adaptive update ratio `ρ ∈ [0,1]`.

- **Long-interval prompt caching:** recompute prompt features every `K_p` steps (e.g. ~100),
  reuse otherwise.
- **Short-interval response caching with adaptive partial update:** fully refresh the response
  every `K_r` steps (e.g. ~5); otherwise do **V-verify** — project all response Values
  (`O(r d^2)`), score `s_j = cos(v_{r,j}^{new}, ṽ_{r,j})`, select the `⌊ρ|y|⌋` tokens with the
  **lowest** `s_j`, recompute their `Q/K/AttnOut/FFNOut` and scatter them back, and reuse the
  cache for the rest. The entire cached `V_r` is **overwritten** with the freshly computed
  `V_r^{new}` (already available from scoring, so free); `K`, `AttnOut`, `FFNOut` are scattered
  only for the selected rows.

## Why each choice

- **Cache four features, not just KV:** the heavy per-layer cost is the attention output and
  the FFN; caching only K/V would still pay them. (KV-only schemes leave this on the table.)
- **Select lowest similarity (most changed):** reusing an unchanged token is near-lossless;
  the stale cache is most wrong exactly for the tokens that moved — so recompute movers, reuse
  stayers.
- **V as the proxy:** the Value is what attention reads out, so its change forecasts the
  `AttnOut`/`FFNOut` change; the Value projection is linear and `n^2`-free, so deciding
  who-to-update costs far less than updating.
- **Cosine, not L2:** captures directional/semantic change and is scale-invariant; L2 conflates
  magnitude and lets large-norm tokens dominate the selection.
- **Overwrite all of `V_r`:** it's computed for scoring anyway, and a fresher Value baseline
  improves the next step's selection.
- **Per-layer, fixed `ρ`:** the selected set is stable across adjacent layers (Lipschitz: if
  `|s_j^l − s_j^{l+1}| ≤ ε` is below the boundary margin `Δ`, bottom-`ρ` membership is
  preserved), so per-layer selection is consistent with the cache; a flat `ρ` keeps the update
  budget simple across depth.
- **`ρ ≈ 0.25`:** balances the fixed overhead of *initiating* selective recompute (gather/scatter,
  kernel launches — a latency floor independent of token count) against the dynamic compute saved.

## Algorithm (per denoising step, per layer `l`)

```
At k = K (init): compute all features; split into C_p (prompt) and C_r (response).
For k = K-1 .. 1, at each layer l:
    refresh_prompt = ((current_step - 1) mod K_p == 0)   or  l == 0
    refresh_gen    = ((current_step - 1) mod K_r == 0)   or  l == 0
    Case (refresh_prompt, refresh_gen):
      (T, T): full refresh — recompute prompt+response, store C_p, C_r       # periodic error reset
      (F, T): recompute whole response, reuse prompt cache
      (T, F): recompute prompt; V-verify adaptive update of response
      (F, F): reuse prompt cache; V-verify adaptive update of response       # the common step
    V-verify adaptive update:
      V_r^new = V_proj(LN(x_r))                          # all response tokens (cheap)
      s_j     = cos(V_r^new[j], C_r.V[j])
      I       = indices of floor(rho*|y|) tokens with lowest s_j
      recompute Q/K/AttnOut/FFNOut for tokens in I (attending over full [C_p.K|C_r.K], [C_p.V|C_r.V])
      scatter K, AttnOut, FFNOut at I into C_r;  overwrite C_r.V <- V_r^new
```

## Complexity and storage

Baseline over `K` denoising steps: `K·T·(8nd^2 + 4n^2 d + 6ndm)` (FFN `6ndm` for a
3-matmul SwiGLU). dLLM-Cache:

```
FLOPs ≈ (K/K_p)·T·(8nd^2 + 4n^2 d + 6ndm)                                   # full refreshes
      + (K/K_r − K/K_p)·T·(8rd^2 + 4rnd + 6rdm)                            # response-only refreshes
      + K·(1 − 1/K_r)·T·(2rd^2 + 8r̂d^2 + 4r̂nd + 6r̂dm)                     # adaptive updates
```
with prompt/response lengths `p, r` (`n = p + r`) and `r̂ = ρ·r`. The dominant reductions:
attention `4n^2 d → 4r̂nd`, FFN `6ndm → 6r̂dm`; V-verify adds only the linear `2rd^2`.

Storage: four `T×d` tensors per layer × `L` layers = `O(4·L·T·d)` (`2·4·L·T·d` bytes in
bf16), one version per layer — flat in `K`.

## Error bound (why it stays stable)

Abstract a step as `y_{k-1} ≈ α_k y_k + (1−α_k)F(y_k)`, `F` Lipschitz with constant `L_F`. With
error `δ_k = y_k − ỹ_k`:
`‖δ_{k-1}‖ ≤ (α_k + (1−α_k)L_F)‖δ_k‖ = C_k‖δ_k‖`, and `C_k > 1` for expressive `F` (`L_F > 1`)
— error amplifies geometrically without intervention. Two controls compose:
- **Periodic refresh `K_r`** sets `δ ≈ 0` every `K_r` steps, so the within-window peak is bounded
  by `‖δ_{k_0−j}‖ ≤ (∏ C)·‖δ_{k_0}‖`, `j < K_r` → `O(C^{K_r})`.
- **V-verify** recomputes the worst `ρ` tokens each step, so only the cached `(1−ρ)` fraction's
  error amplifies: `‖δ_{k-1}‖ ≤ C_k‖P_cache(δ_k)‖ + ε_step`, with
  `‖P_cache(δ_k)‖ ≤ ‖δ_k‖` and strict shrinkage when the selected tokens carry nonzero error,
  giving effective `C'_k < C_k` on those steps and a tighter peak `O((C')^{K_r})`.

## Working code (per-layer block, faithful to the canonical implementation)

```python
import torch
import torch.nn.functional as F
from dllm_cache.cache import dLLMCache


def refresh_index(v_new: torch.Tensor, v_cached: torch.Tensor, transfer_ratio: float):
    """V-verify: per-token cosine similarity of current vs cached Value;
    return indices of the `transfer_ratio` fraction with LOWEST similarity (most changed)."""
    num_replace = int(v_new.size(1) * transfer_ratio)
    cos_sim = F.cosine_similarity(v_new, v_cached, dim=-1)              # [B, gen_len]
    return torch.topk(cos_sim, k=num_replace, largest=False).indices    # [B, num_replace]


def dllm_cache_block_forward(self, x, attention_bias=None):
    """One bidirectional Transformer block under dLLM-Cache, monkey-patched onto the host
    dLLM block. The singleton `dLLMCache` exposes: layer_id, prompt_length, current_step,
    prompt_interval_steps (K_p), gen_interval_steps (K_r), transfer_ratio (rho);
    refresh_prompt()/refresh_gen() (true on the interval); set_cache/get_cache over
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

Typical settings on LLaDA-8B: `K_p ≈ 50–100`, `K_r ≈ 4–8`, `ρ = 0.25`, low-confidence remasking,
semi-autoregressive block decoding. The scheme is training-free — it only reorganizes *which*
features each step recomputes — and drops directly into the existing denoising rollout.
