# Block-Top-K sparse attention, distilled

Block-Top-K is a content-adaptive, parameter-free sparse attention for a frozen causal LLM at
inference time. Block importance is computed from **mean-pooled keys** and **mean-pooled
queries**, so it needs no learned compression and runs on a frozen backbone. For each query
block it scores all causal key blocks, keeps `K` total blocks by taking the top `K-1`
off-diagonal blocks plus the always-included diagonal (current) block, and runs
masked-softmax attention over the kept query-key pairs.

## Problem it solves

Full softmax attention is `O(N^2)` and dominates long-context latency. Goal: drop a sparse
attention into a pretrained model's forward (no retraining) under a fixed density budget
(fraction of `(q,k)` pairs allowed nonzero attention, default 0.25), in parallel-forward mode,
preserving long-context retrieval/QA quality. The important keys are query-dependent and
dynamic, so a static mask fails; scoring every `(q,k)` pair to find them is itself `O(N^2)`.

## Key idea

Attention mass is sparse, dynamic, and **spatially clustered** (nonzero entries arrive in
contiguous blocks). So judge importance at block granularity — which also matches the hardware
(contiguous block reads are fast; Tensor-Core friendly) — chosen per query (content-adaptive).

1. **Block representatives.** Chop keys into contiguous blocks of size `B`; summarize each by
   the **mean** of its key vectors (valid because a clustered block's keys point similarly; the
   mean is the minimum-variance summary; parameter-free, so it works on a frozen model).
   Mean-pool the queries per block too, so the selection score matrix is `(N/B) x (N/B)` — a
   factor `B^2` cheaper than dense, without materializing the full `(N x N)` logits for
   selection.
2. **Block score.** `s[i,j] = (mean q of block i) . (mean k of block j) * scale`,
   `scale = 1/sqrt(D)`, computed in fp32.
3. **Causal block mask.** Set `s[i,j] = -inf` for `j > i`.
4. **Force-include the diagonal block.** Local context is always relevant and the proxy may
   miss it, so mask the diagonal out of the ranking (so it never spends an off-diagonal slot),
   take the top-`kk = K-1` off-diagonal blocks, and append the diagonal. This plays the role
   the dedicated sliding-window branch plays in the trainable design.
5. **Budget -> K.** Keep `K` blocks per query block including the diagonal. Under causal AND,
   the kept causal block-pairs total `sum_{i=0}^{n-1} min(K, i+1) = K(2n-K+1)/2` (for `K<=n`),
   and the causal denominator is `n(n+1)/2`, so block-level density is
   `K(2n-K+1)/(n(n+1))`. Set equal to budget `rho`:
   `K^2 - (2n+1)K + rho*n(n+1) = 0`, take the **smaller** root (fewest blocks)
   `K = ((2n+1) - sqrt((2n+1)^2 - 4*rho*n(n+1)))/2`, floor, clamp `K>=1`.
   At `B=64, N=8k` (`n=128`), `rho=0.25`: `K≈17.2 → 17` blocks → density `0.247 < 0.25`.
   (Non-causal: `K ≈ round(rho*n)`.)
6. **Expand and attend.** Expand the block keep-decision to a token keep-mask
   (`token // B`), trim padding back to `N`, AND with the strict token causal triangle
   (`q_pos >= k_pos`, which also halves the diagonal block). Report
   `last_density = (kept pairs)/(N(N+1)/2)` **measured from the realized mask** (not the block
   algebra). Then masked-softmax over the full logits: `q.k^T*scale`, `-inf` on non-kept pairs,
   softmax, zero out empty rows, times `v`, cast back to input dtype. (A Triton kernel would
   instead gather only the selected contiguous blocks into SRAM; the masked-softmax form
   computes the identical output in pure PyTorch.)

## Relation to prior methods

- **Sliding window / sink+window (StreamingLLM):** static, content-blind — cannot route to
  mid-context evidence. Block-Top-K is query-driven; the diagonal-force-include recovers the
  local/window role.
- **Quest:** query-aware blockwise selection via channelwise key min/max bound, **per head** —
  inflates GQA KV memory (union over the group). The full trainable/kernel setting handles
  this by sharing block scores within a GQA group; this inference module is called after the
  harness has replicated GQA heads and therefore selects per replicated head.
- **InfLLM:** block memory with representative/mean-key units; supplies the parameter-free
  mean-pooled-key representative used here.
- **MInference:** measures the spatial clustering (nearest-nonzero distance ~5) that justifies
  block granularity.

## Working code

```python
import math
import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    """Mean-pooled block scoring with K-1 off-diagonal blocks plus diagonal."""

    BLOCK = 64

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        self.last_density = None

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        Bk = self.BLOCK
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        # Pad to a multiple of BLOCK so block-pooling is exact.
        Npad = ((N + Bk - 1) // Bk) * Bk
        if Npad != N:
            pad = Npad - N
            qp = torch.nn.functional.pad(q, (0, 0, 0, pad))
            kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
        else:
            qp, kp = q, k
        n_blocks = Npad // Bk

        # Mean-pooled q / k per block: (B, H, n_blocks, D).
        q_blocks = qp.view(B, H, n_blocks, Bk, D).mean(dim=3)
        k_blocks = kp.view(B, H, n_blocks, Bk, D).mean(dim=3)

        # Block-pair importance (fp32 for stable masking/softmax).
        scores = torch.einsum('bhmd,bhnd->bhmn',
                              q_blocks.float(), k_blocks.float()) * scale

        idx = torch.arange(n_blocks, device=q.device)
        if is_causal:
            causal_blk = idx[:, None] >= idx[None, :]
            scores = scores.masked_fill(~causal_blk, float('-inf'))

        # Force-include diagonal: drop it from the ranking, append it after.
        diag_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=q.device)
        diag_mask.fill_diagonal_(True)
        scores_nodiag = scores.masked_fill(diag_mask, float('-inf'))

        # Per-query-block keep count K from the density budget (smaller root).
        n_b = n_blocks
        if is_causal:
            disc = max(0.0, (2 * n_b + 1) ** 2
                      - 4 * self.density_budget * n_b * (n_b + 1))
            K_top = max(1, int(((2 * n_b + 1) - math.sqrt(disc)) / 2))
        else:
            K_top = max(1, int(round(self.density_budget * n_b)))
        kk = max(0, min(K_top - 1, n_b - 1))

        if kk > 0:
            topk_idx = scores_nodiag.topk(kk, dim=-1).indices       # (B,H,n_b,kk)
        else:
            topk_idx = torch.empty(B, H, n_blocks, 0,
                                   dtype=torch.long, device=q.device)
        diag_idx = idx.view(1, 1, n_blocks, 1).expand(B, H, n_blocks, 1)
        sel = torch.cat([topk_idx, diag_idx], dim=-1)

        block_keep = torch.zeros(B, H, n_blocks, n_blocks,
                                 dtype=torch.bool, device=q.device)
        block_keep.scatter_(-1, sel, True)
        if is_causal:
            block_keep = block_keep & causal_blk

        # Expand to token-level keep mask, trim padding.
        tok_blk = torch.arange(Npad, device=q.device) // Bk
        token_keep = block_keep[:, :, tok_blk, :][:, :, :, tok_blk]
        token_keep = token_keep[:, :, :N, :N]
        if is_causal:
            tidx = torch.arange(N, device=q.device)
            token_keep = token_keep & (tidx[:, None] >= tidx[None, :])

        # Density measured from the realized mask (harness contract).
        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(token_keep[0, 0].sum().item()) / max(denom, 1.0)

        # Masked-softmax attention over kept (q, k) pairs.
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~token_keep, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
```
