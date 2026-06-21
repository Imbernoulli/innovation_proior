The problem is to make long-context attention in a frozen causal LLM cheap enough to run at inference without retraining, while staying under a hard cap on the fraction of query-key pairs that may receive nonzero attention. Full softmax attention is quadratic in the sequence length and, for contexts like 64k tokens, consumes the majority of decode latency. The natural hope is to exploit the empirical fact that softmax attention is sparse: for any given query, only a small subset of previous keys carries almost all the mass. But the important keys are query-dependent and change from prompt to prompt, so a fixed sparse mask cannot capture them. Worse, finding the important keys by scoring every individual query-key pair is itself quadratic, which defeats the purpose. Existing content-blind baselines such as sliding window and StreamingLLM are fast and contiguous, but they cannot route attention to relevant evidence in the middle of a long context. Content-adaptive methods such as Quest select per head, which inflates memory traffic under grouped-query attention, and methods such as InfLLM rely on heuristic representative tokens rather than a stable block-level summary. What is needed is a selector that is query-driven, block-granular, and parameter-free, so it drops into a frozen model and maps onto efficient contiguous memory access.

The method is Block-Top-K. It treats attention sparsity as a block-level property rather than a token-level property. The key observation is that important attention entries are spatially clustered: neighboring keys tend to be similarly relevant, so the decision of which keys to keep can be made per contiguous block. The block is also the right unit for the hardware, because GPUs and Tensor Cores prefer contiguous block-shaped operands to scattered index gathers. Block-Top-K first partitions the queries and keys into contiguous blocks of size 64. It summarizes each block by the mean of its vectors: mean-pooled keys as block representatives and mean-pooled queries as the block-level query. Because the keys within a block point in similar directions, the mean is a low-variance, parameter-free summary that requires no learned projection. The block-level importance score is then the dot product between the mean query of block i and the mean key of block j, scaled by the usual attention scale. This produces a score matrix of size n_blocks by n_blocks, a factor of block_size squared smaller than the full attention score matrix, and the expensive token-level logits are never materialized for selection.

For each query block, Block-Top-K keeps K total key blocks. The diagonal block, which contains the query's own local neighborhood, is force-included unconditionally because local context is almost always relevant and the coarse proxy might miss it. The remaining K minus one slots are filled by the highest-scoring off-diagonal causal blocks. The diagonal is masked out of the ranking so it does not waste a top-k slot. Causality is enforced twice: first at block granularity during selection by forbidding future blocks, and again at token granularity during attention by applying the strict lower-triangle mask within and especially on the diagonal block. The keep count K is not hand-tuned; it is derived from the density budget by solving a quadratic that counts causal block-pairs. With n blocks and budget rho, the causal block-level density is K times (2n minus K plus 1) divided by n times (n plus 1), so K is the smaller feasible root of the corresponding quadratic, floored to stay safely under budget. For example, with 8k tokens, block size 64, and budget 0.25, this gives 17 blocks out of 128, or about 24.7 percent block density. The actual reported density is measured from the realized token-level mask, which trims padding and halves the diagonal block, so the harness contract is honored exactly.

The implementation below pads the sequence to a multiple of the block size, computes mean-pooled block representatives, scores blocks in float32 for stability, selects top off-diagonal blocks plus the diagonal, expands the block decision to a token keep-mask, applies the strict causal triangle, reports the measured density, and finally runs masked-softmax attention over the kept query-key pairs. A production kernel would gather only the selected contiguous blocks into SRAM; the masked-softmax form here computes the identical output in pure PyTorch and is intended to specify the behavior.

```python
import math
import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    """Content-adaptive block-sparse top-K attention for a frozen causal LLM.

    For each query block, score all causal key blocks by mean-pooled-query dot
    mean-pooled-key, keep K total blocks by taking K-1 top off-diagonal blocks
    plus the always-kept diagonal block, and run masked-softmax attention over
    the kept (q, k) pairs.
    """

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

        # Pad sequence to a multiple of BLOCK so block-pooling is exact.
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

        # Cheap block-pair importance proxy (fp32 for stable masking/softmax).
        scores = torch.einsum('bhmd,bhnd->bhmn',
                              q_blocks.float(), k_blocks.float()) * scale

        idx = torch.arange(n_blocks, device=q.device)
        if is_causal:
            causal_blk = idx[:, None] >= idx[None, :]
            scores = scores.masked_fill(~causal_blk, float('-inf'))

        # Remove diagonal from ranking; it is force-included below.
        diag_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=q.device)
        diag_mask.fill_diagonal_(True)
        scores_nodiag = scores.masked_fill(diag_mask, float('-inf'))

        # Derive per-query-block keep count K from the density budget.
        # Kept causal block-pairs = K(2n-K+1)/2; denom = n(n+1)/2.
        n_b = n_blocks
        if is_causal:
            disc = max(0.0, (2 * n_b + 1) ** 2
                       - 4 * self.density_budget * n_b * (n_b + 1))
            K_top = max(1, int(((2 * n_b + 1) - math.sqrt(disc)) / 2))
        else:
            K_top = max(1, int(round(self.density_budget * n_b)))
        kk = max(0, min(K_top - 1, n_b - 1))

        if kk > 0:
            topk_idx = scores_nodiag.topk(kk, dim=-1).indices
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

        # Expand block decision to a token-level keep mask and trim padding.
        tok_blk = torch.arange(Npad, device=q.device) // Bk
        token_keep = block_keep[:, :, tok_blk, :][:, :, :, tok_blk]
        token_keep = token_keep[:, :, :N, :N]
        if is_causal:
            tidx = torch.arange(N, device=q.device)
            token_keep = token_keep & (tidx[:, None] >= tidx[None, :])

        # Report density measured from the realized mask.
        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(token_keep[0, 0].sum().item()) / max(denom, 1.0)

        # Masked-softmax attention over the kept (q, k) pairs.
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~token_keep, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
```
