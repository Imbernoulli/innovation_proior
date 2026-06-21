StreamingLLM did what I predicted, and across three rungs the diagnosis is now isolated. Qasper F1 came up from BigBird's $0.0871$ to $0.1103$ — within a hair of the oracle's $0.1406$, about 78% recovered — MultiFieldQA-EN stayed in its band at $0.213$, and crucially every density dropped to a flat $0.2344$, comfortably under the $0.25 + 0.02$ ceiling where BigBird had been over budget on two of three. The static floor is now honest. But NIAH stayed pinned at $0.2$ — chance — identical to BigBird. Two completely different static patterns, the same NIAH. That is decisive: the failure is not about *which* static pattern I pick or how I size its window, it is about *staticness itself*. Sink columns cover the first few tokens, the window covers the recent tokens, and a needle in the middle of an 8K haystack lives in neither, so no fixed mask can route to it. The residual gap to the oracle's NIAH $1.0$ is entirely a **routing** gap, and routing means the kept set has to depend on what the query is asking.

I propose **Block-Top-K** — content-adaptive block-level top-K selection in the NSA style. Adaptivity should work at all because softmax attention is empirically sparse: for a given query a small fraction of keys carry almost all the mass, so most of the $O(N^2)$ work is on near-zero weights and attending only to the keys that matter loses little. The catch is that *which* keys matter is query-dependent and moves — no static set works, the two static rungs proved that — and the obvious way to find them, computing $q \cdot k$ for all pairs and keeping the largest, *is* the dense score matrix I am trying to avoid. The escape is to judge importance at a coarser unit than the individual key, and there are two independent reasons that is legitimate. First, where attention lands is *spatially clustered* — the nonzero entries come in contiguous runs, neighboring keys share an importance level — so a whole contiguous block rises and falls together and one score per block loses little. Second, contiguous block reads are what GPUs are fast at and what tiled attention kernels already do; selecting scattered tokens would force non-contiguous gathers. Both point the same way: chop the keys into contiguous blocks of size `BLOCK=64` (128 blocks at 8K) and decide importance per block.

How to score a block cheaply with a frozen model and no learned projection? Summarize each block by a single representative vector and score the query against it. The block's keys are spatially coherent — that was the premise — so they point in similar directions and their *mean* is a low-variance summary close to any key inside the block. So the representative is the mean-pooled key, and $q \cdot \mathrm{mean}(k_\text{block})$ approximates the aggregate $q \cdot k$ over the block: parameter-free, no training, valid precisely because the block clusters. To keep the selection matrix from being $O(N \cdot n_\text{blocks})$ — still quadratic in $N$ — I pool the queries the same way, so the importance matrix is $(n_\text{blocks} \times n_\text{blocks})$, a factor $\text{BLOCK}^2$ cheaper than dense and never materializing the $(N,N)$ logits. Pooling queries is defensible by the same clustering argument on the query axis: adjacent queries ask for similar things, so giving a query block one shared selection is consistent with the structure and is exactly what a block-shaped kernel wants. The proxy is
$$s[i, j] = \mathrm{mean}_q(\text{block } i) \cdot \mathrm{mean}_k(\text{block } j) \cdot \text{scale}.$$

With block scores, the selection rule is top-K per query block, and two things must be handled. Causality first: query block $i$ may only see key block $j \le i$, so I mask the future block scores to $-\infty$ before ranking and AND the kept set with the causal block mask after, and then *inside* the diagonal block enforce the strict token-level triangle $\text{key\_pos} \le \text{query\_pos}$, because a coarse "keep block $i$" would otherwise let an early query in the block peek at a later key in the same block. Two levels of causal masking — blockwise for selection, token-wise for the final attention. Second, the diagonal block: local context is almost always most relevant and must always be kept, but a top-K by the *approximate* proxy gives no guarantee the current block wins a slot — for an early query block the diagonal might score below a content-similar far block. So I force-include it unconditionally: mask the diagonal out of the ranking by setting its score $-\infty$ so top-K never picks it, take the top $K{-}1$ *off-diagonal* blocks, and append the diagonal by hand. This is the role StreamingLLM's recent window played — local context always kept — but it collapses to "always keep your own block," and the *rest* of the budget routes to query-chosen far blocks instead of a fixed recent band. That is the precise upgrade over rung 3, and what NIAH needs.

The keep count $K$ comes from the budget, accounting for causality, because under a causal mask different query blocks have different numbers of admissible blocks. Index query blocks $i = 0..n{-}1$; block $i$ has $i+1$ causal candidates and keeps $\min(K, i+1)$. Total kept block-pairs is
$$\sum_{i=0}^{n-1} \min(K, i+1) = \frac{K(K-1)}{2} + (n - K + 1)K = \frac{K(2n - K + 1)}{2}.$$
Sanity: $K=1$ gives $n$ (each block keeps itself), $K=n$ gives $n(n{+}1)/2$ (full causal). The denominator is that full count, so block-level density is $K(2n - K + 1)/(n(n+1))$. Setting it equal to $\rho$ and solving the quadratic $K^2 - (2n+1)K + \rho\,n(n+1) = 0$ — the kept-pair function is concave and rising over $0 \le K \le n$, so I take the smaller feasible root —
$$K = \frac{(2n+1) - \sqrt{(2n+1)^2 - 4\rho\,n(n+1)}}{2},$$
floored to stay at or under budget. With $n = 128$, $\rho = 0.25$: discriminant $257^2 - 4 \cdot 0.25 \cdot 128 \cdot 129 = 66049 - 16512 = 49537$, $\sqrt{} \approx 222.6$, $K = (257 - 222.6)/2 \approx 17$. So 17 of 128 blocks per query block including the diagonal; realized density $17 \cdot 240 / (128 \cdot 129) \approx 0.247$, just under budget. Then $kk = K - 1 = 16$ off-diagonal top-K plus the diagonal, $K$ clamped $\ge 1$ and $kk$ to $[0, n{-}1]$. (The non-causal fallback would be $K \approx \mathrm{round}(\rho n)$, but `is_causal` is always true.)

A couple of honesty points about the realization. This $K$ controls the *block-pair* density, while the reported `last_density` is the *token-level* mask after the strict token triangle trims the half-filled diagonal block and any final padding block, so the algebra and the realization disagree at the boundary; I trust the algebra only to *choose* $K$ and *measure* `last_density` from the realized token mask, which comes in at or below the block-level $0.247$. Shape mechanics: $N$ may not be a multiple of `BLOCK`, so I pad $q, k$ up to $N_\text{pad} = \lceil N/\text{BLOCK} \rceil \cdot \text{BLOCK}$ with zeros so the reshape and `mean(dim=3)` are exact, then trim the token keep-mask back to $[:N, :N]$ and let the strict causal triangle kill any padded influence. Block scoring is in float32 (inputs are fp16/bf16 and I am about to `masked_fill` with $-\infty$ and softmax), output cast back to `q.dtype`. One GQA note specific to this task: the kernel-level NSA design sums block importance *across the heads in a KV group* so the whole group selects the same blocks, or the loaded KV is the union of each head's picks and the memory sparsity collapses — but the harness has already replicated GQA before my module, so I see 12 heads on both $Q$ and $K/V$ and score and select per replicated head, which is correct here; the group-reduction is only needed against a real GQA cache, which I do not have. This is the faithful inference re-expression, not the trainable kernel, and the final attention is again the masked-softmax over the full logits — a block-gather kernel into SRAM is what a Triton implementation would do for the real speedup, but the masked form computes the identical output and no-Triton is the task constraint. I expect NIAH to *break above* $0.2$ for the first time on the ladder, because the query block asking for the needle scores the needle's key block highly through $\mathrm{mean}_q \cdot \mathrm{mean}_k$ and selects it regardless of where it sits — while the diagonal force-include holds StreamingLLM's local floor on the QA tasks and the routed far blocks keep Qasper and MultiFieldQA at least at $0.1103$ / $0.213$, all under budget. That NIAH rise is the confirmation that closing the routing gap is exactly what separates the static rungs from the oracle.

```python
# EDITABLE region of custom_sparse_attn.py — rung 4: Block-Top-K (NSA-style adaptive)
class SparseAttention(nn.Module):
    """NSA-style content-adaptive block-sparse top-K attention."""

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

        # Pad N up to a multiple of BLOCK so we can pool cleanly.
        Npad = ((N + Bk - 1) // Bk) * Bk
        if Npad != N:
            pad = Npad - N
            qp = torch.nn.functional.pad(q, (0, 0, 0, pad))
            kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
        else:
            qp, kp = q, k

        n_blocks = Npad // Bk

        # Mean-pooled q / k per block: (B, H, n_blocks, D)
        q_blocks = qp.view(B, H, n_blocks, Bk, D).mean(dim=3)
        k_blocks = kp.view(B, H, n_blocks, Bk, D).mean(dim=3)

        # Block-level scores (B, H, n_blocks, n_blocks)
        scores = torch.einsum('bhmd,bhnd->bhmn',
                              q_blocks.float(), k_blocks.float()) * scale

        idx = torch.arange(n_blocks, device=q.device)
        if is_causal:
            causal_blk = idx[:, None] >= idx[None, :]   # (n_b, n_b)
            scores = scores.masked_fill(~causal_blk, float('-inf'))

        # Force-include diagonal: zero its score (irrelevant for topk choice)
        diag_mask = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=q.device)
        diag_mask.fill_diagonal_(True)
        scores_nodiag = scores.masked_fill(diag_mask, float('-inf'))

        # Target top-K per query block under causal AND. With per-row keep K,
        # mean kept block-pairs = K*(2n-K+1)/2, denom = n*(n+1)/2 (causal),
        # so density = K*(2n-K+1)/(n*(n+1)). Solve K(2n+1-K) = budget*n*(n+1)
        # via the quadratic root closer to 0:
        #   K = ((2n+1) - sqrt((2n+1)^2 - 4*budget*n*(n+1))) / 2
        n_b = n_blocks
        if is_causal:
            disc = max(0.0, (2 * n_b + 1) ** 2 - 4 * self.density_budget * n_b * (n_b + 1))
            K_top = max(1, int(((2 * n_b + 1) - math.sqrt(disc)) / 2))
        else:
            K_top = max(1, int(round(self.density_budget * n_b)))
        kk = max(0, min(K_top - 1, n_b - 1))
        if kk > 0:
            topk_idx = scores_nodiag.topk(kk, dim=-1).indices  # (B,H,n_b,kk)
        else:
            topk_idx = torch.empty(B, H, n_blocks, 0, dtype=torch.long, device=q.device)
        diag_idx = idx.view(1, 1, n_blocks, 1).expand(B, H, n_blocks, 1)
        sel = torch.cat([topk_idx, diag_idx], dim=-1)
        block_keep = torch.zeros(B, H, n_blocks, n_blocks,
                                 dtype=torch.bool, device=q.device)
        block_keep.scatter_(-1, sel, True)
        if is_causal:
            block_keep = block_keep & causal_blk

        # Expand block_keep -> token-level (B, H, Npad, Npad)
        q_tok_blk = (torch.arange(Npad, device=q.device) // Bk)
        k_tok_blk = q_tok_blk
        # Index per-(b,h,qtok,ktok)
        token_keep = block_keep[:, :, q_tok_blk, :][:, :, :, k_tok_blk]
        token_keep = token_keep[:, :, :N, :N]
        if is_causal:
            tidx = torch.arange(N, device=q.device)
            token_keep = token_keep & (tidx[:, None] >= tidx[None, :])

        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        # Take per-(b,h) mean for reporting; harness aggregates further.
        self.last_density = float(token_keep[0, 0].sum().item()) / max(denom, 1.0)

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~token_keep, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
```
