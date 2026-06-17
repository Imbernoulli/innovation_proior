**Problem (from rung 1).** The dense oracle fixed the ceiling — NIAH `1.0`, Qasper F1 `0.1406`,
MultiFieldQA-EN F1 `0.3447` — and the NIAH-vs-QA split is the diagnosis: NIAH tests whether the mask
covers the needle's one block, QA tests whether it covers enough distributed spans. The first sparse
rung must spend ≤ 25% of the causal matrix and not abort on density. A *static* pattern — content-blind,
built once per `N`, cached across the 24 layers — is the cheapest starting point.

**Key idea.** Read attention as a graph and keep three edge types, each supplying a property of the
complete graph: **window** (band of neighbor blocks → locality/clustering), **random** (a fixed sample of
non-window blocks per query → expander, `O(log N)` paths), and **global** (the first 2 blocks promoted to
attend-all / be-attended-by-all → the star the whole-sequence reach needs). `BLOCK=64`, so 128 blocks at
8K; global=2, window=3, random sized from the budget.

**Why it works (and how it differs from the paper here).** The frozen-model inference setting strips the
paper's learned `W_Q/W_K/W_V`, the appended CLS-like extended global tokens, and the block-sparse kernel:
I receive `q,k,v` already projected/RoPE'd/GQA-replicated, cannot grow `N`, and have no Triton, so the
fill is a masked-softmax over full logits (faithful *behavior*, not kernel speedup) with global installed
as *internal* roles on the first blocks. Random blocks are sampled **once**, deterministically cached by
`(n_blocks, device, g, w, r)`, and reused every forward so the replayed mask is stable. The random count
is sized against a conservatively discounted budget `round(0.25 · 0.88 · n_blocks) − g − w` (~12% margin)
because the global rows and the causal AND push measured density above the linear estimate, and one layer
over `0.25 + 0.02` aborts the run. `last_density` is **measured** from the realized token mask.

**What to watch.** Expect a usable share of QA F1 (distributed evidence is reachable via window+random),
but NIAH to **collapse** from `1.0`: a fixed mask covers a mid-haystack needle only by luck of window /
global / fixed-random placement. That predicted cliff is the case for making the next rungs route
content-adaptively.

**Hyperparameters.** `BLOCK=64`, `NUM_GLOBAL_BLOCKS=2`, `NUM_WINDOW_BLOCKS=3`, budget margin `0.88`,
fp32 score/softmax, deterministic random seed from `n_blocks` + process seed.

```python
# EDITABLE region of custom_sparse_attn.py — rung 2: BigBird (global + window + random)
class SparseAttention(nn.Module):
    """BigBird — global + window + random block-sparse pattern."""

    BLOCK = 64
    NUM_GLOBAL_BLOCKS = 2  # first 2 blocks (128 tokens) act as global sinks
    NUM_WINDOW_BLOCKS = 3  # band of 3 blocks around the query block

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        self.last_density = None
        # Random-block cache, keyed by (N, device) — same pattern across calls
        # for the same sequence length (deterministic per layer instance).
        self._random_cache = {}

    def _build_block_keep(self, N, device, is_causal):
        Bk = self.BLOCK
        if N % Bk != 0:
            # Pad-aware: round up to whole blocks; the (N,N) mask gets clipped.
            n_blocks = (N + Bk - 1) // Bk
        else:
            n_blocks = N // Bk
        g = min(self.NUM_GLOBAL_BLOCKS, n_blocks)
        w = self.NUM_WINDOW_BLOCKS
        # Solve random-blocks count from the budget at the BLOCK level.
        # The actual measured density (after random-block sampling and
        # causal AND) tends to land slightly above the linear estimate, so
        # apply a ~12% conservative margin to stay clear of the +0.02 slack
        # ceiling at every context length we evaluate.
        target = max(1, int(round(self.density_budget * 0.88 * n_blocks)))
        used = g + w
        r = max(0, min(target - used, n_blocks - used))
        # Build (n_blocks, n_blocks) bool keep
        keep = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=device)
        # global cols (every query block attends to first g blocks)
        if g > 0:
            keep[:, :g] = True
        # global rows (first g blocks attend to everyone)
        if g > 0:
            keep[:g, :] = True
        # window: |bi - bj| <= w//2
        idx = torch.arange(n_blocks, device=device)
        win = (idx[:, None] - idx[None, :]).abs() <= w // 2
        keep |= win
        # random: per query block, sample r blocks from the non-(global|window) pool
        cache_key = (n_blocks, str(device), g, w, r)
        if cache_key not in self._random_cache:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(((0xBB ^ n_blocks) + int(torch.initial_seed()) - 42) & 0xFFFFFFFF)
            rand_keep = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
            base = keep.detach().to('cpu')
            for bi in range(n_blocks):
                avail = (~base[bi]).nonzero(as_tuple=False).flatten()
                if avail.numel() == 0 or r == 0:
                    continue
                pick = avail[torch.randperm(avail.numel(), generator=gen)[:r]]
                rand_keep[bi, pick] = True
            self._random_cache[cache_key] = rand_keep.to(device)
        keep |= self._random_cache[cache_key]
        # Apply causal at block level (a query block i may attend to j<=i)
        if is_causal:
            keep = keep & (idx[:, None] >= idx[None, :])
        return keep, n_blocks

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        Bk = self.BLOCK
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        block_keep, n_blocks = self._build_block_keep(N, q.device, is_causal)
        # Expand block_keep -> token-level (N, N) by index gather
        q_tok_blk = (torch.arange(N, device=q.device) // Bk).clamp(max=n_blocks - 1)
        k_tok_blk = q_tok_blk
        token_keep = block_keep[q_tok_blk][:, k_tok_blk]   # (N, N) bool
        if is_causal:
            idx = torch.arange(N, device=q.device)
            token_keep = token_keep & (idx[:, None] >= idx[None, :])

        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(token_keep.sum().item()) / max(denom, 1.0)

        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~token_keep, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
```
