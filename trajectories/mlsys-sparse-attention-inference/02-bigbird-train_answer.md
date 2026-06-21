The oracle handed me three targets and they split exactly along the fault line I expected: NIAH at 8K came in at $1.0$, full attention retrieves the needle perfectly, while the two QA tasks came in modest — Qasper F1 $0.1406$, MultiFieldQA-EN F1 $0.3447$ — because the 1.5B model's own competence caps them. So the ceiling is not "perfect everywhere." On QA the evidence is spread across the document and a sparse mask covering *some* of it loses F1 gracefully; on NIAH the answer lives in one position and a mask that misses the needle's block falls off a cliff from $1.0$ to chance. NIAH is the discriminator that will punish me hardest, and a content-blind pattern is exactly the kind of thing that can miss the needle by construction. The cheapest honest place to start, though, is *static*: a mask that depends only on $N$, built once per prompt, cached across all 24 layers, costing nothing per layer to decide. The question is which static pattern spends the 25% budget best.

I propose **BigBird** — a global + window + random block-sparse pattern — and the way to see why it is built this way is to read attention as a graph. Put the $N$ token positions as vertices; an edge $i \to j$ means query $i$ attends to key $j$. Full attention is the complete causal graph, which is why the oracle nailed NIAH: a direct edge from every query to the needle. Sparsifying means deleting most edges and asking whether the graph still behaves like the complete one, and two properties of the complete graph are load-bearing. First, short paths — information must reach from any position to any other in few hops, or a fact at position 200 can never influence position 7000 within the model's depth. Second, locality — when people probe trained attention the dominant weight lands on *nearby* positions. A good static pattern has to supply both, and the three edge types each install one piece. **Window** edges arrange the tokens on a line and connect each query block to a band of neighbor blocks — a ring-lattice with high clustering, supplying locality at a constant number of blocks per query block, but with terrible paths ($O(N/\text{window})$ hops to cross the sequence, so the far needle is unreachable). **Random** edges fix exactly that: give each query block a few randomly chosen non-window blocks and the graph becomes an expander, shortest paths drop to $O(\log N)$, a large spectral gap, rapid mixing — the Watts–Strogatz small-world recipe of a local ring plus a sprinkle of long-range shortcuts. But window+random alone is known to fall short of dense, because some tasks need a single position to corral information about the *whole* sequence at once, and no node in a window+random graph sees everything in one hop. **Global** tokens — positions wired to everyone and attended by everyone — plant a star inside the graph and recover that missing reach.

What makes this rung *this task's* method rather than the generic paper version is that none of the paper's machinery is available. The model is frozen at inference: I have no $W_Q, W_K, W_V$ to learn, I receive $q, k, v$ already projected, RoPE'd, and GQA-replicated to 12 heads. I cannot *append* dedicated CLS-like global tokens, because the harness hands me a fixed $(B, H, N, D)$ and reads density over the existing $N(N{+}1)/2$ causal pairs, so growing $N$ would break the contract. And with no Triton I cannot write a block-sparse kernel, so the implementation is a masked-softmax over the full logits — faithful *behavior* under the budget, not a realized speedup. So the three ingredients install as *internal* roles on the existing tokens. With `BLOCK=64` there are 128 blocks at 8K. **Global** promotes the first 2 blocks to dual-role anchors — every query attends to them, they attend to everyone — which is the natural sink position under causal masking, since the earliest tokens are the only ones visible to every later query. **Window** keeps a band of 3 blocks, $|b_i - b_j| \le 1$. The **random** count is what is left of the budget, and here I have to be exact, because the harness aborts on measured density, not on intent.

The naive sizing `random = round(0.25 \cdot 128) - g - w` overshoots, for two interacting reasons: the global *rows* — the first 2 blocks attending to everyone — add causal pairs a flat per-query estimate never counts, and the random sampling under the causal AND lands the realized triangle fraction above the flat `fraction × n_blocks` estimate, especially because early query blocks have few admissible blocks and the fixed window and global already saturate them. So I size the random count against a *conservatively discounted* budget,
$$\text{target} = \mathrm{round}(0.25 \cdot 0.88 \cdot n_\text{blocks}), \qquad r = \text{target} - g - w,$$
clamped non-negative and below the admissible pool. The $0.88$ is not a paper constant; it is a $\sim 12\%$ margin calibrated so the measured density sits clear of the $0.25 + 0.02$ ceiling at every context length the harness evaluates, because a single layer crossing the line aborts the whole run for nothing. The random pattern itself must be *fixed*, not resampled per forward, because the same module replays at every generation step and a drifting mask would make attention non-deterministic and could spike density on some forward; so I sample the random blocks once, keyed by `(n_blocks, device, g, w, r)` with a deterministic seed derived from `n_blocks` and the process seed, cache the $(n_\text{blocks} \times n_\text{blocks})$ boolean keep-matrix, and reuse it. For each query block I draw $r$ blocks from the pool that excludes the global, window, and self blocks, so the random arcs add genuinely new reach rather than re-covering what the others already hold. Then I AND the block-keep matrix with the causal block mask $b_i \ge b_j$, expand it to a token-level $(N, N)$ mask by mapping each token to its block, AND again with the strict token-level causal triangle so the diagonal block is properly lower-triangular, and report `last_density` as the *measured* fraction of the realized token mask over $N(N{+}1)/2$ — not the block algebra, because the diagonal-block trimming and final-block padding make the algebra and the realization disagree at the boundary. The attention is then the masked-softmax form: compute $QK^\top \cdot \text{scale}$ in float32 (the inputs are fp16/bf16 and I am about to `masked_fill` with $-\infty$ and softmax, far more stable upcast), $-\infty$ the non-kept entries, softmax over keys, `nan_to_num` any row that kept nothing, multiply by $V$, cast back. I expect this rung to recover a usable share of the QA F1, where window covers local context and the random expander reaches scattered spans — but to *collapse* on NIAH toward chance, because whether the needle's one block falls in a query's window, the 2 global blocks, or its fixed random sample is purely luck, and a content-blind mask cannot route to a position it did not anticipate. That predicted cliff is the case the next rungs answer.

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
