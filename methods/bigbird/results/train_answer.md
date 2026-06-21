The problem is the quadratic cost of full self-attention. Every query computes an inner product with every key, producing an n × n score matrix whose time and memory grow as O(n²). In practice this caps sequences at about 512 tokens, which rules out long-document QA, summarization, and genomics tasks where the relevant context spans thousands of tokens. The goal is to make attention linear in n without giving up what makes dense attention dependable. Simple fixes fall short: a sliding window respects locality but cannot move information across the sequence without stacking many layers; random sparse edges give short paths and fast mixing but ignore the strong local signal seen in trained attention; and even window plus random together leaves a measurable gap to dense attention, because neither design gives any single position a view of the whole sequence.

The missing piece, and the one that recovers the theoretical guarantees, is a global token. The universal-approximation proof for transformers builds a contextual mapping by computing sequence-wide maxima and minima; that construction stalls if every query only sees a small local or random neighborhood. Adding a token that attends to all positions and is attended by all positions plants a star graph inside the sparse pattern, and the proof goes through. The same idea extends to Turing completeness in the decoder: a one-shot argmin over all past positions is replaced by an associative running min accumulated over a few extra steps, so O(n) inner products per step still suffice. Sparsity is not free—a furthest-neighbor task needs Ω̃(n^{1−o(1)}) layers under standard complexity assumptions—but for ordinary long-context modeling the linear mechanism is both expressive and practical.

The method is BigBird. It views attention as a directed graph on token positions and keeps three edge types: window edges from each query to its nearest neighbors, random edges to a small sample of keys, and global edges through one or a few global tokens. Each query touches only g + w + r keys, so the total cost is O(n). Window edges supply locality and high clustering, matching the empirical fact that most learned attention is local. Random edges supply expander-like connectivity, giving O(log n) paths between any pair of positions. Global edges supply the star subgraph that the universal-approximation proof needs and act as sequence-wide memory sinks. Global tokens can be internal, promoting existing tokens to attend and be attended globally, or extended, appending new CLS-like tokens for extra scratch capacity.

On hardware the pattern is realized block-wise so that sparse connectivity becomes dense block matrix multiplications, which GPUs and TPUs handle efficiently. Tokens are grouped into contiguous blocks of size b. The first and last blocks are treated as global rows and are computed densely. Every middle query block attends to the global blocks, to a rolling band of neighboring key blocks, and to a handful of gathered random key blocks. The window neighbors are produced by stacking rolled copies of the blocked key tensor rather than by scatter-gather, and only the small random component requires an explicit gather. All scores for a query block are then computed with one dense matrix multiply against the packed key tensor.

```python
import math
import numpy as np
import torch
import torch.nn as nn


class BigBirdBlockSparseAttention(nn.Module):
    """Linear-time attention = window + random + global, computed block-wise."""

    def __init__(self, hidden_size, num_heads, block_size=64,
                 num_random_blocks=3, seed=None):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.seed = seed
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x):
        b, n, _ = x.size()
        return x.view(b, n, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    def _rand_blocks(self, num_blocks):
        rng = np.random.RandomState(self.seed)
        r = self.num_random_blocks
        plan = np.zeros((num_blocks - 2, r), dtype=np.int64)
        for block in range(1, num_blocks - 1):
            forbidden = {0, num_blocks - 1, block - 1, block, block + 1}
            if block == 1:
                forbidden.add(num_blocks - 2)
            if block == num_blocks - 2:
                forbidden.add(1)
            choices = [k for k in range(1, num_blocks - 1) if k not in forbidden]
            if len(choices) < r:
                raise ValueError("sequence has too few blocks for this random plan")
            plan[block - 1] = rng.permutation(choices)[:r]
        return torch.tensor(plan, dtype=torch.long)

    def _gather_random(self, blocked, plan):
        b, h, nb, B, d = blocked.shape
        r = plan.shape[-1]
        idx = plan.to(blocked.device).view(1, 1, nb - 2, r, 1, 1)
        idx = idx.expand(b, h, nb - 2, r, B, d)
        src = blocked.unsqueeze(2).expand(b, h, nb - 2, nb, B, d)
        return torch.gather(src, 3, idx).reshape(b, h, nb - 2, r * B, d)

    def _attend(self, q, k, v, scale):
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def forward(self, hidden_states):
        b, n, _ = hidden_states.size()
        B = self.block_size
        assert n % B == 0
        nb = n // B
        assert nb >= 5
        scale = 1.0 / math.sqrt(self.head_size)

        q = self._split_heads(self.query(hidden_states))
        k = self._split_heads(self.key(hidden_states))
        v = self._split_heads(self.value(hidden_states))
        h, d = self.num_heads, self.head_size
        q_blk = q.view(b, h, nb, B, d)
        k_blk = k.view(b, h, nb, B, d)
        v_blk = v.view(b, h, nb, B, d)
        plan = self._rand_blocks(nb)
        rand_k = self._gather_random(k_blk, plan)
        rand_v = self._gather_random(v_blk, plan)

        first = self._attend(q_blk[:, :, 0], k, v, scale).unsqueeze(2)
        last = self._attend(q_blk[:, :, -1], k, v, scale).unsqueeze(2)

        second_k = torch.cat(
            [k_blk[:, :, 0], k_blk[:, :, 1], k_blk[:, :, 2],
             k_blk[:, :, -1], rand_k[:, :, 0]], dim=2,
        )
        second_v = torch.cat(
            [v_blk[:, :, 0], v_blk[:, :, 1], v_blk[:, :, 2],
             v_blk[:, :, -1], rand_v[:, :, 0]], dim=2,
        )
        second = self._attend(q_blk[:, :, 1], second_k, second_v, scale).unsqueeze(2)

        mid_q = q_blk[:, :, 2:-2]
        band_k = torch.cat([k_blk[:, :, 1:-3], k_blk[:, :, 2:-2], k_blk[:, :, 3:-1]], dim=3)
        band_v = torch.cat([v_blk[:, :, 1:-3], v_blk[:, :, 2:-2], v_blk[:, :, 3:-1]], dim=3)
        first_g_k = k_blk[:, :, 0].unsqueeze(2).expand(b, h, nb - 4, B, d)
        first_g_v = v_blk[:, :, 0].unsqueeze(2).expand(b, h, nb - 4, B, d)
        last_g_k = k_blk[:, :, -1].unsqueeze(2).expand(b, h, nb - 4, B, d)
        last_g_v = v_blk[:, :, -1].unsqueeze(2).expand(b, h, nb - 4, B, d)
        mid_k = torch.cat([first_g_k, band_k, rand_k[:, :, 1:-1], last_g_k], dim=3)
        mid_v = torch.cat([first_g_v, band_v, rand_v[:, :, 1:-1], last_g_v], dim=3)
        middle = self._attend(mid_q, mid_k, mid_v, scale)

        second_last_k = torch.cat(
            [k_blk[:, :, 0], k_blk[:, :, -3], k_blk[:, :, -2],
             k_blk[:, :, -1], rand_k[:, :, -1]], dim=2,
        )
        second_last_v = torch.cat(
            [v_blk[:, :, 0], v_blk[:, :, -3], v_blk[:, :, -2],
             v_blk[:, :, -1], rand_v[:, :, -1]], dim=2,
        )
        second_last = self._attend(
            q_blk[:, :, -2], second_last_k, second_last_v, scale
        ).unsqueeze(2)

        out = torch.cat([first, second, middle, second_last, last], dim=2)
        out = out.reshape(b, h, n, d)
        return out.permute(0, 2, 1, 3).reshape(b, n, h * d)
```
