# BigBird: linear-complexity sparse attention for long sequences

## Problem

Full self-attention forms an `n × n` matrix of query–key inner products, so it costs `O(n²)` time and memory. On commodity accelerators this caps the usable sequence length at roughly 512 tokens, which rules out tasks whose signal lives in long contexts: question answering over documents, long-document classification and summarization, and genomics, where a single DNA sequence can be tens of thousands of bases. The goal is an attention mechanism that is **linear in `n`** yet keeps the two theoretical guarantees that make full attention trustworthy — it is a **universal approximator** of continuous sequence-to-sequence functions, and the encoder–decoder form is **Turing complete**.

## Key idea

View attention as a directed graph `D` on the tokens: an edge `i → j` means query `i` attends to key `j`, and full attention is the complete graph. Reducing cost is then graph sparsification. BigBird keeps three kinds of edges, each contributing a property the complete graph has:

- **Window (`w`)** — each query attends to its `w/2` left and `w/2` right neighbors. Supplies locality / high clustering coefficient.
- **Random (`r`)** — each query attends to `r` random keys. Makes the graph an expander with `O(log n)` diameter and a large spectral gap, so information mixes across the whole sequence in a few hops.
- **Global (`g`)** — a few tokens that attend to everything and are attended by everything. They are the load-bearing piece: the graph then contains a *star*, which is exactly what the universal-approximation proof needs, and they address the diagnostic gap left by window+random alone.

Each query touches `g + w + r = O(1)` keys, so the mechanism is `O(n)` in time and memory.

Global tokens come in two flavors. **ITC** (internal) promotes a subset `G` of existing tokens to global: `A(i,:) = A(:,i) = 1` for `i ∈ G`. **ETC** (extended) appends `g` new CLS-like tokens that are global, giving extra scratch capacity.

## Final mechanism

Generalized attention on graph `D` with out-neighbors `N(i)`:

```
Attn_D(X)_i = x_i + Σ_h σ( (x_i W_Q^h)(X_{N(i)} W_K^h)^T / √m ) · (X_{N(i)} W_V^h)
```

with `N(i) = { window(i) } ∪ { r random } ∪ { global }`. Setting `D` to the complete graph recovers full attention.

## Theory (sketch)

**Universal approximation.** For the star graph `S` (`N(i) = {0, i}`, `N(0) = {1..n}`) and any `D ⊇ S`: approximate a continuous `f` by a piecewise-constant `f̄` on a grid of granularity `δ`; build a *contextual mapping* — a unique scalar code per `(X, x_i)` — using a **sparse selective-shift operator** `ψ_u` (shift entries whose `u`-projection lies in a band by `max−min` over `N(i)`) plus the global token at index 0. Running `n` interleaved low/high shifts threads every column's content through the global token; afterward feed-forward layers decode each code to the target output. Swapping hardmax/Φ for softmax/ReLU completes it. The global token is essential because the construction needs a place to compute sequence-wide max/min values while still using only sparse neighborhoods.

**Turing completeness.** With a causal sparse decoder graph, simulate any Turing machine by exploiting associativity of `min`: the one-step "find the symbol last written at the next head cell" (which the full model does with one attention) is broken into `O(√i)` intermediate steps that aggregate a running min, plus a switching layer that holds the machine state across those steps. `O(n)` inner products suffice.

**Cost of sparsity (no free lunch).** Finding each vector's furthest neighbor is one full-attention layer (`Q=−a, K=a`), but under the Orthogonal Vectors Conjecture any sparse pattern with `Õ(n)` edges needs `Ω̃(n^{1−o(1)})` layers.

## Code (block-sparse, GPU-efficient)

Fine-grained sparse matmul is slow on GPUs, so attention is computed on **blocks** of size `b`: blockify `Q,K`, get the block-diagonal scores cheaply, extend to a window by *rolling* copies of the key-block tensor, prepend the fixed global key blocks, and `gather` the few random key blocks. Everything reduces to dense tensor products of cost `O(n(g+w+r)bd)`.

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

    @staticmethod
    def _matmul(a, b):
        return torch.matmul(a, b)

    @staticmethod
    def _matmul_t(a, b):
        return torch.matmul(a, b.transpose(-1, -2))

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
        scores = self._matmul_t(q, k) * scale
        return self._matmul(torch.softmax(scores, dim=-1), v)

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

        # First and last query blocks are global rows: they attend to all keys.
        first = self._attend(q_blk[:, :, 0], k, v, scale).unsqueeze(2)
        last = self._attend(q_blk[:, :, -1], k, v, scale).unsqueeze(2)

        # Second block: first global, local blocks 1 and 2, last global, random.
        second_k = torch.cat(
            [k_blk[:, :, 0], k_blk[:, :, 1], k_blk[:, :, 2],
             k_blk[:, :, -1], rand_k[:, :, 0]],
            dim=2,
        )
        second_v = torch.cat(
            [v_blk[:, :, 0], v_blk[:, :, 1], v_blk[:, :, 2],
             v_blk[:, :, -1], rand_v[:, :, 0]],
            dim=2,
        )
        second = self._attend(q_blk[:, :, 1], second_k, second_v, scale).unsqueeze(2)

        # True middle blocks: first global, three-block window, random, last global.
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

        # Second-last block mirrors the second block near the right boundary.
        second_last_k = torch.cat(
            [k_blk[:, :, 0], k_blk[:, :, -3], k_blk[:, :, -2],
             k_blk[:, :, -1], rand_k[:, :, -1]],
            dim=2,
        )
        second_last_v = torch.cat(
            [v_blk[:, :, 0], v_blk[:, :, -3], v_blk[:, :, -2],
             v_blk[:, :, -1], rand_v[:, :, -1]],
            dim=2,
        )
        second_last = self._attend(
            q_blk[:, :, -2], second_last_k, second_last_v, scale
        ).unsqueeze(2)

        out = torch.cat([first, second, middle, second_last, last], dim=2)
        out = out.reshape(b, h, n, d)
        return out.permute(0, 2, 1, 3).reshape(b, n, h * d)
```

This mirrors the canonical block-sparse implementation: the end blocks are computed densely (they are global), and every middle query block is one dense matmul against a compact packed tensor holding its global, window (built by rolling/stacking neighbor blocks), and gathered random key blocks.
