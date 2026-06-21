A decoder-only language model at the 7B scale is only worth deploying if it is cheap to serve, and autoregressive generation is expensive in two coupled places. Every new token must stream back the cached keys and values of all prior tokens, and full causal attention over a growing history scales with sequence length, so a length-$n$ prefill does $O(n^2)$ score work while decode step $i$ reads a cache proportional to $i$. The persistent state is the keys and values: the current query is transient and never cached, but each prior position contributes one key and one value vector per attention head, so decode-time memory traffic grows with both the number of stored key/value heads and the number of cached positions. The obvious efficiency levers each fall short on their own. Dense multi-head attention gives direct access to all history but pays the full price in cache width and length. Shared key/value-head variants — down to the extreme single-head multi-query form — shrink the per-position width but do nothing to bound how many positions accumulate. A fixed local band cuts the long-sequence cost to linear, but a single local layer no longer sees arbitrary old tokens, which threatens the long-range dependence that language modeling needs. What we want is a compact decoder that keeps a route for old information to influence later predictions while bounding both the width and the length of the most expensive inference state.

I propose Mistral 7B: an otherwise standard LLaMA-style pre-norm decoder — RMSNorm, rotary query/key embeddings, a SwiGLU feed-forward, 32 layers, model dimension 4096, head dimension 128 — whose serving cost is controlled by composing three mechanisms that attack width, work, and cache length separately. The first is grouped-query attention. Because the current step creates queries but caches only keys and values, the query and key/value head counts can differ. We project queries to $h=32$ heads but keys and values to only $g=8$ heads, and after reading the cache each of the 8 key/value heads is repeated $h/g=4$ times just before the attention kernel, so attention still runs with 32 query-compatible heads. The cache stores 8 key heads and 8 value heads instead of 32 of each, cutting both cache memory and decode-time KV bandwidth by exactly $32/8 = 4$. This intermediate point is deliberate: $g=1$ multi-query attention saves the most bandwidth but collapses every query head onto one key/value subspace, a blunt loss of representational room, while $g=8$ keeps most of that room at most of the savings.

The second mechanism is sliding-window attention, which bounds the long-sequence work that grouped-query attention leaves untouched. Each query at position $i$ may attend only to the most recent $W=4096$ keys, using the operational convention $j \in [\max(0,\,i-W+1),\, i]$ so at most $W$ keys are visible. Prefill cost falls from $O(n^2)$ to $O(nW)$ for fixed $W$, and a decode step reads at most $W$ positions rather than the whole past. The immediate objection is that a token far in the past becomes invisible, but stacking local layers defeats this through a transitive receptive field. At layer 1, position $i$ depends on inputs roughly one window back; at layer 2 it attends to layer-1 states that already summarized another window beneath them; inductively, after $k$ layers the state at position $i$ can carry information from about $kW$ positions back. With $W=4096$ and 32 layers the theoretical span is $32 \cdot 4096 = 131072$, roughly 131K tokens. This is reach through depth, not direct single-layer attention to 131K keys, and that distinction is what makes a local window legitimate rather than a truncation of history.

The third mechanism turns the fixed window into a fixed cache. If a query at position $i$ sees at most the last $W$ positions, then once position $i$ is written the oldest still-useful key/value is no older than about $i-W+1$; position $i-W$ has fallen outside every future local window and can never change another result, so its slot may be overwritten. A rolling ring buffer realizes this: absolute position $i$ writes to slot $i \bmod W$, so the cache length stays pinned at $W$ regardless of sequence length. A 32K-token sequence then uses $32768/4096 = 8$ times less cache length than a dense growing cache, and an 8192-token prompt uses half. The cost is that after the buffer wraps, physical slot order no longer matches time, so before a cached chunk is interleaved with a later prompt chunk it must be unrotated — split at $p = \text{seqlen} \bmod W$ and re-concatenated as slots $p{:}$ followed by ${:}p$ to recover chronological order. Two indices are kept distinct throughout: rotary embeddings use the true absolute position because RoPE depends on real sequence position, while only the storage slot is taken modulo the cache size (plus a per-batch offset so different sequences occupy disjoint flattened ranges). When a chunk is longer than the cache, only its last $W$ tokens are written, since earlier ones are already unreachable.

These three mechanisms force a careful treatment of chunked prefill, because the cache update and the attention mask genuinely differ across three cases rather than reducing to one generic sliding mask. On the first prompt chunk there is no prior cache, so the mask is block-diagonal causal attention made local to the window. On a later prompt chunk the attended keys are the unrotated cached keys interleaved with the current chunk's keys, and the mask must align from the bottom right so each new query sees the allowed suffix of prior cache plus the causal positions in the current chunk. On single-token decode the cache is rectangular and padded to the per-layer cache size, and the mask supplies the true valid key lengths after the new token is written. Folding all three into one path would hide a real correctness detail, so the implementation selects the mask — `BlockDiagonalCausalMask.make_local_attention`, `BlockDiagonalMask.make_local_attention_from_bottomright`, or `BlockDiagonalCausalWithOffsetPaddedKeysMask` — by case. The released configuration is `dim=4096`, `n_layers=32`, `head_dim=128`, `hidden_dim=14336`, `n_heads=32`, `n_kv_heads=8`, `sliding_window=4096`, `context_len=8192`, `vocab_size=32000`.

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    sliding_window: int = 4096
    vocab_size: int = 32000
    norm_eps: float = 1e-5


def repeat_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    repeats: int,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.repeat_interleave(keys, repeats=repeats, dim=dim),
        torch.repeat_interleave(values, repeats=repeats, dim=dim),
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional["CacheView"] = None,
        mask: Optional[BlockDiagonalMask] = None,
    ) -> torch.Tensor:
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        xq = self.wq(x).view(seqlen_sum, self.n_heads, self.head_dim)
        xk = self.wk(x).view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
            attn_mask = mask
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
            attn_mask = cache.mask
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim)
            attn_mask = cache.mask

        key, val = repeat_kv(key, val, self.repeats, dim=1)
        out = memory_efficient_attention(xq[None, ...], key[None, ...], val[None, ...], attn_mask)
        out = out.view(seqlen_sum, self.n_heads * self.head_dim)
        return self.wo(out)
```

```python
@dataclass
class CacheInputMetadata:
    positions: torch.Tensor
    to_cache_mask: torch.Tensor
    cached_elements: torch.Tensor
    cache_positions: torch.Tensor
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def get_cache_sizes(
    n_layers: int,
    max_seq_len: int,
    sliding_window: Optional[Union[int, List[Optional[int]]]],
) -> List[int]:
    if sliding_window is None:
        return n_layers * [max_seq_len]
    if isinstance(sliding_window, int):
        return n_layers * [sliding_window]
    assert n_layers % len(sliding_window) == 0
    repeats = n_layers // len(sliding_window)
    return repeats * [w if w is not None else max_seq_len for w in sliding_window]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    # cache has shape (W, H, D); return chronological order.
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    if position == 0:
        return cache
    return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: CacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.metadata = metadata
        self.kv_seqlens = kv_seqlens

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        flat_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if all(s == 0 for s in self.metadata.seqlens):
            return xk, xv
        split_k = torch.split(xk, self.metadata.seqlens)
        split_v = torch.split(xv, self.metadata.seqlens)
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]
        interleaved_k = [x for pair in zip(cache_k, split_k) for x in pair]
        interleaved_v = [x for pair in zip(cache_v, split_v) for x in pair]
        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask
```

```python
class BufferCache:
    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[Union[int, List[Optional[int]]]] = None,
    ):
        self.cache_sizes = get_cache_sizes(n_layers, max_seq_len, sliding_window)
        self.cache_k = {
            i: torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))
            for i, cache_size in enumerate(self.cache_sizes)
        }
        self.cache_v = {
            i: torch.empty((max_batch_size, cache_size, n_kv_heads, head_dim))
            for i, cache_size in enumerate(self.cache_sizes)
        }
        self.kv_seqlens: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self.cache_k[0].device

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> List[CacheInputMetadata]:
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert self.kv_seqlens is not None
        seqpos = self.kv_seqlens.tolist()
        return [self._get_input_metadata_layer(size, seqlens, seqpos) for size in self.cache_sizes]

    def _get_input_metadata_layer(
        self,
        cache_size: int,
        seqlens: List[int],
        seqpos: List[int],
    ) -> CacheInputMetadata:
        masks = [[x >= seqlen - cache_size for x in range(seqlen)] for seqlen in seqlens]
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)
        cached_elements = torch.tensor([sum(mask) for mask in masks], device=self.device, dtype=torch.long)
        positions = torch.cat(
            [torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]
        ).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = positions % cache_size + batch_idx * cache_size

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            assert all(pos == 0 for pos in seqpos)
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(cache_size)
        elif subsequent_prefill:
            assert self.kv_seqlens is not None
            kv_seqlen = [
                s + cached_s.clamp(max=cache_size).item()
                for s, cached_s in zip(seqlens, self.kv_seqlens)
            ]
            mask = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens, kv_seqlen=kv_seqlen)
            mask = mask.make_local_attention_from_bottomright(cache_size)
        else:
            assert self.kv_seqlens is not None
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=cache_size,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=cache_size).tolist(),
            )

        return CacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
```
