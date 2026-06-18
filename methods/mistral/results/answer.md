# Mistral 7B

## Method

Use a LLaMA-style decoder with RMSNorm, RoPE, SwiGLU, 32 layers, hidden size 4096, head dimension 128, 32 query heads, 8 key/value heads, and window size 4096. The serving-side method is the composition of:

- **Grouped-query attention:** project queries to 32 heads, keys and values to 8 heads, cache only the 8 KV heads, then repeat each KV head 4 times after cache read so attention has 32 query-compatible heads.
- **Sliding-window attention:** each query attends to at most `W=4096` recent keys. Dense prefill cost changes from $O(n^2)$ to $O(nW)$ for fixed $W$.
- **Depth-expanded receptive field:** local attention is stacked, so after `k` layers information can move about `kW` positions. With 32 layers and `W=4096`, the theoretical span is `32 * 4096 = 131072` tokens.
- **Rolling KV cache:** each layer stores at most `W` key/value positions per sequence. Absolute position `i` writes to slot `i mod W`; after wraparound the cache is unrotated before interleaving with a later prompt chunk.
- **Chunked prefill:** first prefill, later prefill chunks, and one-token decode use different masks so the attention-visible keys match the fixed local window exactly.

The released table configuration is `dim=4096`, `n_layers=32`, `head_dim=128`, `hidden_dim=14336`, `n_heads=32`, `n_kv_heads=8`, `window_size=4096`, `context_len=8192`, and `vocab_size=32000`.

## Code Artifact

This is the core implementation structure, matching the canonical `mistral_inference` attention and cache mechanics.

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

## Correctness Checks

- GQA cache factor: `n_heads / n_kv_heads = 32 / 8 = 4`; K and V are both reduced by this same factor, so the total KV cache is 4x smaller, not 8x.
- Window boundary: the implementation stores and masks at most `W` keys per query. Writing position `i` to `i mod W` overwrites the entry that has just left the local window.
- Receptive span: the local window is per layer; the theoretical long-range route is transitive across layers, giving about `kW`, not direct single-layer attention to `kW` tokens.
- Cache length reduction: for 32K tokens and `W=4096`, dense cache length divided by rolling cache length is `32768 / 4096 = 8`.
- RoPE positions remain absolute; only storage slots are modulo the cache size.
