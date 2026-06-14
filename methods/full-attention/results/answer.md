# Full-attention KV cache

The full-attention KV cache is the standard way to run a masked-attention decoder's
generation loop without redundant work. It computes each token's key and value exactly once — when
that token is first processed — stores them per layer, and at every later step appends only
the new token's key/value and attends the single new query over the entire stored history. It
keeps *every* key and value and evicts nothing, so generation follows the same mathematical
computation as re-running the model on the full prefix at each step, while removing the
recomputation that makes the naive loop quadratic.

## Problem it solves

Autoregressive generation `p(y) = prod_t p(y_t | y_{<t}, x)` is sequential in `t`: a token
cannot be produced until the previous one is. The naive way to advance a step — re-run the
whole decoder on the growing prefix `[y_1, ..., y_t]` and read the last position — is correct
but recomputes the projections and attention of all earlier positions every step, costing
`Theta(b n^2 d^2)` per layer for batch size `b` under `n <= d`. The task is to make per-step
cost reflect only the new work, with output numerically identical to the parallel
(training-style) pass.

## Key idea

In a causal self-attention layer, attention is masked one-directionally, so the key `k_j` and
value `v_j` of position `j` depend only on the token at `j` and the layers beneath it — never
on any later position. Hence `k_j, v_j` are **frozen** the moment `j` is produced.
Recomputing them every subsequent step is pure waste. Cache them:

- **Cache K and V, only K and V.** The query is transient — each step has exactly one query,
  used once and discarded, so caching `Q` buys nothing. Attention outputs are downstream and
  already consumed. What the current query needs from the past is precisely the set of
  (key, value) pairs to read over.
- **Decode step.** Project the one new token to `q_t, k_t, v_t`; append `k_t, v_t` to the
  per-layer cache (`K <- concat(K_past, k_t)`, `V <- concat(V_past, v_t)` along the sequence
  axis); attend `q_t` over the full `K, V`; emit. The future side of the causal mask is a
  no-op for the single-query decode row: the newest query may see every cached key, because
  every cached position is `<=` it. Padding and batch masks are separate bookkeeping and still
  need to match the combined past-plus-current length.
- **Prefill vs. decode.** The prompt is processed in one parallel masked pass that fills the
  cache; generation then appends one position's `k, v` per step.

## Cost accounting

Assume `k = v = d/h`, `m = n`, `n <= d`. Over `n` decode steps:

- **Arithmetic:** `Theta(b n d^2)` — same as the single parallel pass, and a factor `n`
  cheaper than the naive `Theta(b n^2 d^2)` re-run loop. The cache removes the *redundant*
  arithmetic, not the necessary arithmetic.
- **Memory access:** `Theta(b n^2 d + n d^2)` — reloading the growing `K, V` each step (first
  term) plus reloading the projection weights each step (second term).
- **Ratio memory/arithmetic:** `Theta(n/d + 1/b)`. Near 1 when `n ~ d` (long generation) or
  `b ~ 1` (single-sequence decode), so the step is **memory-bandwidth bound** on hardware
  whose compute throughput is ~100x its bandwidth. The `n/d` term — reloading the cache — is
  the structural offender; the `1/b` term shrinks by raising the batch size.
- **Cache size:** `b * h * (k + v) * L * n` — **linear** in sequence length and depth.

This is the honest ledger of exactness: full-context, no eviction, decoded output following the
full-prefix computation, paid for with state and per-step bandwidth that grow with the sequence.

## Why it is exact

Windowing (cap attention to recent positions) and memory compression (pool past positions
into fewer slots) both bound the cost but **change what the model computes** — the query no
longer reads the true set of past key/value pairs, so the output diverges from the model's
full-context output. The full KV cache keeps every key and value, so its attention read at
every step is over the genuine complete history. It is the exact no-compression execution of
the same decoder.

## Working code

The ordinary `DynamicCache` / `DynamicLayer` path maps `full_attention` layers to
`DynamicLayer`, while an unconfigured dynamic cache lazily creates `DynamicLayer` objects as
layer indices appear through the generic `Cache` container. The no-eviction update is a
`concat` along the sequence axis.

```python
import math
import torch


class DynamicLayer:
    """Per-layer KV cache that grows dynamically as more tokens are generated.
    keys/values: [batch, num_heads, seq, head_dim]; the seq axis (dim=-2) grows."""

    is_sliding = False

    def __init__(self, config=None):
        self.keys = None
        self.values = None
        self.is_initialized = False

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        # k_j, v_j are causally frozen once j is seen: append, never recompute.
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = torch.cat([self.keys, key_states], dim=-2)        # K <- concat(K_past, k_t)
        self.values = torch.cat([self.values, value_states], dim=-2)  # V <- concat(V_past, v_t)
        return self.keys, self.values                                # full history for the read

    def get_seq_length(self):
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_mask_sizes(self, query_length):
        return self.get_seq_length() + query_length, 0

    def get_max_cache_shape(self):
        return -1

    def reset(self):
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()

    def reorder_cache(self, beam_idx):
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))

    def crop(self, max_length):
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats):
        if self.get_seq_length() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


LAYER_TYPE_CACHE_MAPPING = {"full_attention": DynamicLayer}


class Cache:
    """List-like cache container with optional lazy layer replication."""

    def __init__(self, layers=None, layer_class_to_replicate=None):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError("provide either `layers` or `layer_class_to_replicate`, not both")
        if layers is None and layer_class_to_replicate is None:
            raise ValueError("provide `layers` or `layer_class_to_replicate`")
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length, layer_idx):
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def get_max_cache_shape(self, layer_idx=0):
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def reorder_cache(self, beam_idx):
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)


class DynamicCache(Cache):
    """Full-attention slice of a dynamic KV cache container."""

    def __init__(self, ddp_cache_data=None, config=None):
        layers = []
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = ["full_attention" for _ in range(decoder_config.num_hidden_layers)]
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]
            for layer_type in layer_types:
                cache_cls = LAYER_TYPE_CACHE_MAPPING.get(layer_type, DynamicLayer)
                layers.append(cache_cls(decoder_config))

        if ddp_cache_data is not None:
            for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
                if config is None:
                    layers.append(DynamicLayer())
                layers[layer_idx].update(kv_and_optional_sliding[0], kv_and_optional_sliding[1])

        if len(layers) == 0:
            super().__init__(layer_class_to_replicate=DynamicLayer)
        else:
            super().__init__(layers=layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)


def self_attention_step(layer, x_t, cache, layer_idx, attention_mask=None):
    """One masked self-attention layer advanced by one position via the KV cache.
    x_t: [b, 1, d]. Numerically identical to the last row of the parallel masked pass."""
    def split_heads(t):
        b, n, d = t.shape
        return t.view(b, n, layer.h, d // layer.h).transpose(1, 2)   # [b, h, n, d_k]

    def merge_heads(t):
        b, h, n, dk = t.shape
        return t.transpose(1, 2).contiguous().view(b, n, h * dk)

    q = split_heads(x_t @ layer.W_Q)                  # only the new query
    k_new = split_heads(x_t @ layer.W_K)              # this position's key
    v_new = split_heads(x_t @ layer.W_V)              # this position's value

    K, V = cache.update(k_new, v_new, layer_idx)      # append, get full frozen history

    d_k = q.shape[-1]
    scores = (q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # keep softmax unsaturated
    if attention_mask is not None:
        scores = scores + attention_mask              # mask shape spans past + current keys
    weights = scores.softmax(dim=-1)
    o = merge_heads(weights @ V)                      # [b, 1, d]
    return o @ layer.W_O
```

As a decode cache policy this is the no-compression case: no token scoring, no selection step,
and the `update` path only appends the newly computed keys and values. Every stored position
remains available to attention.
