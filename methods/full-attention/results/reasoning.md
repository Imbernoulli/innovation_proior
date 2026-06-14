Let me start from what actually hurts when I run one of these attention decoders to *generate* text, because training feels great and generation feels slow, and I want to know precisely why. In training I have the whole target sequence in hand, so a masked self-attention layer is one big batched `softmax(QK^T / sqrt(d_k)) V` over all `n` positions at once. Every position's query, key, value is a cheap linear projection of that position's layer input, I stack them into matrices, one matmul makes all the pairwise scores, the causal mask sets the strict-upper-triangle logits to `-inf` so position `i` only sees `j <= i`, softmax, one more matmul reads out the values. It is perfectly parallel across the sequence, which is the whole reason these things train fast on a GPU or TPU. But generation is `p(y) = prod_{t=1}^n p(y_t | y_{<t}, x)`, and that product is sequential by construction: I cannot compute `y_t` until I have sampled `y_{t-1}`, because `y_{t-1}` is literally the input token at the next step. The parallelism that made training fast is simply not available — the layer's output at one position decides the token fed in at the next, so the steps are chained.

So how do I actually produce a token? The most literal thing, and the thing I'd write first, is: keep the growing list of tokens `[y_1, ..., y_t]`, feed the whole list through the decoder, look at the last position's logits, take the argmax (or sample), append it, and loop. This is unarguably correct — it is exactly the forward pass training uses, I'm just reading off the final position instead of all of them. Let me run the cost of this loop in my head, because that's where the pain is. At step `t` I feed in a length-`t` prefix. For batch size `b`, the layer projects `Q, K, V` for all `t` positions: three matmuls of a `[b, t, d]` activation against `[d, d]` weights, `Theta(b t d^2)` arithmetic per layer just for the projections, plus `Theta(b t^2 d)` for the attention score/read work. Now sum that over the whole generation, `t = 1, 2, ..., n`: `Theta(b d^2 (1 + 2 + ... + n)) + Theta(b d (1^2 + 2^2 + ... + n^2)) = Theta(b n^2 d^2) + Theta(b n^3 d)`. In the regime I am using for the performance count, `n <= d`, so the projection term dominates and the naive loop is `Theta(b n^2 d^2)` arithmetic per layer. That is a factor of `n` too much relative to a single parallel pass's `Theta(b n d^2)`. That's the wall: the loop is quadratically recomputing work in the length, and I have no idea yet *where* the waste is, only that there is a lot of it.

Let me stare at one step and ask what, of all that arithmetic, is genuinely new. At step `t` I recompute the projections of positions `1, ..., t`. But position `j < t` — its token `y_j` was fixed steps ago, it hasn't changed, and its layer-input hasn't changed either, because the layers below are themselves causal: position `j`'s hidden state at any layer depends only on tokens `<= j`. So the query `q_j`, the key `k_j`, the value `v_j` I compute for position `j` at step `t` are the same mathematical vectors I computed for position `j` at step `j`, and at step `j+1`, and at every step in between. I am recomputing the same vectors over and over. The redundancy isn't an accident of my implementation — it's forced by the loop structure: each step redoes the entire prefix, and the prefix's interior is frozen. So almost all of that `b n^2 d^2` projection work is recomputation of frozen quantities.

Now, *why* are those quantities frozen — I want to be exact about it, because the exactness is what tells me which ones I can safely stash and reuse versus which I genuinely must recompute. The mask makes attention one-directional: position `i` attends to `j <= i`, never to `j > i`. Read that the other way around. What position `j` *advertises* to the queries that will look at it — its key `k_j` — and what it *contributes* — its value `v_j` — are computed from `y_j` and the layers beneath, and they exist before any later position `i > j` exists. No future token can reach back and change `k_j` or `v_j`; the causal mask is precisely the statement that information flows only forward in position. So `k_j` and `v_j` are invariant for the entire rest of the generation once `j` has been produced. That's a hard invariance, not a heuristic — it falls straight out of the masking I already rely on for the autoregressive property. Recomputing `k_j, v_j` at every later step is pure waste.

So the fix writes itself: compute each token's key and value exactly once, the moment that token is first processed, and *store* them. Then at step `t` I don't re-run the whole prefix — I only do the new work: project the single new token into its query `q_t`, its key `k_t`, its value `v_t`; take the stored keys and values of all earlier positions; attend `q_t` over the full set `[k_1, ..., k_t]`, `[v_1, ..., v_t]`; emit. The stored state grows by exactly one key and one value per step, and a step's fresh arithmetic is `Theta(b d^2)` for the one new position's projections plus `Theta(b t d)` for the one query reading over `t` keys — no longer `Theta(b t d^2)`.

But wait — let me be careful about *which* things I store, because it's tempting to store everything and that would be its own waste. Do I cache the queries? The query is needed only at the step that owns it: each generation step has exactly one query, the current position's, and once I've used `q_t` to read out `o_t` and emit `y_{t+1}`, I never look at `q_t` again — there is no later computation that re-attends *for* a past query, because each past position already emitted its token. So caching `Q` buys nothing; `q_t` is born, used once, discarded. What about the attention outputs `o_j`, or the post-FFN hidden states of past positions? Those fed the next layer at their own step and were consumed then; the current step's computation for the current position never reads a past position's output directly — what it needs *from* the past is exactly the set of (key, value) pairs to attend over, nothing else. So the thing to store is precisely the keys and the values, and only the keys and the values. The query is transient, the outputs are downstream and consumed; the keys and values are the frozen, repeatedly-needed past. A key-value cache.

Let me write the single-step layer with this in mind and make sure it's *exactly* the parallel result, because the whole point is that it changes nothing the model computes. The step holds `prev_K`, `prev_V` for this layer (shape `[b, h, m, d_k]` after `m` prior positions). I project the new token `x` of shape `[b, d]` into `q = x W^Q`, `k_new = x W^K`, `v_new = x W^V`. I append: `K = concat(prev_K, k_new)`, `V = concat(prev_V, v_new)` along the position axis, giving length `m+1`. Then `logits = q . K^T / sqrt(d_k)`, `weights = softmax(logits)`, `o = weights . V`, `y = o W^O`. The `1/sqrt(d_k)` factor is the same scaled-dot-product correction as in the parallel layer; it is not part of the cache idea, but dropping it would no longer be the same attention layer. Here's a subtlety I should pin down rather than copy by reflex: do I need the causal `-inf` mask in this step? In the parallel pass I masked because some of the `n` queries were earlier than some of the keys. But here there is exactly *one* query, the newest position, and every key in the cache belongs to a position `<=` it — they're all legal by construction. So the causal future mask is unnecessary for the single-query decode row; the lone query is allowed to see all `m+1` cached keys, which is exactly the lower-triangular row the mask would have left untouched anyway. Padding or batching masks are separate bookkeeping, but the future mask itself has become a no-op. Good — that confirms the incremental step computes the same row of the same attention the parallel pass would, so the cache is lossless, not an approximation.

I should separate the two phases cleanly, because they're different shapes of the same computation. There's the *prompt*: I'm handed `[x_1, ..., x_p]` all at once, with nothing to generate yet, so I run them through the stack as one parallel masked pass — the fast training-style kernel — and that pass, as a side effect, produces `k_1..k_p` and `v_1..v_p` for every layer, which I drop straight into the cache. Then there's *generation*: from there I append one token's `k, v` per step. So the prompt is amortized over a single parallel pass that fills the cache, and only the generated tail pays the one-token-at-a-time price. Prefill fills, decode appends.

Now I want to actually account for what this incremental loop costs, end to end, because I claimed it turns the dominant arithmetic from `n^2` into `n`, and I should check whether the per-step work is now dominated by arithmetic or by something else. Take the standard simplifying assumptions for an apples-to-apples count: keys and values are the same head dimension `k = v = d/h`, the memory length equals the generated length `m = n`, and `n <= d` (the usual regime — subword sequences are shorter than the model width). Across all `n` steps, the arithmetic is dominated by the projections, `Theta(d^2)` per step times `n` steps times the batch, so `Theta(b n d^2)` total. That is the same asymptotic arithmetic as the single parallel pass over `n` positions — which is the point: the cache hasn't reduced the *necessary* arithmetic below the parallel pass, it has removed the *redundant* arithmetic the naive loop added. Against the naive re-run-the-prefix loop's `Theta(b n^2 d^2)`, the cache saves a clean factor of `n` in arithmetic. So far so good; the arithmetic is no longer quadratic.

But here's the thing I'd miss if I only counted FLOPs, and the hardware fact from the background makes me look: on these accelerators the arithmetic throughput can be a couple of orders of magnitude higher than the memory bandwidth, so the binding constraint on a kernel is often not how much it computes but how many bytes it moves per unit of compute. Let me count the *memory traffic* of the incremental loop, not just the arithmetic. At step `t` I have to read the cached `K` and `V` to do the attention read. One of those tensors has size `b h m k = b h (d/h) m = b d m`; reading both changes only a constant factor, so the cache traffic at step `t` is `Theta(b d t)`. Sum over `t = 1..n`: `Theta(b d (1 + 2 + ... + n)) = Theta(b n^2 d)` of cache traffic across the whole generation. On top of that I reload the projection weights `W^Q, W^K, W^V, W^O`, size `Theta(d^2)`, once per step, `Theta(n d^2)` total. So total memory access is `Theta(b n^2 d + n d^2)`.

Now divide memory by arithmetic to get the ratio that decides whether I'm compute-bound or bandwidth-bound: `Theta(b n^2 d + n d^2) / Theta(b n d^2) = Theta(n/d + 1/b)`. Look at that. In the parallel training pass the analogous ratio was tiny — `O(1/k + 1/(bn))`, comfortably compute-bound. But here it's `n/d + 1/b`, and the moment `n` approaches `d` (a long generation) or the batch `b` is small (interactive, one-sequence decoding), this ratio is on the order of 1. A ratio near 1 means I'm moving roughly as many bytes as I'm doing FLOPs, which on hardware with `~100x` more compute than bandwidth means the compute units sit idle waiting on memory — the incremental decode is *memory-bandwidth bound*. And I can see exactly which term is the offender: the `n/d`, which traces back to reloading the cached `K` and `V` at every step. At step `t` those tensors hold `Theta(b t d)` values, and summing that streaming traffic over `t = 1..n` gives the `Theta(b n^2 d)` term. The `1/b` term is the gentle one; I can just push the batch size up, memory permitting, and it shrinks. The `n/d` term is the structural one: the very state I introduced to *avoid* recomputing keys and values now has to be *streamed* from memory on every step, and it grows with the sequence.

And there's the other cost, the one in space rather than time, that I should name plainly: the cache itself. Each generated token adds one key and one value vector, per head, per layer. So the cache size is `b * h * (k + v) * L * n` — linear in the sequence length `n` and in the depth `L`. Linear memory growth, and as I just derived, linear-in-`n` per-step bandwidth to reload it. That's the honest ledger of keeping every key and value: I have made generation correct and non-redundant — exactly the full-context result, every past position attended over, nothing dropped — and the bill for that exactness is a state that grows with the sequence and a per-step cost dominated by streaming that state.

Let me make sure I haven't quietly approximated anything, because the value of this construction is that it's *exact*. The windowing idea — cap attention to a fixed number of recent positions — would have bounded both the memory and the traffic, but it changes what the model computes: a query can no longer see positions outside the window, so any dependency longer than the window is silently dropped. The summarize/compress-the-memory idea has the same character: the query attends over fewer, pooled key/value slots instead of the true set, so the output diverges from the model's full-context output. Both are legitimate speed/quality trades, but neither is the model's actual full-prefix computation. The full key-value cache, by contrast, retains every key and every value and evicts nothing, so the attention read at every step is over the genuine, complete set of past positions, and the decoded sequence follows the same mathematical computation as the naive full-prefix re-run — only without the recomputation. Exactness, full context, no eviction: this is the uncompressed execution of the decoder.

So let me write it as the code I'd actually ship, and the right shape is to make the cache a small object the attention layer talks to. The cache is per-layer: each attention layer owns a dynamic layer holding `keys` and `values` tensors of shape `[b, h, seq, d_k]`. Its one real operation is `update`: given this step's new `key_states` and `value_states`, append them along the sequence axis (`dim=-2`, the `seq` axis of `[b, h, seq, d_k]`) and return the full grown tensors for the attention read. Lazily initialize the tensors as empty on the first call so I don't need to know the length up front. A container can either be built from a configured list of layer objects or start empty and lazily replicate the full-attention layer whenever a new layer index appears. If a model configuration resolves a layer to ordinary full attention, the mapping points to this dynamic layer; other cache layer policies are outside this exact no-eviction object. That `concat`-along-the-sequence-axis *is* the entire no-eviction mechanism — the frozen-past invariance I derived is what makes a plain append correct.

```python
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
        # k_j, v_j are causally frozen once position j is seen, so a step only ever
        # appends its new (k, v) and reuses every stored one unchanged.
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = torch.cat([self.keys, key_states], dim=-2)      # K <- concat(K_past, k_t)
        self.values = torch.cat([self.values, value_states], dim=-2)  # V <- concat(V_past, v_t)
        return self.keys, self.values                              # full history for the read

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
```

And the attention layer's single incremental step is straightforward now: project the one new token, push its key and value into the cache, and read the lone query over the full returned `K, V`; the future part of the causal mask is unnecessary in the one-token no-padding row because there is one query and it is the newest position, while padding and batch masks still have to span the combined past-plus-current length:

```python
import math


def self_attention_step(layer, x_t, cache, layer_idx, attention_mask=None):
    """One masked self-attention layer advanced by one position, using the KV cache.
    x_t: [b, 1, d] hidden state of the single new position.
    Numerically identical to the last row of the parallel masked pass."""
    def split_heads(t):
        b, n, d = t.shape
        return t.view(b, n, layer.h, d // layer.h).transpose(1, 2)   # [b, h, n, d_k]

    def merge_heads(t):
        b, h, n, dk = t.shape
        return t.transpose(1, 2).contiguous().view(b, n, h * dk)

    q = split_heads(x_t @ layer.W_Q)        # [b, h, 1, d_k]  only the new query
    k_new = split_heads(x_t @ layer.W_K)    # [b, h, 1, d_k]  this position's key
    v_new = split_heads(x_t @ layer.W_V)    # [b, h, 1, d_k]  this position's value

    K, V = cache.update(k_new, v_new, layer_idx)     # append, get full frozen history

    d_k = q.shape[-1]
    scores = (q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # 1/sqrt(d_k): keep softmax unsaturated
    if attention_mask is not None:
        scores = scores + attention_mask             # mask shape spans past + current keys
    weights = scores.softmax(dim=-1)
    o = merge_heads(weights @ V)                     # [b, 1, d]
    return o @ layer.W_O
```

Let me trace the whole causal chain back so I'm sure it holds together. Generation is sequential because each token feeds the next, so the parallelism that makes training fast is gone and I'm stuck in a one-step-at-a-time loop. The naive version of that loop re-runs the entire prefix every step, which is `Theta(b n^2 d^2)` arithmetic under `n <= d`, and staring at it I saw the waste is recomputation: the key and value of any past position are frozen the instant that position is produced, because the causal mask means information only flows forward in position, so no later token can revise them. That invariance says to compute each key and value once and store them; queries are transient (one per step, used once) and outputs are downstream and consumed, so the cache holds keys and values and nothing else. The incremental step then appends the new token's `k, v` to the stored set and attends the single new query over the full set; in the no-padding one-token row, all cached keys are legal for the newest query, while any padding mask still spans the combined past-plus-current length. Counting the cost shows arithmetic drops back to `Theta(b n d^2)`, matching the parallel pass and a factor of `n` cheaper than the naive loop, but the memory-to-arithmetic ratio rises to `Theta(n/d + 1/b)`, near 1 for long sequences or small batches, so the step becomes bandwidth-bound on the reload of the growing `K, V`; and the cache itself costs memory linear in sequence length and depth. The landing artifact is a per-layer cache whose update is a single `concat` along the sequence axis, retaining every key and value with no eviction — the exact, full-context reference loop.
