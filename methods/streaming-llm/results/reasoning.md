Let me set up the problem cleanly. I want a pretrained LLM to run in a streaming setting — a chatbot over a day-long conversation, an assistant on an endless input — generating fluent text indefinitely, at constant memory and constant per-token latency, with no fine-tuning. Two things stand in the way. First, the KV cache: a decoder caches the Key and Value of every past token so each new token attends to the whole history, and that cache grows linearly with tokens seen — unbounded memory, rising latency. Second, length extrapolation: feed the model more tokens than its pretraining attention window (4K for Llama-2, say) and quality falls apart, because those long positions are out of distribution. So I need to bound the cache *and* stay stable past the trained window.

The obvious efficiency fix for the first problem is window attention: keep only the most recent L tokens' KV, evict everything older. Constant memory, constant latency once the cache is full — exactly the efficiency profile I want. The diagnostic perplexity curve should tell me whether this actually preserves the model's normal behavior as the stream gets long.

It collapses. The moment the sequence exceeds the cache size — that is, the very first token I have to evict — perplexity spikes, hard. Not a gentle drift; a cliff. And it's a strange cliff, because the tokens I'm evicting are the *oldest* ones, far away from what I'm currently predicting. Why would dropping a token from thousands of positions ago, that the current query barely needs, wreck the model? If anything, the recent window is what should matter for predicting the next token. So my mental model — "keep recent context, the distant past is irrelevant" — is exactly backwards about *something*. The recent window is plainly not what's holding the model together, because I kept it and the model still broke. The breakage is tied to losing the *beginning*.

I look at where the model actually puts its attention, pulling up the attention maps across all layers and heads of a trained model — Llama-2, and the same in MPT, Falcon, Pythia. There's a glaring pattern: past the first couple of layers, a huge fraction of attention mass lands on the *first few tokens of the sequence*, across most layers and heads — and it stays there regardless of what those first tokens actually say. The model is dumping enormous attention on positions 0, 1, 2, 3 even when predicting something a thousand tokens later that has nothing to do with them. That's bizarre on its face. Why would a well-trained model waste most of its attention on irrelevant tokens at the start?

I look at the softmax: attention weights over the attended keys are forced through a softmax, so they must sum to one: SoftMax(x)_i = e^{x_i} / Σ_j e^{x_j}. The denominator can't be zero; the weights can't all be zero. So on *every* step, for *every* head, the model is obligated to distribute a full unit of attention mass across the available tokens — even when the current query has no real match anywhere and the right amount of attention to pay to the context is essentially nothing. But "nothing" isn't an option; softmax won't let the weights vanish. So the model needs a place to dump the surplus attention it doesn't actually want to use. The natural move is to learn to dump it on a fixed, always-available set of tokens.

Why the *initial* tokens specifically, and not some others? Causal masking. Token 0 is visible to every subsequent token; token 5 is visible to everything from position 5 on; a token near the end is visible to almost nothing. So the initial tokens are the only positions guaranteed to be in every query's receptive field across the whole sequence. If the model is going to learn a consistent dumping ground, the only positions that are always there to dump onto are the first ones. So the first tokens get trained into the role — call them attention sinks.

If that story is right, the window-attention collapse should follow mechanically from removing the sinks, not just be "consistent with" it: take a query whose real content match is weak — say one key with logit 2 sitting in the recent window, and three sink keys carrying the inflated logits the model learned to put there, around 8 each, plus a handful of other ordinary keys near 0. With the sinks present, the denominator is roughly e^8·3 + e^2 + (a few)·e^0 ≈ 3·2981 + 7.4 + 5 ≈ 8955. The weight the real key actually receives is e^2/8955 ≈ 7.4/8955 ≈ 0.00083, and each sink holds e^8/8955 ≈ 0.333 — the three sinks together own about 0.999 of the mass. That matches the maps: the content key gets a tiny, *correct* sliver and the sinks soak up essentially everything else. Now evict the three sinks, exactly what window attention does. The denominator drops to e^2 + 5·e^0 ≈ 7.4 + 5 ≈ 12.4, and that same real key's weight jumps to 7.4/12.4 ≈ 0.60. The model intended to place 0.0008 of its attention on that key and is now forced to place 0.6 — a ~700× reallocation onto a key it barely wanted, with no change to the keys themselves. Every head in every layer suffers this renormalization simultaneously the instant the sinks leave the cache. So the cliff isn't about the evicted tokens being *informative*; the arithmetic says it's about them being the denominator's anchor, and the moment the anchor goes, the whole distribution snaps to a shape the model never saw in training. That's the out-of-distribution jump.

This also tells me what to test next. Is it the *position* of those first tokens or their *content* that matters? The mechanism above never used the meaning of the sink tokens — only that they sit where every query can see them and carry large learned logits. If that's true, the content should be replaceable. The brutal test: overwrite the first four tokens with a meaningless linebreak "\n" token and stream again. If perplexity stays broken, I need the opening *semantics* and this is a much harder problem; if it recovers, the sink is purely positional. Running it: the model still pours attention onto those positions and perplexity returns to the with-sinks level. So it is not the semantics — whatever token occupies the absolute starting positions becomes the sink. Good outcome: I don't need to preserve any particular *content*, just *something* in the sink positions to soak up the surplus.

So the fix suggests itself directly. Window attention failed only because it threw away the sinks, so don't. Keep a few of the very first tokens' KV permanently alongside the sliding window of recent tokens. The recent window carries the actual local context for predicting the next token; the sink tokens hold the attention distribution in its trained shape — they keep those big e^{x_sink} terms in everyone's denominator. The cache is two pieces: a small fixed set of initial sink tokens, plus a rolling cache of the most recent tokens. Both are bounded, so memory and latency stay constant, and no fine-tuning is touched — I'm only changing which KV entries I retain. That also fixes the ceiling of what this buys: the model only ever sees the sinks plus whatever sits in the recent window, so this is stable, constant-cost streaming, not extended long-term memory of anything that falls outside that window.

How many sink tokens? Run the start-size ablation: stream with 1 retained initial token, then 2, then 4, and read off the perplexity. One or two don't fully restore window-attention perplexity; around four does. Why more than one? These models weren't pretrained with a single consistent starting token — Llama-2 prepends "<s>" but before text chunking, so the token actually sitting at position zero is more or less random across training samples. With no single dedicated sink, the model spread the sink role across several initial positions, so I need to keep enough of them to recover the full denominator contribution. Four it is. Cheap insurance, negligible cache cost.

Now a real trap in the details: positions. Suppose start_size = 4, recent_size = 3, so the cache holds at most 7 tokens. The stream has run to text position 8, and after eviction the cache physically holds the KV of text tokens [0,1,2,3] (sinks) and [6,7,8] (recent window) — the middle, [4,5], is gone. I'm now decoding the token at text position 9. If I feed the positional encoding their *original text* positions — 0,1,2,3,6,7,8, then 9 for the new token — two things break. There's a gap (3 to 6) the model never sees in a contiguous sequence, and worse, as the stream runs on the recent positions march off toward infinity, far past the 4K pretraining window — reintroducing exactly the extrapolation failure I'm trying to avoid. But the cache only ever holds 7 tokens. So assign positions *within the cache*, contiguously: the seven cached tokens get 0,1,2,3,4,5,6, and the token being decoded gets 7. The text token that was at position 6 is now treated as cache-position 4. The positions are contiguous (no gap) and never exceed 7 (never leave the trained range), no matter how long the stream gets. That's the detail I have to get right: for a cache of start_size + recent_size = C tokens, the assigned positions are always 0..C regardless of how many text tokens have streamed by — bounded by C, always in-distribution, no matter how long the stream runs.

Concretely this depends on the positional scheme being relative, which RoPE and ALiBi both are. For RoPE I have to be careful *when* I apply the rotation. RoPE rotates a key by an angle proportional to its position, and the attention score between query at position m and key at position n depends only on m − n. If I cached the keys *already rotated* by their original text positions, the key that was text-position 6 would carry rotation-angle ∝ 6 baked in, and decoding at cache-position 7 the score would use m − n = 7 − 6 = 1 — but I want it to behave as cache-position 4, i.e. relative offset 7 − 4 = 3. The stale baked-in angle gives the wrong relative offset. So: store the cached keys *unrotated* in `past_key_values`, and apply the rotary transform fresh at each decoding step to the concatenated cached keys using their *within-cache* positions — the text-position-6 key gets rotated as position 4. For ALiBi it's simpler: ALiBi adds a bias linear in query-key distance, so I just apply a contiguous linear bias over the cache (distances 1,2,3,… across adjacent cache slots) rather than a "jumping" bias that would skip the evicted positions and feed the model a distance gap.

That handles deployment of an already-trained model. But the diagnosis suggests something I can fix at the source for *future* models. The whole mess comes from the model having no dedicated place to offload surplus attention, so it conscripts the initial tokens — and conscripts *several* of them because none was consistent. What if I give it one dedicated place? Prepend a single learnable token to *every* training sample — a designated sink token — so the model can dump surplus attention there consistently instead of scattering it across several initial real tokens. Then at streaming time I'd only ever need to keep that one sink token plus the recent window.

There may be an even cleaner framing. The reason a sink is needed at all is that softmax forces the weights to sum to one. What if I relax that? Replace softmax with a variant that adds 1 to the denominator: SoftMax₁(x)_i = e^{x_i} / (1 + Σ_j e^{x_j}). Now the weights are allowed to sum to *less* than one — the "+1" absorbs surplus, so the model can attend to almost nothing when it wants to, with no real token conscripted. This "+1" is exactly the same thing as prepending a token whose Key and Value are all-zero. Take ordinary softmax over the real keys *plus* one extra key with score x₀. Its weight on real key i is e^{x_i} / (e^{x₀} + Σ_{j real} e^{x_j}). Set the extra key's score to x₀ = 0: the denominator becomes 1 + Σ_{j real} e^{x_j}, which is exactly SoftMax₁'s denominator, and the weight on each real key matches term-for-term. So the "+1" is literally an extra key with logit 0. When does a key get logit 0? The attention score is qᵀk (before scaling); if that extra key's Key vector is all-zero, qᵀk = 0 for every query, so its logit is 0 for everyone — giving the "+1" automatically, with no learning. And its contribution to the output is weight · Value; if its Value vector is all-zero, it contributes the zero vector to the output regardless of its weight. So an all-zero (Key, Value) token reproduces SoftMax₁ exactly: it shifts the denominator by +1 and adds nothing to the output — a "zero sink" built into the math rather than learned. To make the training-side fix concrete I'd pretrain small (160M) models from scratch under matched settings: vanilla softmax, the zero-sink variant, and the prepended learnable sink token, then check which lets a model stream with a single anchor. My prediction from the diagnosis is that the dedicated sink (learnable token or zero sink) centralizes the role so one anchor suffices, where vanilla models need the four scattered initial positions — but that's a prediction; I'd want the perplexity numbers from the three runs to confirm it, since I can't read it off the math alone.

Let me write the inference method, since that's the part that works on existing models with no retraining. The implementation has two responsibilities. The cache manager keeps the first `start_size` tokens and the last `recent_size` tokens, evicting the middle. The attention adapter makes RoPE or ALiBi see positions inside the bounded cache rather than original text positions.

```python
import torch

def slice2d(x, start, end):
    return x[:, :, start:end, ...]

def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

def slice1d(x, start, end):
    return x[:, start:end, ...]

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class StartRecentKVCache:
    def __init__(self, start_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
```

The class also needs `evict_range` — a general cut of an arbitrary interior span, for tools that want to drop a mid-stream block rather than just the oldest-after-sinks — and `enable_streaming_llm`, which dispatches on `model.config.model_type` to pick each architecture's k/v sequence dimension and wire in its RoPE or ALiBi position-shift adapter (Llama, MPT, GPT-NeoX, Falcon each lay out the KV tensor's axes differently). Neither adds a new idea over what's above; the full module is in the answer.

Tracing `__call__` against the toy layout: with start_size = 4, recent_size = 3, a stream at seq_len = 9 has `seq_len (9) <= cache_size (7)` false, so it evicts — keeping `k_slice(k, 0, 4)` (indices [0,1,2,3]) concatenated with `k_slice(k, 6, 9)` (indices [6,7,8]), the seven entries [0,1,2,3,6,7,8] with the middle [4,5] dropped, exactly the [sinks | recent] split the position-shift adapter then numbers 0..6. `evict_for_space(num_coming=1)` trims one extra recent slot to make room for the incoming token — `k_slice(k, 7, 9)` leaves [0,1,2,3,7,8], six entries, the four sinks preserved unconditionally either way.
