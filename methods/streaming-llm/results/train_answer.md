We want a pretrained LLM to run in a genuinely streaming setting — a chatbot held over a day-long conversation, an assistant consuming an unbounded input — generating fluent, coherent text indefinitely at constant memory and constant per-token latency, and with no fine-tuning. Two obstacles stand in the way, and they pull against each other. The first is the KV cache: a decoder caches the Key and Value of every past token so each new token can attend to the whole history, and that cache grows linearly with the number of tokens seen, so memory and decoding latency blow up without bound. The second is length extrapolation: feed the model more tokens than its pretraining attention window — 4K for Llama-2, say — and quality collapses, because those long absolute positions are out of distribution. A real solution has to bound the cache *and* stay stable past the trained window simultaneously.

The clean efficiency fix for the first problem is window attention: keep only the most recent $L$ tokens' KV and evict everything older. That gives exactly the profile I want — constant memory, constant latency once the cache fills. But it collapses. The instant the sequence exceeds the cache size, that is, the moment I evict the very first token, perplexity spikes hard — a cliff, not a gentle drift. And it is a strange cliff, because the tokens being evicted are the *oldest* ones, far from what is being predicted now; the recent window, which should be all that matters for the next token, is exactly what I kept. So my intuition "keep recent context, the distant past is irrelevant" is backwards about something. The only other quality-competitive baseline, sliding window with recomputation, rebuilds the recent window's KV from scratch for every generated token; it recovers quality but pays quadratic attention per token, far too slow for real streaming. So neither bounded option works: one is fast and broken, the other correct and unusably slow.

I propose StreamingLLM, built on the phenomenon I call attention sinks. The diagnosis starts by looking at where trained models actually place attention. Across all layers and heads of Llama-2, MPT, Falcon, and Pythia, past the first couple of layers a surprisingly large fraction of attention mass lands on the *first few tokens of the sequence*, regardless of what those tokens say — the model dumps enormous attention on positions 0,1,2,3 even when predicting something a thousand tokens away that has nothing to do with them. The reason is the softmax normalization. Attention weights over the attended keys are forced through a softmax,
$$\mathrm{SoftMax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}},$$
so on every step, for every head, the weights must sum to one — the denominator cannot be zero and the mass cannot vanish. When the current query has no real match anywhere and the right amount of attention to pay is essentially nothing, "nothing" is not an option; the model must offload the surplus mass somewhere. It learns to dump it on a fixed, always-available set of tokens, which become sinks. And why the *initial* tokens specifically? Causal masking: token 0 is visible to every later position, whereas a token near the end is visible to almost nothing, so the opening tokens are the only positions guaranteed to be in every query's receptive field across the whole sequence — the only consistent dumping ground the model can learn.

This explains the window-attention cliff completely. Evicting the first tokens deletes the sinks; every query's softmax denominator loses the large $e^{x_{\text{sink}}}$ terms it was built around, so the entire attention distribution is renormalized into a shape the model never trained on — out of distribution, instantly, everywhere. The collapse was never about the distant tokens being informative; it was about them being the normalizer's anchor. A crucial check is whether it is the *content* or the *position* of those tokens that matters: replacing the first four tokens with a meaningless linebreak "\n" still draws the same heavy attention onto those positions and still restores perplexity, so any tokens sitting in the absolute starting positions serve as the sink. That is the good outcome — I need not preserve any particular content, only keep *something* in the sink slots.

So the fix writes itself: window attention failed only because it threw away the sinks, so don't. Keep a few of the very first tokens' KV permanently — the attention sinks — alongside the rolling sliding window of recent tokens. The recent window carries the actual local context for predicting the next token; the sink tokens hold the attention distribution in its trained shape. The cache is two bounded pieces, a small fixed set of `start_size` initial sink tokens plus a rolling `recent_size` most-recent tokens, with the middle evicted, so memory and latency stay constant and no weights are touched. How many sinks? Roughly four; one or two do not suffice. The reason is that these models were not pretrained with a single consistent starting token — Llama-2 prepends "<s>" but before text chunking, so the token actually at position zero is essentially random across training samples — and with no single dedicated sink the model spread the role across several initial positions. Keeping four is cheap insurance at negligible cache cost.

The one subtle trap that will silently ruin everything is positions. After eviction the cache might hold text positions $[0,1,2,3]$ (sinks) then $[6,7,8]$ (recent), and as the stream runs the recent positions march off toward infinity, far past the pretraining window — reintroducing the very extrapolation failure I am trying to avoid, and leaving a gap where the evicted positions were. The fix is to assign positions *within the cache*, contiguously: for the seven cached tokens use $[0,1,2,3,4,5,6]$, then $7$ for the token being decoded. The positions never exceed the cache size, so they always sit inside the trained range, and they are contiguous so there is no jump. This requires the positional scheme to be relative, which RoPE and ALiBi both are. For RoPE the timing matters: I keep the cached keys *unrotated* in `past_key_values` and apply the rotary transform fresh to the concatenated cached keys at each decoding step using their within-cache positions, so a key currently at cache-position 4 is rotated as position 4 even though it was the 6th token of the text — had I cached the already-rotated keys, they would carry their stale original positions and the gap would be baked in. For ALiBi it is simpler: it adds a linear distance bias to the scores, so I apply a contiguous linear bias over the cache rather than a bias that jumps over the evicted positions.

That handles deployment of an already-trained model with no retraining, which is the part that matters most. The same diagnosis also suggests fixes at the source for future models. Since the whole mess comes from the model having no dedicated place to offload surplus attention, I can give it one: prepend a single learnable Sink Token to every training sample, so the model dumps surplus there consistently and a single sink then suffices at streaming time. An even cleaner framing relaxes the constraint that caused the problem in the first place — the softmax forcing the weights to sum to one — by adding 1 to the denominator,
$$\mathrm{SoftMax}_1(x)_i = \frac{e^{x_i}}{1 + \sum_j e^{x_j}},$$
which lets the weights sum to *less* than one so the "+1" absorbs the surplus and the model can attend to almost nothing when it wants to. This is exactly equivalent to prepending a token whose Key and Value are all-zero: $e^{0}=1$ contributes $1$ to the denominator and the zero Value contributes nothing to the output — a Zero Sink built into the math rather than learned. These are pretraining-time variants; the inference method below works on existing models as they are. It is worth being clear about its scope: the model still only ever sees the sinks plus the recent cache, so this gives stable, constant-cost streaming, not extended long-term memory beyond what is retained.

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

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

def enable_streaming_llm(model, start_size, recent_size):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")

    kv_cache = StartRecentKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
```
