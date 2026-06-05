Let me set up the problem cleanly. I want a pretrained LLM to run in a streaming setting — a chatbot over a day-long conversation, an assistant on an endless input — generating fluent text indefinitely, at constant memory and constant per-token latency, with no fine-tuning. Two things stand in the way. First, the KV cache: a decoder caches the Key and Value of every past token so each new token attends to the whole history, and that cache grows linearly with tokens seen — unbounded memory, rising latency. Second, length extrapolation: feed the model more tokens than its pretraining attention window (4K for Llama-2, say) and quality falls apart, because those long positions are out of distribution. So I need to bound the cache *and* stay stable past the trained window.

The obvious efficiency fix for the first problem is window attention: keep only the most recent L tokens' KV, evict everything older. Constant memory, constant latency once the cache is full — exactly the efficiency profile I want. So let me just try it and measure perplexity as the stream gets long.

It collapses. The moment the sequence exceeds the cache size — that is, the very first token I have to evict — perplexity spikes, hard. Not a gentle drift; a cliff. And it's a strange cliff, because the tokens I'm evicting are the *oldest* ones, far away from what I'm currently predicting. Why would dropping a token from thousands of positions ago, that the current query barely needs, wreck the model? If anything, the recent window is what should matter for predicting the next token. So my mental model — "keep recent context, the distant past is irrelevant" — is exactly backwards about *something*. The recent window is plainly not what's holding the model together, because I kept it and the model still broke. The breakage is tied to losing the *beginning*.

Let me actually look at where the model puts its attention. Pull up the attention maps across all layers and heads of a trained model — Llama-2, and the same in MPT, Falcon, Pythia. There's a glaring pattern: past the first couple of layers, a huge fraction of attention mass lands on the *first few tokens of the sequence*, in basically every head, every layer — and it stays there regardless of what those first tokens actually say. The model is dumping enormous attention on positions 0, 1, 2, 3 even when predicting something a thousand tokens later that has nothing to do with them. That's bizarre on its face. Why would a well-trained model waste most of its attention on irrelevant tokens at the start?

Look at the softmax. Attention weights over the attended keys are forced through a softmax, so they must sum to one: SoftMax(x)_i = e^{x_i} / Σ_j e^{x_j}. The denominator can't be zero; the weights can't all be zero. So on *every* step, for *every* head, the model is obligated to distribute a full unit of attention mass across the available tokens — even when the current query has no real match anywhere and the right amount of attention to pay to the context is essentially nothing. But "nothing" isn't an option; softmax won't let the weights vanish. So the model needs a place to dump the surplus attention it doesn't actually want to use. It learns to dump it on a fixed, always-available set of tokens. Those tokens become sinks for unwanted attention — attention sinks.

Why the *initial* tokens specifically, and not some others? Causal masking. Token 0 is visible to every subsequent token; token 5 is visible to everything from position 5 on; a token near the end is visible to almost nothing. So the initial tokens are the only positions guaranteed to be in every query's receptive field across the whole sequence. If the model is going to learn a consistent dumping ground, the only positions that are always there to dump onto are the first ones. So the first tokens get trained into universal attention sinks.

Now the window-attention collapse makes complete sense. When I evict the first tokens, I delete the sinks. The softmax denominator for every query loses the big e^{x_sink} terms it was built around — I've removed a large chunk of the denominator — and the entire attention distribution gets renormalized into a shape the model was never trained on. Out of distribution, instantly, everywhere. That's the cliff. It was never about the distant tokens being informative; it was about them being the *normalizer's anchor*.

I should check whether it's really the position or the content of those first tokens. If the model needs the *meaning* of the opening tokens, this is a much harder problem. So substitute: replace the first four tokens with a meaningless linebreak "\n" token and run again. The model still pours attention onto those positions, and reintroducing them restores perplexity to where it'd be with the genuine opening tokens. So it is not the semantics — any tokens sitting in those absolute starting positions serve as the sink. That's the good outcome: I don't need to preserve any particular *content*, I just need *something* in the sink positions to soak up the surplus attention.

So the fix writes itself. Window attention failed only because it threw away the sinks. So don't. Keep a few of the very first tokens' KV permanently — the attention sinks — *alongside* the sliding window of recent tokens. The recent window carries the actual local context for predicting the next token; the sink tokens hold the attention distribution in its trained shape. The cache is two pieces: a small fixed set of initial sink tokens, plus a rolling cache of the most recent tokens. Both are bounded, so memory and latency stay constant, and no fine-tuning is touched — I'm just changing which KV entries I retain.

How many sink tokens? My probe said roughly four initial tokens are needed to fully restore window-attention perplexity, and one or two don't cut it. The reason there's more than one is that these models weren't pretrained with a single consistent starting token — Llama-2 prepends "<s>" but before text chunking, so the token actually sitting at position zero is more or less random across training samples. With no single dedicated sink, the model spread the sink role across several initial positions. So I keep four. Cheap insurance, negligible cache cost.

Now a subtle trap that will silently ruin everything if I get it wrong: positions. My cache, after eviction, might hold tokens [0,1,2,3] (sinks) then [6,7,8] (recent window), and I'm decoding the next token. If I feed the positional encoding their *original text* positions — 0,1,2,3,6,7,8 — there's a gap, and as the stream runs on the recent positions march off toward infinity, way past the pretraining window. That reintroduces the extrapolation failure I'm trying to avoid: positions the model never trained on. The cache only ever holds a bounded number of tokens, so I should assign positions *within the cache*, contiguously: 0,1,2,3,4,5,6 for the seven cached tokens, then 7 for the token being decoded. The positions never exceed the cache size, so they're always inside the trained range, and they're contiguous so there's no jump. This is the detail that makes it actually work past the window.

Concretely this depends on the positional scheme being relative, which RoPE and ALiBi both are. For RoPE, I have to be careful *when* I apply the rotation: cache the keys *before* rotating them by position, and apply the rotary transform to the cached keys fresh at each decoding step using their *within-cache* positions — so a key that's currently at cache-position 4 gets rotated as position 4, even though it was the 6th token of the text. (If I'd cached the already-rotated keys, they'd carry their stale original positions and the gap would be baked in.) For ALiBi it's simpler: ALiBi adds a linear bias by distance to the scores, so I just apply a contiguous linear bias over the cache rather than a "jumping" bias that skips the evicted positions.

That handles deployment of an already-trained model. But the diagnosis suggests something I can fix at the source for *future* models. The whole mess comes from the model having no dedicated place to offload surplus attention, so it conscripts the initial tokens. What if I give it a dedicated place? Prepend a single learnable token to *every* training sample — a designated "sink token" — so the model can dump its surplus attention there consistently, instead of scattering it across several initial real tokens. Then at streaming time I'd only ever need to keep that one sink token plus the recent window.

There's an even cleaner framing. The reason a sink is needed at all is that softmax forces the weights to sum to one. What if I just relax that? Replace softmax with a variant that adds 1 to the denominator: SoftMax₁(x)_i = e^{x_i} / (1 + Σ_j e^{x_j}). Now the attention weights are allowed to sum to *less* than one — the "+1" absorbs the surplus, so the model can attend to almost nothing when it wants to, with no need to dump mass on any real token. Notice this "+1" is exactly equivalent to prepending a token whose Key and Value are all-zero: e^{0}=1 contributes 1 to the denominator and the zero Value contributes nothing to the output. So it's a "zero sink" — a sink built into the math rather than learned. I'd want to validate all this by pretraining small (160M) models from scratch under matched settings: vanilla softmax, the zero-sink variant, and the prepended learnable sink token, and check which one lets a model stream with just a single anchor. My expectation from the diagnosis: a single dedicated sink token should let one anchor suffice, where vanilla models need the four scattered ones.

Let me write the inference method, since that's the part that works on existing models with no retraining. The KV cache keeps the first `start_size` tokens (the sinks) and the last `recent_size` tokens, evicting the middle; positions are assigned within the cache.

```python
import torch

class StartRecentKVCache:
    """KV cache that retains `start_size` attention-sink tokens at the front
    plus a rolling window of the most recent `recent_size` tokens."""
    def __init__(self, start_size=4, recent_size=2000, k_seq_dim=2, v_seq_dim=2):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = lambda x, a, b: x[:, :, a:b, ...]   # slice along seq dim
        self.v_slice = lambda x, a, b: x[:, :, a:b, ...]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        # keep [0:start_size] (the sinks) concatenated with the LAST recent_size
        return [
            [
                torch.cat([self.k_slice(k, 0, self.start_size),
                           self.k_slice(k, seq_len - self.recent_size, seq_len)],
                          dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, self.start_size),
                           self.v_slice(v, seq_len - self.recent_size, seq_len)],
                          dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        # proactively drop the middle so num_coming new tokens fit, keeping sinks
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        end_recent = seq_len - self.recent_size + num_coming
        return [
            [
                torch.cat([self.k_slice(k, 0, self.start_size),
                           self.k_slice(k, end_recent, seq_len)], dim=self.k_seq_dim),
                torch.cat([self.v_slice(v, 0, self.start_size),
                           self.v_slice(v, end_recent, seq_len)], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]

@torch.no_grad()
def streaming_generate(model, token_stream, kv_cache):
    past = None
    for token in token_stream:
        past = kv_cache.evict_for_space(past, num_coming=1)
        # positions are assigned WITHIN the cache (contiguous, bounded),
        # not the original text positions -- keeps RoPE/ALiBi in-distribution.
        # For RoPE: keys are cached pre-rotation and rotated by cache-position here.
        logits, past = model(token, past_key_values=past, use_cache=True)
        yield sample(logits)
```

The causal chain: a Transformer's KV cache grows without bound, so streaming needs eviction; window attention bounds it but collapses the instant the first tokens are evicted; attention maps show the model parks a huge, content-independent share of attention on the initial tokens — attention sinks — because softmax forces the weights to sum to one and the model needs somewhere to dump surplus mass, and causal masking makes the always-visible initial tokens the natural dump (confirmed to be about position, not content, via the linebreak swap, with ~4 sinks needed); so keep the first few tokens' KV permanently as sinks alongside the rolling recent window, assign positions within the bounded cache so RoPE/ALiBi stay in-distribution, and the model streams stably at constant memory and latency with no fine-tuning — while a dedicated learnable sink token (or a SoftMax₁ "+1" / zero-sink) baked in at pretraining would let a single anchor do the job.
