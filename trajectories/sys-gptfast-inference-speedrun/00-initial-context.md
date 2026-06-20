## Research question

A 7-billion-parameter Llama-2 model, fully trained, sits on a single A100-80GB. The job is to *serve*
it: feed it a short prompt and let it autoregressively decode a few hundred tokens, one at a time. The
metric is **decoding throughput in tokens per second at batch size 1** — the latency a single user
feels, which at batch 1 equals "tokens/s/user." Higher is better. The hardware is frozen at one
A100-80GB (the multi-GPU rung extends to 8×A100 on the same node), the model is frozen at Llama-2-7B,
the weights are frozen (this is inference, not training), and the decoding semantics are frozen: the
output distribution must stay the model's own, so any speedup that changes what the model would have
said is only allowed if it provably preserves that distribution or costs no observable quality.

The thing that makes this hard, and interesting, is that single-stream autoregressive decoding is one
of the most *inefficient* workloads a GPU ever runs. Each new token requires a full forward pass
through all 32 layers, but it processes exactly **one** token position at a time. A100 tensor cores can
do ~312 TFLOP/s of bf16 matmul; a 7B model is ~13.5 GB of bf16 weights and A100 HBM delivers ~2 TB/s.
To generate one token you must read all 13.5 GB of weights from memory once, and you do only ~2·7B =
14 GFLOP of useful arithmetic against it. The arithmetic intensity is on the order of *one* FLOP per
byte, so the matmuls finish long before the weights finish arriving: decoding is hard
**memory-bandwidth bound**, not compute bound. At 2 TB/s the floor is ~2e12 / 13.5e9 ≈ 148 tokens/s
just to stream the weights once per token — and that is the *ceiling* a perfect implementation would
approach, not the speed a naive one gets.

So the single free variable is the **decoding method**: how the forward pass is scheduled on the
hardware, how the weights are represented in memory, and how many model evaluations it takes to commit
a token. Every rung below is one named change to the decode path that moves the tokens/s number up,
and most of them are aimed squarely at the memory-bandwidth wall — either making the launch overhead
disappear so the GPU actually runs at bandwidth, or shrinking the bytes-per-token that have to cross
that wall.

## The fixed substrate

The model is a standard Llama-2-7B decoder-only transformer in bf16: token embedding, 32
`TransformerBlock`s (each an RMSNorm → multi-head attention with rotary embeddings → residual, then
RMSNorm → SwiGLU feed-forward → residual), a final RMSNorm, and a bias-free linear head to the 32000-
token vocabulary. `dim=4096`, `n_head=32`, `head_dim=128`, `intermediate_size=11008`. The whole thing
is ~6.7B parameters, ~13.5 GB of bf16 weights.

Decoding is the usual two-phase loop. A **prefill** pass runs the prompt (here a tiny 5-token prompt)
through the model once to populate state and emit the first token. Then a **decode** loop runs once
per new token: it takes the single most-recent token, does one forward pass, samples the next token,
and appends it. Sampling is fixed (temperature 0.8, top-k 200 in the benchmark). The loop is the frozen
scaffold; every rung is a change to *how* that forward pass executes or *how* the weights it reads are
stored.

## Background

The decode loop, written the most direct way, looks like this — a per-token Python step that calls the
model and samples:

```python
def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def decode_one_token(model, x, input_pos, **sampling_kwargs):
    # input_pos: [B, 1] — the single most recent token's position
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model, cur_token, input_pos, num_new_tokens, callback, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(model, cur_token, input_pos, **sampling_kwargs)
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()
    return new_tokens, new_probs
```

Two facts about this loop set up everything that follows. First, **each `decode_one_token` is a tiny
amount of GPU work wrapped in a lot of Python and framework overhead.** Launching the dozens of CUDA
kernels that make up one transformer layer — through PyTorch's eager dispatch, autograd bookkeeping,
allocator calls, and Python loop — takes real CPU time, and at batch 1 the GPU kernels are so small
that the GPU can finish them faster than the CPU can queue the next ones. The GPU spends much of its
time *idle*, waiting on the CPU. The hardware's 2 TB/s is there; the loop just never lets the device
run flat-out against it.

Second, **the attention layer keeps growing state.** Each layer caches the keys and values of every
token seen so far (the KV-cache) so the next token can attend back over the whole sequence without
recomputing. A direct implementation grows these tensors by concatenation every step — which means
reallocating and recopying a tensor whose size changes on every iteration. Dynamic shapes like that
are exactly what a kernel-fusing compiler cannot specialize on, and the reallocation is its own
overhead.

The attention itself, for one decoded token, reads the entire KV-cache; the feed-forward and
projections read their weight matrices. For a 7B model that is ~13.5 GB of bf16 weights streamed from
HBM per token — and as established above, *that read is the bottleneck.* Anything that reduces idle
time (so the read actually runs at bandwidth) or reduces the number of bytes read (so there is less to
stream) attacks the real constraint; anything that only saves arithmetic does not, because arithmetic
was never the limiter here.

## Evaluation settings

The yardstick is wall-clock decoding throughput: generate a fixed number of new tokens (200 in the
benchmark) from a fixed 5-token prompt, time it after a warmup/compile pass, and report tokens/s as the
mean over several samples. All numbers are at **batch size 1** on an **A100-80GB power-limited to
330 W**, so tokens/s equals tokens/s/user. The reported `Bandwidth achieved (GB/s)` —
`model_size × tokens_per_sec / 1e9` — is the diagnostic that tells you how close the run is to the HBM
ceiling: a method that is bandwidth-bound is "fast" exactly to the extent it pushes that number toward
the ~2 TB/s peak.

For rungs that change the model's numerics (the quantization rungs), correctness is checked with the
EleutherAI evaluation harness on multiple-choice tasks (`hellaswag`, `winogrande`) via `eval.py`, so a
throughput win that silently wrecked accuracy would show up. Rungs that are provably
distribution-preserving (speculative decoding) or purely scheduling/parallelism changes (compile,
tensor parallelism) leave the output distribution untouched by construction and need no such check.

## Code framework

The serving scaffold is the model definition plus the generate loop. The forward pass and the
per-token step exist; the slots each rung fills are the *representation* of the linear weights, the
*scheduling* of the loop, the *number of model calls* per committed token, and the *sharding* of the
weights across devices.

```python
# model.py — the frozen transformer (Llama-2-7B), bf16
class KVCache(nn.Module):
    # TODO: how the per-layer key/value state is stored across decode steps
    ...

class Attention(nn.Module):
    def forward(self, x, freqs_cis, mask, input_pos):
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        # ... reshape, rotary embed ...
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)   # read/extend the cache
        y = scaled_dot_product_attention(q, k, v)          # attend over the cache
        return self.wo(y)

class Transformer(nn.Module):
    def setup_caches(self, max_batch_size, max_seq_length):
        # TODO: how decoding state is preallocated before the loop
        ...
    def forward(self, idx, input_pos):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, input_pos, self.freqs_cis[input_pos], mask)
        return self.output(self.norm(x))     # logits over the vocabulary

class Linear:
    # the projection layers (wqkv, wo, w1/w2/w3, output) — the bulk of the 13.5 GB
    # TODO: how each weight matrix is represented in memory and multiplied at decode time
    ...

# generate.py — the decode loop (the substrate shown in Background)
def generate(model, prompt, max_new_tokens, *, draft_model, speculate_k, **kw):
    model.setup_caches(max_batch_size=batch_size, max_seq_length=...)
    next_token = prefill(model, prompt, input_pos, **kw)      # phase 1: run the prompt once
    # phase 2: commit max_new_tokens, one model step at a time
    # TODO: how the loop is scheduled, and how many model evaluations each committed token costs
    generated = decode_n_tokens(model, next_token, input_pos, max_new_tokens - 1, **kw)
    return seq

def main(..., compile: bool = ...):
    model = _load_model(checkpoint_path, device, precision, use_tp=...)
    # TODO: compilation of the decode step; optional weight quantization at load; optional TP sharding
    ...
```

Each rung below fills exactly one of these slots: how the loop is scheduled and the cache preallocated
(`setup_caches`, `torch.compile`), how each linear weight is represented (`int8`, `int4+GPTQ`), how
many model calls a committed token costs (speculative decoding), and how the weights are split across
GPUs (`apply_tp`). The reasoning for each rung derives the change; the full scaffold code lands in that
rung's answer.
