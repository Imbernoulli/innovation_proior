## Research question

How fast can Llama-2-7B decode tokens at batch size 1 on a single A100-80GB? The model is frozen: ~6.7B parameters, ~13.5 GB of bf16 weights, and the output distribution must stay the model's own (any approximation has to be checked for quality). The workload is the usual autoregressive loop: prefill a short prompt, then generate one token at a time.

The hard part is not arithmetic. Each new token reads the entire weight set once while doing only ~14 GFLOP of useful compute, so arithmetic intensity is roughly one FLOP per byte. The ceiling is set by HBM bandwidth (~2 TB/s → ~150 tokens/s ideal for a perfect implementation), not by the tensor cores. The free variables are all on the decode path: scheduling, KV-cache layout, weight representation, and how many forward passes are needed per committed token.

## Prior art / Background / Baselines

**Naive eager PyTorch decode.** The loop calls `model(x, input_pos)` from Python for every token, relying on standard autograd dispatch and the framework allocator.
*Limitation:* each step launches dozens of tiny CUDA kernels; at batch size 1 the GPU finishes them faster than the CPU can queue the next ones, so much wall-clock time is idle overhead rather than useful weight traffic.

**Fused attention kernels (e.g., FlashAttention, xFormers memory-efficient attention).** They replace the materialized attention path with a single kernel that keeps the O(n) compute while reducing HBM traffic and kernel count.
*Limitation:* attention is only a fraction of the layer; the model still pays per-layer Python/framework dispatch and still streams every weight matrix for every token, so batch-1 latency stays far below the raw bandwidth ceiling.

**Production serving frameworks (e.g., FasterTransformer, DeepSpeed Inference, vLLM).** They use C++/CUDA fused kernels, optimized memory layouts, batching, or paged KV-cache management to push throughput under production conditions.
*Limitation:* those optimizations are tuned for batched or multi-user serving; at batch size 1 the benefits are smaller—extra communication, memory-management work, or batch-oriented bookkeeping can leave single-stream latency still below the bandwidth ceiling.

## Fixed substrate / Code framework

The model is a standard Llama-2-7B decoder-only transformer in bf16: token embedding, 32 `TransformerBlock`s (each an RMSNorm → multi-head attention with rotary embeddings → residual, then RMSNorm → SwiGLU feed-forward → residual), a final RMSNorm, and a bias-free output head. `dim=4096`, `n_head=32`, `head_dim=128`, `intermediate_size=11008`, vocabulary size 32000.

Decoding is the usual two-phase loop: a **prefill** pass runs the prompt once, then a **decode** loop commits one token per step with temperature 0.8 and top-k 200. The Python scaffold below is frozen; the `TODO` comments mark the editable slots.

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

# generate.py — the decode loop (the substrate shown above)
def generate(model, prompt, max_new_tokens, **kw):
    model.setup_caches(max_batch_size=batch_size, max_seq_length=...)
    next_token = prefill(model, prompt, input_pos, **kw)      # phase 1: run the prompt once
    # phase 2: commit max_new_tokens, one model step at a time
    # TODO: how the loop is scheduled, and how many model evaluations each committed token costs
    generated = decode_n_tokens(model, next_token, input_pos, max_new_tokens - 1, **kw)
    return seq

def main(...):
    model = _load_model(checkpoint_path, device, precision, ...)
    # TODO: optional changes to the decode step, weight representation, and sharding
    ...
```

## Editable interface

The solver may change only these slots inside the frozen scaffold:

1. **KV-cache storage and update.** How keys and values are kept across steps (preallocation, layout, indexing) so the cache does not force reallocations or dynamic shapes.
2. **Decode-step scheduling.** How the per-token forward pass is launched so the GPU stays busy and kernel gaps are minimized.
3. **Weight representation.** The dtype/packing of the linear-layer weights and how they are unpacked and applied during the matmul.
4. **Forward-pass budget per committed token.** How many target-model evaluations are required to emit one token.
5. **Multi-GPU sharding.** How layers are split across devices and how activations are exchanged.

Any change that would alter the output distribution must either provably preserve it or be validated on downstream tasks.

## Evaluation settings

The yardstick is wall-clock decoding throughput at **batch size 1** on an **A100-80GB power-limited to 330 W**: generate 200 new tokens from a fixed 5-token prompt after a warmup pass, and report mean tokens/s over several runs. The diagnostic is achieved bandwidth, `model_size × tokens_per_sec / 1e9`; the closer it is to the ~2 TB/s HBM peak, the better the decode path is using the hardware.

For methods that change numerics, accuracy is checked on `hellaswag` and `winogrande` via `eval.py`. Distribution-preserving or purely scheduling/parallelism changes need no extra accuracy check.
