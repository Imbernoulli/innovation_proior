The task is to serve Llama-2-7B as fast as possible: feed it a short prompt and let it autoregressively decode a few hundred tokens at batch size 1 on a single A100-80GB, with tokens per second as the metric. Before I change anything, I want the most direct implementation's cost on record, because the cost structure is the whole game and I would rather measure it than guess. The workload is memory-bandwidth bound by its nature — to produce one token the model must stream all ~13.5 GB of bf16 weights from HBM once, against only ~2 FLOP of arithmetic per byte — so at the A100's ~2 TB/s peak there is a theoretical ceiling near 148 tok/s. The question this baseline answers is how close the naive loop comes to that ceiling, and the honest expectation is: nowhere near.

What I establish here is the *zero point* — the eager PyTorch decoding baseline. I take the scaffold's loop exactly as written and run it with no acceleration: no `torch.compile`, no static KV-cache, no quantization. The prompt goes through `prefill`, then `decode_n_tokens` calls `decode_one_token` once per new token, samples, and appends, with the KV-cache left in its naive growing form (each step concatenates the new key and value onto the running tensors). Every operation in the forward pass dispatches through PyTorch's normal eager machinery one at a time.

The reason this is the floor, and why it is worth measuring rather than skipping, is the shape of the work at batch 1. One decoded token is a *sliver* of GPU work — a $4096\times4096$ matmul against a single token vector is a tall-skinny GEMM an A100 finishes in microseconds — wrapped in a great deal of CPU work that *precedes* it. Each layer launches dozens of CUDA kernels (the QKV projection, rotary embedding, attention, output projection, two RMSNorms, the three feed-forward matmuls and the SiLU), and each launch is routed through the dispatcher, the autograd engine's per-op bookkeeping (present even under `no_grad`), and the caching allocator, with the Python `for` loop driving all of it. Across 32 layers that is hundreds of launches per token, each costing microseconds of host time, and the GPU kernel each one triggers is so small that the device drains its queue faster than the host can refill it. So the expensive A100 spends much of every decode step *idle*, waiting on the CPU. The defining diagnostic is therefore not just tokens/s but the *achieved bandwidth*, $\text{model\_size}\times\text{tokens\_per\_sec}$: it measures the rate at which the run actually streams weights from HBM, and if it sits far below 2 TB/s the GPU is bandwidth-*starved* (idle on the host) rather than bandwidth-*bound*. That gap is precisely the headroom the next rung claims. Choosing the pure-eager configuration is deliberate — it isolates this per-token launch overhead from everything else, so the very first real rung's gain reads cleanly as exactly "what removing that overhead buys."

```python
def multinomial_sample_one_no_sync(probs_sort): # multinomial sampling without a cuda sync
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def decode_one_token(model, x, input_pos, **sampling_kwargs):
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

# run with compile=False, no quantization, plain eager forward.
```
