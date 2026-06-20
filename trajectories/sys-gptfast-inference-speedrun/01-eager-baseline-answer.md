**Problem.** Serve Llama-2-7B: decode tokens autoregressively at batch 1 on one A100-80GB, as fast as
possible. The metric is tokens/s. The workload is memory-bandwidth bound — one token needs the whole
13.5 GB of bf16 weights streamed from HBM — so the question is how close to the ~2 TB/s ceiling the
implementation runs.

**Key idea (baseline).** Run the most direct loop with no acceleration: eager PyTorch, the cache grown
by concatenation each step. This is the zero point — it isolates per-token CPU launch overhead so every
later rung's gain reads cleanly as "what this change removed."

**Why this is the floor.** At batch 1 each decoded token is a sliver of GPU work (tall-skinny GEMMs an
A100 finishes in microseconds) wrapped in dozens of eager CUDA-kernel launches per layer across 32
layers, each routed through the dispatcher, allocator, and the Python loop. The GPU drains its queue
faster than the host can refill it, so the device sits idle waiting on the CPU. The measured "bandwidth
achieved" will sit far below 2 TB/s — not because the hardware is slow, but because the loop never lets
it run flat-out.

**Change / code.** The scaffold's decode loop, run as written — `decode_one_token` calls the model and
samples; `decode_n_tokens` loops it per token. No `torch.compile`, no static cache, no quantization.

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
