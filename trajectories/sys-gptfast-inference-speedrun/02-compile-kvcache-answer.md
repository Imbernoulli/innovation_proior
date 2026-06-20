**Problem (from baseline).** Eager decode hits 25.5 tok/s and only ~340 GB/s against a ~2 TB/s A100
ceiling: at batch 1 each token is a sliver of GPU work behind hundreds of eager CUDA-kernel launches per
step, so the GPU idles waiting on the host. The binding constraint is CPU launch overhead, not yet
memory bandwidth.

**Key idea.** Delete the launch overhead with two changes that must go together. (1) **Static KV-cache:**
preallocate the per-layer key/value buffers once at full `max_seq_length` and *write into a slot* each
step instead of growing by concatenation — so tensor shapes and addresses never change. (2)
**`torch.compile(..., mode="reduce-overhead")`** of the decode step: the compiler fuses pointwise chains
into fewer kernels, and because the cache is now static the whole step is captured into a **CUDA graph**
and replayed, collapsing per-step launch cost to ~one host call. `fullgraph=True` forbids graph breaks
(which would split the graph and reopen the host round-trip).

**Why it works.** CUDA-graph replay requires identical shapes/addresses every step; the growing cache
broke that, so making the cache static is the precondition that *unlocks* the graph. Together they
convert a workload that was idle-on-the-CPU into one that runs flat-out on the GPU — turning the problem
from latency-bound into genuinely memory-bandwidth-bound, the regime every later rung then attacks by
shrinking bytes-per-token. The output distribution is unchanged (pure scheduling). First call pays a
one-time compile latency, amortized over the generation.

**Change / code.** Static `KVCache` + `setup_caches` in `model.py`; `reduce-overhead` compile of
`decode_one_token` in `generate.py`.

```python
# model.py
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out

class Transformer(nn.Module):
    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length,
                                           self.config.n_local_heads, head_dim, dtype)
        self.freqs_cis = precompute_freqs_cis(self.config.block_size,
            self.config.dim // self.config.n_head, self.config.rope_base, dtype, self.config.rope_scaling)

# generate.py — inside generate(): preallocate before the loop
with torch.device(device):
    model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

# generate.py — inside main(): capture the decode step into a CUDA graph
if compile:
    global decode_one_token, prefill
    decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    if compile_prefill:
        prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
```
