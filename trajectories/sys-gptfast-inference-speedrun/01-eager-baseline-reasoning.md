Before I change anything, I want to know what the most direct implementation actually costs, because
the cost structure is the whole game here and I'd rather measure it than guess. So I take the loop as it
stands — eager PyTorch, no compilation, the cache extended by concatenation each step — and just run it:
the prompt goes through `prefill`, then `decode_n_tokens` calls `decode_one_token` once per token,
samples, and appends, exactly as written in the scaffold.

I run the model in eager mode, which means every operation in the forward pass dispatches through
PyTorch's normal Python machinery one at a time. For one decoded token at batch 1 that is dozens of CUDA
kernel launches per layer × 32 layers — each launch routed through the dispatcher, the autograd engine's
bookkeeping (even under `no_grad` there is overhead per op), and the caching allocator — and the Python
`for` loop sitting on top driving it all. None of that is GPU work; it is CPU work that *precedes* GPU
work. And here is the thing I expect to dominate: the GPU kernels for a single token are *tiny*. A
4096×4096 matmul on one token vector is a tall-skinny GEMM that an A100 finishes in microseconds. So the
device chews through each kernel faster than the CPU can prepare and enqueue the next one. The GPU
stalls, waiting on the host. The expensive A100 spends a large fraction of every decode step idle.

I keep the cache in the naive growing form for this baseline too — each step concatenates the new key
and value onto the running tensors, so attention sees the whole history. That reallocates a
different-sized tensor every iteration, which is more allocator traffic and more copying, and it also
means the shapes the model sees change every step.

This is the configuration to start from, and it's the *right* zero: it isolates the overhead I just
described from everything else, so the first real rung's win can be read as exactly "what removing that
overhead buys." I won't quantize, won't compile, won't touch the cache layout — pure eager decode.

What I'm watching is two numbers. The headline is tokens/s. The diagnostic is the bandwidth this run
*achieves* — `model_size × tokens_per_sec`, the rate at which it is actually streaming the 13.5 GB of
weights from HBM. The A100 can do ~2 TB/s; if this baseline is achieving only a small fraction of that,
then the GPU is bandwidth-*starved* (idle, waiting on the CPU) rather than bandwidth-*bound*, and the
gap between achieved bandwidth and peak is precisely the headroom the next rung has to claim. My
expectation, from the launch-overhead argument above, is that this number comes in low — that eager
decode at batch 1 leaves the device mostly waiting — and that the very first thing worth doing is making
the GPU run flat-out instead of starved. The default fill is in the answer; it is the scaffold's decode
loop, run as written.
