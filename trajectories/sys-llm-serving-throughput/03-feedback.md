Chunked prefill — split long prefills into token-budget chunks and co-batch them with running decodes, so a
long prompt no longer monopolizes a forward pass and the prefill's compute fills the tensor cores that decode
steps leave idle.

This is a **config-sensitive** rung: the gain depends on the model, the GPU, and the workload's prompt/output
length mix, and it is tuned by the chunk size (the prefill token budget per step) — so there is no single
honest multiplier to quote. The effect is reproducible via the shipped throughput benchmark with the enabling
flag.

Measured via `vllm bench throughput` (the in-repo offline throughput harness; the legacy entry point
`benchmarks/benchmark_throughput.py` now redirects to it) with chunked prefill enabled, i.e. the engine flag
`--enable-chunked-prefill` (scheduler config `enable_chunked_prefill`, with `max_num_batched_tokens` setting
the per-step token budget / chunk size). Run the same fixed model on the same fixed GPU(s) and the same
prompt set twice — once with `--enable-chunked-prefill` and once without — and compare tokens/sec at the
fixed-latency operating point; chunked prefill raises sustained GPU utilization by co-running prefill chunks
with decodes and removes the latency spikes from monopolizing prefill steps, with the magnitude set by the
chunk size and the workload.

Role on the ladder: stops prefill from being a throughput-and-latency tax on decode. It keeps the
per-iteration occupancy high (no long-prefill step pushing decodes out) and improves each step's compute
quality (decode-idle tensor cores get filled by prefill chunks). It sits directly on the paged cache (chunks
attend over the same block-table-addressed KV) and on the token-budgeted scheduler (it is a clamp on the
per-step prefill advance).

(Provenance: config-sensitive; measured via `vllm bench throughput --enable-chunked-prefill` — state the
reproduction recipe, not a fabricated number. Code: vllm/v1/attention/ops/chunked_prefill_paged_decode.py;
the scheduler's long_prefill_token_threshold clamp in vllm/v1/core/sched/scheduler.py.)
