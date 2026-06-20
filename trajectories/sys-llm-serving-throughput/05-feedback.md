Speculative decoding — draft-and-verify: a cheap proposer guesses k next tokens, the target model verifies
all k in one forward pass, and the longest matching prefix is accepted. More accepted tokens per big-model
forward pass amortizes the bandwidth-bound decode cost; the output is lossless (identical to plain decoding
from the target).

This is a **config- and workload-sensitive** rung: the speedup is the *mean accepted draft length per pass*,
which depends entirely on (a) how predictable the workload is and (b) which drafter is configured. Extractive
/ repetitive outputs with the n-gram drafter, or a well-matched draft model / EAGLE / Medusa head, accept
many tokens per pass and win large; high-entropy creative generation accepts ≈1 and wins little. There is no
single multiplier to quote — the effect is reproducible via the shipped throughput benchmark with a
speculative config enabled.

Measured via `vllm bench throughput` with a speculative configuration, i.e. the engine flag
`--speculative-config` (`-sc`) selecting the proposer and draft length — for example the n-gram proposer
(`{"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_max": ..., "prompt_lookup_min": ...}`), or
an EAGLE / Medusa / draft-model config. Run the same fixed model on the same fixed GPU(s) and the same prompt
set with and without `--speculative-config` and compare tokens/sec at the fixed-latency point; the gain
tracks the realized acceptance rate for that drafter and workload.

Role on the ladder: attacks the decode step's structural one-token-per-pass limit — the last big source of
idle GPU during the bandwidth-bound decode phase that the earlier rungs left untouched. It reuses the stack
wholesale: verification is the chunked-prefill multi-query attention (rung 3) over the paged cache (rung 1),
and the variable accepted-token count rides the per-iteration token-budgeted scheduler (rung 2).

(Provenance: config-sensitive; measured via `vllm bench throughput --speculative-config ...` — state the
reproduction recipe, not a fabricated number. Code: vllm/v1/spec_decode/ — ngram_proposer.py (the KMP n-gram
lookup), eagle.py, medusa.py for the neural drafters.)
