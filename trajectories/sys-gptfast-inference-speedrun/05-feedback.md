Measured result — speculative decoding (draft + verifier with exact rejection-sampling acceptance), batch
1, A100-80GB (power-limited 330 W). Source: gpt-fast README "Speculative Sampling" section
(`scripts/speculate_70B_int4.sh`); the companion PyTorch blog reports the 7B-verifier combined figure.
Metric: decoding tokens/second, **higher is better**. The output distribution is identical to plain
verifier decoding by construction (no quality loss).

| pairing (verifier / draft) | tokens/second |
|---|---|
| Verifier Llama-70B (int4) / Draft Llama-7B (int4) | **48.4** |

The headline gain is the *speedup factor* per verifier pass, set by the draft's accept rate — largest when
the verifier is expensive: a 70B-int4 verifier drafted by a 7B-int4 runs at 48.4 tok/s (a 70B served far
above its un-speculated rate). On the cheap 7B verifier the absolute throughput is higher but the
amortization is smaller: the companion blog reports the full int4+GPTQ+speculation stack on 7B at ≈244.7
tok/s combined (Llama-7B verifier + TinyLlama-1B draft, ~1.3× over int4 alone). No accuracy check applies
— the rejection-sampling rule preserves the verifier's distribution exactly.
