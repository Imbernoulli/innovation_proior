# Context: low-bit KV-cache quantization for LLM decoding (circa 2024)

## Research question

Autoregressive decoding in a decoder-only transformer caches the key and value tensors of
every past token so that each new token's attention does not recompute them. This key-value
(KV) cache is what makes decoding cheap in compute, but its size grows linearly with sequence
length, batch size, and depth: for an `L`-layer model with `h` heads of dimension `d`, a batch
of `b` sequences of length `n` stores `2·b·n·L·h·d` scalars. At FP16 this is gigabytes for
long contexts, and the bottleneck is not only capacity but the bandwidth of streaming those
tensors between memory and compute every step. The pressure is worst exactly where long
outputs are most needed — multi-step reasoning chains that "think" for thousands of tokens,
long-document QA and summarization, and many concurrent decode paths.

Quantizing the KV cache to a few bits is attractive because, unlike weight quantization, it
needs no retraining or fine-tuning, keeps the full context, and composes with token pruning
and weight compression. But it has a property weight quantization does not: the cache is a
*streaming* object. Key and value tensors arrive one token at a time during decode, so the
quantizer must run on-the-fly with negligible per-step overhead — it cannot do an expensive
offline calibration pass over a fixed matrix. And there is a feedback loop: a quantization
error baked into an early token's cache perturbs the attention of *every* later token, so
errors can accumulate over a long generation and drag the output off course.

The precise problem: design an on-the-fly, training-free, calibration-free quantizer for the
streaming K/V cache that drives the *model's generated output* as close as possible to the
full-precision output at a target bit-width, while staying cheap enough to run inside the
decode loop. The catch that makes this non-trivial is identifying *what* a few bits should be
spent preserving, given that the cache is consumed by attention rather than read back directly.

## Background

**The attention mechanism and the inference workflow.** With token embeddings `X ∈ R^{n×d}`,
a single attention head forms `Q = X W^Q`, `K = X W^K`, `V = X W^V` and outputs
`softmax(QKᵀ/√d + M) V` with a causal mask `M`. During the *prefill* phase the whole prompt
`X_0 ∈ R^{l_prompt×d}` is processed at once and its `K`,`V` are cached. During the *decoding*
phase only the newly generated token's `q`,`k`,`v` are computed; the new `k`,`v` are appended
to the cache, and the new token attends over all cached keys/values. Crucially, the cached
keys are never read on their own — they only ever enter the model through inner products
`q_n·k_iᵀ` that become attention scores, which a softmax turns into the convex weights of a
weighted sum of values. Modern open models use grouped-query attention (Ainslie et al. 2023;
Shazeer 2019): several query heads share one KV head, which already shrinks the cache.

**Asymmetric group-wise integer quantization.** The standard primitive maps a group of reals
`x_1,…,x_n` to `b`-bit integers via `qtz(x_i) = round((x_i − m)/Δ)`, with zero-point
`m = min_i x_i` and scale `Δ = (M − m)/(2^b − 1)`, `M = max_i x_i`; dequantization is
`deq(x̄_i) = x̄_i·Δ + m`. The integers, plus `(m, Δ)` per group, are stored. The bits-per-
element is driven down by the bit-width `b` and amortizing `(m, Δ)` over a larger group, and
up by keeping recent tokens in full precision. Which *axis* the groups run along matters a lot.

**Diagnostic findings about KV tensors.** One robust finding from KV-quantization work sets up
the streaming compression baseline:
- *Outlier structure differs between keys and values.* In the key cache, a few fixed channels
  carry persistently large magnitudes (consistent with the activation-outlier-channel findings
  of LLM.int8(), Dettmers et al. 2022, and AWQ, Lin et al. 2023); in the value cache there is
  no such fixed-channel pattern. Empirically, grouping the key cache *per channel* confines the
  large-magnitude channels' error to their own group, while values are best grouped *per token*
  — at 2-bit, per-channel keys with per-token values measurably beat the alternatives, and a
  per-token quantization of keys produces several-fold larger attention-score error than
  per-channel.

**Error feedback in least-squares quantization (the Optimal Brain lineage).** A separate,
mature line of work quantizes a layer's weight matrix `W` to minimize the *output* error
`‖WX − ŴX‖²` rather than the raw weight error. This objective is quadratic with Hessian
`H = 2XXᵀ`, and the Optimal Brain Surgeon framework (LeCun et al. 1989; Hassibi et al. 1993)
gives, for removing/rounding a single coordinate, a closed-form *compensating* update of all
remaining coordinates that cancels as much of the induced output error as possible. The cost
of doing this exactly — re-inverting the (sub-)Hessian after each coordinate — was brought down
to `O(d³)` by Frantar & Alistarh's Optimal Brain Compression (2022), which updates the inverse
in place by one Gaussian-elimination/Schur step per coordinate, and a block (group-OBS) form
handles `g` coordinates at once. The standard tools here are the block-matrix-inverse identity
and Schur complements: for `M = [[A,B],[C,D]]` with inverse `[[P,Q],[R,S]]`, one has
`A⁻¹ = P − Q S⁻¹ R`, which removes the trailing block's influence from the inverse. This whole
machinery presupposes a meaningful quadratic metric (the data Hessian `XXᵀ`) under which to
measure "error."

## Baselines

**Compression-based KV quantization (KIVI; Liu et al. 2024).** The
tuning-free, calibration-free reference. It quantizes the key cache per channel and the value
cache per token (justified by the outlier findings above), and handles the streaming axis with
a residual buffer: the most recent `R` tokens are kept in FP16; once the buffer accumulates `R`
keys they are quantized together in groups of `G`, so per-channel grouping (which spans tokens)
becomes implementable in a stream. Its objective for both keys and values is to minimize the
reconstruction error of the cached tensor itself, `‖k − deq(k^qtz)‖₂` (and likewise for `v`),
group by group. **Limitation:** it measures fidelity by how close the dequantized tensor is to
the original *in Euclidean norm*, with every coordinate weighted equally — it does not account
for how the residual will be read by attention. Two keys with the same `‖k − deq‖₂` can perturb
the next token's attention score by very different amounts depending on the direction of the
residual, and the criterion is blind to that direction.

**GEAR (Kang et al. 2024).** Tuning-free; reduces quantization error by
storing, alongside the low-bit tensor, a low-rank matrix plus a sparse matrix that capture the
residual. **Limitation:** the auxiliary matrices add storage and compute, and the target it
optimizes is still the tensor reconstruction error, not the attention output.

**KVQuant (Hooper et al. 2024).** Pushes reconstruction fidelity with
pre-RoPE quantization (quantize before rotary position embedding, applying RoPE on-the-fly
after dequant), per-channel-key/per-token-value grouping, outlier isolation, and non-uniform
levels. **Limitation:** every device of this method serves a smaller `‖k − deq‖`; the objective
remains reconstruction of the tensor, so a residual that happens to fall in a direction the
queries actually use is treated the same as one that does not.

**ZipCache (He et al. 2024).** Identifies salient tokens and applies higher precision to them,
mixed-precision across tokens. **Limitation:** it allocates bits per token by a saliency
heuristic but, within a token, still quantizes to minimize tensor reconstruction error.

**Optimal-Brain weight quantizers (OBC, Frantar & Alistarh 2022; GPTQ, Frantar et al. 2022).**
Not KV methods, but the relevant prior art on *error-compensating*
quantization: quantize coordinates one block at a time and update the rest to cancel induced
output error, using the data Hessian `H = 2XXᵀ` as the metric, with an `O(d³)` in-place inverse
downdate; GPTQ adds the observation that quantizing in a *fixed* (rather than greedy) order is
nearly as good on large matrices, which lets the inverse-Hessian information be shared across
all rows. **Limitation as applied here:** this is an *offline* procedure over a static weight
matrix with a *data* Hessian estimated from a calibration set — it presupposes a fixed metric
`XXᵀ` and a calibration pass, neither of which exists for a streaming KV cache that must be
quantized on-the-fly with no calibration data.

The shared gap across the KV baselines: they all minimize the reconstruction error of the
cached tensor, and so spend their bit budget on numerical faithfulness of `k` and `v` per se,
with no notion of which residual directions actually disturb the model's output.

## Evaluation settings

The natural yardsticks for a tuning-free KV quantizer, all pre-existing:

- **Reasoning / knowledge benchmarks** that require long chain-of-thought before a final
  answer — e.g. GSM8K (`openai/gsm8k`, exact final-answer accuracy after numeric
  normalization), MMLU-Pro (Math, Law), GPQA, BBH, IFEval — run through a harness such as
  LM-Eval. These stress the cache because the response is long.
- **Long-context benchmarks** from LongBench (Bai et al. 2023): document QA (HotpotQA, Qasper),
  retrieval (passage retrieval), summarization (QMSum, MultiNews), code completion
  (RepoBench-P, LCC), classification (TREC, TriviaQA, SAMSum), scored with each task's native
  metric (QA F1, retrieval accuracy, code edit similarity, all on a 0–100 scale). Maximum
  sequence length 4,096–8,192 tokens.
- **Needle-in-a-haystack** retrieval: place a canonical sentence inside a long essay-text
  context and score exact phrase retrieval.
- **Visible model / protocol:** an instruction-tuned mid-size model (e.g. `Qwen2.5-3B-Instruct`,
  or Llama-2-7B / Llama-3.1-8B-Instruct / Mistral-7B / DeepSeek-R1-Distill-Llama-8B in related
  KV-quantization work), deterministic greedy decode. Efficiency is reported hardware-
  independently as effective KV bits per element and the implied compression ratio against the
  FP16 footprint (`16 / effective_bits`), with quantization group size, residual length, and
  bit-width as the knobs.

## Code framework

A fixed replay harness owns the model and the deterministic decode loop. At each decode step
after prefill it snapshots the real KV tensors, hands them to a quantizer object, restores the
(de)quantized cache, and advances generation. The quantizer is the one editable piece. It is
called through a fixed interface — what stays *inside* it (grouping, ranges, zero-points,
per-layer presets, residual retention, any prefill-time bookkeeping, helper state, and the bit
accounting) is exactly the algorithm to be designed. Key and value tensors arrive with shape
`[batch, heads, seq_len, head_dim]`.

The skeleton exposes the slots a streaming KV quantizer must fill: an optional prefill-time
observation hook, the two quantize calls, and a bit-estimator. The bodies are empty.

```python
import math
import torch

FP_BITS = 16  # full-precision KV reference

class AdaptiveKVQuantizer:
    """Streaming KV-cache quantizer. The harness calls reset before each example,
    optionally lets it observe the prompt's Q/K/V during prefill, then calls
    quantize_key / quantize_value at every decode step. Must run on-the-fly:
    no offline calibration pass, no retraining."""

    def __init__(self):
        self.bits = 4
        self.group_size = 32        # quantization group for scale / zero-point
        self.residual_length = 32   # most-recent tokens kept in full precision
        # TODO: any extra state/hyperparameters the algorithm we design will need
        self.state = {}

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.state = {}

    def needs_prefill_qkv_observer(self) -> bool:
        # TODO: does the algorithm we design need to look at the prompt before decode?
        return False

    def query_observation_position(self) -> str:
        return "post_rope"

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        # TODO: optional prefill-time bookkeeping the algorithm may want
        return None

    def _residual_keep_length(self, seq_len: int) -> int:
        # keep the most-recent tokens uncompressed (streaming residual window)
        residual_length = max(0, min(seq_len, int(self.residual_length)))
        return seq_len % residual_length if residual_length else 0

    def _minmax_last_dim(self, data, group_size, bits):
        # asymmetric group-wise min-max quantize/dequantize along the last dim
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, 2 ** int(bits) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int
        q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]

    def _quantize_with_residual(self, tensor, quant_fn):
        # keep the most-recent `residual` tokens FP, quantize the rest
        work = tensor.float().clone()
        _, _, seq_len, _ = work.shape
        residual = self._residual_keep_length(seq_len)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        work[:, :, :quant_end, :] = quant_fn(work[:, :, :quant_end, :])
        avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id, key_states, cache_meta):
        # TODO: the key-quantization rule we will design
        raise NotImplementedError

    def quantize_value(self, layer_id, value_states, cache_meta):
        # TODO: the value-quantization rule we will design
        raise NotImplementedError

    def estimate_bits(self, layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float:
        residual = self._residual_keep_length(seq_len)
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
```

The two `quantize_*` bodies, the prefill hook, and whatever extra helpers and state they need are
the slots the method will fill.
