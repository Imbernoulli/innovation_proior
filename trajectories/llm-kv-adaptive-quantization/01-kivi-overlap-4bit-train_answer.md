The cache I have to compress is a *streaming, fake-quant, tensor-level* object: at every decode step a new key row and a new value row are appended to a tensor whose length keeps growing, I am handed the whole `[batch, heads, seq_len, head_dim]` tensor, and I must return one of the identical shape after quantizing-then-dequantizing back to the model dtype. Each of those clauses kills a family of methods. A GPTQ-style per-matrix reconstruction fit is dead on arrival, because the thing I would be fitting changes every step and I am called once per layer per step; a calibration pass is impossible, because prefill itself stays lossless and only the cached copy is touched. What survives all three constraints is exactly one primitive — group-wise round-to-nearest with a per-group scale and zero-point — and the only real freedom is *where I spend bits*: along which axis I group, how big the group is, and how much of the recent cache I keep exact. The floor of the ladder is the most conservative point in that space, but it is still a real policy, not an empty fill.

I propose KIVI K4/V4: one global 4-bit asymmetric round-to-nearest, but with the grouping axis chosen *per cache* for two unrelated structural reasons, plus a streaming residual window that keeps the most recent tokens exact. The quantization primitive is the standard one — over a group, take the min and max, set the zero-point $z = \min$, the scale $s = (\max - \min)/(2^B - 1)$, round $(x - z)/s$ into $[0, 2^B - 1]$, and dequantize as $q \cdot s + z$ on the way into attention. The scale is the noise floor: dropping a bit roughly doubles $s$, so error grows geometrically in the bit cut, which is exactly why I sit at $B = 4$ — uniform low-bit RTN is essentially free at 4-bit and falls off a cliff at 2-bit, and the floor must hold quality rather than collapse. The zero-point sits at the group min rather than symmetrically around zero because KV distributions are not centered at zero (the keys especially are skewed by their outlier channels), so $[\min, \max]$ uses the whole integer codebook where a symmetric band would waste half of it.

The real content of even this conservative policy is the asymmetric axis choice, because per-token everywhere is *not* what I want. The key cache is the activation $X \cdot W_K$, and transformer activations carry persistent large-magnitude outliers in a small number of *fixed* channels — the same channel is huge for essentially every token, vertical stripes in the magnitude map. A per-token group is a slice *across channels within one token*, so it contains the outlier channel; that one huge element sets the group max, the scale blows up, and every normal element sharing the group is quantized against a step it does not need. The remedy is to group along the *channel* dimension for keys — put each channel's elements (across a window of tokens) in their own group with their own scale — so the outlier channel is confined to a group by itself and the normal channels each get a tight scale. Outlier confinement says **keys per-channel**, and since the key only matters through the attention score $\text{softmax}(qK^\top)$, per-channel grouping cuts the attention-score error several-fold versus per-token.

The value cache is where the naive reading trips, because it has no channel outliers, so by raw reconstruction error per-channel and per-token look interchangeable — and that is the trap. The value never enters attention raw; it enters as the *output* $o = \sum_j a_j v_j$, a weighted sum of value rows, and attention is sparse — a query attends strongly to only a handful of tokens. Per-token value quant gives each token its own scale, so each token's error stays inside its row; the unimportant rows are multiplied by $\approx 0$ and contribute nothing, the few important rows are each cleanly quantized. Per-channel value quant groups across tokens within a channel, so the group's scale is set partly by tokens the output never uses, and that error leaks through the shared channel scale into the values of the important tokens, where no sparsity suppresses it. Measured on the mixed output the gap is an order of magnitude, so mixer-error confinement says **values per-token**. The asymmetry — key per-channel, value per-token — comes from two completely different arguments, and even the floor must adopt it because per-token keys would simply be worse for free.

The one genuinely non-obvious piece of machinery is the residual window, forced by streaming. Per-token value quant streams trivially: a fresh value row is a complete group along the channel axis the instant it arrives. Per-channel key quant does *not* stream, because a channel's group runs along the *token* axis — I want, say, 32 tokens of a channel in one group — and tokens arrive one at a time, so I cannot form a complete channel-group until enough tokens have shown up, and the running token count is almost never a clean multiple of 32. So I split the key cache into a *grouped* part whose token count is a whole number of blocks, quantized per-channel, and a *residual* tail of the most recent tokens kept in full precision. With a block length $R$ divisible by the group size, the residual is $r = \text{seq\_len} \bmod R$ — the leftover tokens that have not yet completed a block — and it sawtooths from $0$ up to $R-1$ and resets to zero whenever `seq_len` hits a multiple of $R$. That "block-modulo" rule is exactly what makes per-channel key quant possible in a growing cache: the partial group simply stays exact until it fills. For values I keep a residual too, but with a different rule — a fixed sliding tail of the last $R$ rows, the "tail" policy — because the most recent tokens are the locally relevant ones, and hard generation (a multi-step arithmetic chain, a code completion) is exactly where a little noise on the immediately preceding K/V derails the next token. The residual does double duty: it makes per-channel key quant streamable *and* protects the local tail, at vanishing memory cost since $R$ is tiny against a long sequence.

The constants come from structure, not fiat. The group size: smaller groups give tighter ranges and lower error, but each group costs a stored $(\min, \text{scale})$ pair, so too-small groups drown the savings in metadata; $32$ is the fine-but-amortizable default. The block/residual length must be a multiple of the group size so a flushed key block is a whole number of groups, and large enough to protect a meaningful recent window without letting FP memory dominate — $128$ (four groups) for the LongBench workloads, $32$ for the shorter ones. The bit accounting is analytic and lives in `estimate_bits` at the fixed 4096-token reference span: a quantized token costs $B$ bits per element, an FP residual token costs $16$, so the average is $(\text{quant\_tokens} \cdot B + \text{residual} \cdot 16)/\text{seq\_len}$. At 4096 with $R = 128$ the key residual is $4096 \bmod 128 = 0$, so the key averages exactly 4 bits; the value keeps 128 FP rows, so $(3968 \cdot 4 + 128 \cdot 16)/4096 = 4.375$ bits; across equal-count key and value elements that is $(4 + 4.375)/2 = 4.1875$ bits per cached element, $\approx 3.82\times$ compression. Nothing here observes the prefill, so the observer hook is a no-op and the query-observation position stays at the harness default post-RoPE.

This is the floor by construction, and it is the floor for a diagnosable reason the rungs above will have to beat: it spends precision *flat*, one bit width across all 36 layers, the same 4 bits whether a layer is a fragile retrieval head or a robust concentrated one. The place I expect it to bleed is GSM8K — long greedy generations where two-dimensional error accumulation (this layer's quantization error feeds the next layer, this step's feeds the next step) can compound past the threshold that flips a single arithmetic token, and one flipped operator poisons the whole downstream chain. A uniform policy has no way to put extra bits on the layers where that accumulation bites, because it does not know which layers those are. So I expect respectable but unspectacular quality on the LongBench retrieval/QA/code tasks and NIAH at a modest $\approx 4\times$ compression, and a soft GSM8K. The floor's weakness is *uniformity*, and the question the ladder opens with is where, not whether, to move the bits.

```python
class AdaptiveKVQuantizer:
    """KIVI K4/V4 axes with streaming residual behavior."""

    def __init__(self):
        self.bits = 4
        self.group_size = 32
        self.key_residual_length = 128
        self.value_residual_length = 128

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.bits = 4

    def needs_prefill_qkv_observer(self) -> bool:
        return False

    def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _residual_keep_length(self, seq_len: int, residual_length: int, residual_policy: str) -> int:
        residual_length = max(0, min(seq_len, int(residual_length)))
        if residual_policy == "block_modulo":
            return seq_len % residual_length if residual_length else 0
        if residual_policy == "tail":
            return residual_length
        return 0

    def _minmax_last_dim(self, data: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, int(2**bits) - 1)
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

    def _quantize(self, tensor: torch.Tensor, axis: str, residual_policy: str) -> tuple[torch.Tensor, float]:
        work = tensor.float().clone()
        batch, heads, seq_len, head_dim = work.shape
        residual = self._residual_keep_length(seq_len, self.key_residual_length, residual_policy)
        quant_end = seq_len - residual
        if quant_end <= 0:
            return work.to(tensor.dtype), FP_BITS
        quant_slice = work[:, :, :quant_end, :]
        if axis == "channel":
            usable = quant_slice.shape[-2] - (quant_slice.shape[-2] % self.group_size)
            if usable > 0:
                main = quant_slice[:, :, :usable, :].transpose(2, 3)
                main = main.reshape(batch, heads, head_dim, usable // self.group_size, self.group_size)
                main = self._minmax_last_dim(main, self.bits, self.group_size)
                work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
            fp_tokens = residual + (quant_slice.shape[-2] - usable)
            avg_bits = (usable * self.bits + fp_tokens * FP_BITS) / max(seq_len, 1)
        else:
            flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
            flat = self._minmax_last_dim(flat, self.bits, self.group_size)
            work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
            avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize(key_states, "channel", "block_modulo")

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize(value_states, "token", "tail")

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        policy = "block_modulo" if kv_kind == "key" else "tail"
        residual = self._residual_keep_length(seq_len, self.key_residual_length, policy)
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
```
