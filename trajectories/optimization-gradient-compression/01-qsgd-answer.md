**Problem.** 100× gradient compression in a fixed SGD+momentum loop, starting from the
compressor whose convergence I can actually argue. Exactly one property keeps the standard SGD
theory valid under compression: *unbiasedness*. If `E[Q(g)] = g`, then `Q(g)` is still a
stochastic gradient and the only cost is inflated variance, which plugs straight into the
existing rate — no error feedback, no new proof.

**Key idea (unbiased stochastic quantization).** Flatten the gradient, take its `ℓ₂` norm
`‖v‖`, normalize each coordinate to `a = |v_i|/‖v‖ ∈ [0,1]`, and stochastically round it to one of
`s` evenly spaced levels: up to `(ℓ+1)/s` with probability `a·s − ℓ` (the fractional part), else
down to `ℓ/s`. Times sign times `‖v‖`, this is unbiased per coordinate by construction. The wire
payload is the signed integer levels plus the single norm float; decompress is `(‖v‖/s)·level`.

**This task's fill (departures from the generic quantizer — load-bearing).**
`s = quantum_num = 256` is *fixed* (not `√d`, not derived from `compress_ratio`), so on the
`~10^7`-param VGG the added second-moment factor is `min(d/s², √d/s) ≈ 12`, not the constant
the variance-optimal `s = √d` would give. There is **no bucketing** (norm/levels over the whole
tensor, so `d` is the full layer size). And the gradient is **clipped to norm 1.0 before
quantizing** for stability — a *biased* step with **no error feedback** to repay the
clipped-away mass. So despite the "unbiased ⇒ no EF needed" rationale, the realized compressor
is clip-then-quantize with no memory.

**Why it is the floor.** Unbiased quantization gives a clean story on the easy CIFAR-10 settings
(small models, low noise floor), so ResNet-20/56 should land near uncompressed. But on
VGG-11-BN/CIFAR-100 — largest model (most unquenched variance at fixed `s`), hardest objective,
and clip-without-feedback deleting early large-gradient energy — the failure is a *seed-dependent
collapse*, not a uniform small loss: lucky seeds train, unlucky seeds fall into a bad basin and
never recover.

**Hyperparameters.** `quantum_num = 256` (≈9 bits/level); `clip_norm = 1.0` (per-tensor pre-clip);
no residual state; `compress_ratio = 0.01` is accepted but unused by this fill.

```python
class Compressor:
    """QSGD — Quantized Stochastic Gradient Descent.

    Quantizes each gradient element to one of `s` discrete levels using
    randomized rounding. The quantization is unbiased: E[Q(g)] = g.
    Communication cost: O(n * log(s) / 32) of original, where n = numel.

    Uses s=256 quantization levels for a stable communication/variance tradeoff.

    Note: QSGD is an *unbiased* compressor, so error feedback is not needed
    and can actually hurt convergence. Unlike biased compressors (TopK,
    SignSGD) that systematically lose information, QSGD preserves the
    expected gradient value, making the vanilla SGD convergence guarantees
    applicable with only increased variance.

    Reference: Alistarh et al., "QSGD: Communication-Efficient SGD via
    Gradient Quantization and Encoding", NeurIPS 2017.
    """

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        # QSGD: s = number of quantization levels (~log2(s)+1 bits/element).
        # Var(Q(g)) ~ ||g||^2 * min(d/s^2, sqrt(d)/s). For deep nets with
        # d ~ 1e7 params, small s produces huge variance that interacts with
        # momentum SGD causing divergence on some seeds. s=256 gives ~9
        # bits/element (~3.5x compression) and keeps variance bounded.
        self.quantum_num = 256
        # Per-tensor gradient clip: prevent rare large-norm gradients from
        # amplifying quantization noise into divergence (standard QSGD
        # practice, cf. Alistarh 2017 Algorithm 1 discussion).
        self.clip_norm = 1.0

    def compress(self, tensor, name):
        shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Gradient clipping BEFORE quantization — critical for stability.
        norm = tensor_flat.norm()
        if norm == 0:
            return [tensor_flat.to(torch.int16), norm], shape
        clip_coef = self.clip_norm / (norm + 1e-6)
        if clip_coef < 1.0:
            tensor_flat = tensor_flat * clip_coef
            norm = tensor_flat.norm()
            if norm == 0:
                return [tensor_flat.to(torch.int16), norm], shape

        abs_gradient = tensor_flat.abs()

        # Quantize: level = floor(s * |g_i| / ||g||) with stochastic rounding
        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.rand_like(tensor_flat)
        is_next_level = (prob < (level_float - previous_level)).float()
        new_level = previous_level + is_next_level

        # Store sign and quantized level
        sign = tensor_flat.sign()
        tensor_compressed = (new_level * sign)
        tensor_compressed = tensor_compressed.to(torch.int16)

        return [tensor_compressed, norm], shape

    def decompress(self, compressed_tensors, ctx):
        shape = ctx
        tensor_compressed, norm = compressed_tensors

        # Dequantize: g_hat = (norm / s) * quantized_value
        decode_output = tensor_compressed.float()
        tensor_decompressed = norm / self.quantum_num * decode_output
        return tensor_decompressed.view(shape)
```
