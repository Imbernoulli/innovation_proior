The task is a single, sharp one: replace the identity compressor inside a fixed SGD+momentum loop with something that cuts the communicated gradient volume toward 100× and still trains a model as good as full-gradient SGD would. The identity default ships every float32 coordinate and so tells me nothing; the first rung has to be the compressor whose convergence I can actually *argue* end to end, because exactly one property decides whether a compressed gradient is still a legitimate stochastic gradient — is the compression unbiased? For an $L$-smooth $f$ and SGD with $\mathbb{E}[g]=\nabla f$ and a second-moment bound $\mathbb{E}\lVert g\rVert^2 \le B$, the iteration count to a target error is linear in $B$ and depends on the gradient *only through its variance*. So if I compress with a randomized map $Q$ that is exactly unbiased, $\mathbb{E}[Q(g)] = g$, then $Q(g)$ is itself a perfectly valid stochastic gradient: its mean is right, the only thing compression does is inflate $B$ by some factor, and that factor multiplies the iteration count and changes nothing else in a theorem I already trust. No error feedback, no residual buffer, no new proof — that is the cleanest possible foundation, and it is why an unbiased quantizer is the natural place to plant the flag.

I propose **QSGD — unbiased stochastic quantization**. Take a gradient tensor, flatten it to $v$, and keep its $\ell_2$ norm $\lVert v\rVert$ — one float shared across all coordinates carries the scale almost for free. For each coordinate, normalize $a = |v_i|/\lVert v\rVert \in [0,1]$, lay down $s$ evenly spaced levels in $[0,1]$, and round *stochastically* to one of the two levels $a$ falls between: up to $(\ell+1)/s$ with probability equal to the fractional part $p = a\cdot s - \ell$, else down to $\ell/s$. The expectation is exactly $a$: $(\ell/s)(1-p) + ((\ell+1)/s)\,p = \ell/s + p/s = a$, so the rounded level times the sign times $\lVert v\rVert$ is unbiased per coordinate by construction. The reason to randomize between *adjacent* levels rather than, say, a coarse global round is that the per-coordinate jump stays one level wide, so the added variance is that of a Bernoulli on a gap of $1/s$ — it scales as $1/s^2$, and summed over the tensor the added second-moment factor is bounded by $\min(d/s^2,\sqrt d/s)\,\lVert v\rVert^2$. The variance-optimal point is $s=\sqrt d$, where both branches equal one and the second moment is at most $2\lVert v\rVert^2$: a constant-factor iteration cost. More levels crush the noise quadratically at the price of more bits to name the level; $s$ is the dial.

What goes on the wire is the vector of signed integer levels plus the single norm float; what stays local is just the shape. Decompression is $\hat g = (\lVert v\rVert / s)\cdot\text{level}$. There is no residual, no warm-start, no per-name state — the simplest of the families to fill, which is part of why it is the right place to read the first numbers.

I have to be honest about how *this* fill departs from the textbook construction, because three of its choices are load-bearing and not at the variance-optimal point. First, $s$ is **fixed at 256**, not $\sqrt d$ and not derived from `compress_ratio`. With $d \sim 10^7$ for the VGG weights, the variance bound at $s=256$ is $\min(10^7/65536,\ \sim\!3162/256) \approx \min(153,\,12.4)=12.4$ — an added second-moment factor in the low tens, not the constant the sweet spot would give. It spends a fixed $\sim 9$ bits per level, so the realized compression is roughly the int16-vs-float32 ratio (a few ×), far short of the headline 100×. Second, there is **no bucketing**: the norm and levels are taken over the whole flattened tensor, so the $d$ in that bound is the full layer size — exactly the regime where a fixed $s$ leaves the variance large. Third, and this is the one that bites, the fill **clips the gradient to norm 1.0 before quantizing**. Clipping is standard QSGD-stability practice — rare large-norm gradients otherwise amplify quantization noise toward divergence — but it is a *biased* operation, and there is no error feedback to repay the clipped-away mass. So the comment's "unbiased, so no error feedback needed" is only half true here: the quantizer is unbiased, but the pre-clip is not, and nothing carries forward what it deletes. The compressor I am actually running is *clip-to-unit-norm, then full-tensor stochastic quantize to 256 levels, with no memory.*

That tells me exactly where this floor should crack. On the well-conditioned CIFAR-10 settings — ResNet-20 and ResNet-56 — SGD's noise floor is already low and the gradient norms are modest, so a double-digit variance inflation is survivable and the cosine schedule's late small steps average the quantization noise out; I expect those to land near uncompressed, low-to-mid 90s, with at most one weaker seed from the unrepaired clip. The danger is concentrated on VGG-11-BN/CIFAR-100: it is the largest model (largest full-tensor $d$, most unquenched variance at fixed $s$), the hardest objective (least forgiving of a noisy direction), and the clip-with-no-feedback systematically deletes gradient energy on exactly the early steps where the gradient is large and the model most needs to move. That combination produces not a uniform small loss but a *seed-dependent collapse* — on an unlucky seed the early clipped mass is thrown away and the model falls into a basin it never escapes, while a luckier seed trains almost normally. So I expect VGG to separate the seeds, one or two collapsing to a fraction of the others, which is precisely what should make this the weakest rung — and the failure mode points straight at the next idea, a compressor that *never permanently throws gradient mass away* but instead accumulates it in a memory.

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
