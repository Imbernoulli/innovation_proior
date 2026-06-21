I have a fixed pretrained LLM — a LLaMA / Llama-2 at FP16 — and a single instruction: make it cheaper to serve without retraining. The obvious move is to store the weights in fewer bits, and before reaching for anything clever I want to know exactly how far the simplest possible scheme gets me, because whatever it leaves on the table is precisely the problem the rest of the work has to solve. A free baseline that establishes the loss at each bit-width is therefore the right first rung: it tells me where the cliff is and gives every later method a well-posed gap to close.

The method is **round-to-nearest (RTN)**, uniform integer quantization with a magnitude-derived scale. A weight matrix $W$ is a grid of floats; I pick a step size $\Delta$, divide every entry by it, round to the nearest integer, clamp into the representable range, and store the integers plus $\Delta$. At inference I multiply back by $\Delta$ to recover $\hat W \approx W$. The only real decision is the step size, and the only honest way to set it from the data alone is from the magnitudes: for an $N$-bit symmetric grid I take
$$\Delta = \frac{\max(|W|)}{2^{\,N-1}-1},$$
so the largest-magnitude weight lands at the edge of the range and everything else falls inside. There is no calibration, no Hessian, no search — $\Delta$ comes straight from the tensor's own min/max. That is the whole appeal: it costs nothing, it never overflows the grid, and it is trivially parallel. The quantize-dequantize path is just $\hat W = \Delta \cdot \mathrm{clamp}(\mathrm{round}(W/\Delta),\, -(2^{N-1}),\, 2^{N-1}-1)$.

The first real lever is *granularity*, and it matters because one $\Delta$ for an entire matrix (per-tensor) is crude — different output channels of a linear layer have very different scales, and a single $\Delta$ forces the quiet channels to round against the loud channels' range. The cheap fix that costs almost nothing is **per-channel** scaling: one $\Delta$ per output channel (per row of the weight). The finer-still option is **group-wise**: split each row into contiguous blocks of, say, 128 weights and give each block its own $\Delta$ — and, for an affine grid, its own zero-point — written g128. Group-128 stores a few extra scales but lets $\Delta$ track the local magnitude, so the rounding error per weight falls. I keep both settings in mind but never compare across them, because a g128 number is always going to look better than a per-channel number at the same nominal bit-width, and conflating them would fool me.

How low can the bit-width go before this falls apart? At 8 bits the 256-level grid makes the relative error per weight well under a percent; the model is essentially unchanged, and there is nothing to discover. The interesting regime is 4 bits and below, where the memory savings get large ($4\times$ over FP16) and a batch-1 generation workload — bound by how fast it reads weights from DRAM — actually speeds up. The reason low bits hurt is structural: because $\Delta$ is set by the *maximum* magnitude in the channel, and the weight distribution of a trained Transformer is heavy in the middle and light in the tails, the small weights — the bulk of the matrix — round against a grid far too coarse for them. At 4-bit g128 this is survivable, because each group's local max keeps $\Delta$ from being dominated by a far-away outlier. But at **3-bit per-channel** there are only eight levels for an entire channel, the step is enormous relative to the typical weight, and the accumulated rounding error compounds across a deep stack of layers into a large shift in the output distribution. I expect perplexity to degrade sharply — not gracefully — between 4 and 3 bits, and per-channel to break well before g128.

There is a second, far worse failure I name now even though I am not attacking it yet, because it sets up the entire later ladder. Everything above is *weight-only*: I quantize $W$ and leave the activations $X$ in FP16. If I also push the activations to low bits — which is what running the matmul itself on integer tensor cores would require — RTN does not merely degrade, it detonates. Transformer activations are not heavy-tailed-but-bounded the way weights are; they have a few *persistent channels* whose values are on the order of $100\times$ everything else. A single per-tensor activation $\Delta$ set by those outlier channels makes the step so coarse that every ordinary activation rounds to almost nothing, and the bulk of the signal is annihilated. So at W4A4 I expect perplexity in the thousands and an effectively destroyed model. That is not an arithmetic bug; it is the structural fact the rest of this ladder exists to defeat.

This is the bet I place on RTN as the floor. It is the right *baseline* — free, calibration-free, genuinely fine at 8 bits and tolerable at 4-bit g128 (a Llama-2-7B WikiText perplexity within a few tenths of FP16, on the order of 5.7 against an FP16 $\approx 5.5$) — but it should fall off a cliff at 3-bit *per-channel*, where I expect LLaMA-7B perplexity to climb to the mid-twenties against an FP16 of $\approx 5.68$, a more than $4\times$ blow-up no one would deploy. If that is what the numbers say, the next rung's task is sharp: the low-bit rounding errors are not random noise I must live with, they are a structured loss I should be able to *compensate* — and the lever sits in the calibration activations RTN has so far refused to use.

```python
import torch

def sym_quant(x, scale, maxq):
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)   # round-to-nearest, clamp
    return q, scale

def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

class WeightQuantizer(torch.nn.Module):
    def configure(self, bits, perchannel=False, sym=True):
        self.bits, self.perchannel, self.sym = bits, perchannel, sym
        self.maxq = torch.tensor(2 ** (bits - 1) - 1) if sym else torch.tensor(2 ** bits - 1)

    def find_params(self, x):                       # set Δ from the channel's own max
        x = x.flatten(1) if self.perchannel else x.flatten().unsqueeze(0)
        tmp = torch.zeros(x.shape[0], device=x.device)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        if self.sym:
            xmax = torch.maximum(xmin.abs(), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq                      # Δ = max(|W|) / (2^(N-1) - 1)
            self.zero  = torch.zeros_like(self.scale)
        else:                                                  # affine grid (used for g128)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero  = torch.round(-xmin / self.scale)
        shape = [-1] + [1] * (len(x.shape) - 1)
        self.scale = self.scale.reshape(shape)

    def quantize(self, x):                          # RTN: quantize then dequantize back to FP16 grid
        if self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq)
```
