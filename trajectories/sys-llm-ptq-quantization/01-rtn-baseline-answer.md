**Problem (the floor).** Take a fixed pretrained LLM and store its FP16 weights in fewer bits, with no
retraining and no calibration. We need a baseline that establishes exactly how much quality the
simplest scheme loses at each bit-width, so later rungs have a well-posed gap to close.

**Key idea.** Uniform integer quantization by **round-to-nearest (RTN)**: set the step size Δ straight
from the tensor's own maximum magnitude, divide, round, clamp to the N-bit grid, store integers + Δ.
Use **per-channel** granularity (one Δ per output channel) almost for free, and optionally **group-wise
g128** (one Δ per contiguous block of 128 weights) for finer resolution. No Hessian, no search, no
calibration — Δ is `max(|W|) / (2^(N-1) − 1)`.

**Why it works (and where it stops).** At 8 bits the 256-level grid makes rounding error negligible, so
RTN is essentially free. It degrades gracefully down to 4-bit g128, where the local group max keeps Δ
from being dominated by a far-away large weight. It breaks at low bits *per-channel*: with only a
handful of levels for a whole channel, Δ is set by the channel max and the small weights — the bulk of
the matrix — round against a grid far too coarse for them, and the error compounds across layers. It
breaks far worse if the *activations* are quantized too: a few persistent activation channels are ~100×
the rest, so a per-tensor activation Δ destroys the model — the failure that motivates the entire
later ladder.

**Change / code.** The magnitude-based per-channel step size and the round-to-nearest dequantize path
(the RTN quantizer, the same `WeightQuantizer` later methods reuse as their grid).

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

For g128, the same routine is applied independently to each contiguous group of 128 weights (an affine
grid with a per-group zero-point), trading a few extra stored scales for finer Δ.

**Target.** Be the floor: excellent at 8-bit, acceptable at 4-bit g128, and unusably bad at 3-bit
per-channel — leaving the next rung the task of *compensating* the rounding error using the calibration
activations RTN ignores.
