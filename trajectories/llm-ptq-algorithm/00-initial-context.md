## Research question

Take a pretrained Mistral-7B-v0.1 (7.24B parameters, FP16) and quantize its linear weights to low-bit
integers — INT4 and INT3 — **once**, with no retraining and no gradient update to the original weights,
so that WikiText-2 perplexity stays as close as possible to the FP16 baseline. The only thing being
designed is the per-layer **quantization algorithm**: how a single `nn.Linear`'s weight matrix is mapped
onto a fixed integer grid, given (optionally) a small calibration pass over that layer's inputs.
Everything else — model loading, the layer-by-layer calibration driver, the perplexity evaluation — is
fixed. The challenge is that INT4 has only 16 levels and INT3 only 8, so the rounding residual is large,
and the errors compound across 32 transformer blocks × 7 linear layers each.

## Prior art before the first rung (low-bit PTQ lineage)

The first rung — round-to-nearest — is the crude floor that every scaling effort falls back to, and the
later rungs are the corrections people built on top of it. These are the methods the ladder reacts to.

- **Round-to-nearest (RTN), Jacob et al. 2018.** Map each weight independently to the closest value on a
  uniform symmetric grid, $\Delta=\max(|\mathbf w|)/q_{\max}$, with one scale per row (or per group of
  columns). Trivially scalable — no calibration, no interaction between weights. Gap: it treats every
  weight as equally important and ignores how the layer's *output* depends on the weights, so at 3–4
  bits the per-element residual (≈0.25 of a step on average) accumulates into large output error and
  perplexity blows up at 7B scale, worst of all at INT3.
- **SmoothQuant, Xiao et al. 2023 (arXiv:2211.10438).** Migrate quantization difficulty from activations
  to weights with a per-input-channel equivalence transform $\mathbf W\,\mathrm{diag}(\mathbf s)$,
  $\mathbf s=\mathbf s_X^{\alpha}$ from activation magnitudes, then RTN. Built for W8A8 (8-bit weights
  *and* activations); the smoothing helps activation quantization most. Gap: in a weight-only INT3/INT4
  setting the activation side is already FP16, so a fixed-$\alpha$ smooth + RTN leaves most of the
  low-bit weight error on the table — it does not search the scale against the actual output error.
- **GPTQ, Frantar et al. 2023 (arXiv:2210.17323).** Minimize the layer output error
  $\lVert\mathbf{WX}-\widehat{\mathbf W}\mathbf X\rVert^2$ directly: accumulate the input second moment
  $\mathbf H\propto\mathbf X\mathbf X^\top$ during calibration, then quantize column by column,
  compensating the residual onto the still-free columns with the inverse Hessian. Gap (relative to the
  next rung): it fits an error-feedback regression to the calibration activations, which is a per-column
  greedy that can over-commit to the calibration distribution, and it spends a Cholesky/inverse and a
  column sweep per layer.
- **AWQ, Lin et al. 2024 (arXiv:2306.00978).** Don't compensate error after the fact — *protect* the
  weights that matter before rounding. Identify salient input channels by activation magnitude, scale
  them up (and the activation down) so they get effectively finer resolution, with the per-channel scale
  $\mathbf s=\mathbf s_X^{\alpha}$ chosen by grid-searching $\alpha$ against the layer's real output MSE,
  plus a per-group clip search. No Hessian inverse. Gap it inherits: still a uniform low-bit grid, so the
  extreme INT3 regime remains the hard case.

## The fixed substrate

A layer-by-layer quantize-then-evaluate pipeline is frozen and must not be touched. It loads
Mistral-7B-v0.1 from disk, evaluates the FP16 perplexity once as the reference, then walks the 32
transformer blocks one at a time: it moves a block to the GPU, registers forward hooks that feed each
linear sublayer's input into a `LayerQuantizer.add_batch(inp)`, runs the 128 calibration sequences
through to populate the per-layer statistics, calls `LayerQuantizer.quantize()` on each of the block's
seven linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`),
writes the returned quantized-dequantized weight back in place, re-runs the calibration inputs through
the now-quantized block to produce the inputs for the next block, and moves on. Embeddings, the RMSNorm
layers, and the LM head are left in FP16. The quantizer sees **one linear layer at a time** — there is
no cross-layer scale absorption and no whole-block reconstruction context; whatever a method needs must
be computed from that single layer's weight and its own calibration inputs. Helpers available in scope:
`torch`, `torch.nn`, `F`, `np`, `math`, `copy`, `os`, `time`.

## The editable interface

Exactly one region is editable — lines 26–157 of `gptq/custom_ptq.py`: the three quantization
primitives (`quantize_tensor`, `dequantize_tensor`, `find_scale_zero`) and the `LayerQuantizer` class.
Every method on the ladder is a fill of this same contract. The constructor receives the layer and the
evaluation-set `num_bits` (4 or 3) and `group_size` (128, 64, or −1 for per-channel); `add_batch(inp)`
collects whatever calibration statistics the method needs from inputs of shape `(batch*seq, in_features)`;
`quantize()` returns the quantized-dequantized weight, same shape and dtype as the original, respecting
`num_bits` and `group_size`; `free()` releases the buffers. The grid is symmetric:
$q_{\min}=-2^{b-1}$, $q_{\max}=2^{b-1}-1$, and `find_scale_zero` builds a per-group scale by reshaping
each row into groups of `group_size` consecutive columns and taking $\max|\cdot|/q_{\max}$ per group.

The starting point is the scaffold default: **plain RTN** — `add_batch` only counts samples (it keeps an
`H` buffer for interface compatibility), and `quantize()` calls `find_scale_zero` + round + dequantize
on the raw weight, ignoring calibration entirely. Each later method replaces exactly this region.

```python
# EDITABLE region of gptq/custom_ptq.py (lines 26-157) — default fill: round-to-nearest

def quantize_tensor(x, scale, zero_point, qmin, qmax):
    """Quantize a float tensor to integers given scale and zero point."""
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    """Dequantize integer tensor back to float."""
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    """Compute per-channel (or per-group) quantization parameters."""
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)
        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        if symmetric:
            w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = weight.amin(dim=1, keepdim=True)
            w_max = weight.amax(dim=1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

    return scale, zero_point, qmin, qmax


class LayerQuantizer:
    """Quantizes a single nn.Linear layer's weights. Default: round-to-nearest."""

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.float()
        self.H += inp.T @ inp
        self.nsamples += n

    def quantize(self):
        W = self.layer.weight.data.clone().float()
        scale, zero_point, qmin, qmax = find_scale_zero(
            W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
        )
        W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale, zero_point)
        return W_dq.to(self.layer.weight.dtype)

    def free(self):
        del self.H
        self.H = None
```

## Evaluation settings

One model (Mistral-7B-v0.1), one seed (42), and three quantization settings probing generalizability of
the *algorithm*: `ptq-7b-int4` (INT4, group size 128 — the standard setting), `ptq-7b-int3` (INT3, group
size 128 — only 8 levels, the hard case), and `ptq-7b-int4-g64` (INT4, group size 64 — finer
granularity). Calibration is 128 sequences of 2048 tokens from the WikiText-2 training set. Primary
metric `wikitext2_ppl` — WikiText-2 perplexity after quantization, lower is better; secondary
`degradation` — the perplexity increase over the FP16 reference (4.9071), lower is better. The same
`LayerQuantizer` must run unchanged across all three settings (only `num_bits`/`group_size` differ).
