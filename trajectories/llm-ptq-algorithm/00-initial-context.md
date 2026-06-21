## Research question

Take a pretrained Mistral-7B-v0.1 (7.24B parameters, FP16) and quantize its linear weights to low-bit integers — INT4 and INT3 — **once**, with no retraining and no gradient update to the original weights, so WikiText-2 perplexity stays as close as possible to the FP16 baseline. Only the per-layer **quantization algorithm** is being designed: how a single `nn.Linear`'s weight matrix is mapped onto a fixed integer grid, optionally using a small calibration pass over that layer's inputs. Model loading, the layer-by-layer calibration driver, and perplexity evaluation are fixed. The difficulty is that INT4 has only 16 levels and INT3 only 8, so rounding residuals compound across 32 transformer blocks × 7 linear layers each.

## Prior art / Background / Baselines

- **Round-to-nearest (RTN).** Map each weight independently to the closest value on a uniform symmetric grid, with one scale per row or per group of columns. No calibration, no interaction between weights. Gap: every weight is treated as equally important and the layer's output dependence is ignored, so at 3–4 bits the per-element residual accumulates into large output error and perplexity degrades heavily at 7B scale, worst at INT3.
- **SmoothQuant, Xiao et al. 2023.** Migrate quantization difficulty from activations to weights with a per-input-channel scaling transform based on activation magnitudes, then apply RTN. Built for W8A8, where both weights and activations are quantized. Gap: in a weight-only INT3/INT4 setting the activations stay FP16, so fixed-α smoothing does not address most of the low-bit weight error and does not search the scale against actual output error.
- **GPTQ, Frantar et al. 2023.** Minimize the layer output error directly by accumulating the input second moment during calibration, then quantizing column by column and compensating each residual onto the still-free columns with the inverse Hessian. Gap: it commits to the calibration distribution through a per-column greedy regression and pays for a Cholesky/inverse plus a full column sweep per layer.
- **AWQ, Lin et al. 2024.** Identify salient input channels by activation magnitude, scale them up before quantization (and the activations down), and search the per-channel scale and a per-group clip against the layer's real output MSE, without a Hessian inverse. Gap: it still maps weights onto a uniform low-bit grid, so the extreme INT3 regime remains the hardest case.

## Fixed substrate / Code framework

A layer-by-layer quantize-then-evaluate pipeline is frozen and must not be touched. It loads Mistral-7B-v0.1, evaluates the FP16 perplexity as reference, then walks the 32 transformer blocks one at a time: it moves a block to the GPU, registers forward hooks that feed each linear sublayer's input into `LayerQuantizer.add_batch(inp)`, runs the 128 calibration sequences through to populate per-layer statistics, calls `LayerQuantizer.quantize()` on each of the block's seven linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), writes the returned quantized-dequantized weight back in place, re-runs the calibration inputs through the now-quantized block to produce inputs for the next block, and moves on. Embeddings, RMSNorm layers, and the LM head stay FP16.

The quantizer sees **one linear layer at a time**: there is no cross-layer scale absorption and no whole-block reconstruction; anything a method needs must come from that single layer's weight and its own calibration inputs. Helpers in scope: `torch`, `torch.nn`, `F`, `np`, `math`, `copy`, `os`, `time`.

## Editable interface

Only one region is editable — lines 26–157 of `gptq/custom_ptq.py`: the three quantization primitives (`quantize_tensor`, `dequantize_tensor`, `find_scale_zero`) and the `LayerQuantizer` class. Every method fills the same contract. The constructor receives the layer, `num_bits` (4 or 3), and `group_size` (128, 64, or −1 for per-channel); `add_batch(inp)` collects whatever calibration statistics the method needs from inputs of shape `(batch*seq, in_features)`; `quantize()` returns the quantized-dequantized weight, same shape and dtype as the original, respecting `num_bits` and `group_size`; `free()` releases buffers. The grid is symmetric: `qmin = -(1 << (b-1))`, `qmax = (1 << (b-1)) - 1`, and `find_scale_zero` builds a per-group scale by reshaping each row into groups of `group_size` consecutive columns and taking `max|·| / qmax` per group.

The starting point is the default scaffold: **plain RTN** — `add_batch` only counts samples (it keeps an `H` buffer for interface compatibility), and `quantize()` calls `find_scale_zero` + round + dequantize on the raw weight, ignoring calibration.

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

One model (Mistral-7B-v0.1), one seed (42), and three quantization settings probing algorithmic generalizability:

- `ptq-7b-int4` — INT4, group size 128 (standard setting).
- `ptq-7b-int3` — INT3, group size 128 (only 8 levels, hard case).
- `ptq-7b-int4-g64` — INT4, group size 64 (finer granularity).

Calibration is 128 sequences of 2048 tokens from the WikiText-2 training set. Primary metric is `wikitext2_ppl` — WikiText-2 perplexity after quantization, lower is better. Secondary metric is `degradation` — perplexity increase over the FP16 reference (4.9071), lower is better. The same `LayerQuantizer` must run unchanged across all three settings; only `num_bits` and `group_size` differ.
