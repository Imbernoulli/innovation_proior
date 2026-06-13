**Problem.** Quantize Mistral-7B's linear weights to INT4/INT3 once, no retraining, keeping WikiText-2
perplexity near the FP16 reference (4.9071). The baseline floor is to round each weight to its nearest
grid value — no calibration, no weight interactions, no output-error awareness.

**Key idea (the floor).** Symmetric per-group round-to-nearest. For each group of `group_size`
consecutive columns in a row, one scale $\Delta=\max(|\mathbf w|)/q_{\max}$ (zero point 0), then
$\widehat w=\Delta\cdot\operatorname{clamp}(\operatorname{round}(w/\Delta),q_{\min},q_{\max})$ with
$q_{\min}=-2^{b-1},\,q_{\max}=2^{b-1}-1$. Quantize every weight independently.

**Why it is the floor.** It minimizes per-*weight* error, not the layer's per-*output* error
$\lVert\mathbf{WX}-\widehat{\mathbf W}\mathbf X\rVert^2$, so it ignores which input channels actually
matter. And the scale is set by the single largest weight in each group, so one outlier inflates $\Delta$
for every typical weight in that group. It discards the calibration stream entirely (`add_batch` only
counts samples; `H` is kept solely for interface compatibility with the later Hessian-based fills).

**Scaffold edit / hyperparameters.** The default fill of lines 26–157. Symmetric grid (`symmetric=True`,
zero point 0). `num_bits` and `group_size` come from the evaluation script; the same class runs unchanged
across INT4-g128, INT3-g128, INT4-g64.

**What to watch.** Settings should order by grid resolution: INT4-g64 ≲ INT4-g128 (finer group, small
gain) ≪ INT3-g128 (8 levels, the outlier-inflation problem bites hardest, errors compound across 32
blocks). INT3 is where the floor is lowest and the next rung has the most room.

```python
# EDITABLE region of gptq/custom_ptq.py (lines 26-157) — step 1: round-to-nearest

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
    """RTN quantizer -- simple round-to-nearest, ignores calibration data."""

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
        """Collect calibration data (unused in RTN, kept for interface)."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.nsamples += inp.shape[0]

    def quantize(self):
        """RTN: symmetric per-channel (or per-group) round-to-nearest."""
        W = self.layer.weight.data.clone().float()
        scale, zero_point, qmin, qmax = find_scale_zero(
            W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
        )
        W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale, zero_point)
        return W_dq.to(self.layer.weight.dtype)

    def free(self):
        """Release calibration buffers."""
        del self.H
        self.H = None
```
