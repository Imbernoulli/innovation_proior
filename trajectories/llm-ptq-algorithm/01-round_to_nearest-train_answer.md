The task is to map Mistral-7B's linear weights onto a low-bit integer grid exactly once — no retraining, no gradient touching the original weights — and keep WikiText-2 perplexity near the FP16 reference of 4.9071. Before I spend any cleverness on calibration or error feedback, I want the crudest possible fill of the fixed pipeline as a floor, so that every later rung is a measurable delta against it. The cheapest thing the editable region can do is round each weight independently to its nearest grid point. It costs nothing, depends on no calibration data, and tells me exactly how far plain rounding gets at 7B before any structure is added.

The method is symmetric per-group round-to-nearest (RTN, in the lineage of Jacob et al. 2018). With $b$ bits the symmetric integer range is $q_{\min}=-2^{b-1}$ and $q_{\max}=2^{b-1}-1$, so INT4 gives 16 levels ($-8\ldots7$) and INT3 only 8 ($-4\ldots3$). For a group of weights $\mathbf w$ I set a single step

$$\Delta=\frac{\max(|\mathbf w|)}{q_{\max}},$$

stretched so the largest-magnitude weight in the group lands at the top of the grid, fix the zero point at $0$, and write back the quantized-dequantized value

$$\widehat w=\Delta\cdot\operatorname{clamp}\!\big(\operatorname{round}(w/\Delta),\,q_{\min},\,q_{\max}\big).$$

The grouping is the only locality RTN has: with `group_size > 0`, each output row is reshaped into groups of `group_size` consecutive input columns, one $\max|\cdot|/q_{\max}$ is computed per group, and that scale is broadcast back over the group's columns; with `group_size = -1` there is a single scale per row. So the one free choice RTN makes is the per-group scale, and it makes it by the bluntest rule available — stretch to the absolute max.

I keep the grid symmetric deliberately, because this sets the convention the whole ladder inherits. An asymmetric grid would fit $(\max-\min)/(q_{\max}-q_{\min})$ with a nonzero integer zero point and would use the levels better for a distribution skewed off zero. But LLM weight groups are close to zero-mean, asymmetric costs a stored zero point per group, and — more importantly for a baseline — the symmetric grid has no zero-point degree of freedom to confound later comparisons, so scaling and error compensation can be measured cleanly against it. Zero point fixed at $0$, scale $=\max|\cdot|/q_{\max}$.

What makes this the *floor* rather than merely crude is that two failures are built in and unfixable under this rule. First, the per-element residual is irreducible and large at low bit-width: rounding $w/\Delta$ to the nearest integer leaves a residual uniform on $[-\tfrac12,\tfrac12]$ in grid units, about $0.25\,\Delta$ in absolute weight error on average. At INT4, $\Delta$ is the group max over $7$; at INT3 it is the group max over $3$, more than twice as coarse, so the same fractional residual maps to more than twice the absolute error. Second, the scale is set by the single largest weight in the group, so one outlier inflates $\Delta$ for *every* typical weight in that group — RTN spends its resolution on the extremes and starves the bulk, which is the overwhelming majority of weights and which carries most of the layer's behavior in aggregate.

The deeper indictment, and the thing every later rung attacks, is that RTN optimizes the wrong objective. What actually propagates to the next block is the layer output $\mathbf W\mathbf X$ on real inputs $\mathbf X$, so the honest target is to keep $\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert^2$ small. RTN instead minimizes per-weight error $\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$, and not even that optimally across a group — just element-wise nearest. The two diverge exactly when some input directions are excited far more than others by real text: an error on a weight multiplying a high-variance feature costs the output a lot, an equal error on a weight multiplying a near-dead feature costs almost nothing, and RTN treats them identically. The calibration apparatus the scaffold provides — the `add_batch` hook streaming 128 real sequences through each layer — exists to measure precisely this input structure, and RTN throws it away. So in the floor, `add_batch` does the minimum the interface demands: it reshapes the input to 2-D and counts samples, keeping the `H` buffer allocated only so the class signature matches the later Hessian-based fills, and nothing in `quantize()` reads it. The buffer is retained, not used, so RTN is a clean minimal instance of the shared contract that GPTQ and AWQ will fill differently.

I expect the three settings to order by how much grid resolution RTN has. INT4-g128 is the standard case and should land within a small perplexity gap of FP16; INT4-g64 should be slightly better, since a smaller group means $\Delta$ is set over fewer columns and each outlier pollutes fewer neighbors; and INT3-g128 is where RTN should fall apart, because halving the grid more than doubles the absolute residual, the outlier-inflation problem bites hardest with only 8 levels, and the per-layer errors compound across 32 blocks. INT3 is where the floor is lowest and the next rung has the most room to climb.

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
