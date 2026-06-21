A model is already trained, its weights sit in fp16 or fp32, and I want to store and compute each large linear layer's weight matrix in very small signed integers — four bits per weight, maybe three — so the model is smaller, the memory traffic lighter, and the multiply-accumulate cheaper, while the layer outputs barely move. Two constraints make the problem narrow. I cannot retrain: no gradient may touch these weights, because the entire reason I quantize instead of training a smaller model is to dodge the cost of optimizing a seven-billion-parameter network. And the result has to be a genuine fixed-precision integer encoding — each weight becomes one of a small fixed set of levels, decoded back to an approximate real value by a rule cheap enough to run on the target hardware, not an arbitrary lookup table or learned codebook that is hostile to fast matmul kernels. The obvious options each fall short against this brief. Keeping the weights in floating point preserves the model exactly but leaves the footprint and bandwidth untouched, which defeats the purpose. Non-uniform or codebook schemes fit the weight distribution more tightly at a fixed bit budget, but their decoded value is no longer a shared multiply and offset, so dense matmul and portable deployment suffer. Binary, ternary, and shift-style formats are efficient in specialized kernels but impose a much stronger constraint that almost always demands training or fine-tuning around it. Quantization-aware training and post-quantization fine-tuning can recover accuracy at low precision, but they violate the zero-retraining constraint outright. So the question reduces to something concrete: given a fixed real weight matrix and a tiny bit budget, what is the data-free mapping to integers that hurts the layer's output least?

I propose Round-To-Nearest (RTN): symmetric, zero-zero-point uniform affine integer quantization of the weights, computed from the weights alone at per-channel or per-group granularity, snapping each weight to the nearest grid point with no retraining and no calibration data. Build it from nothing and every piece justifies its cost. With $B$ bits I have $2^B$ codes, so I must pick a finite grid of representable real values and snap each weight onto it. The cheapest grid to decode is one with evenly spaced levels: if levels are $S$ apart, decoding a code $q$ is a single multiply, and the general affine form that lets the grid slide relative to zero is $$r = S\,(q - Z),\qquad q = \mathrm{clamp}\!\big(\mathrm{round}(w/S) + Z,\; q_{\min},\; q_{\max}\big),$$ with $S > 0$ the step size and $Z$ the integer code that decodes to real zero. Uniform spacing is not a convenience; it is the whole reason this is hardware-friendly, because an uneven grid that clusters levels where weights are dense would fit the distribution better but force a slow, SIMD-hostile table read on every decode. The offset $Z$ earns its keep only for lopsided quantities — an all-positive activation after a ReLU, where I want the grid to cover $[0,\max]$ rather than waste half its codes on negatives that never occur. But my quantities are weights of a trained linear layer, and those sit roughly symmetric around zero. So I ask what $Z$ actually buys here. If I keep a general $Z$, then inside the integer matmul the offset does not vanish: expanding $S_w(q_w - Z_w)\,S_x(q_x - Z_x)$ and summing over the contraction index leaves a $Z_w\,\sum_j q_x$ cross-term I must compute and subtract for every output, extra integer work proportional to the matrix size. In return I get only a marginal tightening of a box around an already-centered distribution. The trade runs the wrong way, so I set $Z = 0$. Encode collapses to $q = \mathrm{round}(w/S)$, decode to $\hat w = S\,q$, and the matmul stays clean. This is the symmetric choice, and for weights it is plainly correct; I keep the $Z$ slot in the interface but for weights it is identically zero.

Setting $S$ is everything, because it alone decides how coarse the grid is. The grid must reach the largest-magnitude weight $\max(|w|)$, and I want it symmetric about zero. A signed $B$-bit container ranges over $[-2^{B-1},\,2^{B-1}-1]$ — note that is not symmetric: there is one extra code on the negative side ($-8$ to $+7$at four bits, $-4$ to $+3$ at three). To get a symmetric effective grid I set the scale from the positive half-width $q_{\max} = 2^{B-1}-1$, $$S = \frac{\max(|w|)}{2^{B-1}-1},$$ which places the biggest weight exactly at the grid edge and leaves the most-negative container code $q_{\min} = -2^{B-1}$ unused. That looks wasteful for a moment, but it gives a clean symmetric grid, and the int8 ancestor avoids the same most-negative endpoint for a hardware reason — it keeps the worst signed product out of the accumulation path. Checking the extremes confirms it: the largest weight maps to $\mathrm{round}(\max(|w|)/S) = 2^{B-1}-1$, the top code, no clipping and no wasted reach; a near-zero weight maps to a small code decoded back near zero. The full signed clamp to $[q_{\min}, q_{\max}]$ only makes the function total and matches the integer storage range. As for which rounding: round to nearest. Once the grid is fixed and I spend no further information — no data, no model of how weights interact — the only act per weight is choosing its grid point, and the error is $|S\,q - w|$; among all grid points the nearest minimizes that error by construction, so nearest is the per-weight error-minimizing decision when each weight is treated in isolation. That last clause is the boundary where stronger methods get in, but for the bare scheme nearest is the best snap. One caution comes with "nearest": tie direction. If a low-level right shift breaks half-ties consistently upward, the rounding errors across a matrix pick up a small positive mean, and a mean error is the dangerous kind — it does not average out across the many weights feeding one output but accumulates into a systematic bias on that output. So the primitive must be nearest rounding without an upward tie bias; `torch.round` supplies that before the clamp.

The real design lives in the failure of the bare scheme, which quietly assumed one $S$ for the whole matrix. Picture the matrix as a stack of output rows. Suppose one row reaches magnitude $1.0$ and another only $0.01$ — a $100\times$ spread, not exotic but exactly the per-channel range imbalance documented in trained networks, especially where normalization bakes different scales into different output channels. A single per-tensor $S$ is set by the loud row, $S = 1.0/(2^{B-1}-1)$; applied to the quiet row, the largest quiet weight maps to $\mathrm{round}(0.01/S)$, which at four bits is $\mathrm{round}(0.07) = 0$. The entire quiet row collapses to a handful of codes near zero, its sixteen levels of resolution spent on a range it never uses. The bare scheme is hostage to the loudest channel. The fix follows from the diagnosis: stop sharing one $S$. Give each output row its own scale set from that row's max — per-channel quantization — so the quiet row gets $S = 0.01/(2^{B-1}-1)$ and spreads across the full grid, at the cost of one float per row, negligible against a row of weights. The same logic pushes further: within a single row the input dimension is thousands of columns wide and magnitude varies along it too, so I chop each row into contiguous groups of $g$ columns (say 128 or 64) and give each group its own scale from its own max — per-group quantization. The granularity knob is then `group_size = -1` for per-channel and `group_size = g > 0` for one scale per $g$-column block; finer is a tighter local fit at the cost of one stored scale per group, $\text{in\_features}/g$ per row, a tiny fraction of the row's storage. The per-group bookkeeping is where implementations go wrong, so I pin it down: reshape the $(\text{out\_features},\,\text{in\_features})$ matrix to $(\text{out\_features},\,\text{in\_features}/g,\,g)$ so the last axis is one group, take $\max(|w|)$ over that axis keeping dims, divide by $2^{B-1}-1$ for a per-group scale, then reshape and `repeat_interleave(g)` to pair every weight with its group's scale. One guard: an all-zero group gives $\max(|w|)=0$ and a zero scale, so $w/S$ divides by zero; clamp the per-group max to a floor of $10^{-12}$ first, which only ever catches the degenerate group.

The calibration stream — 128 sequences through each layer, with an `add_batch` hook — I deliberately do not use, and that refusal is the defining property of the scheme rather than laziness. Calibration data describes activations, the distribution of inputs each layer sees; but the quantity I quantize is the weights, and weight ranges are static the instant training ends. The range $\max(|w|)$ of any group is sitting in the weight tensor, available offline with no forward pass. So the entire quantization is: read the weights, compute per-group scales, round, done — zero data, zero overhead, seconds of compute. I keep the `add_batch` signature and even count the samples handed to me so the interface behaves, and I allocate the Hessian-shaped buffer the harness expects only for parity with the stronger calibration-using methods, never writing into it. I can predict the regime before running anything by modeling rounding as additive noise $\eta \sim \mathrm{Unif}(-S/2, S/2)$, whose variance is $S^2/12$, so per-weight MSE is about $S^2/12$ with $S \propto \max(|w|)/(2^{B-1}-1)$. Dropping from four bits to three replaces the half-width $7$ with $3$, more than doubling $S$ and raising the noise power by $\approx (7/3)^2 \approx 5.4\times$, compounded across dozens of stacked layers — which is why INT3 is dramatically harder than INT4. Smaller groups take the max over fewer weights and track local magnitude more tightly, so in this error model group 64 $\le$ group 128 $\le$ per-channel. And I am clear-eyed about the ceiling: RTN minimizes the per-weight error $|\hat w - w|$ in isolation, a loose proxy for the layer's output error $(\hat W - W)\,x$. It ignores which input directions $x$ actually uses — information the discarded calibration data carries — and the coherent accumulation of a row's rounding errors into the output, which a method that quantized columns sequentially and pushed each leftover error into the not-yet-quantized columns could cancel. RTN is therefore the right answer given a refusal to look at the data or model weight interactions, and a deliberately simple floor that data-aware post-training methods build on.

```python
import torch


def quantize_tensor(x, scale, zero_point, qmin, qmax):
    """Encode a float tensor to integer codes: nearest rounding, clamped to grid."""
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    """Decode integer codes back to approximate floats (inverse affine map)."""
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    """Per-channel (group_size=-1) or per-group quantization parameters from weights."""
    qmin = -(1 << (num_bits - 1))          # B=4 -> -8 ; B=3 -> -4
    qmax = (1 << (num_bits - 1)) - 1       # B=4 ->  7 ; B=3 ->  3

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax                       # largest weight at grid edge
            zero_point = torch.zeros_like(scale)       # Z = 0 for centered weights
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
    """RTN: symmetric per-channel/per-group round-to-nearest, computed from the
    weights alone; calibration data is accepted but unused (weight ranges are static)."""

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        # allocated only for interface parity with calibration-using methods; unused
        self.H = torch.zeros(
            (self.in_features, self.in_features), device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.nsamples += inp.shape[0]          # count only; weights need no data

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
