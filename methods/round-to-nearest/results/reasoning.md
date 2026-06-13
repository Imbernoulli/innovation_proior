A model is trained, its weights sit in floating point, and I want to store and compute each big linear layer's weights as small integers — four bits each, maybe three — so that the model is smaller and the arithmetic is cheaper, but the predictions barely move. And there are two walls I am not allowed to climb over. I cannot retrain: no gradient may touch these weights, because the whole reason I am quantizing instead of training a smaller model is that retraining a seven-billion-parameter network is the cost I am trying to dodge. And the result has to be a real fixed-precision integer encoding — each weight ends up as one of a fixed, small number of levels, and I get the approximate real value back by a rule cheap enough to run on the hardware I am targeting. So the question is narrow and concrete: given a real weight matrix and a tiny bit budget, what is the mapping to integers that hurts the layer's output the least? Let me build it from nothing.

Start with the most primitive thing I could possibly do, because I want to understand exactly where it breaks before I reach for anything clever. I have a real number `w` and I want to write it with `B` bits. `B` bits gives me `2^B` distinct codes. So I have to pick a finite set of representable real values — call them the grid — and snap each weight onto that grid. The cheapest grid to decode is one where the levels are evenly spaced: if every level is `S` apart, then decoding a code `q` is just `r = S·q`, a single multiply, and encoding is `q = w/S` rounded to an integer. That even spacing is not a small convenience; it is the whole reason this is hardware-friendly. The alternative — placing levels unevenly so they cluster where the weights are dense — would fit the distribution better, but to decode an uneven grid I need a lookup table, and a table read is slow and SIMD-hostile compared to a multiply. So I commit to a uniform grid. Now there is a subtlety about *where* the grid sits relative to zero. If I let the encode be `q = round(w/S) + Z` for some integer offset `Z`, I can slide the grid so that any chosen real value lands exactly on a code; decode becomes `r = S·(q - Z)`. That affine form `r = S·(q - Z)` is the general uniform quantizer, with `S` the step size and `Z` the integer that decodes to real zero. The offset `Z` earns its keep when the quantity is lopsided — all positive, say, like an activation after a ReLU — because then I want the grid to cover `[0, max]` rather than waste half its codes on negative values that never occur, and a nonzero `Z` shifts the grid to do exactly that. It also guarantees real zero is represented with no error, which matters anywhere exact zeros are structural.

But my quantities are *weights*, and weights of a trained linear layer are not lopsided — they sit roughly symmetric around zero, a spread of positives and negatives of comparable magnitude. So let me ask what `Z` actually buys me here, because every piece I carry forward has to justify its cost. If I keep a general `Z`, then when this quantized weight matrix is multiplied by a quantized activation inside the integer matmul, the offset does not vanish; expanding `S_w(q_w - Z_w)·S_x(q_x - Z_x)` and summing over the contraction index, the `Z_w` term becomes `Z_w · Σ_j q_x`, a cross-term I have to compute and subtract for every output — extra integer work proportional to the matrix size that I would not otherwise pay. And what do I get for that cost? A tighter box around a distribution that is already centered, which for weights is a marginal gain. So the trade is lopsided the other way: drop `Z`. Set `Z = 0`. Then encode collapses to `q = round(w/S)` and decode to `r = S·q` — no offset anywhere, the matmul has no cross-term, the kernel is clean. This is the symmetric choice, and for weights it is clearly the right one. I will keep the `Z` machinery in the interface as a formal slot, but for weights it is identically zero.

Now I have to set `S`, and `S` is everything — it is the single number that decides how coarse my grid is. The grid has to cover the weights, so it has to reach out to the largest-magnitude weight, `max(|w|)`. With `Z = 0` and `B` bits I want a grid symmetric about zero. A signed `B`-bit integer container ranges over `[-2^{B-1}, 2^{B-1} - 1]` — note that is *not* symmetric, there is one extra code on the negative side (for four bits, `-8` to `+7`; for three bits, `-4` to `+3`). If I insist on a symmetric effective grid, I should choose the scale from the positive half-width `qmax = 2^{B-1}-1`. Natural rounded codes then land in `[-qmax, qmax]`, and the most-negative container code `qmin = -2^{B-1}` stays unused. That feels wasteful for a second, but it gives me a clean symmetric grid, and the int8 ancestor has a hardware reason for the same convention: avoiding the `-128` endpoint keeps the worst signed product out of the accumulation path. So I will pin the positive half-width at `2^{B-1} - 1` and place the largest weight at the edge of the grid:
```
S = max(|w|) / (2^{B-1} - 1).
```
Let me sanity-check the extremes. The weight with the biggest magnitude maps to `round(max(|w|)/S) = round(2^{B-1}-1) = 2^{B-1}-1`, the top code — good, it sits right at the grid edge, no clipping, no wasted reach. A weight near zero maps to a small code, decoded back near zero. The container clamp is `q = clamp(round(w/S), -2^{B-1}, 2^{B-1}-1)` and decode is `w_hat = S·q`. With `S` set from the true max, the rounded values naturally stay inside `[-(2^{B-1}-1), +(2^{B-1}-1)]`; the full signed clamp only makes the function total and matches the integer storage range. For `B = 4`, that means `qmin = -8`, `qmax = 7`, but the effective symmetric grid is `-7..7`. For `B = 3`, it is `qmin = -4`, `qmax = 3`, with an effective grid `-3..3`.

There is the question of *which* rounding. Round to nearest. I should make sure I can defend that and not just assume it. Once the grid is fixed and I have decided to spend no further information — no data, no model of how this weight interacts with others — the only thing I am doing per weight is choosing the grid point to represent it, and the error I make on this weight is `|S·q - w|`. Among all grid points, the nearest one minimizes that error; any other choice is strictly worse on this weight by construction. So nearest-rounding is the per-weight error-minimizing decision under the constraint that I treat each weight in isolation. That last clause is important and I will hold onto it — it is the boundary where stronger methods get in — but for the bare scheme, with each weight on its own, nearest is the best snap. One caution the moment I say "nearest": ties and the *direction* of rounding. If a low-level right shift breaks half ties consistently upward, then across a whole matrix the rounding errors pick up a small positive mean, and a mean error is the dangerous kind — it does not average out across the many weights feeding one output, it accumulates into a systematic bias on that output. So the rounding primitive must be nearest rounding without an upward tie bias. In the tensor code I am mirroring, `torch.round` supplies the nearest-integer snap before clamping; the important part here is not to use truncation or an upward-biased right shift.

Let me now look hard at the failure of this bare scheme, because that is where the real design lives. I quietly assumed "one `S` for the whole weight matrix." Picture the matrix as a stack of output rows, each row the weights producing one output coordinate. Suppose one row has weights ranging up to magnitude 1.0 and another row up to 0.01 — a 100× difference, which is not exotic; it is exactly the kind of per-channel range spread that is documented to appear in trained networks, especially after normalization layers bake very different scales into different output channels. With a single per-tensor `S`, that `S` is set by the loud row: `S = 1.0/(2^{B-1}-1)`. Now apply it to the quiet row, whose weights top out at 0.01. The largest weight in the quiet row maps to `round(0.01/S) = round(0.01·(2^{B-1}-1)/1.0)`, which for four bits is `round(0.07) = 0`. The *entire* quiet row collapses to a handful of codes near zero — most of it to exactly zero. Its 16 levels of resolution have been spent on a range it never uses; its relative error is enormous. The bare per-tensor scheme is hostage to the loudest channel. That is the wall.

The fix follows directly from the diagnosis: stop sharing one `S`. If the problem is that different rows have wildly different ranges, give each row its own scale, set from *that row's* max. Per-output-channel quantization. Now the quiet row gets `S = 0.01/(2^{B-1}-1)`, its weights spread across the full grid, and its relative error is back in line with the loud row's. The cost is one extra floating-point number per row — negligible against the row's worth of weights. This is the move that turns the scheme from "fine on big robust models, broken on the rest" into something usable: adapt the step size to local magnitude. And I can push the same logic further. Even within a single output row, the input dimension is long — thousands of columns — and the weight magnitude can vary along it too. So instead of one scale per row, I can chop each row into contiguous *groups* of columns, say 128 or 64 wide, and give each group its own scale set from the max within that group. Sub-channel, or per-group, quantization. The granularity knob is now: `group_size = -1` means per-channel (one scale per row), `group_size = g > 0` means one scale per `g`-column block within each row. The finer I go, the better the local fit, at the cost of one stored scale per group — `in_features/g` scales per row instead of one. At `g = 128` over a 4096-wide input that is 32 scales per row, a tiny fraction of the row's storage, well worth it.

Let me get the per-group bookkeeping exactly right, because this is where an implementation actually goes wrong. The weight matrix is `(out_features, in_features)`. To quantize in groups of `g` columns, I reshape it to `(out_features, in_features/g, g)` so the last axis is one group. Then `max(|w|)` taken over that last axis, keeping dims, gives me a `(out_features, in_features/g, 1)` tensor of per-group maxima; divide by `2^{B-1}-1` to get a per-group scale of the same shape. To apply the scale back to the original `(out_features, in_features)` matrix, I reshape the scales to `(out_features, in_features/g)` and repeat each one `g` times along the column axis — `repeat_interleave(g)` — so every weight is paired with its own group's scale. One guard: a group of all zeros gives `max(|w|) = 0` and a scale of zero, and then `w/S` is a division by zero. So clamp the per-group max to a small floor, `1e-12`, before dividing. With a real weight group that floor never binds; it only catches the degenerate all-zero group.

Now, the constraint I have been handed says I am given calibration data — 128 sequences streamed through each layer — and an `add_batch` hook to consume it. Do I need it? Stop and think about what the data could tell me. Calibration data tells me about *activations* — the distribution of inputs each layer sees. But the quantity I am quantizing here is the *weights*, and the weights are fixed the instant training ends. The range `max(|w|)` of any weight group is sitting right there in the weight tensor; I do not need a single forward pass to know it. Weight quantization parameters are *static* — computable offline, from the weights alone, with no data whatsoever. So for this scheme the calibration stream is simply unused. I will keep the `add_batch` signature so I plug into the harness, and I will even count the samples it hands me so the interface behaves, but I will not gather statistics from it, because there is nothing about the weights it can tell me. That is not laziness; it is the defining property of this scheme — zero data, zero overhead. The entire quantization is: read the weights, compute per-group scales, round, done. (The harness also hands every per-layer quantizer a buffer named like a Hessian accumulator, because the stronger calibration-using methods need one; here I allocate it only to keep the interface uniform and never write into it.)

Let me reason out, before I run anything, how this should behave as I drop bits, so I understand the regime I am in. Model the rounding as additive noise: `w_hat = w + η` where `η` is the snap error, roughly uniform on `[-S/2, +S/2]` because nearest-rounding sends each weight to within half a step. A uniform variable on `[-S/2, S/2]` has variance `S²/12`. So the per-weight mean squared error is about `S²/12`, and `S ∝ max(|w|)/(2^{B-1}-1)`. Going from four bits to three bits replaces `2^{4-1}-1 = 7` with `2^{3-1}-1 = 3`, so `S` more than doubles, and the noise power `S²/12` goes up by roughly `(7/3)² ≈ 5.4×`. That noise rides on every weight, and a transformer stacks dozens of these layers; the per-layer output perturbation compounds. So I should *expect*, without measuring anything, that three-bit quantization is dramatically harder than four-bit — not a little worse, several times the rounding-noise power per weight, multiplied across the depth. Finer groups help on the other axis: smaller groups mean each `max(|w|)` is taken over fewer weights, so it tracks the local magnitude more tightly and the local `S` is smaller where the weights are small, shrinking the noise where it would otherwise be wasted. In this local error model, group 64 has no larger max than the containing group 128, and group 128 has no larger local scale than the whole row; that is the order I expect from the scale formula itself.

The limitation of everything I have built is the boundary every stronger method exploits. I have been minimizing the *per-weight* error `|w_hat - w|`, weight by weight, in isolation. But what the model actually cares about is the *output* error of the layer: `(W_hat - W)·x`, the layer's response to its real inputs. Minimizing each weight's individual error is only a proxy for minimizing the output error, and a loose one. Two things it misses. First, the rounding errors across a row are not independent of the input — if I had known which input directions `x` actually points along, I could have traded a little more error on a rarely-used weight for less error on a heavily-used one, and come out ahead on the output. That is precisely the information the calibration data carries and that this scheme throws away. Second, the per-weight errors in a row add up coherently into the output; a method that quantized columns in sequence and pushed each column's leftover error into the not-yet-quantized columns could cancel that accumulation, where I just let it stand. So the bare scheme leaves real headroom on the table: it is the right answer *given that I refuse to look at the data or model the interactions*, and it becomes insufficient the moment I am willing to spend a calibration pass. That gap is deliberate here — the whole appeal of this scheme is that it costs nothing, runs in seconds, and needs no data — but I want to be clear-eyed that it is the floor, not the ceiling.

So let me write it as the code I would actually ship, filling the empty quantizer slot. Three pieces: a primitive that encodes a float tensor to integer codes given the parameters and range; its inverse that decodes; a routine that computes the per-channel-or-per-group `(scale, zero_point)` from the weights; and the layer quantizer that wires them together and ignores the calibration stream.

```python
import torch


def quantize_tensor(x, scale, zero_point, qmin, qmax):
    # encode: real -> integer code. nearest rounding (unbiased ties), clamped to grid.
    # with symmetric weights zero_point = 0, so this is round(x / scale) clamped.
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    # decode: integer code -> approximate real. inverse of the affine encode.
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    # B-bit signed grid: codes in [qmin, qmax]; symmetric grid uses [-qmax, qmax],
    # the most-negative code qmin is left unused (clean symmetric grid + SIMD slack).
    qmin = -(1 << (num_bits - 1))            # e.g. B=4 -> -8 ; B=3 -> -4
    qmax = (1 << (num_bits - 1)) - 1         # e.g. B=4 ->  7 ; B=3 ->  3

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        # one group = one contiguous block of `group_size` input columns
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            # scale set by the group's largest magnitude; floor guards all-zero groups
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax             # place the biggest weight at the grid edge
            zero_point = torch.zeros_like(scale)   # weights are centered -> Z = 0
        else:
            # asymmetric fallback (kept for completeness; not used for weights)
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)
        # broadcast each group's params back over its `group_size` columns
        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        # per-channel: one scale per output row, from that row's max
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
    """Round-to-nearest: symmetric per-channel/per-group integer quantization
    of the weights, computed from the weights alone -- the calibration data is
    not used (weight ranges are static)."""

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
        # the data describes activations; weight ranges don't need it. count and drop.
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.nsamples += inp.shape[0]

    def quantize(self):
        W = self.layer.weight.data.clone().float()
        scale, zero_point, qmin, qmax = find_scale_zero(
            W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
        )
        W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)   # round to grid
        W_dq = dequantize_tensor(W_q, scale, zero_point)          # decode back to float
        return W_dq.to(self.layer.weight.dtype)

    def free(self):
        del self.H
        self.H = None
```

Let me trace the causal chain once, end to end, to be sure it hangs together. I need integers, cheap to decode, so I take a uniform grid, which forces the affine form `r = S(q - Z)`. The weights are centered, so a nonzero `Z` only buys a marginal range tightening while costing a cross-term in the matmul, so `Z = 0` — symmetric. With `Z = 0` and a symmetric `B`-bit grid I set the step from the largest magnitude, `S = max(|w|)/(2^{B-1}-1)`, placing the biggest weight at the grid edge and leaving the most-negative code unused for a clean grid and SIMD slack. Each weight snaps to its nearest grid point, which is the per-weight error-minimizing choice when I treat weights in isolation, using nearest rounding rather than truncation or an upward-biased half-tie rule. A single per-tensor scale is hostage to the loudest channel — a 100× range spread collapses the quiet channels to near-zero codes — so I give each output row its own scale, and then each contiguous group of columns its own scale, adapting the step to local magnitude at the cost of one float per group. Because weight ranges are fixed at the end of training, the whole thing is computed from the weights with no calibration data, in one pass, which is the entire point: zero overhead. And I know going in that dropping from four to three bits multiplies the per-weight rounding-noise power by roughly `(7/3)²`, that finer groups shrink it, and that the scheme's ceiling is set by my refusal to look at the data — minimizing per-weight error is only a proxy for the layer output error, and that proxy gap is exactly what a calibration-using method would close.
