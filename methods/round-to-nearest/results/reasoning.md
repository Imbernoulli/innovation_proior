A model is trained, its weights sit in floating point, and I want to store and compute each big linear layer's weights as small integers — four bits each, maybe three — so that the model is smaller and the arithmetic is cheaper, but the predictions barely move. And there are two walls I am not allowed to climb over. I cannot retrain: no gradient may touch these weights, because the whole reason I am quantizing instead of training a smaller model is that retraining a seven-billion-parameter network is the cost I am trying to dodge. And the result has to be a real fixed-precision integer encoding — each weight ends up as one of a fixed, small number of levels, and I get the approximate real value back by a rule cheap enough to run on the hardware I am targeting. So the question is narrow and concrete: given a real weight matrix and a tiny bit budget, what is the mapping to integers that hurts the layer's output the least? Let me build it from nothing.

Start with the most primitive thing I could possibly do, because I want to understand exactly where it breaks before I reach for anything clever. I have a real number `w` and I want to write it with `B` bits. `B` bits gives me `2^B` distinct codes. So I have to pick a finite set of representable real values — call them the grid — and snap each weight onto that grid. The cheapest grid to decode is one where the levels are evenly spaced: if every level is `S` apart, then decoding a code `q` is just `r = S·q`, a single multiply, and encoding is `q = w/S` rounded to an integer. That even spacing is not a small convenience; it is the whole reason this is hardware-friendly. The alternative — placing levels unevenly so they cluster where the weights are dense — would fit the distribution better, but to decode an uneven grid I need a lookup table, and a table read is slow and SIMD-hostile compared to a multiply. So I commit to a uniform grid. Now there is a subtlety about *where* the grid sits relative to zero. If I let the encode be `q = round(w/S) + Z` for some integer offset `Z`, I can slide the grid so that any chosen real value lands exactly on a code; decode becomes `r = S·(q - Z)`. That affine form `r = S·(q - Z)` is the general uniform quantizer, with `S` the step size and `Z` the integer that decodes to real zero. The offset `Z` earns its keep when the quantity is lopsided — all positive, say, like an activation after a ReLU — because then I want the grid to cover `[0, max]` rather than waste half its codes on negative values that never occur, and a nonzero `Z` shifts the grid to do exactly that. It also guarantees real zero is represented with no error, which matters anywhere exact zeros are structural.

But my quantities are *weights*, and weights of a trained linear layer are not lopsided — they sit roughly symmetric around zero, a spread of positives and negatives of comparable magnitude. So let me ask what `Z` actually buys me here, because every piece I carry forward has to justify its cost. If I keep a general `Z`, then when this quantized weight matrix is multiplied by a quantized activation inside the integer matmul, the offset does not vanish. Let me expand the product term by term and see exactly what survives. One output coordinate is `Σ_j w_j x_j ≈ Σ_j S_w(q_{w,j} - Z_w)·S_x(q_{x,j} - Z_x)`. Pulling the scales out front, the bracket is `Σ_j (q_{w,j} - Z_w)(q_{x,j} - Z_x)`. Multiplying that out gives four sums: `Σ q_w q_x`, the term I actually want; `-Z_x Σ q_w`; `-Z_w Σ q_x`; and `+ Z_w Z_x · n`. The first is the real integer dot product. The `-Z_x Σ q_w` term depends only on the (fixed) weights, so it can be folded into a per-output bias computed once. But `-Z_w Σ q_x` contains `Σ_j q_{x,j}`, a sum over the *activations*, which changes every forward pass — so a nonzero `Z_w` forces me to compute that running input sum and subtract `Z_w` times it for every output, every time. That is real per-inference integer work proportional to the matrix size, and it exists only because `Z_w ≠ 0`. And what do I get for paying it? A tighter box around a distribution that is already centered — for weights, a marginal gain. So the trade runs the other way: set `Z = 0`. The two offending sums both carry a factor of `Z_w` or `Z_x`, so with `Z_w = 0` the bracket collapses to `Σ q_w q_x` plus `-Z_x Σ q_w` (and the constant vanishes); encode collapses to `q = round(w/S)`, decode to `r = S·q`, and the weight side of the matmul is a clean integer dot product. That is the symmetric choice. I will keep the `Z` machinery in the interface as a formal slot, but for weights it is identically zero.

Now I have to set `S`, and `S` is everything — it is the single number that decides how coarse my grid is. The grid has to cover the weights, so it has to reach out to the largest-magnitude weight, `max(|w|)`. With `Z = 0` and `B` bits I want a grid symmetric about zero. A signed `B`-bit integer container ranges over `[-2^{B-1}, 2^{B-1} - 1]` — note that is *not* symmetric, there is one extra code on the negative side (for four bits, `-8` to `+7`; for three bits, `-4` to `+3`). If I insist on a symmetric effective grid, I should choose the scale from the positive half-width `qmax = 2^{B-1}-1`. Natural rounded codes then land in `[-qmax, qmax]`, and the most-negative container code `qmin = -2^{B-1}` stays unused. That feels wasteful for a second, but it gives me a clean symmetric grid, and the int8 ancestor has a hardware reason for the same convention: avoiding the `-128` endpoint keeps the worst signed product out of the accumulation path. So I will pin the positive half-width at `2^{B-1} - 1` and place the largest weight at the edge of the grid:
```
S = max(|w|) / (2^{B-1} - 1).
```
Let me check the extremes with `B = 4`, where `2^{B-1}-1 = 7`, so `qmin = -8`, `qmax = 7`. Take a small row and run the formula by hand. Say the row is `[1.0, -0.6, 0.31, -0.02]`, so `max(|w|) = 1.0` and `S = 1.0/7 ≈ 0.142857`. Then `round(1.0/S) = round(7.0) = 7` — the largest weight lands exactly on the top code, no clipping, no wasted reach. The others: `round(-0.6/S) = round(-4.2) = -4`, `round(0.31/S) = round(2.17) = 2`, `round(-0.02/S) = round(-0.14) = 0`. Every code sits inside `[-7, 7]`; nothing reached `-8`, so the most-negative container code went unused exactly as intended. Decoding back, `w_hat = S·q = [1.0, -0.571, 0.286, 0.0]`, close to the originals with the largest weight reproduced exactly. So with `S` set from the true max the rounded values naturally stay inside `[-(2^{B-1}-1), +(2^{B-1}-1)]`; the full signed clamp `q = clamp(round(w/S), -2^{B-1}, 2^{B-1}-1)` only makes the function total and matches the integer storage range. For `B = 3` the same arithmetic gives `qmin = -4`, `qmax = 3`, with an effective grid `-3..3`.

There is the question of *which* rounding. The instinct is "round to nearest," but I want to actually argue for it rather than assume it. Once the grid is fixed and I have decided to spend no further information — no data, no model of how this weight interacts with others — the only thing I am doing per weight is choosing the grid point to represent it, and the error I make on this weight is `|S·q - w|`. Among all grid points, the nearest one minimizes that error; any other choice is strictly worse on this weight by construction. So nearest-rounding is the per-weight error-minimizing decision under the constraint that I treat each weight in isolation. That last clause is important and I will hold onto it — it is the boundary where stronger methods get in — but for the bare scheme, with each weight on its own, nearest is the best snap.

The cheaper alternative would be truncation toward zero — a `floor` of `w/S` — which some integer paths reach for because it is one instruction. Let me see how much that actually costs, because if it were close I might take the simpler primitive. Take a long synthetic row, `w` of 2048 draws from `N(0, 0.1²)`, set `S` from its max at `B = 4`, and compare. Round-to-nearest gives a per-weight MSE of about `2.78e-4`; truncation gives about `1.15e-3` — roughly four times the error. So truncation is plainly worse, but the *more* revealing number is the mean error: nearest leaves a mean of `-1.2e-4`, essentially zero, while truncation leaves a mean of `-2.9e-2`. That is the dangerous quantity. Truncation always pulls toward zero, so its errors share a sign, and a coherent mean error does not average out across the many weights feeding one output — it accumulates into a systematic bias on that output, which then propagates through every downstream layer. Nearest rounding has no such drift because its errors straddle zero. The same warning applies to *ties* within nearest rounding: if a low-level right shift breaks every half-way case upward, the matrix picks up a small positive mean error of the same accumulating kind, just milder. So the rounding primitive must be nearest rounding with an unbiased tie rule, not truncation and not an upward-biased shift. In the tensor code I am mirroring, `torch.round` supplies exactly that nearest-integer snap before clamping.

Let me now look hard at the failure of this bare scheme, because that is where the real design lives. I quietly assumed "one `S` for the whole weight matrix." Picture the matrix as a stack of output rows, each row the weights producing one output coordinate. Suppose one row has weights ranging up to magnitude 1.0 and another row up to 0.01 — a 100× difference, which is not exotic; it is exactly the kind of per-channel range spread that is documented to appear in trained networks, especially after normalization layers bake very different scales into different output channels. With a single per-tensor `S`, that `S` is set by the loud row: `S = 1.0/7 ≈ 0.142857` at four bits. Now apply it to the quiet row. Let me take a concrete quiet row, `[0.01, 0.007, -0.009, 0.003, -0.002]`, and actually push it through. The largest entry maps to `round(0.01/0.142857) = round(0.07) = 0`; running the rest, every single one rounds to `0` as well, so the decoded row is `[0, 0, 0, 0, 0]`. The whole quiet row has collapsed to zero — its 16 levels of resolution spent entirely on a range it never uses. Summing `|w_hat - w|` over the five entries, the per-tensor error is `0.031`, which is just the full magnitude of the row, because nothing survived. The bare per-tensor scheme is hostage to the loudest channel. That is the wall.

The fix follows directly from the diagnosis: stop sharing one `S`. If the problem is that different rows have wildly different ranges, give each row its own scale, set from *that row's* max. Per-output-channel quantization. Now the quiet row gets `S = 0.01/7 ≈ 0.0014286`. Re-running the same five weights through the formula, the codes come out `[7, 5, -6, 2, -1]` — the row now uses the full span of the grid instead of one point — and decoding gives `[0.0100, 0.00714, -0.00857, 0.00286, -0.00143]`, tracking the originals closely. The error sum drops from `0.031` to about `0.00129`, a factor of roughly 24× on this row, just from letting the quiet channel own its scale. The cost is one extra floating-point number per row — negligible against the row's worth of weights. This is the move that turns the scheme from "fine on big robust models, broken on the rest" into something usable: adapt the step size to local magnitude. And I can push the same logic further. Even within a single output row, the input dimension is long — thousands of columns — and the weight magnitude can vary along it too. So instead of one scale per row, I can chop each row into contiguous *groups* of columns, say 128 or 64 wide, and give each group its own scale set from the max within that group. Sub-channel, or per-group, quantization. The granularity knob is now: `group_size = -1` means per-channel (one scale per row), `group_size = g > 0` means one scale per `g`-column block within each row. The finer I go, the better the local fit, at the cost of one stored scale per group — `in_features/g` scales per row instead of one. At `g = 128` over a 4096-wide input that is 32 scales per row, a tiny fraction of the row's storage, well worth it.

Let me get the per-group bookkeeping exactly right, because this is where an implementation actually goes wrong. The weight matrix is `(out_features, in_features)`. To quantize in groups of `g` columns, I reshape it to `(out_features, in_features/g, g)` so the last axis is one group. Then `max(|w|)` taken over that last axis, keeping dims, gives me a `(out_features, in_features/g, 1)` tensor of per-group maxima; divide by `2^{B-1}-1` to get a per-group scale of the same shape. To apply the scale back to the original `(out_features, in_features)` matrix, I reshape the scales to `(out_features, in_features/g)` and repeat each one `g` times along the column axis — `repeat_interleave(g)` — so every weight is paired with its own group's scale. One guard: a group of all zeros gives `max(|w|) = 0` and a scale of zero, and then `w/S` is a division by zero. So clamp the per-group max to a small floor, `1e-12`, before dividing. With a real weight group that floor never binds; it only catches the degenerate all-zero group.

Now, the constraint I have been handed says I am given calibration data — 128 sequences streamed through each layer — and an `add_batch` hook to consume it. Do I need it? Stop and think about what the data could tell me. Calibration data tells me about *activations* — the distribution of inputs each layer sees. But the quantity I am quantizing here is the *weights*, and the weights are fixed the instant training ends. The range `max(|w|)` of any weight group is sitting right there in the weight tensor; I do not need a single forward pass to know it. Weight quantization parameters are *static* — computable offline, from the weights alone, with no data whatsoever. So for this scheme the calibration stream is simply unused. I will keep the `add_batch` signature so I plug into the harness, and I will even count the samples it hands me so the interface behaves, but I will not gather statistics from it, because there is nothing about the weights it can tell me. That is not laziness; it is the defining property of this scheme — zero data, zero overhead. The entire quantization is: read the weights, compute per-group scales, round, done. (The harness also hands every per-layer quantizer a buffer named like a Hessian accumulator, because the stronger calibration-using methods need one; here I allocate it only to keep the interface uniform and never write into it.)

Let me reason out how this should behave as I drop bits, so I understand the regime I am in, and check the model against an actual quantized row rather than trusting the algebra alone. Model the rounding as additive noise: `w_hat = w + η` where `η` is the snap error, roughly uniform on `[-S/2, +S/2]` because nearest-rounding sends each weight to within half a step. A uniform variable on `[-S/2, S/2]` has variance `S²/12`, so the per-weight mean squared error should be about `S²/12`. Is that accurate? On the 2048-weight row from before at `B = 4`, `S²/12` predicts `2.86e-4` and the measured MSE was `2.78e-4` — within three percent, so the uniform-noise model is a faithful description of what rounding actually does here, and I can lean on it. Now, `S ∝ max(|w|)/(2^{B-1}-1)`. Going from four bits to three bits replaces `2^{4-1}-1 = 7` with `2^{3-1}-1 = 3`, so `S` more than doubles, and the noise power `S²/12` scales as `S²`, i.e. by `(7/3)² ≈ 5.44×`. That noise rides on every weight, and a transformer stacks dozens of these layers; the per-layer output perturbation compounds. So I should expect three-bit quantization to be dramatically harder than four-bit — not a little worse, about five and a half times the rounding-noise power per weight, multiplied across the depth — though exactly how that translates into end perplexity is something I can only confirm by running the model, since the compounding across layers is not captured by this single-row estimate.

Finer groups help on the other axis: smaller groups mean each `max(|w|)` is taken over fewer weights, so it tracks the local magnitude more tightly and the local `S` is smaller where the weights are small, shrinking the noise where it would otherwise be wasted. The scale formula says group 64 can have no larger local max than the containing group 128, and group 128 no larger max than the whole row, so the error ordering `group64 ≤ group128 ≤ per-channel` should follow — but "no larger max" only bounds the worst case, so let me actually measure it on a row whose magnitude grows across the 4096 columns. Per-channel gives MSE `4.64e-3`; group 128 gives `1.25e-3`; group 64 gives `1.02e-3`. The ordering holds, and the gap from per-channel to grouped is large precisely because that row's magnitude varies a lot along its length, which is exactly the situation grouping is meant to rescue.

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

Before I call this done, let me run the assembled code path once on a tiny weight matrix and check the round trip by hand, since a bookkeeping slip in the reshape or the broadcast would silently corrupt a real layer. Take `W = [[1.0, -0.6, 0.31, -0.02], [0.01, 0.007, -0.009, 0.003]]`, two output rows, `B = 4`, per-channel (`group_size = -1`). Row 0 has `max(|w|) = 1.0`, so `S₀ = 1/7 ≈ 0.142857`; `quantize_tensor` computes `round(W₀/S₀) = [7, -4, 2, 0]`, all inside `[-8, 7]` so the clamp is inert, and `dequantize_tensor` returns `[1.0, -0.5714, 0.2857, 0.0]` — matching the by-hand extremes check from earlier, which is reassuring that the vectorized path agrees with the scalar reasoning. Row 1 has `max(|w|) = 0.01`, so `S₁ = 0.01/7 ≈ 0.0014286`, and `round(W₁/S₁) = [7, 5, -6, 2]`, decoding to `[0.0100, 0.00714, -0.00857, 0.00286]`. Crucially row 1 did *not* inherit row 0's scale — each row's `amax(dim=1)` is independent — so the quiet row keeps its resolution; that is the per-channel mechanism doing its job on the actual code, not just in the argument. If I switch to `group_size = 2`, the reshape to `(2, 2, 2)` makes `[1.0, -0.6]` and `[0.31, -0.02]` separate groups in row 0 with scales `1/7` and `0.31/7`, and `repeat_interleave(2)` lays each scale back over its two columns; running it, the second group's `0.31` now sits at the top code instead of mid-grid, which is the finer local fit the group machinery is supposed to deliver. The round trip is consistent and the granularity knob changes the codes in the direction it should.

So the scheme that falls out is fully determined by the constraints I started with: a uniform grid forced by cheap decode, `Z = 0` forced by centered weights and the matmul cross-term, `S = max(|w|)/(2^{B-1}-1)` forced by the symmetric grid and the no-clip requirement, nearest rounding forced by per-weight error minimization, and per-channel/per-group scaling forced by the loud-channel failure. The whole thing is computed from the weights alone in one pass — the calibration stream is genuinely unused, which is the entire appeal: zero data, zero overhead. I have checked, not just claimed, that nearest beats truncation in both error and bias, that the per-channel fix recovers ~24× on a collapsed row, that the `S²/12` noise model matches measured MSE to a few percent and predicts the ~5.4× INT3 penalty, and that finer groups order as `group64 ≤ group128 ≤ per-channel`. What I have *not* settled, and cannot from these local checks, is the end-to-end perplexity, because the per-layer noise compounds across depth in a way this analysis does not capture — and the scheme's ceiling is set by my refusal to look at the data: minimizing per-weight error is only a proxy for the layer output error, and that proxy gap is exactly what a calibration-using method would close.
