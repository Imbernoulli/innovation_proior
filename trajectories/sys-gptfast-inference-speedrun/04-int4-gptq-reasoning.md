155.58 tokens/s at 1069 GB/s. The int8 bet paid off exactly the way the bandwidth story predicted: I cut
the weight bytes in half and throughput went up ~1.5×, with the achieved GB/s dropping rather than rising
— fewer bytes per token, same wall. So the lever is proven, and the obvious next pull is to keep going:
if 8 bits per weight bought that, what about 4? int4 halves the weight footprint *again* — 13.5 GB of
bf16 down toward ~3.5 GB — and on a bandwidth-bound workload that's the path to roughly another 1.3-1.5×.
The arithmetic budget is nowhere near binding, so the only question, again, is quality: can I drop the
weights to 4 bits without the model getting observably worse?

And here the same trick that made int8 safe stops being enough. With int8 I had per-row scales and ~256
levels, and the rounding noise was a rounding error. At 4 bits I have *16* levels. A single scale per row
now has to cover the whole row's dynamic range with sixteen rungs, and 4096 weights in a row do not all
live in the same small band — there are local clusters, occasional large entries. Sixteen levels stretched
across a wide range means the rounding step is coarse, and the per-weight error stops being negligible.
Two things have to change to make 4 bits survivable.

First, finer scale granularity. Instead of one scale per *row*, I quantize in small *groups* of
consecutive weights along the input dimension — group size G — each group getting its own scale and zero.
With G=32, every block of 32 weights is fit to the 4-bit grid with a scale and offset tuned to *that
block's* local range, so a quiet block gets a fine step and a loud block gets a coarse one, and no block
is forced onto a grid sized for some distant outlier. This is asymmetric/affine now, not symmetric — I
keep a min and a scale per group (a zero-point), because at 4 bits asymmetry buys real resolution when a
group's values aren't centered at zero. The cost is storage overhead: one (bf16) scale and zero per 32
weights is ~1 extra byte per 32 weights, a few percent on top of the 0.5 bytes/weight, which is why G=32
(rather than the larger G=128 default) is the aggressive choice — finer groups, better fidelity, slightly
more metadata. The group quantizer fits each block:

```python
def get_group_qparams(w, n_bit=4, groupsize=128):
    to_quant = w.reshape(-1, groupsize)
    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int     # span / 15
    zeros = min_val + scales * (2 ** (n_bit - 1))              # affine offset
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(torch.bfloat16).reshape(w.shape[0], -1)
```

Finer groups help, but on their own they still leave 4-bit error that, summed over a 32-layer network,
drifts the outputs. The second change is the one that actually matters at 4 bits, and it comes from
asking the right question. Round-to-nearest minimizes the error of *each weight in isolation*. But I
don't care about the weights in isolation — I care about the layer's *output*, `y = Wx`. The thing to
minimize is the output error `||Wx − Ŵx||²` over the real input distribution `x`, and round-to-nearest is
the wrong objective for that: rounding weight `i` up perturbs the output in a way that rounding a
*correlated* weight `j` could partly cancel, and naive rounding throws that cancellation away.

So: minimize output error, layer by layer, and exploit the correlations. Expand the output error over the
calibration inputs. `E[||(W − Ŵ)x||²] = E[ tr( (W−Ŵ) x xᵀ (W−Ŵ)ᵀ ) ] = tr( (W−Ŵ) H (W−Ŵ)ᵀ )`, where
`H = E[x xᵀ]` is the second-moment (Hessian) of the inputs — a `columns × columns` matrix I can estimate
by running a calibration set (a couple thousand short sequences from wikitext) through the layer and
accumulating `x xᵀ`. Now the quantization error is a *quadratic form weighted by H*, not an unweighted sum
of per-weight errors. The directions in weight space that H says the inputs excite strongly are the ones
where error is expensive; the directions inputs rarely excite are cheap. Round-to-nearest is blind to H;
I want to be greedy about it.

This is the GPTQ procedure. Quantize the columns of `W` one at a time, in order. When I quantize column
`i` to the grid, I incur an error `(w_i − ŵ_i)`; instead of letting that error sit, I *push it forward*
onto the not-yet-quantized columns, adjusting them so the layer's output is corrected for what I just did.
The exact amount to push is dictated by H: the optimal correction distributes the residual along the
inverse-Hessian-weighted directions, and the per-column denominator is the diagonal of the
inverse-Hessian. Concretely, for each column the quantization step is `q = round`, the local error is
`err = (w − dq) / d` with `d` the inverse-Hessian diagonal entry, and that error is propagated to the
remaining columns via the inverse-Hessian row. Working with the *Cholesky* of the inverse Hessian makes
the propagation a stable, ordered triangular sweep instead of a fresh solve per column, and a small
damping `percdamp · mean(diag H)` on the diagonal keeps H invertible when calibration directions are
degenerate:

```python
def faster_quant(self, H, W):
    # ... damp and invert H, take its (upper) Cholesky -> Hinv ...
    damp = self.percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        # ... block of columns ...
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            if groupsize != -1 and (i1 + i) % groupsize == 0:   # new group -> fresh qparams
                cur_qparams = self.get_qparams_func(W[:, (i1+i):(i1+i+groupsize)])
                all_qparams.append(cur_qparams)
            q  = self.quantize_func(w.unsqueeze(1), cur_qparams).flatten()
            dq = self.dequantize_func(q.unsqueeze(1), cur_qparams).flatten()
            err1 = (w - dq) / d                                 # error scaled by inv-Hessian diag
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))   # push onto remaining cols
```

The two ideas compose: grouping (G=32) sets the *grid* finely enough that 4 bits has real local
resolution, and GPTQ chooses *which way to round each weight* so the surviving error is the part the
inputs least excite. That's how 4 bits goes from "noticeably degraded" to "minimal" loss. The calibration
is offline and one-time — trace the model with `torch._dynamo.export`, accumulate H per linear from a
wikitext calibration set, run `faster_quant` per layer — so it costs nothing at decode time. The runtime
linear stores the int4-packed weights and the per-group scales-and-zeros and calls the fused int4 matmul
kernel:

```python
class WeightOnlyInt4Linear(torch.nn.Module):
    def forward(self, input):
        input = input.to(torch.bfloat16)
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)
```

backed by `torch.ops.aten._weight_int4pack_mm`, which reads the 4-bit-packed weights from HBM, dequants
on the fly using the group scales/zeros, and does the matmul — again, the win is the *read*: 4-bit storage
streamed across the bottleneck, widened inside the kernel.

The prediction, against 155.58 tok/s at 1069 GB/s. Halving the weight bytes once more should buy roughly
another 1.3× — I'm expecting throughput up near 195-200 tok/s, with the achieved GB/s dropping again
(same bandwidth-bound signature: fewer bytes/token, lower absolute GB/s). The risk is entirely on the
quality side, and it's realer than it was at int8: 16 levels is genuinely coarse, and if grouping + GPTQ
*don't* recover the output, the harness will show it. My bet is that the H-weighted error-feedback plus
G=32 grouping holds the degradation to *minimal* — small enough to be acceptable for the throughput, not
the zero-observable-loss I claimed for int8. If hellaswag/winogrande drop more than marginally, the
grouping is too coarse or the calibration too thin and I'd revisit G or the calibration set. The change
is the group quantizer, the GPTQ `faster_quant` runner, and the int4-packed runtime linear; full scaffold
code in the answer.
