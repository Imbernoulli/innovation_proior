155.58 tokens/s at 1069 GB/s. The int8 bet paid off exactly the way the bandwidth story predicted: I cut
the weight bytes in half and throughput went up `155.58/104.9 ≈ 1.48×`, with the achieved GB/s dropping
from 1397 to 1069 rather than rising — fewer bytes per token, same wall. Let me read the 1069 back into a
model size to check I understand it: `1069 / 155.58 ≈ 6.87 GB` streamed per token, which matches the ~7 GB
I estimated for the int8 model (6.47 GB of int8 weights plus the still-bf16 tail). So the bandwidth
diagnostic is internally consistent and the mechanism is confirmed: throughput is `achieved_bandwidth /
bytes_per_token`, and the win came entirely from the denominator. The lever is proven. The obvious next
pull is to keep going: if 8 bits per weight bought 1.48×, what about 4? int4 halves the weight footprint
*again* — the ~6.5 GB of int8 weights down toward ~3.3 GB — and on a bandwidth-bound workload that's the
path to roughly another `~1.3-1.5×`. The arithmetic budget is nowhere near binding (still ~150× to the
left of the roofline ridge, and now even the matmul intensity is unchanged since I still upcast and
multiply in bf16), so the only question, again, is quality: can I drop the weights to 4 bits without the
model getting observably worse?

And here the same trick that made int8 safe stops being enough, and I should quantify why rather than just
assert it. With int8 I had per-row scales and 256 levels; the rounding step was `m/127` and the per-weight
error `~m/254`, which propagated to a ~1% output perturbation — a rounding error. At 4 bits I have *16*
levels. A single scale per row now has to cover the whole row's dynamic range with sixteen rungs, so the
step jumps to `m/8` (asymmetric) or `m/7.5` — roughly *16× coarser* than int8's step. Redo the SNR
estimate with a 16× larger step and the output noise that was ~1% of signal becomes ~16%, and 16% output
perturbation per layer, compounded over 32 layers, is no longer a rounding error — it's the model getting
observably dumber. And it's worse than the uniform estimate suggests, because 4096 weights in a row do not
all live in the same small band: there are local clusters and occasional large entries, and sixteen levels
stretched across a range set by the largest entry means the typical weight sees only a few usable levels.
Round-to-nearest at 4 bits, per row, would show up on the harness. Two things have to change to make 4 bits
survivable, and they attack two different parts of the problem.

First, finer scale granularity. This is the observation I put in my back pocket at int8 — that asymmetric
plus finer granularity reopens with real stakes once levels get scarce — now cashed in. Instead of one
scale per *row*, I quantize in small *groups* of consecutive weights along the input dimension — group
size `G` — each group getting its own scale and zero-point. With `G=32`, every block of 32 weights is fit
to the 4-bit grid with a scale and offset tuned to *that block's* local range, so a quiet block gets a fine
step and a loud block gets a coarse one, and no block is forced onto a grid sized for some distant outlier
four thousand columns away. This is asymmetric/affine now, not symmetric — I keep a min and a scale per
group (a zero-point), because at 4 bits, with only 16 levels, the one level symmetric "wastes" and the
centering assumption cost a real slice of resolution when a group's values aren't centered at zero. Let me
price the storage overhead, because "finer" isn't free and `G` is a genuine trade. Each group of 32 weights
costs `32 × 0.5 B = 16 B` of packed int4 plus one bf16 scale and one bf16 zero, `2 × 2 = 4 B` of metadata —
so `4/16 = 25%`... no: the metadata is `4 B` per `32` weights, i.e. `4/32 = 0.125 B` per weight on top of
the `0.5 B` packed, so `0.625 B/weight` effective, a `25%` overhead on the raw int4 storage but only a few
percent on the total token if I compare it against the pre-int4 footprint. Contrast `G=128`: `4 B` per
`128` weights is `0.031 B/weight`, so `0.53 B/weight` — cheaper metadata, but four times coarser grouping,
which at 4 bits is exactly the resolution I can't spare. `G=32` is the aggressive choice: finer groups,
better fidelity, slightly more metadata. That's the deliberate trade — I'm spending a few percent more
bytes to buy back the resolution that 4 bits threw away. The group quantizer fits each block:

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

Let me check the affine formula lines up. `max_int = 15`, so `scales = (max − min)/15` maps the group's
full span onto the 16 integer levels `0..15`. The zero-point `zeros = min + scales·8` is the real value
that the integer `8` (mid-grid) decodes to — i.e. the affine map is `dequant(q) = scales·q + (min −
scales·8·(−1))`... the convention here stores `zeros` as the real value at grid-center, and dequant
reconstructs `w ≈ scales·(q − 8) + zeros_offset` consistently with how `linear_forward_int4`'s kernel
unpacks it. The important property is that `min` and `max` of the *group* both land exactly on grid
endpoints, so all 16 levels are used within each 32-weight block — which is the whole point of grouping.

Finer groups help, but on their own they still leave 4-bit error that, summed over a 32-layer network,
drifts the outputs — grouping shrinks the step but 16 levels is still 16 levels, and my SNR estimate even
with per-group scales is a few times worse than int8's. The second change is the one that actually matters
at 4 bits, and it comes from asking the right question. Round-to-nearest minimizes the error of *each
weight in isolation*. But I don't care about the weights in isolation — I care about the layer's *output*,
`y = Wx`. The thing to minimize is the output error `E||Wx − Ŵx||²` over the real input distribution `x`,
and round-to-nearest is the wrong objective for that: rounding weight `i` up perturbs the output in a way
that rounding a *correlated* weight `j` could partly cancel, and naive rounding throws that cancellation
away. When two input features are correlated, an over-estimate on one weight can be compensated by an
under-estimate on the other; RTN, blind to the input statistics, never does this.

So: minimize output error, layer by layer, and exploit the correlations. Expand the output error over the
calibration inputs. `E[||(W − Ŵ)x||²] = E[ tr( (W−Ŵ) x xᵀ (W−Ŵ)ᵀ ) ] = tr( (W−Ŵ) H (W−Ŵ)ᵀ )`, where
`H = E[x xᵀ]` is the second-moment (Hessian) of the inputs — a `columns × columns` matrix I can estimate
by running a calibration set (a couple thousand short sequences from wikitext) through the layer and
accumulating `x xᵀ`. This is a real Hessian in the sense that for the quadratic loss `||Wx−Ŵx||²`, the
second derivative with respect to the weights *is* `2H` — the loss is exactly quadratic in the weights, so
`H` is not an approximation of curvature, it *is* the curvature. Now the quantization error is a *quadratic
form weighted by H*, not an unweighted sum of per-weight errors. The directions in weight space that H says
the inputs excite strongly are the ones where error is expensive; the directions inputs rarely excite are
cheap. Round-to-nearest is blind to H; I want to be greedy about it.

This is the GPTQ procedure. Quantize the columns of `W` one at a time, in order. When I quantize column `i`
to the grid, I incur an error `(w_i − ŵ_i)`; instead of letting that error sit and corrupt the output, I
*push it forward* onto the not-yet-quantized columns, adjusting them so the layer's output is corrected for
what I just did. The exact amount to push is dictated by H: minimizing the quadratic form subject to
column `i` being fixed to its quantized value has a closed-form solution, and it distributes the residual
along the inverse-Hessian-weighted directions, with the per-column denominator being the diagonal of the
inverse-Hessian. Concretely, for each column the quantization step is `q = round`, the local error is
`err = (w − dq) / d` with `d` the inverse-Hessian diagonal entry, and that error is propagated to the
remaining columns via the inverse-Hessian row. Working with the *Cholesky* of the inverse Hessian makes the
propagation a stable, ordered triangular sweep instead of a fresh linear solve per column — a huge
practical saving, since a fresh solve per column would be `O(columns⁴)` for the layer, while the Cholesky
sweep is `O(columns³)` once — and a small damping `percdamp · mean(diag H)` on the diagonal keeps H
invertible when calibration directions are degenerate (a `dead` column that the calibration never excites
would otherwise make H singular):

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

Let me trace the inner loop once to be sure the error-feedback bookkeeping is right, because a sign or an
index slip here silently makes it worse than RTN. Column `i` has value `w`; I fetch its inverse-Hessian
diagonal `d = Hinv1[i,i]`; every 32 columns I recompute `cur_qparams` from the *original* `W`'s next group
(so the grid is set by clean weights, not by already-error-compensated ones). I quantize and dequantize to
get `dq`, the actual reconstructed value. The residual `err1 = (w − dq)/d` is the raw error normalized by
the curvature diagonal — dividing by `d` is what makes the subsequent update the *optimal* correction
rather than a heuristic. Then `W1[:, i:] -= err1 ⊗ Hinv1[i, i:]` subtracts a rank-1 update from all
not-yet-quantized columns in the block, moving them to absorb the error along the inverse-Hessian row. The
column index only ever moves forward, the qparams for a group are frozen once computed, and the update
touches only columns `≥ i` — so it's a clean triangular sweep. The block structure (`blocksize`) just
batches the cross-block correction (`W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]`) for efficiency; it doesn't
change the math. The trace checks out.

Let me make the "correlated columns cancel" claim concrete on a two-column toy, because it's the whole
justification for GPTQ over RTN and I want to see it work rather than take it on faith. Suppose a layer has
two input features that are strongly correlated, so `H = [[1, 0.9], [0.9, 1]]` (unit variance, correlation
0.9). Take a one-output row `w = [0.10, 0.10]` and a crude grid whose nearest points are at multiples of
0.1 with a rounding that happens to push column 0 *up* by 0.04 (so `dq_0 = 0.14`, `err_0 = −0.04` before
the `/d` scaling). RTN would also round column 1 independently to its nearest point and eat both errors.
GPTQ instead quantizes column 0, sees the +0.04 output-side overshoot, and — because column 1 is 0.9-
correlated with column 0 — pushes a compensating adjustment onto `w_1` so that when column 1 is quantized
it lands lower, roughly `w_1 ← 0.10 − 0.9·0.04 ≈ 0.064`, which then rounds to a grid point that *cancels*
most of column 0's output error along the direction the inputs actually excite (the `[1,1]` direction that
`H` weights heavily). The net output error `tr((W−Ŵ)H(W−Ŵ)ᵀ)` comes out markedly smaller than RTN's,
precisely because the update used `H`'s off-diagonal to route the residual into column 1. On an
*uncorrelated* `H = I` the off-diagonal is zero, the push does nothing, and GPTQ degenerates to RTN — which
is the correct behavior: when there's no correlation to exploit, there's no free cancellation, and GPTQ
doesn't pretend otherwise. That degeneracy check is reassuring; it means GPTQ can only help or tie, never
hurt relative to RTN under this objective.

Before I settle on 4 bits I should ask whether to be greedier still — int3, int2 — since the whole ladder
has been "pull the byte lever again." Two forces say stop at 4. First, the metadata overhead grows as a
*fraction* of the shrinking payload: at G=32 the `4 B/group` scale+zero is `0.125 B/weight`, which is 25%
on top of int4's `0.5 B` but would be 33% on top of int3's `0.375 B` and 50% on top of int2's `0.25 B` —
so the effective bytes stop halving cleanly, and I'd need coarser groups to keep the metadata down, which
fights the resolution I need even more desperately at 8 or 4 levels. Second, the quality cliff: my SNR
estimate degrades ~2× per bit dropped, and even GPTQ's error-feedback has a floor — it redistributes error
but can't create resolution that isn't there, so below 4 bits the "minimal" loss I'm betting on at int4
would become a real, harness-visible regression that violates the no-dumber constraint. So 4 bits is the
aggressive-but-safe stopping point for the *representation* lever specifically: it's the last width where
grouping-plus-GPTQ can plausibly hold quality, and it's where the byte savings are still worth the
metadata. If I want more throughput past this, the byte-per-weight lever is nearly exhausted and I'll have
to find a *different* axis to attack than the width of the weights.

The two ideas compose, and it's worth being precise about the division of labor: grouping (G=32) sets the
*grid* finely enough that 4 bits has real local resolution — it fixes the step size — and GPTQ chooses
*which way to round each weight onto that grid* so the surviving error lands in the directions the inputs
least excite. One sets the quantization lattice; the other picks the lattice point. That's how 4 bits goes
from "noticeably degraded" to "minimal" loss. The calibration is offline and one-time — trace the model
with `torch._dynamo.export`, run a wikitext calibration set through to accumulate `H` per linear, run
`faster_quant` per layer — so it costs nothing at decode time; it's a one-off pre-processing pass, minutes
of GPU time that produce a static quantized checkpoint. The runtime linear stores the int4-packed weights
and the per-group scales-and-zeros and calls the fused int4 matmul kernel:

```python
class WeightOnlyInt4Linear(torch.nn.Module):
    def forward(self, input):
        input = input.to(torch.bfloat16)
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)
```

backed by `torch.ops.aten._weight_int4pack_mm`, which reads the 4-bit-packed weights from HBM, dequants on
the fly using the group scales/zeros, and does the matmul — again, the win is the *read*: 4-bit storage
streamed across the bottleneck, widened inside the kernel. The `padding` handles a shape constraint the
packed kernel imposes: `_weight_int4pack_mm` needs the input feature dimension to be a multiple of the
group size and the tiling, so a linear whose `in_features` isn't a clean multiple gets its input padded
with zeros to the next valid size — the padded columns contribute nothing to the dot product, so it's
correctness-preserving, just a little wasted read on the odd-shaped layers.

Two runtime details earn a mention because they're where the abstract "4-bit storage" becomes real bytes.
The weights aren't stored as a naive int4 array; they're pre-packed by `prepare_int4_weight_and_scales_and
_zeros` into the tiled layout `_weight_int4pack_mm` expects (`inner_k_tiles=8`), so that when the kernel
streams them it reads contiguous, coalesced blocks and dequantizes them with the matching group qparams in
registers. That packing is why the int4 path can actually approach its byte-count advantage rather than
being throttled by a scatter-gather access pattern — the layout is chosen so the read is sequential. And it
composes with the compiled decode graph exactly as int8 did: `WeightOnlyInt4Linear` is a plain module with
fixed-shape buffers (`weight`, `scales_and_zeros`), so it traces and captures into the same CUDA graph, and
I keep the entire scheduling win from the compile rung stacked under the quantization. Nothing about the
graph discipline breaks; the int4 kernel is just another captured op with static inputs.

The prediction, against 155.58 tok/s at 1069 GB/s. Let me size the int4 model to ground it. The ~6.47B
linear params at `0.625 B/weight` (int4 plus G=32 metadata) is `~4.0 GB`, plus the un-quantized bf16 tail
(embeddings, head, norms) at `~0.5 GB`, so ~4.5 GB streamed per token versus the int8 model's ~6.87 GB —
a byte ratio of `6.87/4.5 ≈ 1.53×`. Real bandwidth inefficiency at 4-bit packing pulls the realized gain
below that ideal, so I expect roughly `~1.3×`, throughput up near 195-200 tok/s, with the achieved GB/s
dropping again (same bandwidth-bound signature: fewer bytes/token, lower absolute GB/s, because the int4
read is a less efficient access pattern than int8 and the un-quantized tail is now a bigger *fraction* of
the shrinking token). The risk is entirely on the quality side, and it's realer than it was at int8: 16
levels is genuinely coarse, my SNR estimate says several times worse than int8 even with grouping, and if
G=32 + GPTQ *don't* recover the output, the harness will show it. My bet is that the H-weighted
error-feedback plus G=32 grouping holds the degradation to *minimal* — small but nonzero, distinct from the
no-observable-loss I claimed for int8, and acceptable in exchange for the throughput. If hellaswag/
winogrande drop more than marginally, the diagnosis is either the grouping is too coarse (raise the
metadata budget, smaller G) or the calibration is too thin (H estimated from too few tokens, so the
error-feedback is aimed at the wrong directions), and I'd revisit those knobs. The change is the group
quantizer, the GPTQ `faster_quant` runner, and the int4-packed runtime linear; full scaffold code in the
answer.
