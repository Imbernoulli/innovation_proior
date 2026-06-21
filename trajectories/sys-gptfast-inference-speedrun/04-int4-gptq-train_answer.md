int8 weight-only landed at 155.58 tok/s with no observable quality loss, and the achieved bandwidth dropped (1069 GB/s) exactly as a bandwidth-bound win should: fewer bytes per token, same wall. The lever is proven, so the obvious next pull is to keep going — if 8 bits per weight bought $1.5\times$, what about 4? int4 halves the weight footprint again, ~13.5 GB of bf16 toward ~3.5 GB, which on a bandwidth-bound workload is the path to roughly another $1.3\times$. The arithmetic budget is nowhere near binding, so the only real question, again, is quality.

And here the trick that made int8 safe stops being enough. int8 had ~256 levels and per-row scales, so rounding was a rounding error. At 4 bits I have *16* levels, and a single per-row scale now has to cover the whole row's dynamic range with sixteen rungs. The 4096 weights in a row do not all live in one small band — there are local clusters and occasional large entries — so sixteen levels stretched across a wide range makes the step coarse and the per-weight error stops being negligible. I propose **int4 weight-only with groupwise scales at $G=32$ plus GPTQ**, two changes that compose to make 4 bits survivable.

The first change is finer scale granularity. Instead of one scale per *row*, I quantize in small *groups* of $G=32$ consecutive weights along the input dimension, each group getting its own scale and offset tuned to *that block's* local range, so a quiet block gets a fine step and a loud one a coarse step, and no block is forced onto a grid sized for some distant outlier. This is *affine* (asymmetric) now, not symmetric — I keep a per-group min and scale (a zero-point) — because at 4 bits asymmetry buys real resolution when a group's values are not centered at zero. The quantizer fits each block by mapping its span onto the 4-bit grid: $\text{scale} = (\max - \min)/(2^{n}-1)$, with $\text{zero} = \min + \text{scale}\cdot 2^{\,n-1}$. The cost is metadata — one bf16 scale and zero per 32 weights, ~1 extra byte per 32, a few percent on top of the 0.5 bytes/weight — which is why $G=32$ rather than the larger default of 128 is the aggressive choice: finer groups, better fidelity, slightly more overhead.

Finer groups alone still leave 4-bit error that, summed over 32 layers, drifts the outputs, because round-to-nearest minimizes the error of *each weight in isolation* — the wrong objective. I do not care about weights in isolation; I care about the layer's output $y = Wx$. The quantity to minimize is the output error $\mathbb{E}\,\lVert (W-\hat W)x\rVert^2$ over the real input distribution, and rounding weight $i$ up perturbs the output in a way that rounding a *correlated* weight $j$ could partly cancel — naive rounding throws that cancellation away. Expanding,
$$\mathbb{E}\,\lVert (W-\hat W)x\rVert^2 = \operatorname{tr}\!\big((W-\hat W)\,H\,(W-\hat W)^\top\big),\qquad H = \mathbb{E}[x x^\top],$$
where $H$ is the input second-moment (Hessian), a columns×columns matrix I estimate by running a wikitext calibration set through the layer and accumulating $xx^\top$. The error is now a *quadratic form weighted by $H$*, not an unweighted sum of per-weight errors: directions the inputs excite strongly are expensive, directions they rarely excite are cheap, and round-to-nearest is blind to this.

This is GPTQ. I quantize the columns of $W$ one at a time, left to right. Quantizing column $i$ incurs an error $(w_i - \hat w_i)$; instead of letting it sit, I *push it forward* onto the not-yet-quantized columns, correcting the layer's output for what I just did, in the amount $H$ dictates — the optimal correction distributes the residual along inverse-Hessian-weighted directions, with the per-column denominator being the diagonal of the inverse Hessian. Concretely, the local error is $\text{err} = (w - dq)/d$ with $d$ the inverse-Hessian diagonal entry, propagated to the remaining columns via the inverse-Hessian row. Working with the *Cholesky* of $H^{-1}$ makes the propagation a stable, ordered triangular sweep instead of a fresh solve per column, and a small damping $\text{percdamp}\cdot\operatorname{mean}(\operatorname{diag}H)$ keeps $H$ invertible when calibration directions are degenerate. The two ideas compose cleanly: grouping at $G=32$ sets the *grid* finely enough that 4 bits has real local resolution, and GPTQ chooses *which way to round each weight* so the surviving error is the part the inputs least excite — together turning 4 bits from "noticeably degraded" into "minimal" loss.

The calibration is offline and one-time, so it costs nothing at decode time. The runtime linear stores the int4-packed weights and the per-group scales-and-zeros and calls a fused int4 matmul kernel, `torch.ops.aten._weight_int4pack_mm`, which reads the 4-bit-packed weights from HBM, dequantizes on the fly using the group scales/zeros, and does the matmul — once more, the win is the *read*: 4-bit storage streamed across the bottleneck, widened inside the kernel, not int4 arithmetic. I expect another ~$1.3\times$, near 195–200 tok/s, with achieved GB/s dropping again (same bandwidth-bound signature). The risk is realer than at int8 — 16 levels is genuinely coarse — so the bet is that the $H$-weighted error-feedback plus $G=32$ holds the degradation to *minimal* (small enough to take, not the zero-observable-loss of int8), checked on hellaswag/winogrande via `eval.py`; if it drops more than marginally, the grouping is too coarse or the calibration too thin.

```python
def get_group_qparams(w, n_bit=4, groupsize=128):
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    to_quant = w.reshape(-1, groupsize)
    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(torch.bfloat16).reshape(w.shape[0], -1)

class WeightOnlyInt4QuantHandler:
    def __init__(self, mod, groupsize=128, inner_k_tiles=8, padding=True):
        self.mod, self.groupsize = mod, groupsize
        self.inner_k_tiles, self.padding = inner_k_tiles, padding
        assert groupsize in [32, 64, 128, 256]
    @torch.no_grad()
    def create_quantized_state_dict(self, use_cuda=True):
        device = "cuda" if use_cuda else "cpu"
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                weight = mod.weight.data
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16).to(device), self.groupsize, self.inner_k_tiles)
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to('cpu')
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to('cpu')
        return cur_state_dict

# GPTQ: error-feedback column sweep weighted by the input Hessian H = E[x xᵀ]
def faster_quant(self, H, W):
    percdamp, blocksize, groupsize = self.percdamp, self.blocksize, self.groupsize
    W = W.detach().float()
    rows, columns = W.shape[0], W.shape[1]
    device = W.device
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=device); H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    DQ = torch.zeros_like(W); all_qparams = []
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns); count = i2 - i1
        W1 = W[:, i1:i2].clone(); DQ1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]; d = Hinv1[i, i]
            if groupsize != -1 and (i1 + i) % groupsize == 0:
                cur_qparams = self.get_qparams_func(W[:, (i1+i):(i1+i+groupsize)])
                all_qparams.append(cur_qparams)
            q  = self.quantize_func(w.unsqueeze(1), cur_qparams).flatten()
            dq = self.dequantize_func(q.unsqueeze(1), cur_qparams).flatten()
            DQ1[:, i] = dq
            err1 = (w - dq) / d
            W1[:, i:] -= err1.to(Hinv1.dtype).unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        DQ[:, i1:i2] = DQ1
        W[:, i2:] -= Err1.to(Hinv.dtype).matmul(Hinv[i1:i2, i2:])
    all_qparams = self.combine_qparams_list_func(all_qparams)
    Q = self.quantize_func(DQ, all_qparams)
    return Q, DQ.to(W.dtype), all_qparams

class WeightOnlyInt4Linear(torch.nn.Module):
    def forward(self, input):
        input = input.to(torch.bfloat16)
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)

def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    return c.reshape(origin_x_size[:-1] + (out_features,))

# build offline: python quantize.py --mode int4-gptq --groupsize 32 --calibration_tasks wikitext --calibration_seq_length 2048
```
