**Problem (from baseline).** int8 weight-only reached 155.58 tok/s by halving the weight bytes with no
observable quality loss. The workload is still bandwidth-bound, so the next lever is the same: fewer bytes
per token. But going to 4 bits leaves only 16 levels — a single per-row scale is too coarse, and
round-to-nearest drifts the layer outputs over 32 layers.

**Key idea.** int4 weight-only, made survivable by two changes. (1) **Groupwise scales, G=32:** quantize
consecutive blocks of 32 weights affinely (per-group scale + zero), so each block's grid matches its local
range — finer than per-row, at a few percent metadata overhead. (2) **GPTQ:** quantize columns left to
right minimizing the *output* error `tr((W−Ŵ) H (W−Ŵ)ᵀ)` (H = E[xxᵀ] from wikitext calibration), pushing
each column's rounding residual onto the not-yet-quantized columns along inverse-Hessian directions
(stable triangular sweep via the Cholesky of H⁻¹, with `percdamp` damping). Grouping sets a fine grid;
GPTQ chooses which way to round so the surviving error is the part the inputs least excite.

**Why it works.** Throughput ≈ bandwidth / bytes-per-token, so 4-bit storage (~3.5 GB vs 13.5 GB) buys
another ~1.3×; the win is int4 *storage* streamed across HBM and dequantized inside the
`_weight_int4pack_mm` kernel, not int4 arithmetic. Round-to-nearest minimizes per-weight error, the wrong
objective; GPTQ minimizes per-layer output error under the real input covariance, which is what
correlated weights let it partly cancel. The accuracy harness (hellaswag/winogrande, via `eval.py`) shows
**minimal** quality loss — not the zero-observable-loss of int8, but small enough to take the throughput.

**Change / code.** Group quantizer + GPTQ runner (offline) + int4-packed runtime linear.

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
