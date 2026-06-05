# GPTQ

**Problem.** Quantize the weights of a very large pretrained Transformer (up to 175B parameters) to 3–4 bits in one shot — no retraining, a few hours of compute, a tiny calibration set — while keeping perplexity essentially unchanged. Naive round-to-nearest scales but collapses below 8 bits; accurate second-order methods (OBQ) are too slow because their per-row cubic runtime cannot reach billions of parameters.

**Key idea.** Solve the per-layer output-reconstruction problem $\min_{\widehat{\mathbf W}}\lVert\mathbf{WX}-\widehat{\mathbf W}\mathbf X\rVert_2^2$ with the Optimal-Brain-Surgeon update, but make three changes that turn an $O(d_{\text{row}}d_{\text{col}}^3)$ algorithm into a scalable one:

1. **Fixed shared column order.** Greedy ordering barely helps on large over-parameterized layers, so quantize *all rows in the same left-to-right order*. The reconstruction Hessian $\mathbf H = 2\mathbf X\mathbf X^\top$ depends only on the inputs, so a single inverse is shared across rows and downdated once per column instead of once per weight — runtime drops to $O(\max\{d_{\text{row}}d_{\text{col}}^2,\,d_{\text{col}}^3\})$.
2. **Lazy batch updates.** Process columns in blocks of $B=128$, keeping per-column error compensation contained to the block, then apply the block's accumulated error to all columns to the right in a single GEMM — converting a bandwidth-bound rank-one update into compute-bound matrix multiplies.
3. **Cholesky reformulation.** The OBS inverse downdate is exactly symmetric Gaussian elimination, so all the inverse-Hessian row-tails the algorithm needs are read from one numerically stable Cholesky factorization of $\mathbf H^{-1}$ (with mild dampening $\lambda=1\%$ of the mean diagonal), instead of thousands of accumulating in-place downdates that drift indefinite at scale.

**Algorithm (per linear layer).** Build $\mathbf H^{-1}=(2\mathbf X\mathbf X^\top+\lambda\mathbf I)^{-1}$, take its Cholesky factor. Sweep blocks of $B$ columns; inside a block, for each column $j$: round to grid $\mathbf Q_{:,j}=\mathrm{quant}(\mathbf W_{:,j})$, form per-row error $\mathbf E_{:,j}=(\mathbf W_{:,j}-\mathbf Q_{:,j})/[\mathbf H^{-1}]_{jj}$, and update the remaining in-block columns $\mathbf W_{:,j:}\mathrel{-}=\mathbf E_{:,j}\,\mathbf H^{-1}_{j,j:}$. After the block, $\mathbf W_{:,\text{rest}}\mathrel{-}=\mathbf E\,\mathbf H^{-1}_{\text{block},\text{rest}}$. The method is grid-agnostic: it combines with per-group scales (and group scales can be fit against the already-updated weights), enabling the 2-bit / ternary extreme regime.

```python
import torch

def add_batch(self, inp):
    inp = inp.reshape(-1, inp.shape[-1]).t().float()       # d_col x tokens
    n = inp.shape[1]
    self.H *= self.nsamples / (self.nsamples + n)
    self.nsamples += n
    inp *= (2.0 / self.nsamples) ** 0.5                    # so sum -> 2 X X^T
    self.H += inp.matmul(inp.t())

def quantize_layer(self, quantizer, blocksize=128, percdamp=0.01):
    W = self.W.clone()
    H = self.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.d_col)
    H[diag, diag] += damp

    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)            # rows give H^{-1} tails

    Q = torch.zeros_like(W)
    for i1 in range(0, self.d_col, blocksize):
        i2 = min(i1 + blocksize, self.d_col)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, Err1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for i in range(i2 - i1):
            w = W1[:, i]; d = Hinv1[i, i]
            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])         # block GEMM update

    self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
```

The driver quantizes one Transformer block at a time: register hooks that feed `add_batch` from calibration activations, run the calibration data, quantize each linear layer, then re-run the calibration inputs through the now-quantized block to supply inputs for the next block. At inference, a dedicated quantized-weight × FP16-vector kernel dequantizes weights on the fly, giving memory-bound generation speedups with no activation quantization required.
