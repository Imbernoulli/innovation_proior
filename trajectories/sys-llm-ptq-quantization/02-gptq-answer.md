**Problem (from step 1).** RTN rounds each weight to the grid independently, optimizing per-weight
distance instead of the thing we care about — the layer's output WX. At 3-bit per-channel that loses
catastrophically: LLaMA-7B blows up to 25.54 perplexity against an FP16 of 5.68. The rounding error is
structured, not noise, so it should be *compensable*.

**Key idea.** **GPTQ** (Frantar 2022). Solve the per-layer output-reconstruction least squares
min‖WX − ŴX‖²_F directly with the Optimal-Brain-Surgeon update: quantize weights one column at a time,
and after each one apply the closed-form OBS compensation — governed by the input Hessian H = 2XXᵀ — to
the *not-yet-quantized* weights so they repair the error just introduced. Make OBS scale to billions of
parameters with three changes:

1. **Fixed shared column order.** Greedy ordering barely helps on huge over-parameterized layers, so
   quantize all rows in the *same* left-to-right order. H depends only on inputs, so one inverse is
   shared across rows and downdated once per *column*, not once per weight.
2. **Lazy batch updates.** Process columns in blocks of B=128; keep compensation contained to the block,
   then apply the block's accumulated error to all columns to the right in a single GEMM — turning
   bandwidth-bound rank-one updates into compute-bound matmuls.
3. **Cholesky reformulation.** The OBS inverse-downdate is exactly symmetric Gaussian elimination, so all
   the scaled inverse-Hessian row-tails are read off one stable upper Cholesky factor U of H⁻¹ (with ~1%
   diagonal dampening), instead of thousands of in-place downdates that drift indefinite at scale.

**Why it works.** Minimizing the layer output rather than per-weight grid distance lets the error from
already-quantized weights be absorbed by downstream weights, recovering most of the gap to FP16. It is
grid-agnostic — the inner quantize is just RTN onto per-channel *or* g128 scales — and group scales can
be fit against the already-updated weights, which unlocks the 2–3-bit regime where RTN died.

**Change / code.** `add_batch` accumulates the Hessian from calibration activations; `compress`
Cholesky-factors H⁻¹ once and sweeps blocks of 128 columns with OBS compensation.

```python
import torch

def add_batch(self, inp):
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    batch = inp.shape[0]
    inp = inp.reshape(-1, inp.shape[-1]).t().float()      # d_col x tokens
    self.H *= self.nsamples / (self.nsamples + batch)
    self.nsamples += batch
    inp *= (2.0 / self.nsamples) ** 0.5                   # scaled Hessian average  H = 2 X Xᵀ
    self.H += inp.matmul(inp.t())

def compress(self, quantizer, blocksize=128, percdamp=0.01, groupsize=-1):
    W, H = self.W.clone(), self.H.clone()
    if not quantizer.ready():
        quantizer.find_params(W, weight=True)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.d_col, device=H.device)
    H[diag, diag] += damp                                 # dampen for a stable Cholesky
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    U = torch.linalg.cholesky(H, upper=True)              # scaled inverse-Hessian row-tails, computed ONCE
    Q = torch.zeros_like(W)
    for i1 in range(0, self.d_col, blocksize):
        i2 = min(i1 + blocksize, self.d_col)
        W1, U1 = W[:, i1:i2].clone(), U[i1:i2, i1:i2]
        Q1, Err1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for i in range(i2 - i1):
            w = W1[:, i]; d = U1[i, i]
            if groupsize != -1 and (i1 + i) % groupsize == 0:
                quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
            q = quantizer.quantize(w.unsqueeze(1)).flatten()   # round THIS column to the grid
            Q1[:, i] = q
            err = (w - q) / d                                  # scaled OBS error
            W1[:, i:] -= err.unsqueeze(1) * U1[i, i:].unsqueeze(0)   # compensate within the block
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(U[i1:i2, i2:])            # one GEMM updates the rest of the rows
    self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
```

**Target.** Bring 3-bit per-channel LLaMA-7B down from RTN's 25.54 toward single digits — making low-bit
weight-only quantization nearly lossless and leaving the next question as *which* weights deserve
protection.
