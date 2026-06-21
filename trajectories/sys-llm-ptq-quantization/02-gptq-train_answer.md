Round-to-nearest gave me the floor exactly as advertised: fine at 4-bit g128 (Llama-2-7B at 5.73 perplexity), and a wreck at 3-bit per-channel — LLaMA-7B at **25.54** against an FP16 of 5.68. That 25.54 is the number to beat, and what nags me is that RTN throws away information it does not have to. It rounds every weight to its nearest grid point *independently*, as if each weight stood alone. But the weights in a linear layer do not act alone; they are summed against an input to produce an output, and what I actually care about is not whether each weight is close to the grid but whether the *layer's output* is close to its FP16 value. RTN optimizes the wrong objective.

Written honestly, for a linear layer with weight $W$ and calibration inputs $X$ the quantity I want to preserve is the output $WX$, so the per-layer problem is
$$\min_{\hat W \text{ on the grid}} \; \lVert WX - \hat W X \rVert_F^2.$$
This is a constrained least-squares problem: find the grid-valued $\hat W$ that reproduces the FP16 output on the calibration data. RTN ignores $X$ entirely — it is the solution you get if you pretend $X = I$. The moment $X$ is put back, I can do much better, because the errors made on already-quantized weights can be compensated by adjusting the weights *not yet* quantized: round one weight down, see the output pushed up, and nudge a still-unquantized weight to push it back. RTN has no mechanism for this; the least-squares objective does.

The classical tool for exactly this is Optimal Brain Surgeon / Optimal Brain Quantization: quantize weights one at a time, and after each rounding apply a closed-form update to *all remaining* weights that optimally compensates the error just introduced. The compensation is governed by the Hessian of the layer objective, which here is beautifully simple — the objective is quadratic in $W$, so $H = 2XX^\top$ depends only on the inputs, not on the weights. The OBS step rounds a weight, incurs error $(w - \mathrm{quant}(w))$, and redistributes it onto the surviving weights weighted by the rows of $H^{-1}$. Run to completion and you get the least-squares-optimal grid assignment, far better than independent rounding.

The reason this is not already standard on 175B-parameter models is that OBS is far too slow: it re-chooses which weight to quantize next and downdates the inverse Hessian after every single weight, which is cubic in the layer width per row and quadratic in the number of rows. The method I propose, **GPTQ**, keeps the OBS-quality compensation but makes it scale through three real algorithmic changes, not tweaks.

First, *kill the greedy order*. OBQ's per-weight cleverness is choosing, for each row, the least-damaging weight to do next — but on large, heavily over-parameterized layers that advantage all but vanishes, because there is so much redundancy that almost any order works about as well. Since $H = 2XX^\top$ depends only on the inputs and is therefore *shared across all rows*, I quantize every row in the *same* fixed left-to-right column order. The inverse-Hessian information each row needs is then identical, so I compute one $H^{-1}$ and downdate it *once per column* instead of once per weight, collapsing per-weight work into per-column work.

Second, *batch the updates*. Even shared across rows, applying the rank-one error-compensation to all columns to the right after every single column is bandwidth-bound — lots of memory traffic, tiny arithmetic. So I process columns in *blocks* of $B = 128$: inside a block I do the per-column updates but keep the compensation contained to the block, and then once per block I apply the block's whole accumulated error to all columns to the right in a *single* matrix multiply. That turns a stream of memory-bound rank-one updates into one compute-bound GEMM per block — exactly what a GPU wants.

Third, *stop downdating the inverse*. OBS keeps an explicit inverse Hessian and downdates it in place; at this scale, with thousands of accumulating downdates, that running inverse drifts numerically and eventually goes indefinite, and the algorithm collapses. But the OBS inverse-downdate is, examined carefully, *exactly symmetric Gaussian elimination* on $H^{-1}$. That means all the scaled inverse-Hessian row-tails the update needs can be read directly off a single **Cholesky factor** of $H^{-1}$, computed once, up front, in a stable routine — instead of thousands of in-place downdates. I add a mild dampening of $\approx 1\%$ of the mean diagonal to $H$ so the Cholesky is well-conditioned, take the upper factor $U$ with $H^{-1} = U^\top U$, and then for column $j$ the row-tail $U[j, j{:}] / U[j,j]$ *is* the sequentially-downdated inverse row I need. No running inverse at all.

Putting the three together: build $H = 2XX^\top$ from calibration activations, dampen and Cholesky-factor $H^{-1}$ once, then sweep blocks of 128 columns left to right; inside each block round each column to the grid, form the scaled error $(w - q)/U_{jj}$, push it onto the remaining in-block columns via the Cholesky row-tail, and after the block apply the accumulated error to all columns to the right with one GEMM. It is grid-agnostic, which matters: the inner `quantize` is just RTN onto whatever grid I hand it, so this compensation composes with per-channel *or* g128 scales, and the group scales can even be fit against the *already-updated* weights — which is what unlocks the extreme 2–3-bit regime where RTN died.

The bet is that replacing independent rounding with output-reconstruction-optimal compensation recovers most of the gap to FP16: 3-bit per-channel LLaMA-7B coming down from 25.54 to single digits, into the 8-ish range — making low-bit weight-only quantization nearly lossless, and turning the next question from *how* to round into *which* weights deserve protection.

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
