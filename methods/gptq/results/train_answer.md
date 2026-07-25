The problem is how to compress the weights of a large pretrained Transformer down to 3 or 4 bits in a single pass, with no retraining and only a small calibration set, while keeping language-modeling perplexity essentially intact. The naive approach is round-to-nearest, which is fast but ignores how weights interact through the layer output; at 8 bits it is acceptable, but at 3 bits the accumulated error makes perplexity explode. More accurate second-order methods such as OBQ exist, but they maintain a separate inverse-Hessian trajectory for every row of the weight matrix, giving cubic runtime that tops out around a hundred million parameters. So the real gap is between scalable-but-crude and accurate-but-slow quantization.

The way forward is to preserve the layer's output rather than the weights themselves. For a linear layer with weights W and calibration inputs stacked as columns of X, the right objective is to minimize the reconstruction error ||WX - W_hat X||_2^2. This objective decomposes by output row, and every row shares the same curvature because the Hessian is H = 2 X X^T, which depends only on the inputs. The crucial empirical observation is that greedy ordering of which weight to quantize next barely helps on the heavily over-parameterized layers found in large Transformers. If all rows are quantized in the same fixed left-to-right order, one shared inverse can serve every row, and the per-row cubic cost disappears.

The method is GPTQ. It takes the Optimal Brain Surgeon update and restructures it so it scales to models with hundreds of billions of parameters. The core loop processes columns in fixed order: quantize the current column, compute its per-row residual, and push that residual onto the still-unquantized columns using the inverse Hessian. Because the order is fixed and shared, the inverse only needs to be downdated once per column rather than once per weight. To avoid bandwidth-bound rank-one updates, columns are processed in blocks, typically of size 128. Within a block the compensation is kept local; after the block is finished, the accumulated error is applied to all remaining columns in a single GEMM, which is efficient on a GPU. For numerical stability the repeated explicit downdates are replaced by a single upper Cholesky factor of H^{-1}. The OBS update only needs the scaled rightward tail of each inverse row, and those tails are exactly the rows of the Cholesky factor. A small amount of dampening, around 1% of the mean diagonal of H, is added before inversion to keep the matrix well-conditioned.

GPTQ is also grid-agnostic, so it composes cleanly with per-group scaling. When grouping is enabled, each group's scale is recomputed from the already-compensated weights at the moment that group is reached. This means the second-order error compensation and the finer granularity of grouping reinforce each other, which is what makes the extreme 2-bit and even ternary regimes usable. The final algorithm is one-shot, gradient-free, and requires only the input second moment accumulated from a small calibration set.

Concretely, each linear layer is wrapped by an accumulator object that keeps the running Hessian self.H, the layer's own weight self.W, and a hand to a separate quantizer object that already knows the target bit-width and grid: add_batch folds each calibration mini-batch's activations into a running, reweighted average of H so that batches seen at different times are combined correctly, and compress runs the block-wise OBS sweep just described — damping and Cholesky-factoring H once, then walking column blocks, quantizing each column through the quantizer, propagating its scaled error onto the rest of the block explicitly and onto the remaining blocks via one batched matrix multiply — before writing the quantized weights back into the layer:

```python
import torch

def add_batch(self, inp, out=None):
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    batch = inp.shape[0]
    inp = inp.reshape(-1, inp.shape[-1]).t().float()       # d_col x tokens
    self.H *= self.nsamples / (self.nsamples + batch)
    self.nsamples += batch
    inp *= (2.0 / self.nsamples) ** 0.5                    # scaled Hessian average
    self.H += inp.matmul(inp.t())

def compress(self, quantizer, blocksize=128, percdamp=0.01, groupsize=-1):
    W = self.W.clone()
    H = self.H.clone()
    if not quantizer.ready():
        quantizer.find_params(W, weight=True)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.d_col, device=H.device)
    H[diag, diag] += damp

    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    U = torch.linalg.cholesky(H, upper=True)               # scaled inverse row-tails

    Q = torch.zeros_like(W)
    Losses = torch.zeros_like(W)
    for i1 in range(0, self.d_col, blocksize):
        i2 = min(i1 + blocksize, self.d_col)
        W1, U1 = W[:, i1:i2].clone(), U[i1:i2, i1:i2]
        Q1, Err1 = torch.zeros_like(W1), torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            w = W1[:, i]; d = U1[i, i]
            if groupsize != -1 and (i1 + i) % groupsize == 0:
                quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * U1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        W[:, i2:] -= Err1.matmul(U[i1:i2, i2:])            # block GEMM update

    self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
    return torch.sum(Losses)
```
