The problem is how to compress the weights of a large pretrained Transformer down to 3 or 4 bits in a single pass, with no retraining and only a small calibration set, while keeping language-modeling perplexity essentially intact. The naive approach is round-to-nearest, which is fast but ignores how weights interact through the layer output; at 8 bits it is acceptable, but at 3 bits the accumulated error makes perplexity explode. More accurate second-order methods such as OBQ exist, but they maintain a separate inverse-Hessian trajectory for every row of the weight matrix, giving cubic runtime that tops out around a hundred million parameters. So the real gap is between scalable-but-crude and accurate-but-slow quantization.

The way forward is to preserve the layer's output rather than the weights themselves. For a linear layer with weights W and calibration inputs stacked as columns of X, the right objective is to minimize the reconstruction error ||WX - W_hat X||_2^2. This objective decomposes by output row, and every row shares the same curvature because the Hessian is H = 2 X X^T, which depends only on the inputs. The crucial empirical observation is that greedy ordering of which weight to quantize next barely helps on the heavily over-parameterized layers found in large Transformers. If all rows are quantized in the same fixed left-to-right order, one shared inverse can serve every row, and the per-row cubic cost disappears.

The method is GPTQ. It takes the Optimal Brain Surgeon update and restructures it so it scales to models with hundreds of billions of parameters. The core loop processes columns in fixed order: quantize the current column, compute its per-row residual, and push that residual onto the still-unquantized columns using the inverse Hessian. Because the order is fixed and shared, the inverse only needs to be downdated once per column rather than once per weight. To avoid bandwidth-bound rank-one updates, columns are processed in blocks, typically of size 128. Within a block the compensation is kept local; after the block is finished, the accumulated error is applied to all remaining columns in a single GEMM, which is efficient on a GPU. For numerical stability the repeated explicit downdates are replaced by a single upper Cholesky factor of H^{-1}. The OBS update only needs the scaled rightward tail of each inverse row, and those tails are exactly the rows of the Cholesky factor. A small amount of dampening, around 1% of the mean diagonal of H, is added before inversion to keep the matrix well-conditioned.

GPTQ is also grid-agnostic, so it composes cleanly with per-group scaling. When grouping is enabled, each group's scale is recomputed from the already-compensated weights at the moment that group is reached. This means the second-order error compensation and the finer granularity of grouping reinforce each other, which is what makes the extreme 2-bit and even ternary regimes usable. The final algorithm is one-shot, gradient-free, and requires only the input second moment accumulated from a small calibration set.

```python
import torch

class LayerQuantizer:
    BLOCK_SIZE = 128
    PERCDAMP = 0.01

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        self.H = torch.zeros((self.in_features, self.in_features),
                             device=self.dev, dtype=torch.float32)

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.float()
        self.H += inp.T @ inp
        self.nsamples += n

    def quantize(self):
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()
        if self.nsamples > 0:
            H /= self.nsamples

        num_bits = self.num_bits
        group_size = self.group_size
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1

        # Dampen and invert H once.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = self.PERCDAMP * torch.mean(torch.diag(H))
        H += damp * torch.eye(self.in_features, device=self.dev)

        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)

        Q = torch.zeros_like(W)
        Err = torch.zeros_like(W)

        for col_start in range(0, self.in_features, self.BLOCK_SIZE):
            col_end = min(col_start + self.BLOCK_SIZE, self.in_features)
            W_block = W[:, col_start:col_end].clone()
            U_block = U[col_start:col_end, col_start:col_end]

            for j in range(col_end - col_start):
                col = col_start + j
                w = W_block[:, j]
                d = U_block[j, j]

                if group_size > 0 and col % group_size == 0:
                    g_end = min(col + group_size, self.in_features)
                    W_group = W[:, col:g_end]
                    g_max = W_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                    group_scale = (g_max / qmax).squeeze(1)

                scale = group_scale if group_size > 0 else w.abs().max().clamp(min=1e-12) / qmax
                q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
                Q[:, col] = q

                err = (w - q) / d
                Err[:, col] = err
                W_block[:, j:] -= err.unsqueeze(1) * U_block[j, j:].unsqueeze(0)

            if col_end < self.in_features:
                W[:, col_end:] -= Err[:, col_start:col_end] @ U[col_start:col_end, col_end:]

        return Q.to(self.layer.weight.dtype)

    def free(self):
        del self.H
        self.H = None
```
