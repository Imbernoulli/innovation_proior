# LLM.int8()

**Problem.** Run the feed-forward and attention-projection matrix multiplications of a pretrained Transformer, up to 175B parameters, in 8-bit integers with no retraining or post-quantization tuning. The obstacle appears around the 6.7B scale: ordinary Int8 quantization starts losing predictive quality just when extreme activation features become systematic.

**Why ordinary Int8 fails.** The hidden state is $\mathbf X\in\mathbb R^{s\times h}$: rows are sequence positions, columns are feature dimensions. At scale, a tiny number of columns develop values up to about 20x larger than normal features. Around 6.7B parameters, those outliers appear in all layers and in about 75% of sequence positions, yet they are concentrated in about 6 dimensions. They are not removable noise: zeroing them cuts mean top-1 attention probability from about 40% to about 20% and inflates validation perplexity by 600-1000%, while zeroing the same number of random dimensions has negligible effect. A shared activation scale, even a row scale, must cover the huge value and the normal values in the same token row, so the normal values collapse into too few Int8 bins.

**Vector-wise quantization.** Treat $\mathbf X\mathbf W$ as row-column inner products. For $\mathbf X\in\mathbb R^{s\times h}$ and $\mathbf W\in\mathbb R^{h\times o}$, use one absmax dequantization scale per row of $\mathbf X$, $\mathbf c_x\in\mathbb R^s$, and one per column of $\mathbf W$, $\mathbf c_w\in\mathbb R^o$. The Int8 product accumulates in Int32 and is dequantized by the outer product:
$$\mathbf C_{f16}\approx \mathbf C_{i32}\odot(\mathbf c_x\otimes\mathbf c_w).$$
This gives every output inner product its own scale pair while preserving a single GEMM.

**Mixed-precision decomposition.** Let
$$O=\{j:\max_i|\mathbf X_{ij}|\ge \alpha\},\qquad \alpha=6.0.$$
Split the contraction dimension. Keep the outlier feature dimensions in FP16, quantize the remaining dimensions with row/column vector-wise scales, and add the partial sums:
$$\mathbf C_{f16}\approx \mathbf X_{f16}^{[:,O]}\mathbf W_{f16}^{[O,:]}+\left(\mathbf X_{i8}^{[:,\bar O]}\mathbf W_{i8}^{[\bar O,:]}\right)\odot(\mathbf c_x\otimes\mathbf c_w).$$
The recombination is exact as a decomposition of the contraction sum; the only approximation is in the non-outlier Int8 part. Since $|O|$ is at most about 7 in the analyzed models, more than 99.9% of values stay in Int8 and the FP16 side product is tiny. Once the one-sided outliers are removed from the Int8 path, symmetric absmax is sufficient and the zeropoint advantage disappears.

```python
import torch

class Int8Linear(torch.nn.Module):
    def __init__(self, weight_f16, threshold=6.0):
        super().__init__()
        self.threshold = threshold
        W = weight_f16.detach().to(torch.float16)                 # h x o

        c_w = W.float().abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127.0
        W_i8 = torch.clamp(torch.round(W.float() / c_w), -127, 127).to(torch.int8)
        self.register_buffer("W_i8", W_i8)
        self.register_buffer("c_w", c_w)

        # FP16 rows for the exact outlier side product.
        self.W_f16_rows = W.cpu()

    def _weight_rows_f16(self, idx, device):
        return self.W_f16_rows.index_select(0, idx.cpu()).to(device=device, dtype=torch.float16)

    def forward(self, x_f16):                                      # s x h
        x = x_f16.to(torch.float16)
        idx = (x.abs().amax(dim=0) >= self.threshold).nonzero().flatten()

        x_reg = x.clone()
        x_reg[:, idx] = 0
        c_x = x_reg.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        x_i8 = torch.clamp(torch.round(x_reg.float() / c_x), -127, 127).to(torch.int8)

        acc_i32 = torch.matmul(x_i8.to(torch.int32), self.W_i8.to(torch.int32))
        out = acc_i32.float() * c_x * self.c_w                     # row scale x column scale

        if idx.numel() > 0:
            W_out = self._weight_rows_f16(idx, x.device)
            side = torch.matmul(x[:, idx], W_out).float()
            out = out + side
        return out.to(x_f16.dtype)
```

Only the parameter-heavy linear layers are quantized; the parameter-free attention operation remains 16-bit. The thresholded FP16 side path protects the emergent outlier dimensions, and the row/column Int8 path carries the rest of the matrix multiplication.
