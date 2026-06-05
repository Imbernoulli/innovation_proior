# LLM.int8()

**Problem.** Run the feed-forward and attention-projection matrix multiplications of a pretrained Transformer (up to 175B parameters) in 8-bit integers — halving memory — with *zero* degradation in predictive quality, immediately, without retraining or post-quantization tuning. Simple per-tensor 8-bit quantization works up to ~2.7B but collapses around 6.7B.

**Why it fails.** As models scale, a tiny set of *feature dimensions* (columns of the hidden state) develop extreme magnitudes — up to ~20× larger than other dimensions, concentrated in ~6 of thousands of dimensions, persistent across layers and sequence positions. They emerge in a sharp phase shift right at 6.7B parameters. Zeroing these ~6 dimensions cuts mean top-1 attention probability from ~40% to ~20% and inflates perplexity by 600–1000%, while zeroing the same number of random dimensions changes almost nothing. A single per-tensor scale set by such an outlier crushes all normal values (which live in ~$[-3.5,3.5]$) into a couple of quantization bins, extinguishing their information. Per-row / vector-wise scaling does not help, because the outliers run along the contraction (feature) axis and remain grouped with the values they poison.

**Key idea — two parts.**

1. **Vector-wise quantization.** Treat the matmul $\mathbf X\mathbf W$ as a grid of inner products; give each row of $\mathbf X$ its own absmax scale $\mathbf c_x\in\mathbb R^s$ and each column of $\mathbf W$ its own scale $\mathbf c_w\in\mathbb R^o$. Run the Int8 GEMM (Int32 accumulate), then dequantize by the outer product $\mathbf c_x\otimes\mathbf c_w$.
2. **Mixed-precision decomposition (the core).** Let $O$ be the set of feature dimensions containing any value of magnitude $\ge\alpha$ (with $\alpha = 6.0$). Split the contraction sum: run the $\sim$99.9% well-behaved dimensions in Int8 (vector-wise), run the $\le 7$ outlier dimensions in an exact FP16 matmul, and add the two partial sums:
$$\mathbf C_{f16}\ \approx\ \sum_{h\in O}\mathbf X_{f16}^{h}\mathbf W_{f16}^{h}\ +\ \mathbf S_{f16}\cdot\!\!\sum_{h\notin O}\mathbf X_{i8}^{h}\mathbf W_{i8}^{h}.$$
Because matrix multiplication is a sum over the contraction index, partitioning the index set and re-adding is exact in how the pieces combine; the only loss is in the Int8 part, which now has no outliers. Cost: ~0.1% extra memory; >99.9% of the multiply-adds stay in 8-bit. Because the isolated outliers were one-sided (which is why asymmetric zeropoint quantization beat absmax pre-decomposition), once they are removed the remaining values are symmetric and plain absmax suffices.

```python
import torch

class Int8Linear(torch.nn.Module):
    def __init__(self, weight_f16, threshold=6.0):
        super().__init__()
        self.threshold = threshold
        W = weight_f16.float()                                       # h x o
        self.c_w = W.abs().amax(0, keepdim=True).clamp(min=1e-8) / 127.0   # 1 x o
        self.W_i8 = torch.clamp(torch.round(W / self.c_w), -127, 127).to(torch.int8)
        self.W_f16 = weight_f16                                      # for outlier path

    def forward(self, x_f16):                                        # s x h
        x = x_f16.float()
        outlier = x.abs().amax(0) >= self.threshold                 # per feature dim
        idx = outlier.nonzero().flatten()

        x_reg = x.clone(); x_reg[:, idx] = 0
        c_x = x_reg.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0   # s x 1
        x_i8 = torch.clamp(torch.round(x_reg / c_x), -127, 127).to(torch.int8)
        acc = torch.matmul(x_i8.to(torch.int32), self.W_i8.to(torch.int32))
        out = (c_x * acc.float()) * self.c_w                         # dequant by c_x (x) c_w

        if idx.numel() > 0:                                          # exact 16-bit outliers
            out = out + torch.matmul(x[:, idx], self.W_f16[idx, :].float())
        return out.to(x_f16.dtype)
```

Only the parameter-heavy linear layers are quantized; the (parameter-free) attention function stays in 16-bit. The result loads a 16/32-bit checkpoint, converts the linear weights to Int8, and serves up to 175B parameters at full accuracy on a fraction of the GPUs.
