I want the big Transformer linear layers, the feed-forward matrices and the attention projections, to run in 8-bit integers. Those layers are where almost all the parameters live, and Int8 tensor cores are already available. The demand is stricter than "usually close": load a pretrained 16/32-bit checkpoint, convert the linear layers, and use it immediately. No retraining, no calibration loop, no hidden tuning step. If this is going to work at 175B parameters, the quantizer has to respect whatever structure appears only when the model is large.

The simplest path is symmetric absmax quantization. Pick the largest magnitude in the tensor, map it to 127, round everything else into the same signed 8-bit range. Written with a multiplier, $\mathbf X_{i8}=\lfloor(127/\lVert\mathbf X\rVert_\infty)\mathbf X\rceil$; written with a dequantization scale, $c_x=\lVert\mathbf X\rVert_\infty/127$ and $\mathbf X\approx c_x\mathbf X_{i8}$. For a matrix product, quantize $\mathbf X$ and $\mathbf W$, do the Int8 multiply with Int32 accumulation, then multiply by $c_xc_w$ to return to FP16. It is clean because the scale lives outside the contraction sum.

The first sign of trouble is not subtle. Per-tensor absmax can survive small and mid-scale models, then the scaling trend breaks. Zeropoint quantization, with an extra shift that uses the full range for asymmetric values, survives longer, but it is still not enough at larger scale. That tells me the issue is not just "8 bits are too few." Something in the activation distribution is growing in a way that changes the effective precision of ordinary quantizers.

So I look at the object being quantized. The hidden state is $\mathbf X\in\mathbb R^{s\times h}$: sequence positions as rows, feature dimensions as columns. The large magnitudes are not scattered uniformly. They concentrate in a tiny handful of feature columns, and those columns can be about 20 times larger than the normal dimensions. Around the 6.7B scale, the same outlier dimensions appear across essentially every layer and most sequence positions. A 2048-token sequence can contain about 150,000 individual outlier values, but they sit in about six feature dimensions. Six columns out of thousands carry the extreme range.

I cannot treat those columns as harmless spikes. Removing the outlier dimensions before the attention projections cuts mean top-1 attention probability from about 40% to about 20% and drives validation perplexity up by 600-1000%; removing the same number of random dimensions barely moves either quantity. The features are sparse, but they are load-bearing. Any quantization scheme that rounds them badly, clips them, or lets them destroy the precision of nearby normal values is attacking exactly the wrong part of the model.

Now the ordinary failure is almost mechanical. Suppose one outlier column reaches magnitude 60 while the normal feature values mostly live around $[-3.5,3.5]$. A single tensor scale maps 60 to 127, so the normal values are squeezed into a narrow set of integer levels. Most of the available bins are unused, and the values that make up 99.9% of the tensor get rounded coarsely just to preserve the range of a tiny number of entries. The outlier does not merely suffer quantization error itself; it spends the precision budget for everything else.

The standard answer to that kind of range problem is more scales. Block-wise scaling is one version. Row-wise activation scaling is another: each token row of $\mathbf X$ gets its own absmax, so an outlier no longer sets the scale for every token. But the matrix product has a more natural granularity. Output $(i,k)$ is the dot product of row $i$ of $\mathbf X$ and column $k$ of $\mathbf W$. If row $i$ has scale $c_{x,i}$ and weight column $k$ has scale $c_{w,k}$, then the integer dot product for that output can be dequantized by $c_{x,i}c_{w,k}$. Across the whole matrix, the dequantization is an outer product $\mathbf c_x\otimes\mathbf c_w$. That gives each inner product its own row/column scale pair and still keeps the core as one Int8 GEMM.

This row/column scaling is the right way to spend the precision budget within the GEMM, but it still has the wrong axis for the activation outliers. The outliers are columns of $\mathbf X$. A token row that contains one giant outlier value also contains all the normal feature values for that token, and the row's absmax scale is still set by the giant value. I could imagine giving each activation feature column its own scale, but that scale sits on the contraction dimension $h$. It cannot be pulled out as an output rescale, because each output is $\sum_j X_{ij}W_{jk}$ and a different scale for each $j$ must be applied inside the sum. That would break the clean Int8 tensor-core path.

So the scaling approach has reached its limit. The outlier dimensions are too important to quantize coarsely, too rare to justify slowing the whole product, and too systematic to pretend they are random noise. I do not need a better scale for them. I can remove them from the Int8 product.

Let $O$ be the set of feature dimensions where at least one activation has magnitude at least a threshold $\alpha$. Split the contraction sum into $O$ and its complement. The non-outlier dimensions are now well-behaved enough for row/column absmax quantization, so they go through the Int8 GEMM. The outlier dimensions stay in FP16 and form a small dense product. Then I add the two outputs:

$$\mathbf C_{f16}\approx \mathbf X_{f16}^{[:,O]}\mathbf W_{f16}^{[O,:]}+\left(\mathbf X_{i8}^{[:,\bar O]}\mathbf W_{i8}^{[\bar O,:]}\right)\odot(\mathbf c_x\otimes\mathbf c_w).$$

The important point is that the split itself is not an approximation. Matrix multiplication is a sum over the feature index, so partitioning that index and adding the partial sums gives the same contraction structure. The approximation is only in the regular complement, where the outliers no longer set the scale. Since the outlier set is on the order of six or seven dimensions, the FP16 side product is a sliver and the large product stays Int8.

The threshold has to be low enough to catch the harmful feature dimensions and high enough not to turn normal dimensions into an expensive side path. The diagnostic criterion that matters is magnitude about 6.0: values at or above that level are the ones that must leave the Int8 path. For analyzing the phenomenon across models, I also want recurrence criteria, because a single spike is not the same thing as an emergent feature: same feature dimension, magnitude at least 6, present in at least 25% of layers and at least 6% of sequence positions. For the inference decomposition, the operational rule is simpler: if the current activation column crosses $\alpha=6.0$, that feature dimension goes to the FP16 side.

This also explains why zeropoint helped before. The outlier columns are usually one-sided, purely positive or purely negative across the sequence dimension. Symmetric absmax wastes half of its representable range on the unused sign. Zeropoint shifts the distribution and uses the whole signed range, so it handles those one-sided columns better. Once the outlier columns leave the Int8 path, that special advantage disappears. The remaining activation values are closer to symmetric and ordinary absmax is enough. I still keep the row/column scales, because the weight columns need their own ranges just as the activation rows do; otherwise the output columns inherit unnecessary quantization error from a shared weight scale.

The layer now has the shape I need: store the main weight matrix in Int8 with one scale per output column; on each forward pass, find activation feature columns whose magnitude crosses 6.0; zero those columns before quantizing the activation rows; run the regular product in Int8 and dequantize by row and column scales; run the selected activation columns and matching weight rows in FP16; add the two partial sums.

```python
import torch

class Int8Linear(torch.nn.Module):
    def __init__(self, weight_f16, threshold=6.0):
        super().__init__()
        self.threshold = threshold
        W = weight_f16.detach().to(torch.float16)                 # h x o

        # Weight side of the row/column scaling: one scale per output column.
        c_w = W.float().abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127.0
        W_i8 = torch.clamp(torch.round(W.float() / c_w), -127, 127).to(torch.int8)
        self.register_buffer("W_i8", W_i8)
        self.register_buffer("c_w", c_w)

        # FP16 rows for the exact outlier side path.
        self.W_f16_rows = W.cpu()

    def _weight_rows_f16(self, idx, device):
        return self.W_f16_rows.index_select(0, idx.cpu()).to(device=device, dtype=torch.float16)

    def forward(self, x_f16):                                      # x_f16: s x h
        x = x_f16.to(torch.float16)

        # Columns whose activations cross the outlier threshold leave the Int8 path.
        idx = (x.abs().amax(dim=0) >= self.threshold).nonzero().flatten()

        x_reg = x.clone()
        x_reg[:, idx] = 0

        # Activation side of the row/column scaling: one scale per token row.
        c_x = x_reg.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        x_i8 = torch.clamp(torch.round(x_reg.float() / c_x), -127, 127).to(torch.int8)

        acc_i32 = torch.matmul(x_i8.to(torch.int32), self.W_i8.to(torch.int32))
        out = acc_i32.float() * c_x * self.c_w

        if idx.numel() > 0:
            W_out = self._weight_rows_f16(idx, x.device)
            side = torch.matmul(x[:, idx], W_out).float()
            out = out + side

        return out.to(x_f16.dtype)
```

The reason this works is now narrow and concrete: row/column Int8 quantization is enough for the ordinary dimensions, but emergent outlier feature columns at scale are both essential and too large to share an activation row scale with normal values. Splitting the contraction index protects those columns in FP16 while leaving almost the entire matrix multiplication in Int8, and the two paths recombine because they are just complementary sums over the same feature dimension.
