The goal is concrete and demanding: take a pretrained 16/32-bit Transformer, up to 175B parameters, and run its parameter-heavy linear layers — the feed-forward matrices and the attention projections, which hold roughly 95% of the weights and most of the compute — in 8-bit integers on Int8 tensor cores, immediately, with no retraining, no calibration loop, and no hidden post-quantization tuning, and with no loss of predictive quality. The natural first tool is symmetric absmax quantization: pick the largest magnitude in a tensor, map it to 127, round everything else into the signed range $[-127,127]$, so with dequantization scale $c_x=\lVert\mathbf X_{f16}\rVert_\infty/127$ we have $\mathbf X_{i8}=\lfloor\mathbf X_{f16}/c_x\rceil$ and $\mathbf X\approx c_x\mathbf X_{i8}$; the product becomes an Int8 GEMM with Int32 accumulation, rescaled by $c_xc_w$ afterward, the scale conveniently living outside the contraction sum. This survives small and mid-scale models and then breaks. Zeropoint quantization, which adds an integer shift so a one-sided distribution can use the full range, survives a little longer but still fails at scale, and its extra cross-terms $(A_{i8}+zp_a)(B_{i8}+zp_b)$ are awkward on GPUs. The fact that two different ordinary quantizers both degrade at the same scale tells me the problem is not that 8 bits are simply too few; something in the activation distribution is changing as the model grows.

Looking at what is being quantized makes the cause plain. The hidden state $\mathbf X\in\mathbb R^{s\times h}$ has sequence positions as rows and feature dimensions as columns, and the large magnitudes are not scattered — they concentrate in a tiny set of feature columns that can be about 20x larger than ordinary dimensions. Around the 6.7B-parameter scale these emergent outlier features become systematic: a 2048-token sequence can carry about 150,000 individual outlier values, yet they sit in only about six feature dimensions, and the same dimensions light up across essentially every layer and most sequence positions (the fraction of layers carrying them jumps from about 65% to 100%, and affected positions from about 35% to about 75%). They are not removable noise. Zeroing the outlier dimensions cuts mean top-1 attention softmax probability from about 40% to about 20% and inflates validation perplexity by 600–1000%, whereas zeroing the same number of random dimensions moves top-1 probability by at most about 0.3% and perplexity by about 0.1%. The outliers are sparse but load-bearing. And the failure is now mechanical: if one outlier column reaches magnitude 60 while the normal features live near $[-3.5,3.5]$, a single tensor scale maps 60 to 127 and crushes the 99.9% of normal values into a handful of integer levels. The outlier does not merely suffer its own quantization error; it spends the precision budget for everything else.

The standard cure for a range problem is more scales, and the right granularity for a matmul is the inner product itself. Output $(i,k)$ of $\mathbf X\mathbf W$ is the dot product of row $i$ of $\mathbf X$ with column $k$ of $\mathbf W$, so I give each activation row its own absmax scale $c_{x,i}$ and each weight column its own $c_{w,k}$; the integer product for that output dequantizes by $c_{x,i}c_{w,k}$, and across the whole matrix the dequantization is the outer product $\mathbf c_x\otimes\mathbf c_w$ — I call this vector-wise quantization, and it keeps the core as one Int8 GEMM,
$$\mathbf C_{f16}\approx \mathbf C_{i32}\odot(\mathbf c_x\otimes\mathbf c_w),\qquad \mathbf c_x\in\mathbb R^s,\ \mathbf c_w\in\mathbb R^o.$$
This is the correct way to spend the precision budget within the product, but it still has the wrong axis for the activation outliers. The outliers are columns of $\mathbf X$; a token row containing one giant value also contains that token's normal features, and the row's absmax scale is still set by the giant value. The obvious fix — give each activation feature column its own scale — fails because that scale would sit on the contraction dimension $h$. Each output is $\sum_j X_{ij}W_{jk}$, and a per-$j$ scale must be applied inside the sum, so it cannot be pulled out as an output rescale, and the clean tensor-core path is destroyed. The scaling approach has reached its limit: the outlier dimensions are too important to quantize coarsely, too rare to justify slowing the whole product, and too systematic to treat as noise. I do not need a better scale for them — I can take them out of the Int8 product entirely.

I propose LLM.int8(): vector-wise quantization for the bulk of the matrix combined with a mixed-precision decomposition that keeps the emergent outlier features in FP16. Let
$$O=\{j:\max_i|\mathbf X_{ij}|\ge \alpha\},\qquad \alpha=6.0,$$
be the set of feature dimensions in which some activation crosses the threshold. I split the contraction sum along the feature index into $O$ and its complement $\bar O$: the non-outlier dimensions are now well-behaved enough for row/column absmax and go through the Int8 GEMM, while the handful of outlier dimensions stay in FP16 as a small dense product, and the two partial outputs are added,
$$\mathbf C_{f16}\approx \mathbf X_{f16}^{[:,O]}\mathbf W_{f16}^{[O,:]}+\left(\mathbf X_{i8}^{[:,\bar O]}\mathbf W_{i8}^{[\bar O,:]}\right)\odot(\mathbf c_x\otimes\mathbf c_w).$$
The load-bearing point is that the split is not itself an approximation: matrix multiplication is a sum over the feature index, so partitioning that index and adding the partial sums reproduces the exact same contraction. The only approximation lives in the regular complement, where the outliers no longer set the scale — and because $|O|$ is at most about seven in the analyzed models, more than 99.9% of the values stay in Int8 and the FP16 side product is a sliver. The threshold $\alpha=6.0$ is the chosen knob: low enough to catch the dimensions whose values reach the harmful magnitude $\ge 6.0$, high enough that ordinary dimensions are not needlessly diverted into the expensive side path. (For characterizing the phenomenon across models I also use recurrence criteria — same feature dimension, magnitude at least 6, present in at least 25% of layers and at least 6% of sequence positions — but the operational inference rule is just the magnitude crossing.) This also explains why zeropoint helped earlier: the outlier columns are usually one-sided, so symmetric absmax wasted half its representable range on the unused sign, and zeropoint's shift used the whole range. Once the outliers leave the Int8 path the remaining activations are nearly symmetric, ordinary absmax suffices, and that zeropoint advantage disappears. I keep the per-column weight scale regardless, because the weight columns need their own ranges just as the activation rows do, or the output columns inherit unnecessary error from a shared weight scale. Concretely the layer stores the weight in Int8 with one scale per output column; on each forward pass it finds the activation columns crossing 6.0, zeros those columns before quantizing the activation rows, runs the regular product in Int8 and dequantizes by the row and column scales, runs the selected activation columns against the matching weight rows in FP16, and adds the two sums. Only the parameter-heavy linear layers are quantized; the parameter-free attention operation stays in 16-bit.

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
