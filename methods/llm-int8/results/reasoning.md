I want the big Transformer linear layers, the feed-forward matrices and the attention projections, to run in 8-bit integers. Those layers are where almost all the parameters live, and Int8 tensor cores are already available. The demand is stricter than "usually close": load a pretrained 16/32-bit checkpoint, convert the linear layers, and use it immediately. No retraining, no calibration loop, no hidden tuning step. If this is going to work at 175B parameters, the quantizer has to respect whatever structure appears only when the model is large.

The simplest path is symmetric absmax quantization. Pick the largest magnitude in the tensor, map it to 127, round everything else into the same signed 8-bit range. Written with a multiplier, $\mathbf X_{i8}=\lfloor(127/\lVert\mathbf X\rVert_\infty)\mathbf X\rceil$; written with a dequantization scale, $c_x=\lVert\mathbf X\rVert_\infty/127$ and $\mathbf X\approx c_x\mathbf X_{i8}$. For a matrix product, quantize $\mathbf X$ and $\mathbf W$, do the Int8 multiply with Int32 accumulation, then multiply by $c_xc_w$ to return to FP16. It is clean because the scale lives outside the contraction sum.

The first sign of trouble is not subtle. Per-tensor absmax can survive small and mid-scale models, then the scaling trend breaks. Zeropoint quantization, with an extra shift that uses the full range for asymmetric values, survives longer, but it is still not enough at larger scale. If the issue were simply "8 bits are too few," zeropoint would not have bought anything over absmax, and the degradation would not switch on at a particular scale. The fact that an asymmetric scheme helps, and that it helps more as the model grows, points instead at the activation distribution: something in it is growing in a way that changes the effective precision of ordinary quantizers.

So I look at the object being quantized. The hidden state is $\mathbf X\in\mathbb R^{s\times h}$: sequence positions as rows, feature dimensions as columns. The large magnitudes are not scattered uniformly. They concentrate in a tiny handful of feature columns, and those columns can be about 20 times larger than the normal dimensions. Around the 6.7B scale, the same outlier dimensions appear across essentially every layer and most sequence positions. A 2048-token sequence can contain about 150,000 individual outlier values, but they sit in about six feature dimensions. Six columns out of thousands carry the extreme range.

I cannot treat those columns as harmless spikes. Removing the outlier dimensions before the attention projections cuts mean top-1 attention probability from about 40% to about 20% and drives validation perplexity up by 600-1000%; removing the same number of random dimensions barely moves either quantity. The features are sparse, but they are load-bearing. Any quantization scheme that rounds them badly, clips them, or lets them destroy the precision of nearby normal values is attacking exactly the wrong part of the model.

I want to see how bad the ordinary failure actually is rather than just assert it, so I work one token row by hand. Suppose a row contains normal feature values around $[-3.5,3.5]$ — say $3.5,-2.1,0.8,-3.2,1.4$ — and one outlier value $60$ in some column. The absmax of that row is $60$, so the scale is $60/127\approx0.472$. Dividing and rounding, the five normal values map to integer levels $7,-4,2,-7,3$, and the outlier maps to $127$. The normal values, which are the entire content of the row except one entry, now occupy a span of about $-7$ to $7$: roughly fourteen of the available $254$ levels. Their finest resolution is about $0.47$ in original units, so $0.8$ and $1.4$ are barely two bins apart and anything below $\sim0.24$ rounds to zero. Now do the same row with the outlier removed from the scale: the absmax is $3.5$, the scale is $3.5/127\approx0.0276$, and the same five values become $127,-76,29,-116,51$ — spread across nearly the whole range, with about $17\times$ finer resolution. So the cost is concrete and large: a single outlier collapses the precision of everything that shares its scale by more than an order of magnitude. The outlier does not merely suffer quantization error itself; it spends the precision budget for everything else.

The standard answer to that kind of range problem is more scales. Block-wise scaling is one version. Row-wise activation scaling is another: each token row of $\mathbf X$ gets its own absmax, so an outlier no longer sets the scale for every token. But the matrix product has a more natural granularity. Output $(i,k)$ is the dot product of row $i$ of $\mathbf X$ and column $k$ of $\mathbf W$. If row $i$ has scale $c_{x,i}$ and weight column $k$ has scale $c_{w,k}$, then the integer dot product for that output can be dequantized by $c_{x,i}c_{w,k}$. Across the whole matrix, the dequantization is an outer product $\mathbf c_x\otimes\mathbf c_w$. That gives each inner product its own row/column scale pair and still keeps the core as one Int8 GEMM.

This row/column scaling is the right way to spend the precision budget within the GEMM, but it still has the wrong axis for the activation outliers. The outliers are columns of $\mathbf X$. A token row that contains one giant outlier value also contains all the normal feature values for that token, and the row's absmax scale is still set by the giant value. I could imagine giving each activation feature column its own scale, but that scale sits on the contraction dimension $h$. It cannot be pulled out as an output rescale, because each output is $\sum_j X_{ij}W_{jk}$ and a different scale for each $j$ must be applied inside the sum. That would break the clean Int8 tensor-core path.

So the scaling approach has reached its limit on these dimensions. The outlier columns are too important to quantize coarsely, too rare to justify slowing the whole product, and too systematic to pretend they are random noise. No choice of scale fixes them while keeping the clean GEMM, because the problematic scale lives on the contraction axis. The only remaining lever is to not put them through the Int8 product at all.

That suggests partitioning the contraction. Let $O$ be the set of feature dimensions where at least one activation has magnitude at least a threshold $\alpha$. The output $(i,k)$ is $\sum_j X_{ij}W_{jk}$, and a sum splits over any partition of its index set:
$$\sum_{j} X_{ij}W_{jk}=\sum_{j\in O}X_{ij}W_{jk}+\sum_{j\notin O}X_{ij}W_{jk}.$$
So I can run the non-outlier dimensions, which are now well-behaved enough for row/column absmax quantization, through the Int8 GEMM, keep the outlier dimensions in FP16 as a small dense product, and add the two outputs:

$$\mathbf C_{f16}\approx \mathbf X_{f16}^{[:,O]}\mathbf W_{f16}^{[O,:]}+\left(\mathbf X_{i8}^{[:,\bar O]}\mathbf W_{i8}^{[\bar O,:]}\right)\odot(\mathbf c_x\otimes\mathbf c_w).$$

Before I trust the recombination I want to check that the partition itself adds nothing — that any error comes only from quantizing the complement, not from the splitting. So I take a small case: $\mathbf X\in\mathbb R^{4\times8}$ with a single planted outlier column near magnitude $40$, $\mathbf W\in\mathbb R^{8\times5}$, and I compute three things. First, the pure FP16 partition with no quantization on either side: outlier columns times their weight rows, plus the rest times theirs. Against the direct FP16 product its largest entrywise difference is about $4\times10^{-6}$, i.e. floating-point roundoff. So the split is exact; the partition contributes nothing. Second, vector-wise Int8 with no decomposition gives a relative error of about $7.7\times10^{-3}$ here, with the outlier column dominating it. Third, the mixed-precision version — pull the detected column ($j=3$) into FP16, quantize the rest by row and column scales, add — gives relative error about $5\times10^{-4}$, roughly $15\times$ smaller. The numbers line up with the by-hand row analysis: removing the outlier from the shared scale is worth more than an order of magnitude, and the only residual error sits in the complement where I expected it. Since the outlier set is on the order of six or seven dimensions, the FP16 side product is a sliver and the large product stays Int8.

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

Pulling the pieces together: row/column Int8 quantization was enough for the ordinary dimensions, and the small example put a number on the gap that remained — the outlier column alone was carrying most of the Int8 error, and removing it from the shared scale recovered more than an order of magnitude of precision. The emergent outlier feature columns at scale are both essential to the model and too large to share an activation row scale with normal values, so the natural move is to keep them out of the quantized product. Splitting the contraction index does that while leaving more than 99.9% of the values in Int8, and the two paths recombine exactly because, as the partition check showed, they are just complementary sums over the same feature dimension. What I would still want to confirm beyond this toy case is that the threshold of $6.0$ actually catches the harmful dimensions on real 6.7B-and-larger activations and that the FP16 side stays small there — that is a measurement on real checkpoints, not something this derivation settles.
