I want to run the big linear layers of a Transformer — the feed-forward and attention projections, which are where almost all the parameters and most of the compute live — in 8-bit integers, and I want zero loss in predictive quality, at 175B parameters, with no retraining. 8-bit halves the memory and Int8 tensor cores exist on the GPUs I have. The catch: every prior 8-bit Transformer result either degraded accuracy or needed extra tuning, and none went past 350M parameters. So before I trust anything I should actually push the simplest quantizers up the size curve and watch where and why they break.

Start with the standard tool: absmax. Pick the largest magnitude in the tensor, scale so it maps to 127, round. $\mathbf X_{i8} = \lfloor (127/\lVert\mathbf X\rVert_\infty)\,\mathbf X\rceil$. One scale per tensor. For the matmul $\mathbf X\mathbf W$, quantize both, do the Int8 product accumulating in Int32, and divide out by $c_x c_w$ to come back to FP16. Clean and fast.

I run this across model sizes on a perplexity benchmark. Up to ~2.7B it's fine. Then it starts diverging from FP16, and around the 6–7B mark it falls off a cliff — perplexity wrecked. Per-tensor zeropoint quantization, which uses the full $[-127,127]$ range by also fitting a shift, does better and survives a bit longer, but by 13B even that is failing. Whatever is going wrong gets worse with scale, fast. I need to know what.

So I go look at the hidden states themselves. I take the activation matrix $\mathbf X\in\mathbb R^{s\times h}$ — rows are token positions, columns are feature dimensions — and ask where the large magnitudes are. They are not scattered. They sit in a tiny handful of *feature dimensions* (columns), and within those dimensions the magnitude is huge — order 20× bigger than anything in the other columns. And the same dimensions are the offenders across nearly every layer. In a 6.7B model I count something like 150,000 individual outlier values per 2048-token sequence, but when I ask which *dimensions* they live in, it's about six. Six columns out of thousands carry essentially all the extreme mass, persistently, layer after layer.

Are these just noise I can ignore? I test it directly: zero out those ~6 feature dimensions before the attention projections and see what happens. The mean top-1 attention softmax probability drops from about 40% to about 20%, and validation perplexity blows up by 600–1000%. Then the control: zero out the *same number* of randomly chosen dimensions instead. Top-1 probability barely moves (a fraction of a percent), perplexity changes by ~0.1%. So these six dimensions, which are ~0.1% of the features, are doing a wildly disproportionate share of the model's work. They are not noise. They are load-bearing, and they must be preserved with high precision.

Now the failure makes sense. With a single scale per tensor, that one outlier column with magnitude ~60 sets the scale for the whole tensor. The scale becomes $127/60\approx 2$, so every *normal* value — which lives in roughly $[-3.5, 3.5]$ — gets mapped into only a couple of integer levels. Most of the 256 bins are empty; the small values that carry the bulk of the information get rounded to zero or to one of two levels. The information in 99.9% of the tensor is extinguished to protect the range of 0.1% of it. And I can see in the data that the median outlier magnitude jumps sharply right at the 6.7B phase shift — that jump is what tips the per-tensor scale past the point where normal values survive. That's the cliff.

The standard fix for "one outlier poisons the whole tensor" is more scales. Block-wise, or row-wise: give each row of $\mathbf X$ its own scale, so an outlier only poisons its own row. Let me think about whether row-wise saves me. A matmul $\mathbf X\mathbf W$ can be read as a grid of inner products: output $(i,k)$ is row $i$ of $\mathbf X$ dotted with column $k$ of $\mathbf W$. If I give each *row* of $\mathbf X$ its own scale $c_{x,i}$ and each *column* of $\mathbf W$ its own scale $c_{w,k}$, then I can quantize each independently, do the Int8 inner product, and recover the true value by dividing by $c_{x,i}c_{w,k}$. Over the whole matmul that's dequantizing the Int32 result by the *outer product* $\mathbf c_x \otimes \mathbf c_w$, with $\mathbf c_x\in\mathbb R^s$, $\mathbf c_w\in\mathbb R^o$. This is the finest scaling I can apply while still doing one clean Int8 GEMM — a separate normalization per inner product. Call it vector-wise quantization, and it does help: it pushes the no-degradation frontier out to ~2.7B and is strictly better than per-row alone because it also gives the weights per-column precision.

But it does not save me at 6.7B, and now I can see exactly why, and it's almost embarrassing. The outliers run along *columns* — feature dimensions of $\mathbf X$. Row-wise and vector-wise scaling assign a scale per *row* (per token). A given row still contains both the outlier column's giant value and all the normal columns' tiny values, sharing one scale. I've changed which axis gets its own scale, but the outlier and the values it's poisoning are still in the *same scaling group*, because they're in the same row. To actually isolate the outliers I'd want a per-*column* scale on the activations — but a per-column activation scale is exactly per-feature scaling along the inner/contraction dimension of the matmul, and that cannot be folded into the GEMM as a simple output rescaling: the scale would have to be applied *inside* the sum over $h$, which breaks the clean Int8 tensor-core path. So I can't simply scale the troublesome dimension away.

Let me stare at what I actually have. The outliers are (1) catastrophic if quantized, (2) extremely sparse — about 0.1% of feature dimensions, never more than ~7 columns even at 13B — and (3) systematic: they sit in the *same* columns across layers and across almost all sequence positions. Sparse, systematic, and known. I don't need to quantize them at all. I can pull them out.

Split the contraction dimension $h$ into two sets: $O$, the feature dimensions that contain any value above a magnitude threshold $\alpha$, and the rest. Do the matmul in two pieces. For the rest — 99.9% of dimensions, all well-behaved — quantize to Int8 (vector-wise) and run the fast Int8 GEMM. For the $O$ columns, keep them in full 16-bit and do a tiny 16-bit matmul. Add the two partial sums. Because matrix multiplication is just a sum over the contraction index, splitting the index set and summing the partial products is exactly equal to the full product — no approximation in how I combine them:

$$\mathbf C_{f16} \approx \sum_{h\in O}\mathbf X_{f16}^{h}\mathbf W_{f16}^{h}\ +\ \mathbf S_{f16}\cdot\!\!\sum_{h\notin O}\mathbf X_{i8}^{h}\mathbf W_{i8}^{h},$$

where the second sum is the Int8 accumulation dequantized by the vector-wise denormalization $\mathbf S_{f16}$, and the first is the exact 16-bit contribution of the outlier dimensions. The whole trick rests on the contraction being a sum: I get to partition it however I like and still land on the same matrix. And the cost is negligible — $|O|\le 7$ columns means the 16-bit side is a sliver, about 0.1% extra memory, while >99.9% of the multiply-adds run in 8-bit, so I keep nearly the full ~2× memory reduction (for BLOOM-176B, 1.96×).

What threshold $\alpha$? I want it low enough to catch everything that hurts and high enough not to drag normal dimensions into the expensive 16-bit path. Empirically, treating any dimension with a value of magnitude $\ge 6$ as an outlier is exactly where the mixed-precision decomposition stops degrading perplexity — push the threshold higher and some harmful dimensions leak into Int8 and quality drops; push it lower and I needlessly inflate the 16-bit set. So $\alpha = 6.0$. (To *detect* an outlier dimension robustly for the analysis I also require it to recur — a value $\ge 6$ in at least 25% of layers in the same dimension and across at least ~6% of sequence positions — which is what tells me it's a systematic feature dimension rather than a one-off spike. For the inference decomposition itself, the magnitude threshold on the current input is what matters.)

There's a loose end I noticed earlier: zeropoint quantization beat absmax in the scaling study. Now I understand it. The outliers are almost always *one-sided* — a given outlier column is essentially all-positive or all-negative. A symmetric absmax scale wastes half its range on the sign that never occurs; an asymmetric zeropoint shifts and uses the full $[-127,127]$, so it tolerates one-sided outliers better. That's why zeropoint limped further up the curve. But once I do mixed-precision decomposition, the outliers are gone from the Int8 path entirely — the remaining dimensions are well-behaved and roughly symmetric — so the asymmetric advantage evaporates and plain absmax vector-wise is enough. I don't need to pay for zeropoint's awkward fused-instruction matmul. I'll keep vector-wise absmax for the 8-bit part, because comparing row-wise vs vector-wise *after* decomposition still shows vector-wise ahead — the extra per-column weight precision genuinely matters for matching FP16.

So the method is the combination: vector-wise absmax quantization for the bulk, plus mixed-precision decomposition that routes the sparse, systematic outlier feature dimensions through an exact 16-bit matmul. Let me write the layer forward.

The causal chain: simple per-tensor 8-bit quantization fails at exactly 6.7B because emergent outlier features — a handful of columns ~20× larger than everything else, but causally essential — force a tensor scale that crushes all the normal values; finer per-row/vector scaling doesn't help because the outliers run along the contraction axis and stay grouped with the values they poison; so I split the matmul's contraction sum into a fast 8-bit part for the 99.9% well-behaved dimensions and an exact 16-bit part for the ~0.1% outlier dimensions (threshold 6.0), which is lossless to combine because the contraction is a sum, and recovers full FP16 accuracy at scale at ~0.1% memory cost.

```python
import torch

class Int8Linear(torch.nn.Module):
    def __init__(self, weight_f16, threshold=6.0):
        super().__init__()
        self.threshold = threshold
        # vector-wise: one absmax scale per WEIGHT output column (the o dimension);
        # outliers live in activations, so the weight quantizes cleanly column-wise.
        W = weight_f16.float()                       # h x o
        self.c_w = W.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127.0   # 1 x o
        self.W_i8 = torch.clamp(torch.round(W / self.c_w), -127, 127).to(torch.int8)
        self.W_f16 = weight_f16                       # kept for the 16-bit outlier path

    def forward(self, x_f16):                         # x_f16: s x h
        x = x_f16.float()
        # find outlier feature dimensions: columns with any value above threshold
        col_max = x.abs().amax(dim=0)                 # h
        outlier = col_max >= self.threshold           # ~0.1% of dims True
        idx = outlier.nonzero().flatten()

        # --- 8-bit path over the well-behaved 99.9% of dimensions ---
        x_reg = x.clone(); x_reg[:, idx] = 0          # drop outlier cols from int8 side
        c_x = x_reg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0    # s x 1 (per-row)
        x_i8 = torch.clamp(torch.round(x_reg / c_x), -127, 127).to(torch.int8)
        acc_i32 = torch.matmul(x_i8.to(torch.int32), self.W_i8.to(torch.int32))
        out = (c_x * acc_i32.float()) * self.c_w      # dequant by outer product c_x (x) c_w

        # --- exact 16-bit path over the sparse outlier dimensions ---
        if idx.numel() > 0:
            out = out + torch.matmul(x[:, idx], self.W_f16[idx, :].float())

        return out.to(x_f16.dtype)
```
