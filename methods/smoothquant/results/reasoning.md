I want real inference speedups from quantization, which means W8A8 — both weights and activations in INT8 — so the linear layers actually run on the hardware INT8 GEMM units, not just smaller storage with FP16 math. So I quantize a few large linear layers to INT8 and look at the accuracy. The weights quantize beautifully: their distribution is flat, INT8 (even INT4) barely moves perplexity. The activations are a disaster. Something in the activations is roughly 100× larger than the typical value, and once that sits in the tensor it ruins everything else.

Let me make the damage precise rather than just assert it's bad. Per-tensor INT8 uses one step $\Delta = \max(|\mathbf X|)/(2^{N-1}-1) = m/127$, set by the single largest magnitude $m$ in the whole matrix. A normal channel whose own max magnitude is $m_i \ll m$ only ever occupies values up to $m_i$, so it uses about $127\cdot m_i/m$ of the available codes. Put numbers in: if $m_i/m = 1/100$ that's $127/100 \approx 1.3$ codes; even $m_i/m = 3/100$ is only $\approx 3.8$. So a normal channel is supposed to be represented by between one and four integer codes. To see what that actually does, take an outlier-dominated tensor with $m=100$ so $\Delta \approx 0.787$, and quantize a normal channel of eight values in $[-1,1]$:

```
normal channel values : [-0.166  0.441 -1.    -0.395 -0.706 -0.815 -0.627 -0.309]
quantized codes        : [   0      1    -1     -1     -1     -1     -1      0  ]
distinct codes used    : {-1, 0, 1}
dequantized            : [ 0.   0.787 -0.787 -0.787 -0.787 -0.787 -0.787  0. ]
```

The whole channel collapses onto $\{-1,0,1\}$. The value $-0.166$ rounds to $0$ — gone — and five genuinely different values ($-0.706, -0.815, -0.627, \dots$) all map to the single code $-1$ and dequantize to the identical $-0.787$. The channel is essentially erased. The outliers are spending the entire dynamic range and leaving nothing for the values that carry the information.

So where are the outliers? I look at the activation matrix $\mathbf X\in\mathbb R^{T\times C_i}$ — rows are tokens, columns are input channels. The big values are not scattered; they sit in a small set of *channels* (columns), and within those channels they're large for essentially every token. If I look at the variance, it's lopsided: for a single token, the variance *across* channels is huge (a few channels enormous, the rest small); for a single channel, the variance *across* tokens is small (an outlier channel is consistently large, a normal channel consistently small). The difficulty is organized by channel, and it's *static* in the sense that the same channels are the troublemakers regardless of which tokens flow through.

That points at the granularity. If I could give each input channel its own step $\Delta$ — per-channel activation quantization — then an outlier channel gets a big $\Delta$ all to itself and a normal channel gets a small $\Delta$, and nobody borrows anyone else's range. I check this against the diagnostic: simulating per-channel activation quantization closes the gap to FP16, while per-token barely helps. That ordering matches the structure I just measured — per-token keeps one shared scale across a token's whole row, so the outlier channel's giant value still co-occupies the range with all the normal channels, whereas per-channel is the one granularity aligned with where the difficulty actually lives. So the fix wants to be per-channel.

Then I hit the hardware wall. The hardware INT8 GEMM cannot apply a per-channel activation scale. The kernel is a tight march of tensor-core MMAs over the contraction dimension $C_i$, and it will not let me insert a different scale for each $C_i$ index *inside* that accumulation — that would mean a lower-throughput conversion in the hot loop and the kernel falls apart. Dequant scales can only be applied as an *epilogue*, after the integer sum finishes, and that restricts them to the matmul's *outer* dimensions: the token dimension $T$ of $\mathbf X$ and the output-channel dimension $C_o$ of $\mathbf W$. Concretely the kernel can do $\mathbf Y = \mathrm{diag}(\boldsymbol\Delta_{\mathbf X})\,(\bar{\mathbf X}\bar{\mathbf W})\,\mathrm{diag}(\boldsymbol\Delta_{\mathbf W})$ — a per-token activation scale on the left, a per-output-channel weight scale on the right. A per-token activation scale is fine. A per-*input*-channel activation scale lives on $C_i$, the contraction axis, exactly where I'm not allowed to put it. So the granularity that fixes accuracy is the one granularity the fast kernel forbids. That's the bind.

Let me sit with the actual asymmetry instead of fighting it. Weights are easy to quantize. Activations are hard, and the hardness lives in a few input channels. What if I don't try to *scale away* the outlier on the activation side at all, but instead *move the difficulty off the activations and onto the weights*, where there's room to spare? The weight matrix is multiplied against the same activation, channel by channel: output is $\mathbf Y = \mathbf X\mathbf W$, and in this mathematical layout $\mathbf W\in\mathbb R^{C_i\times C_o}$, so channel $j$ of $\mathbf X$ multiplies row $j$ of $\mathbf W$. If I shrink activation channel $j$ by some factor $s_j$ and simultaneously grow that weight row by the *same* $s_j$, the product is unchanged. That's just inserting $\mathrm{diag}(s)^{-1}\mathrm{diag}(s)$ between them:

$$\mathbf Y = \big(\mathbf X\,\mathrm{diag}(\mathbf s)^{-1}\big)\big(\mathrm{diag}(\mathbf s)\,\mathbf W\big) = \hat{\mathbf X}\hat{\mathbf W}.$$

I should make sure this really is identity-preserving and not just plausibly so, because the whole approach dies if the transform shifts the output. Take a small $\mathbf X$ ($4\times 5$) with a 100× outlier injected into channel 2, a random $\mathbf W$ ($5\times 3$), and a random positive $\mathbf s$, and compare $\mathbf X\mathbf W$ against $(\mathbf X\,\mathrm{diag}(\mathbf s)^{-1})(\mathrm{diag}(\mathbf s)\,\mathbf W)$: the max absolute difference comes out $1.4\times10^{-14}$, i.e. floating-point zero. Exact equivalence, no approximation. And critically, $\mathrm{diag}(\mathbf s)$ acts along $C_i$ — but it's a *constant* rebalancing of the two operands done *offline*, not a per-channel scale applied inside the GEMM. After I've rebalanced, $\hat{\mathbf X}$ and $\hat{\mathbf W}$ are quantized with ordinary hardware-friendly scales: per-token or per-tensor on activations, and per-tensor or output-channel epilogue scales on weights. The per-channel structure has been baked into the operands, so the runtime kernel never sees a contraction-axis scale.

Does the rebalancing even cost anything at runtime? The activation scaling $\hat{\mathbf X} = \mathbf X\,\mathrm{diag}(\mathbf s)^{-1}$ has to happen somewhere. But $\mathbf X$ is the output of whatever came before — a LayerNorm, or a previous linear. I can fold $\mathrm{diag}(\mathbf s)^{-1}$ into *that* layer's parameters: divide the LayerNorm's affine weight and bias, or divide the previous linear's output-channel weights and bias, by $\mathbf s$ offline. Then $\hat{\mathbf X}$ comes out of the previous layer already smoothed, at zero extra kernel launches. (When the input is a residual add rather than a clean prior layer, I add an explicit scaling on the residual branch.) The weight side $\hat{\mathbf W} = \mathrm{diag}(\mathbf s)\mathbf W$ is just a one-time offline reweighting.

Now the real question: how do I choose $\mathbf s$? I want the smoothed activation $\hat{\mathbf X}$ to be easy to quantize. Quantization difficulty per channel is set by that channel's max magnitude; per-tensor difficulty is set by the *largest* channel max. The dynamic range is best used — the most effective codes for everyone — when all channels have the *same* max magnitude, so no channel hogs the range. A first guess: to make activation channel $j$ have a uniform max, divide it by its own max, $s_j = \max(|\mathbf X_j|)$, so every $\hat{\mathbf X}$ channel has max 1, perfectly flat.

But I should check what that does to the weights before believing it, because $s_j$ is shared between the two operands. With the same $4\times5$ tensor from before, where channel 2 holds the 100× outlier, set $s_j = \max(|\mathbf X_j|)$ and look at the smoothed weight maxima per channel:

```
alpha=1 (s_j = max|X_j|)
  smoothed activation max per channel: [1, 1, 1, 1, 1]
  smoothed weight     max per channel: [4.50, 3.39, 150.02, 3.29, 3.70]
```

There it is. The activations went perfectly flat, but the weight outlier channel jumped to 150 against ~3–4 elsewhere — a ~40× spread that the formerly-flat weights now have to absorb, and they'll quantize as badly as the activations did. I moved the problem, I didn't solve it. The symmetric extreme is just as bad: pick $s_j = 1/\max(|\mathbf W_j|)$ to flatten the *weights* completely, and by the same arithmetic the activation maxima blow up by the outlier factor. The two extremes are mirror images — each sends one operand to all-ones and dumps the entire dynamic range onto the other. So neither endpoint works; the difficulty has to be *split*, with activations and weights *both* tolerable, neither perfectly flat nor catastrophic.

I want, per channel, the smoothed activation max and the smoothed weight max to be comparable — to share the difficulty rather than concentrate it. Introduce a knob $\alpha$ that controls how much difficulty migrates from activations to weights, and interpolate the two extremes in the exponent:

$$s_j = \frac{\max(|\mathbf X_j|)^{\alpha}}{\max(|\mathbf W_j|)^{1-\alpha}}.$$

The endpoints recover the two cases I just walked: at $\alpha=1$, $s_j = \max(|\mathbf X_j|)$ (full activation-flattening that overloaded the weights to 150); at $\alpha=0$, $s_j = 1/\max(|\mathbf W_j|)$ (full weight-flattening that overloads the activations). What happens in between is the question. The smoothed activation max of channel $j$ is $\max(|\mathbf X_j|)/s_j = \max(|\mathbf X_j|)^{1-\alpha}\max(|\mathbf W_j|)^{1-\alpha}$, and the smoothed weight max is $\max(|\mathbf W_j|)\cdot s_j = \max(|\mathbf X_j|)^{\alpha}\max(|\mathbf W_j|)^{\alpha}$. Setting these two equal forces $\max(|\mathbf X_j|)^{1-\alpha}\max(|\mathbf W_j|)^{1-\alpha} = \max(|\mathbf X_j|)^{\alpha}\max(|\mathbf W_j|)^{\alpha}$, which holds exactly at $\alpha=1/2$, where both sides become $\big(\max(|\mathbf X_j|)\max(|\mathbf W_j|)\big)^{1/2}$ — the geometric mean of the two maxima, shared equally.

I'd rather see this than trust the algebra, so I recompute the smoothed maxima on the same tensor at $\alpha=0.5$:

```
alpha=0.5 (s_j = sqrt(max|X_j| / max|W_j|))
  smoothed activation max per channel: [2.12, 1.84, 12.25, 1.81, 1.92]
  smoothed weight     max per channel: [2.12, 1.84, 12.25, 1.81, 1.92]
  geometric mean sqrt(max|X|*max|W|) : [2.12, 1.84, 12.25, 1.81, 1.92]
```

The activation and weight rows are now identical, channel by channel, and equal to the geometric mean — exactly the balance the algebra predicted. And note the outlier channel: it didn't vanish (its shared max is still 12.25 vs ~2 elsewhere), but it dropped from the activations' raw 98 and the $\alpha=1$ weights' 150 down to 12.25 on *both* sides. Neither operand is flat, but neither is catastrophic; that's the point. So $\alpha=0.5$ splits the difficulty evenly, and for the same-quantizer-on-both-sides case that's the natural choice. For most models (all the OPT and BLOOM sizes) $\alpha=0.5$ is the sweet spot. For a model whose activation outliers are unusually severe (e.g. GLM-130B, with a far larger outlier fraction), the balanced point isn't quite even — I'd want to push more difficulty onto the (still-easy) weights with a larger $\alpha$ like 0.75, and I'd confirm that by a small grid search rather than assume it.

The activation max per channel is input-dependent, so I can't read it off the weights — I estimate $\max(|\mathbf X_j|)$ over calibration samples drawn from the pre-training data. In the experiments this is 512 random Pile sentences, and the suitable $\alpha$ is picked with a quick grid search on a Pile validation subset. The weight max is exact. With $\alpha$ and both sets of statistics in hand, $\mathbf s$ is a closed-form per-channel vector, no gradients or iterative optimization loop.

That's the whole method. There is no backprop and no mixed FP16 outlier-decomposition path — just a per-channel rebalancing chosen by a single $\alpha$, folded into adjacent layers, after which a plain hardware INT8 GEMM runs both operands. I apply the offline smoothing where the input channel statistics are shared by a preceding normalization and one or more following linears — for example the attention LayerNorm feeding Q/K/V together, and the feed-forward LayerNorm feeding the first MLP projection — then all the linear layers are replaced by W8A8 linears. If the input comes from a residual add, I add explicit scaling on the residual branch. The attention batched matmuls go INT8 too, while Softmax, LayerNorm and residuals stay FP16 since they're cheap and not the bottleneck.

The causal chain: W8A8 is needed for real speedups, but activation outliers — concentrated in a few persistent input channels, ~100× the normal scale — destroy per-tensor activation quantization (a normal channel collapses to a couple of codes), and the per-channel scaling that would fix it can't go on the contraction axis of a hardware INT8 GEMM; so instead of applying per-channel scales inside the runtime GEMM, smooth activations by a per-channel equivalence transform and push the inverse factor into the weights; choose the factor as $\max(|\mathbf X_j|)^{\alpha}/\max(|\mathbf W_j|)^{1-\alpha}$ to *split* the difficulty between the (easy) weights and the (hard) activations — the extremes $\alpha\in\{0,1\}$ just flatten one operand and wreck the other, while $\alpha=0.5$ equalizes both at the geometric mean — and fold the activation factor into the preceding LayerNorm/linear when possible so smoothing adds no extra kernel.

```python
import torch

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # nn.Linear stores weights as [out_features, in_features], so dim=0
    # gives the per-input-channel max across output channels.
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)                       # shape C_i

    # split the difficulty: s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    s = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    # fold diag(s)^-1 into the previous LayerNorm
    ln.weight.div_(s)
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(s)
    # fold diag(s) into the following linears: (X diag(s)^-1)(diag(s) W) == X W
    for fc in fcs:
        fc.weight.mul_(s.view(1, -1))

# after smoothing each LayerNorm-fed group, linears and attention BMMs are quantized W8A8
# and run on hardware INT8 GEMM; Softmax / LayerNorm / residual stay FP16.
```
