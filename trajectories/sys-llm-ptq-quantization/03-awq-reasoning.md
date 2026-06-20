GPTQ closed most of the weight-only gap: 3-bit per-channel LLaMA-7B went from RTN's 25.54 down to 8.07,
against FP16's 5.68. That is a real win, and it makes me want to push harder on the low-bit weight-only
regime, because that is the one that actually speeds up batch-1 generation — where the bottleneck is
reading weights from DRAM, FLOPs are nearly free, and quantizing weights to 3–4 bits raises the
arithmetic intensity several-fold. But two things about GPTQ bother me when I try to go lower still.

First, it leans on a full second-order reconstruction: a Hessian per layer, a Cholesky inverse,
column-by-column error feedback. That is heavy machinery, and because it fits the quantized weights to
the calibration set's input statistics, I worry it can *overfit* to the calibration distribution — it is
regressing weights against a particular few hundred sequences. Second, and more fundamentally, GPTQ
treats every weight as equally deserving of accuracy. It minimizes the total output error, but it has
no notion that *some weights matter far more than others*. I want to interrogate that, because if a tiny
fraction of weights carries most of the model's quality, then spending the bit budget uniformly is
wasteful, and protecting exactly that fraction might beat reconstructing everything.

So let me run the diagnostic. Suppose I keep a small fraction of weight channels — say 0.1% to 1% — in
full FP16 and quantize all the rest to INT3. Which channels should I keep? The obvious guess is the
channels with the largest weight magnitude. I try it: keeping the top-magnitude channels in FP16 barely
helps. That is a surprise, and it is informative. The magnitude of a weight is *not* what makes it
important. Then I try selecting the channels by the *activations* that flow through them — keep in FP16
the input channels whose activations have the largest average magnitude — and this recovers almost all
of the FP16 accuracy from an otherwise-INT3 model. So the salient channels are activation-defined, not
weight-defined. That makes sense in hindsight: a weight only matters in proportion to how large the
input it multiplies tends to be; a big weight on a channel that is always near zero does nothing, and a
modest weight on a high-magnitude activation channel dominates the output.

Now I have the right notion of saliency, but the obvious implementation is a dead end. Keeping 1% of
channels in FP16 and the rest in INT3 means storing a *mixed-precision* tensor — a ragged layout with
some FP16 columns interleaved among INT3 columns. That is a nightmare for a GPU kernel; the whole point
of quantization was a clean, regular, hardware-friendly weight matrix, and a mixed FP16/INT layout
throws that away. I need the *protection* the FP16 channels gave, without the irregular layout.

Here is the move. I do not have to store the salient channels in higher precision; I can *scale them up*
before quantizing so that, on the integer grid, they get more effective resolution, and then divide the
corresponding activations down by the same factor so the layer's function is unchanged. For a linear
layer this is an exact equivalence transform:

  W X = (W · diag(s)) · (diag(s)⁻¹ X).

Multiply a weight channel by s and divide its input channel by s; the product WX is identical before
any quantization. The payoff is what happens *after* quantization. With a round-to-nearest quantizer
Q(w) = Δ·round(w/Δ) and step Δ = max(|w|)/2^(N−1), scaling a salient channel by s > 1 changes its
quantized contribution from Q(w)·x to Q(ws)·(x/s), and the relative error works out to

  Err(Q(ws)(x/s)) / Err(Q(w)x) = (Δ'/Δ) · (1/s).

If s is moderate enough that the scaled-up salient weight does not become the new group maximum, then
Δ' ≈ Δ, and the salient channel's quantization error shrinks by about 1/s. I have protected the
important channel — given it effectively finer resolution — using nothing but a per-channel scale that
folds away at inference. No mixed precision, no ragged layout: the stored tensor is still a plain
group-wise INT3/INT4 matrix.

The catch is the same Δ' in that ratio. If I scale a salient channel too hard, the scaled weight *does*
become the group maximum, Δ' grows, and now every *other* weight in that group rounds against a coarser
grid — I have protected one channel by hurting its neighbors. So there is a sweet spot, and I should not
try to derive it analytically. Instead: the right scales should track activation magnitude, so I
parametrize s as the per-input-channel average activation magnitude raised to a power, s = s_X^α, and
*search* α over a small grid, directly minimizing the layer's output MSE against the FP16 reference. No
gradients, no Hessian, no per-weight regression against the calibration set — just a handful of forward
passes choosing one scalar α per layer. At α = 0 there is no scaling (back to RTN); as α grows the
salient channels get more protection until the group-max penalty starts to bite; the grid search finds
the balance.

```python
x_scale = get_act_scale(x)                 # per-input-channel mean |activation| over calibration
best_loss, best_scales = float("inf"), None
for grid in range(n_grid):                 # search the exponent α
    alpha = grid / n_grid
    scales = x_scale.pow(alpha).clamp(min=1e-4)
    scales = scales / (scales.max() * scales.min()).sqrt()
    for fc in linears:                     # W · diag(s), quantize, divide back by diag(s)
        fc.weight.mul_(scales)
        fc.weight.data = pseudo_quantize_tensor(fc.weight.data, n_bit, q_group_size) / scales
    loss = (org_out - module(x)).float().pow(2).mean().item()   # output MSE vs FP16
    if loss < best_loss:
        best_loss, best_scales = loss, scales
```

Then the chosen scale is *folded into the surrounding ops* — divided into the preceding LayerNorm or
previous linear's output, multiplied into the current linear's weights — so the network function is
exactly preserved and there is zero runtime overhead. The stored model is an ordinary group-wise INT3 or
INT4 weight-only model with FP16 activations. The activation statistic decided which channels deserved
resolution; the equivalence transform delivered it without an irregular layout. This is **activation-
aware weight quantization**: protect by scaling, not by storing.

The bar and the bet. The honest comparison is at the *same* grouping, because g128 numbers and per-
channel numbers are not interchangeable, and AWQ is naturally a group-wise method. So I pin it against
RTN-g128: on Llama-2-7B, RTN at INT3-g128 is 6.66. My claim is that one searched per-channel scale per
layer — pure forward passes, no second-order machinery, no risk of regressing onto the calibration set
— closes most of the remaining gap to FP16 (≈5.47) by spending the grid where the activations say it
matters. I am betting INT3-g128 comes down toward ~6.2, and that at the easier INT4-g128 the gap to FP16
nearly vanishes (into the ~5.6 range). The risk is the group-max coupling — over-scale and I trade one
channel's accuracy for its group-mates' — which is exactly why α is searched against true output MSE
rather than set by a formula. If it holds, I have a lighter, calibration-robust weight-only method that
matches or beats the heavy second-order one — and the whole weight-only frontier is in good shape. Which
means the next pressure has to come from the side I have been ignoring entirely: the *activations*.
