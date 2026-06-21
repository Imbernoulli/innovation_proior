The core cost of running a large autoregressive Transformer is not the floating-point arithmetic itself but the memory bandwidth needed to stream dense FP16 weight matrices from DRAM and the energy consumed by those multiply-accumulates on every forward pass. Shrinking the weight representation therefore has a direct, roughly linear payoff in both latency and power, but the standard recipe — train in FP16, then quantize — breaks down at extreme bit widths. Post-training quantization at one bit rounds weights that were never trained to live on a binary grid, so the rounding error per weight is comparable to the weight itself and the summed error inside each dot product destroys perplexity. Quantization-aware training is only a fake-quant overlay: the optimized parameter is still a float, and the forward pass never actually uses a one-bit value. What is needed is a linear projection whose forward weight is genuinely drawn from {−1, +1} on every training step and every inference step, yet remains trainable, numerically stable when stacked dozens of layers deep, and efficient to evaluate.

The obstacle is that the natural mapping to a binary value, the sign function, has zero derivative almost everywhere. If the forward pass applies sign directly to the learnable weight, backpropagation multiplies by that zero derivative and the weight never updates. A second obstacle is that forcing every weight magnitude to one throws away the global scale of the tensor and inflates the layer output. A third is that activations are harder to quantize than weights because they contain persistent outlier channels, so aggressively binarizing both operands at once is dangerous. The right design must solve all three: get a usable gradient through the sign, preserve an appropriate scale, and treat the two operands asymmetrically.

The method is BitNet, implemented through its BitLinear layer. BitLinear replaces every standard nn.Linear in the Transformer with a drop-in module that stores a high-precision latent weight but uses a binary weight during the forward pass. The binary weight is obtained by taking the sign of the latent weight and rescaling it by the per-tensor mean absolute value of that latent weight. This scale is not hand-tuned: it is the l2-optimal scalar for approximating the latent weight by a signed binary tensor. Minimizing ||W − αB||² over B ∈ {−1, +1}ⁿ and α > 0 gives B* = sign(W) and α* = mean(|W|). The sign operation is made trainable by the straight-through estimator: in the forward pass it acts as sign, but in the backward pass it is treated as the identity so gradients flow back to the latent weight. The latent weight is what the optimizer updates; it accumulates the tiny noisy steps that SGD requires, while the binary forward weight is computed fresh each step and discarded at inference.

Centering the latent weight before taking the sign is a useful capacity refinement. Because the binary code is symmetric about zero, a non-zero mean wastes one of the two symbols; subtracting the per-tensor mean before sign gives a more even split. The scale remains the absmean of the uncentered latent weight, keeping the variance analysis clean. For activations, BitLinear uses an asymmetric W1A8 scheme: weights are one bit, but activations stay at eight-bit absmax quantization. Weights have flat, uniform distributions that binarize cleanly, whereas activations carry large-magnitude outlier features in fixed channels that would be crushed by a one-bit grid. Eight bits provides enough levels to absorb those outliers while still turning the core matmul into integer accumulation. The clip and round operations in the activation quantizer also use straight-through gradients.

To keep a deep stack stable, the output variance of the binarized matmul must stay near one. Treating the operands as independent with zero mean gives Var(y) = n · E[w̃²] · E[x̃²]. The scaled binary weight has E[w̃²] = β² where β = mean(|W|), and under standard initialization mean(|W|) scales like 1/√n, so n · β² is order one. The remaining factor is the second moment of the quantized activation. A sub-layer normalization placed immediately before activation quantization forces that second moment to one, matching the stability of a full-precision layer and removing the risk of variance explosion as the network deepens. At scale, whole-tensor statistics would require cross-shard communication and break model parallelism, so the statistics are computed in groups along the partition dimension to keep each shard independent.

Training a one-bit layer also requires a larger-than-usual learning rate. Small updates to the latent weight usually fail to cross the zero threshold and therefore fail to flip any actual bit, which is especially harmful early in training when the sign pattern needs to reorganize quickly. A large learning rate, combined with the regularizing effect of the low-bit path and the variance-preserving normalization, makes the bits flip fast enough for learning to proceed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(weight):
    """Binarize latent weight with sign and absmean scale, using STE.

    Returns (binary_weight, beta) where binary_weight is in {-1, +1}
    during the forward pass and beta = mean(|W|).
    """
    beta = weight.abs().mean()
    center = weight.mean()
    w_bin = torch.where(weight > center,
                        torch.ones_like(weight),
                        -torch.ones_like(weight))
    w_ste = weight + (w_bin - weight).detach()
    return w_ste, beta


def activation_quant(x):
    """8-bit symmetric absmax quantization with STE.

    Returns (quantized_x, gamma / Qb) so the matmul can be rescaled.
    """
    Qb = 128.0
    gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_int = (x * (Qb / gamma)).round().clamp(-Qb, Qb - 1)
    x_ste = x + (x_int - x).detach()
    return x_ste, gamma / Qb


class BitLinear(nn.Module):
    """Drop-in BitLinear layer: 1-bit weights, 8-bit activations.

    self.weight is the high-precision latent weight updated by the optimizer.
    It is binarized on the fly every forward pass and is not stored at inference.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.norm(x)
        x_int, x_scale = activation_quant(x)
        w_bin, w_scale = weight_quant(self.weight)
        out = F.linear(x_int, w_bin) * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
```
