What actually costs money when I serve one of these models is the linear projections. A Transformer is, to first approximation, a stack of matmuls — Q, K, V, the attention output, the feed-forward matrices — and at inference nearly all of the arithmetic and nearly all of the memory traffic live there. In FP16 each of those is a floating-point multiply-and-accumulate, and on a modern process node the multiply is the expensive operation, several times the cost of an add; since the chip's hard ceiling is power, those FP multiplies are literally what limits throughput. On top of that I have to haul the weight matrices out of DRAM into on-chip SRAM before I can compute at all, and that bandwidth grows with model size. So if I want a model that is cheap to run, the lever is plain: stop the linear layers from doing floating-point multiplies, and make the weights small enough that loading them is cheap.

The reflex is quantization — fewer bits per weight — but the prevailing recipe is post-training: train a normal float model, then snap its weights onto a coarse grid afterward. That has to break at low bit-width, and it is worth being precise about why. Post-training quantization can only place quantized weights *near* the float values it was handed; it has no way to go searching for a different solution, and there is no reason the good low-bit network lives anywhere near the float optimum, which was found in a continuous landscape the grid does not respect. Empirically this is exactly what happens: 8-bit is essentially free, 4-bit holds with effort, 2-bit is already lossy, and 1-bit is a catastrophe with perplexities exploding by twenty-plus orders of magnitude. Weight-only second-order methods like GPTQ and QuIP push the frontier to 2-bit weights but keep activations in FP16 — so the matmul stays a float operation — and they too snap around the trained float values and cannot reach 1-bit without collapse. The conclusion I draw is that the discreteness has to be trained *in* from the start — quantization-aware training — so the optimizer searches directly in a relaxation of the low-bit space and finds a network that is genuinely good under the low-bit forward pass, not a float network mutilated afterward.

I propose BitNet b1.58: a native low-bit linear layer, BitLinear, that replaces every `nn.Linear` in the attention and SwiGLU projections of a LLaMA-style decoder with weights drawn from the ternary set $\{-1, 0, +1\}$ and activations held to 8 bits, applied identically on every forward pass during training and inference. Three discrete weight values carry $\log_2 3 \approx 1.58$ bits each, which is where the name comes from. Building the layer means answering, in order, four coupled questions: how to get a gradient through a non-differentiable quantizer, how to keep the discrete weights from stagnating, what discretizer and scale to use for the weights and activations, and how to keep the whole thing numerically stable at billions of parameters.

Take the gradient first, because without it nothing else matters. The forward operator that maps a real weight onto a discrete value — sign, or round-then-clip — is piecewise constant, so its derivative is zero almost everywhere and backprop transmits nothing. So I lie about it on the backward pass with the straight-through estimator: forward uses the discrete value, backward pretends the quantizer was the identity and lets the gradient flow straight to the latent weight. It is biased, but for a single layer it has the right sign — pushing the latent weight up tends to push the discrete weight up — so a gradient that says "raise the latent weight to lower the loss" points the right way even if its magnitude is wrong. In code it is the one-liner $x_q = x + (\text{quant}(x) - x).\text{detach}()$: the forward value is $x + (\text{quant}(x) - x) = \text{quant}(x)$, and the detached bracket contributes no gradient, so $\partial x_q / \partial x = 1$. The second wall is that tiny SGD steps almost never push a real weight across a quantization boundary, so the discrete weight stagnates while the latent weight drifts. The fix, inherited from the binary-network literature, is to keep a full-precision *latent* weight that accumulates the updates and is quantized on the fly each forward pass; the hairs of drift accumulate until the latent weight finally crosses a boundary and the discrete weight flips, and at inference the latent copy is discarded. So the optimizer touches the latent weight, the matmul touches the discrete weight, and the STE bridges them.

Now the discretizer and scale. The crudest choice, the sign, throws away all magnitude, so I need a scalar to rescale the discrete tensor back into the neighborhood of the real one: $W \approx \alpha\,B$ with $B = \text{sign}(W)$ and $\alpha > 0$ chosen to minimize the squared error. Expanding $J(B,\alpha) = \|W - \alpha B\|^2 = \alpha^2 B^\top B - 2\alpha W^\top B + W^\top W$, and using $B^\top B = n$ (a constant since $B \in \{\pm 1\}^n$) with $W^\top W$ fixed, minimizing over $B$ is the same as maximizing $W^\top B = \sum_i W_i B_i$ term by term, which gives $B^* = \text{sign}(W)$ and confirms the sign is the right discretizer for this objective. Then $\partial J/\partial \alpha = 2\alpha n - 2 W^\top B = 0$ yields the L2-optimal scale $\alpha^* = W^\top \text{sign}(W)/n = (1/n)\sum_i |W_i| = \text{mean}(|W|)$ — the absmean. It is not a tuned knob; it falls out of the data, and a matmul against a $\{-1,+1\}$ tensor needs no multiplies, only additions and subtractions plus one scalar multiply by $\alpha$ at the end.

But binary is precisely what post-training failed at, and two values are too few: every weight is forced to participate at full magnitude $\pm\beta$, and a weight that is essentially zero — a connection the network would rather switch off — still gets shoved to $+\beta$ or $-\beta$ at full strength, picking up whichever side of zero noise happened to land it on. The two-valued set cannot express *absence*. What I want is a third option, a way for the quantizer to set a weight to actually-zero and gate that connection off. Adding a zero buys two things: capacity in the counting sense — three values carry $\log_2 3 \approx 1.58$ bits instead of one, barely more than a bit for a qualitatively richer representation — and, the real prize, a weight set to exactly zero drops out of the matmul entirely, so the zero is explicit feature filtering, free sparsity a kernel can exploit. So the set is ternary $\{-1, 0, +1\}$.

I cannot reuse the binary derivation directly, since the sign only ever emits $\pm 1$; I need a rule that rounds a scaled weight to the nearest of $\{-1, 0, +1\}$. The clean construction normalizes the weight by a scale $\gamma$, rounds to the nearest integer, and clips into $[-1, 1]$:
$$\widetilde{W} = \text{RoundClip}\!\left(\frac{W}{\gamma + \epsilon}, -1, 1\right), \qquad \text{RoundClip}(x, a, b) = \max(a, \min(b, \text{round}(x))), \qquad \gamma = \frac{1}{nm}\sum_{ij} |W_{ij}|.$$
Round-then-clip *is* "snap to the nearest of $\{-1, 0, +1\}$": a normalized magnitude below $0.5$ rounds to $0$, between $0.5$ and $1.5$ rounds to $\pm 1$, and anything beyond $\pm 1$ clips. So a weight is gated to zero exactly when $|W| < \gamma/2$, which means the small, weak connections are the ones switched off — precisely the feature filtering I was reaching for — while the above-half-average weights survive as $\pm 1$. The $\epsilon$ guards against dividing by zero on a degenerate tensor. For $\gamma$ I want the same parameter-free, data-driven answer the binary case gave, so I carry over the absmean over the whole matrix. It doubles as the sparsity dial: a larger $\gamma$ raises the threshold $\gamma/2$, pushing more weights to zero; absmean is the neutral setting that can make the three values nearly balanced. A learnable elastic scale was the alternative, but it adds parameters and is touchier to train, and I specifically need to crank the learning rate, so I take the most stable parameter-free quantizer I can. As for why ternary and not another small set: $\{0, 1\}$ breaks symmetry — the matmul can no longer subtract, the representation is biased, and the optimizer fights a one-sided alphabet (it is observed to be unstable with exploding gradients and early divergence); richer sets like $\{-2,-1,0,1\}$ cost bits and packing complexity for no demonstrated gain. The first set offering negative, zero, and positive is $\{-1, 0, +1\}$, and its symmetry keeps the matmul a balanced signed add/subtract — so I stop there.

The activations flowing in are still real, and an integer-friendly matmul wants them quantized too. The established tool is absmax: to map a tensor into the signed $b$-bit range with $Q_b = 2^{b-1}$ ($Q_b = 128$ for $b = 8$), scale by $Q_b / \|x\|_\infty$, round, clip, and dequantize by the inverse; the official code writes this as scale $127/\max|x|$ and clamps to the int8 range $[-128, 127]$. Absmax is parameter-free and symmetric, and symmetry matters because $[-Q_b, Q_b]$ maps zero to zero, so the quantized matmul stays a clean signed integer operation with no zero-point offset to carry around — simpler than the asymmetric $[0, Q_b]$ pre-nonlinearity shift the 1-bit predecessor used. The known trap is that past roughly the 6.7B scale a tiny fraction of activation features (about a tenth of a percent) grow systematically huge and appear in every layer; with a single per-tensor scale that one outlier sets $\|x\|_\infty$ and squashes every other value into a sliver of the range. The fix is granularity: take the absmax *per token* (per row), so an outlier only blows up the scale of its own row and the rest keep their precision.

Two stability arguments close the design. First, variance. With i.i.d. assumptions an output coordinate $y = \sum_j \tilde w_j \tilde x_j$ over $n$ terms has $\text{Var}(y) = n\,\mathbb{E}[\tilde w^2]\,\mathbb{E}[\tilde x^2] = n\beta^2\,\mathbb{E}[\tilde x^2]$, where $\beta = \text{mean}(|W|)$; at the intended scale $n\beta^2 \approx 1$, so $\text{Var}(y) \approx \mathbb{E}[\tilde x^2]$. A full-precision layer with standard initialization keeps $\text{Var}(y)$ near $1$, which is the condition that keeps deep Transformers stable, so to inherit that I need $\mathbb{E}[\tilde x^2] \approx 1$ entering the matmul — guaranteed by normalizing the activations *before* quantizing them. That normalization is a sub-layer RMSNorm placed at the front of the layer, and I fuse it into BitLinear so the surrounding decoder block can drop its own pre-attention and pre-FFN norm: one normalization, in the right place, doing double duty. Second, the latent-update-doesn't-flip problem is worst at the very start, when I most want the weights to move; the cure is a large learning rate that makes latent steps big enough to cross boundaries, and there is a pleasant asymmetry — a float Transformer would diverge at that rate, but the discrete model tolerates it because the discretization itself regularizes the runaway. The refinement is a two-stage schedule, because the low-bit loss curve is S-shaped with the big drop coming late: a high peak rate in the first half drives fast flipping, then a decay midway lets the model settle into the late descent. Weight decay follows the same two-stage logic. A latent weight's magnitude is a confidence score for its discrete value — large magnitude is far from any boundary and won't flip, near-zero is on the fence — and decay shrinks magnitudes, lowering confidence and increasing flips. So I use the LLaMA recipe's $0.1$ early, when churn helps the model find its assignment of $\{-1, 0, +1\}$, then switch decay off in the second half so the ternary weights commit and the model converges. The optimizer is Adam with betas around $(0.9, 0.95)$, a short warmup, and sequence length about 2048. Only the parametric projections go low-bit; the embeddings, norms, and residuals stay high precision, the embeddings especially so the model can produce high-precision sampling probabilities.

The trained object is exactly the following layer. `self.weight` (from `nn.Linear`) is the latent FP weight that accumulates updates; each quantizer returns its *dequantized* value, so the tensor entering `F.linear` is already on scale and no separate scalars have to be tracked through the matmul; and the two `.detach()` wrappers are the straight-through estimator.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x):
    """Per-token absmax quantization to 8 bits (symmetric, no zero-point).
    x: (..., d). Scale each token by 127 / max|x|, round into the int8 range, divide back."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """Ternary {-1, 0, +1} absmean quantization.
    scale = 1 / mean|W|; round to nearest of {-1,0,1}, clip, divide back.
    A weight with |w| < mean|W| / 2 rounds to 0 (feature filtering)."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def rmsnorm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class BitLinear(nn.Linear):
    """Ternary-weight, 8-bit-activation linear, applied identically in train and eval.
    Built on nn.Linear so self.weight is the high-precision latent parameter.
    RMSNorm is fused in, so the surrounding LLaMA block removes its pre-projection norm."""

    def forward(self, x):
        w = self.weight
        x_norm = rmsnorm(x)
        # Straight-through estimator: forward = quantized, backward = identity
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
```

To build the model from a LLaMA LLM, replace every `nn.Linear` in attention and the SwiGLU feed-forward with `BitLinear` and remove the RMSNorm before attention and the feed-forward, since BitLinear carries it. At inference the weights are offline-quantized once to ternary, the STE is dropped, the RMSNorm is fused with the activation quantizer, and `F.linear` is replaced by a low-bit kernel that does the integer add/subtract and applies the dequant scales afterward.

```python
import torch
import torch.nn as nn


def activation_norm_quant(x, eps=1e-5):
    """Fused RMSNorm + per-token 8-bit absmax quantization (kernel-fusable).
    Returns the int8 activations and the per-token dequant scale."""
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)   # RMSNorm
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale


class BitLinear(nn.Linear):
    """Inference BitLinear. self.weight holds ternary values; self.weight_scale = 1 / mean|W|."""

    def forward(self, x):
        w = self.weight                 # ternary {-1, 0, +1}
        w_scale = self.weight_scale     # = 1 / mean|W| from offline quantization
        x_quant, x_scale = activation_norm_quant(x)
        y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale   # integer add/sub matmul + dequant
        return y
```
