The scaffold ships a pass-through `BitLinear` that leaves the weights in float, so it is just FP GPT-2 wearing a quantizer costume and tells me nothing about the regime I actually care about. To find the honest floor of the native-low-bit idea I should start from its most aggressive form: one bit per weight, $\{-1,+1\}$, applied on every forward pass and trained from scratch, so I can measure how much of a 355M decoder's language ability survives when each projection weight is collapsed to a single sign.

I propose the binary-sign `BitLinear`: the forward weight is $\mathrm{sign}(W)$ rescaled by a per-tensor scalar, the latent weight stays float, and the gradient is carried through the threshold by a straight-through estimator. The starting move is forced on me by the sign function, whose derivative is zero everywhere except at the origin: ordinary backprop multiplies by that derivative, the gradient reaching $W$ is zero almost surely, and nothing trains. So I lie in the backward pass — apply the sign forward, treat it as the identity backward — using the detached-difference idiom $w_q = (\mathrm{sign}(W) - W)\texttt{.detach()} + W$. Forward this evaluates to $W + \mathrm{sign}(W) - W = \mathrm{sign}(W)$; the detached parenthesis contributes no gradient, so $\partial w_q/\partial W = 1$ — exactly the wire I want, with no custom backward. This is the STE of Bengio, Léonard and Courville (2013); it is licensed rather than arbitrary, since the deterministic version is the shadow of the provably unbiased estimator $(h - \mathrm{sign}(a))\cdot L$ for a stochastic binary neuron, a special case of the score-function trick. And the float latent weight matters for a reason particular to binary training (BinaryConnect, Courbariaux, Bengio and David 2015): a single bit cannot integrate SGD's tiny noisy steps, because a small update rarely flips a sign, so I keep $W$ in float as the optimizer's accumulator and binarize it on the fly each pass.

A bare $\mathrm{sign}(W)$ forces every forward weight to unit magnitude, which inflates each output dot product by a large systematic factor and throws away the one cheap piece of magnitude information I could keep. So I approximate $W$ not by $\mathrm{sign}(W)$ but by $\alpha\cdot\mathrm{sign}(W)$ for a positive scalar — one number per tensor, negligible storage, factoring out of the matmul as a single post-multiply at inference. I do not guess $\alpha$; I solve for it. Minimizing $J(B,\alpha) = \|W - \alpha B\|^2$ over $B \in \{-1,+1\}^n$ and $\alpha > 0$ and expanding gives $J = \alpha^2 B^\top B - 2\alpha W^\top B + W^\top W$; since every $B_i = \pm 1$, $B^\top B = n$ is constant, so with $\alpha$ fixed the only $B$-dependent term is $-2\alpha W^\top B$, and minimizing $J$ means maximizing $W^\top B = \sum_i W_i B_i$, which decouples coordinatewise to $B_i = \mathrm{sign}(W_i)$. The sign is therefore the provably optimal binary direction, not a heuristic. With $B$ at that optimum, $\partial J/\partial\alpha = 2\alpha n - 2W^\top B = 0$ gives $\alpha^\star = (1/n)\sum_i |W_i| = \mathrm{mean}(|W|)$ — the absmean. So the L2-best one-bit summary of a weight tensor is $\mathrm{mean}(|W|)\cdot\mathrm{sign}(W)$, both pieces falling out of a single least-squares problem (XNOR-Net, Rastegari et al. 2016).

This is the task's binary fill, and it deliberately keeps less machinery than the textbook binarized Transformer. It uses the *uncentered* $\mathrm{sign}(W)$ rather than $\mathrm{sign}(W - \mathrm{mean}(W))$ — the centering is a capacity refinement I leave for later — and it adds **no** sub-layer normalization inside `BitLinear`. That omission is correct, not careless: each GPT-2 block already applies a `LayerNorm` immediately before its projections, so the input reaching every quantizer is already normalized. Checking the variance confirms it. For $y = \sum_i \tilde w_i \tilde x_i$ over $n$ inputs, treating entries as i.i.d., $\mathrm{Var}(y) = n\cdot E[\tilde w^2]\cdot E[\tilde x^2]$; the scaled binary weight has magnitude $\beta = \mathrm{mean}(|W|)$ on every entry, so $E[\tilde w^2] = \beta^2$ and $\mathrm{Var}(y) = n\beta^2\,E[\tilde x^2]$. Under the $\texttt{std}=0.02$ init, $n\beta^2$ is a controlled constant, and $E[\tilde x^2]$ is held near 1 by the block's pre-projection `LayerNorm` — so the binarized layer inherits the float layer's variance scale, and the absmean does double duty as both the L2-derived magnitude and the thing that keeps $n\beta^2$ in check.

A matmul has two operands, and leaving the activations in float would solve only half the cost problem — the expensive floating-point multiply would still sit on the activation side. So I quantize activations too, but asymmetrically, because activations are a harder animal: weight distributions are flat and binarize cleanly, while LLM activations carry a few outlier feature channels whose magnitudes run tens to a hundred times larger than the rest and recur in the same channels across tokens. One bit on activations would mean quantizing the very hardest thing with two levels in the presence of huge outliers, so I keep activations at a *modest* 8 bits, where there are still enough levels to absorb the outlier dynamic range, and reserve the extreme treatment for the well-behaved weights. The scheme is symmetric absmax: divide by the absolute maximum, scale into the signed int8 range with $Q_b = 127$ so values land in $[-127, 127]$, round, clip, and carry $\mathrm{max}|x|/Q_b$ as the dequant scale, with the same detached-difference STE carrying the gradient through the non-differentiable round and clip. I take a single **per-tensor** absmax over the whole activation tensor rather than a per-token maximum — per-token would confine an outlier's damage to its own row and is the tighter scheme, but per-tensor is the plain one, held identical across every rung of this ladder so that the only thing changing rung to rung is the *weight* grid.

I leave the inherited recipe untouched — `CONFIG_OVERRIDES` empty, peak LR `6e-4`, the FP GPT-2 setting — so the measurement isolates the cost of binarization alone rather than binarization plus a re-tuned schedule. There is a known optimization wrinkle I am choosing not to act on yet: because the gradient to the latent weight is computed through the binarized forward weight, a small step usually does not flip any sign, and the standard lever is a larger learning rate to push latent weights across zero; if the floor is bad, that is the first thing the diagnosis points at. What I expect from one bit is a model that *trains* — the absmean scale and the STE make it a real language model, the LayerNorms keep it stable — but lands well above the float loss, because a two-valued set has no way to express "this connection is off": a near-zero latent weight is still shoved to $\pm\beta$ at full strength with whichever sign noise gave it. That missing off-state is precisely the cheapest thing the next rung can buy back, by turning $\{-1,+1\}$ into $\{-1,0,+1\}$ with the same absmean machinery.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- step 1: binary sign {-1, +1}
def weight_quant(weight):
    """Binary quantization: sign(W) * mean(|W|) with STE.

    Forward: w_q = sign(W), scale = mean(|W|)
    Backward: STE (gradient passes through sign as identity)
    """
    scale = weight.detach().abs().mean()
    # STE: forward uses sign, backward treats sign as identity
    w_q = (weight.sign() - weight).detach() + weight
    return w_q, scale


def activation_quant(x):
    """Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling, following the original BitNet paper.
    """
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    """BitNet linear layer with binary {-1, +1} weights.

    During both training and eval: weights are binarized via sign function,
    activations are quantized to int8 range. Output is rescaled by
    weight_scale * activation_scale.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        w_q, w_scale = weight_quant(self.weight)
        x_q, x_scale = activation_quant(x)
        out = F.linear(x_q, w_q, None)
        out = out * (w_scale * x_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
```
