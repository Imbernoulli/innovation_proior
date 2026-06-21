Binary GPT-2 trained but landed at a validation loss of 2.7352, with WikiText-2 perplexity 81.3, LAMBADA 110.2, ARC-Easy 47.0, and HellaSwag 28.7 — barely above the 25 chance floor on the four-way completion task. The loss curve was stable and the run finished cleanly, so this is a *capacity* failure, not an optimization one, and I can name the specific deficiency. A two-valued set $\{-1,+1\}$ forces every weight to participate at full magnitude $\pm\beta$: a near-zero latent weight — a connection the network would rather drop — is still shoved to $\pm\beta$ with whichever sign noise gave it, so the binary code cannot express *absence*. In a 355M Transformer a large fraction of trained weights genuinely want to sit near zero, and binary converts each of those into a full-strength, essentially-random-sign term, which is pure injected noise in the matmul. Buying back that off-switch costs almost nothing in bits, so it is the right thing to test first.

I propose the ternary $\{-1, 0, +1\}$ fill — BitNet b1.58. Adding a single zero level moves the alphabet from $\log_2 2 = 1$ bit to $\log_2 3 \approx 1.58$ bits per weight, but the real prize is not the bit count: a weight set to exactly zero contributes *nothing* to the dot product, its term drops out entirely, so the layer can learn which connections to switch off, and the random-sign noise that binary injected into every near-zero weight disappears because those weights now map to a clean $0$ instead of $\pm\beta$. That is the direct removal of the noise source I diagnosed. The construction reuses the binary machinery but cannot use $\mathrm{sign}$, which only ever emits $\pm 1$; I need to round a scaled weight to the *nearest* of three levels. I keep the same absmean scale $\gamma = \mathrm{mean}(|W|)$ — it was the L2-optimal scale for the sign grid, it carries over as the natural parameter-free unit, and it is the same cheap statistic that kept $n\gamma^2$ controlled in the variance accounting. Normalize $w_n = W/\gamma$, then snap $w_n$ onto $\{-1,0,+1\}$. The threshold this induces is the whole point of the zero: a normalized value below $0.5$ in magnitude rounds to $0$, between $0.5$ and $1.5$ rounds to $\pm 1$, beyond $\pm 1$ clips to $\pm 1$ — so a weight is gated off precisely when $|W| < \gamma/2 = \mathrm{mean}(|W|)/2$, i.e. when its magnitude is below half the average, exactly the genuinely weak connections, while the above-average survivors become $\pm 1$. The absmean does double duty once more: the L2 unit *and* the dial that places the zero-threshold at the sensible spot.

One detail in *how* the snap is written is worth being explicit about, because this fill orders the two non-differentiable operations differently from the textbook. The canonical ternary quantizer is round-then-clip, $\mathrm{clip}(\mathrm{round}(w_n), -1, 1)$, where the clip almost never fires because round already produces an integer and only the rare $|w_n| \ge 1.5$ would round to $\pm 2$. This fill *clips first* to $[-1, 1]$, then rounds. Folding the saturation into a continuous pre-round value pins anything with $|w_n| > 1$ to exactly $\pm 1$ before rounding ($\mathrm{round}(\pm 1) = \pm 1$), while values inside $[-1, 1]$ round to the nearest of $\{-1,0,+1\}$ exactly as before — so the two orders agree on the resulting ternary level for *every* weight; the set mapping to $0$, $+1$, $-1$ is identical. The only difference is which intermediate the STE attaches its identity gradient to: since the STE is $(\,\hat w\texttt{.round()} - \hat w)\texttt{.detach()} + \hat w$ wrapped around the *clamped* value, the gradient flows as identity through the clamped value, which is the saturating behavior I want — a weight driven past $\pm 1$ stops receiving a magnitude-increasing gradient through the round. The clamp-first ordering is therefore a faithful ternary quantizer with sensible saturation, not a different method.

The STE itself is the same wire I needed for binary, and for the same reason: $\mathrm{round}$ and $\mathrm{clip}$ have zero derivative almost everywhere, so I treat them as the identity on the backward pass via the detached-difference idiom, and the float latent weight keeps accumulating the optimizer's tiny noisy steps that a discrete variable could never integrate. Nothing about the latent-weight story changes from rung 1. As before, this is the task's fill, not the textbook one: the canonical b1.58 BitLinear is built on a LLaMA-style backbone that *fuses an RMSNorm* before the activation quantizer and scales activations *per token*; this fill has neither, and correctly so, because the GPT-2 block already applies a `LayerNorm` immediately before each projection, so the variance bookkeeping the fused RMSNorm would do is done by the frozen substrate — $\mathrm{Var}(y) = n\gamma^2 E[\tilde x^2]$ with $E[\tilde x^2]$ held near 1. The activation quantizer is the *same* per-tensor 8-bit absmax the binary rung used ($Q_b = 127$, clip $[-127, 127]$, STE), held identical as deliberate ladder hygiene so that the only change from binary to ternary is the weight grid.

It is worth pinning down why ternary and not some other small set, since I am spending a measurement on it. Two values is already shown to be too rigid. A non-symmetric set like $\{0,1\}$ breaks what made binary stable: with no negative value the matmul cannot subtract, the representation is biased, and the optimizer fights a one-sided alphabet. The symmetry of $\{-1,0,+1\}$ keeps the matmul a balanced signed add/subtract centered on zero. Richer sets — $\{-2,-1,0,1\}$ or the four-level grid of the next rung — each cost bits and kernel complexity, and the *first* set that gives me negative, zero, and positive is $\{-1,0,+1\}$, so I stop here and let the controlled question be narrow: does adding *only* the zero, at 1.58 bits, beat one bit? I expect a *modest* drop below 2.7352 — the zero removes a real noise source but is only one extra level on a tiny set, so I am buying back a fraction of the gap to float, not closing it: WikiText-2 toward the high 70s, LAMBADA toward the high 100s, and downstream essentially flat near binary's 47.0 / 28.7, because the zero helps the loss but adds no magnitude resolution. If the gain is as small as I suspect, the diagnosis for the next rung is already written: the missing thing was never the off-switch but *magnitude resolution* — three coarse levels still cannot say "this weight is small-but-not-zero" — and the fix is to spend the next bit on finer levels rather than on a zero.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 38-115) -- step 2: ternary {-1, 0, +1} (1.58-bit)
def weight_quant(weight):
    """Ternary quantization: {-1, 0, +1} via absmean with STE.

    Forward: normalize by absmean, round-then-clip to {-1, 0, +1}
    Backward: STE (gradient passes through rounding as identity)
    """
    scale = weight.detach().abs().mean().clamp(min=1e-12)
    w_normed = weight / scale
    # STE round: (round(x) - x).detach() + x
    w_q = w_normed.clamp(-1, 1)
    w_q = (w_q.round() - w_q).detach() + w_q
    return w_q, scale


def activation_quant(x):
    """Absmax 8-bit activation quantization with STE.

    Quantizes activations to 127 levels (int8 range) using per-tensor
    absmax scaling, following the BitNet b1.58 paper.
    """
    Qb = 127  # int8 range
    scale = x.detach().abs().max().clamp(min=1e-12)
    x_normed = x / scale
    x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    # STE: forward uses quantized, backward passes through
    x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    return x_q, scale / Qb


class BitLinear(nn.Module):
    """BitNet b1.58 linear layer with ternary {-1, 0, +1} weights.

    During both training and eval: weights are ternarized via absmean
    + round-clip, activations are quantized to int8 range. Output is
    rescaled by weight_scale * activation_scale.
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
