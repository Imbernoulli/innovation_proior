Let me look hard at the one part of the Transformer I keep treating as fixed furniture: the feed-forward sublayer. Between every pair of attention layers I run each position's hidden vector x through `f(xW1)W2` — project up to a wide hidden dimension, apply one pointwise nonlinearity, project back down. That's it. And this is not a small corner of the model: with the usual 4× expansion the two matrices W1 and W2 hold something like two-thirds of all the parameters, and a big chunk of the FLOPs, of the whole network. So if I want to spend my fixed budget better, this is the obvious place to push, because it's where most of the budget already is. Attention mixes information across positions; this layer is the entire per-position transformation, the only place each token's representation gets reshaped on its own. And the only nonlinearity in it is that single f. Everything the layer can express about a token, it expresses by taking *one* linear view xW1 and bending it pointwise.

What is f today? Originally ReLU, `max(0, xW1)`. The way I read ReLU now is as a hard, sign-based gate: `ReLU(z) = z · 1[z>0]`, i.e. multiply z by a 0/1 mask that depends only on whether z is positive. Crisp, cheap, gradient of exactly 1 on the positive side, which is why it flows so well and became the default. But the all-or-nothing decision throws away magnitude information and kills the negative half entirely. The smoother replacements I have on hand soften that gate. GELU is `z·Φ(z)`, Φ the standard normal CDF — derived as the expectation of a stochastic 0/1 mask `m ~ Bernoulli(Φ(z))`, whose mean transform is `Φ(z)·z + (1−Φ(z))·0 = zΦ(z)`. Its derivative is `Φ(z) + zφ(z)`, with `φ` the standard-normal density, so the hard ReLU corner becomes a smooth slope that changes continuously through the origin. Swish is `z·σ(βz)`, the same shape with a logistic gate instead of a Gaussian one; its derivative is `σ(βz) + βzσ(βz)(1−σ(βz))`. At β=1 it is `z·σ(z)`, smooth, slightly non-monotonic, and it traces almost exactly the same curve as GELU — no surprise, since GELU's own cheap approximation is `z·σ(1.702 z)`, which is just Swish with β≈1.702.

So pause on that. Every one of these activations I'm choosing between — ReLU, GELU, Swish — has the *same internal anatomy*. Each is the raw preactivation z multiplied by a gate that is some function of z: ReLU is `z · 1[z>0]`, GELU is `z · Φ(z)`, and Swish_β is `z · σ(βz)`. The activation function is already a gating operation. It takes a value (z itself) and multiplies it by a gate (a function of z) that decides how much of the value to let through. And the Swish activation search found exactly this by brute force: when it searched over compositions of primitives for good activation functions, the winners overwhelmingly had the form `b(z, g(z))` — the raw preactivation z reused, multiplied by some gate g(z). The structure that keeps winning is value-times-gate.

Now here is what bothers me once I see the layer that way. In `f(xW1)W2`, the value and the gate come from the *same* projection. GELU's value is z = xW1 and its gate is Φ(z) = Φ(xW1); both are functions of the single linear map xW1. There is exactly one learned linear view of x feeding this whole layer, and the gate is forced to be a fixed function of the very thing it gates. The layer never gets to ask one question of x to decide *how much*, and a different question of x to decide *what*. The gating is real but it is not independent — it can't be, because there's only one projection to build it from.

What would it take to make the gate its own thing? I want the gate to be a separately-learned linear function of x, not a fixed function of the value. So instead of one up-projection xW1, give the layer two: a value projection xV and a gate projection xW. Form the hidden vector as their elementwise product, with a nonlinearity sitting on the gate path:

  hidden = f(xW) ⊗ (xV),

and then project back down with W2 as before. Now the layer takes two independent linear views of the same input — one decides, via f, how much each hidden unit should fire; the other carries the content to be modulated. The amount and the content are no longer tied to the same projection.

I should check this isn't a brand-new gadget I'm inventing from nothing, because I've seen this exact shape before, just not in a Transformer FFN. Dauphin, Fan, Auli and Grangier's gated convolutional language model is built on precisely this unit: `(X∗W + b) ⊗ σ(X∗V + c)` — a content projection times a sigmoid gate, one elementwise product of two linear maps of the input. They called it a gated linear unit, and the whole point of the architecture was to put this gating on the path so the network can select which features matter for predicting the next word, without the recurrence of an LSTM. So the move I'm reaching for — split the FFN's single projection into a gate and a value and multiply them — is the gated linear unit, lifted out of the convolutional sequence model and dropped into the Transformer's position-wise FFN in place of `f(xW1)`.

And there's a second, older thread that says the un-gated version of this is already meaningful. If I drop the activation entirely and just take `(xW) ⊗ (xV)`, that's a pure bilinear interaction — a product of two learned linear maps of the same input — which is exactly the kind of multiplicative coupling Mnih and Hinton used in their log-bilinear language model to combine distributed word representations. A product of two linear views of x can represent pairwise, multiplicative feature interactions that no single-projection-then-pointwise map can: each hidden unit becomes a degree-two function of x rather than a degree-one preactivation bent by a fixed curve. So even before I add any nonlinearity, going from one projection to a product of two is a genuine increase in what the layer can express per unit, not just a rearrangement.

Why should I believe the *gated* version trains well and not just expresses more? Here the gated-conv work gives me the argument I actually trust, and it's about gradients, which is what I always worry about when I stack many of these layers. Suppose I'd used the LSTM-style gate, content through a tanh and gate through a sigmoid — the "gated tanh unit" `tanh(X) ⊗ σ(X)`. Differentiate it:

  ∇[tanh(X) ⊗ σ(X)] = tanh'(X)∇X ⊗ σ(X) + σ'(X)∇X ⊗ tanh(X).

Look at the two paths: the upstream gradient ∇X gets multiplied by saturating activation *derivatives* — tanh', in `[0,1]`, or σ', in `[0,1/4]`. There is no derivative-free content path through this unit; once either nonlinearity saturates, the gradient has to pass through a small derivative factor. Stack a dozen of these and the signal is attenuated a dozen times over; it's documented that this kind of unit's gradient fades as you go deep. Now contrast the linear-gated version `X ⊗ σ(X)` — content X carried *linearly*, only the gate passed through σ:

  ∇[X ⊗ σ(X)] = ∇X ⊗ σ(X) + X ⊗ σ'(X)∇X.

The first term is ∇X ⊗ σ(X): the upstream gradient multiplied only by the gate *value* σ(X) ∈ (0,1), not by any activation derivative. For units the gate has opened (σ(X) near 1) the gradient passes essentially undiminished — a clean, almost-linear highway back through the layer, a multiplicative skip connection. That's the structural reason to keep the *content* path linear and put the nonlinearity only on the *gate*: it buys you GLU's expressivity without paying the per-layer gradient-downscaling tax that the both-paths-nonlinear unit pays. So the design isn't "two projections, nonlinearity wherever" — it's specifically value carried linearly, gate carried through f, product taken. That matches the shape I wrote: `f(xW) ⊗ (xV)`, with V's path linear.

Now which f for the gate? The gated-conv unit used σ. But the FFN already has a serious smooth alternative to ReLU in GELU: it is value-weighting rather than hard sign-gating, and it behaves almost like the Swish shape that activation search keeps rediscovering. The minimal, consistent move is to make the gate use that same activation family, rather than reaching for an unrelated function. Put GELU on the gate path:

  hidden = GELU(xW) ⊗ (xV).

I can also see why GELU (or Swish) is a better gate than a bare sigmoid, not just a familiar one. A sigmoid gate is strictly in (0,1): it can only ever *attenuate* the content, scale it down toward zero, never pass it at more than unit strength and never flip or amplify it. The GELU gate output is not bounded above by 1 the same way — `GELU(z) = zΦ(z)` grows like z for large positive z and dips slightly negative for moderately negative z (the same non-monotonic "bump" Swish has). So a GELU gate can pass content through at greater-than-unit gain when the gate preactivation is large and positive, and can softly suppress or sign-flip near zero, which is a strictly richer modulation than σ's pure squashing. And it keeps the smoothness and the good behavior at the origin that made me prefer GELU to ReLU in the first place. Sigmoid, ReLU, GELU, Swish, or nothing-at-all on the gate are all available — they give the whole family, GLU / ReGLU / GEGLU / SwiGLU / Bilinear — but GELU on the gate is the one that lines up with the FFN I'm actually improving.

So the layer is `(GELU(xW) ⊗ xV) W2`. Three weight matrices now — the gate projection W, the value projection V, and the down-projection W2 — where the old FFN had two. And that is a problem I have to confront immediately, because the whole premise of the comparison is matched budget. If I just bolt a third matrix on, of course it might do better — it has more parameters and does more compute. Any quality gain would be confounded with spending more. I have to hold parameters and FLOPs fixed against the baseline FFN, which means the extra projection has to be paid for by shrinking the hidden width.

Let me do the bookkeeping. Write d for d_model and let the baseline FFN use hidden width d_ff. The baseline has W1 of shape d×d_ff and W2 of shape d_ff×d, so its parameter count is 2·d·d_ff (ignore biases, which the T5 convention drops anyway). My gated layer has W, V each of shape d×d_ff' and W2 of shape d_ff'×d, for 3·d·d_ff' parameters, where d_ff' is the hidden width I'm allowed. Match them:

  3·d·d_ff' = 2·d·d_ff   ⇒   d_ff' = (2/3)·d_ff.

So the gated FFN must reduce its hidden dimension to two-thirds of the baseline's. The FLOP count scales the same way — each of the three matmuls is d×d_ff' work per token, three of them is 3·d·d_ff' = 2·d·d_ff, the same as the baseline's two d×d_ff matmuls — so matching parameters matches compute here too. Concretely, if the baseline uses the standard 4× expansion, d_ff = 4d, then

  d_ff' = (2/3)·4d = (8/3)·d ≈ 2.667·d.

That's the magic number: where the plain FFN went up to 4d, this gated FFN goes up to (8/3)d, and then it has three matrices instead of two, and the two layouts cost exactly the same before hardware rounding. In the T5-scale model where d_ff = 3072, that's d_ff' = 2048 exactly. In a GPT-style model where the natural width is `4 * n_embd`, I compute the target `(8/3) * n_embd` and round it *up* to the next multiple of 64 so the matmul tiles align with the hardware — a sub-1% nudge at common widths, not a real change in budget. I omit biases on all three matrices, both to keep the parameter accounting clean and because the FFN baseline I'm matching already uses the bias-free T5 convention.

Let me write the forward pass concretely to make sure the shapes click. Input x is (batch, length, d). Gate projection w1 takes x to (batch, length, d_ff') and I apply GELU to it. Value projection w3 takes x to (batch, length, d_ff') with no activation. Multiply them elementwise — same shape, (batch, length, d_ff'). Down-projection c_proj takes that back to (batch, length, d), and dropout on the output. Same `(batch, length, d)` in, same out: a true drop-in for the FFN, touching nothing in attention, normalization, the data, or the optimizer.

Do I have an honest mechanistic *proof* that this lowers loss? No — and I should be careful not to invent one. What I have are three converging design-time reasons to expect it to help at matched budget: it forms each hidden unit from a product of two independent linear views of x, so the layer can express multiplicative feature interactions a single-projection FFN cannot; it keeps the content path linear so the gradient gets a clean almost-undiminished highway back through every layer, the way the linear-gated unit does and the both-paths-nonlinear unit does not; and it reuses the same GELU I already preferred at this layer, as a richer gate than a plain sigmoid because it can amplify and softly sign-flip rather than only attenuate. Whether those add up to a real improvement is something I'd validate on the held-out language-modeling perplexity, comparing strictly against the matched-budget GELU FFN. The reasoning gets me to the architecture; the architecture is simple, budget-matched before the explicit tile rounding, and changes nothing else — which is exactly what makes it worth trying.

So here is the layer I'd actually write, filling the open slot in the FFN sublayer — the gated form at two-thirds width, GELU on the gate, content carried linearly, projected back down:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Gated feed-forward sublayer: GELU(xW) ⊗ (xV), projected back down.

    Two independent up-projections of x — a GELU-gated path and a linear value
    path — are multiplied elementwise, then projected down. The hidden width is
    set to 8/3 * d so the three matrices cost the same parameters and FLOPs as the
    baseline two-matrix FFN at 4x expansion. (B, T, n_embd) in, same shape out.
    """

    def __init__(self, config):
        super().__init__()
        # match the 2-matrix FFN's budget: 3 * d_ff' = 2 * (4d)  ->  d_ff' = (8/3) d
        hidden_dim = int(8 / 3 * config.n_embd)                          # (8/3) * d
        hidden_dim = ((hidden_dim + 63) // 64) * 64                     # round up to matmul-friendly width
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)      # gate projection W
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)      # value projection V
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)  # down-projection W2
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):                                  # x: (B, T, n_embd)
        # gate (nonlinear) ⊗ value (linear), then project down:
        # GELU(xW) keeps the content path xV linear, so its gradient gets a clean
        # highway (no activation-derivative factor) back through the layer.
        return self.dropout(self.c_proj(F.gelu(self.w1(x)) * self.w3(x)))
```

To recap the causal chain: the FFN holds most of the model's budget yet forms each hidden unit from a single linear view of x bent by one fixed pointwise function — and every such function, ReLU, GELU, Swish, is itself already a value-times-gate, with the value and gate tied to the same projection. Untying them — one projection for the gate, one for the value, multiplied elementwise — is the gated linear unit, which I recognized from gated convolutional language models and, in its un-activated bilinear form, from log-bilinear models; it lets each unit express a multiplicative interaction of two independent linear views of x. Keeping the value path linear and putting the nonlinearity only on the gate gives the gradient a clean multiplicative-skip highway through every layer, unlike the both-paths-nonlinear unit whose gradient is downscaled by an activation derivative on every path; and putting GELU on that gate keeps it consistent with the smooth value-weighting I already prefer for this layer while giving a richer-than-sigmoid modulation that can amplify and softly sign-flip. Because three matrices replace two, I shrink the hidden width to (2/3)·d_ff = (8/3)·d, round up only for 64-wide matmul tiles, and omit biases so parameters and FLOPs match the baseline FFN before that explicit hardware-friendly nudge — so the only thing that changed is the *form* of the per-position transformation, which is exactly the variable I wanted to isolate.
