I keep treating one part of the Transformer as fixed furniture: the feed-forward sublayer. Between every pair of attention layers I run each position's hidden vector x through `f(xW1)W2` — project up to a wide hidden dimension, apply one pointwise nonlinearity, project back down. That's it. And this is not a small corner of the model: with the usual 4× expansion the two matrices W1 and W2 hold something like two-thirds of all the parameters, and a big chunk of the FLOPs, of the whole network. Putting a number on that "two-thirds": at d_model = 768 with d_ff = 4·768 = 3072, one FFN's two matrices are 2·768·3072 = 4,718,592 parameters; one attention block's four projections (Q, K, V, O, each 768×768) are 4·768² = 2,359,296. So per layer the FFN already carries 4.72M/(4.72M+2.36M) ≈ 67% of the non-embedding weight — the "two-thirds" is real, and it says plainly where the budget is. Attention mixes information across positions; this layer is the entire per-position transformation, the only place each token's representation gets reshaped on its own. And the only nonlinearity in it is that single f. Everything the layer can express about a token, it expresses by taking *one* linear view xW1 and bending it pointwise. If I want to spend a fixed budget better, this is the place to push, both because it holds most of the budget and because it is where the per-position modeling capacity actually lives.

What is f today? Originally ReLU, `max(0, xW1)`. The way I read ReLU now is as a hard, sign-based gate: `ReLU(z) = z · 1[z>0]`, i.e. multiply z by a 0/1 mask that depends only on whether z is positive. Crisp, cheap, gradient of exactly 1 on the positive side, which is why it flows so well and became the default. But the all-or-nothing decision throws away magnitude information and kills the negative half entirely. The smoother replacements I have on hand soften that gate. GELU is `z·Φ(z)`, Φ the standard normal CDF — derived as the expectation of a stochastic 0/1 mask `m ~ Bernoulli(Φ(z))`, whose mean transform is `Φ(z)·z + (1−Φ(z))·0 = zΦ(z)`. Its derivative is `Φ(z) + zφ(z)`, with `φ` the standard-normal density, so the hard ReLU corner becomes a smooth slope that changes continuously through the origin. Swish is `z·σ(βz)`, the same shape with a logistic gate instead of a Gaussian one; its derivative is `σ(βz) + βzσ(βz)(1−σ(βz))`. At β=1 it is `z·σ(z)`, smooth, slightly non-monotonic, and it traces almost exactly the same curve as GELU — no surprise, since GELU's own cheap approximation is `z·σ(1.702 z)`, which is just Swish with β≈1.702.

So pause on that. Every one of these activations I'm choosing between — ReLU, GELU, Swish — has the *same internal anatomy*. Each is the raw preactivation z multiplied by a gate that is some function of z: ReLU is `z · 1[z>0]`, GELU is `z · Φ(z)`, and Swish_β is `z · σ(βz)`. The activation function is already a gating operation. It takes a value (z itself) and multiplies it by a gate (a function of z) that decides how much of the value to let through. And the Swish activation search found exactly this by brute force: when it searched over compositions of primitives for good activation functions, the winners overwhelmingly had the form `b(z, g(z))` — the raw preactivation z reused, multiplied by some gate g(z). The structure that keeps winning is value-times-gate.

Now here is what bothers me once I see the layer that way. In `f(xW1)W2`, the value and the gate come from the *same* projection. GELU's value is z = xW1 and its gate is Φ(z) = Φ(xW1); both are functions of the single linear map xW1. There is exactly one learned linear view of x feeding this whole layer, and the gate is forced to be a fixed function of the very thing it gates. The layer never gets to ask one question of x to decide *how much*, and a different question of x to decide *what*. The gating is real but it is not independent — it can't be, because there's only one projection to build it from.

What would it take to make the gate its own thing? I want the gate to be a separately-learned linear function of x, not a fixed function of the value. So instead of one up-projection xW1, give the layer two: a value projection xV and a gate projection xW. Form the hidden vector as their elementwise product, with a nonlinearity sitting on the gate path:

  hidden = f(xW) ⊗ (xV),

and then project back down with W2 as before. Now the layer takes two independent linear views of the same input — one decides, via f, how much each hidden unit should fire; the other carries the content to be modulated. The amount and the content are no longer tied to the same projection.

This shape feels familiar. Dauphin, Fan, Auli and Grangier's gated convolutional language model is built on precisely this unit: `(X∗W + b) ⊗ σ(X∗V + c)` — a content projection times a sigmoid gate, one elementwise product of two linear maps of the input. They called it a gated linear unit, and the whole point of the architecture was to put this gating on the path so the network can select which features matter for predicting the next word, without the recurrence of an LSTM. So the move I'm reaching for — split the FFN's single projection into a gate and a value and multiply them — is the gated linear unit, only there it lives in a convolutional sequence model and here it would replace `f(xW1)` inside the Transformer's position-wise FFN.

And there's a second, older thread that says the un-gated version of this is already meaningful. If I drop the activation entirely and just take `(xW) ⊗ (xV)`, that's a pure bilinear interaction — a product of two learned linear maps of the same input — which is exactly the kind of multiplicative coupling Mnih and Hinton used in their log-bilinear language model to combine distributed word representations. This has to actually buy expressivity and not just be two matrices I could fold into one. Take a single hidden unit i: its value is `(xW)_i (xV)_i = (Σ_j W_ji x_j)(Σ_k V_ki x_k) = Σ_{j,k} W_ji V_ki x_j x_k`. That's a quadratic form in x — every hidden unit is a degree-2 function carrying all the pairwise products x_j x_k, weighted by the rank-1 matrix W_{·i} V_{·i}ᵀ. A single projection-then-pointwise unit `f(xW1)_i = f(Σ_j (W1)_ji x_j)` is f applied to a degree-1 preactivation; it can never produce a cross term x_j x_k for j≠k, because f acts on one scalar. So the product of two views genuinely reaches functions the single view cannot — and it is not collapsible, since the two matmuls are separated by the elementwise product, not by a linear op. Even before any nonlinearity, going from one projection to a product of two is a real increase in what each unit can express.

Why should I believe the *gated* version trains well and not just expresses more? Here the gated-conv work gives me the argument I actually trust, and it's about gradients, which is what I always worry about when I stack many of these layers. Suppose I'd used the LSTM-style gate, content through a tanh and gate through a sigmoid — the "gated tanh unit" `tanh(X) ⊗ σ(X)`. Differentiate it:

  ∇[tanh(X) ⊗ σ(X)] = tanh'(X)∇X ⊗ σ(X) + σ'(X)∇X ⊗ tanh(X).

Look at the two paths: the upstream gradient ∇X gets multiplied by saturating activation *derivatives* — tanh' or σ'. How big can those factors be? tanh'(x) = 1 − tanh²(x), which peaks at x=0 where tanh=0, giving tanh'=1, and decays toward 0 as |x| grows; σ'(x) = σ(x)(1−σ(x)), maximized at x=0 where σ=½, giving σ'=½·½ = ¼, and again →0 away from the origin. So even in the best case (units sitting right at the origin) one path scales the gradient by at most 1 and the other by at most ¼, and away from the origin both shrink. There is no derivative-free content path through this unit; once either nonlinearity saturates, the gradient has to pass through a small derivative factor. Put a number to "stack a dozen": if each of 12 layers contributes even a mild saturated factor of 0.5, the gradient through the stack is scaled by 0.5¹² ≈ 2.4×10⁻⁴ — gone. That's the documented gradient-fade of this unit, made concrete.

Now contrast the linear-gated version `X ⊗ σ(X)` — content X carried *linearly*, only the gate passed through σ:

  ∇[X ⊗ σ(X)] = ∇X ⊗ σ(X) + X ⊗ σ'(X)∇X.

The first term is ∇X ⊗ σ(X): the upstream gradient multiplied only by the gate *value* σ(X) ∈ (0,1), not by any activation derivative. For units the gate has opened (σ(X) near 1) the gradient passes essentially undiminished. The contrast with the both-paths-nonlinear case is sharp when I run the same depth calculation: where the saturated unit multiplied by ≤¼ on its gated path, here an opened gate multiplies by a *value* close to 1. Take a gate value of 0.9 per layer; over 12 layers that's 0.9¹² ≈ 0.28, the gradient retains roughly a quarter of its magnitude instead of vanishing to 10⁻⁴. The difference is the whole ballgame: a clean, almost-linear highway back through the layer, a multiplicative skip connection, versus a per-layer downscaling tax. That's the structural reason to keep the *content* path linear and put the nonlinearity only on the *gate*. So the design isn't "two projections, nonlinearity wherever" — it's specifically value carried linearly, gate carried through f, product taken. That matches the shape I wrote: `f(xW) ⊗ (xV)`, with V's path linear.

Now which f for the gate? The gated-conv unit used σ. But the FFN already has a serious smooth alternative to ReLU in GELU: it is value-weighting rather than hard sign-gating, and it behaves almost like the Swish shape that activation search keeps rediscovering. The minimal, consistent move is to make the gate use that same activation family, rather than reaching for an unrelated function. Put GELU on the gate path:

  hidden = GELU(xW) ⊗ (xV).

GELU on the gate, not the σ the original GLU used — but that preference should rest on more than familiarity. Comparing the two gates numerically: a sigmoid gate is strictly in (0,1): whatever the preactivation, it can only ever scale the content down toward zero. GELU's gate `GELU(z) = zΦ(z)` is not bounded that way. Tabulating both at a few z:

  z    GELU(z)=zΦ(z)   σ(z)
  −3     −0.0040       0.047
  −1     −0.1587       0.269
  −0.5   −0.1543       0.378
   0      0.0000       0.500
   0.5    0.3457       0.622
   1      0.8413       0.731
   2      1.9545       0.881
   3      2.9960       0.953

Two things fall out that I would only have guessed at. First, GELU crosses 1 — solving zΦ(z)=1 numerically puts the crossing near z≈1.14, and by z=2 the gate is ≈1.95, by z=3 it is ≈3.0. So a GELU gate can pass content through at *greater than unit gain* when its preactivation is large and positive; σ never can, it tops out below 1. Second, on the negative side GELU goes slightly below zero — scanning z<0 the minimum is ≈−0.170 at z≈−0.75 — so the gate can softly *sign-flip* and suppress, not merely attenuate. σ stays positive throughout. So GELU on the gate is a strictly richer modulation than σ: amplify, suppress, or softly flip, versus σ's pure (0,1) squashing. And it keeps the smoothness and continuity through the origin that made me prefer GELU to ReLU in the first place. Sigmoid, ReLU, GELU, Swish, or nothing-at-all on the gate are all available — they give the whole family, GLU / ReGLU / GEGLU / SwiGLU / Bilinear — but GELU on the gate is the one that lines up with the FFN I'm actually improving.

So the layer is `(GELU(xW) ⊗ xV) W2`. Three weight matrices now — the gate projection W, the value projection V, and the down-projection W2 — where the old FFN had two. And that is a problem I have to confront immediately, because the whole premise of the comparison is matched budget. If I just bolt a third matrix on, of course it might do better — it has more parameters and does more compute. Any quality gain would be confounded with spending more. I have to hold parameters and FLOPs fixed against the baseline FFN, which means the extra projection has to be paid for by shrinking the hidden width.

The bookkeeping: write d for d_model and let the baseline FFN use hidden width d_ff. The baseline has W1 of shape d×d_ff and W2 of shape d_ff×d, so its parameter count is 2·d·d_ff (ignore biases, which the T5 convention drops anyway). My gated layer has W, V each of shape d×d_ff' and W2 of shape d_ff'×d, for 3·d·d_ff' parameters, where d_ff' is the hidden width I'm allowed. Match them:

  3·d·d_ff' = 2·d·d_ff   ⇒   d_ff' = (2/3)·d_ff.

So the gated FFN must reduce its hidden dimension to two-thirds of the baseline's. The FLOP side needs its own check, since parameters and compute don't always track each other. Per token, the baseline does one d×d_ff matmul up and one d_ff×d matmul down: 2·d·d_ff multiply-adds. The gated layer does two d×d_ff' up-matmuls (gate and value) and one d_ff'×d down-matmul: 3·d·d_ff' multiply-adds. Substituting d_ff' = (2/3)·d_ff gives 3·d·(2/3)·d_ff = 2·d·d_ff — identical to the baseline. The elementwise GELU and the elementwise product are O(d_ff') pointwise work, negligible next to the matmuls. So at this width parameters and FLOPs match simultaneously; the (2/3) rule settles both.

Plugging in real numbers, to see whether the rule lands on a clean width or some awkward fraction. With the standard 4× expansion, d_ff = 4d, then

  d_ff' = (2/3)·4d = (8/3)·d ≈ 2.667·d.

At the T5/GPT scale d = 768: (8/3)·768 = 2048 exactly. And it is exactly matched — baseline 2·768·3072 = 4,718,592 params, gated 3·768·2048 = 4,718,592 params, difference 0. At this width the rule isn't approximate — it's an identity. In a GPT-style model where the natural width is `4 * n_embd`, I compute the target `(8/3) * n_embd` and round it *up* to the next multiple of 64 so the matmul tiles align with the hardware. How much does that rounding actually disturb the budget? "Round up" could quietly re-inflate the parameter count. At d=768 it's a no-op: 2048 is already a multiple of 64, so zero drift. At d=1024, (8/3)·1024 ≈ 2730.7 → round to 2752, which pushes the gated params to +0.78% over baseline; at d=1600, → 4288, +0.50%. So at the widths I care about the round-up is a sub-1% nudge, not a real change in budget. (It is not universally tiny — at d=512, (8/3)·512 ≈ 1365.3 → 1408 is +3.1%, because 64 is a coarser grid relative to a smaller width — so if I were at a small d I'd want to keep an eye on it. At the scales here it's negligible.) I omit biases on all three matrices, both to keep the parameter accounting clean and because the FFN baseline I'm matching already uses the bias-free T5 convention.

The forward pass: gate projection w1 takes x from (batch, length, d) to (batch, length, d_ff') and I apply GELU to it; value projection w3 takes x to the same shape with no activation; the elementwise product stays (batch, length, d_ff'); down-projection c_proj takes that back to (batch, length, d), then dropout. Same shape in, same shape out — a true drop-in for the FFN, touching nothing in attention, normalization, the data, or the optimizer.

Do I have an honest mechanistic *proof* that this lowers loss? No — and I should be careful not to invent one. What I have are three computed reasons to expect it to help at matched budget: each hidden unit is a quadratic form Σ W_ji V_ki x_j x_k, so the layer reaches multiplicative cross-terms a single-projection FFN provably cannot; the linear content path gives the gradient a factor of the gate *value* (≈1 when open) instead of a saturating derivative (≤¼), the difference between 0.9¹²≈0.28 and 0.5¹²≈2×10⁻⁴ over twelve layers; and the GELU gate, numerically, can amplify past unit gain (≈1.95 at z=2) and softly sign-flip (min ≈−0.17), where a sigmoid only ever attenuates. Whether those add up to a real improvement on the actual objective is something I'd validate on held-out language-modeling perplexity, comparing strictly against the matched-budget GELU FFN; the design-time arguments tell me where to expect the gain, not that it is guaranteed. The reasoning gets me to the architecture; the architecture is simple, budget-matched (exactly at d=768, sub-1% after the explicit tile rounding at the widths I use), and changes nothing else — which is exactly what makes it worth trying.

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
