The substrate is frozen and the only architectural slot I own is the `MLP` class, so before I touch
anything I have to decide where the design pressure actually is. The block is `x = x + attn(ln_1(x)); x =
x + mlp(ln_2(x))`, and the MLP is `Linear(d,4d) → GELU → Linear(4d,d)`. With `d = n_embd = 1024`, the
up-projection `c_fc` is `d×4d = 1024×4096 = 4,194,304` weights and the down-projection `c_proj` is `4d×d`,
another `4,194,304`; together `8d² = 8,388,608 ≈ 8.4M` parameters per block. The attention projections in
the same block — fused `q,k,v` at `d×3d` and the output projection `d×d` — come to `4d² ≈ 4.2M`. So inside
one block the FFN carries `8d²` against attention's `4d²`, two-thirds of the per-block transform, a clean
2-to-1; over 24 layers that is `24·8d² ≈ 201M` of the model's ~355M, the rest being the tied `50257×1024 ≈
51M` embedding and attention. This layer is where the budget already sits, and it is also the *entire*
per-position transformation — attention mixes tokens across positions, but the MLP is the only place each
token's own vector gets reshaped on its own. So if I want to spend a fixed budget better, this is where to push.

And the whole per-position transform hangs on a single nonlinearity, that one `GELU`. Everything the layer
can express about a token, it expresses by taking *one* linear view `xW1 = c_fc(x)` and bending it
pointwise. The family I'm choosing from is small and shares an anatomy. GELU is `GELU(z) = z·Φ(z)`, Φ the
standard-normal CDF — the expectation of a stochastic 0/1 mask `m ~ Bernoulli(Φ(z))`, so `z` multiplied by
a smooth value in roughly (0,1) that rises with `z`. Its predecessor ReLU is the hard version of the same
idea, `z·1[z>0]`. Swish is `z·σ(βz)`; at β=1 it traces almost exactly GELU's curve (GELU's own cheap
approximation is `z·σ(1.702 z)`). Every activation I could drop into this slot — ReLU, GELU, Swish — is the
raw preactivation `z` multiplied by a gate that is some function of `z`. The activation is already a gating
operation: a value (`z` itself) times a gate (a function of `z`) that decides how much of the value to let
through. The automated activation search that turned up Swish made this explicit — its winners
overwhelmingly had the form `b(z, g(z))`, the raw preactivation reused and multiplied by a gate of itself.
The structure that keeps winning is value-times-gate.

Here is what bothers me once I see the layer that way. In `f(xW1)·W2`, the value and the gate come from the
*same* projection. GELU's value is `z = xW1` and its gate is `Φ(z) = Φ(xW1)`; both are functions of the
single linear map `c_fc(x)`. The layer never gets to ask one question of `x` to decide *how much*, and a
different question of `x` to decide *what*. The gating is real, but it is not independent — it can't be,
because there is only one projection to build it from.

So what to do with the one slot. I could leave the structure exactly as it is and only reshape the
*pointwise function*, swapping GELU for some other curve — the smallest move, but it does nothing about the
"one view" limitation; reshaping the bend is a lever on an orthogonal axis that I'd rather isolate later
than confound with a structural change now. I could spend expressivity on *depth* — three matrices in
series, `d→h→h→d`. But run the budget: `2dh + h² = 8d²` gives `h² + 2dh − 8d² = 0`, so `h = 2d`, a serial
FFN collapsing to width `2d = 2048` with a square `h×h` middle matrix, and for all that it still computes a
*composition* of degree-one bends — no unit becomes a product of two views of `x` — while stacking two
nonlinear bends in one residual branch is exactly the compounding-derivative setup I am about to argue
against in the gradient. Serial depth spends the third matrix on the wrong thing. The third option is to
spend the extra projection on a *multiplicative* interaction instead of depth, and that is the one that
speaks to the actual diagnosis.

What would it take to make the gate its own thing? Give the layer two up-projections — a gate `xW` and a
value `xV` — and form the hidden vector as their elementwise product, with the nonlinearity on the gate
path, then project back down:

  hidden = f(xW) ⊗ (xV),   then  hidden·W2.

Now the layer takes two independent linear views of the same input — one decides, via `f`, how much each
hidden unit fires; the other carries the content. The amount and the content are no longer tied to the same
projection. This is not a gadget from nothing: the gated convolutional language model of Dauphin, Fan, Auli
and Grangier is `(X∗W) ⊗ σ(X∗V)` — a content projection times a sigmoid gate, one elementwise product of
two linear maps of the input — which they called a gated linear unit, to let the network select which
features matter without recurrence. And the un-activated version `(xW) ⊗ (xV)` is a pure bilinear
interaction, the multiplicative coupling Mnih and Hinton used in their log-bilinear language model. A
product of two linear views makes each hidden unit a degree-two function of `x` rather than a degree-one
preactivation bent by a fixed curve — the genuine increase serial depth could not buy at any width, since
composition of degree-one maps stays degree-one in each factor but a product raises the polynomial degree
of the unit itself. So the move is the gated linear unit, lifted out of the convolutional sequence model
and dropped into this MLP slot in place of `GELU(xW1)`.

Why should I believe the *gated* version trains well, not just expresses more? I'm stacking 24 of these, so
the gradient is what I worry about, and the placement of the nonlinearity — gate path, value path, or both
— is a real fork that taxes every layer if I get it wrong. Suppose I used the LSTM-style gated-tanh unit,
content through `tanh` and gate through `σ`: `tanh(X) ⊗ σ(X)`, whose gradient is
`tanh'(X)∇X ⊗ σ(X) + σ'(X)∇X ⊗ tanh(X)`. Every term multiplies the upstream gradient by an activation
*derivative* — `tanh' ≤ 1`, or `σ' ≤ ¼`. Now the linear-gated version `X ⊗ σ(X)`, content carried linearly,
only the gate through σ: `∇X ⊗ σ(X) + X ⊗ σ'(X)∇X`. Evaluate the through-branch gain on an *opened* unit,
`X` large positive — the case that matters, the units the layer wants to pass forward. Linear-value:
`σ(X) + X·σ'(X) → 1 + 0 = 1`, the gradient passing essentially undiminished, a clean almost-linear highway,
a multiplicative skip. The same open unit under gated-tanh: `tanh'·σ + σ'·tanh → 0·1 + 0·1 = 0`, a
*saturated* gated-tanh unit has essentially zero local gradient — the very units the layer decided to fire
are the ones through which it can no longer learn. At the origin both give gain ½; the divergence is
entirely at the opened end. Tabulating the through-branch gain across `X = 1, 2, 3`: linear-value
`σ(X) + X·σ'(X)` reads `0.93, 1.09, 1.09` — rising slightly above unity in the moderately-open regime then
settling near 1, a genuine highway; gated-tanh `tanh'σ + σ'tanh` reads `0.46, 0.16, 0.05`, falling toward
zero as the gate opens — a sevenfold gap already at `X=2`, past twentyfold by `X=3`. So keep the *content*
path linear and put the nonlinearity only on the *gate*: GLU's expressivity without the per-layer gradient
collapse on exactly the units it most wants to train. The residual `x = x + mlp(...)` always leaves an
identity path, so no choice here strands the gradient entirely; what linear-value protects is the gradient
*through the FFN's own transform* — the signal that trains this layer's three matrices — which is precisely
what the saturated gated-tanh Jacobian throws away. So the design is specifically value carried linearly,
gate through `f`, product taken: `f(xW) ⊗ (xV)` with V's path linear.

Which `f` for the gate? The gated-conv unit used σ, but this MLP's default is GELU, a smooth value-weighting
behaving almost like the Swish shape the activation search keeps rediscovering — so the minimal consistent
move is to keep that same activation family: `hidden = GELU(xW) ⊗ (xV)`. GELU is also a better gate than a
bare sigmoid, not just a familiar one. A sigmoid gate is strictly in (0,1): it can only *attenuate* content
toward zero. `GELU(z) = zΦ(z)` grows like `z` for large positive `z` and dips slightly negative for
moderately negative `z`, so as a multiplier of the value a GELU gate contributes `1.95` at preactivation
`+2` (amplifying nearly twofold), `0.84` at `+1`, and `−0.17` at `−0.75` (a soft sign-flip), where a sigmoid
gate at those points is `0.88, 0.73, 0.32` — always a positive fraction, never above 1, never flipping. That
extra span from amplification through suppression through gentle inversion is expressivity the layer gets
once the gate is untied from the value. The choice of `f` — σ, ReLU, GELU, SiLU, or identity — parameterizes
the whole GLU family (GLU / ReGLU / GeGLU / SwiGLU / Bilinear); GELU is where I start because it lines up
with the incumbent activation, and I am deliberately *not* yet claiming it is the optimal member — only that
it is the conservative first fill.

That settles the form: `(GELU(xW) ⊗ xV)·W2` — GeGLU, three matrices where the old MLP had two. Since the
whole premise is matched budget — frozen substrate, fixed iteration count, `val_loss` read at the end — the
extra projection has to be paid for by shrinking the hidden width, or any gain is confounded with spending
more. Baseline: `c_fc` `d×4d`, `c_proj` `4d×d`, count `2·d·(4d) = 8d²`. Gated: `W, V` each `d×d_ff'` and
`W2` of `d_ff'×d`, count `3·d·d_ff'`. Match them: `3·d·d_ff' = 8d² ⇒ d_ff' = (8/3)·d ≈ 2.667·d`. FLOPs scale
identically — three `d×d_ff'` matmuls equal the baseline's two `d×4d` (`3·(8/3)d² = 8d² = 2·4d²`) — so
matching parameters matches compute here too. At `n_embd = 1024` the exact width is `8·1024/3 = 2730.67`,
and I round *up* to the next multiple of 64, `2752`, so the matmul tiles align. That round-up costs
`3·1024·2752 = 8,454,144` against the baseline's `8,388,608`, a ratio of `1.0078`, `+0.78%`; truncating to
2730 instead would land `0.024%` *under* baseline. Either way the deviation from matched is under one
percent, so the comparison is honest, and I take 2752 for the aligned tiles. Biases are off on all three
matrices (`config.bias = False`), matching the bias-free baseline FFN.

The forward is a true drop-in: input `x` `(B, T, n_embd)`, gate `w1(x)` with GELU and value `w3(x)` linear,
both `(B, T, hidden)`; elementwise product; `c_proj` back to `(B, T, n_embd)`; dropout — same shape in and
out, touching nothing in attention, normalization, data, or optimizer. And if the gate
`f` were the identity the layer would be `(xW) ⊗ (xV)·W2`, the pure bilinear form — so GeGLU sits one
activation away from a known-meaningful special case, not out in unexplored territory. `CONFIG_OVERRIDES`
stays empty — I'm changing the *form* of the MLP, not the schedule, and the whole point is to isolate that
one variable. The literal edit (gate `w1`, value `w3`, down `c_proj`, `F.gelu(self.w1(x)) * self.w3(x)`) is
in the answer.

I have no mechanistic proof this lowers loss, and I won't invent one — the multiplicative interaction, the
linear-value gradient highway, and the richer-than-sigmoid GELU gate are design-time reasons to expect help
at matched budget, not a guarantee. Whether they add up is what the run measures.

One caveat on wall-clock: compute is matched to within the 0.78% width nudge, but three narrower matmuls
launch more kernels over smaller tiles than two wide ones, plus an extra elementwise product to fuse, so
`elapsed` could come in a hair *above* the default's even at matched FLOPs — an artifact of matrix shape,
not of doing more work.

The one choice I made most casually is the gate's activation: GELU because it's what the MLP already ran,
not because I derived it as optimal. The family is parameterized by exactly that choice at the *same* 8/3
width and three-matrix budget, so swapping the gate is free and fully controlled — the obvious next
experiment. A second thread I'm deferring: I haven't questioned the *activation itself*, only whether to
gate it — reshaping the pointwise nonlinearity inside a plain two-matrix FFN is an orthogonal lever this
step says nothing about.
