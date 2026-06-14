The 2-bit grid finally broke the plateau. Validation loss fell from the 2.72 the binary and ternary
rungs sat on to 2.4392, WikiText-2 from ~78 to 54.1, LAMBADA from ~110 to 82.0, and — the tell I was
watching for — downstream *moved*: ARC-Easy 47→53.8, HellaSwag 28→31.5. So the resolution diagnosis was
right: the binding constraint on the earlier rungs was not the off-switch but the inability to express a
graded weight magnitude, and the five-level grid `{−1, −2/3, 0, +2/3, +1}` that the int2 fill actually
realizes (its `×1.5` scaling plus half-to-even rounding lands on five symbols, not the four its
docstring names) fixed exactly that. But the win also tells me where the *next* limit is. The thing that
moved the model was giving each weight more places to land; the thing I have *not* touched on any rung is
*where those places are*. On all three baselines the grid is anchored by one number per tensor — the
absmean `s = mean(|W|)` — and that number is chosen to minimize the squared reconstruction error
`‖W − s·grid‖²`, i.e. to make the discrete weights *close to the float weights*. But I do not care
whether the discrete weights are close to the float weights; I care whether they make the *loss* small.
Those are different objectives, and at two or three bits, where only a handful of levels exist per layer,
the gap between them is exactly the gap between int2 and a usable model.

Let me make the deficiency concrete, because it is the whole motivation. With so few levels, the single
scalar `s` decides everything about level placement: too small and I clip away the tails of the weight
distribution, throwing the large-magnitude weights into the top level and losing them; too large and I
waste my precious few levels on coarse spacing, so the dense core of the distribution near zero gets one
level where it wanted three. The absmean picks `s` to balance reconstruction across the whole tensor —
but reconstruction error weights every weight equally, while the loss does not: some weights matter
enormously and some not at all, and the right `s` for the loss is the one that places levels where the
loss-important weights are, not where the average weight is. There is no formula for that `s`. So I
should stop computing it from a statistic and *learn* it — make the per-layer step a trainable parameter,
optimized jointly with the weights against the actual next-token cross-entropy. That is the move: a
single extra scalar per `BitLinear`, the step size, trained by the same AdamW that trains the weights, so
the network tunes its own quantization grid to minimize the thing I measure.

The obstacle is immediate and it is the same wall every rung hit, but sharper. The quantizer is
`ŵ = round(clip(w/s, −Q_N, Q_P)) · s`. The round is flat almost everywhere, so the path from `s` through
`w/s` into the integer code has zero ordinary derivative — ordinary backprop sees the final multiply by
`s` but is blind to how `s` moves weights *toward or away from bin transitions*, which is the whole
reason `s` matters. The standard fix for the round is the STE I have used throughout: treat round as the
identity on the backward pass. But here I must be more careful than on the baselines, because they only
needed a gradient to the *weight*; now I need a correct gradient to the *step* too, and the easy thing —
cancel the round entirely — would zero out the interior step-gradient and learn nothing useful. So I
apply STE to the round node only, and differentiate the divide, the clip, and the multiply honestly.

Take the interior region, where `−Q_N < w/s < Q_P` so the clip is inactive: `ŵ = round(w/s)·s`.
Differentiate w.r.t. `s` by the product rule: `∂ŵ/∂s = [∂round(w/s)/∂s]·s + round(w/s)`. STE says treat
round as identity for the first term, so `round(w/s) ≈ w/s` there and `∂(w/s)/∂s = −w/s²`; times `s`
gives `−w/s`. The second term is `round(w/s)`. So in the interior `∂ŵ/∂s = round(w/s) − w/s`. Write
`z = w/s`, `n = round(z)`, `r = z − n` (the residual, between about `−½` and `½`): then
`∂ŵ/∂s = n − z = −r`, the negative signed residual between `z` and the level it rounds to. Stare at that,
because it is the payoff. The step-size gradient is *largest in magnitude exactly when `z` sits near a
bin transition* (`|r|` near ½) and *zero when `z` sits right on a level* (`r = 0`). That is precisely the
sensitivity I want: a weight near a transition is the one a small change in `s` will flip to a different
integer code, producing a large jump in `ŵ`, so its gradient to `s` should be large. The fixed-absmean
baselines had no version of this at all — their `s` was a constant of the weights, not a parameter, so it
could never respond to where the weights sit relative to the grid. The clipped regions are simpler: if
`w/s ≤ −Q_N` then `ŵ = −Q_N·s` and `∂ŵ/∂s = −Q_N`; if `w/s ≥ Q_P` then `∂ŵ/∂s = Q_P`. And the data
gradient is the usual STE: `∂ŵ/∂w = 1` inside the range, `0` outside (clipped weights get no gradient,
which is correct — pushing a clipped latent weight further changes nothing in the forward pass).

There is one more thing I have to get right or the learned step will destabilize training, and it is the
part the careless versions of this idea miss. I now have one scalar `s` per layer being optimized by the
same AdamW, with the same global learning rate, as the layer's hundreds of thousands of weights. Training
behaves well when, across parameters, the ratio of update magnitude to parameter magnitude is in the same
band — if `s` gets updates that are huge relative to its own size it overshoots and oscillates; if tiny,
it stalls. Let me check the ratio `R = (∇_s L / s) / (‖∇_w L‖ / ‖w‖)`. For the parameter sizes:
`‖w‖ ∝ √N_W` times a typical weight magnitude, and the typical weight magnitude is about `s·√Q_P` (with
`Q_P` levels the step shrinks like `1/√Q_P`), so `‖w‖/s ≈ √(N_W·Q_P)`. For the gradients: `∇_s L` is a
sum over all `N_W` weights of `(∂L/∂ŵ)·(∂ŵ/∂s)`, and treating the per-weight loss gradients as
uncorrelated zero-mean with the `∂ŵ/∂s` factor an order-one per-element constant, `E[(∇_s L)²] ≈
N_W·E[(∂L/∂ŵ)²]` — the *same order* as `E[‖∇_w L‖²]`. So numerator and denominator of `R` differ exactly
by the `‖w‖/s` factor: `R ≈ √(N_W·Q_P)`. That is not 1 — it grows with layer width and precision, so the
step would be over-driven by roughly that factor, worst in the widest layers. The fix is to cancel it
directly: scale the gradient flowing to `s` by `g = 1/√(N_W·Q_P)`. I inject this as a transparent
gradient multiplier with the same detach trick the STE uses — `gradscale(s, g) = (s − g·s).detach() +
g·s` is `s` in the forward pass (the step value is untouched) and multiplies the gradient by `g` in the
backward pass. So `s` is divided by `√(N_W·Q_P)` in its gradient and trains in the same update/parameter
band as the weights, and one AdamW learning rate serves both.

Now adapt all of this to *this task's* substrate, because the canonical LSQ recipe assumes things the
harness does not provide and I must not import them blindly. Canonical LSQ *fine-tunes from a trained
full-precision model* with momentum SGD and cosine decay; here I am pretraining from scratch with AdamW
on the fixed 13,535-iteration schedule, so there is no FP teacher to initialize from — the latent weights
start from the scaffold's `std=0.02` init and the step must be initialized from them. The scale-aware
initializer `s = 2⟨|w|⟩/√Q_P` does exactly that from the initial weights, so I set the step parameter to
`2·mean(|W|)/√Q_P` at construction and let AdamW take it from there; the `√Q_P` denominator correctly
says "more levels ⇒ finer initial step." Canonical LSQ keeps the first and last layers at 8 bits; the
harness ties `wte` to `lm_head` and quantizes every projection uniformly, so I keep the uniform treatment
rather than carve out exceptions the scaffold is not built to express. And I keep the activation path
*identical* to the three baselines — 8-bit per-tensor absmax, `Q_b = 127`, STE — for two reasons: it
isolates the contribution to the *weight* quantizer (the learned step), which is the honest comparison
against int2; and learning a second step for activations from scratch, on top of an unstable-by-nature
low-bit pretraining run, is a risk the controlled experiment does not need. The bit-width I take is
three: signed range `Q_N = 2^{b−1} = 4`, `Q_P = 2^{b−1}−1 = 3`, an eight-level grid `{−4,…,+3}`. Three
bits keeps me firmly in the few-bit native-low-bit regime this task is about, and it deliberately gives
the learned step *more* levels to place than int2's effective five — so the finale tests both halves of
the thesis at once: more levels, and levels placed by the loss rather than by a reconstruction formula.
No SubLN inside the layer, same as every rung: the block's pre-projection `LayerNorm` holds the variance.

So the finale fill adds three things to the scaffold and changes one: a learnable `weight_scale`
parameter per `BitLinear` initialized to `2·mean(|W|)/√Q_P`; the two helper ops `roundpass` (STE round)
and `gradscale` (the `1/√(N_W·Q_P)` step-gradient multiplier); and a `weight_quant` that clips `w/s` to
`[−Q_N, Q_P]`, STE-rounds, and rescales by the gradient-scaled `s`. The activation path and the latent-
weight machinery are unchanged from the baselines (the full module is in the answer).

Here is the bar this has to clear, against int2's measured numbers, and what I would validate. The thing
int2 left on the table is loss-aware level placement, and three bits gives more room to place; so I
expect validation loss *below* int2's 2.4392 — into the low-2.4s or high-2.3s, in the neighborhood of the
~2.37 that the strongest non-baseline run on this task reached, which is itself evidence that headroom
below int2 exists. I expect WikiText-2 below int2's 54.1 and LAMBADA below 82.0, and downstream ARC-Easy
and HellaSwag at or above int2's 53.8 / 31.5 — because graded, loss-placed magnitudes are exactly what
the completion tasks reward. The way I would be wrong is if the learned step does not train cleanly under
AdamW from a scratch init — if the `√(N_W·Q_P)` gradient scaling is even slightly off, the step either
oscillates and the grid thrashes, or stalls and I have a fixed-step quantizer with one wasted parameter
landing right back at int2. I would validate by watching the per-layer `weight_scale` trajectories: if
they move smoothly away from their absmean-flavored init and settle, the learned placement is doing real
work; if they sit pinned at init, the gradient scaling needs the per-layer `N_W` it already uses but
perhaps a different AdamW treatment. The falsifiable claim is simply: a *learned* step at three bits
beats the *fixed* absmean step at int2's effective resolution — val_loss strictly under 2.4392.
