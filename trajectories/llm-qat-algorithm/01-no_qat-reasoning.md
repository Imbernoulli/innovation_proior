The training-side algorithm is the whole point, but it bolts onto a fixed format — per-group symmetric
round-to-nearest — and the floor I have to start from is that format with *nothing* trained on top of
it: pure post-training quantization, the irreversible rounding applied once to the pretrained weights
and nothing else. Before I can reason about what training buys, I need to know exactly what the
rounding does on its own, because that is the error every later rung is trying to claw back. So the
first rung is deliberately the do-nothing control: zero training steps, materialize the integer model,
measure. Its value is precisely that it isolates the damage of the format from the contribution of any
algorithm.

Let me nail the irreversible step, since it is the substrate the whole ladder stands on. An integer
code `q` is, by itself, just an index; it only becomes a number because I attach a rule `q ↦ r`. The
first thing forced on me is the *shape* of that rule. A layer is a matrix multiply — sums of products
of weights and activations — and I want to do that accumulation on the codes and only convert back at
the end, so the products and sums of the codes must track the products and sums of the reals. The only
`g` for which "multiply the reals" reduces to "do integer work on the codes plus a cheap fix-up" is an
*affine* one: `r = A·q + B`. Anything nonlinear and `g(q1)·g(q2)` stops being expressible through
`q1·q2` plus corrections, and the whole reason for integers evaporates. So the map is affine — forced,
not chosen.

Now look at `B`. Convolutions and the attention/MLP plumbing pad and zero things; in the quantized
world that padding is some code, and whatever real value the code denotes is what actually gets added.
If real 0 is *not* exactly representable, every padded position contributes the same small nonzero
value — the same error, every time, which is a *bias*, the one kind of error that compounds across depth
rather than averaging out. So I need real 0 to land exactly on a code. Reparameterize the same affine
map as `r = S·(q − Z)`: set `r = 0` at `q = Z` and `S` cancels, leaving `Z = −Z`, i.e. the constraint
is satisfied by construction for any `S`. So the correspondence is `r = S(q − Z)`: a positive real step
size `S` and an integer zero-point `Z` (the code that maps to real 0). But a trained weight tensor is
roughly zero-centered and symmetric, so the natural grid is centered at zero and `Z = 0` falls out,
collapsing the map to `r = S·q`. That is not just tidy: in the integer matmul the general scheme
expands `(q1 − Z1)(q2 − Z2)` into the core `q1·q2` plus cross terms in the zero-points that I would have
to subtract off; with `Z = 0` on the weights, all the weight zero-point cross terms vanish, leaving the
leanest kernel. This is the entire reason integer quantization earns its keep — the `O(N³)` accumulation
runs on small integers and only a per-output scalar multiplier `S1·S2/S3` survives in floating point —
and it is why the format is what it is.

That leaves a single knob, the scale `S`. With no appetite for tuning the range against data — this is
the *no-calibration* control — the honest choice is the max-magnitude one: in the signed code range
`qmin = −2^{B−1}`, `qmax = 2^{B−1} − 1`, tie the scale to the positive endpoint, `S = max|w| / qmax`,
the smallest step that covers the weights without clipping. I should name the trade-off this resolves,
because it is the crux of the whole ladder: a *smaller* `S` would clip the range tighter than `max|w|`,
shrinking the rounding error on the bulk at the cost of clipping the few weights beyond the tighter
range to the rail — trading rounding error for clipping error, a balance that requires looking at the
data and optimizing. The max-abs scale is exactly the choice that needs *no* data: the smallest `S` that
covers both rails with no clipping. The extra negative code is simply outside the observed no-clip range.
And round-to-nearest, because for a one-shot conversion it minimizes each element's error,
`|error| ≤ S/2`, with residual variance about `S²/12` under the uniform-within-a-cell model. I
deliberately do *not* reach for stochastic rounding here even though it is pointwise unbiased: its
unbiasedness only earns its keep when the *same* value is rounded over and over, so a tiny per-step bias
accumulates linearly while the extra variance averages down. PTQ rounds each weight exactly once. There
is no accumulation to fight, so stochastic rounding would buy variance for nothing.

The one real refinement is to make the scale per-group rather than per-tensor. A single tensor-wide `S`
is wrecked by the two known one-shot failure modes: weight ranges that differ by over a hundredfold
across output channels — so a narrow channel gets a grid far too coarse for its own magnitude — and
lone outlier weights that inflate `max|w|` for the whole tensor and coarsen everyone. Partitioning each
row into contiguous blocks of `group_size = 128` columns, each with its own `max|w|` and its own `S`,
localizes both: an outlier only coarsens the 128 weights in its own group, and a narrow channel's group
gets a scale matched to its own magnitude. It costs one extra scale per 128 weights — negligible — and
directly attacks both diseases. So the irreversible step is fixed: per-group symmetric
round-to-nearest, `S_g = max|w_g| / qmax`, with a `1e-12` floor on `max|w_g|` so an all-zero group does
not divide by zero.

Now the wall I will be staring at for the rest of the ladder. This conversion is *blind*: it looks only
at the weights, never at the data or the loss, and once it rounds, the error is frozen. The floor is
`S/2`, and at INT2 `qmax = 1`, so `S = max|w_g|` and that floor is `max|w_g| / 2` — half the largest
weight in the group. The signed container has four codes `{−2, −1, 0, 1}`, but with the no-clip scale
and weights inside `[−max|w_g|, max|w_g|]`, the in-range reconstruction points are spaced by
`max|w_g|`; the extra negative code does not make the cells finer. Grouping localizes the outlier and
range-spread damage but does nothing to the underlying coarseness — within a group of 128 weights
snapped onto that spacing, most weights are still thrown onto a brutally coarse grid. There is no
mechanism in this scheme to *repair* the error: I round once and freeze. That is exactly why every
later rung exists — to let the weights (and a learnable grid) move *before* the rounding, which needs
the loss, gradients, and optimization that this control forbids.

So this rung's edit is the one that makes the harness do nothing. The scaffold default fakes-quant in
the forward and would, if I left it, run a 500-step finetune through the straight-through estimator —
but that is already the next rung's idea (STE QAT), not this one. The point of the PTQ control is *no
training at all*, so I set `num_steps = 0` and `learning_rate = 0`, which turns the AdamW loop into a
no-op and leaves the pretrained fp32 weights untouched. Then the fixed post-step does the only real
work: it calls the no-grad `quantize_dequantize_weight` once on every wrapped linear, snapping the
fp32 weights to their per-group integer grid, and evaluates. Because training is off, the wrapper's
forward does not need to fake-quant inside any loop — by eval time the weight has already been replaced
by its dequantized version, so the wrapper just calls a plain linear on the already-quantized weight.
This is the one place the wrapper diverges from the scaffold default: the default's `forward` applies
`fake_quantize_weight` every call (it expects to be trained through), but the PTQ control's `forward`
is the bare `F.linear` on `self.linear.weight`. The differentiable `fake_quantize_weight` is still
present and correct — it just is never exercised, because zero gradients are ever taken. And I keep the
deliberate exclusions: activations stay full precision (this is weight-only), and the LM head
(`embed_out` for this GPTNeoX backbone) is restored to a plain Linear so the accuracy-critical output
projection escapes quantization. The full scaffold module is in the answer.

What must this floor do, and why run it? At INT4 the grid is fine enough — 16 codes, in-range points
spaced by `max|w_g|/7` — that even blind RTN should leave perplexity only modestly above full
precision; this is the regime where the older 8-bit-style inference work lived, and the damage is real
but survivable. At INT3 — 8 codes, spacing `max|w_g|/3` — the floor climbs sharply and I expect a clear
degradation, perplexity well above the `fp16_ppl = 13.2033` baseline. At INT2 — 4 codes, spacing
`max|w_g|`, floor `max|w_g|/2` — I expect the model to be effectively destroyed: every weight smashed
onto a four-level grid is a perturbation so large that the composed function bears little resemblance
to the trained one, and perplexity should blow up by orders of magnitude rather than points. That
collapse is the entire motivation for the ladder. The diagnosis is already pointed: this is not a
learning problem and not a tuning problem — it is a *no-repair* problem. The grid is fixed and the
rounding is irreversible, so the only leverage is to move the weights before I round, which means
turning training back on. The very next rung does exactly that, with the cheapest possible mechanism
for sending a gradient through the round: the straight-through estimator the scaffold default already
carries.
