Ternary barely moved the needle, and that is the most informative result so far. Validation loss fell
from binary's 2.7352 to 2.7213 — a drop of 0.014, about half a percent — with WikiText-2 from 81.3 to
77.9 and LAMBADA essentially flat at 109.8, and downstream ARC-Easy and HellaSwag if anything a hair
*worse* (46.7 vs 47.0, 28.4 vs 28.7, both inside noise). So adding the explicit zero bought back almost
nothing. That falsifies the hypothesis I went into rung 2 with. I had argued the binary floor was
dominated by the missing off-switch — that near-zero weights forced to `±β` with random signs were the
main noise source — and that gating them to a clean 0 would recover a real chunk of the gap to float.
The numbers say no: the off-switch alone is worth half a percent of loss, not the several percent that
separates these models from a usable one. The deficiency that actually bites is somewhere else, and the
flat-to-slightly-worse downstream accuracies point right at it.

Here is the re-diagnosis. Ternary can say a weight is `0`, `+β`, or `−β`. What it *cannot* say is that a
weight is small-but-not-zero — there is no `+β/3`-flavored level, no way to express a connection that is
present but weak. So every surviving connection is slammed to full magnitude `±β`, and the only
expressive choice the layer has per weight is "off, or full-strength in one of two directions." The zero
removed the random-sign noise on the truly-dead weights, which is why the loss improved at all; but the
*resolution* problem — the inability to represent a graded magnitude — is untouched, and that is what
the completion tasks need. HellaSwag and ARC-Easy probe whether the model has learned finely-weighted
combinations of features; a three-level weight cannot encode a finely-weighted anything. So the next bit
I spend should not buy another structural state — it should buy *magnitude resolution*. I want a grid
with more levels per weight, spaced to give the layer graded magnitudes.

The natural next step is two genuine bits — four levels — but here I have to read the task's actual fill
carefully rather than the textbook 2-bit grid, because they differ, and the difference is the whole
reason this rung will leap rather than crawl. The clean textbook 2-bit grid is the symmetric uniform set
`{−1, −1/3, +1/3, +1}`, four levels at exactly `log₂4 = 2` bits, *without* an exact zero — it trades
ternary's off-state for four graded magnitudes. That would test resolution-without-the-zero. But the
fill in front of me does something subtly different, and I worked out what it actually computes rather
than trust its docstring. It normalizes by the absmean, multiplies by `1.5` so the grid lands at
`{−1.5, −0.5, 0.5, 1.5}` in scaled space, clamps the *scaled* value to `[−2, 2]`, rounds to the nearest
integer, clamps the result to `[−1.5, 1.5]`, then divides back by `1.5`. Trace where the levels land:
round sends a scaled value to one of the integers `{−2, −1, 0, 1, 2}`; the outer clamp pins `±2` back to
`±1.5`; and PyTorch's round is half-to-even, so `±0.5` rounds to `0` (zero is even) while `±1.5` rounds
to `±2` (two is even). Divide the surviving integers by `1.5` and the realized grid is
`{−1, −2/3, 0, +2/3, +1}` — **five** levels, not four, and it *contains an exact zero*. So this fill is
neither the 4-level paper grid nor ternary; it is a 5-level symmetric grid at `log₂5 ≈ 2.32` bits that
keeps the off-state *and* adds two intermediate magnitudes `±2/3`. That is the best of both prior rungs
at once: the zero that ternary had, plus the graded magnitude that ternary lacked. The docstring claims
`{−1, −1/3, +1/3, +1}`, but the arithmetic the code runs produces the five-level grid, and the
measurement is of the code, so I reason about the five levels.

It is worth being precise about why the round-half-to-even convention is what creates the extra zero
level rather than the symmetric four-level grid the docstring evidently intended. If round broke ties away
from zero, `±0.5` would go to `±1` and `±1.5` to `±2`, the outer clamp would pin `±2`→`±1.5`, and after
dividing by `1.5` the realized set would be `{−1, −1/3, +1/3, +1}` with no zero — the textbook 2-bit
grid. Half-to-even flips two of those mappings: `0.5`→`0` (even) and `1.5`→`2` (even), which both
*adds* the `0` level and *removes* the `±1/3` levels in favor of `±2/3`. So the five-level grid is an
artifact of the rounding convention interacting with the `×1.5` scaling, not a designed choice — but it
is exactly what the model is trained on, and it happens to land on a strictly more capable grid than the
nominal one. I take the grid the code computes, and the grid the code computes is five symmetric levels.

Why does this matter so much for my expectation? Because I just diagnosed that the binding constraint is
magnitude resolution, and this grid attacks exactly that while *also* keeping the off-switch I confirmed
is worth a (small) real amount. A weight can now be off (`0`), weakly on (`±2/3`), or strongly on
(`±1`), in either sign — five symbols, `log₂5 ≈ 2.32` bits, against ternary's three at 1.58. That is a
qualitatively richer alphabet than three slammed levels: the layer can finally express a
present-but-weak connection, which is the thing ARC-Easy and HellaSwag were starved of. And the extra
0.74 bits per weight, multiplied across the 355M projection weights, is a large amount of restored
representational budget. I should expect this rung to break out of the 2.72 plateau that binary and
ternary sat on, not inch below it — the gap between three and five graded levels is far bigger than the
gap between two levels and three.

Let me make sure the rest of the layer is sound for this grid, because the scale and the STE have to be
right or the resolution gain evaporates. The scale stays the absmean `s = mean(|W|)` — the L2-optimal
per-tensor unit from the same least-squares argument that gave the sign grid its scale, cheap and
outlier-robust, and it places the typical weight near the grid's `±1` so the bulk of the distribution
falls inside the representable range. Contrast absmax, which one outlier weight would inflate, crushing
all ordinary weights toward a single level and wasting the five levels on representing the outlier; so
absmean for weights, deliberately. The two non-differentiable operations — the round and the two clamps
— are bridged by the same detached-difference STE I used on both prior rungs: `(w_rounded −
w_scaled).detach() + w_scaled` makes the forward use the grid value and the backward pass identity to
the normalized weight, and the clamp inside `w_rounded` defines the forward saturation (a scaled weight
driven past `±1.5` is pinned, stops receiving a magnitude-increasing gradient through the round). The
float latent weight keeps accumulating the optimizer's tiny noisy steps exactly as before; nothing about
the latent-weight machinery changes from rung 1.

The activation path stays *identical* to the two earlier rungs — and this is the controlled-experiment
spine of the whole ladder. It is 8-bit symmetric per-tensor absmax, `Q_b = 127`, clip `[−127, 127]`,
STE, with the dequant scale `max|x|/127`. I keep activations at int8 rather than pushing them low,
because the diagnostic from the quantization literature is decisive: weights are far easier to quantize
than activations — weight distributions are flat and round cleanly, while activations carry per-token
outlier channels with huge dynamic range — so the contribution should stay in the *weight* grid and the
activations should be only as low as is safe. And I use absmax rather than absmean here, the opposite of
the weight choice, because the failure modes are opposite: a clipped activation is a destroyed
activation, whereas a clipped weight is just a slightly misplaced grid assignment, so the activation
scale must be set by the maximum to guarantee nothing clips. Holding this path fixed across all three
rungs means that when int2 beats ternary, I will know it was the five-level weight grid and nothing
else. And as on the earlier rungs there is no normalization inside `BitLinear`: the block's
pre-projection `LayerNorm` already holds `E[x̃²] ≈ 1`, so the variance estimate `Var(y) = n·s²·E[g²]·
E[x̃²]` (with `g` a grid value, `E[g²]` a fixed grid constant) stays at the float layer's order, no
SubLN needed.

Assemble the fill: `weight_quant` returns `(five-level-grid-value, mean(|W|))` via the
multiply-by-1.5 / clamp / round / clamp / divide-by-1.5 path with STE; `activation_quant` is the
unchanged int8 absmax; `forward` runs `F.linear(x_q, w_q)` and rescales by the product of the two scales
before the bias. Recipe inherited — `CONFIG_OVERRIDES` empty, peak LR `6e-4` — so it remains a clean
grid-only comparison (the distilled module is in the answer).

So the delta from ternary is one thing: where ternary offered `{−1, 0, +1}` — off or full-strength —
this grid offers `{−1, −2/3, 0, +2/3, +1}`, adding two intermediate magnitudes while keeping the zero.
Here is what I expect against the measured numbers, and how I would be wrong. I expect a *large* drop in
validation loss, decisively below the 2.72 plateau — into the mid-2.4s if the resolution diagnosis is
right, because graded magnitudes are exactly what the model has been starved of. WikiText-2 should fall
sharply from ~78 toward the mid-50s and LAMBADA from ~110 toward the low 80s — these are
perplexity-of-text metrics and respond directly to better weight fidelity. And critically, downstream
should *move* this time where it did not for ternary: ARC-Easy up from ~47 toward the low-to-mid 50s and
HellaSwag up from ~28 toward the low 30s, because the finely-weighted feature combinations those tasks
need are now representable. If instead int2 lands only a hair below ternary, like ternary landed only a
hair below binary, then I was wrong twice and the bottleneck is not the weight grid at all but the
*fixed* absmean scale — a per-tensor statistic chosen to minimize reconstruction error, not task loss —
and the next move would be to stop fixing the scale by a formula and learn it against the loss instead.
