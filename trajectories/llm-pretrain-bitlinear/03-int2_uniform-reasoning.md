Ternary barely moved the needle, and that is the most informative result so far. Validation loss fell
from binary's 2.7352 to 2.7213 — a drop of 0.0139, about half a percent — with WikiText-2 from 81.3 to
77.9 (a 4.1% improvement, the largest of the lot) and LAMBADA essentially flat at 109.8 (110.2 before, a
0.3% move inside noise). Downstream did nothing or slightly regressed: ARC-Easy 46.7 against binary's
47.0, HellaSwag 28.4 against 28.7, PIQA 59.7 against 58.9, WinoGrande 50.8 against 51.5 — every one of
those swings is a few tenths of a point, which at a single seed is noise. So adding the explicit zero
bought back almost nothing measurable. That falsifies the hypothesis I went into rung 2 with. I had argued
the binary floor was dominated by the missing off-switch — that near-zero weights forced to `±β` with
random signs were the main noise source — and that gating them to a clean 0 would recover a real chunk of
the gap to float. The numbers say no: the off-switch alone is worth half a percent of loss, not the
several percent that separates these models from a usable one. And the fact that WikiText-2 moved 4% while
the downstream accuracies moved not at all is itself a clue — the zero cleaned up a little text-modeling
noise but added no new *discriminative* capability, which is exactly what I would expect if resolution,
not the off-state, is the binding constraint. The deficiency that actually bites is somewhere else, and
the flat-to-slightly-worse downstream accuracies point right at it.

Let me set the bar honestly before I design the fix, because a single-seed board tempts over-reading. The only binary→ternary move that clears plausible run-to-run noise is WikiText-2 — a 3.35-perplexity, 4.1% drop; val_loss moved 0.0139 (0.5%), LAMBADA 0.38 (0.3%), and every downstream swing is a few tenths, well inside what one seed's initialization and data order can manufacture. So the truthful one-line summary of rung 2 is: the off-switch bought one metric's worth of text-modeling cleanup and nothing discriminative. That fixes what int2 has to do to count as a real result rather than a second half-percent shuffle. It must move val_loss by an amount that *dwarfs* 0.0139 — a change measured in tenths of a nat, not thousandths — because a gain of the ternary size would tell me resolution is as weak an axis as the off-switch was. And it must move at least one downstream accuracy by more than a point, since those are the metrics that have been pinned near their floors through two rungs and are the actual test of whether graded magnitude buys discrimination rather than just a lower cross-entropy. If int2 lands another 0.5% below ternary with flat downstream, resolution was not the binding constraint either and I was wrong twice, and the diagnosis would have to jump off the weight grid entirely.

Here is the re-diagnosis. Ternary can say a weight is `0`, `+β`, or `−β`. What it *cannot* say is that a
weight is small-but-not-zero — there is no `+β/3`-flavored level, no way to express a connection that is
present but weak. So every surviving connection is slammed to full magnitude `±β`, and the only
expressive choice the layer has per weight is "off, or full-strength in one of two directions." The zero
removed the random-sign noise on the truly-dead weights, which is why the loss improved at all; but the
*resolution* problem — the inability to represent a graded magnitude — is untouched, and that is what
the completion tasks need. HellaSwag and ARC-Easy probe whether the model has learned finely-weighted
combinations of features; a three-level weight cannot encode a finely-weighted anything. This lines up
with the ARC-vs-HellaSwag split I flagged after binary: coarse associative signal survives, fine
discrimination does not, and neither binary nor ternary can fix the second because both lack graded
magnitude. So the next bit I spend should not buy another structural state — it should buy *magnitude
resolution*. I want a grid with more levels per weight, spaced to give the layer graded magnitudes.

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

Let me convert the realized grid back into a threshold statement in the original weight units, because
that tells me which weights land where and lets me check the levels are actually populated. Working in
`w_n = W/s` with `s = mean(|W|)`: a weight maps to `0` when its scaled value `1.5·w_n` rounds to `0`,
i.e. `|1.5·w_n| ≤ 0.5` (the `= 0.5` case ties to even `0`), so `|w_n| ≤ 1/3`, i.e. `|W| ≤ s/3`. It maps
to `±2/3` when `1.5·w_n` rounds to `±1`, i.e. `0.5 < |1.5·w_n| < 1.5`, so `1/3 < |w_n| < 1`, i.e.
`s/3 < |W| < s`. And it maps to `±1` when `1.5·w_n ≥ 1.5` (rounds to `±2`, clamped), i.e. `|w_n| ≥ 1`,
i.e. `|W| ≥ s`. So the zero-threshold here is `s/3`, *lower* than ternary's `s/2` — the five-level grid
gates off *fewer* weights than ternary did, and reassigns the freed band to the new `±2/3` level. Put init
numbers on it: `s/3 = 0.266σ`, and a Gaussian puts `2Φ(0.266) − 1 ≈ 21%` inside, so ~21% gate to zero (down
from ternary's 31%). The `±2/3` band `0.266σ < |W| < 0.798σ` holds `2(Φ(0.798) − Φ(0.266)) ≈ 37%`, and the
`±1` band `|W| > 0.798σ` holds `2(1 − Φ(0.798)) ≈ 42%`. So at init the five levels are populated ~21% /
37% / 42% across `{0, ±2/3, ±1}` — and the key number is that *37%* of the weights land on the brand-new
`±2/3` level, the "present-but-weak" magnitude that neither binary nor ternary could express. More than a
third of the layer immediately uses the capability ternary lacked. That is the mechanism I am betting on,
and it is not a paper abstraction — it is a third of every projection matrix.

It is worth being precise about why the round-half-to-even convention is what creates the extra zero
level rather than the symmetric four-level grid the docstring evidently intended. If round broke ties away
from zero, `±0.5` would go to `±1` and `±1.5` to `±2`, the outer clamp would pin `±2`→`±1.5`, and after
dividing by `1.5` the realized set would be `{−1, −1/3, +1/3, +1}` with no zero — the textbook 2-bit
grid. Half-to-even flips two of those mappings: `0.5`→`0` (even) and `1.5`→`2` (even), which both
*adds* the `0` level and *removes* the `±1/3` levels in favor of `±2/3`. So the five-level grid is an
artifact of the rounding convention interacting with the `×1.5` scaling, not a designed choice — but it
is exactly what the model is trained on, and it happens to land on a strictly more capable grid than the
nominal one. Let me trace a few weights through to be sure I have the map right. `w_n = 0.3`: `×1.5 =
0.45`, clamp leaves it, `round(0.45) = 0`, `/1.5 → 0`. `w_n = 1/3 ≈ 0.333`: `×1.5 = 0.5`, `round(0.5)` ties
to even `0`, `→ 0` — so the boundary weight gates off. `w_n = 0.5`: `×1.5 = 0.75`, `round = 1`, `/1.5 →
2/3`. `w_n = 0.9`: `×1.5 = 1.35`, `round = 1`, `/1.5 → 2/3`. `w_n = 1.0`: `×1.5 = 1.5`, `round(1.5)` ties to
even `2`, clamp to `1.5`, `/1.5 → 1`. `w_n = 3`: `×1.5 = 4.5`, clamp to `2`, `round = 2`, clamp `1.5`, `/1.5
→ 1`. Every case lands on `{0, ±2/3, ±1}` exactly as the threshold analysis said. I take the grid the code
computes, and the grid the code computes is five symmetric levels.

I can see exactly where the intent and the arithmetic part ways, which makes me trust the reading. The
`×1.5` is a standard trick: the intended four-level grid `{−1, −1/3, +1/3, +1}` has spacing `2/3`, so to
round onto it you rescale by `3/2` to make the spacing 1 and then round — *if* you round to the nearest
member of the intended targets. But the intended targets after `×1.5` are the half-integers
`{−1.5, −0.5, +0.5, +1.5}`, and rounding to nearest *half-integer* is not what `round` does — `round` snaps
to nearest *integer*, whose targets are `{−2, −1, 0, 1, 2}`. So the code rounds to a grid that is offset by
half a step from the one it meant to hit, and that offset is precisely what inserts the `0` (an integer,
hit by the round) and drops the `±1/3` half-integers (never hit). The `×1.5` is right for making the
spacing unit; the mismatch is that a four-level grid centered *between* integers cannot be reached by
integer rounding. That is a clean, mechanical account of the artifact, and it tells me the five-level grid
is robust — it is not a floating-point accident that might vanish, it is what integer rounding onto this
scaled range always produces.

And the artifact is strictly favorable, which is worth stating because it is the reason this rung is
special. Compare the grid the author *intended* against the one the code *runs*. The intended
`{−1, −1/3, +1/3, +1}` has four graded magnitudes but *no zero* — it would have thrown away ternary's
off-switch to buy resolution, testing one axis while sacrificing the other. The realized
`{−1, −2/3, 0, +2/3, +1}` keeps the zero *and* adds graded magnitude — it improves on ternary along both
axes at once. At init the intended grid would gate 0% of weights off (no zero level, every weight forced
to one of four magnitudes), whereas the realized grid keeps ~21% off and still gives the survivors two
magnitudes. So the code, by accident, lands on the union of everything the prior two rungs offered rather
than a trade between them. That is why I expect a jump and not a wash: I am not swapping the off-state for
resolution, I am adding resolution on top of the off-state.

Two checks that this grid is the strict improvement I think it is. First, the alphabet's entropy at init:
with the `21% / 18.5%-each / 21%-each` split across `{0, ±2/3, ±1}`, `H = −0.21·log₂0.21 −
2·0.185·log₂0.185 − 2·0.21·log₂0.21 ≈ 0.47 + 0.90 + 0.95 ≈ 2.32` bits, essentially the maximum `log₂5 ≈
2.32` — so all five levels are well-used, none is a wasted symbol. Second, the grid *contains* ternary's:
`{−1, 0, +1} ⊂ {−1, −2/3, 0, +2/3, +1}`, so a correctly-trained five-level model can always reproduce
ternary by simply never selecting `±2/3`, which means — barring optimization noise — it should not do
*worse* than ternary, and any gain is pure upside from the two new levels. That containment is why I can be
confident about the direction of the move even though I cannot predict its size.

The storage premium for those two extra levels is modest, which keeps the move honest as a *low-bit*
method rather than a slide back toward full precision. Five levels pack efficiently because `5³ = 125 ≤
256`, so three symbols fit in one byte at `8/3 ≈ 2.67` bits per weight (the information-theoretic floor is
`log₂5 ≈ 2.32`); the 355M projection weights are then ~118 MB against ternary's 71 MB and `bfloat16`'s 710
MB — still a 6× compression, about 47 MB more than ternary. So I am spending ~0.74 extra bits and ~47 MB
to add graded magnitude on top of the off-switch, which is a good trade if resolution is really the
binding constraint.

Why does this matter so much for my expectation? Because I just diagnosed that the binding constraint is
magnitude resolution, and this grid attacks exactly that while *also* keeping the off-switch I confirmed
is worth a (small) real amount. A weight can now be off (`0`), weakly on (`±2/3`), or strongly on
(`±1`), in either sign — five symbols, `log₂5 ≈ 2.32` bits, against ternary's three at 1.58. That is a
qualitatively richer alphabet than three slammed levels: the layer can finally express a
present-but-weak connection, which is the thing ARC-Easy and HellaSwag were starved of. And the extra
0.74 bits per weight, multiplied across the 355M projection weights, is a large amount of restored
representational budget. I should expect this rung to break out of the 2.72 plateau that binary and
ternary sat on, not inch below it — the gap between three and five graded levels is far bigger than the
gap between two levels and three, and unlike the binary→ternary step it adds resolution rather than only
a structural state.

Let me make sure the rest of the layer is sound for this grid, because the scale and the STE have to be
right or the resolution gain evaporates. The scale stays the absmean `s = mean(|W|)` — the L2-optimal
per-tensor unit from the same least-squares argument that gave the sign grid its scale, cheap and
outlier-robust, and it places the typical weight near the grid's `±1` so the bulk of the distribution
falls inside the representable range. Contrast absmax, which one outlier weight would inflate, crushing
all ordinary weights toward a single level and wasting the five levels on representing the outlier; so
absmean for weights, deliberately. Check the variance stays sane with the new levels: `E[g²]` at init is
`0.37·(2/3)² + 0.42·1 ≈ 0.163 + 0.425 ≈ 0.587`, so `Var(y) = n·s²·E[g²]·E[x̃²] = n·(2/π)σ²·0.587·E[x̃²] ≈
0.37·n·σ²·E[x̃²]` — quieter still than ternary's 0.44× and binary's 0.64×, because the `±2/3` levels carry
less squared magnitude than `±1`, but the same order and cleanly renormalized by the block's LayerNorm.
The two non-differentiable operations — the round and the two clamps — are bridged by the same
detached-difference STE I used on both prior rungs: `(w_rounded − w_scaled).detach() + w_scaled` makes the
forward use the grid value and the backward pass identity to the normalized weight, and the clamp inside
`w_rounded` defines the forward saturation (a scaled weight driven past `±1.5` is pinned, stops receiving
a magnitude-increasing gradient through the round). The float latent weight keeps accumulating the
optimizer's tiny noisy steps exactly as before; nothing about the latent-weight machinery changes from
rung 1.

One thing about this rung's `×1.5 / ÷1.5` wrapping worried me — does the extra scaling distort the
gradient the way it distorts the forward levels? — so let me differentiate it and check. The returned
weight is `w_q = w_ste / 1.5` with `w_ste = (w_rounded − w_scaled).detach() + w_scaled` and `w_scaled =
1.5·(W/s)`. The detached parenthesis contributes nothing, so `∂w_q/∂W = (1/1.5)·∂w_scaled/∂W =
(1/1.5)·(1.5/s) = 1/s` — the two factors of `1.5` cancel exactly, and the gradient to the latent weight is
`1/s`, identical to ternary and binary. Since the layer then rescales its output by `s`, the net gradient
through the quantizer is order 1, the same wire every rung has used. So the `×1.5` is a pure *forward*
device to make integer rounding realize the intended spacing; it is gradient-neutral, and the STE is
unchanged. Good — the resolution gain is not bought with a distorted backward pass.

I should refine one word in that check, because "the gradient to the latent weight is `1/s`, identical to ternary and binary" is only true in the *interior*, and where the two rungs part ways in the tails is worth pinning down. Look at where each fill puts its clamp. The ternary fill clamped `w_normed` *before* the detached round — `clamp(−1, 1)` sat in the differentiable path — so a weight past `|w_n| > 1` had a saturated forward value *and* a zeroed backward gradient; at the `std=0.02` init that knee is `|W| > γ`, roughly 42% of the tensor parked with no data-gradient, only its sign under active control. This fill puts the `clamp(−2, 2)` *inside* `w_rounded`, which is detached, while the differentiable branch adds back the *unclamped* `w_scaled`. Differentiate it: `∂w_q/∂W = (1/1.5)·∂w_scaled/∂W = (1/1.5)·(1.5/s) = 1/s` for *every* weight, including the extreme tails whose forward value is pinned to `±1`. So int2 saturates the forward value without saturating the gradient. That matches binary — whose bare-sign STE also passes `1` everywhere, with no clamp at all — but differs from ternary, which alone among the three zeroes the gradient on its saturated band. The tradeoff is real and I will not pretend it is free: a weight clipped to the top level keeps receiving an outward push it cannot act on, the classic STE latent-inflation pathology, so some latent magnitudes will drift with no forward consequence. But the upside is that no weight freezes — the whole tensor stays under gradient where ternary parked its `±1` band. I cannot cleanly separate how much of the leap I expect is the two new levels versus this un-parked tail; both push the same way, and the added levels are the larger, *designed* effect, so I attribute the leap primarily to resolution while noting honestly that the backward path also opened up.

Before I commit I should ask whether the cleaner move is just to spend *more* bits — go to three or four
levels-per-side with a proper uniform integer grid — rather than take this particular five-level artifact.
The answer for this rung is no, on two grounds. First, ladder discipline: I want to change exactly one
thing from ternary, and the smallest change that adds graded magnitude is this two-bit-ish grid; jumping
straight to a wider integer grid would confound "does resolution help?" with "how much resolution?".
Second, each extra bit roughly doubles the kernel's level count and the packing complexity, and I have not
yet demonstrated that resolution is even the right axis — that is precisely what this rung tests. So I take
the five-level grid the fill computes, measure whether resolution breaks the plateau, and only then decide
whether more of it is worth the bits.

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
pre-projection `LayerNorm` already holds `E[x̃²] ≈ 1`, so the variance estimate stays at the float
layer's order, no SubLN needed.

Assemble the fill: `weight_quant` returns `(five-level-grid-value, mean(|W|))` via the
multiply-by-1.5 / clamp / round / clamp / divide-by-1.5 path with STE; `activation_quant` is the
unchanged int8 absmax; `forward` runs `F.linear(x_q, w_q)` and rescales by the product of the two scales
before the bias. Recipe inherited — `CONFIG_OVERRIDES` empty, peak LR `6e-4` — so it remains a clean
grid-only comparison (the distilled module is in the answer).

So the delta from ternary is one thing: where ternary offered `{−1, 0, +1}` — off or full-strength —
this grid offers `{−1, −2/3, 0, +2/3, +1}`, adding two intermediate magnitudes while keeping the zero.
Here is what I expect against the measured numbers, and how I would be wrong. I expect a *large* drop in
validation loss, decisively below the 2.72 plateau — well down into the 2.4s if the resolution diagnosis is
right, because graded magnitudes are exactly what the model has been starved of. WikiText-2 should fall
sharply out of the high-70s, plausibly into the 50s or low 60s, and LAMBADA down from ~110 into the 80s or
90s — these are perplexity-of-text metrics and respond directly to better weight fidelity. And critically,
downstream should *move* this time where it did not for ternary: ARC-Easy up from ~47 by several points and
HellaSwag up from ~28 into the low-to-mid 30s, because the finely-weighted feature combinations those tasks
need are now representable — that HellaSwag move is the specific thing I said any real fix has to produce.
If instead int2 lands only a hair below ternary, like ternary landed only a hair below binary, then I was
wrong twice and the bottleneck is not the weight grid at all but the *fixed* absmean scale — a per-tensor
statistic chosen to minimize reconstruction error, not task loss — and the next move would be to stop
fixing the scale by a formula and learn it against the loss instead.
