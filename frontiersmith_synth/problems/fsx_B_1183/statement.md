# Two-Shot Wavefront Sign Recovery

## Problem

A mirror's shape error (its "wavefront") is described by four unknown real
coefficients `t0, t1, t2, t3`. Coefficient `t0` is special: it is the mode
that is *degenerate with defocus itself* -- "the mirror bulges out"
(`t0>0`) versus "the mirror dimples in" (`t0<0`). Modes `t1, t2, t3` are
three other, fixed wavefront shapes that do not interact with defocus.

A camera has `S` detector pixels. Pixel `s` has four known response
coefficients `g0[s], g1[s], g2[s], g3[s]` (given in the input, `g0[s] > 0`).
At a chosen defocus setting `d`, pixel `s`'s coherent response and measured
intensity are

```
R_s(d) = g0[s]*(t0 + d) + g1[s]*t1 + g2[s]*t2 + g3[s]*t3
I_s(d) = R_s(d)^2
```

Intensity only ever depends on the *square* of the response, so from a
single image you can measure `|R_s(d)|` but never its sign: a mirror that
"bulges" and one that "dimples" (together with correspondingly adjusted
`t1,t2,t3`) can produce numerically identical images at one defocus setting
-- for a FIXED `d`, the whole coefficient vector `(t0,t1,t2,t3)` and its
"companion" `(-t0-2d, -t1, -t2, -t3)` give exactly the same image. You are
given TWO images taken at two DIFFERENT, known defocus settings `d1 != d2`.
Resolving the true coefficients requires using both together -- the
companion that fools image 1 is a different companion from the one that
would fool image 2, so only the true vector fits both.

Your program must output an estimate `t0_hat, t1_hat, t2_hat, t3_hat`. It is
scored by how well it **predicts a third, held-out image** at a third known
defocus setting `d3` (`d3 != d1`, `d3 != d2`), which you never observe --
only `d3` itself is given, not the intensity there.

*Illustrative example ONLY (not the hidden law you must find; it merely
shows the input/output shapes):* if `t=(1,0,0,0)` and `g0=1,g1=g2=g3=0`
everywhere, then `I_s(d) = (1+d)^2` at every pixel, for any `d`.

## Input (stdin)
```
testId S
d1 d2 d3
g0_1 g1_1 g2_1 g3_1 I1_1 I2_1
...
g0_S g1_S g2_S g3_S I1_S I2_S
```
`S` (8..26) is the number of pixels. `d1, d2, d3` are the three defocus
settings (all distinct). Each of the `S` following lines gives one pixel's
four response coefficients and its two OBSERVED intensities `I1 = I_s(d1)`,
`I2 = I_s(d2)`. `g0[s] > 0` always. `testId` is provided for reference only.

## Output (stdout)
Exactly one line with four real numbers:
```
t0_hat t1_hat t2_hat t3_hat
```

## Feasibility
Exactly four finite (`not nan/inf`) numbers, each with `|value| <= 10`.
Any violation scores `Ratio: 0.0`.

## Objective (maximize)
Using your `t_hat`, the checker forward-computes the predicted held-out
intensity `I_hat_s(d3) = (g0[s]*(t0_hat+d3)+g1[s]*t1_hat+g2[s]*t2_hat+g3[s]*t3_hat)^2`
for every pixel `s`, and compares it against the TRUE held-out intensity
(computed from the true, hidden coefficients). Let `err` be the mean squared
difference over the `S` pixels, and `errB` the mean squared difference for
the "assume no aberration" guess `t=(0,0,0,0)`. Define `F = 1/(1+err)` and
`B = 1/(1+errB)`. The printed score is

```
Ratio = min(900, 100*F/B) / 1000
```

so guessing all-zero scores exactly `0.100`, and better predictions of the
held-out image score higher (capped below `0.900` so there is always
headroom above the reference solutions).

## Constraints
`8 <= S <= 26`, `0.60 <= |t0| <= 0.95`, `|t1|,|t2|,|t3| <= 0.15`,
`0.5 <= g0[s] <= 1.5`, `|g1[s]|,|g2[s]|,|g3[s]| <= 1`. Time limit 5s.

## Example (worked score)
Suppose `S=2`, `d1=-0.3, d2=0.22, d3=0.06`, true `t=(-0.8,0.1,-0.05,0.05)`.
A submission that outputs `t_hat=(0.8+2*0.3, -0.1, 0.05, -0.05)` (the
sign-flipped companion that fits the `d1` image equally well) will predict
`I_hat_s(d3)` far from the true held-out intensity on most pixels -- scoring
much lower than a submission that recovers the true `(-0.8, 0.1, ...)` by
combining both images.
