# Palate-Adaptation Flight Order

## Problem
A judge tastes a flight of `N` wine samples, each with a true intensity
`V_i` (`1..100`). You choose the **presentation order**, and may splice in
up to `K` **palate-cleanser** breaks anywhere in the sequence. Every sample
`1..N` must be tasted exactly once; cleanser breaks are not samples.

The judge's palate carries two kinds of state between tastings, tracked
through a running adaptation level `a` (an exponential moving average of
recently-tasted true intensities, starting at the neutral `CENTER`):

**Gain-control adaptation.** Just before tasting a sample, let
`dev = |a - CENTER*SCALE|`. The gain is
`gain = SCALE*D^3 / (D^3 + dev^3)` (in `[0, SCALE]`, `SCALE=1000`) — wide
open while `a` stays within roughly `D` (scaled) of center, collapsing
fast once `a` drifts well past it. The perceived base reading is
`CENTER + gain*(V - CENTER)` (scaled): the farther `a` has drifted, the
more the reading compresses toward center regardless of the true value.
After tasting, `a` updates toward the just-tasted value at rate
`ALPHA/SCALE`.

**Sequential contrast carryover.** The reading is further displaced by the
jump from the immediately preceding *tasted* sample's true value: add
`CC_NUM*(V - V_prev)*SCALE / CC_DEN` to the base reading (a big jump from
the previous sample exaggerates the current reading away from the truth).

A palate-cleanser break resets `a` to `CENTER*SCALE` and clears the
contrast memory (the next tasted sample's `V_prev` is `CENTER`), but a
sample right after a cleanser still starts from a neutral, un-adapted
gain.

Some samples are **flagship** (high judging weight `W_i`); the per-sample
perception error is weighted by `W_i`, so mis-reading a flagship costs far
more than mis-reading a filler sample.

## Input (stdin)
```
N K
ALPHA D CC_NUM CC_DEN CENTER
V_1 V_2 ... V_N
W_1 W_2 ... W_N
```

## Output (stdout)
One line: a sequence of tokens, each either a sample index `1..N` (every
index must appear exactly once, in any order) or `0` (a cleanser break).
The number of `0` tokens must not exceed `K`.

## Feasibility
Output scores `Ratio: 0.0` if: it is empty, unparsable, or has a
non-integer token; any token is outside `{0,1,...,N}`; any sample index
`1..N` is missing or repeated; or more than `K` cleanser tokens are used.

## Objective
Simulate the tasting in the printed order (cleansers reset state as
above). At each tasted sample compute the perceived reading, its absolute
error from the true value, weighted by `W_i`, and sum these into a total
error `E`. **Maximize** `F = BOUND - E`, where `BOUND` is fixed per
instance (larger `F` means smaller total perceptual error).

## Scoring
The checker builds its own reference `E_b`: taste the samples in the exact
order given in the input, using no cleansers at all. Let
`margin = max(1, E_b // 10)`, `BOUND = E_b + margin`, `B = margin`. Then
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = max(0.0, sc / 1000.0)
```
Matching the reference order exactly scores `0.1`; cutting the total error
enough to reach `F = 10*B` caps the score at `1.0`.

## Constraints
`1 <= N <= 300`, `0 <= K <= N`, `1 <= ALPHA <= 1000`, `1 <= D`,
`1 <= CC_NUM`, `1 <= CC_DEN`, `1 <= CENTER <= 100`, `1 <= V_i <= 100`,
`1 <= W_i <= 20`. Time limit 5s, memory 512MB.

## Example
`N=3 K=1`, `ALPHA=300 D=20000 CC_NUM=1 CC_DEN=8 CENTER=50`,
`V = [10, 90, 50]`, `W = [1, 1, 1]`. Tasting in input order `1 2 3` with no
cleanser gives total weighted error `12880` (scaled by 1000): sample 2
(`V=90`) follows a low reading, so contrast pushes its perceived value
further from 90; sample 3 (`V=50`) follows that very high reading and
picks up a large contrast displacement of its own. Presenting `1 2 0 3`
instead — same two samples first, then the cleanser before the third —
drops the error to `7880`: sample 3 no longer inherits any contrast pull
from the 90 and tastes with a freshly centered palate. With
`E_b = 12880`, `margin = 1288`, `BOUND = 14168`, `B = 1288`, this scores
`Ratio ≈ 0.488` versus the baseline's `0.1`. The pattern scales with `N`:
it is the sommelier's uninterrupted monotone sweep, not a scrambled or
interrupted one, that eventually drives `a` far from center and keeps it
there.
