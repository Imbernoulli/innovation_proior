# Fiber Reach Tuning

## Problem

A straight optical line runs from a transmitter at position `x_0 = 0` km
past `N-1` downstream nodes at strictly increasing positions
`x_1 < x_2 < ... < x_{N-1}` (km). Any node may host an inline amplifier,
but the transmitter (node 0) always does. You choose a subset of nodes to
equip with amplifiers, and for each amplifier you choose one **launch
power** from a given discrete menu. Together these choices decide every
downstream node's signal quality.

Between one active amplifier and the next, the signal launched at power
`P` degrades over distance `d` (km) by fiber loss `alpha` (dB/km):
received signal `= P * 10^(-alpha*d/10)`. Two kinds of noise pile up as
light travels: linear amplifier noise and cubic Kerr nonlinear noise.
Every amplifier you install adds a fixed per-amplifier noise term `c0`,
plus noise that accrues continuously with distance `d` traveled since the
last amplifier at rate `c_ase` per km, plus Kerr nonlinear noise that
accrues as `c_kerr * d * P^3` — proportional to distance **and to the
cube of that span's launch power**. All noise accumulated on prior spans
stays baked in; the SNR any node experiences is its received signal
divided by everything accumulated up to it. A node counts as **reached**
if its SNR is at least a given threshold.

**Why launch power is not "more is better":** the received signal grows
only *linearly* with `P`, but the Kerr noise term grows with `P^3`. For
any fixed span, SNR as a function of `P` rises, peaks, and then falls —
there is an interior best power for that span, and it depends on the
span's own length and on how much noise already piled up before it.
Different spans can have very different best powers; the input's actual
numbers (not just this shape) determine where each optimum sits.

**Objective (maximize):** the count of downstream nodes (`1..N-1`) whose
SNR is at least the threshold.

## Input (stdin)

```
N
x_0 x_1 ... x_{N-1}
alpha c0 c_ase c_kerr
thresh
K
p_1 p_2 ... p_K
```
`N` node positions (`x_0=0`, strictly increasing, km). Four physical
constants. The SNR threshold `thresh`. Then `K` allowed discrete launch
power levels `p_1 < ... < p_K` (a launch power for any amplifier must be
exactly one of these).

## Output (stdout)

```
m
i_1 i_2 ... i_m
P_1 P_2 ... P_m
```
`m`: number of active amplifiers (`1 <= m <= N`). `i_1..i_m`: their
node indices, strictly increasing, `i_1 = 0`. `P_1..P_m`: the launch
power for the span starting at each amplifier (each must be a value
from the input's allowed list). The span for amplifier `k` runs from
`i_k` to `i_{k+1}` (or to `x_{N-1}` if `k=m`).

## Feasibility

`1 <= m <= N`; indices strictly increasing and within `[0, N-1]`;
`i_1 = 0`; each `P_k` exactly matches one allowed level. Any violation,
or non-finite/non-numeric tokens, or a truncated output, scores
`Ratio: 0.0`.

## Scoring

The checker replays your plan through the accumulation model above to
get your reach count `F`. It also computes `B`: the reach achieved by
amplifying *every* candidate node at the *smallest* allowed power (a
plain, dense-but-timid feasible plan). Your score is `100 * F / B`,
capped at `1000`, printed as `Ratio: <score/1000>` on the final line.

## Worked example (illustrative shape only)

Two nodes, `x = [0, 60]`, one span. Suppose (numbers here are just to
show the shape, not the real constants) `alpha=0.2, c0=0.05, c_ase=0.001,
c_kerr=0.00001, thresh=2.0`, powers `{2,5,9}`. At `P=2`: noise is small
but signal is weak. At `P=9`: `c_kerr*d*P^3` dominates and swamps the
gain from the larger numerator. `P=5` lands closer to the interior peak
of `SNR(P) = P*atten(d) / (c0 + c_ase*d + c_kerr*d*P^3)`. Which discrete
level is actually best depends on `d` and on the constants the real
input gives you — that is exactly what you must compute, not assume.

## Constraints

`2 <= N <= 14`, `1 <= K <= 8`, positions and powers are positive
integers, time limit 5s, memory 512MB.
