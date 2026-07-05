# Alloy Flow-Stress Constitutive Law: Extrapolate a Coupon Test to High Strain

## Title
Alloy Flow-Stress Constitutive Law

## Problem
A metallurgy lab is characterising a new structural **alloy**. A small tensile coupon is pulled
in a **stress test** rig, and the lab records the material's **flow stress** `sigma` (in MPa) at
a grid of two controlled quantities:

- `s` — the imposed **plastic strain** (dimensionless), and
- `r` — the **strain rate** (dimensionless, relative to a quasi-static reference).

Materials science says `sigma` obeys a smooth **constitutive law** in these two knobs. It is
**not** a straight line: past yield the alloy work-hardens, but the hardening *rate* decays as
strain accumulates (a concave hardening curve), while raising the strain rate stiffens the
response. There is also an irreducible **yield floor** below which the stress does not fall.

You are given a table of measured points `(s, r, sigma)` taken from the coupon's **mild,
low-strain / low-rate operating envelope** — the safe region where repeated tests do no damage.
Your job is to **discover a closed-form constitutive law** `sigma = f(s, r)` that not only fits
these points but, crucially, **extrapolates** to a **hidden high-strain, high-rate regime** (the
alloy pushed toward necking and impact loading) that lies entirely outside the sampled region.

## Input (stdin)
```
Ntrain
s_1 r_1 sigma_1
s_2 r_2 sigma_2
...
s_Ntrain r_Ntrain sigma_Ntrain
```
- Line 1: integer `Ntrain`, the number of measured operating points.
- Each following line: three real numbers `s r sigma` (strain, strain rate, flow stress in MPa).
- All `s > 0`, `r > 0`, `sigma > 0`. The measurements contain small multiplicative noise.

## Output (stdout)
A single line: a **closed-form expression** for `sigma` as a function of the variables `s` and
`r`, written as a Python expression string.

Allowed tokens:
- variables `s` and `r`;
- numeric literals (e.g. `0.85`, `3`, `1e-2`);
- binary operators `+  -  *  /  **`;
- unary `+` / `-`;
- parentheses;
- the constants `pi`, `e`;
- the functions `exp`, `log`, `log10`, `sqrt`, `abs`, `pow`, `sin`, `cos`.

Nothing else (no other names, attributes, indexing, comprehensions, or calls to other functions).

**Example of the OUTPUT FORMAT ONLY** (this is an *illustrative shape*, deliberately a
different, unrelated form — it is **NOT** the hidden law; do not submit it):
```
250 + 30 * sin(s) + log(1 + r)
```

## Feasibility
The submitted expression must:
- parse using only the allowed tokens above;
- evaluate to a **finite real number** at every held-out evaluation point (positive `s`, `r`);
- contain no `nan`/`inf` and raise no math error on the held-out inputs.

Any violation scores `0`.

## Objective
**Minimise** the **root-mean-square extrapolation error** of your law on a hidden **held-out
region** of larger `s` and larger `r` (the high-strain, high-rate regime), regenerated
deterministically by the grader. Simpler laws are preferred: a mild complexity penalty applies to
very large expressions, so a compact law that generalises beats an over-parameterised one that
memorises the training noise.

## Scoring
Deterministic. Let `F` be the (complexity-adjusted) held-out RMSE of your expression and let `B`
be the held-out RMSE of the grader's internal baseline (a single product power-law
`sigma = k * s^a * r^b` fitted by log-linear regression on the training table). The score is

```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```

Reproducing the baseline scores ≈ `0.10`; a law that extrapolates ~10x better caps at `1.0`.
Irreducible measurement noise on the held-out region keeps the maximum below `1.0`.

## Constraints
- `Ntrain` = 48 (an 8x6 grid of strain x strain-rate).
- Held-out region: 16 points at strictly larger `s` and `r` than any training point.
- Expression length ≤ 5000 characters.
- Grader runs in well under the time/memory limits (pure CPU, no randomness in the score).

## Example (worked score)
Suppose the true flow stress in the high-strain regime has RMS magnitude ~700 MPa, the internal
log-linear baseline mispredicts the held-out region with RMSE `B = 90.0`, and your submitted law
achieves held-out RMSE `F = 22.5` with no complexity penalty. Then
`sc = min(1000, 100 * 90.0 / 22.5) = 400` and `Ratio = 0.400`. A constant or a law of the wrong
functional shape would land near `F ≈ B`, i.e. `Ratio ≈ 0.10`.
