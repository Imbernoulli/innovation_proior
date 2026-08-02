# Graded Bond-Line Design Under a Thermal-Cycling Edge Concentration

## Problem
Two substrates with mismatched thermal-expansion coefficients are bonded
along a line split into `N` discrete segments. A library of `M` candidate
adhesive layer types is given, sorted by increasing shear stiffness `k_j`;
each type also has a shear-strength capacity `s_j` (also increasing in
`k_j` -- a stiffer adhesive is individually a stronger one too). You choose
one adhesive type for every segment: this per-segment choice **is** the
artifact -- a compliant-layer grading profile along the bond line.

**The stress model (already fully specified; you do not need to re-derive
it).** Using the chosen per-segment stiffness `k_i` and strength `s_i`
(`i=1..N`), the checker runs two forward sweeps (`i=1..N`, arrays 0-indexed
from a leading 0):

```
Homogeneous sweep (math tool, not a mechanical test):
  H[0]=0, slip[0]=1
  shear[i] = k_i*slip[i-1]; H[i] = H[i-1]+shear[i]; slip[i] = slip[i-1]+Csub*H[i]

Thermal sweep (dAlpha = fixed CTE-mismatch forcing per unit dT):
  T[0]=0, tslip[0]=0
  tshear[i] = k_i*tslip[i-1]; T[i] = T[i-1]+tshear[i]
  tslip[i] = tslip[i-1] + Csub*T[i] + dAlpha
```

Let `d0 = -T[N]/H[N]` (this enforces zero net force at both free bond-line
edges). The per-segment shear stress under a unit temperature swing is
`thermal_i = tshear[i] + d0*shear[i]`. Define the worst normalized stress
`R = max_i(|thermal_i| / s_i)` -- this is where stiffness bites: a shorter
mismatch-strain decay length dumps more of the strain onto fewer, more
stressed segments, and `R` grows with uniform stiffness. Given the held-out
cycling profile `dT_1..dT_C` from the input and a **fixed** fatigue exponent
`p=3`: `Q = sum_c |dT_c|^p`, and the scored objective (cycling life) is
`F = 1 / (Q * R^p)`.

Below a size threshold, a stiffer *uniform* bond line still has a lower `R`
than a soft one (matches ordinary intuition: stiffer = stronger). Past that
threshold `R` for a uniform-stiff design blows up and `F` collapses toward 0
-- while a design that keeps the edges compliant and only the interior stiff
avoids the blow-up without giving up the interior's advantage.

## Input (stdin)
```
N M
Csub dAlpha
C
dT_1 ... dT_C
k_0 s_0
...
k_{M-1} s_{M-1}
```
`Csub` is the combined substrate axial compliance, `dAlpha` the CTE
mismatch (both fixed positive constants for the instance). The library lines
are sorted by increasing `k_j` (and `s_j`).

## Output (stdout)
Exactly `N` whitespace-separated integers `a_1..a_N`, each in `[0, M-1]`:
the adhesive type used at bond-line segment `i`.

## Feasibility
Exactly `N` tokens; each must parse as finite and be (within `1e-6`) an
integer in `[0, M-1]`. Wrong token count, non-numeric/non-finite tokens, or
an out-of-range index score the whole case `0`.

## Scoring
Let `F` be the cycling life above. The checker also builds its own
reference: the uniform SOFTEST type (index 0) on every segment -- always
well-defined and never collapses. Call its value `B`. Then
`Ratio = min(1000, 100*F/B) / 1000.0`. Matching the reference scores ≈0.1;
your final score is the mean `Ratio` over 10 test cases of varying bond-line
size and severity, several of which are placed specifically past the
uniform-stiffness collapse threshold.

## Constraints
`10 <= N <= 130`, `5 <= M <= 7`, `6 <= C <= 10`. Time limit 5s.

## Example (illustrative FORM only, small made-up numbers -- not the hidden
calibration)
`N=3, M=2`: types `(k=1, s=2)` and `(k=8, s=5)`, `Csub=0.05`,
`dAlpha=0.02`, one cycle `dT=10`. Using type 0 everywhere: `shear=[1, 1.05,
1.1525]`, `H=[0,1,2.05,3.2025]`; thermal sweep gives `T=[0,0,0.02,0.061]`,
so `d0=-T[3]/H[3]=-0.019048`; `thermal=[-0.019048, ~0, 0.019048]`,
`R=0.019048/2=0.009524`, `Q=10^3=1000`, `F=1/(1000*0.009524^3)≈1157.6`.
This IS `B` (type 0 everywhere is the checker's own reference). Any other
assignment is scored the same way and compared against it.
