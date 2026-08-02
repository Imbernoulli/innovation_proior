# A Mold That Decides the Pore Shape: Coupled Zeolite Template Selection

## Story

You are choosing an **organic structure-directing agent** (template) and a
**crystallization condition** to synthesize a target zeolite framework. A
template's molecular geometry "molds" the pore during nucleation — the closer
its size, charge density, and shape suit the target pore, the more strongly it
directs the framework to form (its *structure-directing index*, SDI). But a
template can only work inside its own **chemical stability window**
(temperature, pH), which must also overlap the **framework's own crystallization
window** — you run ONE batch at ONE (T, pH) point, so template choice and
operating conditions are locked together, not chosen one after the other. Once
the crystal has grown, the template must be burned out by calcination; bulkier,
more rigid templates cost more thermal budget to remove and risk collapsing the
pore, discounting the final yield.

The template that molds the pore best is often the hardest to actually run.

## Input (stdin)

```
K c
D_target q_target
Tf_lo Tf_hi pHf_lo pHf_hi
w1 w2 w3
s_0 q_0 f_0 Tlo_0 Thi_0 pHlo_0 pHhi_0 Topt_0 pHopt_0 R_0 r_0
...  (K lines total, one per template, index 0..K-1)
```
`c` in {1,2,3} is the target's channel connectivity. `D_target` (Å) and
`q_target` are the target pore's kinetic diameter and net charge density.
`[Tf_lo,Tf_hi]`x`[pHf_lo,pHf_hi]` is the framework's own crystallization
window. `w1+w2+w3=1` weight size/charge/shape fit. For template `i`:
`s_i,q_i,f_i` are its size/charge/flexibility, `[Tlo_i,Thi_i]`x`[pHlo_i,pHhi_i]`
is its stability window, `(Topt_i,pHopt_i)` is its kinetically-preferred
nucleation point (always inside its own window), `R_i` is its process
robustness radius, `r_i` in [0,0.9) is its removal-cost fraction.

Fixed constants (same every case): `T_NORM=100`, `PH_NORM=3`, `Q_NORM=2`,
`F_NORM=1`, `f_ideal(c) = 0.15*c + 0.1`.

## Output (stdout)

One line: `idx T pH` — the template you pick and the (T, pH) you will run the
batch at.

## Feasibility

`idx` must be a valid integer in `[0,K-1]`. Both `T` and `pH` must be finite and
lie inside **both** windows simultaneously: `Tf_lo<=T<=Tf_hi`,
`Tlo_idx<=T<=Thi_idx`, `pHf_lo<=pH<=pHf_hi`, `pHlo_idx<=pH<=pHhi_idx` (1e-6
tolerance). Any violation, parse failure, or non-finite value scores `0.0` for
that case.

## Objective (what the score rewards)

```
size_match(i)   = max(0, 1 - |s_i - D_target| / D_target)
charge_match(i) = max(0, 1 - |q_i - q_target| / Q_NORM)
shape_match(i)  = max(0, 1 - |f_i - f_ideal(c)| / F_NORM)
SDI(i)          = w1*size_match(i) + w2*charge_match(i) + w3*shape_match(i)

dist(T,pH,i)    = sqrt( ((T-Topt_i)/T_NORM)^2 + ((pH-pHopt_i)/PH_NORM)^2 )
proximity(T,pH,i)= max(0, 1 - dist(T,pH,i) / R_i)

yield = SDI(i) * proximity(T,pH,i) * (1 - r_i)
```

`SDI` alone measures geometric complementarity — how well the template molds
the pore. `proximity` measures how close your *feasible* operating point is to
the template's kinetic sweet spot, which is often located outside the
framework's crystallization window even when the two windows technically
overlap. `(1-r_i)` discounts for calcination damage. All three mechanisms
multiply into one number: a perfect-fit template with an unreachable sweet
spot or crushing removal cost is not a good choice.

## Scoring

The checker computes your `yield` (call it `F`) and an internal baseline `B`
(a fixed, always-feasible reference template run at its own sweet spot).
```
Ratio = min(1.0, F / B * 0.1)
```
printed as `Ratio: <value>` (10x the baseline maps to 1.0). Ten test cases are
averaged (see config.yaml); several cases are engineered so that the
best-*geometric-fit* template's sweet spot is unreachable, its window is
chemically incompatible with the framework, or its removal cost dominates its
SDI advantage — a strategy that first picks the best-fitting template and only
afterward checks if it can be run lands at `Ratio=0` or far below a strategy
that jointly weighs fit, reachable conditions, and removal cost across the
whole template library.

## Objective

**Maximize** the mean `Ratio` over all cases.

## Example (worked, illustrative form only — not the actual scoring formula's constants)

If `SDI=0.8`, `dist=0.2`, `R=0.5` -> `proximity=1-0.2/0.5=0.6`; with `r=0.3`,
`yield = 0.8*0.6*0.7 = 0.336`. If the baseline template yields `B=0.4`, then
`Ratio = min(1, 0.336/0.4*0.1) = 0.084`.
