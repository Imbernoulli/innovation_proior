# Drop-Away Booster Staging

You design a launch vehicle made of **S drop-away stages**. Stage 1 (bottom)
burns first and is jettisoned first; stage S sits just below the payload and
burns last. Every stage is built from **identical engine modules** (each of dry
mass `m_e`, thrust `T`, exhaust velocity `v_e`) plus a propellant tank whose
structural mass is a fixed fraction `kappa` of the propellant it carries. For
each stage you choose an **integer engine count** `n_i` (`1 ≤ n_i ≤ E_max`) and
a **propellant mass** `p_i ≥ 0`. Gravity is `g`, the fixed payload mass is `P`.

Stage `i` has **dry mass** `n_i·m_e + kappa·p_i` and **wet mass**
`n_i·m_e + (1+kappa)·p_i`. The vehicle is legal iff
`P + Σ_i (n_i·m_e + (1+kappa)·p_i) ≤ M_total`.

## Ascent model (how the score is computed)

The grader integrates the climb stage by stage. Let the mass still on the
vehicle at ignition of stage `i` be
`m_start_i = P + Σ_{j≥i} (n_j·m_e + (1+kappa)·p_j)` and the mass at burnout be
`m_end_i = m_start_i − p_i` (only the propellant leaves; the tank stays attached
until jettison). Then:

- Tsiolkovsky gain: `dv_i = v_e · ln(m_start_i / m_end_i)`
- Burn duration: `burn_i = p_i · v_e / (n_i · T)` (propellant ÷ mass-flow rate)
- Loss: `loss_i = L_i · g · burn_i` — an **altitude/stage-indexed** gravity-and-drag
  penalty that grows with how long stage `i` burns. The table `L_1 … L_S` is
  given in the input and differs per stage.
- Net: `net_i = dv_i − loss_i`, after which the whole stage `n_i·m_e + kappa·p_i`
  is jettisoned (irreversible) before stage `i+1` ignites.

Final payload velocity is `V = Σ_i net_i`. **Maximize `V`.**

## Why local per-stage tuning fails

More engines on a stage shorten its burn (less loss) but add dry mass and eat
budget (less `dv`). More propellant on a stage adds `dv` but must be lifted by
every stage below it and burns longer (more loss). Because the loss table `L_i`
is skewed across stages, the mass split that maximizes `V` is dictated by the
**downstream jettison-and-loss schedule**, not by any single stage's local
efficiency. In particular, the loss-free geometric staging that a textbook
prescribes piles propellant onto the bottom stage — and when that stage carries
a high `L_i`, its long burn bleeds away the delta-v. The winning vehicle shortens
the burns that sit under a high `L_i` (more engines and/or less propellant there)
and reallocates the freed mass up the stack where the cascade still pays.

## Input (stdin)

```
S P M_total kappa
m_e T v_e g E_max
L_1 L_2 ... L_S
```

All values are real except `S` and `E_max` (integers). `3 ≤ S ≤ 4`.

## Output (stdout)

Exactly `S` lines. Line `i` holds two numbers `n_i p_i`: the integer engine
count and the propellant mass for stage `i` (stage 1 = first to burn).

## Scoring

Feasible outputs are scored `Ratio = min(1000, 100·V / B) / 1000`, where `B` is
the final velocity of the grader's own **one-engine, equal-propellant-split**
vehicle. That baseline scores ≈ 0.1; reaching ten times its velocity would cap
at 1.0. Any infeasibility — wrong line count, `n_i` outside `[1, E_max]`,
negative or non-finite `p_i`, or busting `M_total` — scores `Ratio: 0.0`.

## Example

For `S = 3`, one legal artifact is:

```
4 5200.0
3 4100.0
2 2600.0
```

meaning 4 engines + 5200 propellant on the bottom stage, and so on. The grader
integrates the three burns, subtracts each stage's duration-scaled loss, and
reports `V`. (Illustrative shape only — not an optimal split.)
