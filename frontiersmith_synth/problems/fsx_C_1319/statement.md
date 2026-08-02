# Self-Assembly Scheduling: Outrunning the Kinetic Trap

## Problem
`N` single-valence monomers (each holding **at most one bond** at a time) should
self-assemble into a target structure: `K` "correct" monomer pairs forming a
node-disjoint matching. You are also given `D` extra "decoy" bond options
between other monomer pairs; decoys compete for the same sticky sites as the
correct bonds but are **not** part of the target. Every one of the
`M = K + D` bond options `j` carries an integer **bond strength** `s_j`.

Assembly runs for `T` discrete time steps. At step `t` the temperature is
`theta(t) = theta0 - floor(theta0*(t-1)/(T-1))` (so `theta(1)=theta0`,
`theta(T)=0`, non-increasing). At every step:
1. Every bond option already **enabled** (see below) and not currently formed
   attempts to grab its two monomers, processed **in the fixed order the
   bonds are listed in the input**. It forms if both monomers are free.
2. Every currently formed bond `j` with `s_j < theta(t)` then **breaks** (its
   monomers free up) — weak bonds are thermally reversible while hot. A bond
   with `s_j >= theta(t)` survives instead: once temperature has fallen below
   a bond's strength that bond is **frozen and can never break again**, even
   if a better option later appears for those monomers. This is the kinetic
   trap: an early, wrongly-placed strong bond permanently blocks its monomers
   from ever forming the intended pair.

You choose, for every bond option, an **enable time** `t_j in [1,T]`: option
`j` cannot attempt to form before step `t_j`, but keeps trying every step from
`t_j` onward while unformed. This staged schedule — not the physics — is your
only lever.

## Input (stdin)
```
N M T theta0
u_1 v_1 s_1 type_1
...
u_M v_M s_M type_M
```
`type` is `T` (part of the target structure) or `D` (decoy). Monomers are indexed
`0..N-1`. Type-`T` bonds are node-disjoint.

## Output (stdout)
`M` integers (whitespace/newline separated), in the same order as the input
bonds: the enable time `t_j` you choose for bond `j`.

## Feasibility
Your output is rejected (score `0`) unless it contains exactly `M` integer
tokens, all finite, each satisfying `1 <= t_j <= T`.

## Objective (maximize)
Run the deterministic simulation above with your chosen enable times. Let `y(t)`
be the total strength of type-`T` bonds that are formed **and currently stable**
at the end of step `t`, and `Y` the total strength of all type-`T` bonds. The
objective is the time-averaged fraction of the target assembled:
```
F = (1/T) * sum_{t=1..T} y(t) / Y
```
This rewards both a high final target yield AND reaching it early — a schedule
that locks the whole target in by step 5 beats one that only gets there by step
`T-1`, even if both eventually reach 100%.

## Scoring
The checker independently simulates its own simple reference schedule (every
bond enabled simultaneously at one fixed, late step) to get a baseline `B` from
the same physics. Then
```
Ratio = min(1000, 100 * F / B) / 1000
```
so matching the reference scores `0.1`; beating it roughly 10x caps the ratio at
`1.0`. Scoring is deterministic integer/rational arithmetic throughout.

## Constraints
- `5 <= K <= 14`, `theta0 = 14` (fixed).
- Target bond strengths in `[1,4]`; decoy bond strengths in `[8,12]`.
- `15 <= T <= 60`. Time limit `5s`.

## Example (worked, illustrative shape only)
`N=4 M=3 T=5 theta0=3`, bonds listed in this input order: `0 2 3 D`, `0 1 1 T`,
`2 3 1 T`. Here `theta(t) = [3,3,2,1,0]` for `t=1..5`.

Enable both `T` bonds at `t=1` and the `D` bond only at `t=5`: at `t=1,2,3` the
`T` bonds form each step but break again (`1 < theta`); at `t=4`, `theta(4)=1`
so `1 >= theta(4)` — both freeze forever. At `t=5` the `D` bond finally becomes
eligible but its monomers are already taken, so it never binds. `y(t) =
0,0,0,2,2`, `Y=2`, giving `F = (1/5)*(0+0+0+1+1) = 0.4`.

Enable the `D` bond at `t=1` too (it is listed first, so it is attempted
before either `T` bond every step): at `t=1` it forms on its free monomers
`0,2`, and since `3 >= theta(1)=3` it freezes immediately, permanently
occupying them. Neither `T` bond can ever form again, so `F` collapses to
`0.0` — forming the strongest bond first maximized its own early success but
destroyed the target.
