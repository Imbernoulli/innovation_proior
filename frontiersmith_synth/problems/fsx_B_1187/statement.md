# Aliased Rotor: Naming the Mover from Its Folded Micro-Doppler

## Problem

A radar stares at one rotating scatterer (a drone rotor, a helicopter rotor,
a propeller, a turbofan, a wind turbine — the *class*) for a short dwell. As
the platform-target geometry sweeps, the dwell contains looks at several
different **aspect angles**. At aspect angle theta (degrees, measured from
the rotation axis) the micro-Doppler line produced by a scatterer with
`blade_count` blades turning at `rate` rotations/second sits at

```
f_true(theta) = blade_count * rate * sin(theta)      [Hz]
```

— only the radial component of blade-tip velocity Doppler-shifts, so
sensitivity scales with `sin(theta)`. The radar's pulse repetition frequency
`PRF` can only unambiguously represent frequencies in `[0, PRF/2]`
(Nyquist). Any true line above that **aliases**: it folds back into band at
`|f_true - PRF * round(f_true / PRF)|`. A fast rotor can therefore
masquerade as a slow one at whichever angle happens to alias — the raw
spectral peak alone does not tell you the true rate.

You are given, for one dwell: `PRF`, `K` aspect angles, the already-folded
observed peak frequency at each of those angles, and a table of `C`
candidate target classes, each with its blade/scatterer count and a
plausible rotation-rate range (RPS). Name the class and estimate its
rotation rate.

## Input (stdin)

```
PRF K C
theta_1 theta_2 ... theta_K
f_obs_1 f_obs_2 ... f_obs_K
name_1 blade_1 rate_min_1 rate_max_1
...
name_C blade_C rate_min_C rate_max_C
```
`PRF` in Hz; angles in degrees, `f_obs_i` in Hz (already PRF-folded); each
candidate row gives an integer blade/scatterer count and its own plausible
rate range. `4 <= K <= 7`, `C = 5`.

## Output (stdout)

Two whitespace-separated tokens: `class_id rate`, where `class_id` is a
0-based index into the printed candidate rows and `rate` is your rotation
-rate estimate (RPS, real number).

## Feasibility

`class_id` must be a valid row index. `rate` must be finite and lie within
that row's own `[rate_min, rate_max]` — you may only claim a rate that is
physically plausible for the class you name. Any violation scores 0.

## Objective and Scoring

Your answer is scored by how well it **explains every given aspect-angle
observation**, not just one. The checker re-applies the forward model with
your `(blade_count, rate)` at each angle, PRF-folds the prediction the same
way the observations were built, and compares it to `f_obs_i`. Each angle
contributes a score between a small positive floor and 1 that shrinks as the
Hz mismatch grows past a tolerance band; `F` is the sum over all `K` angles.
The checker also builds its own naive one-shot reference (candidate row 0 at
its own range midpoint, ignoring the data) to get a baseline `B > 0`. Final
score:
```
ratio = min(1, F / (10 * B))
```
printed as `Ratio: <ratio>`. An answer that is consistent across the whole
aspect-angle spread scores far higher than one that only fits a single look.

## Constraints

Time limit 5s, memory 512MB per test case; 10 test cases.

## Example (worked score, illustrative — NOT one of the graded cases)

Suppose `PRF=100`, `K=2`, angles `20 60`, observed `3.420201 8.660254`
(these were built from `blade=2, rate=5.0`), and two candidate rows:
row 0 `"B" 3 10.0 20.0`, row 1 `"A" 2 4.0 6.0`.

- Checker baseline: row 0 at its midpoint, `rate=15.0` (blade 3). Predicted
  folded lines: `15.39` Hz and `38.97` Hz — both far from the observations,
  so both angles land on the score floor: `B = 0.12 + 0.12 = 0.24`.
- Submission `"1 5.0"` (row 1 = class A, blade 2, rate 5.0): predicted lines
  are `3.420201` and `8.660254` Hz, matching exactly at both angles: `F =
  1.0 + 1.0 = 2.0`.
- `ratio = min(1, 2.0 / (10 * 0.24)) = min(1, 0.8333) = 0.8333`.

## Notes for solvers

An angle that happens to be free of aliasing at the true rate agrees with a
"no-fold" fit; an angle where the true line exceeds `PRF/2` will not. A
class+rate that only explains the loudest single angle, but disagrees with
the others, has found a decoy — real recovery must be consistent with the
whole aspect-angle diversity of the dwell.
