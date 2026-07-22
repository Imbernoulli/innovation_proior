# Steady-State Titration Ward

A ward runs a fixed, published **probe protocol** on every patient: a short
sequence of test doses is administered and the resulting drug concentration
is measured (noisily) after each one. The drug is a one-compartment
accumulator: at every step a known **retention fraction** `rho` (0 < rho < 1)
of the current concentration carries over, and the new dose adds
`S_i * dose`, where `S_i` is that patient's **hidden sensitivity** (constant,
unknown, drawn from a population range). Formally, with `C_0 = 0`:

```
C_{t+1} = rho * C_t + S_i * dose_t
```

After the probe you must commit, **in one shot with no further feedback**, to
the entire remaining dosing plan for every patient, aimed at keeping
concentration inside a therapeutic window without drifting into toxicity.
`rho` is known and public; `S_i` is not — you must infer it from the noisy
probe curve.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs isolated; sees only
the public instance.

### Public instance (stdin)

```json
{
  "instance_id": 3,
  "rho": 0.88,
  "target": 11.4,
  "window_low": 10.49, "window_high": 12.31, "toxic_ceiling": 15.39,
  "probe_doses": [1.5, 2.5, 2.0, 3.0],
  "treatment_steps": 8,
  "dose_max": 80.0,
  "patients": [
    {"patient_id": 0, "probe_readings": [2.6, 6.9, 10.2, 14.8]},
    ...
  ]
}
```

`probe_readings[t]` is the (noisy) concentration observed right after probe
dose `t` was given, following the SAME recursion above with `C_0 = 0`, the
patient's true `S_i`, and the published `probe_doses`.

### Answer (stdout)

```json
{ "doses": [ [d0, d1, ..., d7], [d0, ..., d7], ... ] }
```

One dose sequence per patient, **in the same order** as `patients`, each of
length `treatment_steps`. Every dose must be a finite number in
`[0, dose_max]`. Any shape mismatch, wrong length, out-of-range or
non-finite dose, a crash, a timeout, or non-JSON output scores that whole
instance `0.0`.

## Objective

**Maximize** a therapeutic-quality score across a fixed, seeded family of 10
instances, spanning a spread of retention regimes (some patients clear the
drug fast, others very slowly) and several target/window scales.

## Scoring (deterministic)

The evaluator replays the TRUE dynamics forward from each patient's true
end-of-probe concentration through YOUR proposed doses (true `S_i`, true
`rho` — you never see either directly). Each of the `treatment_steps` steps
earns a reward: full credit while concentration sits inside
`[window_low, window_high]`; partial credit growing from 0 as it approaches
the window from below; credit that decays as it rises above `window_high`
toward `toxic_ceiling`; and a growing **penalty** once it exceeds
`toxic_ceiling`. A patient's score is the mean reward over the treatment
steps; an instance's score is the mean over its patients. Instance scores are
then mapped through a fixed, publicly-undisclosed affine window and clamped
to `[0, 1]` — a naive constant-dose recipe clusters low, and headroom is left
above what even a well-modeled controller reaches (some instances stay noisy
or hard enough that perfect targeting isn't achievable).

The reported **Ratio** is the mean instance score; **Vector** lists the
per-instance scores.

## Suggested strategies

1. **Ignore the probe**: dose everyone the same fixed amount.
2. **Titrate to the last reading**: estimate sensitivity from the single most
   recent probe point and hold that constant dose — assumes the drug has no
   memory.
3. **Fit the whole curve, model the memory**: regress sensitivity from every
   probe point against the known retention-weighted response, then solve for
   the dose that targets the true steady-state fixed point under `rho`.
4. **Adaptive ramp**: correct the very first treatment dose toward target
   before settling into the steady-state dose, for faster convergence.
