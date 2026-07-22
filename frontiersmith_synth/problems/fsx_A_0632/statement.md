# Mystery Converter — recovering the hidden loss law from a commissioning band

A power-converter unit was bench-tested during commissioning. The test rig only
ever ran it at **partial load, between 20% and 60% of rated load**, while ambient
temperature was swept widely. You are handed that commissioning log and must
recover a closed-form law for the unit's **power loss** `y` (watts) as a function
of **load fraction** `L` and **ambient temperature** `T` (degrees C), so that it
can be trusted for an **overload study at 80%-110% of rated load** — a regime the
commissioning rig never reached.

The true loss decomposes into three physical mechanisms, all present at every
load, all shaping `y` simultaneously:

1. A constant **standby loss**.
2. A **resistive loss** growing with `L^2`, whose effective coefficient drifts
   linearly with ambient temperature (a temperature coefficient).
3. A **saturation loss** that grows as `L` to some power `p` (a real number,
   not necessarily an integer) — the signature of the magnetic core beginning to
   saturate. At 20-60% load this term is a small fraction of the total (it is
   *nearly invisible* next to the resistive term and the measurement noise), but
   it grows much faster than `L^2` and becomes a dominant contributor once load
   climbs toward and past 100%.

Multiplicative measurement noise is present in every reading, and is larger on
overload readings (less controlled test conditions) — so even a perfectly
recovered law will not predict overload readings exactly.

## Input (stdin)
- Line 1: two integers `n` and a case id.
- Next `n` lines: `L T y`, one commissioning-log reading each (floats). All
  `L` values lie in `[0.20, 0.60]`.

## Output (stdout)
One line: a closed-form Python expression for `y` in variables `L` and `T`.
Allowed: `+ - * / **`, unary `-`, numeric constants, and the functions
`sqrt log exp sig tanh absv`. Example (illustrative **form only — NOT the
hidden law**): `sqrt(absv(T)) * L + 2.5`. No other names are accepted.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out overload grid** (`L` in
`[0.80, 1.10]`), regenerated deterministically inside the grader from the same
case id — you never see it. Let `p_i` be your prediction and `t_i` the true
(noisy) loss at held-out point `i`:

```
metric   = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor mean(train y)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
predictor scores about `0.1`. `LAMBDA` is a small parsimony weight. Non-finite
or complex-valued predictions score `0`.

## Why the obvious fit is a trap
A practitioner who takes the stated mechanisms 1-2 at face value (constant +
`L^2` resistive term with a temperature coefficient) and fits exactly that —
or fits a generically flexible low-order polynomial in `L` that nails the
20-60% band — gets a law that is accurate where it was trained and **wrong by
a growing margin as load rises past 60%**, because no term in it can express
the faster-than-quadratic saturation growth. In the training band the
saturation term contributes well under 2% of the loss near `L=0.2` and only
climbs to roughly 15-20% by `L=0.6` — a faint but real curvature that a fit
restricted to `{1, L^2, T*L^2}` (or a naively flexible cubic) partially
absorbs into a biased `L^2` coefficient rather than resolving. Only searching
for the actual exponent `p` — mechanism by mechanism, not just curve by curve
— recovers a law that survives into the overload band, where the same term
can exceed the resistive loss.

## Constraints
- Time limit 5 s, memory 512 MB; `n` = 200.
- Held-out noise leaves irreducible error, so even a correctly-shaped law does
  not reach `Ratio = 1.0` — there is room above the reference solutions.
