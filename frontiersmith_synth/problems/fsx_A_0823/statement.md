# Two Tuning Forks in a Box: Extrapolating a Coupled-Mode Resonance

## Problem
A sealed box contains **two damped mechanical oscillators, linearly coupled
to each other**. Only the first is driven, by an external force of unit
amplitude at angular frequency `w`; a lock-in amplifier records the first
oscillator's steady-state complex response `X1(w) = Xre + i*Xim` (in-phase
and quadrature channels). The response is fixed by five hidden physical
constants: two natural frequencies `Om1 < Om2`, two damping rates `g1, g2`,
and a coupling constant `kc`. You are **not told these numbers** — you must
infer them from data, then predict the response far beyond where you
measured it.

For safety, the lab log only records drive frequencies **well below both
resonances**: a smooth, featureless, monotonically-rising curve with no
visible peak. Your job is to predict the response **amplitude** `|X1(w)|`
everywhere, including through the resonance region and beyond — where, if the
oscillators are coupled, the single naive resonance splits into **two**
peaks (an "avoided crossing").

## Input (stdin)
```
N testId
w_1 Xre_1 Xim_1
w_2 Xre_2 Xim_2
...
w_N Xre_N Xim_N
```
`N` noisy measurements, all at frequencies `w` well below `Om1`.

## Output (stdout)
One line: a closed-form Python expression for the amplitude `|X1|` in the
single variable `w`. Allowed: `+ - * / **`, unary `-`, numeric constants, and
the functions `sqrt absv`. Example (illustrative **form only — NOT the
hidden law**): `absv(3.1*w - 0.4) + sqrt(w)`. No other names are accepted.

## Feasibility
The expression must parse under the whitelist above, and must evaluate to a
finite real number (no `nan`/`inf`, no complex result from a negative base
raised to a fractional power) at every held-out point. Any violation scores
`Ratio: 0.0`.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out grid**, regenerated inside the
grader from the test id: a fresh same-band check, a short tail extension,
the **full resonance region** (both peaks and the avoided-crossing dip
between them), and the **high-frequency asymptote**. Let `p_i` be your
prediction and `t_i` the true (noisy) amplitude at held-out point `i`:
```
metric      = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))  # bounded rel. error
O           = metric * (1 + LAMBDA * nodes)      # nodes = expr size
base_metric = the same metric for the constant predictor mean(train amplitude)
baseline    = base_metric * (1 + LAMBDA * 1)     # baseline pays the same 1-node charge
Ratio       = min(1000, 100 * baseline / O) / 1000
```
Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
predictor scores about `0.1`. `LAMBDA` is a small parsimony weight. Held-out
measurement noise leaves irreducible error, so even a correct discovery does
not reach `Ratio = 1.0` — there is room above the reference solutions.

## Why the obvious fit is a trap
The training tail is smooth and featureless — nothing in it "looks like" two
resonances. The obvious move is to fit the textbook single driven-damped
oscillator, `A = c / sqrt((w0^2-w^2)^2 + (gamma0*w)^2)`, discarding the phase
channel entirely; this matches the tail beautifully (any smooth curve is
locally well-approximated by one pole pair) and gives no hint anything is
wrong. But a single pole pair can only ever produce **one** resonance peak
and a `1/w^2` high-frequency falloff. The true box has **two** coupled
oscillators: they produce two split peaks and a `1/w^4` falloff. The single
Lorentzian is not "close but under-fit" — it is structurally the wrong shape,
and no amount of retuning its three constants against more tail data closes
that gap.

The insight is to stop fitting an amplitude curve and instead **commit to
the physical model class**: fit all five physical constants
(`Om1, Om2, g1, g2, kc`) at once against the phase-resolved `(Xre, Xim)`
data, using the exact two-oscillator transfer function. A single Lorentzian
has no slot for a second natural frequency, however its three numbers are
tuned; the two-mode form does, so even an imperfect fit to noisy tail-only
data reproduces both resonances, the splitting, and the correct falloff once
extrapolated. **Committing to the right model class, not gathering more
tail data, is what makes the extrapolation correct.**

## Constraints
- `N = 44` training rows; time limit 5s, memory 512MB.
- Each `.in` file is well under 1MB.
