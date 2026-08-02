# Settling In: Forecasting a Bridge Deck's Irreversible Drift

A displacement sensor on a bridge deck logs, every few days, the **elapsed
time** `t` (days since monitoring began), the **deck temperature** `T`
(degrees C), and the **displacement** `d` (millimetres, referenced to zero at
the start of monitoring). Your job: predict `d` far into the future.

The true displacement is the SUM of two physically different processes, both
present in every reading:

1. **Irreversible settlement (creep).** The deck sinks into its footing. This
   process only ever grows: an initial fast phase that decelerates toward a
   near-steady rate, plus a slow secondary creep that keeps drifting forever.
   It never reverses.
2. **Reversible thermal expansion.** Heat makes the deck material expand and
   contract with the ambient/deck temperature `T`. This part fully reverses
   every season — it contributes zero to the long-run settled position.

You are given a monitoring log confined to a **visible span** early in the
structure's life. You will be graded on your prediction **several seasonal
cycles further out** — a horizon you never observe directly.

## Input (stdin)
- Line 1: `n id` — `n` logged readings and a case id.
- Next `n` lines: `t T d`, one reading each (floats), `t` increasing.

## Output (stdout)
One line: a closed-form Python expression for `d` in variables `t` and `T`.
Allowed: `+ - * / **`, unary `-`, numeric constants, and the functions
`sqrt log exp sig tanh absv`. Example (illustrative **form only — NOT the
hidden law**): `sig(t) * T + 2.5`. No other names are accepted.

## Scoring (deterministic, maximisation)
Your expression is evaluated on a **held-out horizon** — `t` several seasonal
cycles beyond the visible span — regenerated deterministically inside the
grader from the same case id, together with the temperature reading `T` the
grader supplies at each held-out point (you never see this data). Let `p_i` be
your prediction and `y_i` the true (noisy) displacement at held-out point `i`:

```
metric   = mean_i  min(1, |p_i - y_i| / (|p_i| + |y_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor mean(train d)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). A constant
predictor scores about `0.1`. `LAMBDA` is a small fixed parsimony weight.
Non-finite or complex-valued predictions score `0`.

## Why the raw trace is a trap
Nothing in the log tells you which part of `d` is settlement and which part
is thermal — both move together. A method that treats the whole visible trace
as one undecomposed trend, and extrapolates a rate estimated over the visible
span forward from the most recent reading, silently assumes that reading's
value is entirely "current state." On several of the logs, the visible window
happens to end almost exactly at a seasonal temperature peak: the last
readings are pushed up by the full reversible thermal swing, not by extra
settlement. A trend that anchors on that inflated reading carries the swing,
unchanged, across every future prediction — even after several more seasons
have passed and the true temperature (and its contribution to `d`) has long
since come back down. The error compounds with distance from the visible
window rather than washing out.

Separating the two processes first — using the temperature reading itself to
identify how much of each observed displacement is reversible — and
extrapolating *only* the irreversible piece is what survives the held-out
horizon.

## Constraints
- Time limit 5 s, memory 512 MB. `n = 200`.
- Held-out sensor and process noise (larger than in training) leaves
  irreducible error, so even a correctly-shaped law does not reach
  `Ratio = 1.0` — there is room above the reference solutions.
