# Static on the Line: Pre-Distorting a Memory Channel

A transmitter must push a message through a fixed, **known** nonlinear
channel before it ever reaches the receiver: a saturating amplifier that
**remembers**. The amplifier's output at time `t` is not a function of the
input symbol `x_t` alone — it also smears in the last `L` input symbols
(inter-symbol interference from the amplifier's internal state), and the
result is squashed by a saturating nonlinearity:

```
v_t = h_0*x_t + h_1*x_{t-1} + ... + h_L*x_{t-L}     (x with a negative index is 0)
y_t = A * tanh(v_t / A)
```

`h_0, ..., h_L` are the channel's FIR "memory taps" (`h_0 != 0`), and `A` is
the saturation level. You do **not** control the channel — you control the
**input sequence** `x_0, ..., x_{N-1}` fed into it, before transmission
starts. Your job: choose an input sequence whose channel OUTPUT lands close
to a given target message, while not spending more input energy than you
have to (energy is a real transmit-power cost, and every symbol you push
harder than necessary drifts the amplifier closer to saturation on
whatever traffic follows).

Because the channel has memory, a symbol you chose two steps ago is still
leaking into today's output. Inverting the channel one symbol at a time —
ignoring what your own recent symbols are still contributing — throws that
leakage away and lets it compound.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**.
It runs in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute an input sequence ...
print(json.dumps({"x": x}))
```

### Public instance (stdin)

```json
{
  "name": "chan205",
  "n": 40,                          // N, sequence length
  "L": 2,                           // memory order (0 <= L <= 3)
  "h": [0.55, 0.40, 0.22],          // FIR taps h_0..h_L, h_0 != 0
  "A": 2.0,                         // saturation level (> 0)
  "xmax": 3.0,                      // max allowed |x_t|
  "lambda": 0.05,                   // energy weight in the cost
  "target": [0.31, -1.02, 0.88, ...]  // N target outputs, each in (-A, A)
}
```

### Answer (stdout)

```json
{ "x": [0.42, -1.10, 0.77, ...] }   // length N, each |x_t| <= xmax
```

A layout is **valid** iff `x` is a list of exactly `N` finite numbers with
`|x_t| <= xmax` (small tolerance). Any invalid output (wrong length, a
non-numeric or out-of-range entry, non-finite value), a crash, a timeout, or
non-JSON output makes that instance score `0.0`.

## Objective

**Minimize**, across a fixed, seeded family of 10 instances (varying
sequence length, memory order, tap strength, and target amplitude —
including harder, higher-amplitude held-out cases), the mean of:

```
cost(x) = mean_t (y_t - target_t)^2  +  lambda * mean_t (x_t^2)
```

where `y_t` is what the stated channel model actually outputs when fed your
`x`. Both terms matter: matching the target message (first term) and not
overspending input energy (second term).

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_zero`  = `cost` of the all-zero input (transmit nothing: `y=0`
  everywhere, zero energy),
- `q_cand`  = `cost` of **your** input sequence,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_zero - q_cand) / q_zero, 0, 1 )
```

- Doing no better than transmitting nothing scores `~0.1`.
- Doing **worse** than transmitting nothing (e.g. a precoder whose
  per-symbol corrections compound through the channel's memory into a
  larger mismatch than silence) scores **below** `0.1`, clamped to `0`.
- Driving reconstruction error and energy to zero simultaneously is
  generally impossible (a nonzero target needs nonzero energy), so `1.0`
  is an unreachable ideal — real headroom remains even for strong solvers.

The reported **Ratio** is the mean of `r` over all 10 instances; **Vector**
lists the per-instance scores.

## Suggested strategies

1. **Do nothing**: transmit the all-zero sequence (the scoring anchor).
2. **Per-symbol static inversion**: invert the saturating nonlinearity
   symbol by symbol using only the leading tap `h_0`, ignoring `h_1..h_L`.
   Cheap, and fine when the memory taps are small — but the channel's
   memory doesn't go away just because you ignored it.
3. **Model the channel's memory**: since the channel is causal (`y_t` only
   depends on `x_t` and *earlier* symbols), process the sequence in order
   and account for what your own already-chosen recent symbols are still
   contributing to the current target before inverting — a whole-sequence,
   decision-feedback precoder rather than N independent inversions.
4. Within your amplitude budget `xmax`, weigh where to spend correction
   energy versus where a small residual mismatch is cheaper than the
   symbol energy it would take to erase it.
