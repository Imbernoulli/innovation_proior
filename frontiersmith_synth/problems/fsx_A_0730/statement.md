# Firefly Census: a Tiny Automaton Counts Blinks

A meadow biologist records a night's stream of firefly blink-codes as a sequence
of integers (each blink is coded 0..m-1). She wants to know how many blinks in
the stream matched a **target pattern class**, defined by a modular filter: blink
code `x` is a **hit** iff `(x mod k)` lies in a given set of residues. Her field
logger, though, is a tiny chip: a **finite-state machine with at most `s`
internal states**. The stream can run for hundreds of blinks, so remembering the
exact running tally (which needs one state per possible count, `0..L`) is
usually impossible within the budget — the chip has to live with an
**approximate** count.

You will design the logger: a Moore machine (states, a transition rule, an
output table) with `<= s` states. The evaluator feeds your machine, symbol by
symbol, through many hidden overnight streams (drawn from a fixed, seeded
family for this instance) and compares the machine's final readout against the
true hit count of each stream.

## Candidate program contract

Your program is called **once per instance**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your machine) to **stdout**. It
runs in an isolated subprocess and never sees the hidden streams.

```python
import sys, json
inst = json.load(sys.stdin)
# ... design a machine tailored to inst ...
print(json.dumps({"n_states": ..., "start": ..., "trans": ..., "out": ...}))
```

### Public instance (stdin)

```json
{
  "name": "monsoon-dense-A",
  "m": 18,              // alphabet size: raw blink codes are 0..m-1
  "s": 7,                // state budget: your machine may use at most s states
  "k": 3,                 // modular filter base
  "target_residues": [0, 1],   // symbol x is a HIT iff (x % k) in target_residues
  "l_min": 250, "l_max": 350,  // hidden streams have length uniformly in [l_min, l_max]
  "n_streams": 45              // how many hidden streams this instance will average over
}
```

### Answer (stdout)

```json
{
  "n_states": 6,
  "start": 0,
  "trans": [[...18 ints in [0,6)...], ...6 rows...],   // trans[state][symbol] -> next state
  "out":   [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]               // out[state] = reported hit-count estimate
}
```

- `1 <= n_states <= s` (you may use fewer than the full budget).
- `trans` has exactly `n_states` rows of exactly `m` integers, each a valid
  next-state index — the transition may depend **only** on the current state
  and the raw symbol (no position/step counters, no hidden memory).
- `out` has exactly `n_states` finite numbers.
- Any wrong shape, out-of-range index, non-finite output, crash, timeout, or
  non-JSON output scores that instance `0.0`.

## Scoring (deterministic)

For this instance the evaluator itself generates `n_streams` seeded blink
streams of length in `[l_min, l_max]` over alphabet `0..m-1`, and knows each
stream's true hit count. Your machine (fetched once from your program) is then
**simulated by the evaluator** over every hidden stream: run the transition
table symbol by symbol, and read your `out` table at the final state as the
estimate. Per-stream relative error:

```
e = |estimate - true_count| / (true_count + 1)
```

The instance's mean error `q_cand` (averaged over its streams) is compared to
a fixed weak reference `q_base` = mean error of always guessing `0`, and the
unreachable ideal `q_lb = 0`:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(q_base, 1e-9), 0, 1 )
```

Always guessing `0` scores ~`0.1` on every instance; a perfect machine would
score `1.0` (unreachable in general, since `s` states cannot encode every
possible count exactly). The reported **Ratio** is the mean of `r` over all 10
instances (some have small alphabets/short streams, others — held out — use
larger alphabets, tighter budgets, and much longer streams).

## Things to think about

- With a generous budget, an exact tally works well — but when streams run
  long and hits are common, `s` states cannot enumerate every count up to the
  true maximum.
- A plain saturating tally (increment, clamp at `s-1`) stays exact for a
  while, then freezes forever — costly once the true count climbs well past
  the budget.
- `m`, `k`, `target_residues`, `l_min`, `l_max` tell you the hit rate and the
  plausible range of true counts *before* any stream arrives — use them to
  size your machine, not just to classify symbols.
