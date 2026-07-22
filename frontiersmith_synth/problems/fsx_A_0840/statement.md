# Blindfolded Blacksmith: Tuning the Quench

A blacksmith runs a **frozen** search over a hidden bit-string landscape: each
of `steps` moves she picks a uniformly random bit, tentatively flips it, and
keeps the flip if it doesn't lower total reward, else keeps it anyway with
probability `exp(delta / T)` (standard Metropolis). She can't change this
mechanism or see the landscape. **You** control only the temperature `T` over
time, including when to reheat. Your program is called **once per instance**
and must emit a whole control policy in advance — the evaluator then runs its
frozen searcher under that policy.

## The landscape (shape is public; two constants are hidden)

The `N` bits split into consecutive **blocks** (sizes/types given). For a block
of size `b` with `u` of its bits set to 1:

- **onemax block**: `reward(u) = u * slope` — more 1s is always locally better.
- **trap block**: `reward(u) = (b - u) * slope` for `u < b`, and
  `reward(b) = b * slope * BONUS` (`BONUS > 1`) at the unique global optimum
  `u = b`. Reward *drops* every time a 0-bit flips to 1, until the very last
  flip completes the block and reward jumps far above the `u=0` value.
  Hill-climbing alone drives a trap block straight to `u=0` and stays there.

Total reward = sum over blocks; the true optimum is all bits = 1. `slope` and
`BONUS` are **fixed per instance but withheld** — only block sizes/types, `N`,
and `steps` are public. The right temperature to escape a trap varies between
instances and must be inferred from what your policy observes, not assumed.

## Candidate contract

```python
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({
  "T0": 3.0, "alpha": 0.85, "window": 300, "stagnation_window": 300,
  "accept_floor": 0.9, "reheat_factor": 1.6, "restart_mode": "restart_best",
  "max_events": 30
}))
```

### Public instance (stdin)
```json
{"name": "...", "N": 28, "steps": 14000, "T0_max": 15.0,
 "reheat_factor_max": 3.0, "max_events_cap": 80,
 "blocks": [{"type": "trap", "size": 6}, {"type": "onemax", "size": 4}, ...]}
```

### Answer (stdout) — your control policy
- `T0`: float in `[0, T0_max]` — initial temperature.
- `alpha`: float in `(0, 1]` — per-window decay factor.
- `window`: int in `[5, steps]` — moves per temperature-check window; **inside
  a window the temperature is held constant.**
- `stagnation_window`: int `>= 1` — a window counts as "stagnant" only if the
  best score hasn't improved for at least this many moves.
- `accept_floor`: float in `[0, 1]` — a window's realised acceptance rate must
  fall below this to count as "stuck".
- `reheat_factor`: float in `[1, reheat_factor_max]`.
- `restart_mode`: `"reheat"` (keep bits, raise T), `"restart_best"` (jump to
  the best bits found so far, then raise T), or `"restart_random"` (jump to a
  fresh evaluator-generated random bit-string, then raise T).
- `max_events`: int in `[0, max_events_cap]` — hard cap on triggered events.

**Kernel semantics.** `T = leg_T0 * alpha^k` (constant within a window), `k` =
windows since the current "leg" started (0 initially and after every event).
At each window's end: if acceptance rate `< accept_floor` **and** the run is
stagnant **and** budget remains, an event triggers — the current window's `T`
is multiplied by `reheat_factor` (compounding across repeated events) to give
the new leg's `leg_T0` (`k` resets to 0), and `restart_mode` is applied.
Otherwise `k` advances and `T` decays by `alpha`. Malformed/out-of-range
fields, non-JSON output, a crash, or a timeout score that instance `0.0`.

## Scoring (deterministic)

The evaluator tracks the best total reward visited over the whole run and
reports it as your objective `obj`. Per instance: `baseline` = the same
kernel run under the always-zero-temperature policy (pure hill-climbing,
never escapes a trap); `opt` = the closed-form true maximum (all bits = 1, no
search needed). Then `r = clamp(0.1 + 0.9 * (obj - baseline) / (opt -
baseline), 0, 1)`. **Ratio** is the mean of `r` over all 10 fixed, seeded
instances (some larger/deeper traps are held out for generalization).

## Things to think about

- A single guessed temperature, decayed once across the whole budget, gets one
  shot at each block's barrier — fine if it matches the hidden `slope`,
  useless otherwise.
- A window's acceptance rate is a scale-free readout of whether the current
  temperature is right for THIS instance, independent of what `slope` is.
- Compounding escalation on repeated, genuine stagnation searches upward
  through whatever scale is needed, giving every trap block many independent
  attempts instead of one.
