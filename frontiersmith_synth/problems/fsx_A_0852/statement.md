# Sleeper Solvers: Budgeting a Node-Expansion Portfolio

## Story

A search lab has one hard planning instance and a **portfolio of heuristic
solvers**, each able to attack it independently. Every solver improves its
best-found solution **quality** the longer it runs, but the *shape* of that
improvement curve differs by solver and by instance: some sprint out of the
gate then stall at a mediocre ceiling; others crawl for a long time before
their structure pays off and they streak past everyone else; most just aren't
very good. Which is which is **instance-dependent** and unknown to you — but
you get a **short informativeness probe** for free: a handful of early
progress samples per solver, at checkpoint sizes far smaller than the total
**node-expansion budget** `B`. These samples cost nothing — the whole of `B`
is still yours. Decide how to invest all of `B` across the portfolio to
maximize the best final quality you obtain.

## Candidate program contract

Standalone program: read ONE JSON object from **stdin**, write ONE JSON object to
**stdout**. Isolated subprocess; you see only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide an allocation ...
print(json.dumps({"alloc": {"h0": 400, "h1": 1200, ...}}))
```

### Input (stdin)

```json
{
  "budget": 2600,
  "heuristics": [
    {"id": "h0", "checkpoints": [26, 65, 130, 234], "probe": [0.0138, 0.0431, 0.128, 0.257]},
    {"id": "h1", "checkpoints": [26, 65, 130, 234], "probe": [0.001, 0.0021, 0.0041, 0.0079]},
    ...
  ]
}
```

- `budget`: integer `B > 0`, total node-expansions available across the whole
  portfolio.
- `heuristics`: a list of solvers. Each has a unique string `id`, a list of 4
  increasing checkpoint sizes `checkpoints` (each far smaller than `B`), and
  `probe[i]` = the quality that solver would reach if run **alone from
  scratch** for exactly `checkpoints[i]` total node-expansions. Every solver's
  true quality curve is continuous, non-decreasing in its own expansion count,
  and bounded — but its full shape past the probe window is unknown to you.

### Output (stdout)

```json
{ "alloc": {"h0": 900, "h1": 0, "h2": 1700} }
```

- `alloc`'s keys must all be solver ids from the instance (an omitted id means
  0). Every value must be a non-boolean, **finite, non-negative integer**, and
  the values must sum to **at most `B`**.
- Only the *total* expansions you finally give a solver matters, not the order
  you "spend" them in — submit the totals you land on after weighing probes.

Any key that is not one of the instance's solver ids, a negative, non-integer,
or non-finite value, a sum over `B`, a crash, a timeout, or non-JSON output
makes that instance score `0.0`.

## Objective & scoring (deterministic)

The evaluator holds the **true** quality curve of every solver (which generated
the probes). Your realized quality is:

```
quality = max over solvers h of curve_h( alloc[h] )
```

i.e. you keep the best solution any funded solver reaches. Per instance the
evaluator also computes, from the true curves, two references you never see
directly:

- `best`  = quality from putting the **whole** budget `B` on whichever single
  solver turns out strongest — the best single heuristic in hindsight.
- `worst` = quality from putting the whole budget on whichever single solver
  turns out weakest.

Normalized score, with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (quality - worst) / max(best - worst, 1e-9), 0, 1 )
```

Your final score is the mean of `r` over a fixed family of 10 instances
(varied portfolio sizes and budgets, including larger held-out cases).

## Notes

- Fully deterministic and seeded — the score never depends on wall-clock or
  hardware speed, only on your submitted allocation and whether it finishes
  in time.
- A solver whose probe rises fast and flattens is not your best final bet; one
  whose probe is small but visibly **accelerating** (later increments bigger
  than earlier ones) may build toward a much higher ceiling than its probe
  window shows. Reading a probe's *trend*, not just its latest value, and
  hedging rather than dumping the whole budget on today's leader, is the game.
- No single textbook rule dominates: pure exploitation of the loudest early
  signal, pure equal-split, and pure trend-following each fail on some
  instances here.
