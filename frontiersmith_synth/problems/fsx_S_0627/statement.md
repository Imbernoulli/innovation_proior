# Twenty Questions Against a Sealed Physics Box

A sealed box hides a scalar law `f(x)` on the interval `[0, GR]`. Nobody gets to
open the box. Instead you get a limited number of **probes**: name a real `x` in
the **queryable window** `[0, QR]` (a strict prefix of the full domain, `QR < GR`)
and the box returns a **noisy reading** `y = f(x) + noise`. You have a total probe
budget `Q`, spent across up to `R` adaptive rounds — each round you see every
reading collected so far and choose your next batch of probes. When the budget is
exhausted (or you stop early), the box is resealed and you must **predict** `f` on
a fixed dense grid of `G` points spanning the *entire* domain `[0, GR]` — including
the tail `(QR, GR]`, which you could never probe directly.

The hidden law is not arbitrary: it is **piecewise-linear** with a handful of
unknown breakpoints (slope changes / jumps), and it obeys a hidden **point
symmetry**: there exist an unknown center `c` and value `b = f(c)` such that
`f(2c - x) = 2b - f(x)` for every `x`. The center `c` sits inside the queryable
window but past its midpoint, so the reflection `2c - x` of a tail point `x > QR`
can land back inside `[0, QR]` — the only way to know anything about the tail is
through this symmetry.

## Candidate program contract

Your solution is a **standalone program**, re-invoked by the evaluator **once per
round** (stateless between calls — all history is handed back to you each time).
Read ONE JSON object from stdin, write ONE JSON object to stdout:

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide ...
print(json.dumps(answer))
```

### Input each call

```json
{
  "name": "trap1", "phase": "query" | "predict",
  "QR": 12.0, "GR": 20.0, "G": 400, "Q": 36,
  "c_lo": 10.0, "c_hi": 11.5,
  "budget_left": 24, "round": 1, "R": 5, "max_this_round": 36,
  "history": [[x0, y0], [x1, null], ...]
}
```

`history` lists every probe issued so far as `[x, y]`; `y` is `null` if that `x`
fell outside `[0, QR]` (it still counted against your budget). `c_lo`/`c_hi` bound
where the hidden center `c` can be.

### Your answer

- `phase == "query"`: `{"queries": [x1, x2, ...]}` — any reals; only those in
  `[0, QR]` get a real (noisy) reading, others burn budget for `null`. At most
  `min(budget_left, max_this_round)` of your listed queries are honored per call;
  extras are ignored (not charged). You may return `{"queries": []}` to skip a
  round.
- `phase == "predict"` (final call): `{"pred": [y_0, ..., y_{G-1}]}`, one value per
  grid point `g_j = GR * j/(G-1)`.

Any crash, timeout, non-JSON output, or wrong-shaped answer on **any** call scores
that whole instance `0.0`.

## Scoring (deterministic)

For each of 10 fixed seeded instances (varying breakpoint count and how steep the
un-probeable tail is — several are adversarial "traps"), the evaluator computes,
itself, the true `f` on the grid and:

```
err_ref = mean_j |f(g_j)|                  # error of predicting flat zero
err     = mean_j |pred_j - f(g_j)|
quality = clip(1 - err/err_ref, 0, 1)      # 0 for predict-zero, -> 1 for perfect
r       = 0.10 + 0.82 * quality             # max attainable r = 0.92
```

Predicting all zeros scores exactly `0.10`. Noise, un-pinned breakpoints, and
reflection slack keep even a strong strategy below `0.92`, leaving headroom. The
final **Ratio** is the mean of `r` over the 10 instances; **Vector** lists the
per-instance scores.

## Why this is not plain regression

A probe's value is what it tells you about *structure* — where the law bends, and
where the symmetry center sits — not the single noisy number it returns. Spending
the whole budget on a uniform sweep of `[0, QR]` resolves smooth stretches you
didn't need, smears every breakpoint, and leaves you blind on the tail (which, on
the trap instances, carries the steepest structure of the whole domain — it's the
*mirror* of the interior). A stronger strategy spends a few probes to triangulate
`(c, b)`, adaptively bisects toward slope-kinks to pin breakpoints, and reflects
the tail through the identified symmetry.
