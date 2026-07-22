# Last Curtain Call: Dynamic Ticket Pricing

A venue is selling tickets for a one-night event. There are `C0` tickets in
stock and `T` pricing periods (days) counting down to showtime. Any ticket
still unsold when the curtain rises is worth exactly **zero** — it perishes.
Each period you post one price; the number of buyers who show up that period
is a deterministic, price-elastic function of that price. Buyers also
remember recent prices: if you've been discounting, they expect a bargain and
arrive more slowly at any given price next period — a "reference price" that
drifts toward whatever you actually charged. So today's price shapes not
only today's sales but tomorrow's willingness to pay.

Write a program that posts a full pricing schedule to **maximize total
revenue** across a fixed, seeded family of 10 instances.

## Demand model

Let `r` be the current reference price, starting at `r1`. In period `t`
(0-indexed), if you post price `p`, the number of arrivals (before capping to
remaining stock) is:

```
A_t(p) = max(0, (a[t] + g[t]*r) - (b[t] + g[t]) * p)
```

Realized sales this period = `min(A_t(p), stock currently remaining)`.
Revenue this period = `p * sales`. Stock decreases by `sales` and carries
into the next period; the reference price updates:

```
r_next = alpha * r + (1 - alpha) * p
```

`a[t]`, `b[t] > 0`, `g[t] >= 0` are given per period; `alpha` in `[0,1]` and
`r1` are given once. Any stock remaining after period `T-1` perishes worthless.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**.
It runs in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a full price schedule ...
print(json.dumps({"prices": prices}))
```

### Public instance (stdin)

```json
{
  "name": "late_surge",
  "T": 24,               // number of pricing periods
  "C0": 160.0,            // initial stock
  "p_max": 120.0,         // maximum allowed price any period
  "alpha": 0.75,          // reference-price persistence, in [0,1]
  "r1": 48.0,             // initial reference price
  "a": [ ... T floats ... ],   // per-period demand intercept
  "b": [ ... T floats ... ],   // per-period price sensitivity (> 0)
  "g": [ ... T floats ... ]    // per-period reference-price sensitivity (>= 0)
}
```

### Answer (stdout)

```json
{ "prices": [p_0, p_1, ..., p_{T-1}] }
```

- `prices` must be a list of **exactly `T`** finite numbers.
- Every price must satisfy `0 <= p_t <= p_max`.

Any invalid output (wrong length, a non-numeric or out-of-range price, `NaN`/
`Infinity`), a crash, a timeout, or non-JSON output makes that instance score
`0.0`.

## Scoring (deterministic)

For each instance, the evaluator forward-simulates your **full** price
schedule with the exact rules above (it, not you, drives the simulation loop)
to get total revenue `obj`. It also computes, internally, `base`: the revenue
of a single flat price — the monopolist price implied by the horizon's
*average* demand parameters, held constant for all `T` periods and
forward-simulated with the same real dynamics. Your score for the instance is:

```
r = clamp( 0.42 * obj / base, 0, 1 )
```

The reported **Ratio** is the mean of `r` over all 10 instances; the
**Vector** lists the per-instance scores. A curve-blind flat guess scores
well under `0.1`; myopically reposting each period's own revenue-maximizing
price does noticeably better but still leaves real money on the table on
several instances; accounting for how much stock and how much time remain
does substantially better still — and the normalization is built so even a
very good policy stays below `1.0`.

## Suggested strategies

1. **Flat guess** (baseline): one price from period 0's curve, never changed.
2. **Per-period monopolist**: repost each period's revenue-maximizing price,
   reacting to the schedule but ignoring remaining stock and time.
3. **Inventory pacing**: target selling at `remaining stock / remaining
   periods`; scarcity-relative-to-time-left pushes price up (ration), and
   abundance-relative-to-time-left pushes price down (clear before it perishes).
4. **Look-ahead optimization**: the full schedule is known upfront, so
   re-solve (DP, rolling re-optimization, learned pacing corrections) for a
   schedule that beats simple pacing.
