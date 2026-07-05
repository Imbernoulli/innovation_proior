# Ski-Resort Lift Spares: Two-Echelon Safety-Stock Sizing (METRIC)

A large ski resort runs `N` chair/gondola lifts. Each lift is driven by a
drive-motor **controller** that fails at random; when a lift's controller fails
and no spare is on hand, the lift is **down** (a backorder) until a replacement
arrives. Spares are managed by a two-echelon, continuous-review, one-for-one
`(S-1, S)` base-stock system:

- A single **central warehouse** (echelon 0) holds spares and is resupplied
  from an external vendor with lead time `T0` (years).
- Each **lift** `i` holds spares locally and is resupplied from the warehouse
  with nominal transit time `T[i]` (years). Every failure at a lift immediately
  triggers a one-for-one replenishment order on the warehouse.

Controller failures at lift `i` are **Poisson** with rate `lam[i]` per year, so
the warehouse sees aggregate Poisson demand of rate `Lambda = sum(lam)`.

### Coupling (Sherbrooke's METRIC)
Warehouse pipeline demand is Poisson with mean `m0 = Lambda * T0`. If the
warehouse order-up-to level is `s0`, its **expected backorders** are
`EBO0 = E[(D0 - s0)^+]`, `D0 ~ Poisson(m0)`. By Little's law each lift order
waits an extra `delay = EBO0 / Lambda` on average, so lift `i`'s effective
lead-time demand is Poisson with mean `m_i = lam[i] * (T[i] + delay)`.
For an order-up-to level `s_i`, the lift's expected backorders and on-hand are
`EBO_i = E[(D_i - s_i)^+]` and `OH_i = EBO_i + (s_i - m_i)` (clipped at 0);
likewise `OH_0 = EBO0 + (s0 - m0)`.

### Decision
Choose the warehouse level `s0` (integer in `[0, S0_max]`) and every lift level
`s[i]` (integer in `[0, S_max]`).

### Objective (MINIMIZE, dollars/year)
```
C = h0 * OH_0  +  sum_i h[i] * OH_i  +  p * sum_i EBO_i
```
holding cost of on-hand spares (warehouse `h0`, lift `h[i]`) plus a downtime
penalty `p` per expected backordered controller across the resort.

### Service-level constraint (hard SLA)
```
sum_i EBO_i  <=  cap
```
Any answer that violates the SLA — or is malformed / out of range / non-integer
— is **infeasible and scores 0**.

## Input (public instance, one JSON object on stdin)
```
{
  "N": int,                # number of lifts (30)
  "lam": [float]*N,        # per-lift annual failure (Poisson) rates
  "T":   [float]*N,        # per-lift warehouse->lift transit times (years)
  "T0":  float,            # vendor->warehouse lead time (years)
  "h":   [float]*N,        # per-lift holding cost ($/spare/year)
  "h0":  float,            # warehouse holding cost ($/spare/year)
  "p":   float,            # downtime penalty ($/expected-backorder/year)
  "cap": float,            # SLA: max total expected lift backorders
  "Lambda": float,         # sum(lam), provided for convenience
  "S_max": int,            # per-lift cap on s[i]
  "S0_max": int            # cap on s0
}
```

## Output (one JSON object on stdout)
```
{"s0": <int>, "s": [<int>, ... N of them]}
```

## Scoring
Deterministic. The evaluator computes the exact METRIC cost `C` of your design.
A trivial-construction cost `B` (decoupled, near-perfect warehouse + equal
per-lift backorder budget) is the reference. Your per-instance score is
`min(1, 0.1 * B / C)` (lower cost is better), averaged over 10 held-out
instances. A trivial design scores about `0.1`; beating it requires
per-lift marginal economics and genuine two-echelon co-optimization, and there
is deliberate headroom above any single heuristic.

The candidate runs in an isolated sandbox and sees only the public instance
above; the objective and constraint are evaluated in the parent process.
