# Ridgeline Grid: Aging-Aware Battery Arbitrage

You control one grid-scale battery trading energy against a fixed, fully
known day-ahead price series `prices[0..T-1]`. At every timestep `t` you may
**charge** (buy grid energy), **discharge** (sell stored energy), or hold.
The whole series is given up front — offline planning, not a reactive game.

Two frictions compose into one objective:

1. **Round-trip efficiency loss.** Charging has efficiency `eta_c`,
   discharging has efficiency `eta_d`. Spending `x` grid-side units charging
   stores only `x * eta_c`. Withdrawing `d` battery-side units discharging
   sells only `d * eta_d`. A round trip only nets money once the sell price
   clears roughly `buy_price / (eta_c * eta_d)` — spreads narrower than that
   are a guaranteed loser even before aging.

2. **Cycling-depth aging.** Every discharge step of size `d`, taken while
   usable capacity is `cap`, fades that capacity by
   `aging_coeff * capacity0 * (d / cap)^2` — **quadratic** in that step's
   depth of discharge. The dollar cost `degradation_price * fade` is
   subtracted from profit immediately, and the capacity loss shrinks all
   *future* headroom. Because the penalty is convex, splitting the same
   total energy across several shallower discharge steps costs strictly
   less aging than dumping it in one deep step. Charging never ages the
   battery here.

The series is mostly small noise (many locally tempting but marginal
spreads) punctuated by a handful of genuinely large swings. Trading every
attractive-looking ripple over-cycles the battery: efficiency tax plus
quadratic aging erodes the gains, and the battery is often capacity-starved
by the time a real swing arrives. The better play: trade only spreads wide
enough to clear the true marginal cost (read from this instance's own
fields), and shallow-cycle so capacity survives for the big swings.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from stdin,
write ONE JSON answer to stdout. Runs isolated, sees only the instance.

```python
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"actions": actions}))
```

### Public instance (stdin)

```json
{
  "name": "grid901", "T": 60,
  "prices": [50.1, 49.7, 52.3, ...],  // T floats
  "capacity0": 60.0, "power_max": 14.0,   // max usable capacity, max |x_t|
  "eta_c": 0.93, "eta_d": 0.93,           // charge / discharge efficiency
  "soc0": 30.0,                           // starting state of charge
  "aging_coeff": 0.03,                    // depth-aging coefficient
  "degradation_price": 6.0                // $ per unit of capacity lost
}
```

### Answer (stdout)

`{ "actions": [x_0, ..., x_{T-1}] }` — `x_t > 0` charges (buy) `x_t`
grid-side units; `x_t < 0` discharges (sell) `-x_t` battery-side units;
`x_t == 0` holds. Must be a list of exactly `T` finite numbers.

## Feasibility and objective (checked exactly, on the full instance)

Starting from `soc = soc0`, `cap = capacity0`, for `t = 0..T-1` in order:

- `|x_t| <= power_max` (tolerance 1e-6).
- If `x_t > 0`: require `soc + x_t*eta_c <= cap`; pay `x_t*price_t`; then
  `soc <- soc + x_t*eta_c`.
- If `x_t < 0`, let `d = -x_t`: require `d <= soc`; earn
  `d*eta_d*price_t`; then `dod = d/cap`,
  `fade = aging_coeff*capacity0*dod^2`, pay `degradation_price*fade`,
  `cap <- max(cap - fade, 0.05*capacity0)`, `soc <- soc - d`.

Any shape violation, an infeasible step, a crash, a timeout, or non-JSON
output scores that instance `0.0`. Otherwise your **profit** is total
earnings minus total payments above. **Maximize** profit (as a normalized
ratio, see below) across a fixed seeded family of 10 instances — varying
length, capacity, power limit, efficiencies, aging severity, and
noise/swing structure; several are larger held-out cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself, a **loose, deliberately
unreachable** upper bound `q_ideal`: the true optimal profit under the real
`capacity0`/`soc0`/`power_max`/efficiency constraints but with **zero aging
cost** (capacity never fades). Since any real feasible trace pays aging
cost `>= 0`, this always dominates a real strategy's profit. Then:

```
r = clamp( 0.1 + 0.9 * cand_profit / q_ideal, 0, 1 )
```

Doing nothing scores exactly `0.1`. The reported **Ratio** is the mean of
`r` over instances; **Vector** lists the per-instance values.

## Suggested strategies

1. **Do nothing** — the reference floor.
2. **Fixed-margin threshold trading** — full power whenever price crosses a
   flat band around the series mean.
3. **Breakeven-aware valley/peak selection** — derive the true breakeven
   from this instance's own fields, trade only spreads that clear it,
   shallow-cycle the execution.
4. **Global capacity budgeting** — allocate capacity across every
   surviving opportunity jointly, not leg-by-leg.
