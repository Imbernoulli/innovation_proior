# Glass Furnace Product Wheel: Dilution-Routed Changeovers

## Story

A glass furnace produces `K` coloured products against `K` steady demand streams by
running a repeating **wheel**: a cyclic list of production campaigns. Each campaign
is one colour and one lot size (tons). Colours sit on a continuous **tint line**;
each colour `j` has a purity threshold `tau_j` -- how much foreign pigment its spec
tolerates.

Switching the melt from colour `i` to colour `j` is **not** a fixed changeover
matrix. The residual pigment from `i` must be flushed out with fresh melt, and each
flush ton dilutes the residual multiplicatively (exponential decay, rate `lambda` per
ton). The number of flush tons needed to bring the mismatch `|tint_i - tint_j|` below
`j`'s own threshold `tau_j` is:

```
diff = |tint_i - tint_j|
steps = 0                                        if diff <= tau_j
steps = ceil( log(diff / tau_j) / log(1/lambda) ) otherwise
waste(i, j) = flush_cost * steps                  (tons, wasted, direction matters)
```

Because `waste(i, j)` depends on the **target's own threshold**, not a symmetric
"distance", this cost is not a fixed matrix: routing `i -> k -> j` by actually
**producing** an intermediate colour `k` can flush far fewer total tons than `i -> j`
direct, whenever `k`'s spec is loose (cheap to reach) even though `k` sits far from
`i`. A colour can *catalyze* other colours' changeovers -- but inserting it costs a
real campaign (at least `k`'s minimum lot), which changes the cycle time and every
colour's inventory swing.

## Input (public instance, one JSON object on stdin)

```json
{"name": "wheel07", "k": 7,
 "colors": [{"id": 0, "tint": 12, "tau": 30, "demand": 0.07,
             "hold": 1.8, "back": 11.4, "min_lot": 9}, ...],
 "lambda": 0.5, "flush_cost": 5, "waste_price": 3,
 "cycles": 12, "max_campaigns": 15, "max_lot": 500}
```

- `colors[i]`: `tint` (int), `tau` (int, purity threshold), `demand` (tons/period,
  float `> 0`), `hold`/`back` (holding / backlog cost per ton per period),
  `min_lot` (minimum campaign size, int).
- `lambda` (0<lambda<1): per-flush-ton dilution decay rate.
- `flush_cost` (int): tons wasted per flush step. `waste_price`: cost per wasted ton.
- `cycles` (`R`): the wheel repeats `R` times; this scales total cost.
- `max_campaigns`, `max_lot`: bounds on your answer.

## Output (one JSON object on stdout)

```json
{"wheel": [{"color": 0, "lot": 40}, {"color": 3, "lot": 12}, ...]}
```

`1 <= len(wheel) <= max_campaigns`. Each entry's `color` is an integer in `[0,k)`
(colours MAY repeat across entries) and `lot` is an integer with
`colors[color].min_lot <= lot <= max_lot`. **Every colour id `0..k-1` must appear at
least once** -- every demand stream must actually be served each cycle. This is one
full cycle of your wheel; it repeats `cycles` times. Anything else (bad types,
out-of-range values, a missing colour, a crash, a timeout, non-JSON) scores that
instance `0.0`.

## Objective and scoring (deterministic, minimize)

Build the cyclic timeline of your `wheel`: before every campaign (including the
wrap-around from the last campaign back to the first), pay `waste(prev_color,
this_color)` tons, then produce `lot` tons of `this_color`. This fixes one cycle's
length `T_cyc`. Your wheel then repeats back-to-back for `cycles` laps. Every
colour's inventory starts at 0 and is tracked continuously across ALL laps (no
reset): it jumps up at that colour's own campaigns and drains at its `demand` rate
the rest of the time. If a colour's lot per lap is less than `demand * T_cyc`, it
falls further behind on every lap -- backlog is real and compounds, it is not
averaged away. Total cost = `waste_price * (tons wasted per lap) * cycles` + every
colour's holding cost (`hold * excess`, integrated) and backlog cost (`back *
shortage`, integrated) over the WHOLE horizon.

The evaluator also builds a weak reference wheel (visit every colour once, in the
given input order, lot sized to a generic demand share -- ignoring routing
entirely) with cost `cost_base`. Your score on an instance is:

```
r = clamp( 0.1 * cost_base / max(cost_candidate, eps), 0, 1 )
```

Reproducing the naive wheel scores ~0.1; less waste and less imbalance raise your
score, capped at `1.0`. Final score is the mean `r` over 10 instances of varying
size, utilization, and colour geometry, including held-out cases.

## Notes

- Scoring never measures wall-clock time; treat the limit as a compute budget.
- Your program runs in an isolated subprocess and sees only the public instance.
