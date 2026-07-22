# Loyalty Ledger: A Year of Grocery-App Coupons

## Story

A grocery app sells one product to several customer **segments** over a 52-week
year. Every segment silently tracks a **reference price** — an exponential moving
average (EMA) of the net prices it actually paid — which is exactly what "feels
fair" to it. A coupon widens the gap between today's price and the segment's
*current* reference price, lifting *this* week's sales — but the EMA also folds
today's price into next week's reference. Discount repeatedly and the reference
price chases the discount down; once it catches up, the bargain feeling is gone and
you are simply selling at a permanently thinner margin.

Each segment has a hidden **memory constant** `alpha` (how fast its reference price
adapts to — and recovers from — a change) and a hidden **sensitivity** `beta` (how
much a price gap moves demand). Before the year starts, a short **pilot** ran per
segment: one discount-shock week, then four full-price weeks. Its sales are
public — `alpha`, `beta` are recoverable from that data in closed form (not given
directly).

## You write a program

Standalone process: read **one** JSON object from **stdin**, write **one** JSON
object to **stdout**.

### Input (stdin)

```json
{
  "name": "year301", "n_weeks": 52, "price": 10.0, "cost": 4.0,
  "max_discount": 0.50, "pilot_depths": [0.30, 0.0, 0.0, 0.0, 0.0],
  "segments": [
    {"id": 0, "size": 1840, "base_rate": 0.41,
     "pilot_demand": [865.3, 700.8, 730.1, 745.9, 754.3]},
    ...
  ]
}
```

- `price`, `cost` are shared; margin per unit at full price is `price - cost`.
- `max_discount`: the largest discount depth (fraction off `price`) you may use.
- `pilot_depths`: the 5 fixed pilot depths (week 0 = shock; weeks 1–4 hold full
  price so the reference can recover).
- Each segment gives `size`, PUBLIC `base_rate` (baseline purchase fraction when
  price equals reference), and `pilot_demand` (units bought each pilot week —
  divide by `size` for fractions).

### Output (stdout)

```json
{ "schedule": [[0.0, 0.0, 0.20, 0.0, ...], ...] }
```

One length-`n_weeks` list per segment (same order as `segments`), each entry a
discount depth in `[0, max_discount]`.

## The law (governs both the pilot and the real year)

Purchase fraction has two components: a **permanent** piece reacting to how far
below list `price` today's net price is (never erodes — cheaper is always somewhat
more attractive), and a **reference-relative** piece reacting to price versus the
*habituated* reference (the part discounting erodes). Fixed split: 25% permanent /
75% reference-driven.

Each week, for a segment with current reference price `ref` (starting at `price`):
```
net_price  = price * (1 - depth)
struct_gap = price - net_price                                 # never erodes
ref_gap    = ref - net_price                                   # erodes with use
frac       = clamp(base_rate + beta*0.25*struct_gap/price
                             + beta*0.75*ref_gap/price,  0, 1)  # purchase fraction
units      = size * frac
profit    += units * (net_price - cost)
ref        = alpha * net_price + (1 - alpha) * ref              # EMA update
```
`alpha`, `beta` are hidden per segment but identifiable: in week 0 `ref = price`
exactly, so `struct_gap == ref_gap` there and week 0's demand pins down `beta`
directly; the four full-price recovery weeks (`struct_gap = 0`) show
`(base_rate - frac_t)` decaying geometrically with ratio `(1 - alpha)`, pinning
down `alpha`.

## Validity

`schedule` must have exactly one row per segment, each row exactly `n_weeks`
finite numeric entries in `[0, max_discount]`. Any shape mismatch, out-of-range
value, non-finite entry, crash, timeout, or non-JSON output scores that instance
`0.0`.

## Scoring (deterministic)

Per instance, the evaluator computes total year-profit (summed over segments) for
three schedules: **weak** = never discount (computable from public data alone),
**oracle** = the exact profit-maximizing schedule with the true hidden `alpha`/`beta`
(a privileged reference, never revealed), and **cand** = your schedule's realized
profit. With an affine anchor:
```
r = clamp(0.1 + 0.9 * (cand - weak) / max(oracle - weak, 1), 0, 1)
```
Never discounting scores ≈0.1; matching the oracle would score 1.0 (not realistically
reachable — you never see `alpha`/`beta` directly and must commit to a full year from
limited pilot data). Final score is the mean `r` over 10 fixed segment populations
(fast-adapting, slow, mixed, plus two larger held-out populations).

## Notes

- Fully deterministic and seeded — no wall-clock or hardware timing.
- Your program only sees the public instance and runs in an isolated sandboxed
  subprocess; hidden parameters and the oracle computation stay in the evaluator.
- No single dominant recipe: whether to discount a segment at all, how deep, and how
  often to space discounts apart all trade off against each other.
