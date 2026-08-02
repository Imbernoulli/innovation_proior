# Ward Roster: Flexible Cover for a Predictable Surge

## Problem

A hospital ward runs a 24-hour clock, hour `0..23`. Staffing is bought in
three fixed, non-overlapping **8-hour shift blocks**: block 0 = hours
`0..7`, block 1 = hours `8..15`, block 2 = hours `16..23`. For each block
you decide a headcount of two skill tiers: **junior** (`J`) and **senior**
(`S`). A junior worker can only treat **low-acuity** patients. A senior
worker can treat low- OR high-acuity patients (skill superset), but each
senior is one physical person -- an hour they spend on a high-acuity
patient is an hour they are not covering a low-acuity one.

Patients arrive every hour with a **low-acuity** count `L(t)` and a
**high-acuity** count `H(t)`, one unit = one patient needing one staff-hour.
You are given `K` **held-out arrival days**, each a full 24-hour `(L, H)`
profile; your roster is built once and then replayed against all `K` days.
Arrivals are non-stationary: some days carry a sharp, recurring evening
surge (with a heavier mix of high-acuity patients) concentrated in a few
hours of block 2; other days stay close to flat all day.

If your scheduled roster falls short in some hour, the shortfall must be
covered, per skill:
- **Overtime**: staff already scheduled in that block can be asked to do
  more. Available overtime capacity at skill `k` in a block is
  `floor(ot_num_k * scheduled_k / ot_den_k)` -- proportional to how many of
  that skill you actually scheduled there (a block you left empty gets no
  overtime), at a per-unit overtime cost.
- **Agency**: unlimited call-in staff of either skill, at a per-unit cost
  that is always strictly more expensive than overtime.

High-acuity demand is served, in order: scheduled senior capacity (free --
already paid for), senior overtime, then senior agency. Low-acuity demand
is served, in order: scheduled junior capacity (free), idle scheduled
senior capacity left over after high-acuity is served (free), junior
overtime, idle senior overtime left over after high-acuity, then junior
agency. This is cheapest-eligible-source-first throughout (free scheduled
capacity, then overtime, then agency) -- exactly how the checker computes
cost, so the fill order is not a hidden extra decision for you to make.

**Illustrative form only** (not a real test instance): with one block,
`base_S=2` and a hour needing `H=1, L=3`: the senior covers the 1 high-acuity
patient, its 1 leftover senior unit covers 1 of the 3 low-acuity patients,
and the remaining 2 low-acuity patients fall to junior/overtime/agency per
the order above.

## Input (stdin)
```
T
n_starts
start_0 ... start_{n_starts-1}
MAX_PER_SLOT
cost_base_J cost_base_S
ot_num_J ot_den_J ot_num_S ot_den_S
cost_ot_J cost_ot_S
cost_agency_J cost_agency_S
K
(K blocks of T lines: "L(t) H(t)")
```
`T=24`, `n_starts=3`, blocks are 8 hours each in the order given by
`start_i`. `MAX_PER_SLOT` bounds any one block's headcount for either skill.

## Output (stdout)
Exactly `n_starts` lines, one per shift block in the given order:
```
J_count S_count
```

## Feasibility
Exactly `n_starts` lines, two finite integers each, `0 <= J_count,
S_count <= MAX_PER_SLOT`. No missing/extra tokens. Any violation scores
`Ratio: 0.0`.

## Objective (minimize)
`F = base_wage_cost(roster) + average over the K days of that day's
overtime + agency spend`, where base wage cost pays `cost_base_J` /
`cost_base_S` per scheduled unit per block regardless of hours actually
used, and the daily overtime/agency spend is computed by the fill order
above.

## Scoring
The checker also evaluates the empty roster (`0` everywhere -- always
feasible, all demand via agency) as its own baseline `B`. Your score is
`min(1.0, 0.1 * B / F)`.

## Constraints
`K` up to 10, `MAX_PER_SLOT` up to 30, all costs and demand small positive
integers. Time limit 5s.
