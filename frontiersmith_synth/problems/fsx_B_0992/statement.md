# The Flywheel Furnace: 300-Day Blend & Purge Planning

## Problem
You run a secondary aluminum melt shop for `T` days. Every day the furnace
can melt up to `CAP` tons, blended from three sources:

- **Virgin ingot**: 0 ppm contaminant, cost `CV` $/ton, drawn from a single
  **lifetime** budget `V` shared across all `T` days (not replenished daily —
  once it's spent, it's gone for the rest of the run).
- **That day's scrap offers**: `K_t` lots, each `(avail, ppm, price)`. You may
  buy any amount `0..avail` of a lot; unbought volume expires at day's end.
- **Internal return scrap**: drawn from a pool that behaves like a single
  well-mixed tank. A fixed fraction `RETURN_FRAC` of *whatever mass you melt
  on any day* physically becomes internal scrap and rejoins this pool `LAG`
  days later, carrying the ppm of the blend that produced it — mixing into
  the pool as a mass-weighted average. Whatever you draw from the pool on a
  given day comes out at that day's pool-average ppm. (The remaining
  `1 - RETURN_FRAC` of what you melt is sold and leaves the system.)

The furnace physically cannot run an out-of-spec charge: the blended ppm of
everything melted on a day must not exceed the hard cap `PPM_CAP`. Sale price
is not a step function of the cap, though — it falls smoothly the closer you
run to it: `price = P0 * (1 - BETA * (ppm/PPM_CAP)^2)`.

## Input
```
T
CAP V RETURN_FRAC LAG PPM_CAP P0 CV BETA
K_1
avail_1 ppm_1 price_1        (K_1 lines)
K_2
...                           (repeated for t = 1..T)
```

## Output
For `t = 1..T`, in order, print `virgin_t x_{t,1} ... x_{t,K_t} return_t` —
all non-negative reals (tons).

## Feasibility (every day; any violation scores the whole run 0)
- `virgin_t >= 0`; cumulative `sum(virgin_t) <= V`.
- `0 <= x_{t,j} <= avail_{t,j}` for every lot.
- `0 <= return_t <=` the pool's mass on day `t` (after that day's inflow).
- `virgin_t + sum_j x_{t,j} + return_t <= CAP`.
- The day's blended ppm `<= PPM_CAP`.

## Objective (maximize)
For each day, `sold = (1-RETURN_FRAC) * mass_t`, `revenue = price(ppm_t) *
sold`, `cost = CV*virgin_t + sum_j price_{t,j}*x_{t,j}` (return scrap is
free). `F = sum_t (revenue_t - cost_t)` over the whole run.

## Scoring
The checker computes its own baseline `B`: the profit of melting **only**
virgin metal, every day, up to `min(CAP, remaining V)` — no scrap, no pool.
Your score on a test is `ratio = min(1000, 100*max(F,0)/B) / 1000`; the
reported score is the mean ratio over 10 tests.

## Constraints
- `1 <= T <= 300`, `1 <= K_t <= 8`, all economic constants positive reals
  read from the input (never assume the numbers above are the actual test
  values).
- Time limit: 4 s. Memory limit: 256 MB. Fully deterministic.
- Running a blend right at `PPM_CAP` never violates feasibility, but its
  nonlinear quality discount taxes revenue on every ton sold that day.
- The pool only "remembers" the ppm of what you fed it: keeping today's
  blend clean is the only lever that lowers the pool's ppm `LAG` days from
  now — no action taken on the draw side changes the pool's ppm, only its
  mass.

## Example
`CAP=10 V=5 RETURN_FRAC=0.4 LAG=1 PPM_CAP=1000 P0=2000 CV=900 BETA=0.6`, `T=2`.
Day 1 offers one lot `avail=10 ppm=500 price=300`; day 2 offers one lot
`avail=10 ppm=1300 price=100` (too dirty to use alone).

**Day 1**: buy all 10 tons of the lot (`virgin=0, x=10, return=0`). `mass=10,
ppm=500`, quality `= 1-0.6*0.5^2=0.85`, price `=1700`, sold `=6`,
`revenue=10200`, `cost=3000`, `profit=7200`.

**Day 2**: the pool now holds `0.4*10=4` tons at `ppm=500` (LAG=1). Buy
`x=4` of the dirty lot, draw the whole pool (`return=4`), and spend `virgin=2`
to dilute: `mass=10, ppm=(4*1300+4*500)/10=720 <= 1000`, quality
`=1-0.6*0.72^2≈0.689`, price `≈1377.9`, sold `=6`, `revenue≈8267.5`,
`cost=900*2+100*4=2200`, `profit≈6067.5`.

`F ≈ 13267.5`. Baseline: `massB=min(20,5)=5`, `B=2000*0.6*5-900*5=1500`.
`ratio = min(1000, 100*13267.5/1500)/1000 ≈ 0.885`.
