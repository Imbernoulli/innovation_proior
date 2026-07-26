# Starter Peaks

## Problem

You run a sourdough bakery with `T` starter-culture tanks and `H` days of
horizon. Each tank `i` has an integer **vitality** `v` (0..200, `SCALE=200`)
and a **satiety counter** `n` (starts at `v=0, n=0`). Every day you may feed
each tank at most once, spending flour from a shared total `BUDGET`. Feeding
tank `i` costs `COST_i` units of flour.

Each tank has fixed integer parameters `GROW, BASE_KICK, DECAY, SAT,
CRASH_DIV`. Vitality updates **once per day, for every tank**, by whether that
tank was fed that day:

* **Fed:** `n += 1`. If `n > SAT` (overfed / acid crash):
  `v = v // CRASH_DIV`, then `n` resets to `0`.
  Otherwise (still within its safe window) vitality gets a **logistic**
  growth kick: `v = min(SCALE, v + (GROW*v*(SCALE-v)) // SCALE^2 + BASE_KICK)`.
* **Unfed:** `n = max(0, n-1)` (cools down); vitality decays:
  `v = (v * (1000 - DECAY)) // 1000`.

So a tank fed every day is NOT a ramp to a plateau — it is a **pulse**: it
climbs, crashes once satiety overflows, climbs again. Feeding for a long time
does not raise the ceiling; it just cycles the pulse. Feeding *less*, timed to
land the climb exactly where you need it, can beat feeding *more*.

There are `M` dated orders. Order `j` needs delivery on day `DAY_j` (a **bake
day**, i.e. after `DAY_j` daily updates from the start): it is fulfilled iff
**some** tank's vitality on that day is `>= THETA_j`, and pays `VALUE_j`.
Any tank may satisfy any number of orders across different days; nothing is
consumed by baking itself.

## Input (stdin)

```
T H M BUDGET
GROW BASE_KICK DECAY SAT CRASH_DIV COST      (one line per tank, T lines)
DAY THETA VALUE                              (one line per order, M lines)
```
All values are non-negative integers; `1 <= DAY_j <= H`.

## Output (stdout)

```
K
tank_1 day_1
...
tank_K day_K
```
`K` feed events (any order), each an integer tank index (`0..T-1`) and day
(`0..H-1`), meaning "feed this tank on this day". No `(tank, day)` pair may
repeat.

## Feasibility

* All tokens must be present, integer, and finite; `K >= 0` and every
  `(tank, day)` in range with no duplicates.
* Total flour spent, `sum(COST_tank)` over all feed events, must be
  `<= BUDGET`.
Any violation scores `Ratio: 0.0`.

## Objective & Scoring

Let `F` = sum of `VALUE_j` over fulfilled orders under your schedule.
The checker also simulates its own baseline `B`: feed only the single
cheapest tank, every day of the whole horizon, and sum the orders that
passive strategy fulfills. Score:

```
Ratio = min(1.0, F / (10 * B))
```

printed as `Ratio: <value>`. Matching the baseline scores `0.1`; ten times
the baseline saturates the cap.

## Constraints

`2 <= T <= 8`, `16 <= H <= 60`, `3 <= M <= 10`, time limit 5s, memory 512m.

## Example (worked score)

`T=1,H=15,M=1,BUDGET=100`, tank `GROW=110 BASE_KICK=2 DECAY=20 SAT=10
CRASH_DIV=15 COST=5`, order `DAY=11 THETA=100 VALUE=100`.

Feeding tank 0 continuously from day 0 (the "obvious" plan, 11 feed days)
gives vitality `0,2,5,9,15,24,37,55,78,106,135,9` on days `0..11` — the 11th
feed pushes satiety past `SAT=10` and crashes it right on delivery day
(`9 < 100`): **order missed**, and it cost `11*5=55` flour.

Feeding tank 0 only on days `1..10` (10 feed days, `50` flour — start the
burst one day later instead of day 0, so the crash lands one day *after* the
deadline instead of on it) gives `v(11)=135` (`0,0,2,5,9,15,24,37,55,78,106,135`
for days `0..11`) — comfortably above `100`: **order met**, for less flour.
That is the whole game: choose *when* to start feeding so the peak lands on
the calendar, not just *how much* to feed.
