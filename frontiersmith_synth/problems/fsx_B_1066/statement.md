# Rainproof Season

## Problem

A league of `N` teams (even) plays a single round robin: every unordered
pair of teams meets exactly once, so there are `G = N*(N-1)/2` games. The
season calendar has `D` dates and `C` courts per date (a game needs one
court on one date; no team can play twice on the same date).

You must output a full schedule: assign every game to a `(date, court)`
slot. This fixes the schedule that goes to print *before* the weather is
known — the same schedule must then survive every scenario below.

The input also lists `K` possible weather scenarios. Scenario `s` is a
**fixed set of dates that get rained out entirely** — no game, original or
rescheduled, can be played on a cancelled date in that scenario. Exactly one
scenario will occur, but you don't know which, so your schedule is judged by
its **worst** scenario.

**Fixed makeup rule.** Given your schedule and a scenario, every game whose
date was cancelled is postponed. Process the postponed games in ascending
order of `(original date, original court)`. For each one, in that order,
find the *smallest* date strictly after its original date that is (a) not
itself cancelled, (b) not already full-court at this point in the
processing, and (c) has both teams still free at that point in the
processing (a team is "free" on a date if no *currently active* game of
theirs — original or already-relocated — sits there). Send the game to that
date, on the smallest free court there. If no such date exists within `D`,
keep searching later dates beyond the calendar (courts there still cap at
`C`, and never cancelled) — this always terminates. Because postponed games
are resolved one at a time in this fixed order, an early postponement can
consume the very slot a later one needed, cascading it further out.

**Costs.** `delay(game) = final_date - original_date` (0 if never
postponed). For each team, sort the final dates of its `N-1` games; every
consecutive gap smaller than `R` contributes `(R - gap)` to that team's
rest shortfall. `scenario_cost = sum(delay) + LAMBDA * sum(rest shortfall
over all teams)`. Your objective value is `max` over the `K` scenarios of
`scenario_cost` — minimize it.

## Input (stdin)

```
N D C K R LAMBDA
b_1 date_1 date_2 ... date_{b_1}
...
b_K date_1 ... date_{b_K}
```
Line 1: team count, calendar length, courts/date, scenario count, minimum
rest gap, rest-penalty weight. Then `K` lines, one per scenario: `b_s`
followed by its `b_s` cancelled dates (sorted, in `[1, D]`).

## Output (stdout)

`G = N*(N-1)/2` lines, in the fixed order `(1,2),(1,3),...,(1,N),(2,3),...,
(N-1,N)` (team indices `1..N`): each line `date court` for that pair's game.

## Feasibility

- Exactly `G` `(date,court)` pairs, `date` in `[1,D]`, `court` in `[1,C]`.
- No two games share a `(date,court)` slot.
- No team appears in two games on the same date.
Any violation scores `Ratio: 0.0`.

## Scoring

Let `F` be your worst-case scenario cost. The checker also builds its own
naive feasible schedule internally and computes its worst-case cost `B`.
Score `= min(1000, 100*B/max(1e-9,F)) / 1000`, printed as `Ratio: <value>`.
A schedule matching the naive baseline scores near `0.1`; a schedule with a
third of the naive worst-case cost scores near `0.3`, and so on, capped at
`1.0`.

## Constraints

`6 <= N <= 16`, `C = N/2`, `2 <= K <= 10`, `R = 2`, `LAMBDA = 3`,
`D` large enough that a feasible schedule always exists. Time limit 5s.

## Example (worked, illustrative shape only)

`N=4, D=4, C=2, K=1, R=2, LAMBDA=3`, one scenario cancelling date `{1}`.
Feasible schedule: date 1 = `(1,4)` court 1, `(2,3)` court 2; date 2 =
`(1,3)` court 1, `(2,4)` court 2; date 3 = `(1,2)` court 1, `(3,4)` court 2
(date 4 unused). The scenario cancels date 1, postponing `(1,4)` and
`(2,3)` (processed in that court order). `(1,4)`: date 2 has team 1 busy
(`(1,3)`); date 3 has team 1 busy (`(1,2)`); date 4 is free for both ->
delay 3. `(2,3)`: date 2 has team 2 busy; date 3 has team 2 and team 3 both
busy; date 4 now has team 1 and team 4 occupied but a second court is open
and teams 2,3 are free there -> delay 3. `delay_sum = 6`. Final dates:
team 1 plays on `{2,3,4}`, team 2 on `{2,3,4}`, team 3 on `{2,3,4}`, team 4
on `{2,3,4}` — every team has two consecutive gaps of 1 (< R=2), so each
team's rest shortfall is `(2-1)+(2-1)=2`, giving `rest_pen_sum = 8`.
`scenario_cost = 6 + 3*8 = 30 = F`.
