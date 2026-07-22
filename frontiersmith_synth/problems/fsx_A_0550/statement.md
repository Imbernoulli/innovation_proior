# Firebreak Dispatch on the Advancing Front

A wildfire is loose on an `N x N` grid of cells. It ignites at a set of **seed** cells
and spreads deterministically outward, driven by fuel and wind. You command a small
number of fire crews. Write a program that decides **where and when** to send them to
build firebreaks so as to **maximize the total value of grid area still unburned** at a
fixed horizon.

Your program reads ONE public instance as JSON on stdin and writes ONE JSON answer on
stdout.

## Fire model (deterministic)

Cells are 4-connected. Each cell `v` has an integer fuel-resistance `res[v] >= 1`. Fire
spreads from a burning cell `u` to an orthogonal neighbour `v` with integer step cost

```
cost(u -> v) = max(1, res[v] + wind_strength * align(u -> v))
align(u -> v) = -( (v.r - u.r, v.c - u.c) . wind )     # dot product with the wind vector
```

So moving **down-wind** is cheap (fire runs fast) and **up-wind** is costly. Seed cells
ignite at time `0`; every other cell's ignition ("arrival") time is the minimum total
step cost of any path from a seed — an ordinary weighted-BFS / shortest-path field. A
cell **burns** at its arrival time; a cell whose arrival time exceeds `horizon` never
burns this episode. A cell that burns at time `a` is credited with the **fraction
`a / horizon` of its value it survived**; a cell that never burns (arrival beyond the
horizon, or protected by a firebreak) keeps its **full** value. So delaying the front is
worth something, but keeping ground out of the fire's reach entirely is worth the most.

## Crews, commitment and firebreaks

You may dispatch crews to build firebreaks. A firebreak at cell `x` dispatched at step
`d` **completes at step `d + build_time`**. While a crew builds it is **locked** — you
have `crews` teams, so at every instant **at most `crews` builds may be in progress**
(a build occupies the interval `[d, d + build_time)`).

A completed firebreak is a permanent barrier — its cell never burns and fire cannot
spread through it — **but only if it finishes before the fire arrives**: it counts iff
`d + build_time <= arrival(x)`. A break whose cell ignites during construction is
overrun and does nothing; the cell burns as ordinary fuel. Building a barrier changes
the arrival field of the cells behind it (they must now be reached another way, or not
at all).

## Input (stdin, one JSON object)

```
{ "name": str, "n": N,
  "res":   [[...], ...],   # N x N ints >= 1
  "value": [[...], ...],   # N x N ints >= 0
  "seeds": [[r, c], ...],  # ignition cells
  "wind":  [wr, wc],       # each in {-1,0,1}
  "wind_strength": ws,     # int >= 0
  "build_time": b,         # steps per firebreak
  "crews": C,              # max simultaneous builds
  "horizon": T }           # scoring horizon
```

## Output (stdout, one JSON object)

```
{ "breaks": [[r, c, d], ...] }   # firebreak at (r,c), dispatched at step d
```

Requirements (any violation ⇒ that instance scores 0):
- each `[r, c, d]`: integers with `0 <= r,c < N` and `d >= 0`;
- all firebreak **cells distinct**;
- **lock-in capacity**: for every instant `t`, the number of breaks with `d <= t < d + b`
  is at most `C`.
An **empty** list is legal — it dispatches nobody (the do-nothing baseline).

## Objective and scoring

`saved(breaks)` = the survival-weighted value defined above (each cell's `value` times the
fraction of the horizon it survives; unburned/protected cells count fully).
Per instance, with `y_base = saved([])` (do-nothing), `y_ub = ` sum of all cell values,
and `y_cand = saved(your breaks)`:

```
r = clamp( 0.1 + 0.9 * (y_cand - y_base) / (y_ub - y_base), 0, 1 )
```

Doing nothing scores ~0.1; saving more valuable ground scores higher. The cells near the
seed always burn and crews are scarce, so the score stays well below 1 — there is room
above any reference solution. Your final score is the mean of `r` over all instances.

**The catch.** Throwing crews at the cells the fire is touching *now* wastes them: the
front sweeps around and past a short wall while the crews sit locked. The high-value
ground is protected by narrow places the wind and fuel will funnel the fire through
*later*. Read where the front is going and stand a barrier there before it arrives.

Determinism only: no wall-clock, no randomness. Time limit 5s, memory 512 MB.
