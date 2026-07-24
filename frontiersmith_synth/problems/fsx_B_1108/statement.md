# Chill the Grid: Sinking Heat Against Fourth-Power Radiation

A power electronics panel is a network of `n` heat-generating junctions joined by
conductive thermal straps. Every junction also radiates heat to space, but the
radiative flux grows with the **fourth power** of temperature, so moderate
junctions barely radiate at all. You may bolt down `k` **perfect heat sinks**
(each clamps its junction to the ambient temperature 0). Place them to keep the
hottest junction in the whole network as cool as possible.

## Physics

Every junction `i` generates heat at a fixed rate `g_i > 0` and radiates with
coefficient `a_i > 0`. An undirected edge `(i, j)` conducts heat between its
endpoints at rate `c_ij * (T_i - T_j)`, where `c_ij > 0` is its conductance.
Ambient temperature is 0, so at steady state every junction that does **not**
carry a sink satisfies

```
g_i  =  a_i * T_i^4  +  sum over neighbours j of  c_ij * (T_i - T_j)
```

A junction carrying a sink is clamped at `T_i = 0` (the sink absorbs whatever
heat arrives). This system is the gradient of a strictly convex energy, so a
unique non-negative steady-state temperature field `T` exists for any sink set.
The checker recovers it by damped Newton iteration converged to a residual
below `1e-9`.

## Input (stdin)

```
n m k
g_0 g_1 ... g_{n-1}
a_0 a_1 ... a_{n-1}
u v c              (m lines: undirected conductive edges, 0-indexed)
...
```

`n` junctions, `m` edges, sink budget `k`. All `g_i`, `a_i`, `c` are positive
decimals. The network is connected.

## Output (stdout)

```
M
i_1
i_2
...
i_M
```

`M` on the first line, then the `M` distinct junction indices (0-indexed) where
you place sinks, one per line.

## Feasibility

`0 <= M <= k`; every index in `[0, n)`; no duplicates; no extra tokens. Any
violation (including non-integer or non-finite tokens) scores `0`.

## Objective & Scoring

Let `F = max_i T_i` be the steady-state hotspot temperature under your sink
placement (smaller is better). The checker independently computes a baseline
`B` = the hotspot temperature when the `k` sinks are placed at junctions
`0 .. k-1`. Your score is

```
Ratio = min(1.0, 0.1 * B / F)
```

so replicating the first-`k`-junctions baseline scores `0.1`, and you climb by
pushing the hotspot far below what that naive placement achieves.

## What makes it hard

Because radiation scales like `T^4`, only genuinely hot junctions dump
significant heat to space; everywhere else heat must leave by **conduction**.
The steady state is a global fixed point: a junction's temperature is set not
by how much heat it makes but by how little of that heat can escape — through
thin conductive necks and weak local radiators. The networks in the tests are
wired so that the loudest heat *sources* sit in well-ventilated regions, while
modest sources huddled behind narrow conductive bottlenecks accumulate far more
heat. Sinks placed where heat is *generated* leave those traps untouched;
sinks placed where heat *cannot leave* dismantle the true hotspot. Balancing
how many sinks each region deserves requires evaluating the fixed point, not
reading the generation field.

## Example scoring

Suppose `B = 10.0` (first-`k` sinks leave a hotspot of 10) and your placement
achieves `F = 4.0`. Then `Ratio = min(1, 0.1 * 10.0 / 4.0) = 0.25`. A placement
with `F = 20.0` (worse than baseline) scores `0.05`.

## Constraints

`60 <= n <= 320`, `k <= 9`, `m <= 2200`. Time limit 5s, memory 512MB. All
scoring is deterministic: same placement, same score, on any machine.
