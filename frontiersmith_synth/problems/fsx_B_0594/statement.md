# The Frozen Scriptorium: Insulating a Monastery at the Thermal Min-Cut

Winter has come to a drafty monastery. The building is a **thermal resistor network**:
rooms are nodes, walls are edges, and the freezing outside air is a single grounded node.
The abbot gives you a fixed budget of insulation layers. Spend them well.

## Model

There are `N` rooms (ids `1..N`) plus the **outside**, id `0`, fixed at temperature `Tout`.
Some rooms are **occupied** (cells, scriptorium): their thermostats hold them exactly at
`Tstar`. The rest are **unheated buffer rooms** (cloisters, the nave) whose temperatures
float to steady state.

Each wall connects two nodes and has a base thermal resistance `R0`. Putting `k` insulation
layers on a wall raises its resistance in **series** to `R0 + k*rho`; its heat conductance is
`g = 1 / (R0 + k*rho)`. Some walls are **stained-glass windows**: they have `kmax = 0` and
**cannot be insulated** at all.

At steady state, every floating room `i` satisfies conservation `sum_j g_ij (T_i - T_j) = 0`
over its incident walls (occupied rooms and outside contribute their fixed temperatures).
The quantity you pay for is the **total heat loss to outside**

```
H = sum over walls touching node 0 of  g * (T_room - Tout)
```

which equals the total heat the thermostats must pump in. **Minimize `H`.**

## Input (stdin)

```
N M K Tstar Tout
occ_1 occ_2 ... occ_N          # 1 = occupied (held at Tstar), 0 = floating buffer
a b R0 rho kmax                # wall 1: endpoints a,b in [0..N]; 0 = outside
...                            # M wall lines total
```
`R0`, `rho` are decimals; `Tstar`, `Tout` are integers with `Tstar > Tout`.

## Output (stdout)

`M` integers `k_1 ... k_M`, one per wall in input order: the layers you place on that wall.

## Feasibility

- exactly `M` integers, each `0 <= k_e <= kmax_e` (so windows must get `0`);
- total layers `sum k_e <= K`.

Any violation scores `0`.

## Scoring

Let `B` be the loss `H` of the **do-nothing** allocation (all `k_e = 0`) and `H` your loss.
Because this is a minimization,

```
score = min(1000, 100 * B / H) / 1000
```

Insulating nothing scores `0.10`; halving the loss scores `0.20`; a tenfold reduction caps at
`1.0`. The stained-glass shorts and the finite budget keep the optimum well below the cap, so
there is always headroom.

## What makes this hard

Rooms couple in **series and parallel**. An occupied room may lose heat not through its own
outside wall but through an interior **doorway** into a buffer room whose stained-glass window
shorts that buffer straight to the cold — no amount of insulating that buffer's *exterior*
walls closes the leak. The heat source-to-outside **min-cut** can therefore run through an
*interior* wall the obvious audit never inspects. Ranking walls by their standalone heat flow,
or by area, or insulating only exterior walls, misspends the budget: each layer's true value is
its **marginal** effect on the whole network, and the optimum equalizes marginal resistance
along the cut. The interaction coefficients (`R0`, `rho`, which windows are shorted) live in the
input, so you must read the network, not pattern-match.

## Example

```
2 4 5 20 -20
1 0
1 2 0.50 0.50 8
1 0 1.00 0.50 8
2 0 0.40 0.00 0
2 0 1.00 0.50 8
```
Room 1 is the heated cell (held at 20); room 2 is an unheated buffer with a stained-glass
window (`kmax=0`, resistance 0.40) it cannot insulate. Do-nothing loss is `B = 83.33`.
Dumping all 5 layers on room 1's outside wall (the auditor's pick) leaves the doorway wide open
to the shorted buffer and reaches only `H ~ 60` (score `~0.14`). Splitting layers between the
**doorway** (wall 1) and room 1's outside wall thermally isolates the cell from the lost-cause
buffer and reaches `H ~ 36` (score `~0.23`).

## Constraints

`N <= 12`, `M <= 26`, `1 <= K <= 25`, per-wall `kmax <= 8`. Time limit 5 s, memory 512 MB.
Scoring is exact rational arithmetic — fully deterministic.
