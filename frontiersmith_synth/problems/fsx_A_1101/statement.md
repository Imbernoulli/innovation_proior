# Night Depot LIFO

An electric-bus depot charges its fleet overnight. The depot has `R`
drive-through parking rows. Buses enter a row from the back and leave from
the front, so **a bus can exit only after every bus parked ahead of it in its
row has left**: each row is a FIFO lane, and the exit order inside a row is
exactly the order in which the row was filled. To pull out a blocked bus, the
depot must *shunt*: every bus still ahead of it in the row is dragged out and
pushed back, costing **1 shunt move per blocking bus** (shunted buses keep
their positions).

You choose where each bus parks and when it charges. Get every bus fully
charged by its departure hour while spending as little as possible.

## Timeline and rules

The horizon is `T` discrete hours `0 .. T-1`. Bus `i` (`0 <= i < n`):

- **arrives** at the start of hour `a_i` and is parked immediately — you
  choose its row; it takes the next free spot at the back of that row;
- **departs** at the start of hour `d_i` (`a_i < d_i`), and must have
  received exactly `E_i` kWh by then;
- may charge only during hours `h` with `a_i <= h < d_i`, while parked.

All departure hours in an instance are distinct, so the morning exit order is
forced: buses leave in increasing `d_i`.

**Chargers.** Each row has one charger at its head delivering at most `P`
kWh per hour in total to buses of that row. Its cable reaches only the first
`W` positions of the row: at hour `h` a bus may charge only if fewer than `W`
buses that are still present were parked ahead of it in its row. (When a bus
exits, the buses behind it roll forward, so positions are recomputed every
hour among present buses.)

**Transformer.** In hour `h` the total energy delivered to all rows may not
exceed `cap[h]` kWh, and each kWh delivered in hour `h` costs `prc[h]`
(integer). Prices and capacities vary over the night: the evening peak is
expensive, the small hours are cheap but the transformer is tight, the early
morning is moderate again.

## Input (stdin)

```
n R W T P SH
prc_0 ... prc_{T-1}
cap_0 ... cap_{T-1}
a_0 d_0 E_0
...
a_{n-1} d_{n-1} E_{n-1}
```

All integers. `SH` is the cost of one shunt move. `3 <= R <= 8`,
`2 <= W <= 3`, `n <= R*W`, `T = 36`, `a_i <= 4 < 12 <= d_i`, `P <= E_i <= 2P`.

## Output (stdout)

1. `n` lines `bus row`: the parking sequence. It must list every bus exactly
   once in nondecreasing arrival hour (ties in any order you like); `row` in
   `[0, R)`. Each bus is appended to the back of its row in this order.
2. One line with the `n` bus ids in exit order (must be nondecreasing `d_i`).
3. One line `C`, then `C` lines `bus hour kwh` (`kwh >= 1` integer):
   charging records.

## Feasibility

Every bus charged to exactly `E_i`; charging only inside `[a_i, d_i)`; cable
reach respected at the charging hour; per-row per-hour total `<= P`; per-hour
grid total `<= cap[h]`; both sequences valid permutations with the required
orderings. Any violation scores 0.

## Objective (minimize) and scoring

```
F = sum_h prc[h] * (energy delivered in hour h)  +  SH * (total shunt moves)
```

The checker builds a reference plan (balanced arrival-order parking,
earliest-possible charging) with cost `B` and reports
`Ratio = min(1, 0.1 * B / F)`: the reference scores 0.1; beating it by 10x
caps at 1.0. Lower `F` is better.

## Example (illustrative, not a generated case)

`n=3, R=2, W=2, T=6, P=4, SH=10`; `prc = [8,3,2,2,5,6]`;
`cap = [8,8,4,4,8,8]`; buses: `0:(a=0,d=5,E=4)`, `1:(0,4,4)`, `2:(1,3,2)`.

Plan: park `0->row0, 1->row1, 2->row0`; exit `2 1 0` (bus 2 has one bus
ahead: 1 shunt); charge `2 kwh` to bus 0 at hours 1 and 3, `2 kwh` to bus 1
at hours 2 and 3, `2 kwh` to bus 2 at hour 2. All caps, rates, reach limits
hold; energy cost `2*3 + 4*2 + 4*2 = 22`, shunt cost `10`, so `F = 32`.
The reference plan charges everything at hours 0-1 (`B = 80`), giving
`Ratio = 0.1 * 80 / 32 = 0.25`.

## Hints

Parking by arrival convenience or by departure buckets both ignore that a row
replays its fill order at exit. If arrival and departure orders disagree, any
such layout manufactures blocking pairs — and every shunt is pure loss.
