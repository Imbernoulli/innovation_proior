# The River God's Delta

You are a river god asked to grow a fertile farming delta. A spring pours water
down a sloping land; you may resculpt the ground with a limited number of
dig/fill edits. Then the water runs and carries silt. You are graded on how much
**silt settles inside a chosen target zone**.

## The land

An `N x N` integer heightfield `h[y][x]`. Row `0` is the high country; the bottom
row is the **sea** (height `0`). One cell is the **spring** `(sy, sx)`. A
rectangle `[zy0..zy1] x [zx0..zx1]` is the **target zone**.

## Your edits (stdout)

Print `K`, then `K` lines `y x d`: add integer `d` to cell `(y, x)` (dig if
`d<0`, fill if `d>0`). Rules, all strictly enforced (any violation scores `0`):
- `0 <= K <= N*N`; each `(y,x)` inside the grid; each `d` a plain integer;
- total spent `sum |d| <= V`;
- every final height stays in `[0, HMAX]`.

## The erosion simulation (how you are scored)

After your edits the field is **fixed**, and the god runs `T` identical water
**pulses** over it (a passive sediment tracer). Each pulse starts at the spring
carrying load `L = 0` and repeatedly:
1. If the current cell height `<= 0` (the sea): the water and its remaining load
   leave the map. Pulse ends.
2. Otherwise look at the 4 neighbours and step to the **lowest** one (ties broken
   in the fixed order up, right, down, left). If no neighbour is strictly lower
   (a pit), the water pools: it **drops all of `L` here** and the pulse ends.
3. Let `slope = h[cur] - h[next] > 0`. The transport **capacity** is
   `cap = CAP_K * slope`.
   - If `cap > L` (spare capacity) the water **entrains**: `L += min(cap - L,
     EROSION)` (bounded so it never draws a cell below sea+1).
   - If `cap < L` (over capacity) the water **deposits**: it drops
     `min(L - cap, DEPOSIT)` here, subtracting from `L`.
   Then it moves to `next`.

Only sediment dropped on a zone cell counts. `CAP_K, EROSION, DEPOSIT, T, HMAX`
are given on the first input line.

## Objective

Maximise `F` = total sediment deposited inside the target zone over all `T`
pulses. The checker normalises against its own routing baseline `B`:
`Ratio = min(1000, 100 * F / B) / 1000` (a do-the-obvious routing scores about
`0.1`; ten times better caps at `1.0`).

**Key idea.** Capacity tracks *slope*, so deposition happens where the slope
**drops**. Silt only settles if the water arrives *already loaded* and then
*decelerates* on target. A straight, uniformly steep canal keeps `cap >= L`
everywhere and rushes the silt out to sea; a shallow ditch never loads up. What
pays is the **derivative of the terrain**: steepen the approach to entrain a big
load, then flatten exactly at the zone to make it settle.

## Input (stdin)

```
N V T CAP_K EROSION DEPOSIT HMAX SEA
sy sx
zy0 zx0 zy1 zx1
h[0][0] ... h[0][N-1]
...            (N rows)
```

## Constraints

`24 <= N <= 42`, `T = 40`, `HMAX = 250`, `SEA = 0`. Time limit 5 s, memory
512 MB. On some tests the zone is offset from the spring column, so water
flowing straight down bypasses it entirely — you must route as well as stall.

## Example (illustration)

With `L = 10` arriving at a cell whose downhill `slope = 1` (`cap = 1`) and
`DEPOSIT = 6`, the water drops `min(10-1, 6) = 6` there, keeping `4`; the next
gentle cell drops more. Over a steep-then-flat approach the same water would have
arrived carrying far more than `10`. (Numbers illustrative only.)
