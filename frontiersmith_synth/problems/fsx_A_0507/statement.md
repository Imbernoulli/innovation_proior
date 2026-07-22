# Lava-Tunnel Vault: Cooling-Corridor Placement

## Problem
A server vault is carved into cooling rock as an `H x W` grid of cells. The entire outer
**border** is solid **wall** (a perfect insulator — no heat crosses it) **except** for a few
**vent** cells cut into the border, which are held at the cold ambient temperature `T = 0`.
Every **interior** cell (`1 <= r <= H-2`, `1 <= c <= W-2`) is open rock. Some interior cells
hold heat **sources** that inject a fixed integer wattage.

Each cell has a thermal **conductivity**: `1` for ordinary rock. You are given a budget `K`
and may **upgrade** up to `K` interior cells to high conductivity `KHI` (installing cooling
plating). The conductance of the edge between two adjacent open cells is the **harmonic mean**
of their conductivities, `2*ka*kb/(ka+kb)` — so a lone upgraded cell surrounded by rock is
almost useless (its edges are `~2`), while a *contiguous run* of upgraded cells conducts at
`~KHI` internally.

The vault reaches **steady state**: for every interior cell, the net heat flow is zero,
```
sum over open neighbours v of  g(u,v) * (T_v - T_u)  +  P_u  =  0,
```
where `g(u,v)` is the edge conductance, `P_u` is the injected wattage at `u` (0 if none), vent
neighbours contribute with `T = 0`, and wall neighbours carry no flow. This linear system has a
unique positive solution. Let `Tmax` be the largest interior temperature.

You want the vault to run **cool**: minimize `Tmax`.

## Input (stdin)
```
H W KHI K
S
<S lines: r c p>      # a source of wattage p at interior cell (r,c)
NV
<NV lines: r c>       # a vent cell (r,c) on the border, held at T=0
```

## Output (stdout)
```
M
<M lines: r c>        # upgrade interior cell (r,c) to conductivity KHI
```
Print the number of upgraded cells `M`, then their coordinates.

## Feasibility
An output is valid iff **all** hold:
- `0 <= M <= K`, and exactly `M` coordinate pairs follow;
- every `(r,c)` is an **open interior** cell (`1 <= r <= H-2`, `1 <= c <= W-2`), never a vent;
- the upgraded cells are pairwise distinct;
- all tokens are finite integers.
Any violation scores `Ratio: 0.0`.

## Objective
Minimize `F = Tmax`, the maximum steady-state interior temperature under your upgrades.

## Scoring
The checker builds its own baseline `B` = the **do-nothing** maximum temperature (no cells
upgraded), then, with the minimization convention,
```
sc    = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Upgrading nothing gives `F = B` and `Ratio = 0.1`; cooling the hottest spot `10x` caps at `1.0`.
The exact wattages and vent positions live in the input — you must read and exploit them.

## Constraints
- `24 <= H <= 50`, `13 <= W <= 19`, `KHI = 24`.
- `1 <= S <= 4` sources; `1 <= NV <= 3` vents; budget `K` up to `H` cells.
- Steady state is solved by exact sparse linear algebra (deterministic).
- Time limit 5s, memory 512m.

## Example
Take a tall vault with one deep central source and vents only at the top and bottom border
midpoints. Blanketing a disk of plating around the source lowers the gradient right next to it,
but the heat must still crawl the full length of ordinary rock to a vent, so `Tmax` barely
drops. Installing a single **contiguous corridor** of plating from the source straight to the
nearest vent instead removes the dominant series resistance on that path and drops `Tmax`
several-fold — the score is set by the worst bottleneck on the source-to-vent path, so budget
belongs in corridors, not blankets.
```
```
