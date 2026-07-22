# Delta Foreman: Scheduling Mud Floods to Out-Build a Sinking Marsh

A coastal marsh is drowning. You command a diversion that can release engineered
mud floods into it over **T** discrete steps. You are given a fixed sediment
**budget** `Mtot`. Your job is to choose how much mass to release at each step so
that, after the last release, as many grid cells as possible sit **at or above the
waterline** (elevation `0`). The catch: deposits are not merely score — they are
terrain, and terrain steers where the *next* flood goes. Your schedule is a program
that reprograms the router it flows through.

## The world

An `H x W` grid holds bedrock elevations `z[r][c]` (all below `0`: a sunk marsh
that deepens offshore as the column index `c` grows). The **inlet** is cell
`(sr, 0)`; column `W-1` is the open **deep-water edge**. Constants `d0`, `kslope`,
`beta` are given in the input.

## Deterministic deposition simulation

Terrain starts at the bedrock. For each step `t = 0..T-1`:

1. **Subsidence (the sinking marsh).** If `t > 0`, every unit of *already-deposited*
   sediment relaxes toward bedrock: for each cell, `z = bed + (z - bed)*(1 - beta)`.
   (Bedrock never subsides; only your mud does. Later deposits subside fewer times.)
2. **Release & routing.** Inject mass `m_t` at the inlet, then route it downhill
   over the **current** terrain by processing cells in order of **descending
   elevation** (ties by cell index `r*W+c`). At a cell holding in-transit mass `q`:
   - Let `smax` be the largest positive drop `z[cell]-z[nb]` to a 4-neighbour.
   - A fraction `dep = d0 + (1 - d0)/(1 + kslope*smax)` **settles here** (raising
     this cell's elevation by that mass); low slope ⇒ more settles.
   - The remainder `q*(1-dep)` is split among the strictly-lower 4-neighbours in
     proportion to their drops and flows onward.
   - Special cases: an interior cell with **no lower neighbour** (a pit) keeps all
     of `q`; at the **sea edge** (`c = W-1`) only the settled fraction stays and the
     rest is **lost to deep water**.
3. Deposits from step `t` are added to the terrain and therefore reroute every
   later release.

After step `T-1` (no trailing subsidence), your **score count** `F` is the number of
cells with final `z >= 0`.

Because low slope means more settling, mud you place early can flatten a patch and
turn it into a trap for later mud — but the marsh keeps sinking under it. Place mud
too late and, over un-shaped terrain, the flood runs straight down the deepest line
to the sea edge and is lost.

## Input (stdin)

```
H W T
Mtot d0 kslope beta sr
z[0][0] ... z[0][W-1]
...
z[H-1][0] ... z[H-1][W-1]
```

## Output (stdout)

Exactly `T` real numbers `m_0 ... m_{T-1}` (whitespace-separated): the mass released
at each step.

## Feasibility

Each `m_t` must be finite and `>= 0`, and `sum(m_t) <= Mtot` (a tiny tolerance is
allowed). Any other count, a negative/`nan`/`inf` value, or an over-budget total
scores `0`.

## Objective & scoring

Maximize `F`. Let `B` be the count produced by the reference schedule that dumps the
**entire budget in the final release** (the simulator computes it). Your ratio is

```
Ratio = min(1.0, 0.1 * F / max(1, B))
```

so the reference lands near `0.1` and roughly ten-times-better would cap at `1.0`.
The optimum is unknown; many schedules are viable.

## Example

`T = 4`, `Mtot = 40`. The schedule `0 0 0 40` dumps everything last: over the raw
channel it channelises to the sea edge — that is the reference `B`. A schedule like
`3 5 9 23` instead spends small early pulses to aggrade a near-shore platform, then
lands the bulk on that engineered terrain; it can retain far more land and score
well above `0.1`. (Numbers are illustrative, not tuned.)
