# Overhang-Supported Additive Print

## Problem

A 3D printer builds a solid **target shape** out of unit voxels on an `Lx x Ly x Lz` integer
grid, one horizontal layer at a time, strictly bottom-up (`z = 0, 1, 2, ...`). Once a layer
is printed it is final: nothing about it, or the overhangs it left unsupported, can ever be
revisited or repaired later.

Unsupported overhangs sag and ruin the print, so the printer may also deposit **support
material**: temporary voxels that are not part of the target shape but give a foothold to
whatever is printed above them, and are later removed. Support material costs money; some
of it, however, ends up permanently sealed inside the finished part and can never be
recovered.

Output a complete build plan: for every voxel the printer deposits, its coordinates and
whether it is **part** material (`P`) or **support** material (`S`).

### Feasibility rules

1. **Part voxels must equal the target exactly.** The set of `P` voxels in your plan must
   be exactly the given target voxel set — no target voxel may be missing, and no extra `P`
   voxel may be printed.
2. **Layer-monotone overhang support.** Consider the set of all voxels you print (`P` or
   `S` together) at layer `z`. Every voxel of layer `z > 0` needs a foothold from layer
   `z - 1`: at least one of the 5 cells `(x,y,z-1)`, `(x-1,y,z-1)`, `(x+1,y,z-1)`,
   `(x,y-1,z-1)`, `(x,y+1,z-1)` must also be printed (`P` or `S`) in your plan. (This models
   the familiar ~45-degree self-supporting overhang: a voxel may rest on the cell directly
   below it, or "lean" one cell sideways onto its neighbor's footprint.) Layer `z = 0` always
   rests safely on the build plate.

Any violation of either rule scores `0`.

### Objective (minimize)

Each support (`S`) voxel costs **1**, unless it is **trapped**: a support voxel at
`(x,y,z)` is trapped if the target shape contains a part voxel at the same `(x,y)` column
at *any* height above it (`z' > z`). A trapped support voxel can never be withdrawn once
the part above it is printed, and costs **3** instead of 1. Minimize the total support
cost. (Part voxels are free — you must print all of them regardless.)

Because a layer, once built, cannot be revised, support decisions must be made for the
*future*: a support voxel placed low in the print is only useful for the part voxels it
will (or will not) end up enabling several layers later — and whether it becomes trapped is
entirely determined by what gets printed above it later on.

## Input (stdin)

```
Lx Ly Lz
T
x_1 y_1 z_1
...
x_T y_T z_T
```
`Lx, Ly, Lz` are the grid dimensions; `T` is the number of target (part) voxels, each given
by 0-indexed integer coordinates with `0 <= x < Lx`, `0 <= y < Ly`, `0 <= z < Lz`.

## Output (stdout)

```
K
x_1 y_1 z_1 t_1
...
x_K y_K z_K t_K
```
`K` is the total number of voxels you print; each of the next `K` lines gives a voxel's
coordinates and its type `t_i` in `{P, S}`. Coordinates must be in-grid integers, and no
coordinate may repeat.

## Scoring

Let `F` be your total support cost. The checker also builds its own simple, always-feasible
reference: it fills the entire axis-aligned bounding box of the target (in `x` and `y`, from
`z=0` up to the target's tallest voxel) with support wherever a target voxel isn't already
present, and calls its cost `B`. Your score is `min(1000, 100*B/max(1e-9,F)) / 1000`:
matching the reference scores `0.1`, and doing 10x better caps the score at `1.0`.

## Example (illustrative, not to scale)

Target: `(0,0,0)` and `(2,0,1)`, both at `z<=1`. Printing only `{(0,0,0):P,(2,0,1):P}` is
infeasible: `(2,0,1)` has no filled neighbor at `z=0`. Adding `(1,0,0):S` fixes it — it is a
neighbor of `(2,0,1)`, and `z=0` always rests on the plate. Cost `= 1` (column `(1,0)` has
no target voxel above it, so it is not trapped).

## Constraints

`1 <= Lx, Ly, Lz <= 100`; `1 <= T <= Lx*Ly*Lz`. Time limit 5s, memory 512MB.
