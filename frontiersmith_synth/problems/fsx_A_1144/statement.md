# Build the Sculpture in Fewest Moves

## Problem
You are given a target **sculpture**: a finite set of `V` unit voxels at
integer coordinates `(x,y,z)` in 3-space. You must reconstruct it *exactly*
with a **straight-line program over four growth macros**. Each line of your
program creates one new *assembly* (a set of voxels), numbered `0,1,2,...`
in the order created:

```
U                creates the single voxel {(0,0,0)}
T i dx dy dz     creates assembly i translated by (dx,dy,dz)
R i a            creates assembly i reflected through the origin along
                 axis a in {0,1,2} (negates that coordinate)
M i j            creates the UNION of assembly i and assembly j
```

`i` and `j` must refer to an assembly created on an earlier line (no forward
references). The **final assembly** is the one created by your program's
last line. Any assembly, once built, may be reused as the source of any
number of later `T`, `R`, or `M` lines — reuse is free, it just costs the
one line that referenced it.

## Input
```
V
x_1 y_1 z_1
...
x_V y_V z_V
```
The target voxel set, one voxel per line.

## Output
Your straight-line program, one macro per line, in the schema above.

## Feasibility
The final assembly must equal the target voxel set **exactly** (same
voxels, no extra, none missing). Any malformed line, out-of-range index,
non-integer/`nan`/`inf` token, or a mismatched final assembly scores `0.0`.
Coordinates/deltas are bounded by `1e7`; at most 300000 lines; the checker
aborts (score `0.0`) if the total voxel-instances touched while replaying
your program exceeds a generous work budget — you cannot abuse the union
trick below to blow up the checker's memory for a fake score.

## Objective
Minimise `F`, the number of lines in your program (fewer moves is better).

## Scoring
Let `B_hi = 2V - 1`, the cost of the naive construction (one voxel placed
at a time, combined pairwise). Let `B_lo = ceil(log2 V) + 1`: since `U`
starts a size-1 assembly and every subsequent op at best *doubles* the size
of the largest assembly built so far (a `T`/`R` preserves size, an `M` at
best sums two equal maximal assemblies), no correct program can ever use
fewer than `B_lo` lines — it is a safe, unreachable floor. Your score:

```
score = clip( 0.1 + 0.7 * (ln B_hi - ln F) / (ln B_hi - ln B_lo),  0, 1 )
```

`F = B_hi` scores `0.10`; reaching `B_lo` would score `0.80`; there is
always room above whatever you achieve.

## The insight
Placing one voxel at a time is `O(V)`. Noticing that a straight strip of `k`
voxels repeats and caching *that one strip* (an "obvious" optimization) only
helps where such simple, local repeats exist. A sculpture built by
recursively duplicating and transporting a smaller sub-assembly — the same
sub-shape appearing, translated (or mirrored) to many places, at many
scales — is a **shape addition chain**: canonicalize sub-assemblies bottom
up (any two regions with the same relative occupancy, up to translation or
a single mirror, are the *same* assembly) and reuse across the *whole*
recursive structure, not just within one row. A target built from `L`
levels of `m` repeated sub-copies needs only `O(L*m)` lines to reconstruct
exactly `m^L` voxels — logarithmic in `V`, not linear.

## Constraints
`1 <= V <= 25000`. Time limit 5 s, memory 512 MB. Each input file `<= 5 MB`.

## Example (scoring only, illustrative shape — not the intended solution)
A target with `V = 8` (a 2x2x2 cube of voxels at `x,y,z in {0,1}`) has
`B_hi = 15`. One correct program: `U` (asm 0, `{(0,0,0)}`); `T 0 1 0 0`
(asm 1); `M 0 1` (asm 2, a length-2 edge); `T 2 0 1 0` (asm 3); `M 2 3`
(asm 4, a 2x2 face); `T 4 0 0 1` (asm 5); `M 4 5` (asm 6, the full cube) —
`F = 7` lines, doubling three times instead of placing 8 voxels one by one.
With `B_lo = ceil(log2 8)+1 = 4`, this scores `0.1 + 0.7*(ln15-ln7)/(ln15-ln4) = 0.504`.
