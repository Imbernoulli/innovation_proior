# Scaffold Hop: Same Effect, Different Molecule

## Problem

A drug-discovery team has a **known active** molecule and wants a **novel**
molecule with the same biological activity but different enough structure to
be a real scaffold hop. A molecule here is a *chain* of fragments drawn from
a fixed library. Fragment `j` (0-indexed, `j = 0..L-1`) contributes exactly
one atom, placed at global coordinate `(j*STEP + dx, dy, dz)`, where
`(dx, dy, dz)` are properties of the fragment id chosen for position `j`, and
`STEP` is a fixed constant from the input. Each atom has a feature type: `D`
(donor), `A` (acceptor), `R` (aromatic), `H` (hydrophobic), or `X` (no
pharmacophoric feature).

The **known active** is itself such a chain (given in the input, `L_ref`
fragment ids). Its biological activity is captured by `K` **pharmacophore
anchors**: 3D points, each with a required feature type and a tolerance. The
known active satisfies all `K` anchors by construction.

## Input (stdin)

```
M L_max STEP BUDGET
M lines: id cost type dx dy dz        (the fragment library, ids 0..M-1)
L_ref
L_ref ids                             (the known active, as a fragment-id chain)
K
K lines: x y z type tol               (pharmacophore anchors)
```

## Output (stdout)

Your candidate molecule: a line with `L` (`1 <= L <= L_max`), then a line
with `L` fragment ids (each in `[0, M-1]`) -- your own chain, fragment `j` at
global position `(j*STEP+dx, dy, dz)`.

## Feasibility

- `1 <= L <= L_max`; every id is a valid library index; the second line has
  exactly `L` tokens.
- Total synthetic cost (sum of the chosen ids' `cost`) must be `<= BUDGET`.

Any violation scores 0.

## Objective

Three multiplied factors:

1. **Pharmacophore match** `P` = (# of the `K` anchors matched by some atom of
   your molecule -- same feature type, Euclidean distance `<= tol` of the
   anchor's point) `/ K`.
2. **Novelty** `N = 1 - overlap / max(L, L_ref)`, where `overlap` is the
   multiset intersection (Counter-min) between your fragment-id sequence and
   the known active's. Copying the known active exactly gives `N = 0`.
3. **Synthetic-accessibility efficiency** `= min(1, cost(known active) /
   cost(yours))` -- 1.0 if you are at least as cheap as the known active,
   otherwise the ratio (no bonus for being far cheaper).

The score-relevant quantity is `F = P * N * efficiency`. The checker also
builds its own reference construction (the known active plus a little
padding with the library's cheapest fragment) as a baseline `B`, and reports
`Ratio = min(1, F / (10*B))`.

## Strategy notes

Maximizing `P` alone by copying the known active drives `N` to 0 -- no
novelty, no score. Chasing pure novelty (fragments never used by the known
active) without checking an anchor's *type* and *chain position* tanks `P`.
The library is deliberately redundant: for every pharmacophoric type,
several *different* fragments (different id, different cost) place their
feature atom at the *same* offset, so an anchor's type-and-position
requirement can be honored without reusing the known active's exact fragment
identity there -- this decouples activity preservation from novelty.
Watch the near-miss fragments: some fragments carry the right type but at an
offset just outside `tol`, and are not actually usable for any anchor.

## Example (worked, illustrative shape only -- not the hidden instance)

Suppose `L_ref=3`, chain `[0,1,0]`, fragment `0 = (cost 4, X, 0,0,0)`,
fragment `1 = (cost 9, D, 2,0,0)`. One anchor: `(x=1002, y=0, z=0, type=D,
tol=0.5)` (position 1 contributes `1*STEP+2=1002`). If the library also
has fragment `6 = (cost 6, D, 2,0,0)` that is absent from the reference,
output `3\n0 6 0` still satisfies the anchor (a `D` atom lands at global
`(1002,0,0)`) while differing from the known active at that position and
costing less -- exactly the kind of substitution that raises `N` and
`efficiency` without lowering `P`.

## Constraints

`M = 15`, `4 <= L_ref <= 20`, `L_max = L_ref + 8`, `2 <= K <= L_ref`, all
costs positive integers `<= 20`, `tol = 0.5`. Time limit 5s, memory 512MB.
