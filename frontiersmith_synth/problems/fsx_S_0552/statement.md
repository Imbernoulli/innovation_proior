# Resonant Folds: Binding an H/P Chain Without Self-Trapping

## Problem
You are handed a linear chain of `L` residues. Each residue is either **H** (a
binding residue with a positive charge) or **P** (inert filler). You must fold
the chain onto the 2D square lattice as a **self-avoiding walk**: residue `0`
sits at the origin `(0,0)`, and each subsequent residue is placed on a lattice
cell adjacent to the previous one. Placement is one residue at a time and
**permanent** — no cell may ever be occupied twice.

Two residues **bind** when they land on adjacent lattice cells (unit apart) but
are **non-consecutive** in the chain (their chain indices differ by more than 1).
A bond between residues `i` and `j` is worth `w[i] * w[j]`, where `w` is the
charge vector (`w[i] = 0` exactly at P residues). Your score grows with the total
binding value; maximize it.

Note the lattice's checkerboard structure: every unit step flips a cell's colour,
so residue `i`'s colour is fixed by the parity of `i`. Adjacent cells always have
opposite colours, hence **only residues of opposite parity (odd chain-distance)
can ever bind.** No fold changes that; it only decides which admissible pairs
actually touch. Folding too eagerly into a compact pocket strands later residues
on the surface where they can no longer reach a partner.

## Input (stdin)
```
L
s          # H/P string, length L
w[0] w[1] ... w[L-1]   # L integers, w[i] > 0 iff s[i] == 'H', else 0
```
`3 <= L <= 1200`, `1 <= w[i] <= 3` at H residues.

## Output (stdout)
`L-1` integer move codes (whitespace-separated) placing residues `1..L-1`, where
`0`: x+1, `1`: x-1, `2`: y+1, `3`: y-1. The walk must be self-avoiding.

## Feasibility
Rejected (score 0) unless the output has exactly `L-1` move codes, each an
integer in `{0,1,2,3}`, and all `L` visited cells are distinct.

## Objective
Maximize `F = sum over non-consecutive lattice-adjacent pairs {i,j} of w[i]*w[j]`.

## Scoring
Let `B` be the binding value of the reference **hairpin** fold (a 2-row
serpentine of width `ceil(L/2)`), computed by the grader. Your score is
```
Ratio = min(1000, 100 * F / max(1e-9, B)) / 1000
```
so the hairpin baseline scores ≈ 0.1 and a 10×-better fold caps at 1.0. There is
no known optimum: the fold that maximizes `F` depends on where the charged
residues resonate, and the best fold width is generally not a round number.

## Example
`L=6`, `s=HPPHPH`, `w=1 0 0 2 0 3`. The straight fold `0 0 0 0 0` places all
residues in a line: no non-consecutive adjacencies, `F=0`, `Ratio=0.0`. A fold
`0 0 2 1 1` bends the chain so residues `0` and `5` land on adjacent cells; if a
bond of value `w[0]*w[5]=3` forms, `F=3`. The grader compares your `F` to its
hairpin baseline `B` and prints `Ratio`.
