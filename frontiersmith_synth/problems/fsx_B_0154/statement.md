# Debris-Tug Rendezvous Routing on a Constellation Coupling Map

## Problem
An orbital debris-cleanup constellation has `n` physical **service slots** (chaser satellites),
wired by a fixed set of inter-satellite **transfer links** — this is your hardware *coupling map*.
A joint capture maneuver between two logical **debris tugs** can only be executed when the two tugs
currently occupy two slots that are directly linked. Tugs may be relocated only by **SWAP**ing the
occupants of two directly-linked slots.

You are given a batch of `k` required maneuvers (a QAOA-style commuting interaction set: each maneuver
is a `ZZ` rendezvous between two logical tugs; the maneuvers all commute, so they may be executed in
any order). Starting from a fixed initial placement of tugs on slots, emit a program of SWAP moves and
maneuver executions that performs **every required maneuver exactly once**, each on a linked pair of
slots. Minimize the number of inserted SWAP moves.

This is the standard quantum-circuit *routing / transpilation* task: make the logical circuit
executable on limited hardware connectivity while adding as few SWAP gates as possible, with the routed
program verified for exact functional equivalence to the target circuit.

## Input (stdin)
```
n m k
u v            (m lines: undirected coupling-map link between slots u and v)
p0 p1 ... p(n-1)   (init[p] = logical tug initially on physical slot p; a permutation of 0..n-1)
a b            (k lines: required maneuver between logical tugs a and b)
```
All indices are 0-based. Slots and tugs are both numbered `0..n-1`.

## Output (stdout)
A program, one instruction per line (blank lines ignored, keywords case-insensitive):
```
SWAP p q       swap the tugs currently on physical slots p and q (p,q must be linked)
GATE g         execute required maneuver g (its two tugs must currently sit on linked slots)
```

## Feasibility
- Every `SWAP p q` must use an existing coupling-map link, with `p != q`.
- Every `GATE g` must reference a valid maneuver index whose two logical tugs currently occupy two
  linked slots.
- Each of the `k` maneuvers must be executed **exactly once** (executing one twice, or omitting one,
  breaks functional equivalence). Any violation scores 0.

## Objective
Minimize `F` = number of `SWAP` instructions emitted (fewer inserted SWAPs is better).

## Scoring
Minimization against an internal baseline `B` (the checker's in-order shortest-path router):
`Ratio = min(1000, 100 * B / F) / 1000`. Reproducing the baseline scores ≈ 0.1; halving its SWAP
count scores ≈ 0.2; a 10×-better routing caps at 1.0. Deterministic and reproducible.

## Constraints
- `n ≤ 20`, `k ≤ 22` on the graded ladder; the coupling map is a connected grid-like lattice.
- Program length is bounded generously (≤ 200000 instructions).

## Example
Coupling path `0-1-2`, initial placement identity (`init = [0,1,2]`), one maneuver between tugs `0`
and `2`. They sit on slots 0 and 2, not linked. `SWAP 0 1` moves tug 0 onto slot 1 (linked to slot 2
holding tug 2), then `GATE 0` executes the maneuver: `F = 1` SWAP. The in-order baseline also needs 1
SWAP here, so `Ratio = 100*1/1 / 1000 = 0.1`.
