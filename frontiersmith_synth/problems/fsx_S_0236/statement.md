# Quantum Lab Wiring: Twin-Cryostat Module Split

## Problem

Your lab is commissioning a superconducting quantum processor. The processor is delivered
as `n` **modules** (chip tiles) that must be mounted across **two cryostats**, cryostat `0`
and cryostat `1`. To keep the two dilution refrigerators thermally symmetric, exactly half of
the modules go into each cryostat (`n` is always even, so each cryostat holds exactly `n/2`
modules).

Pairs of modules are joined by physical **wires**. Every wire has a *type* and a *value*:

- A **coupling wire** (type `0`) carries a microwave qubit-coupling line. Coupling wires are
  happiest when their two modules sit in **different** cryostats, because the physical
  separation suppresses crosstalk. A coupling wire earns its value only if it is *cut*
  (its endpoints land in different cryostats).
- A **control-bus wire** (type `1`) carries a shared DC bias / flux-control bus. A control bus
  must stay short, so it is happiest when its two modules sit in the **same** cryostat. A
  control-bus wire earns its value only if it is *uncut* (its endpoints land in the same
  cryostat).

You choose which cryostat each module goes into (respecting the exact-half rule) so as to earn
as much total wire value as possible. Choosing the best split is NP-hard, so any balanced
assignment is accepted and graded by the value it earns.

## Input

- Line 1: two integers `n m` — the number of modules and the number of wires.
- Next `m` lines: four integers `u v t w`:
  - `u v` — a wire between module `u` and module `v` (`1 <= u,v <= n`, `u != v`),
  - `t` — the wire type (`0` = coupling wire, `1` = control-bus wire),
  - `w` — the wire value (`1 <= w <= 100`).

Modules are numbered `1..n`. Wires are undirected; there are no self-loops and no duplicate
module pairs. `n` is even.

## Output

Print `n` integers `c_1 c_2 ... c_n` (whitespace-separated), where `c_i` is the cryostat
(`0` or `1`) that module `i` is mounted in.

## Feasibility

- Each `c_i` must be `0` or `1`.
- Exactly `n/2` of the modules must be assigned to cryostat `1` (hence `n/2` to cryostat `0`).

Any output violating these rules scores `0`.

## Objective

Maximize the earned wire value

```
F = sum of w over all coupling wires (t=0) with c_u != c_v
  + sum of w over all control-bus wires (t=1) with c_u == c_v.
```

A coupling wire that is *not* cut earns nothing; a control-bus wire that *is* cut earns nothing.

## Scoring

Let `B` be the earned value of the **reference split** in which modules `1..n/2` go to
cryostat `0` and modules `n/2+1..n` go to cryostat `1` (the checker computes `B` itself; the
generator guarantees `B > 0`). With `F` the earned value of your feasible split, the raw score
is

```
score = min(1000, 100 * F / B)
```

and the reported ratio is `score / 1000`. So the reference split earns ratio `0.1`, and earning
ten times the reference value caps the ratio at `1.0`. Higher is better.

## Constraints

- `2 <= n <= 600`, `n` even.
- `1 <= m <= n*(n-1)/2`.
- `t` in `{0,1}`, `1 <= w <= 100`.
- Time limit: 5 s. Memory limit: 512 MB.

## Example

Input:

```
4 4
1 2 0 5
2 3 1 9
3 4 0 4
4 1 1 7
```

The reference split is `c = [0,0,1,1]` (modules 1,2 in cryostat 0; modules 3,4 in cryostat 1).
- Wire `1-2` (coupling): same cryostat, not cut, earns `0`.
- Wire `2-3` (bus): different cryostats, cut, earns `0`.
- Wire `3-4` (coupling): different cryostats, cut, earns `4`.
- Wire `4-1` (bus): different cryostats, cut, earns `0`.

So `B = 4`.

Now consider the balanced split `c = [0,1,1,0]`:
- Wire `1-2` (coupling): different cryostats, cut, earns `5`.
- Wire `2-3` (bus): same cryostat, uncut, earns `9`.
- Wire `3-4` (coupling): different cryostats, cut, earns `4`.
- Wire `4-1` (bus): same cryostat, uncut, earns `7`.

So `F = 25`, and `score = min(1000, 100 * 25 / 4) = 625`, i.e. ratio `0.625`.
