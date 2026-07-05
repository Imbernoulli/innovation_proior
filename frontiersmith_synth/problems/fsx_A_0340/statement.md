# Server Hall Cooling: Thermal-Aware Job Placement

## Story

A data hall is an `N x N` grid of rack slots. Each slot `(r, c)` has an integer
**cooling capacity** `cool[r][c] >= 0` -- how much dissipated heat that slot can
absorb. Cold-aisle airflow is strong right under the CRAC vents and weak in the dead
zones between them, so `cool` is high in a few pockets and low elsewhere.

You must schedule `J` compute **jobs** onto the racks. Job `j` runs hot with an
integer heat **load** `loads[j]`, and it must occupy one rack slot; no two jobs may
share a slot, so you choose `J` **distinct** slots.

Heat recirculates locally. A job of load `w` placed at slot `(r, c)` deposits heat
onto its `3 x 3` neighborhood through a fixed thermal **kernel**:

```
kernel = [[1, 2, 1],
          [2, 4, 2],
          [1, 2, 1]]
```

That is, slot `(r+dr, c+dc)` (clipped to the hall) receives `w * kernel[dr+1][dc+1]`.
The heat arriving at any slot is the **sum** of deposits from every job whose `3 x 3`
footprint reaches it. A slot runs hot when arriving heat exceeds its cooling:

```
over(a, b) = max(0, deposit(a, b) - cool[a][b])
```

Hotspots stress hardware super-linearly, so you **minimize the total squared
over-temperature**:

```
penalty = sum over all slots (a, b) of over(a, b) ** 2
```

Two tensions pull against each other. Packing jobs close together stacks their `3 x 3`
footprints -- a shared slot then sees `4*w1 + 2*w2 + ...` and overheats badly. And
ignoring the cooling map wastes the strong-vent pockets. A good schedule spreads the
**hottest** jobs apart **and** parks them on **high-cooling** slots. Total deposited
heat exceeds total hall cooling on every instance, so some over-temperature is
unavoidable -- you are shaving a penalty that never reaches zero.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "hall301",
  "n": 6,
  "j": 14,
  "loads": [ ... J ints ... ],
  "cool": [[ ... N ints ... ], ... N rows ... ],
  "kernel": [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
}
```

- `n` (int): the hall is `n x n`.
- `j` (int): number of jobs (`j <= n*n`).
- `loads` (list of `j` ints `>= 0`): heat load of each job, in listed order.
- `cool` (list of `n` lists of `n` ints `>= 0`): cooling capacity per slot.
- `kernel` (`3 x 3` list of ints): the fixed thermal footprint weights above.

## Output (one JSON object on stdout)

```json
{"place": [[r0, c0], [r1, c1], ...]}
```

- Exactly `J` entries; `place[j] = [r, c]` is the slot assigned to job `j`
  (`loads[j]`).
- Each entry is a pair of integers `[r, c]` with `0 <= r < n` and `0 <= c < n`.
- The `J` slots must be **pairwise distinct**.

Any of the following makes the instance score `0.0`: wrong number of entries, a
coordinate out of range, duplicate slots, non-integer coordinates, a crash, a
timeout, or output that is not the JSON object above.

## Objective and scoring (deterministic)

For each instance the evaluator computes:

- `P_base` = penalty of the overlap-blind, cooling-blind reference schedule that puts
  job `j` at the row-major slot `(j // N, j % N)`. Jobs pack into a corner, their
  footprints stack, and the vents go unused. This is the weak baseline.
- `P_cand` = penalty of your schedule.

and normalizes with an affine anchor (weak baseline -> `0.1`, the unreachable
perfect-hall penalty `0` -> `1.0`):

```
r = clamp( 0.1 + 0.9 * (P_base - P_cand) / max(1e-9, P_base), 0, 1 )
```

Reproducing the row-major reference scores about `0.1`; doing worse scores below
`0.1`; cutting the squared over-temperature scores higher. Because total deposited
heat exceeds total cooling, `P_cand` can never reach `0`, so even excellent schedules
stay well below `1.0` -- there is always headroom. Your final score is the mean of
`r` over all instances (a mix of hall sizes, job counts, and cooling maps, including
harder held-out halls with tighter cooling and hotter jobs).

## Notes

- Scoring depends only on your emitted `place`; it never measures wall-clock time.
  Treat the per-instance limit as an operation budget for search-based methods
  (hottest-first greedy, relocation / swap local search, annealing, restarts).
- Your program is run in an isolated subprocess and sees only the public instance
  above.
