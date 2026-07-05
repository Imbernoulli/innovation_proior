# Conveyor Handoff Routing: Minimal-SWAP Transpilation on a Warehouse Floor

## Problem

A warehouse floor is a grid of **docking bays** connected by **conveyor tracks**.
Each bay holds exactly one **robot**; there are `V` bays (ids `0..V-1`) and `V`
robots (ids `0..V-1`). Robots can only move by riding a conveyor track: a single
**SWAP** exchanges the two robots sitting on the two ends of one track.

You are given an ordered list of `m` **handoff tasks**. Task `k` names two robots
`(a, b)` that must transfer a parcel; a transfer can only happen when the two
robots occupy **bays joined by a track** (i.e. adjacent bays). The tasks must be
executed **in the given order**.

This is the quantum qubit-routing / transpilation problem in disguise: the floor
is a hardware **coupling map**, robots are qubits, handoffs are the two-qubit
gates of a circuit (e.g. a QAOA layer), and conveyor SWAPs are inserted SWAP
gates. You choose the initial qubit placement and the SWAP schedule; the checker
verifies the routed program is **functionally equivalent** (every gate runs on
adjacent hardware qubits, in order) and then **counts the SWAPs**.

You choose:
1. the **initial placement** — which bay each robot starts on, and
2. a schedule of **SWAP** moves interleaved with the handoffs,

so that every handoff is executed on adjacent bays, in order. **Minimize the total
number of SWAP moves.**

## Input (stdin)

```
V E m
u v            (E lines: a conveyor track between bays u and v)
a b            (m lines, in order: handoff between robots a and b)
```

The bays form a connected rectangular grid, so any placement is reachable by SWAPs.

## Output (stdout)

First the initial placement, then the schedule:

```
MAP p_0 p_1 ... p_{V-1}
<steps...>
```

* `MAP` is followed by `V` integers: `p_i` is the bay on which robot `i` starts.
  The `p_i` must be a **permutation** of `0..V-1` (every bay holds one robot).
* Each step line is one of:
  * `S u v` — SWAP the robots currently on bays `u` and `v`. Requires `(u,v)` to
    be an existing conveyor track.
  * `G` — execute the **next pending handoff** (they are consumed in input order).
    Requires the two named robots to currently sit on adjacent bays.

Exactly `m` `G` steps must appear, in order; any extra/fewer, any SWAP on a
non-track, or any handoff on non-adjacent bays makes the output **infeasible**.

## Feasibility

The output is feasible iff `MAP` is a permutation of `0..V-1`, every `S` acts on a
real track, and each `G`, in order, finds its two robots on adjacent bays, with
exactly `m` handoffs executed. Infeasible output scores `0`.

## Objective

Minimize `F` = the number of `S` (SWAP) steps.

## Scoring

Let `B` be the checker's internal baseline: identity placement (robot `i` on bay
`i`) with each handoff routed by moving robot `a` along a shortest track-path
until adjacent to `b`. With `F` your SWAP count:

```
Ratio = min(1.0, 0.1 * B / F)
```

So reproducing the baseline scores `0.1`; halving its SWAPs scores `0.2`; a
10x reduction caps at `1.0`. Optimal routing is NP-hard (token swapping under an
ordered gate list), so the ceiling is genuinely open.

## Constraints

* `9 <= V <= 42`, grid floor.
* `m` grows with the instance (roughly `V + 3*testId` handoffs).
* Most handoffs are within hidden **work zones** whose membership is scrambled
  relative to robot id — so identity placement scatters each zone across the
  floor, but a good placement gathers it.

## Example (worked score)

Suppose an instance yields baseline `B = 120` SWAPs. A submission that executes
all handoffs feasibly using `F = 48` SWAPs scores
`Ratio = min(1, 0.1 * 120 / 48) = min(1, 0.25) = 0.25`. A submission that needs
`F = 12` SWAPs scores `min(1, 0.1 * 120 / 12) = 1.0`. An output whose `MAP`
repeats a bay, or that runs a handoff on non-adjacent bays, scores `0.0`.
