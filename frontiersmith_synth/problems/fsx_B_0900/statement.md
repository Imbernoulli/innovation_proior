# Doorway Tolls: Cost-Routed Pallet Reshuffling

## Problem
A warehouse has `n` pallet slots, partitioned into `m` segments (aisles). Right now slot
`i` holds some pallet `perm[i]`; every pallet's home is the slot with its own number, so
pallet `p` belongs at slot `p`. There are also `K` scratch registers ("staging bays"),
each able to hold at most one pallet, all empty right now and required to be empty again
once you finish.

You control a forklift that executes a sequence of operations:
- `M a b` (**MOVE**): `b` must currently be empty and `a` must currently hold a pallet;
  afterwards `b` holds that pallet and `a` is empty.
- `S a b` (**SWAP**): `a` and `b` must both currently hold pallets; their contents are
  exchanged.

`a` and `b` each name either a real slot (an integer `0 <= idx < n`) or a scratch
register (a token `R0`, `R1`, ..., up to `R{K-1}`).

**Cost.** If both `a` and `b` are real slots in the *same* segment, the operation costs
`1`. If they are real slots in *different* segments, it costs `D` (crossing a doorway is
expensive). If either endpoint is a scratch register, it costs `1` flat — a register is a
small depot cheaply reachable from any aisle, but it can only hold one pallet at a time
and never appears "inside" a segment.

Your program must bring every pallet home: after executing it in order, slot `i` must
hold pallet `i` for every `i`, and every register must be empty. Minimize the total cost.

## Input (stdin)
```
n m K D
seg[0] seg[1] ... seg[n-1]
perm[0] perm[1] ... perm[n-1]
```
`seg[i]` is the segment id (`0..m-1`) of slot `i`. `perm` is a permutation of `0..n-1`
(`perm[i]` = the pallet currently at slot `i`). `2 <= n <= 200000`, `1 <= K <= 16`,
`1 <= D <= 1000`.

## Output (stdout)
```
T
op_1
op_2
...
op_T
```
`T` operation lines follow, each `M a b` or `S a b` as described above, and nothing else
(no trailing tokens after the `T`-th line). `0 <= T <= 8n + 64`.

## Feasibility
Every token must parse as a well-formed opcode/slot/register (no `nan`/`inf`/out-of-range
indices, no malformed lines). Every operation must be legal in the CURRENT state (a MOVE
destination empty and source occupied; a SWAP with both endpoints occupied). Simulating
the whole program must leave every slot `i` holding pallet `i` and every register empty.
Any violation scores `Ratio: 0.0`.

## Objective
Minimize `F`, the sum of every operation's cost under the rule above.

## Scoring
The checker builds its own reference program directly from `perm` (never from your
output): decompose `perm` into cycles and, for each cycle, chain the standard
star-of-transpositions from its smallest-index element — but realize every one of those
transpositions the hard way, through a shared register, at cost `edgecost + 2` instead of
a plain `SWAP`'s `edgecost`. Call this cost `B`. Then:
```
Ratio = min(1, 0.1 * B / F)
```
Smaller `F` is better. A program that matches `B` scores `0.1`; a program `10x` cheaper
than `B` saturates at `1.0`.

## Example
Toy instance: `n=4, m=2, K=1, D=5`, segments `[0,0,1,1]`, `perm = [1,0,3,2]` (two local
2-cycles, illustrative shape only — not the doorway-relay structure of the real test
cases). One valid, cheap program: `S 0 1` (cost 1, same segment) then `S 2 3` (cost 1),
`T=2`, `F=2`. The checker's reference `B` for this instance is `2*(1+2)=6`, so this
program scores `min(1, 0.1*6/2) = 0.3`.

## Constraints
- `2 <= n <= 200000`, `1 <= m <= n`, `1 <= K <= 16`, `1 <= D <= 1000`, `0 <= T <= 8n + 64`.
- Time limit 5s, memory 512MB. Deterministic scoring; no timing measurements.
