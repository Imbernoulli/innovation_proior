# The Weakest Isolation That Is Still Correct

## Problem
You are tuning a database's per-transaction isolation level. There are `N`
transaction programs; program `i` has a **read-set** and a **write-set**
(sets of key ids it touches) and a throughput **weight** `w_i` (its call
frequency). You must assign every transaction one of three isolation
levels:

- **0 = READ COMMITTED** — fastest, weakest.
- **1 = SNAPSHOT** — moderate cost; stops lost updates, nothing else.
- **2 = SERIALIZABLE** — slowest; always correct on its own.

Two hazard classes are determined *statically*, from the read/write sets
alone (no runtime scheduling is simulated):

- A **read-write hazard edge** `i -> j` exists whenever `R_i` and `W_j`
  share a key (transaction `i` reads something `j` writes). A read-write
  edge is *exposed* only if **both** endpoints sit at level `<= 1`. A
  **directed cycle** made entirely of exposed edges is a write-skew
  anomaly: every transaction on the cycle can read stale data and commit
  something jointly inconsistent. Breaking a cycle needs only **one** of
  its members promoted all the way to level 2 — SERIALIZABLE anywhere on
  the ring kills every hazard edge touching it, so the whole cycle stops
  being all-exposed.
- A **write-write pair** `{i, j}` exists whenever `W_i` and `W_j` share a
  key. It is exposed only if **both** `i` and `j` sit at level 0 — level 1
  already stops the lost update, no cycle reasoning is needed here.

## Input (stdin)
```
N K
w_1 nR_1 r_{1,1} ... r_{1,nR_1} nW_1 wr_{1,1} ... wr_{1,nW_1}
...
```
Line 1: `N` transactions, `K` distinct key ids (`0..K-1`). Then `N` lines,
one per transaction (1-indexed by input order): weight, read-set size and
members, write-set size and members.

## Output (stdout)
`N` integers `L_1 ... L_N`, each in `{0,1,2}` — the chosen isolation level
per transaction, in input order (whitespace-separated, any layout).

## Feasibility
Output scores `Ratio: 0.0` if: token count `!= N`; any token is
non-finite or not (numerically) an integer in `{0,1,2}`; the exposed
read-write graph (edges with both endpoints `<= 1`) contains any directed
cycle; or any write-write pair has both endpoints at level 0.

## Objective
Maximize total throughput
```
F = sum_i w_i * speed(L_i),   speed = {0: 4.0, 1: 2.0, 2: 1.0}
```
subject to the feasibility rules above.

## Scoring
The checker's own reference `B` is the always-safe construction "run every
transaction at level 2": `B = sum_i w_i * speed(2)`.
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching `B` scores `0.1`; four times `B` (every transaction safely at
level 0) caps at `0.4` — headroom stays open above any reference solution.

## Constraints
`3 <= N <= 20`, `1 <= K <= 40`, `1 <= w_i <= 25`. Read/write sets are
small (each `<= 4` keys). Time limit 5s, memory 512MB.

## Example
`N=4`. T1: `w=10, R={0}, W={1}`. T2: `w=10, R={1}, W={0}`. T3: `w=8, R={},
W={2}`. T4: `w=8, R={}, W={2}`. T1 and T2 form a 2-cycle (`1->2` via key 0,
`2->1` via key 1); T3 and T4 share write key 2.

Output `2 0 1 0` (T1 SERIALIZABLE, T2 READ COMMITTED, T3 SNAPSHOT, T4 READ
COMMITTED): the cycle is broken because T1 is no longer exposed, and the
write-write pair is safe because T3 is not level 0. `F = 10*1.0 + 10*4.0 +
8*2.0 + 8*4.0 = 98`, `B = 36*1.0 = 36`, `Ratio = min(1000, 272.2)/1000 =
0.2722`.

This is a *different, unrelated* illustrative shape from the harder
planted cases — do not assume every instance is one 2-cycle plus one
write-write pair; larger cases plant multiple cycles of different lengths
(2, 3, 4 transactions) simultaneously, each needing its own promoted
member, plus unrelated write-write clusters and fully independent
transactions mixed in. The trap: output `1 1 0 0` instead (weaken both
cycle members to SNAPSHOT because each only conflicts with one other
transaction) — the cycle is still made entirely of exposed edges (both
endpoints `<= 1`), so this scores `Ratio: 0.0` even though every
transaction individually "looks lightly loaded."
