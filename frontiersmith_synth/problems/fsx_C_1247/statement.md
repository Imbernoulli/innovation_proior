# Weak Rows, Strong Rows: Retention-Binned DRAM Refresh Scheduling

## Problem

A DRAM chip has `Bnum` independent banks. Bank `b` has `Rb` rows, indexed `0..Rb-1`. Time
advances in discrete slots `0, 1, ..., T-1`. Each bank has its own refresh port: **at most one
refresh command can be issued per bank per slot** (different banks never contend with each
other — refreshing bank 0 never blocks anything in bank 1).

Row `i` of bank `b` has a retention bound `rho[b][i]`: consider the sequence of slots at which
that row is refreshed, together with a virtual boundary at slot `-1` (the row starts "fresh" at
time 0) and a virtual boundary at slot `T` (data must still be valid through the end of the
horizon). Every gap between consecutive entries of this sequence (including the two boundary
gaps) must be `<= rho[b][i]`. Rows vary enormously in how long they can safely go between
refreshes — the input tells you exactly how long each row can hold.

You are also given an access trace: `M` requests, each `(slot, bank, weight)`. If your schedule
issues a refresh command in `bank` at exactly `slot`, any request `(slot, bank, weight)` in the
trace **stalls**, costing `weight`. Your goal: choose WHEN and for WHICH row to issue refresh
commands so every row's retention bound holds, while minimizing total stall weight summed over
the access trace.

## Input (stdin)

```
T Bnum
Rb_0
rho[0][0] rho[0][1] ... rho[0][Rb_0-1]
Rb_1
rho[1][0] ... rho[1][Rb_1-1]
...                              (one Rb line + one rho line per bank, Bnum banks total)
M
slot_1 bank_1 weight_1
...
slot_M bank_M weight_M
```
All values are non-negative integers. It is guaranteed `rho[b][i] >= Rb` for every row (so a
plain round-robin refresh of every row, one per slot per bank, is always a valid — if wasteful —
schedule; the instance is never infeasible).

## Output (stdout)

```
K
slot_1 bank_1 row_1
...
slot_K bank_K row_K
```
`K` refresh commands (any order). Each line issues a refresh of `row_j` in `bank_j` at
`slot_j`. Two commands may not share the same `(bank, slot)` pair.

## Feasibility

An output is rejected (score 0) if: any token is missing/non-integer/out of declared bounds;
any `(bank, slot)` pair is used by more than one refresh command; or any row's retention bound
is violated by its refresh sequence (checked with the `-1`/`T` boundary rule above).

## Scoring

Let `F` = total stall weight your schedule incurs (sum of `weight` over access requests whose
`(slot, bank)` is occupied by one of your refresh commands). Let `B` = total weight of ALL
access requests (the stall the checker's own trivial "refresh every slot" construction would
incur, since that construction occupies every slot). Score:

```
ratio = min(1.0, 0.1 * B / max(1e-9, F))
```

Fewer, better-placed refreshes -> smaller `F` -> higher score. Refreshing everything, always,
scores about 0.1.

## Example

Toy instance, NOT a worked score: 1 bank, 2 rows, `T=6`, `rho=[2,4]` (row 0 weak, row 1
strong), one access request `(3, 0, 5)`. Row 0's bound is tight: it MUST fire at `0,2,4` (no
other legal choice). Row 1's looser bound gives real freedom. Treating row 1 like row 0
(uniform worst-case, period 2) fires it at `1,3,5`, colliding with the request at slot 3 and
costing `F=5`. Recognizing row 1's true bound instead needs only two refreshes — `1,5` (gaps
`2,4,1`, all `<=4`) — which both avoids slot 3 (`F=0`) and uses one fewer refresh command,
illustrating that binning by true retention (not the bank's worst row) changes both how OFTEN
and WHERE you refresh.

## Constraints

`1 <= Bnum <= 6`, `1 <= Rb <= 130`, `1 <= T <= 4200`, `0 <= M <= 2*T*Bnum`, weights `>= 1`.
Time limit 5s, memory 512MB.
