# Tail-Risk Backups: Abandoning the Nastiest Disasters

## Problem
You run `N` data shards; shard `i` is worth `v_i`. You have `M` backup machines; machine `j`
can host at most `cap_j` replica copies (of any shards). You may create at most `R` replicas
in total (the **replica budget**); each replica places one copy of one shard on one machine,
and you may never place two replicas of the *same* shard on the *same* machine.

A **disaster** destroys exactly two *distinct* machines simultaneously. A shard **survives**
a disaster iff at least one of its replicas sits on a machine that was not destroyed. For
every one of the `M*(M-1)/2` possible machine pairs, the surviving total value (the sum of
values of shards that survive that particular pair going down) is a number -- this gives
`M*(M-1)/2` disaster-outcome values, one per pair.

You are **not** scored by the worst outcome (that would be minimax). Instead the input gives
an index `K`: sort all outcome values ascending; your score is the `K`-th smallest one. This
means you may deliberately let the `K-1` worst outcomes collapse to nothing -- choosing which
disasters to write off, and spending your whole budget defending everything else, is part of
the decision.

## Input (stdin)
```
N M R K
v_1 v_2 ... v_N
cap_1 cap_2 ... cap_M
```
All 1-indexed. `1 <= K <= M*(M-1)/2`.

## Output (stdout)
```
P
t_1 m_1
t_2 m_2
...
t_P m_P
```
`P` = the number of replicas you place (`P <= R`); line `k` places one replica of shard `t_k`
on machine `m_k`.

## Feasibility
An output is valid iff **all** hold:
- every `t_k` is an integer with `1 <= t_k <= N`, every `m_k` an integer with `1 <= m_k <= M`;
- no `(t_k, m_k)` pair repeats (a shard is never double-replicated on one machine);
- `P <= R`;
- for every machine `m`, the number of output lines naming `m` is `<= cap_m`.
Any violation (including malformed/non-finite tokens) scores `Ratio: 0.0`.

## Objective
Let `S` be the ascending-sorted list of the `M*(M-1)/2` machine-pair surviving-value totals
(a shard survives pair `(a,b)` iff it has a replica on some machine outside `{a,b}`; a shard
with zero replicas never survives anything). Maximize `F = S[K]` (1-indexed).

## Scoring
The checker builds its own weak reference allocation: spend only **40%** of the replica
budget (rounded, value-oblivious triage that gives up early), handing each covered shard
exactly **one** replica, cycling machines in index order `1,2,...,M,1,2,...` and skipping
any machine already at capacity. Let `B` be that construction's own `S[K]` value (at least
`1e-9`). Then, with maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the reference construction scores `Ratio = 0.1`; ten times better caps at `1.0`.

## Constraints
`8 <= N <= 40`, `5 <= M <= 12`, `N <= R < 2*N`, `cap_j >= 2`, `1 <= v_i <= 999`. Time limit
5s, memory 512m.

## Example
Illustrative shape only, not a real test case. `N=3` shards worth `10, 10, 10`; `M=3`
machines, `cap = (2,2,2)`; `R=4`, `K=2` (there are 3 pairs: `(1,2),(1,3),(2,3)`; the worst 1
of them may be sacrificed). Place shard 1's replicas on machines 1 and 2, and shard 2's
single replica on machine 3 (shard 3 gets nothing). Pair `(1,2)`: shard 1 dies, shard 2
survives (machine 3 intact) -> value `10`. Pair `(1,3)`: shard 1 survives (machine 2 intact),
shard 2 dies -> value `10`. Pair `(2,3)`: shard 1 survives (machine 1 intact), shard 2 dies
-> value `10`. `S = [10, 10, 10]`, so `F = S[2] = 10`. A different split of the same budget
trades shard 1's spread for coverage of shard 3, changing which pair ends up worst -- picking
that trade-off well is the whole game.
