# Plate-Stack Codegen

## Problem

A juggler keeps every plate spinning in the air — there is no table to set one down on.
You must compile an arithmetic expression DAG (a computation with **shared subterms**,
each possibly used by several later computations, and by several designated final
outputs) into a program for a tiny stack machine. The machine has a fast stack — but
only a **cheap permutation group** reaching its top three slots — and a slow "table"
(memory) that can hold anything but costs eight times as much to use. Values that fall
out of the cheap reach must either be paid for via the table or recomputed from scratch.
Your job: minimize total weighted instruction cost while producing every output's exact
value, in order.

All arithmetic is performed modulo the prime `P = 998244353`.

## Input (stdin)

```
P
N M K
v_1 v_2 ... v_N
op_1 childL_1 childR_1
...
op_M childL_M childR_M
out_1 out_2 ... out_K
```
`N` leaves have fixed values `v_1..v_N` (leaf `i` is referenced as `Li`). There are `M`
internal nodes (node `i`, referenced as `Ni`), each `op_i in {ADD, SUB, MUL}` combining
two earlier components (a leaf `Lj` or an earlier node `Nj`, `j<i`): value(`Ni`) =
value(`childL_i`) `op_i` value(`childR_i`) (mod `P`). Finally, `out_1..out_K` name `K`
node references (`N#`) that must be emitted, **in this exact order**.

## Output (stdout)

A straight-line program: one instruction per non-empty line, from:
```
PUSH i        push value of leaf i (1<=i<=N)                        cost 1
OP ADD|SUB|MUL   pop a=top, pop b=new top, push (b OP a) mod P       cost 1
DUP           push a copy of the top element                        cost 1
SWAP          swap the top two elements                              cost 1
OVER          push a copy of the 2nd-from-top element                cost 1
ROT           let (v0,v1,v2) = (top, 2nd, 3rd); after: (top,2nd,3rd) = (v2,v0,v1)   cost 1
STORE s       pop the top element into memory slot s (0<=s<10^6)     cost 8
LOAD s        push a copy of memory slot s (must be previously STOREd)  cost 8
OUTPUT        pop the top element and emit it as the next output     cost 1
```
`DUP`/`OVER`/`ROT` are the *only* ways to reach a value without recomputing or paying for
memory, and together they can only ever touch the top three stack slots — anything
buried deeper is unreachable except via `STORE`/`LOAD` or by rebuilding it from its
children again.

## Feasibility

The program is infeasible (score 0) if: any instruction underflows the stack; any
`PUSH`/`STORE`/`LOAD` index is out of range; any `LOAD s` occurs before a `STORE s`;
the program does not contain exactly `K` `OUTPUT` instructions; or the `j`-th `OUTPUT`
does not pop the true value of `out_j` (mod `P`).

## Objective & Scoring

Minimize total weighted instruction cost `F` (sum of the per-instruction costs above).
The checker computes its own baseline `B` — the cost of recomputing every output's
subtree completely independently, from leaves, with zero sharing (no shuffles, no
memory) — and reports `Ratio = min(1, 0.1 * B / F)` (lower `F` scores higher).

## Constraints
`1<=N<=200`, `1<=M<=2000`, `1<=K<=200`. Time limit 2-5s.

## Example (worked score, illustrative shape only — not real test data)

DAG: leaves `L1=3, L2=5`; `N1 = L1 ADD L2` (=8); outputs = `N1, N1` (used twice, e.g. it
feeds two different downstream reports). Baseline `B` recomputes `N1` twice from
scratch: `(PUSH1 PUSH2 OP) x2 + OUTPUT x2` = `3+3+1+1 = 8`. A smarter program computes
`N1` once and keeps a spare on the stack for the second use: `PUSH1 PUSH2 OP DUP OUTPUT
OUTPUT` = `1+1+1+1+1+1 = 6` (the second `OUTPUT` pops the spare `DUP` left behind).
`Ratio = min(1, 0.1*8/6) ≈ 0.133`.
