# Canal-Exchange Routing on a Reservoir Dam Network

## Problem
A regional water authority operates a **reservoir dam network** laid out as a `rows x cols`
grid of reservoirs. Two reservoirs are joined by a **canal** if and only if they are
orthogonally adjacent in the grid. A reservoir at grid position `(r, c)` has id
`r*cols + c` (0-indexed, row-major).

There are `V = rows*cols` labelled **water batches**. Initially batch `i` sits in
reservoir `i` (the identity placement).

Engineering issues a list of `K` scheduled **mixing operations**. Operation `t` names two
batches `(a_t, b_t)` that must be brought into two **canal-connected** reservoirs so their
contents can be mixed at a shared sluice. The mixing operations all commute: you may
perform them in **any order**.

The only way to move batches is a **canal exchange**: pick two canal-connected reservoirs
and swap the batches currently held in them. Your job is to emit a program of exchanges and
mixings that performs every operation, using **as few exchanges as possible**.

This is the classic hardware-routing / token-swap problem skinned as water logistics: it is
NP-hard, has no known polynomial optimum, and admits many distinct heuristics.

## Input (stdin)
```
rows cols
K
a_1 b_1
...
a_K b_K
```
`2 <= rows,cols <= 40`, `1 <= K`, `0 <= a_t,b_t < V`, `a_t != b_t`. Batch `i` starts in
reservoir `i`.

## Output (stdout)
```
M
<move_1>
...
<move_M>
```
`M` is the number of moves. Each move is one of:
- `S u v` — a canal exchange: swap the batches in reservoirs `u` and `v`. `u` and `v` must
  be canal-connected (orthogonally adjacent) at all times (the grid never changes).
- `X t` — perform mixing operation `t`. Its two batches must currently sit in
  canal-connected reservoirs. Each `t in [0,K)` may be performed **at most once**.

## Feasibility
Rejected (score 0) unless: every `S` is on a real canal; every `X`'s two batches are
canal-connected at that moment; every operation index is valid and performed exactly once;
**all K operations are performed**; and every token is a plain integer.

## Objective (minimize)
`F` = number of `S` (canal exchange) moves.

## Scoring
Let `B` be the number of exchanges used by the reference **naive fixed-order router** (it
handles the operations in the given order, each time walking batch `a` to batch `b` along a
row-then-column shortest canal path). Then

```
Ratio = min(1, 0.1 * B / F)
```

Reproducing the naive router scores about `0.1`; halving its exchanges roughly doubles the
score; a 10x reduction caps at `1.0`. Fewer exchanges is always better.

## Constraints
Deterministic integer scoring only. `M <= 5,000,000`.

## Example (worked score)
Suppose the naive router needs `B = 40` exchanges on an instance and your program performs
all operations with `F = 25` exchanges. Then `Ratio = min(1, 0.1 * 40 / 25) = 0.16`. A
sharper router reaching `F = 16` would score `0.1 * 40 / 16 = 0.25`.
