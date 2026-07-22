# Prune the Hydra Before It Clones Its Heads

## Problem
You are given a closed term built from three fixed **combinators** `S`, `K`, `I` and
opaque **atoms** (which never reduce), combined by binary application `(f a)`. Three
rewrite rules apply, matched only against the *exact* left-spine shape shown (no other
rules exist):

```
I x           -->  x                          cost 1
K x y         -->  x            (y discarded)  cost 1
S x y z       -->  (x z) (y z)  (z duplicated)  cost 1 + 2*size(z)
```

`size(z)` is the number of nodes (`S`/`K`/`I`/atom/application) in `z` **at the moment
the rule fires**. Firing an `S`-redex physically creates a second copy of whatever `z`
currently looks like — the more of `z` is still unresolved, the more expensive the
copy. A `K`-redex, by contrast, *deletes* `y` outright; nothing inside `y` is ever paid
for if `y` is discarded before being touched.

Every input term is guaranteed to reach a unique normal form (no more redexes) under
**any** terminating reduction order — but different orders pay wildly different total
cost to get there. Your job is to steer the reduction to normal form as cheaply as
possible.

## Input (stdin)
One line, prefix notation, whitespace-separated tokens:
- `S`, `K`, `I` — a bare combinator atom.
- `L <id>` — an opaque leaf atom with integer id `<id>` (never itself a redex).
- `@ <f> <a>` — the application `(f a)`.

Example: `@ @ @ S K I L 0` denotes `((S K) I) L0`.

## Output (stdout)
```
R
addr_1 addr_2 ... addr_R
```
`R` is the number of rewrite steps you perform. Each `addr_i` names the node to rewrite,
as a path from the **current** term's root (after all previous steps): `.` denotes the
root itself; otherwise a string over `{0,1}`, where `0` descends into the function side
and `1` into the argument side of an application. The node named must, at that moment,
exactly match one of the three rule shapes above.

## Feasibility
Replaying `addr_1..addr_R` in order must: (1) always name a node that is currently a
valid redex of one of the three shapes; (2) end with a term containing **no** remaining
redex; (3) that final term must equal the term's unique normal form. Any parse error,
out-of-range `R`, malformed address, address running off the tree, a named node that
isn't a redex, or a final term short of normal form scores `0`.

## Objective
Minimize `F`, the sum of the costs of the `R` steps you fired (as defined above).

## Scoring
The checker computes its own internal baseline `B`: the cost of the naive "always fully
reduce both sides of an application before combining them" strategy on the same input
(eager / call-by-value to normal form). Then:

```
Ratio = min(1.0, 0.14 * B / max(1e-9, F))
```

Matching the baseline's cost scores `0.14`; roughly halving it moves you well above
that; the true minimum cost is not the eager baseline and is not generally known in
closed form for these instances, so headroom remains above any fixed strategy.

## Constraints
- Term size: at most a few thousand nodes. Time limit 5s, memory 512MB.
- `0 <= R <= 20000`.
- Exact term-rewriting semantics; no floating point, no timing.

## Example
Take `((S K) I) L0` (a leaf `L0` of size 1). Firing the `S`-redex at `.` costs
`1 + 2*size(L0) = 3`, giving `(K L0) (I L0)`. That whole term *itself* now matches the
`K`-redex shape directly (`X=L0`, `Y=(I L0)`) — firing it at `.` again costs `1` and
discards `(I L0)` **without ever reducing the `I` inside it**, for a total `F = 4` and
final answer `L0`. The eager baseline instead normalizes `(I L0)` down to `L0` *before*
noticing the `K`-redex (paying the extra `I`-step), giving `B = 5`; this submission's
`Ratio = min(1.0, 0.14*5/4) = 0.175`. Real test instances plant many such duplication
and discard opportunities together, at scales where the choice of order changes the
total cost by a large factor.
