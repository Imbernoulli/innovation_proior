# Confluent Collapse: Minimum-Step Term Rewriting

## Problem
You are given a term built from a tiny rewriting system with exactly two rules:

- `u` is an atomic leaf.
- `(drop T)` is a unary node; its rule is `drop(x) -> x` (delete the wrapper).
- `(dup T)` is a unary node; its rule is `dup(x) -> (pair x x)` (duplicate the argument).
- `(pair T1 T2)` is a binary constructor with **no rule** — once built it is permanent.

`drop` and `dup` are the only two symbols with rules. Each rule's left-hand side matches only its own root symbol, unconditionally (it does not matter what is inside the argument), and `pair` never reduces at all. So this system has **zero critical pairs**: it is confluent by construction. Every terminating reduction of the same starting term reaches the **same** normal form (a tree built only from `pair`/`u`), no matter which live redex you fire at each step, in what order. Termination is likewise guaranteed: every step only copies or discards material that was already present in the original finite term — no rule ever manufactures a new `drop`/`dup` symbol out of nothing.

Your job: reach the normal form in as **few rewrite steps as possible**. Confluence means you are free to choose the order — any terminating order gets you to the right (unique) answer — but the order changes the step count a great deal, because firing a `dup` before its argument is fully reduced duplicates whatever un-finished `drop`-work is still sitting inside it, and you then pay for that work again in *every* copy.

## Input
Line 1: an integer N — the node count of the term (informational).
Line 2: the term as a whitespace-separated S-expression over `u`, `(drop T)`, `(dup T)`, `(pair T1 T2)`.

## Output
Line 1: an integer M — the number of rewrite steps you perform.
Next M lines: the **position** rewritten at that step — a string over `{0,1}`, the path of child indices from the root (`0` = first child, `1` = second, read left to right), or `.` for the root itself. At each step the checker applies whichever rule matches the *current* symbol at that position (`drop` or `dup`); addressing a `pair`/`u` node, or a position that no longer exists, is illegal.

## Feasibility
Every step must address a live `drop` or `dup` node in the term as it stands after the previous steps. After your last step the term must contain **no** `drop`/`dup` node anywhere — a genuine normal form. Any illegal step, or stopping short of normal form, scores 0.

## Objective and Scoring
Minimize M. The checker computes its own reference count B by running the naive **leftmost-outermost** strategy (always contract the shallowest live redex, ties broken left) on the same input term, then scores
```
Ratio = min(1000, 100 * B / M) / 1000
```
Matching B scores about 0.1; using 10x fewer steps than B caps the ratio at 1.0.

## Constraints
1 <= N <= a few hundred nodes. Time limit 5s, memory 256MB.

## Example
Term: `(dup (dup (drop (drop (drop u)))))`, i.e. `dup(dup(drop(drop(drop(u)))))`.

**Sequence A** (leftmost-outermost, naive): fire the outer `dup` first — this duplicates the *whole unreduced* `dup(drop(drop(drop u)))` subtree into two independent copies (1 step). Each copy's own `dup` then fires next, duplicating its *still-unreduced* `drop(drop(drop u))` argument again — now there are **four** independent copies of that 3-`drop` chain (2 more steps). Each of the four copies still needs its own 3 `drop`s (12 more steps). Total: 1 + 2 + 12 = **15 steps**.

**Sequence B** (innermost, shrink-before-duplicate): first collapse the 3 `drop`s down to `u` (3 steps, giving `dup(dup(u))`), then fire the inner `dup` (1 step, giving `dup(pair(u,u))`), then the outer `dup` (1 step, giving `pair(pair(u,u),pair(u,u))`) — **5 steps** total, reaching the *same* normal form as Sequence A.

Here B (the checker's leftmost-outermost baseline) = 15. Sequence A: `100*15/15 = 100` -> Ratio 0.100000. Sequence B: `100*15/5 = 300` -> Ratio 0.300000. Both reach `pair(pair(u,u),pair(u,u))`; only the step count differs — that gap is exactly what confluence lets you exploit.
