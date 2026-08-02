# Delivered-Cost Retrosynthesis: Convergent Routes Beat Short Ones

## Problem
You are given a library of chemical **molecules** (integer ids) and
**reactions**. Each reaction turns 1 or 2 input molecules into one output
molecule, consuming a whole batch of inputs and delivering only a fraction as
product: it has a **yield** `y` (0 < y <= 1) and a **step cost** `c`. To get
`a` units of the output you must feed `a / y` units of *each* input and pay
`c * (a / y)`.

A **target** molecule `T` must be delivered, exactly 1 unit. Some molecules
are **purchasable raw materials** at a fixed cost per unit; everything else
is built by chaining reactions backwards from `T` to raw materials — a
**disconnection tree**. A molecule can sometimes be reached in more than one
way: a short **linear** chain, or a **convergent** route where two
independently-built fragments join in a final coupling reaction. Some
molecules can only pass through a later reaction if a sensitive site was
first **protected** (an extra protect/react/deprotect detour — more steps,
each at high yield); the alternative is a direct "unprotected" reaction that
is available but has a much lower yield. Which choice is cheaper is *not*
obvious from the step count: yields compound **multiplicatively** along every
path from a raw material to the target, so the shortest chain can need far
more raw material — and pay far more amplified step-cost — than a route that
uses *more total reactions* but keeps every individual path to the target
short.

## Input (stdin)
```
N M T P
p_1 cost_1
...
p_P cost_P
rid output k in_1 [in_2] yield_pct cost
... (M lines)
```
`N` bounds molecule ids (`0 <= id < N`), `M` is the number of reactions, `T`
is the target molecule, `P` is the number of purchasable raw materials (each
with an integer per-unit cost). Each reaction line gives its id, output
molecule, arity `k` (1 or 2), its `k` input molecules, its yield as an
integer percent (1-100), and its integer step cost. The reaction hypergraph
is acyclic: no molecule appears among the inputs of any reaction reachable
from itself.

## Output (stdout)
A synthesis program, one instruction per line, each introducing a fresh
positive integer **instance id**:
```
BUY <molecule_id> <instance_id>
REACT <reaction_id> <instance_id> <input_instance_id_1> [<input_instance_id_2>]
ROOT <instance_id>
```
`BUY` purchases one instance of a raw material. `REACT` applies a reaction
(by id) to previously-defined instances, producing a new instance of that
reaction's declared output. `ROOT` names the single instance delivered as the
target. Every instance must be defined before it is referenced, and consumed
as an input **at most once**.

## Feasibility
Reject (score 0) unless: every `REACT` uses a real reaction id whose declared
input molecule multiset exactly matches the molecules of its cited input
instances; every `BUY` molecule is in the purchasable list; every instance is
consumed at most once; exactly one `ROOT` line exists and its molecule is
`T`; all tokens are finite integers.

## Objective
Minimize the total delivered cost `F` to produce 1 unit of `T`, propagated
top-down from the root: delivering `a` units of a reaction's output costs
`cost * (a/yield)` plus, for each input, its own delivered cost to supply
`a/yield` units of it; buying `a` units of a raw material costs
`purchase_cost * a`.

## Scoring
Let `B` be the cost of the route obtained by always picking the
**smallest-id** reaction available for whichever molecule is currently
needed (a fixed, always-feasible reference route). With your feasible cost
`F`: `Ratio = min(1.0, B / (10 * F))`.

## Constraints
`1 <= N <= 600`, `1 <= M <= 30`, `1 <= P <= 5`, yields are integers 1-100,
costs are nonnegative integers `<= 20`. Time limit 5s.

## Example (worked score, illustrative shape only)
Molecule 1 purchasable at cost 2. Reactions: id0 makes molecule 2 from
molecule 1 (yield 80%, cost 1); id1 makes target 0 directly from molecule 1
(yield 40%, cost 1); id2 makes target 0 from molecule 2 (yield 80%, cost 2).
Naive smallest-id baseline takes id1 directly: `B = (2+1)/0.4 = 7.5`. The
two-step route `BUY 1 1` / `REACT 0 2 1` / `REACT 2 3 2` / `ROOT 3` costs
`F = ((2+1)/0.8 + 2)/0.8 = 7.1875`, giving `Ratio = min(1, 7.5/71.875) ~= 0.104`
— fewer steps is not always cheaper once yields are accounted for.
