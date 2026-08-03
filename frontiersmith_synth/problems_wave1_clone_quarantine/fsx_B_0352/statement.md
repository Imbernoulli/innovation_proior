# Greenhouse Zones — Sensor-Predicate Prefix-Cache Layout (Format B, isolated)

A greenhouse-automation controller runs a fixed daily **stream of irrigation control
checks**, one after another, over its zones. Each check evaluates a **set of sensor
predicates** (e.g. "zone-3 soil moisture < 30%", "canopy temp > 28C", "reservoir level
OK"). Evaluating a predicate costs some *tokens* of work.

The controller memoizes results in a **prefix cache** (a trie). To use it, the
controller lays out **all** predicates in **one global order**, chosen once. Each check
then evaluates *its own* predicates **in that global order**. When a check's leading
run of predicates exactly matches the leading run of an **earlier** check, that shared
**prefix is a cache hit** — its token cost is reimbursed for free; only the first
divergent predicate onward must be paid.

Because checks that belong to the same "family" share many core predicates, a smart
global layout makes those shared predicates line up at the **front** of every check, so
long prefixes are reused. A careless layout (for instance, one that lets a check's rare,
quirky predicates sort ahead of its shared core) shatters the prefixes and wastes work.

**Your job: choose the global predicate order that maximizes the total cache-hit
tokens over the whole fixed stream.**

## The prefix-cache model (exactly how you are scored)

You return a permutation `order` of the predicate indices `0..M-1`. Let `rank[a]` be the
position of predicate `a` in `order`. The stream of checks is processed **in the given
order**, maintaining a trie that starts empty:

For each check with predicate set `S` (processed in stream order):
1. Form its sequence `seq` = the predicates of `S` sorted **ascending by `rank`**.
2. Walk `seq` down the trie from the root. Let the **matched prefix** be the longest
   leading run of `seq` whose nodes already exist in the trie. Add
   `sum(weights[a] for a in matched_prefix)` to your hit total.
3. Insert the whole `seq` into the trie (creating nodes for the unmatched suffix).

Your objective is the **total hit tokens** summed over every check. Higher is better.
(Identical predicate sets always share their whole sequence regardless of layout; the
layout only changes how much *overlapping-but-different* checks share.)

## Public instance (stdin JSON)
```json
{
  "M":       <int>,               // number of distinct predicates (indices 0..M-1)
  "weights": [<int>, ...],        // weights[a] = token cost of predicate a (length M)
  "queries": [[a, b, ...], ...]   // the fixed stream; each entry is a check's predicate set
                                  //   (a sorted list of distinct indices in 0..M-1)
}
```

## Answer (stdout JSON)
```json
{ "order": [ <a permutation of 0..M-1> ] }
```
Requirements: `order` is a list of exactly `M` integers that is a permutation of
`0..M-1`. Any other shape (wrong length, repeats, out-of-range, non-integers) is
**infeasible** and scores **0**.

## Scoring

Let `H(order)` be the total hit tokens your layout achieves, and `H_id` the hit tokens
of the identity layout `[0,1,...,M-1]` (the evaluator computes both itself). Your
per-instance score is
```
min(1.0, 0.1 * H(order) / H_id)
```
so the identity layout scores exactly **0.1** and better layouts score higher. The final
report is the mean over all instances (a held-out set of harder, noisier streams is
included). There is no closed-form optimum — choosing the layout that maximizes reused
prefixes across many overlapping check families is a hard combinatorial ordering
problem, and several heuristics (frequency-first, weight-first, greedy construction)
trade off differently.

## Isolation

Your program is run in an OS sandbox: it reads only the public instance on stdin and
writes only the answer on stdout. It cannot see the evaluator's internals, the hit
totals, or anything else — only the layout you print is read back, and the evaluator
replays the stream itself to compute your score.
