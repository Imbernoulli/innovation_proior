# Vineyard Irrigation — Season-Weighted Moisture-Predicate Prefix-Cache Layout (Format B, isolated)

A vineyard's irrigation controller runs a fixed library of **irrigation decision
checks** over its vine blocks. Each check evaluates a **set of moisture / canopy
sensor predicates** (e.g. "block-7 root-zone moisture < 25%", "canopy temp > 30C",
"drip-line pressure OK"). Evaluating a predicate costs some *tokens* of work.

Over a whole growing **season** each distinct check is scheduled a fixed number of
times — its **season weight** `w_q`. Some checks run every irrigation cycle; others
only a handful of times all season.

The controller memoizes predicate results in a **prefix cache** (a trie). To use it,
the controller lays out **all** predicates in **one global order**, chosen once. Each
check then evaluates *its own* predicates **in that global order**. When a check's
leading run of predicates exactly matches the leading run of an **earlier** check, that
shared **prefix is a cache hit** — its token cost is reimbursed for free; only the
first divergent predicate onward must be paid. Because a check is run `w_q` times over
the season, its hit tokens are counted **`w_q` times**.

Because checks that belong to the same "family" share many core predicates, a smart
global layout makes those shared predicates line up at the **front** of every check,
so long prefixes are reused. A careless layout (for instance, one that lets a check's
rare, quirky predicates sort ahead of its shared core) shatters the prefixes and wastes
work. And because reuse is scored by season weight, the layout should favor the cores
of the **most frequently scheduled** check families.

**Your job: choose the global predicate order that maximizes the total season-weighted
cache-hit tokens over the whole fixed check library.**

## The prefix-cache model (exactly how you are scored)

You return a permutation `order` of the predicate indices `0..M-1`. Let `rank[a]` be the
position of predicate `a` in `order`. The checks are processed **in the given list
order**, maintaining a trie that starts empty:

For each check with predicate set `S` and season weight `w`:
1. Form its sequence `seq` = the predicates of `S` sorted **ascending by `rank`**.
2. Walk `seq` down the trie from the root. Let the **matched prefix** be the longest
   leading run of `seq` whose nodes already exist in the trie. Add
   `w * sum(weights[a] for a in matched_prefix)` to your hit total.
3. Insert the whole `seq` into the trie (creating nodes for the unmatched suffix).

Your objective is the **total season-weighted hit tokens** summed over every check.
Higher is better. (Identical predicate sets always share their whole sequence
regardless of layout; the layout only changes how much *overlapping-but-different*
checks share.)

## Public instance (stdin JSON)
```json
{
  "M":        <int>,               // number of distinct predicates (indices 0..M-1)
  "weights":  [<int>, ...],        // weights[a] = token cost of predicate a (length M)
  "queries":  [[a, b, ...], ...],  // the fixed check library; each entry is a check's
                                   //   predicate set (sorted list of distinct indices)
  "qweights": [<int>, ...]         // qweights[i] = season weight of queries[i] (>=1)
}
```

## Answer (stdout JSON)
```json
{ "order": [ <a permutation of 0..M-1> ] }
```
Requirements: `order` is a list of exactly `M` integers that is a permutation of
`0..M-1`. Any other shape (wrong length, repeats, out-of-range, non-integers, booleans)
is **infeasible** and scores **0**.

## Scoring

Let `H(order)` be the total season-weighted hit tokens your layout achieves, and `H_id`
the hit tokens of the identity layout `[0,1,...,M-1]` (the evaluator computes both
itself). Your per-instance score is
```
min(1.0, 0.1 * H(order) / H_id)
```
so the identity layout scores exactly **0.1** and better layouts score higher. The final
report is the mean over all instances (a held-out set of larger, noisier check libraries
is included). There is no closed-form optimum — choosing the layout that maximizes
season-weighted reused prefixes across many overlapping check families is a hard
combinatorial ordering problem, and several heuristics (frequency-first, season-weighted
frequency, greedy construction) trade off differently.

## Isolation

Your program is run in an OS sandbox: it reads only the public instance on stdin and
writes only the answer on stdout. It cannot see the evaluator's internals, the hit
totals, or anything else — only the layout you print is read back, and the evaluator
replays the weighted check library itself to compute your score.
