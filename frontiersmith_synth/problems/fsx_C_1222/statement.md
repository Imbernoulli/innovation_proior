# Reconciling Divergent Replica Logs Without Losing Writes

`R` replicas of a key-value store issued writes independently before a
reconciliation pass. Every write carries a **vector clock** of length `R`
(replica `r` increments its own coordinate on each write, and only carries a
neighbour's coordinate forward if it had already learned that state). For
writes `a`, `b` on the *same* key, `a` **happened before** `b` if `a`'s
clock is `<=` `b`'s clock in every coordinate (and they differ) — otherwise
they are **concurrent**: two operators disagreed about the key's value at
the same causal moment, and neither is "more correct".

For each key, its **frontier** is the writes to that key that no other write
to that key causally supersedes. Frontier size 1 = no real conflict. Size
`>= 2` = a genuine concurrent conflict. The frontier depends only on the
vector clocks, so any replica recomputing it gets the *same* set — that is
what lets replicas converge without talking again.

You must output, for **every** key, one decision:
* **Pick** one op touching the key; its value becomes final.
* **Merge**, if the key allows it: combine the *entire* frontier's values
  with the key's own merge operator (`SUM` or `MAX`, given in the input),
  spending that key's posted cost from a shared reconciliation **budget**.

Each op carries a positive integer **weight** — how much it matters that
this write's contribution survives. A pick earns its op's weight **only if
that op is in the key's frontier** (a causally superseded pick is feasible
but earns nothing — it was already overwritten by history). A merge earns
the *sum of the weights of every frontier member*, but only if the key
allows merging and its frontier has `>= 2` members. **Maximize total weight
earned**, without exceeding the budget.

## Input (stdin)
```
R K N BUDGET
mtype_1 ... mtype_K        (0=NONE, 1=SUM, 2=MAX)
mcost_1 ... mcost_K        (positive int, budget cost to merge that key)
replica key value weight timestamp vc_0 ... vc_{R-1}    (N such lines)
```
Op lines are 1-indexed by position (1..N). `0<=replica<R`, `0<=key<K`,
`weight>=1`, `vc` entries are non-negative integers. `timestamp` is a
wall-clock reading — informative, but **not** part of the causal history.

## Output (stdout)
Exactly `K` lines (any order), 4 whitespace-separated tokens each:
```
key_id mode ref value
```
`mode` is `P` (pick) or `M` (merge). For `P`, `ref` is the 1-indexed id of an
op touching `key_id`, and `value` must equal that op's value. For `M`, `ref`
is ignored (any integer) and `value` must equal the exact merge (`SUM`/`MAX`)
of every current frontier member's value for `key_id`. Every key `0..K-1`
must appear exactly once.

## Scoring
The checker recomputes every key's frontier from the vector clocks, checks
your output structurally (every key covered once, values match exactly,
merges legal, budget respected — **any** violation scores the whole
submission `Ratio: 0.0`), then computes your earned weight `F` as above. It
also builds its own unoptimized baseline `B`: if a frontier has size 1 it
must use that op (forced); if size `>= 2` it arbitrarily uses the
*minimum-weight* frontier member and never merges. `Ratio = min(1.0, 0.1 *
F / B)`.

## Structure you can exploit
A **timestamp says nothing about whether a write is causally current**.
Keeping the newest timestamp per key ("last-writer-wins") always converges,
but on a concurrent key it throws away every other frontier member's weight
for free, and nothing stops the largest timestamp from belonging to the
*least* important writer. The frontier — not the timestamp — is what every
replica can agree on independently. Once you have each key's true frontier,
deciding which multi-writer frontiers are worth spending shared budget to
merge (vs. taking the frontier's own best-weighted member for free) is a
knapsack over the keys.

## Constraints
`2 <= R <= 5`, `2 <= K <= 9`, `N <= 130`, `1 <= BUDGET <= 60`. Time limit 5s,
memory 512MB. Exact integer arithmetic; deterministic.

## Example
Key 7's frontier has op A (weight 12, value 30) and op B (weight 3, value
9), `mtype=SUM`, `mcost=6`. Picking A earns 12; merging (if budget allows)
earns 12+3=15 via `7 M -1 39`. (Illustration only — an unrelated toy key,
not part of any real test case.)
