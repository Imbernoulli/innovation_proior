# Watchtower Scan-Directive Prefix Cache

## Setting

A network of forest-fire watchtowers is coordinated by a central fire-control server.
Every operating period the server issues a **log of scan directives** in a fixed order.
Each directive is a **set of sensor check-clauses** drawn from a shared catalog of `C`
clause types (thermal, smoke plume, wind shear, humidity, camera pan, lightning strike,
regional relay, master check, ...).

The inference server evaluates a directive as an **ordered sequence** of its clauses and
memoizes results by **prefix** (a prefix KV-cache / trie). When a directive's leading
clauses — in the emitted order — exactly match a prefix that was already computed by an
**earlier** directive in the log, that leading work is served from cache (a **HIT**);
otherwise it must be recomputed (a **MISS**). Each clause `c` carries an integer compute
weight `weights[c]` (its token cost).

You get to fix **one global canonical clause order** — a permutation of all `C` clause
types. Every directive is then emitted with **its own** clauses sorted by this canonical
order, and run through the prefix cache in log order. Your job: choose the canonical order
that **maximizes the weighted prefix-cache hit rate** over the whole log.

Ordering matters because prefix sharing depends on *which clauses co-occur*, not only on
how often each clause appears: hoisting the clauses that many directives share to the
front lengthens the shared prefixes, but the best arrangement of co-occurring clause
clusters is a hard combinatorial choice with no easy optimum.

## Candidate program contract

Your program reads ONE JSON object (the public instance) from **stdin** and writes ONE
JSON object (your answer) to **stdout**. It is run in an isolated subprocess.

### Public instance (stdin)
```json
{
  "n_clauses": 10,                       // C: number of clause types (indices 0..C-1)
  "weights":  [1, 2, 1, 1, 2, 1, 2, 1, 8, 11],   // compute weight of each clause type
  "directives": [ [2, 8, 9], [1, 4, 8, 9], ... ] // log of directives, each a sorted set
                                                 //   of distinct clause indices, in log order
}
```

### Answer (stdout)
```json
{ "order": [9, 8, 2, 4, 1, 6, 0, 3, 5, 7] }   // a permutation of 0..C-1 (the canonical order)
```
`order` must be a list of length `C` that is exactly a permutation of `0,1,...,C-1`. Any
other shape (wrong length, repeats, out-of-range, non-integer, missing key) is rejected
and scores 0.

## How your order is scored (deterministic)

Given your `order`, define `pos[c]` = the rank of clause `c` in `order`. Process the
directives in log order over a shared, initially empty prefix cache:

1. Emit directive `d` as the sequence `seq` = its clauses sorted by `pos`.
2. Let `p` be the length of the longest prefix of `seq` that is already in the cache
   (the cache is prefix-closed). Clauses `seq[0..p-1]` are **hits**; `seq[p..]` are
   **misses**. Add `weights[c]` for every clause to `total`, and to `hit` for the hit
   clauses.
3. Insert every prefix of `seq` into the cache.

The instance objective is `obj = hit / total` (the weighted hit rate, in `[0,1)`).

Let `b` = the objective of the **identity order** `[0,1,...,C-1]` (computed by the
evaluator). Your per-instance normalized score is

```
r = min(1, 0.1 * obj / b)
```

so the identity order scores exactly `0.1`, and an order whose hit rate is `k×` the
identity baseline scores `min(1, 0.1·k)`. The reported **Ratio** is the mean of `r` over
all instances; a malformed / non-permutation answer scores `0`.

**Objective: maximize.** There is no wall-time in the score (the time limit is only a
safety gate). Scoring is fully deterministic over a fixed, seeded set of instances.
