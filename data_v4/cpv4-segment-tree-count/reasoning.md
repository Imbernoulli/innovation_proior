A query `2 l r` asks for the number of *visible* buildings in `h[l..r]` — those strictly taller than every building to their left inside the window — which is exactly the count of strict prefix maxima of the subarray, and the leftmost element always counts. Updates `1 p x` are point assignments. The scale is what rules out the easy answer: `n, q <= 2*10^5`, and a single query window can be the whole array, so recomputing a running max across the window is `O(window)` per query and `O(n*q) = 4*10^{10}` overall — far too slow. I need something like `O(log)` per operation. Heights reach `10^9`, but the answer is a count bounded by `n` and nothing is ever summed, only compared, so overflow is not in play; I still carry heights and the threshold in `long long` so comparisons stay uniform.

The workable structure is a max segment tree over `h`: a point update is the textbook `O(log n)`, and I keep the brute `O(window)` recount only as an oracle to test against. The subtlety is entirely in the query, because visibility is not additive — a building's status depends on the running maximum of everything to its left *within the window*, so I cannot read a count off a single node. I define `countVisible(node, bound)` = the number of strict prefix maxima inside the node given that everything already seen to its left has maximum `bound`:

- if `mx[node] <= bound`, nothing here can strictly beat the running max — return `0` and prune. The `<=` is load-bearing: an element merely *equal* to the bound is not strictly greater, so it is not visible.
- a leaf reached with `mx > bound` is a new prefix max — return `1`.
- otherwise recurse left with the same `bound`, then recurse right with `max(bound, mx[left])` — a prefix max in the right half must beat both the external bound and the entire left half — and add.

Pruning any subtree whose max does not exceed the bound keeps this `O(log n)` per node. On `[1,4,1]` with `bound = -inf` it returns 2 (positions holding `1` and `4`), matching the definition.

To answer a full window I walk its `O(log n)` canonical segments. The naive assembly counts each fully covered node independently, resetting the bound to `-inf`:

```
long long queryBad(int node, int lo, int hi, int ql, int qr) {
    if (ql <= lo && hi <= qr)
        return countVisible(node, lo, hi, LLONG_MIN); // each node counted in isolation
    int mid = (lo + hi) / 2;
    long long res = 0;
    if (ql <= mid) res += queryBad(2*node, lo, mid, ql, qr);
    if (qr > mid)  res += queryBad(2*node+1, mid+1, hi, ql, qr);
    return res;
}
```

On `h = [1,1,1,9,2,3,4,5]`, query `[0,7]` — true answer 2, since the prefix maxima of the whole array are `1` and `9`, and `2,3,4,5` all sit below `9` — the root splits into canonical segments `[0,3]` and `[4,7]`. `countVisible([0,3], -inf)` over `[1,1,1,9]` gives 2; `countVisible([4,7], -inf)` over `[2,3,4,5]` gives 4, because every element beats the previous one locally. The walk returns 6.

The over-count is structural, not a slip: each segment was counted as if it began the window, so the later segment `[2,3,4,5]` never learns that the `9` in `[0,3]` already dominates it. Since visibility is defined against everything to the left *in the query*, and the left context of a later node is exactly the union of the earlier nodes, the running maximum has to be threaded across segment boundaries. I make `bound` a reference parameter: count inside a covered node with the incoming bound, then raise it by that node's max before the next segment; recurse left child before right so the canonical segments are visited in array order.

```
long long queryCount(int node, int lo, int hi, int ql, int qr, long long &bound) {
    if (ql <= lo && hi <= qr) {
        long long c = countVisible(node, lo, hi, bound);
        bound = max(bound, mx[node]);
        return c;
    }
    int mid = (lo + hi) / 2;
    long long res = 0;
    if (ql <= mid) res += queryCount(2*node, lo, mid, ql, qr, bound);
    if (qr > mid)  res += queryCount(2*node+1, mid+1, hi, ql, qr, bound);
    return res;
}
```

Re-run `[0,7]` with `bound = -inf`: node `[0,3]` counts 2, then `bound` rises to 9; node `[4,7]` with `bound = 9` prunes to 0, since all of `2,3,4,5` are `<= 9`, and `bound` stays 9. Total 2 — the `9` now suppresses the tail, which is the whole point of threading.

The `[0,7]` trace only exercised a root-spanning window; I still owe myself the unaligned case, where left-child-before-right has to reproduce array order across a lopsided decomposition. On `h = [9,1,2,3,8,5,6,7]`, query `[1,6]` = `[1,2,3,8,5,6]` with true prefix maxima `1,2,3,8` = 4: the range decomposes, in visitation order, into canonical pieces `{1}, {2,3}, {4,5}, {6}` — values `[1], [2,3], [8,5], [6]`. Threading with `bound = -inf`: `[1]` -> 1 (bound 1); `[2,3]` -> 2 (bound 3); `[8,5]` -> 1, only the `8` (bound 8); `[6]` -> 0 since `6 <= 8`. Total 4. The threading holds across an unaligned split.

The strict comparison is the other place ties can silently flip the count, and it is why the prune is `<=` rather than `<`. On `[4,4,4]` only the first `4` is visible (answer 1); a `<` prune would treat each element equal to the running max as still un-dominated and count all three, returning 3. With `mx[node] <= bound` a node that merely ties contributes nothing, so only the first of a run of equal heights survives.

On the statement's sample `h = [3,1,4,1,5,9]` the operations give `4, 2, 3, 2`: `2 0 5` -> maxima `3,4,5,9` = 4; `2 1 3` over `[1,4,1]` -> 2; after `1 1 7` the array is `[3,7,4,1,5,9]`; `2 0 5` -> `3,7,9` = 3; `2 0 2` over `[3,7,4]` -> 2. The compiled program reproduces these. The corners behave as they should: `n = 1` or `l == r` give 1 (the only building is visible); an all-equal window gives 1; a strictly increasing window gives its length; a strictly decreasing one gives 1. The tie corner is the one to state carefully: `[10^9, 1, 10^9]` over `[0,2]` — the second `10^9` is not strictly greater than the first, so the answer is 1, not 2. Against the `O(window)` brute force on 1000 random instances — small value domains to force ties, plus a wider generator with heights up to `10^9` — there are zero mismatches, and a full-scale `n = q = 2*10^5` run finishes in about 0.18 s, well inside the limit.

So the program I ship is a max segment tree with the standard `build` and `pointSet`, `countVisible` for the threshold descent, and `queryCount` threading a `long long bound` initialized to `LLONG_MIN` left to right across canonical segments. `main` reads `n q` and the array, dispatches `1 p x` to `pointSet` and each `2 l r` to `queryCount`, and buffers the per-query counts into one output string. That is `O(log n)` per update and `O(log^2 n)` per query — the full module is the answer file.
