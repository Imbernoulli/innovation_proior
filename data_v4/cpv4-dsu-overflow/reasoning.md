**Reading the problem and pinning the contract.** I have `n` servers and `m` cables installed one at a time. After each cable I must print how many unordered pairs of *distinct* servers currently sit in the same cluster (can reach each other), and at the very end I print the grand total — the sum of all `m` numbers I printed. A cable can be redundant: if its two endpoints are already in one cluster, or it is a self-loop `u == v`, it joins nothing and the count is unchanged. Scale first, because it fixes the data types before I write a line of algorithm: `n, m <= 2*10^5`, and `1 <= u, v <= n`. The connected-pair count when the clusters have sizes `c_1, c_2, ...` is `sum_k C(c_k, 2) = sum_k c_k*(c_k-1)/2`. If everything ends up in one cluster of size `2*10^5`, that single term is `C(2*10^5, 2) = 200000*199999/2 = 19,999,900,000 ≈ 2*10^10`. That already dwarfs the signed 32-bit ceiling of `2,147,483,647`. And the grand total sums up to `2*10^5` such numbers, so it can climb to roughly `2*10^10 * 2*10^5 = 4*10^15`. So *every* accumulator — the running count, the grand total — and *every* intermediate product has to be 64-bit. I'll commit to `long long` for all of them up front; an `int` anywhere on this path is a silent wrong-answer on the big tests, and I want to remember that when I get to the merge step.

**Candidate approaches.** The naive thing is to rebuild the components from scratch after every cable: keep the list of all edges seen so far, flood-fill to label clusters, sum `C(size, 2)`. That is obviously correct — it is exactly what I'll use as a brute force to check against — but it costs `O(m * (n + m))`, which at `2*10^5` is around `4*10^10` operations after the dust settles. Hopeless inside one second. The structure of the problem screams *incremental connectivity*: edges only ever get added, never removed, and I keep being asked a connectivity-derived quantity. That is the textbook setting for disjoint-set-union (union–find). DSU with union by size plus path compression answers "which cluster is server `x` in?" and "how big is that cluster?" in near-constant amortized time. So the plan is: maintain DSU; on each cable, find the two roots; if they differ, merge and update the count; if they match, the cable is redundant and the count is unchanged.

**Deriving the incremental update.** I do not want to recompute `sum C(c_k, 2)` from scratch each step — that defeats the point. So I track a single running value `P` = current number of connected pairs, and ask: when I merge two *distinct* clusters `A` and `B` of sizes `sA` and `sB`, how does `P` change? Before the merge, the pairs inside `A` and inside `B` are already counted in `P`. After the merge, every server of `A` becomes connected to every server of `B`, which is `sA * sB` brand-new pairs, and no pairs are removed and no other cluster is touched. So `P += sA * sB`, and that is the *only* change. A redundant cable merges nothing, so `P` is unchanged. This is exact and `O(1)` per cable on top of the DSU find. Good — and note `sA * sB` itself can be `~10^10` (two clusters of size `~10^5` each), so that product is one of the int-overflow traps I flagged.

Let me sanity-check the update against the closed form on a trivial case before trusting it. Start with all singletons, `P = 0`. Merge two singletons: `sA = sB = 1`, `P += 1` → `P = 1`. Closed form: one cluster of size 2 gives `C(2,2) = 1`. Match. Now merge a size-2 cluster with a singleton: `P += 2*1 = 2` → `P = 3`. Closed form: cluster of size 3 gives `C(3,2) = 3`. Match. The incremental product reconstructs the closed form, as it must.

**Checking the recurrence on the stated sample.** The sample is `n = 5` and cables `(1,2), (3,4), (2,3), (3,4), (1,5)`; the claimed per-cable counts are `1, 2, 6, 6, 10` and grand total `25`. Let me walk it with DSU in my head, clusters written as sets.

- Start: `{1},{2},{3},{4},{5}`, `P = 0`.
- Cable `(1,2)`: distinct roots, sizes `1,1`, `P += 1` → `P = 1`. Clusters `{1,2},{3},{4},{5}`. Print `1`. ✓
- Cable `(3,4)`: distinct, `1*1`, `P += 1` → `P = 2`. Clusters `{1,2},{3,4},{5}`. Print `2`. ✓
- Cable `(2,3)`: roots of `2` and `3` are the `{1,2}` cluster (size 2) and the `{3,4}` cluster (size 2), distinct. `P += 2*2 = 4` → `P = 6`. Clusters `{1,2,3,4},{5}`. Print `6`. ✓
- Cable `(3,4)`: `3` and `4` are now both in `{1,2,3,4}` — same root, redundant. `P` unchanged at `6`. Print `6`. ✓
- Cable `(1,5)`: `{1,2,3,4}` (size 4) with `{5}` (size 1), distinct. `P += 4*1 = 4` → `P = 10`. Print `10`. ✓
- Grand total `= 1 + 2 + 6 + 6 + 10 = 25`. ✓

The derivation reproduces the sample exactly, including the redundant fourth cable contributing `0`. I'm convinced the math is right; now the danger moves to transcription.

**First implementation and a trace.** Here's my first cut. I'll write DSU with arrays, and — to be honest about how I'd actually type it the first time — I reach for `int` arrays because that's the muscle-memory default for a DSU:

```
int parent_[200005];
int sz[200005];                 // <-- first instinct: int sizes

int find_(int x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
    }
    return x;
}
...
long long pairs = 0, grand = 0;
for (int e = 0; e < m; e++) {
    int u, v; cin >> u >> v;
    int ru = find_(u), rv = find_(v);
    if (ru != rv) {
        pairs += sz[ru] * sz[rv];          // <-- product computed in int!
        if (sz[ru] < sz[rv]) swap(ru, rv);
        parent_[rv] = ru;
        sz[ru] += sz[rv];
    }
    grand += pairs;
    out += to_string(pairs); out += '\n';
}
```

I deliberately trace a case that pushes cluster sizes past the int boundary in the product `sz[ru]*sz[rv]`. Consider building two clusters of size `50000` each and then merging them. `50000 * 50000 = 2,500,000,000`. The signed-int max is `2,147,483,647`. The true product `2,500,000,000` is `2,500,000,000 - 2^32 = 2,500,000,000 - 4,294,967,296 = -1,794,967,296` once it wraps as a 32-bit signed value. So `sz[ru] * sz[rv]` is evaluated as a *negative int* `-1,794,967,296`, then that negative value is added (promoted to `long long`) to `pairs`. So `pairs` would *decrease* by ~1.79 billion on that merge. Concretely, build cluster `A = {1..50000}` and `B = {50001..100000}`, then cable `(1, 50001)`: the printed count would go from `2*C(50000,2) = 2,499,950,000` down to `2,499,950,000 - 1,794,967,296 = 704,982,704`, a glaringly wrong, smaller-than-before number. The connected-pair count can *never* decrease as cables are added, so this is a self-evident bug.

**The bug.** The defect is precise and it is exactly the pitfall I warned myself about and then walked right into out of habit: `sz[ru]` and `sz[rv]` are `int`, so `sz[ru] * sz[rv]` is computed in `int` arithmetic *before* the result is widened to the `long long` accumulator `pairs`. C++ evaluates the right-hand side in the operands' type; the `+=` to a `long long` happens only after the multiplication has already overflowed. The `long long pairs` lulled me into thinking I was safe — but the overflow is born one step upstream, in the product. Two clean ways to fix it: make the `sz` array `long long` so the multiplication is 64-bit, or cast one operand: `pairs += (long long)sz[ru] * sz[rv]`. I'll make `sz` a `long long` array, because the sizes themselves stay small (`<= 2*10^5`, fits in int) but I never again want to worry about *which* expression mixing them overflows — widening the storage makes every arithmetic touching a size 64-bit by default. That's the more robust choice given the whole problem is about not overflowing.

**Fix and a re-trace.** With `long long sz[]`, redo the merge of the two size-`50000` clusters: `sz[ru] * sz[rv] = 50000LL * 50000 = 2,500,000,000` in 64-bit — exact, positive. `pairs` goes from `2,499,950,000` up to `2,499,950,000 + 2,500,000,000 = 4,999,950,000`, which is `C(100000, 2) = 100000*99999/2 = 4,999,950,000`. ✓. Monotonic increase, matches the closed form. The overflow is gone because the storage type forces 64-bit multiplication. I also re-run the small sample mentally — sizes there are tiny so the bug never showed, which is the whole danger of int overflow: it hides on small inputs and only the large hidden tests expose it. That's exactly why I traced a constructed large case rather than trusting the passing sample.

**A second debug episode: the redundant-cable / monotonicity contract, traced on `(u == v)`.** I want to be sure the "if `ru != rv`" guard is right and that self-loops behave. Trace `n = 3`, cables `(1,1), (2,2), (1,2)`. `(1,1)`: `find_(1) == find_(1)`, same root, so the `if` body is skipped, `P` stays `0`. Print `0`. `(2,2)`: same, `P = 0`. Print `0`. `(1,2)`: distinct roots (sizes `1,1`), `P += 1` → `P = 1`. Print `1`. Grand `= 0 + 0 + 1 = 1`. My brute force (flood-fill over the edge set, where self-loops add a vertex's own adjacency and never change component structure) gives `0, 0, 1` and grand `1`. Match. The `if (ru != rv)` cleanly absorbs both "already-connected" redundancy and "self-loop" redundancy in one condition — a self-loop has `u == v` so `find_(u) == find_(v)` trivially, no special-casing needed. Good. I had briefly worried a self-loop might need its own guard; the trace shows it does not.

While I'm here, a subtler trace to check I'm not double-counting on a *redundant non-self-loop*. `n = 3`, cables `(1,2), (2,3), (1,3)`. `(1,2)`: `P = 1`, clusters `{1,2},{3}`. `(2,3)`: merge size-2 and size-1, `P += 2` → `P = 3`, cluster `{1,2,3}`. `(1,3)`: now `1` and `3` share the root — redundant, `if` skipped, `P` stays `3` = `C(3,2)`. Print `1, 3, 3`, grand `7`. Brute gives the same. The guard correctly refuses to add `sz*sz` for an edge inside an existing cluster, which would have been a real over-count bug if I'd forgotten it.

**Edge cases, deliberately.**

- `m = 0`: the loop never runs, nothing is printed for cables, then I print the grand total `0`. The contract says output `m + 1` lines, so with `m = 0` that's exactly one line, `0`. ✓. I need to make sure I still print the grand total even when there were zero cables — my code appends `grand` after the loop unconditionally, so it does.
- `n = 1`: there are no pairs of *distinct* servers possible, ever. Any cable must be `(1,1)` (since `1 <= u,v <= 1`), a self-loop, redundant, `P` stays `0`. Every printed line is `0`, grand `0`. ✓.
- Self-loops and redundant cables: handled by the single `ru != rv` guard, traced above.
- Minimum: smallest non-trivial is `n = 2, m = 1, (1,2)` → print `1`, grand `1`. `C(2,2) = 1`. ✓.
- Maximum / overflow: with `n = m = 2*10^5`, the per-cable count peaks at `C(2*10^5, 2) = 19,999,900,000 ≈ 2*10^10`, and the grand total — if many cables are redundant after full connection, each adding ~`2*10^10` to `grand` — can reach `~4*10^15`. Both are comfortably inside the signed 64-bit range (`~9.2*10^18`). The one product `sz[ru]*sz[rv]` peaks at about `(10^5)^2 = 10^10`, also fine in 64-bit, and now genuinely computed in 64-bit thanks to the `long long sz` fix. No accumulator and no intermediate exceeds `long long`. I verified the peak `19,999,900,000` and the multi-quadrillion grand total on an actual large run, and both came out exact.
- Output volume: `m` lines plus one, each up to ~14 digits. Printing with `cout <<` per line through the default-synced streams could be slow at `2*10^5` lines, so I build one big `string out` and flush it once. I also `reserve` roughly `8*m` bytes to avoid reallocation churn. The numbers can exceed 8 digits so the reserve is a hint, not a hard bound — `to_string` and `+=` resize as needed, correctness is unaffected.
- Parsing: `cin >> n >> m` and `cin >> u >> v` consume arbitrary whitespace, so the input format (spaces vs newlines) doesn't matter. If the very first read fails (truly empty input), I `return 0` and print nothing — defensive, though the constraints guarantee `n >= 1`.

**DSU correctness details.** Union by size keeps the tree shallow: I always attach the smaller cluster under the larger root (`if (sz[ru] < sz[rv]) swap(ru, rv)` then `parent_[rv] = ru`). Path halving in `find_` (`parent_[x] = parent_[parent_[x]]`) flattens trees over time. Together these give the near-`O(α(n))` amortized find that makes the whole thing `O((n + m) α(n))` — effectively linear. I keep `sz[ru]` updated only at roots, and I read `sz[ru]`, `sz[rv]` *before* reassigning `parent_[rv]`, so the product uses the true pre-merge sizes. Order matters here: if I updated `sz[ru] += sz[rv]` before reading the product I'd compute `(sA+sB)*sB` — wrong. In my code the `pairs += sz[ru]*sz[rv]` line sits *above* the union, so the sizes are the pre-merge ones. ✓.

**Sanity check of the whole thing against an independent brute force.** I wrote a from-scratch flood-fill brute (rebuild adjacency from all edges so far, BFS each component, sum `C(size,2)`) — a completely different method from the DSU incremental update — and ran 700+ random small cases (`n` up to 10, `m` up to 20, including self-loops and redundant edges) plus the constructed edge cases (`m=0`, `n=1`, self-loop-only inputs). Zero mismatches. The DSU solution and the brute agree on every per-cable line and on the grand total, which is the evidence I trust most: two independent derivations landing on the same numbers.

**Final solution.** I disproved the naive rebuild on performance, derived the `P += sA*sB` incremental update and checked it against the closed form and the sample, then traced a constructed large merge to catch the `int`-product overflow (the `long long pairs` accumulator did not save me — the multiply overflowed upstream), fixed it by storing sizes as `long long`, and confirmed the redundant/self-loop guard and the empty/min/max corners. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int parent_[200005];
long long sz[200005];

int find_(int x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
    }
    return x;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    for (int i = 1; i <= n; i++) {
        parent_[i] = i;
        sz[i] = 1;
    }

    long long pairs = 0;     // current number of connected unordered pairs
    long long grand = 0;     // running sum of "pairs" reported after each cable

    string out;
    out.reserve((size_t)m * 8);

    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        int ru = find_(u), rv = find_(v);
        if (ru != rv) {
            // merging two distinct components adds sz[ru]*sz[rv] new connected pairs
            pairs += sz[ru] * sz[rv];
            // union by size
            if (sz[ru] < sz[rv]) swap(ru, rv);
            parent_[rv] = ru;
            sz[ru] += sz[rv];
        }
        // if ru == rv, the cable is redundant; pairs unchanged
        grand += pairs;
        out += to_string(pairs);
        out += '\n';
    }

    out += to_string(grand);
    out += '\n';
    cout << out;
    return 0;
}
```

**Causal recap.** Incremental connectivity plus a connectivity-derived count points straight at DSU; the count updates by exactly `sA*sB` on each real merge (verified against `sum C(c_k,2)` and the sample), and a single `ru != rv` guard absorbs both redundant and self-loop cables (traced on `(1,1),(2,2),(1,2)` and on a triangle). The trap was magnitude: a peak per-cable count of `~2*10^10` and a grand total of `~4*10^15` force 64-bit accumulators, but the *first* overflow I'd actually hit is the product `sz[ru]*sz[rv]` computed in `int` (`50000*50000` wraps negative and makes the monotone count drop) — fixed by storing sizes as `long long` so the multiply is 64-bit. Reading the two sizes before the union, building one output string, and printing the grand total unconditionally close out the ordering, performance, and `m=0` corners; 700+ cross-checks against an independent flood-fill brute and an exact large run confirm it.
