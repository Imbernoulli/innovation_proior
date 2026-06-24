**Reading the problem and pinning the contract.** I have `n` stars in a line, numbered `1..n`, and `m` operations. Each operation `(l, r)` with `1 <= l <= r <= n` declares "make every star in the inclusive range `l, l+1, ..., r` mutually connected". Bridges are permanent, operations can overlap or repeat or be single points, and at the end I count connected clusters — a star that no operation ever links to a neighbour stays a singleton. I print one integer, the cluster count. Let me fix the scale first because it decides the algorithm class: `n, m <= 2*10^5`. The values are all small (indices bounded by `n <= 2*10^5`), so there is no overflow worry on the *data* — `int` holds every star index and every count comfortably. The danger here is not arithmetic width; it is **time**, and it is the **boundary** of the range union. I will keep that front of mind: this is a connectivity problem (disjoint-set union is the obvious vehicle) where the entire difficulty is "what exactly does range `[l, r]` connect, and how do I do it without quadratic blowup".

**Restating what a range actually connects, very concretely.** I want to be ruthless about the boundary before I write a line of code, because that is precisely where this problem is designed to bite. Range `[l, r]` makes stars `l, l+1, ..., r` one cluster. The minimal set of adjacency links that achieves that is the chain `l-(l+1), (l+1)-(l+2), ..., (r-1)-r`. How many links is that? The links are indexed by their left endpoint `i`, which ranges over `i = l, l+1, ..., r-1`. That is `r - l` links. So:
- A range with `l == r` (a single star) introduces `r - l = 0` links — it connects **nothing new**. Star `r` is "touched" but its cluster is unchanged.
- A range `[2, 4]` introduces links `2-3` and `3-4`: two links, three stars in one cluster. Good.
- Two ranges that *touch* at a shared endpoint, `[1,3]` then `[3,5]`, share star `3`, so the two chains fuse into one cluster `{1,2,3,4,5}`. They merge.
- Two ranges separated by a one-index gap, `[1,3]` then `[4,6]`, never share a star and there is no link `3-4`, so they stay two clusters `{1,2,3}` and `{4,5,6}`. They do **not** merge.

That last contrast — touch-at-endpoint merges, gap-of-one does not — is the inclusive/exclusive trap in plain sight. The number of links is `r - l`, the *left endpoints* run `l .. r-1`, and the topmost link is `(r-1) -> r`. Any code that creates a link with left endpoint `r` (i.e. `r -> r+1`) has reached **outside** the range and will wrongly merge `r` with `r+1`. I am writing that sentence down now so that when I implement the loop I can check it against this invariant: the largest left endpoint I ever pass to `unite` must be `r - 1`, never `r`.

**Candidate approaches.** Two routes, and I want the one I can both prove correct *and* prove fast.

- *Plain DSU, link every adjacent pair in every range.* For each `(l, r)`, run `for i in l..r-1: unite(i, i+1)`. This is transparently correct — it literally builds the chain I described. Its cost is `O(r - l)` per operation, so `O(sum of range lengths)` total. With `m` ranges each as long as `n`, that is `O(n*m) = 4*10^{10}` union calls on the worst input. That will time out by orders of magnitude under a 1-second limit. So this is my **reference oracle** for small inputs, not my submission.
- *DSU plus a "next unconsumed index" skip pointer.* The wasted work in the plain version is re-touching links that already exist: if `[1,100]` already chained `1..100`, then a later `[1,100]` re-calls `unite` 99 times for no effect. I want each adjacency link `i -> i+1` to be *created at most once over the entire run*. So I keep a second structure `nxt[i]` = "the smallest index `>= i` whose link to its successor has not yet been built". When I walk a range and build link `i -> i+1`, I mark `i` as consumed by setting `nxt[i] = i + 1`; a future walk hitting `i` will skip straight past it. Because each index is consumed at most once, the total number of `unite` calls across all operations is at most `n - 1`, and with path compression on both DSU and the skip pointer the whole thing is near-linear, `O((n + m) * alpha)`. This is the submission — *if* I can nail the loop boundary.

**Designing the skip-pointer walk.** I maintain two DSU-like arrays. `par[]` is ordinary connectivity union-find over the stars. `nxt[]` is a second union-find-ish pointer where `nxt[i]` chains forward to the next index that still needs its outgoing link built; a self-loop `nxt[i] == i` means "index `i` is still active (its link `i -> i+1` not yet built)". To process `(l, r)`:

1. Jump to the first active index at or above `l`: `i = fnext(l)` (follow `nxt` with path compression to the representative).
2. While `i < r`: build the link `unite(i, i+1)`; consume `i` by `nxt[i] = i + 1`; advance `i = fnext(i+1)`.
3. Stop when `i >= r`.

The crucial line is the loop guard. The links I must build have left endpoints `l, l+1, ..., r-1`. So the cursor `i` (a left endpoint) must be allowed to take the value `r - 1` but must **stop before `r`**. That is `while (i < r)`, strictly less. If I wrote `while (i <= r)` I would, when `i` reaches `r`, call `unite(r, r+1)` — building a link with left endpoint `r`, which violates the invariant I wrote down earlier (largest left endpoint must be `r - 1`) and wrongly glues `r` to `r+1`. I have a strong prior this `< r` vs `<= r` choice is exactly the off-by-one the problem is built around, so I will trace it explicitly once the code exists rather than trust my eyes.

**Deriving the final count.** After all operations, the number of clusters is the number of DSU roots among `1..n`: `comps = #{ v in 1..n : find(v) == v }`. Stars never touched are their own roots and correctly counted as singletons. This part has no boundary subtlety; the subtlety is all upstream.

**First implementation.** I size the arrays `n + 2` so that index `i + 1` is always valid even when `i = n` momentarily appears during a `fnext`, and I seed `nxt[i] = i` (every index active) and `par[i] = i`. Here is the first cut of the operation loop, written deliberately the way I might write it on autopilot, to see whether the boundary survives a trace:

```
int i = d.fnext(l);
while (i <= r) {              // <-- suspicious guard, written on purpose
    d.unite(i, i + 1);
    d.nxt[i] = i + 1;
    i = d.fnext(i + 1);
}
```

**First trace, hunting the off-by-one.** I pick the smallest input that can expose an inclusive/exclusive error. The cleanest is a single two-star range: `n = 3`, one operation `(1, 2)`. The correct answer is clusters `{1,2}` and `{3}`, i.e. `2`. Let me run my first cut by hand. Start: `par = [_,1,2,3]`, `nxt = [_,1,2,3]` (index 0 unused). `i = fnext(1) = 1`.

- Iteration, check `i <= r` -> `1 <= 2` true. `unite(1, 2)` -> now `1` and `2` share a root. `nxt[1] = 2` (consume `1`). `i = fnext(2) = 2`.
- Check `2 <= 2` true. `unite(2, 3)` -> **this glues star 2 to star 3**. `nxt[2] = 3`. `i = fnext(3) = 3`.
- Check `3 <= 2` false. Stop.

Now `1, 2, 3` are all one cluster, so the count is `1`. But the correct answer is `2`: range `(1,2)` must connect only stars `1` and `2`, never star `3`. The code is **wrong**, and it is wrong for exactly the reason I anticipated: with `i <= r`, when `i` reached `r = 2` I called `unite(2, 3)` — a link with left endpoint `2 = r`, reaching outside the inclusive range to drag in star `r + 1 = 3`. This is the off-by-one. The number of links a range adds is `r - l = 1` here, but the buggy loop added `2` links.

**The bug, stated precisely.** The loop builds one link per iteration whose left endpoint is the current `i`. The legal left endpoints are `l .. r-1`. The guard `i <= r` permits `i = r`, building the illegal link `r -> r+1`. The fix is the strict guard `i < r`, which lets `i` rise to `r - 1` (building the top legal link `(r-1) -> r`) and then stop. This is the inclusive-range / exclusive-loop mismatch: the range is inclusive in `r` for *stars*, but the loop over *left-endpoint links* must be exclusive in `r`.

**Fix and re-trace the same case.** Change the guard to `while (i < r)`. Re-run `n = 3`, op `(1, 2)`:

- `i = fnext(1) = 1`. Check `1 < 2` true. `unite(1, 2)`. `nxt[1] = 2`. `i = fnext(2) = 2`.
- Check `2 < 2` false. Stop.

Stars `1, 2` are one cluster, star `3` untouched. Roots among `1..3`: `find(1)` and `find(2)` share a root (say `2`), `find(3) = 3`. Count `= 2`. Correct. The single case that broke now passes, and it broke for precisely the cause I fixed.

**Second debug episode — a subtler boundary: does the skip pointer ever over-skip and miss a needed link?** Fixing `<` vs `<=` is necessary but I am not done, because the skip pointer introduces its own boundary hazard: when I advance `i = fnext(i + 1)`, could I land *past* `r` and skip a link that I still needed, or conversely land somewhere that builds a link I should not? Let me stress the case where ranges overlap so the skip pointer actually fires across operations. Take `n = 6` and two operations `(1, 3)` then `(3, 5)`. Correct answer: `(1,3)` -> `{1,2,3}`; `(3,5)` shares star `3` -> merges to `{1,2,3,4,5}`; star `6` alone. Clusters `{1,2,3,4,5}`, `{6}` = `2`.

Process `(1, 3)` with the fixed guard. `nxt = [_,1,2,3,4,5,6]`.
- `i = fnext(1) = 1`. `1 < 3` true. `unite(1,2)`. `nxt[1] = 2`. `i = fnext(2) = 2`.
- `2 < 3` true. `unite(2,3)`. `nxt[2] = 3`. `i = fnext(3) = 3`.
- `3 < 3` false. Stop. Cluster so far `{1,2,3}`. `nxt = [_,2,3,3,4,5,6]` (with `nxt[1]=2, nxt[2]=3`).

Process `(3, 5)`.
- `i = fnext(3)`. `nxt[3] = 3`, so `fnext(3) = 3`. `3 < 5` true. `unite(3,4)` -> star `4` joins the `{1,2,3}` cluster, now `{1,2,3,4}`. `nxt[3] = 4`. `i = fnext(4) = 4`.
- `4 < 5` true. `unite(4,5)` -> star `5` joins, `{1,2,3,4,5}`. `nxt[4] = 5`. `i = fnext(5) = 5`.
- `5 < 5` false. Stop.

Roots among `1..6`: `{1,2,3,4,5}` one root, `{6}` one root. Count `= 2`. Correct, and notice the skip pointer correctly did **not** rebuild links `1-2` or `2-3` during the second operation — it started the walk at `3`, the first still-active index `>= 3` happens to be `3` itself. Good. Now the harder sub-case: what if the second operation's `l` lands *inside* an already-consumed run? Take `n = 6`, ops `(1, 4)` then `(2, 6)`. After `(1,4)`: links `1-2,2-3,3-4` built, `nxt = [_,2,3,4,4,5,6]`, cluster `{1,2,3,4}`. Process `(2, 6)`: `i = fnext(2)`. `nxt[2] = 3 -> nxt[3] = 4 -> nxt[4] = 4`, so `fnext(2) = 4` (the skip pointer jumps over the already-built `2-3,3-4` links straight to the first active index `4`). Check `4 < 6` true. `unite(4,5)`, `nxt[4]=5`, `i=fnext(5)=5`. `5<6` true. `unite(5,6)`, `nxt[5]=6`, `i=fnext(6)=6`. `6<6` false. Stop. Cluster `{1,2,3,4,5,6}`, plus nothing else. Count `= 1`. Correct: range `(2,6)` together with the prior `(1,4)` spans everything. The skip pointer **started at 4, not at 2**, skipping the redundant links — and crucially it did not skip too far: it stopped exactly at the first index whose outgoing link was unbuilt. So the skip pointer's boundary is sound: `fnext` lands on the first active left endpoint, and the `< r` guard still bounds it.

**A third boundary trace — the single-point range must connect nothing.** This is the `l == r` corner, the place where the link count `r - l = 0` and any off-by-one would manufacture a phantom link. `n = 5`, op `(3, 3)`. `i = fnext(3) = 3`. Guard `3 < 3` is **false** immediately, so the loop body never runs — zero links built. Star `3`'s cluster is unchanged; every star is still its own root. Count `= 5`. Correct. Had I kept `i <= r`, the guard `3 <= 3` would have been true and I would have called `unite(3, 4)`, fabricating a link out of a zero-link operation. This single-point case is the most concentrated form of the same off-by-one, and `< r` handles it for free.

**Sanity-checking the derivation on the stated sample.** The contract's example is `n = 8`, ops `(2,4), (4,6), (7,7)`, expected `4`. Let me run the fixed logic. Process `(2,4)`: `i=fnext(2)=2`, `2<4` -> `unite(2,3)`, `nxt[2]=3`, `i=3`; `3<4` -> `unite(3,4)`, `nxt[3]=4`, `i=4`; `4<4` false. Cluster `{2,3,4}`. Process `(4,6)`: `i=fnext(4)=4`, `4<6` -> `unite(4,5)` (joins `{2,3,4}` to `5`), `nxt[4]=5`, `i=5`; `5<6` -> `unite(5,6)`, `nxt[5]=6`, `i=6`; `6<6` false. Cluster `{2,3,4,5,6}`. Process `(7,7)`: `i=fnext(7)=7`, `7<7` false — no link. Final clusters: `{1}`, `{2,3,4,5,6}`, `{7}`, `{8}`. Count `= 4`. Matches the expected output. The derivation and the sample agree.

**Edge cases, deliberately.**
- `m = 0` (no operations): the operation loop never runs; every star is its own root; count `= n`. With the `if (!(cin >> n >> m)) return 0;` guard, a present `n` with `m = 0` reads fine and outputs `n`. Correct.
- `n = 1`: there are no valid links at all (the only possible range is `(1,1)`, zero links). Count is always `1`. The array sizing `n + 2 = 3` keeps index `i + 1` in bounds even though it is never used. Correct.
- Repeated / fully-overlapping ranges: e.g. `(1,n)` applied `m` times. The first application builds all `n - 1` links; every subsequent application does `i = fnext(1)` which jumps straight to `fnext` of the first active index — but after the first pass *all* indices `1..n-1` are consumed, so `fnext(1)` resolves to `n`, and `n < n` is false: zero further work. So `m` copies of `(1,n)` cost `O(n + m*alpha)`, not `O(n*m)`. This is why the skip pointer matters, and it is the worst case for a naive solution.
- Array bounds / "overflow": no arithmetic overflow is possible — all indices are `<= n + 1 <= 2*10^5 + 1`, well within `int`. The only out-of-range risk is touching `nxt[n+1]` or `par[n+1]`; I size both at `n + 2` and initialise `nxt[n+1] = n+1`, `par[n+1] = n+1`, so any transient `fnext(n+1)` or `find(n+1)` is well-defined (and in fact never produces a wrong link because the `< r` guard with `r <= n` stops the cursor at `n - 1` at the latest, so `i + 1 <= n`). Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace so the line structure of the input is irrelevant to parsing.

**Complexity.** Each adjacency link `i -> i+1` is built at most once over the whole run (it is consumed by `nxt[i] = i + 1` and skipped thereafter), so total `unite` calls `<= n - 1`. Each `fnext` and `find` is amortised `O(alpha)` with path compression. Reading `m` operations is `O(m)`. Total `O((n + m) * alpha(n))` time, `O(n)` memory. On the `n = m = 2*10^5` adversarial input this runs in a couple of hundredths of a second.

**Final solution.** I disproved the brute-force route on *time* (quadratic on overlapping ranges), adopted the skip-pointer DSU, and then spent my effort where the problem is sharp: the loop boundary. A trace of the two-star range `(1,2)` on `n=3` caught the `i <= r` off-by-one (it dragged in star `r+1`, giving `1` cluster instead of `2`); the fix is the strict guard `i < r`; I then re-verified the skip pointer never over-skips (overlapping `(1,3),(3,5)` and `(1,4),(2,6)` cases) and that the single-point range `(l,l)` builds zero links, and confirmed the stated sample gives `4`. Here is the file I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

// DSU over stars 1..n, with a "next free" skip pointer to union a whole
// contiguous range in near-linear amortized total time.
struct DSU {
    vector<int> par;     // connectivity parent
    vector<int> nxt;     // nxt[i] = smallest index >= i not yet "consumed" by a range-union step
    DSU(int n) : par(n + 2), nxt(n + 2) {
        for (int i = 0; i <= n + 1; i++) { par[i] = i; nxt[i] = i; }
    }
    int find(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a != b) par[a] = b;
    }
    // skip-pointer find over nxt[]
    int fnext(int x) { while (nxt[x] != x) { nxt[x] = nxt[nxt[x]]; x = nxt[x]; } return x; }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    DSU d(n);

    for (int op = 0; op < m; op++) {
        int l, r;
        cin >> l >> r;
        // Union all stars in [l, r] together. We walk from l upward using the
        // skip pointer nxt[] so each star is "consumed" (its nxt advanced) at
        // most once across the whole run, giving near-linear total work.
        //
        // Boundary: a range [l, r] has (r - l) adjacent links l-l+1, ..., (r-1)-r.
        // We start at the first not-yet-consumed index >= l, and keep linking it
        // to its successor while we are still strictly below r. The cursor i must
        // never pass r, and the LAST link we add is (r-1) -> r, so the loop runs
        // while i < r (NOT i <= r): linking i to i+1 when i == r would touch r+1,
        // which is outside the range.
        int i = d.fnext(l);
        while (i < r) {
            d.unite(i, i + 1);   // connect star i with star i+1
            d.nxt[i] = i + 1;    // i is consumed; future range-walks skip past it
            i = d.fnext(i + 1);  // jump to next not-yet-consumed index
        }
    }

    // Count connected components among all stars 1..n.
    int comps = 0;
    for (int v = 1; v <= n; v++)
        if (d.find(v) == v) comps++;

    cout << comps << "\n";
    return 0;
}
```

**Causal recap.** A range `[l, r]` connects the inclusive set of stars `l..r` but is realised by the *exclusive* chain of `r - l` links with left endpoints `l..r-1`; mixing those two boundaries is the trap. Plain DSU that links every pair is correct but quadratic on overlapping ranges, so I switched to a skip-pointer DSU where each link is built once. My first loop used `while (i <= r)`, and a hand-trace of `(1,2)` on `n=3` returned `1` cluster instead of `2` because at `i = r` it built the out-of-range link `r -> r+1`; the strict guard `i < r` fixes it, stops the cursor at `r - 1`, makes single-point ranges build zero links, and — re-verified on overlapping `(1,3)/(3,5)` and `(1,4)/(2,6)` plus the `n=8` sample — yields the correct cluster count in near-linear time.
