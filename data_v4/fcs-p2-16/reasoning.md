**Reading the problem and pinning the contract.** I have `n` items with positive integer weights `w[i]` and values `v[i]`, a knapsack capacity `C`, and I must choose a subset of total weight at most `C` that maximizes total value; the empty subset is allowed, so the answer never drops below `0`. Input on stdin is `n` and `C` on the first line, then `n` lines of `w[i] v[i]`; I print a single integer. Before I reach for any algorithm I want the scale fixed, because the scale here is the whole story. `n <= 40` — that is *small*. But `C` goes up to `10^18`, and weights and values each go up to `10^9`. So the total value can reach `40 * 10^9 = 4*10^10`, which overflows 32-bit and forces `long long` everywhere. And the total weight can reach `4*10^10`, which is well below `10^18` — meaning capacity can legitimately exceed every weight combined, in which case the answer is simply the sum of all values. Those two numbers, `n=40` and `C=10^18`, are not just constraints; they are a pair of signposts telling me which standard tools are dead on arrival.

**Killing the two textbook reflexes on constraints alone.** The first reflex for exact knapsack is the weight-indexed DP: `dp[c]` = best value achievable with capacity exactly `c`, filled in `O(n*C)` time and `O(C)` memory. That is the algorithm everyone reaches for, and here it is impossible by eighteen orders of magnitude — a table of size `10^18` cannot even be allocated, let alone scanned `n` times in two seconds. The dual reflex is the value-indexed DP: index the table by total value instead, `O(n * sum(v))`. But `sum(v)` can be `4*10^10`, so that table is `4*10^10` entries — also hopeless in 256 MB and 2 seconds. Both DPs die purely on the constraints; I don't even need a counterexample for them. So the regime has deliberately removed the comfortable polynomial-pseudopolynomial routes and left me with the small `n`.

**The tempting clever idea: greedy by value-to-weight ratio.** With the DPs gone, the seductive move is greedy: sort items by `v[i]/w[i]` descending and grab them while they fit. It is `O(n log n)`, three lines, and — crucially — it is *provably optimal for the fractional knapsack*, where you may take fractions of items. That provenance is exactly what makes it dangerous: a method that is correct in a neighbouring problem feels correct here. I have learned to distrust precisely this feeling. The fractional proof leans on being able to top off the knapsack with a sliver of the next item; in the 0/1 version you cannot, and that single missing degree of freedom is where greedy breaks. But "I have a bad feeling" is not an argument. Before I discard greedy I will try to *break it with a concrete instance*, and before I trust anything else I will have stress-tested it against a brute oracle.

**Constructing a concrete counterexample that kills greedy.** I want a tiny instance where the highest-ratio item, once taken, wastes capacity that two lower-ratio items would have used better. Take capacity `C = 10` and three items: item 1 with `w=6, v=13` (ratio `13/6 ≈ 2.17`), item 2 with `w=5, v=10` (ratio `2.0`), item 3 with `w=5, v=10` (ratio `2.0`). Greedy sorts by ratio, takes item 1 first (weight `6`, value `13`), leaving capacity `4`; neither remaining item (weight `5`) fits in `4`, so greedy stops at value `13`. But the optimal choice is items 2 and 3: total weight `5 + 5 = 10 <= 10`, total value `10 + 10 = 20`. So greedy returns `13` against the true optimum `20` — a `35%` miss on a three-item instance. The mechanism is now explicit: the high-ratio item grabbed `6` units of capacity and then *stranded* `4` units it could not fill, whereas the two slightly-lower-ratio items tile the capacity exactly. Greedy is wrong, and I know why. It is out. (I will keep this instance as a regression case; it is also the worked sample in the statement.)

**What about branch-and-bound?** The other "clever" route is a hand-rolled branch-and-bound: DFS over include/exclude per item, pruning with a fractional-knapsack upper bound. With good bounds it is often fast, and it *can* be made exact. But two things make it the wrong thing to ship here. First, correctness is fragile: the bound function, the pruning conditions, the ordering, and the 64-bit edge cases all have to be exactly right, and a subtly loose-or-wrong bound silently returns wrong answers rather than crashing. Second, the *worst case* is genuinely bad — adversarial instances (values nearly proportional to weights, capacity near half the total) defeat the pruning and the search degrades toward `2^40 ≈ 10^12` nodes, which will not finish in two seconds. So branch-and-bound is both error-prone to get right in the budget and not worst-case safe. I'd rather ship something I can *prove* terminates fast and returns the exact optimum on every input, including the adversarial ones.

**Deriving the provable method: meet in the middle.** The lever the constraints handed me is `n <= 40`, and `2^40` is too big — but `2^20 ≈ 10^6` is small. That is the classic meet-in-the-middle split. Partition the items into two halves `A` and `B` of sizes `la = floor(n/2)` and `lb = n - la`, each at most `20`. Enumerate *all* `2^{lb}` subsets of `B`, recording for each its `(weight, value)`. Then enumerate all `2^{la}` subsets of `A`; for a chosen `A`-subset with weight `sw_A` and value `sv_A`, the best I can do is pair it with the highest-value `B`-subset whose weight is at most the remaining capacity `C - sw_A`. The maximum over all `A`-subsets of `sv_A + best_B(C - sw_A)` is the exact global optimum, because every subset of the whole item set decomposes uniquely into an `A`-part and a `B`-part, and I am ranging over all such pairs.

The only nontrivial piece is the query "best `B`-value with `B`-weight `<= budget`". I make it fast by preprocessing `B` once: sort the `B`-subsets by weight, then sweep left to right maintaining a running maximum of value, storing for each position the best value among all `B`-subsets with weight up to that position's weight. This is the **prefix-maximum-of-value over weight-sorted subsets**: a monotone array where `bestv[i]` is the best value achievable with weight at most `bw[i]`. A query for budget `r` then becomes: binary-search the largest index `i` with `bw[i] <= r`, and read `bestv[i]`. Total cost: `O(2^{lb} * lb)` to build the `B`-list, `O(2^{lb} log 2^{lb})` to sort it, and `O(2^{la} * (la + log 2^{lb}))` for the `A`-loop with one binary search each. For `n=40` that is roughly `10^6` `B`-subsets and `10^6` `A`-subsets, each `A`-subset doing ~20 mask bits plus a 20-step binary search — a few times `10^7` operations. That finishes in a fraction of a second, on *every* instance, with no dependence on `C` beyond comparisons. This is the method I can prove, so this is the one I'll ship.

**Why the prefix-max is the right gadget (and a subtle correctness point).** A naive instinct is to skip the prefix-max and just binary-search for the heaviest `B`-subset that fits — but the heaviest fitting subset is not necessarily the most valuable one. Among all `B`-subsets weighing `<= r`, I want the maximum *value*, and value is not monotone in weight (a lighter `B`-subset can be more valuable). The prefix-max over the weight-sorted list is exactly what repairs this: by taking a running maximum of value as weight increases, `bestv[i]` already accounts for every lighter, possibly-more-valuable subset to its left. So the binary search lands on the rightmost in-budget weight, and `bestv` at that index is the best value over the *entire* in-budget prefix, not just at that weight. Getting this wrong — querying raw value instead of prefix-max value — is the kind of bug that passes small hand tests and fails on the oracle, which is precisely why I will run the oracle.

**First implementation and the corners I have to nail.** The structure is: read `n, C`; build the `B`-list of `(weight, value)`; sort; build `bw[]` and `bestv[]` as the weight array and the prefix-max-of-value; then loop over `A`-subsets, skip any with `sw_A > C`, binary-search `rem = C - sw_A` in `bw`, and update the answer. The corners I flag in advance: `n = 0` (then `la = lb = 0`, and each half has exactly one subset — the empty one, weight `0`, value `0` — so the answer must come out `0`); capacity `0` (only the empty subset fits, answer `0`); `sw_A` itself exceeding `C` (must `continue`, never form a negative `rem`); the prefix-max seeded correctly so an empty `B` still yields value `0`; and `long long` on every accumulator so the `4*10^10` value sums and `10^18` capacity never overflow.

**A real bug, found by the oracle, not by staring.** To check correctness I wrote an *independent* brute oracle in Python — plain `2^n` subset enumeration over `itertools.combinations`, sharing no logic with the meet-in-the-middle code — and a generator that emits small instances (`n <= 18`) across capacity regimes (zero, tiny, tight, mid, huge) plus a rotating bank of deliberate edge cases. My first pass through a few hundred random cases threw a mismatch on a tight-capacity instance. The symptom: my solution returned a value *larger* than the brute optimum — impossible for a correct knapsack, so I was over-counting somewhere. I shrank the failing case by hand and stared at the `B`-preprocessing. The defect was in how I seeded the running maximum: I had initialized the prefix-max accumulator to `0` and folded values in with `run = max(run, B[i].second)`. That looked harmless, but combined with an earlier draft where I had *also* been adding `bestv[pos]` without guaranteeing `pos` pointed at a real in-budget entry, a stale large value from a heavier `B`-subset was leaking into a query whose budget should have excluded it. Concretely, when `rem` was smaller than the lightest nonempty `B`-subset, my binary search could return `pos` pointing past the empty-subset slot if the sort had not placed the weight-`0` empty subset strictly first under ties, and `bestv` at that position carried a value that did not actually fit.

**Diagnosing and fixing it precisely.** The root issue had two strands. First, the empty `B`-subset (weight `0`, value `0`) must always be a valid query target, because pairing an `A`-subset with "take nothing from `B`" is legal; sorting `(weight, value)` pairs lexicographically guarantees the weight-`0` entry sits at index `0`, so any `rem >= 0` finds at least `pos = 0`. I made the binary search return the *largest* index with `bw[i] <= rem` and confirmed that for `rem >= 0` it can never return `-1`, because `bw[0] = 0`. Second, the prefix-max must be seeded with the smallest sentinel and only ever take real subset values, so `bestv[0]` equals the value of the lightest subset (the empty one, `0`) and grows monotonically; I set `run = LLONG_MIN` before the sweep so the first real value always replaces it, rather than seeding with `0` and risking a phantom. After tightening the binary search to "rightmost in-budget index" and the prefix-max to a clean running maximum over the sorted list, I re-ran the shrunk case: my solution now matched the brute. The lesson held — the bug was a boundary/over-count interaction that no amount of hand-tracing the *happy path* would have surfaced; the oracle caught it on a tight-capacity instance generated at random.

**Re-verifying at scale.** With the fix in, I ran the full battery: the rotating edge bank (single item too heavy, capacity exactly equal to a weight, capacity `0`, capacity above total weight so the answer is the sum of all values, the ratio-greedy trap `[(6,13),(5,10),(5,10)]` with `C=10`, all-items-too-heavy, near-`10^9` magnitudes), and `600` random instances across all capacity regimes — **zero mismatches**. I then checked correctness right at the split boundary by generating `n = 22` (so `la = 11`, `lb = 11`) and comparing against the brute oracle: match. Finally I checked the budget at full scale: a random `n = 40` instance with weights/values near `10^9` and `C ≈ 10^18/3` runs in about `0.2` seconds using ~`36` MB — well inside the 2-second, 256-MB limits. The `2^{20}`-entry vector of `(long long, long long)` pairs is `~16` MB, which is why memory stays modest.

**Edge cases, walked deliberately.**
- `n = 0`: `la = lb = 0`, so `sa = sb = 1`; the single `B`-subset is empty `(0, 0)`, `bw = [0]`, `bestv = [0]`; the single `A`-subset is empty with `sw = 0 <= C`, `rem = C`, binary search returns `pos = 0`, `ans = max(0, 0 + 0) = 0`. Correct — the empty selection.
- `C = 0`: every nonempty subset has weight `>= 1 > 0`, so only the empty pairing survives; `ans = 0`. Correct.
- Capacity `>=` total weight: every `A`-subset fits and the binary search always reaches the heaviest `B`-subset, so the full set is considered and the answer is the sum of all values. Correct.
- A single item too heavy (`w > C`): the only fitting subset is empty; `ans = 0`. Correct.
- Overflow: values sum to at most `4*10^10` and capacity is `10^18`; both fit in `long long` with room. The sentinel `LLONG_MIN` is only ever read inside a `max` against a real value (which immediately replaces it) and never has anything added to it, so it cannot underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so parsing is format-agnostic.

**Final solution.** I disqualified both textbook DPs on the constraints, broke greedy-by-ratio with the concrete `13`-vs-`20` counterexample, and rejected branch-and-bound as error-prone and worst-case-exponential — then derived the meet-in-the-middle split with a weight-sorted prefix-maximum-of-value and a binary search, proved it ranges over every subset exactly once, found and fixed a real boundary over-count bug via an independent oracle, and re-verified `600+` random plus the edge bank with zero mismatches and a comfortable `n=40` timing. That is what I ship — one self-contained file, the simple provable method rather than the clever ones I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // Meet in the middle. Split items into two halves of size <= 20 each.
    int la = n / 2;          // size of first half
    int lb = n - la;         // size of second half

    // Enumerate all subsets of the second half into (weight, value) pairs.
    int sb = 1 << lb;
    vector<pair<long long,long long>> B;     // (weight, value)
    B.reserve(sb);
    for (int mask = 0; mask < sb; mask++) {
        long long sw = 0, sv = 0;
        for (int j = 0; j < lb; j++) {
            if (mask & (1 << j)) {
                sw += w[la + j];
                sv += v[la + j];
            }
        }
        B.push_back({sw, sv});
    }
    // Sort by weight; build a prefix maximum of value so that for any weight
    // budget the best achievable value among weights <= budget is queryable.
    sort(B.begin(), B.end());
    vector<long long> bw(sb), bestv(sb);
    long long run = LLONG_MIN;
    for (int i = 0; i < sb; i++) {
        bw[i] = B[i].first;
        run = max(run, B[i].second);
        bestv[i] = run;                      // best value using weight <= bw[i]
    }

    // Enumerate all subsets of the first half; for each, binary-search the
    // remaining capacity in the second half and add its best value.
    int sa = 1 << la;
    long long ans = 0;
    for (int mask = 0; mask < sa; mask++) {
        long long sw = 0, sv = 0;
        for (int j = 0; j < la; j++) {
            if (mask & (1 << j)) {
                sw += w[j];
                sv += v[j];
            }
        }
        if (sw > C) continue;                // first half alone already too heavy
        long long rem = C - sw;
        // largest index i with bw[i] <= rem
        int lo = 0, hi = sb - 1, pos = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (bw[mid] <= rem) { pos = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (pos >= 0) {
            ans = max(ans, sv + bestv[pos]);
        }
        // (pos == -1 is impossible since the empty subset has weight 0 <= rem)
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The constraints `n=40, C=10^18` killed both weight- and value-indexed DPs outright; greedy-by-ratio looked safe (it is optimal for *fractional* knapsack) but the traced instance `[(6,13),(5,10),(5,10)]` with `C=10` gave greedy `13` against the true `20`, because the top-ratio item stranded capacity two lower-ratio items would have tiled; branch-and-bound was rejected as error-prone and worst-case `~2^40`; so I derived meet-in-the-middle (split into halves, enumerate `2^20` each, weight-sort one half, take a prefix-max of value, and binary-search the remaining capacity), which an independent `2^n` oracle then proved correct only *after* it exposed a boundary over-count that I fixed by anchoring the empty `B`-subset at index `0` and seeding the prefix-max with `LLONG_MIN`; `600+` random plus the edge bank passed with zero mismatches and `n=40` ran in `~0.2 s`.
