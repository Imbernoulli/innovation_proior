I have a text `t[0..n-1]` and a pattern `p[0..m-1]` of integers that can be negative, zero, or positive, and the pattern *resonates* at start `i` when some single offset `c` makes `t[i+j] = p[j] + c` for all `j` in `[0, m-1]` — the window equals the pattern shifted up or down by one fixed amount. Two things about the numbers decide the shape of the solution before any algorithm. Nothing here gets accumulated, but the instant I subtract two entries, `t[i+1] - t[i]` can reach `10^9 - (-10^9) = 2*10^9`, and the 32-bit signed ceiling is about `2.147*10^9` — so a single difference *just* fits in `int`, right at the edge and one nudge from overflow. I put everything in `long long` and stop worrying about the type. The other observation is the lever for a fast algorithm: because `c` is free per position, absolute heights are irrelevant and only the *shape* of the sequence matters.

That shape is captured by consecutive differences. Subtracting consecutive equations, `t[i+j+1] - t[i+j] = (p[j+1] + c) - (p[j] + c) = p[j+1] - p[j]`; the `c` cancels. So a resonance at `i` forces the length-`(m-1)` difference sequence of the window to equal the difference sequence of the pattern. The converse holds too, which is what makes the reduction usable: if all `m-1` differences match, set `c = t[i] - p[0]` and induct upward — `t[i+j+1] = t[i+j] + (t[i+j+1]-t[i+j]) = (p[j]+c) + (p[j+1]-p[j]) = p[j+1] + c`. Matching differences is therefore *equivalent* to resonance, not merely necessary, so nothing is lost. The problem becomes: find every occurrence of the pattern-difference sequence `pd` (length `m-1`) as a contiguous block of the text-difference sequence `td` (length `n-1`) — exact substring search over an integer alphabet, which KMP does in `O(n+m)`.

The alternative, a naive windowed check that fixes `c = t[i] - p[0]` and verifies all `m` entries, is obviously correct and makes a fine brute-force oracle, but an adversary who makes almost every window almost-match — a near-constant text against a constant pattern — drags the inner loop out to `2*10^5 * 2*10^5 = 4*10^10` operations, far past a one-second budget. So KMP is the shipping path and the quadratic version survives only as a comparison oracle.

The one place the reduction can quietly go wrong is the index bookkeeping. KMP scans `td` (length `tn = n-1`) and reports a full match when the prefix counter `k` reaches `pm = m-1`. A match ending at `td` index `e` covers `td[e-pm+1 .. e]`, so it starts at difference-array index `s = e - pm + 1`. That block of `pm` differences describes the originals `t[s], t[s+1], ..., t[s+pm]` — `pm+1 = m` consecutive values from original index `s`. So the difference-array start index *is* the text start index; there is no shift between the two coordinate systems. On the given sample `t = [-1,0,-2,2,3,1]`, `p = [5,6,4]`: `pd = [1,-2]`, `td = [1,-2,4,1,-2]`, and `[1,-2]` occurs ending at `e=1` (start `1-2+1 = 0`) and `e=4` (start `4-2+1 = 3`) — positions `0 3`, matching the stated answer.

The generic KMP path silently assumes `m >= 2`, and the two short pattern lengths break it in different ways, so both need explicit handling before any difference array is built. For `m = 0` (empty pattern) the path computes `pm = m-1 = -1` and then `vector<long long> pd(pm)` — a vector of size `-1`, which as a `size_t` is about `1.8*10^19`, an instant `bad_alloc`. And an empty pattern has no window to anchor `c` against, so its answer is `0` positions regardless of the text. I intercept it at the very top:

```
if (m == 0) { cout << 0 << "\n" << "\n"; return 0; }   // count, then empty positions line
```

For `m = 1` the difference sequence is empty, so the KMP loop reports nothing — yet a length-1 window `t[i]` always matches `p[0]` via `c = t[i] - p[0]`, for every value including negatives and zeros, so all `n` positions resonate. The trap inside this branch is `n = 0`: I print the count `0`, the position loop runs zero times, and if the newline lives only inside the loop's separator ternary I emit *no* second line at all — a malformed one-line output where the contract demands exactly two. So the branch prints the count, the positions, and an explicit empty line when `n == 0`:

```
if (m == 1) {
    cout << n << "\n";
    for (int i = 0; i < n; i++) cout << i << (i + 1 < n ? ' ' : '\n');
    if (n == 0) cout << "\n";
    return 0;
}
```

One more length corner is `m > n`: no window fits, answer `0`. The KMP path would in fact reach `0` on its own — with `tn = n-1 < pm = m-1`, `k` can never climb to `pm` — except when `n = 0`, where `td` has size `n-1 = -1`, the same negative-size disaster. So I guard `m > n` after the two base cases. Once past all three guards I am assured `n >= m >= 2`, hence `n-1 >= 1`, and no negative-size vector can ever be constructed.

With `m >= 2` and `n >= m` I build `pd` and `td`, compute the standard KMP failure function over `pd`, then scan `td` for full matches:

```
int k = 0;
for (int i = 0; i < tn; i++) {
    while (k > 0 && td[i] != pd[k]) k = fail[k - 1];
    if (td[i] == pd[k]) k++;
    if (k == pm) { hits.push_back(i - pm + 1); k = fail[k - 1]; }
}
```

On the sample (`pd = [1,-2]`, `fail = [0,0]`, `td = [1,-2,4,1,-2]`) this fires at `i=1`, pushing `1-2+1 = 0`, and at `i=4`, pushing `4-2+1 = 3` — hits `0 3`, as the index map predicted. The reason sign never enters into it: KMP only ever asks `pd[i] != pd[k]` and `td[i] != pd[k]`, plain integer (in)equality on `long long` values. There is no ordering comparison, no positivity test, no hashing; the sign of a value is simply not something the algorithm branches on, so negatives and zeros pass through untouched.

Two hand-traced instances pin the sign and overflow claims. An all-negative instance with no matching shape, `t = [-5,-2,-9,-1]`, `p = [-1,-5]`: `pd = [-4]`, `td = [3,-7,8]`, and `-4` occurs nowhere, so the answer is `0` — the negativity produces no spurious hit. And the overflow case, `t = [10^9, -10^9, 10^9, -10^9]`, `p = [10^9, -10^9]`: `td = [-2*10^9, 2*10^9, -2*10^9]` and `pd = [-2*10^9]`, every entry overflowing 32-bit but exact in `long long`, matching at difference indices `0` and `2` for positions `0 2`. That instance is exactly where the type choice earns its keep.

For real confidence I wrote the `O(n*m)` brute (fix `c = t[i]-p[0]`, verify the window) and a generator that oversamples the danger zone: `n` in `[0,8]`; value modes including a tiny `[-2,2]` alphabet so shifts collide and resonances are common, an all-negative mode, and a zero-heavy mode; pattern length drawn from `0` through `n+1` so `m=0`, `m=1`, and `m>n` all appear; and half the time the pattern is a shifted copy of a real text window, so the matching path and not just the misses gets exercised. Comparing count plus ordered positions over roughly a thousand seeds gives zero mismatches. The two defects the length corners invite — the `m=0` fall-through into a negative-size vector and the dropped line-2 newline at `n=0, m=1` — no longer occur, and the full self-contained module is in the answer.
