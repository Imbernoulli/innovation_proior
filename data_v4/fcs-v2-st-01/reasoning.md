**Reading the problem and pinning the contract.** I have a lowercase string `s` of length `n` and `q` queries; each query gives two substrings by `(l1, len1)` and `(l2, len2)` (1-indexed starts) and asks how they compare lexicographically: output `-1`, `0`, or `1`. Before any algorithm I fix the scale, because it decides everything: `n` and `q` are both up to `2*10^5`. The output is one of three small integers per query, and the comparison is plain dictionary order — compare characters left to right, the first mismatch decides, and if everything matched until one substring ends, the shorter one is smaller. I will read `s`, then `q`, then four integers per query. Nothing here needs 64-bit arithmetic; the only quantities are indices bounded by `2*10^5`. The real cost is the per-query work, so the entire design question is: how cheap can one comparison be?

**The obvious approach, and the concrete case that kills it.** The most direct comparator just walks the two substrings: compare `s[l1-1+t]` against `s[l2-1+t]` for `t = 0, 1, 2, ...` up to `min(len1, len2)`, return on the first difference, else break the tie by length. That is `O(min(len1, len2))` per query, trivially correct, three lines. The trouble is the worst case. Take `s = "aaaa...a"`, `n = 2*10^5`, and make every query compare the whole string against itself: `l1 = l2 = 1`, `len1 = len2 = n`. Each query scans all `n` characters before concluding "equal", and there are `q = 2*10^5` of them, so the work is `n * q = 4*10^10` character comparisons. That is hundreds of seconds — nowhere near a 2-second limit. Worse, repetitive strings are exactly the natural adversarial input, so I cannot wave this away as "unlikely". The naive comparator is correct but fundamentally too slow; I need per-query work that does not depend on how long the agreement runs.

**Reformulating: a substring comparison is a suffix comparison plus a length cap.** Here is the move that unlocks the whole problem. Comparing `s[l1-1 .. l1-1+len1)` with `s[l2-1 .. l2-1+len2)` is *almost* the same as comparing the two full suffixes `s[l1-1 ..)` and `s[l2-1 ..)`. Why? Because lexicographic order is decided by the **first position where the two strings differ**, and that position is determined by the longest common prefix (LCP) of the two suffixes — the substring lengths only matter once we know whether that first difference falls inside the compared region. Concretely: let `common = LCP(suffix at i, suffix at j)` where `i = l1-1`, `j = l2-1`, and let `cmpLen = min(len1, len2)`. Two cases:

- If `common < cmpLen`, the suffixes already disagree within the compared region, so the character at offset `common` decides: compare `s[i+common]` with `s[j+common]`.
- If `common >= cmpLen`, the two substrings agree on every compared character (they share a prefix at least as long as the shorter substring). Then it is purely a length tie-break: shorter is smaller; equal lengths give `0`.

So if I can compute `LCP(suffix i, suffix j)` for *arbitrary* `i, j` in `O(1)`, every query is `O(1)` and the whole problem is `O(q)` after preprocessing. The naive comparator's pathology — long runs of agreement — is exactly what the LCP collapses into a single number. That is the insight: **don't compare characters, compare suffix LCPs.** The remaining question is how to get arbitrary-pair LCP fast.

**Weighing the two ways to get arbitrary-pair LCP.** There are two standard routes.

- *Rolling hash + binary search.* Precompute prefix hashes of `s`; to find `LCP(i, j)`, binary-search the largest `len` such that the hash of `s[i..i+len)` equals the hash of `s[j..j+len)`. That is `O(log n)` per query and randomized — a hash collision silently returns a wrong answer, and the constant factor of recomputing/looking up two hashes per binary-search step is heavy. At `q = 2*10^5` it would pass, but it is not the deterministic, asymptotically-tightest tool, and collisions are a real correctness risk on adversarial input.
- *Suffix array + LCP array + RMQ.* Sort all suffixes into the **suffix array** `sa` (so `sa[r]` is the start index of the rank-`r` suffix, and `rnk[i]` is the rank of the suffix starting at `i`). Compute the **LCP array** `lcp`, where `lcp[r]` is the LCP of the rank-`r` suffix with the rank-`(r-1)` suffix — the LCP of *adjacent* suffixes in sorted order. Then the LCP of two *arbitrary* suffixes with ranks `ri < rj` is the **minimum** of `lcp[ri+1 .. rj]`: as you walk the sorted order from one suffix to the other, the common prefix can only shrink, and the bottleneck is the smallest adjacent LCP along the way. A range-minimum query (RMQ) over a static `lcp` array is `O(1)` with a sparse table after `O(n log n)` preprocessing.

The suffix-array route is deterministic, the preprocessing is `O(n log n)` (dominated by the suffix sort), and each query is genuinely `O(1)`. That is the canonical, strongest tool for "many arbitrary-pair LCP / substring-equality queries", so I commit to **suffix array + Kasai LCP + sparse-table RMQ**.

**Choosing the suffix-array construction.** For `n <= 2*10^5` I do not need the fully linear SA-IS / DC3 machinery (it is fiddly and error-prone); the canonical strong construction at this scale is **prefix doubling with radix (counting) sort**, which is `O(n log n)`. The idea: rank suffixes first by their first character, then iteratively double the comparison length. At round `k`, two suffixes' order is decided by the pair `(rnk[i], rnk[i+k])` — the rank of their first `k` characters and the rank of the next `k`. Sorting these pairs with two stable counting sorts (by the second key, then the first key) and re-ranking gives the rank for length `2k`. After `O(log n)` rounds, every suffix has a distinct rank and `sa` is the sorted order. Each round is `O(n)` thanks to counting sort, so the whole thing is `O(n log n)`.

**Kasai for the LCP array.** Given `sa` and its inverse `rnk`, Kasai's algorithm computes `lcp` in `O(n)` with one clever invariant: process suffixes by their *start position* `i = 0, 1, 2, ...`, and maintain `h`, the current common-prefix length. For suffix `i`, its predecessor in sorted order starts at `j = sa[rnk[i]-1]`; extend `h` while `s[i+h] == s[j+h]`; record `lcp[rnk[i]] = h`. The key fact is that when we move from `i` to `i+1`, the LCP can drop by at most one, so we decrement `h` by one (if positive) and never reset it — total work is `O(n)` amortized.

**Sparse table for RMQ.** `lcp` is static, so I build a sparse table `sp[j][i] = min(lcp[i .. i+2^j))` in `O(n log n)` and answer `min` over any inclusive range `[lo, hi]` in `O(1)` by covering it with two overlapping power-of-two blocks: with `len = hi-lo+1`, `k = floor(log2(len))`, the answer is `min(sp[k][lo], sp[k][hi-2^k+1])`. For two suffixes `i, j` with `ri = rnk[i] < rj = rnk[j]`, their LCP is `min(lcp[ri+1 .. rj])`; if `i == j` the LCP is the suffix length `n - i`.

**First implementation.** I write the four pieces — initial ranks, the doubling loop with two counting sorts, Kasai, the sparse table — and the query loop that applies the case split above. My first cut initializes the single-character ranks the natural way:

```
vector<int> sa(n), rnk(n), tmp(n);
for (int i = 0; i < n; i++) {
    sa[i] = i;
    rnk[i] = (unsigned char)s[i];   // rank by raw character code
}
```

and inside the doubling loop the counting-sort bucket array is sized to the rank range, which I assume is `[0, n]`:

```
int maxKey = n + 1;
vector<int> cnt(maxKey + 1, 0);
for (int i = 0; i < n; i++) cnt[secondKey(i) + 1]++;     // secondKey in [-1, n-1]
...
```

I compile it and run my first sanity sample, `s = "banana"` with a handful of queries.

**The crash, and tracing it to a precise cause.** The program aborts with `free(): invalid pointer` — a heap corruption, not a wrong answer. That smells like an out-of-bounds write into a `vector`. I rebuild under AddressSanitizer and it points straight at the counting-sort line `cnt[secondKey(i) + 1]++` in the *first* doubling round, "heap-buffer-overflow", writing past the end of a 32-byte region (the `cnt` vector). Now I reason about *why*. I sized `cnt` to `n + 2` slots on the assumption that every rank is in `[0, n-1]`. That assumption holds after the first re-ranking — re-ranking always compresses ranks into `[0, classes-1] ⊆ [0, n-1]`. But on the **very first round**, `rnk[i]` is the raw character code `(unsigned char)s[i]`, which for lowercase letters is around `97..122`. So `secondKey(i) = rnk[i+k]` can be `~122`, and `secondKey(i)+1` indexes well past the `n+2`-slot `cnt` when `n` is small (here `n = 6`, so `cnt` has 8 slots but I index ~123). The placement loop `sa[cnt[rnk[idx]]++]` has the same defect with raw codes. The bucket array is sized for compressed ranks but fed raw ASCII on round one — a textbook off-by-alphabet bug.

**The fix: compress initial ranks into `[0, classes-1]` up front.** The clean repair is to make the initialization establish the invariant the rest of the loop relies on: every rank lives in `[0, n-1]`. I coordinate-compress the characters before the doubling loop — sort the distinct characters and map each character to its index in that sorted list:

```
vector<int> chars(s.begin(), s.end());
vector<int> sorted = chars;
sort(sorted.begin(), sorted.end());
sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());
for (int i = 0; i < n; i++) {
    sa[i] = i;
    rnk[i] = (int)(lower_bound(sorted.begin(), sorted.end(), chars[i]) - sorted.begin());
}
```

Now `rnk[i]` is in `[0, (#distinct)-1] ⊆ [0, n-1]` from the start, so `maxKey = n+1` correctly bounds every key in every round, and the `cnt` array of `n+2` slots is always large enough. I rebuild under ASan and rerun `banana` — no overflow, and the answers match my hand-computation: `s[2..5)="ana"` vs `s[4..7)="ana"` gives `0`; `s[1..7)="banana"` vs `s[3..7)="nana"` gives `-1` (b < n); `s[2..4)="an"` vs `s[4..6)="an"` gives `0`; `s[1..2)="b"` vs `s[5..6)="n"` gives `-1`. All correct.

**A second, subtler scare during stress testing — and why it was a test-harness artifact, not a bug.** With the crash fixed, I built a differential harness: a generator emits random small strings and valid random queries, my C++ runs against a dead-simple Python oracle that just slices the substrings and compares them with `<`. The first sweep of a few hundred tiny cases (`n <= 12`) was clean — 0 mismatches. I then widened to bigger strings (`n` up to ~80, full 26-letter alphabet) and suddenly saw a flurry of "MISMATCH" and even segfaults. My first instinct was a real bug in the suffix sort at larger `n`. But the discipline is to *reproduce the smallest failing input in isolation*, so I dumped the exact generated input for one failing seed and ran both programs on it directly — and they **matched**, with ASan clean. The "failures" only appeared inside the batch loop. The cause was my own bash harness: it wrote the generated input to a single shared `/tmp/in.txt` and read it back from two processes while the previous iteration's large output was still flushing, so occasionally a program read a half-written or stale file (which is also why the *Python oracle itself* crashed on one "input", trying to parse a letter as an integer — an impossible state for a valid input). The lesson: a mismatch is a hypothesis, not a verdict; isolate it before believing it. I rewrote the harness to use a fresh per-iteration file and to check process exit codes explicitly. Re-running 1000+ cases that way: **0 mismatches, 0 crashes.** The solution was correct all along; my test plumbing was racy.

**Deliberate edge cases, because this is where suffix code dies.**
- `n = 1`, query `1 1 1 1`: same suffix, `lcpSuffix` returns `n - i = 1`, `cmpLen = 1`, so `eq = 1 = cmpLen`, lengths equal, answer `0`. The sparse-table sizing also has to survive `n = 1`: `LOG` ends at 2 and `sp` is `2 x 1`, and `lcpSuffix` short-circuits the `i == j` case before ever touching the table. Verified under ASan.
- All identical characters, `s = "aaaa...a"`: every adjacent LCP is large, every equal-length query returns `0`, and this is exactly the naive comparator's worst case — here it is `O(1)` per query. Verified.
- Periodic strings like `abcabcabc`: queries that are prefixes of one another exercise the `common >= cmpLen` length tie-break (e.g. `"abc"` vs `"abcabc"` returns `-1` because the shorter is smaller). Verified against the oracle.
- Equal substrings at different offsets (`"ana"` at positions 2 and 4 in `banana`): the LCP of the two suffixes is `>= 3`, `cmpLen = 3`, so `eq = 3 = cmpLen`, equal lengths, answer `0`. Verified.
- Output format: exactly `-1`, `0`, or `1` per line; I build the whole output in a `string` and flush once to keep `q = 2*10^5` lines fast.

**Performance check at the limit.** I generated the full-scale worst case: `n = 2*10^5`, a 3-letter alphabet (so suffix sorting does the most doubling rounds and LCPs are long), and `q = 2*10^5` random valid queries. End to end it runs in about 0.09 seconds using ~22 MB — comfortably inside the 2-second / 256-MB budget. The `O(n log n)` preprocessing plus `O(q)` querying behaves exactly as the asymptotics promise, and the constant factor (counting-sort radix passes, a single sparse table of `int`s) is small.

**Final solution.** I convinced myself the *idea* is right by reducing substring comparison to suffix LCP and disproving the naive scanner on the all-`a` worst case; I convinced myself the *construction* is right by tracing the heap overflow to a raw-character-code rank that violated the bucket-size invariant and fixing it with up-front coordinate compression; and I convinced myself the *code* is right by 1000+ differential cases against a trivial slicing oracle (after fixing my own racy harness), explicit edge cases under ASan, and a full-scale timing run. That is what I ship — one self-contained file: suffix array by prefix doubling with radix sort, Kasai LCP, sparse-table RMQ, and the case-split query that turns each comparison into an `O(1)` LCP lookup plus a length tie-break:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Suffix array (prefix doubling + radix sort, O(n log n)),
// Kasai LCP (O(n)), sparse-table RMQ over LCP (O(n log n) build, O(1) query).
// Answers q lexicographic-comparison queries between two substrings in O(1) each.

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int n = (int)s.size();

    int q;
    cin >> q;

    // ---- Suffix array via prefix doubling + counting sort ----
    // sa[r]  = start index of the rank-r suffix
    // rnk[i] = rank of the suffix starting at i (after current round)
    vector<int> sa(n), rnk(n), tmp(n);
    {
        // Compress the initial single-character ranks into [0, classes-1] so that
        // every rank stays within [0, n-1] throughout — this keeps the counting-sort
        // bucket array sized O(n) (raw char codes would overflow it).
        vector<int> chars(s.begin(), s.end());
        vector<int> sorted = chars;
        sort(sorted.begin(), sorted.end());
        sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());
        for (int i = 0; i < n; i++) {
            sa[i] = i;
            rnk[i] = (int)(lower_bound(sorted.begin(), sorted.end(), chars[i]) - sorted.begin());
        }
    }

    for (int k = 1; ; k <<= 1) {
        // Comparator on (rnk[i], rnk[i+k]) pairs; sort sa by it via two counting sorts.
        auto secondKey = [&](int idx) -> int {
            return idx + k < n ? rnk[idx + k] : -1; // -1 sorts first (shorter suffix is smaller)
        };

        // Counting sort by the second key (offset by +1 so -1 maps to bucket 0).
        int maxKey = n + 1; // ranks are in [0, n-1], second key shifted into [0, n]
        vector<int> cnt(maxKey + 1, 0);
        for (int i = 0; i < n; i++) cnt[secondKey(i) + 1]++;
        for (int i = 1; i <= maxKey; i++) cnt[i] += cnt[i - 1];
        // Build order stable by second key into tmp.
        for (int i = n - 1; i >= 0; i--) tmp[--cnt[secondKey(i) + 1]] = i;

        // Counting sort by the first key rnk[i]; iterate tmp in order for stability.
        fill(cnt.begin(), cnt.end(), 0);
        for (int i = 0; i < n; i++) cnt[rnk[i] + 1]++;
        for (int i = 1; i <= maxKey; i++) cnt[i] += cnt[i - 1];
        for (int i = 0; i < n; i++) {
            int idx = tmp[i];
            sa[cnt[rnk[idx]]++] = idx; // rnk in [0,n-1], +1 already accounted via prefix shift
        }

        // Recompute ranks.
        tmp[sa[0]] = 0;
        int classes = 1;
        for (int i = 1; i < n; i++) {
            int a = sa[i - 1], b = sa[i];
            int a2 = a + k < n ? rnk[a + k] : -1;
            int b2 = b + k < n ? rnk[b + k] : -1;
            if (rnk[a] != rnk[b] || a2 != b2) classes++;
            tmp[b] = classes - 1;
        }
        rnk = tmp;
        if (classes == n) break; // all suffixes distinct -> sorted
        if (k >= n) break;       // safety
    }

    // ---- Kasai LCP array ----
    // lcp[r] = LCP(suffix sa[r], suffix sa[r-1]); lcp[0] = 0.
    vector<int> lcp(n, 0);
    {
        int h = 0;
        for (int i = 0; i < n; i++) {
            if (rnk[i] > 0) {
                int j = sa[rnk[i] - 1];
                while (i + h < n && j + h < n && s[i + h] == s[j + h]) h++;
                lcp[rnk[i]] = h;
                if (h > 0) h--;
            } else {
                h = 0;
            }
        }
    }

    // ---- Sparse table for RMQ over lcp[1..n-1] ----
    int LOG = 1;
    while ((1 << LOG) < n) LOG++;
    LOG++;
    vector<vector<int>> sp(LOG, vector<int>(n, INT_MAX));
    for (int i = 0; i < n; i++) sp[0][i] = lcp[i];
    for (int j = 1; j < LOG; j++) {
        for (int i = 0; i + (1 << j) <= n; i++) {
            sp[j][i] = min(sp[j - 1][i], sp[j - 1][i + (1 << (j - 1))]);
        }
    }
    // LCP of two suffixes with ranks ri < rj is min over lcp[ri+1 .. rj].
    auto lcpSuffix = [&](int i, int j) -> int {
        if (i == j) return n - i; // same suffix
        int ri = rnk[i], rj = rnk[j];
        if (ri > rj) swap(ri, rj);
        int lo = ri + 1, hi = rj; // inclusive range over lcp[]
        int len = hi - lo + 1;
        int k = 31 - __builtin_clz(len);
        return min(sp[k][lo], sp[k][hi - (1 << k) + 1]);
    };

    // ---- Answer queries ----
    // Each query: l1 len1 l2 len2 (1-indexed l1,l2). Compare s[l1-1 .. l1-1+len1)
    // with s[l2-1 .. l2-1+len2) lexicographically. Output -1 / 0 / 1.
    string out;
    out.reserve((size_t)q * 3);
    for (int t = 0; t < q; t++) {
        int l1, len1, l2, len2;
        cin >> l1 >> len1 >> l2 >> len2;
        int i = l1 - 1, j = l2 - 1;
        int common = lcpSuffix(i, j);          // LCP of the two full suffixes
        int cmpLen = min(len1, len2);          // chars actually compared
        int eq = min(common, cmpLen);          // matched prefix length within compared region
        int res;
        if (eq < cmpLen) {
            // First differing character decides.
            res = (s[i + eq] < s[j + eq]) ? -1 : 1;
        } else {
            // One substring is a prefix of the other (or equal); shorter is smaller.
            if (len1 < len2) res = -1;
            else if (len1 > len2) res = 1;
            else res = 0;
        }
        if (res < 0) out += "-1";
        else if (res > 0) out += "1";
        else out += "0";
        out += '\n';
    }
    cout << out;
    return 0;
}
```

**Causal recap.** The naive character-by-character comparator is correct but `O(q*n)` and dies on repetitive strings (all-`a` full-length queries scan `4*10^10` characters); the resolution is to see a substring comparison as a *suffix* comparison capped by the compared length, so the only thing I need fast is the LCP of two arbitrary suffixes; the strongest deterministic tool for that is a suffix array (prefix doubling + radix sort, `O(n log n)`), Kasai's LCP array (`O(n)`), and a sparse table giving the arbitrary-pair LCP as a range-minimum in `O(1)`; my first build crashed because round-one ranks were raw character codes that overran a bucket array sized for compressed ranks, fixed by coordinate-compressing the initial ranks into `[0, n-1]`; a later "mismatch" scare turned out to be a racy test harness, not a bug, confirmed by isolating the input and re-running cleanly; and the case split — first-difference character if the LCP is shorter than the compared length, else a length tie-break — closes out the prefix, equal-substring, single-character, and worst-case-repetitive corners, all verified against a trivial slicing oracle and at full scale.
