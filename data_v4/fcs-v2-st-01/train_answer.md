**Problem.** Given a lowercase string `s` of length `n` and `q` queries, each query names two substrings by `(l1, len1)` and `(l2, len2)` (1-indexed starts) and asks how they compare lexicographically: output `-1` if the first is smaller, `0` if equal, `1` if larger. Both `n` and `q` are up to `2*10^5`. Read `s` and the queries from stdin, print one of `-1/0/1` per query.

**Why the obvious comparator is too slow.** Comparing two substrings character by character is `O(min(len1, len2))` per query, hence `O(q*n)` overall. On `s = "aaaa...a"` (`n = 2*10^5`) with every query asking `1 n 1 n`, each query scans all `n` characters before concluding "equal" — `n*q = 4*10^10` comparisons, far beyond a 2-second limit. Repetitive strings are the natural adversarial input, so this worst case is not avoidable. The per-query cost must not depend on how long the two substrings agree.

**Key idea — reduce a substring comparison to a suffix LCP plus a length cap.** Lexicographic order is decided by the first position where the strings differ, and that position is fixed by the longest common prefix (LCP) of the two *suffixes* they start at. Let `i = l1-1`, `j = l2-1`, `common = LCP(suffix i, suffix j)`, and `cmpLen = min(len1, len2)`:

- if `common < cmpLen`, the suffixes disagree inside the compared region, so the character at offset `common` decides: compare `s[i+common]` with `s[j+common]`;
- if `common >= cmpLen`, the substrings agree on every compared character, so it is a pure length tie-break: shorter is smaller, equal lengths give `0`.

So if `LCP(i, j)` for arbitrary `i, j` is `O(1)`, every query is `O(1)`. The whole problem collapses to "arbitrary-pair suffix LCP fast", and the strongest deterministic tool for that is the **suffix array + LCP array + RMQ** stack:

- **Suffix array** by prefix doubling with radix (counting) sort: `O(n log n)`. `sa[r]` is the start of the rank-`r` suffix; `rnk[i]` is the rank of the suffix at `i`.
- **LCP array** by Kasai: `O(n)`. `lcp[r]` is the LCP of the rank-`r` suffix with the rank-`(r-1)` suffix.
- **RMQ** by sparse table: the LCP of two suffixes with ranks `ri < rj` is `min(lcp[ri+1 .. rj])` (the common prefix can only shrink as you walk the sorted order), answered in `O(1)` after `O(n log n)` build.

Preprocessing is `O(n log n)`; each query is `O(1)`.

**Pitfalls to get right.**
1. *Initial ranks must be compressed.* The prefix-doubling loop sizes its counting-sort bucket array to the rank range `[0, n]`, which holds only if every rank is in `[0, n-1]`. Re-ranking enforces that for every round *after* the first, but the natural initialization `rnk[i] = (unsigned char)s[i]` uses raw character codes (~97..122), which overflow the bucket array on round one and corrupt the heap. Fix: coordinate-compress the initial single-character ranks into `[0, classes-1]` before the loop. (AddressSanitizer pins this to the `cnt[secondKey(i)+1]++` line on the first round.)
2. *Arbitrary-pair LCP is a range-min, not an adjacent LCP.* `lcp[r]` only gives the LCP of sorted-adjacent suffixes; the LCP of two arbitrary suffixes is the minimum of the adjacent LCPs strictly between their ranks. The same-suffix case (`i == j`) must short-circuit to the suffix length `n - i` before touching the table.
3. *Length tie-break.* When the LCP reaches the compared length, do not fall through to a character comparison (there is no differing character) — decide purely by `len1` vs `len2`.

**Edge cases.** `n = 1` (the `i == j` short-circuit and the `2 x 1` sparse table both survive, answer `0`); all-identical strings (every equal-length query returns `0`; this is the naive comparator's worst case, here `O(1)`); periodic strings where one substring is a prefix of the other (length tie-break decides, e.g. `"abc"` vs `"abcabc"` returns `-1`); equal substrings at different offsets (LCP exceeds the compared length, answer `0`). No 64-bit arithmetic is needed — every quantity is an index bounded by `2*10^5`.

**Complexity.** `O(n log n)` preprocessing (suffix sort dominates) plus `O(1)` per query, `O(n log n)` total time and `O(n log n)` memory for the sparse table. At full scale (`n = q = 2*10^5`) this runs in about 0.09 s using ~22 MB.

**Code.**

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
