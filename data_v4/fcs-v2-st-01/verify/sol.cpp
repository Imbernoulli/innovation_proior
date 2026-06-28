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
