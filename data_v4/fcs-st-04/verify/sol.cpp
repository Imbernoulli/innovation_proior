#include <bits/stdc++.h>
using namespace std;

/*
 * Minimum palindromic factorization.
 *
 * We build the eertree (palindromic tree) of s online. For each node (a distinct
 * palindrome) we keep:
 *   len[v]   : its length
 *   link[v]  : the longest proper palindromic suffix (suffix link)
 *   slink[v] : the "series link" -- the longest palindromic suffix u of v with
 *              diff(u) != diff(v), where diff(v) = len[v] - len[link[v]].
 *   diff[v]  : len[v] - len[link[v]].
 *   sdp[v]   : a rolling best-dp value for the whole series (arithmetic
 *              progression of palindrome lengths) that v heads.
 *
 * The set of palindromic suffixes of any prefix decomposes into O(log n) chains,
 * each an arithmetic progression with common difference diff. Series links jump
 * between chains, so the per-position dp update is O(log n), total O(n log n).
 *
 * dp[i] = min number of palindromes to partition s[0..i-1] (prefix of length i).
 * dp[0] = 0. For a palindromic suffix p of s[0..i-1] of length L,
 *   dp[i] = min over p of dp[i-L] + 1.
 * Using the series-link grouping we evaluate all such p in O(log n).
 */

const int INF = 0x3f3f3f3f;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    // Read the (possibly empty) string token. If there is no token, s stays "".
    cin >> s;
    int n = (int)s.size();

    if (n == 0) {
        cout << 0 << "\n";
        return 0;
    }

    // Eertree storage. Max nodes = n + 2 (two roots).
    int maxNodes = n + 2;
    vector<int> len(maxNodes), link(maxNodes), diff(maxNodes), slink(maxNodes), sdp(maxNodes);
    // transitions: map per node from char -> node. Use array of size 26 via vector.
    // For |s| up to 1e6 a 26-wide table is 26*(n+2) ints ~ 100MB at 4 bytes; that
    // is too much. Use unordered_map per node would be slow. Instead use a single
    // hashed transition table keyed by (node, char).
    // We use a flat hash map (open addressing) for transitions.

    // Open-addressing hash for (node*32 + ch) -> child.
    // capacity is a power of two >= 2*(number of edges). Number of edges <= n.
    size_t cap = 1;
    while (cap < (size_t)(2 * (n + 2))) cap <<= 1;
    size_t mask = cap - 1;
    vector<long long> hkey(cap, -1);
    vector<int> hval(cap, 0);

    auto hfind = [&](long long key) -> size_t {
        // splitmix64-style mix, then linear probe.
        unsigned long long x = (unsigned long long)key;
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
        size_t h = (size_t)x & mask;
        while (hkey[h] != -1 && hkey[h] != key) h = (h + 1) & mask;
        return h;
    };

    // Node indices:
    //   0 : imaginary root with len = -1 (link of itself)
    //   1 : empty-string root with len = 0, link -> 0
    int sz = 2;
    len[0] = -1; link[0] = 0; diff[0] = 0; slink[0] = 0;
    len[1] = 0;  link[1] = 0; diff[1] = 0; slink[1] = 1;

    int last = 1; // current longest palindromic suffix node

    // dp over prefix lengths: dp[i] = min palindromes for s[0..i-1].
    vector<int> dp(n + 1, INF);
    dp[0] = 0;

    auto getChild = [&](int node, int c) -> int {
        long long key = (long long)node * 32 + c;
        size_t h = hfind(key);
        if (hkey[h] == key) return hval[h];
        return 0; // 0 means "no edge" (node 0 is a root, never a child)
    };
    auto setChild = [&](int node, int c, int child) {
        long long key = (long long)node * 32 + c;
        size_t h = hfind(key);
        hkey[h] = key;
        hval[h] = child;
    };

    for (int i = 0; i < n; i++) {
        int c = s[i] - 'a';

        // Find X = longest palindromic suffix that can be extended by s[i].
        int cur = last;
        while (true) {
            int l = len[cur];
            if (i - l - 1 >= 0 && s[i - l - 1] == s[i]) break;
            cur = link[cur];
        }

        int child = getChild(cur, c);
        if (child != 0) {
            last = child;
        } else {
            // Create a new node.
            int now = sz++;
            len[now] = len[cur] + 2;

            // suffix link of the new node
            if (len[now] == 1) {
                link[now] = 1; // single char -> empty root
            } else {
                int t = link[cur];
                while (true) {
                    int l = len[t];
                    if (i - l - 1 >= 0 && s[i - l - 1] == s[i]) break;
                    t = link[t];
                }
                link[now] = getChild(t, c);
                if (link[now] == 0) link[now] = 1; // safety, shouldn't trigger for len>=2
            }

            setChild(cur, c, now);

            diff[now] = len[now] - len[link[now]];
            if (diff[now] == diff[link[now]])
                slink[now] = slink[link[now]];
            else
                slink[now] = link[now];

            last = now;
        }

        // ---- DP update using series links ----
        // We want dp[i+1] = min over palindromic suffixes p (length L) of dp[i+1-L] + 1.
        // Walk the series chains via slink. For each series headed by node v:
        //   The series is the arithmetic progression of palindrome lengths
        //   {len[v], len[v]-diff[v], ..., down to (but excluding) the next series}.
        //   Let firstLen = the length of the *shortest* palindrome in this series
        //                 = len[slink[v]] + diff[v].
        //   sdp[v] caches min over the series (excluding v's own contribution from
        //   the immediate predecessor) of dp[(i+1) - those lengths].
        int idx = i + 1; // we are filling dp[idx]
        dp[idx] = INF;
        for (int v = last; len[v] > 0; v = slink[v]) {
            // position where the longest palindrome of this series starts:
            // start index in dp = idx - len[slink[v]] - diff[v].
            sdp[v] = dp[idx - len[slink[v]] - diff[v]];
            if (diff[v] == diff[link[v]]) {
                // link[v] is in the same series; fold its cached best in.
                sdp[v] = min(sdp[v], sdp[link[v]]);
            }
            // sdp[v] now = min dp over the entire series of v.
            dp[idx] = min(dp[idx], sdp[v] + 1);
        }
    }

    cout << dp[n] << "\n";
    return 0;
}
