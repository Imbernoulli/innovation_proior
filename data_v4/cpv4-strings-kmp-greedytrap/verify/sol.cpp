#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s, t;
    if (!(cin >> s >> t)) return 0;     // missing input -> nothing to do
    int m = (int)s.size();
    int n = (int)t.size();

    // reach[i] = length of the longest prefix of s that matches t starting at position i.
    // Build via the Z-function of  c = s + '\x01' + t  (separator absent from the alphabet).
    // For position p inside the t-part, Z[p] (capped at m) is exactly that longest prefix length,
    // and because every shorter prefix of s is a prefix of that match, the set of block lengths
    // usable at i is precisely {1, 2, ..., reach[i]}.
    string c;
    c.reserve(m + 1 + n);
    c += s;
    c.push_back('\x01');
    c += t;
    int N = (int)c.size();
    vector<int> Z(N, 0);
    int l = 0, r = 0;
    for (int i = 1; i < N; i++) {
        if (i < r) Z[i] = min(r - i, Z[i - l]);
        while (i + Z[i] < N && c[Z[i]] == c[i + Z[i]]) Z[i]++;
        if (i + Z[i] > r) { l = i; r = i + Z[i]; }
    }
    vector<int> reach(n, 0);
    int base = m + 1;                   // index in c where the t-part begins
    for (int i = 0; i < n; i++) reach[i] = min(Z[base + i], m);

    // Minimum number of blocks = minimum jumps to advance from index 0 to index n,
    // where from i you may land on any j in [i+1, i+reach[i]].
    // This is Jump Game II: the LEVEL/BFS greedy (extend the current reachable window to the
    // farthest its members can reach, counting one block per extension) is optimal. The tempting
    // "take the longest prefix each step" (i += reach[i]) greedy is NOT optimal and can deadlock.
    if (n == 0) { cout << 0 << "\n"; return 0; }
    long long blocks = 0;
    int curEnd = 0;                     // farthest index settled with `blocks` jumps
    int farthest = 0;                   // farthest index reachable with one more jump
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (i > curEnd) { ok = false; break; }   // index i unreachable -> infeasible
        if (i + reach[i] > farthest) farthest = i + reach[i];
        if (i == curEnd) {              // exhausted current window: must spend a block
            if (farthest <= curEnd) { ok = false; break; }  // cannot progress
            blocks++;
            curEnd = farthest;
            if (curEnd >= n) break;
        }
    }
    if (!ok || curEnd < n) cout << -1 << "\n";
    else cout << blocks << "\n";
    return 0;
}
