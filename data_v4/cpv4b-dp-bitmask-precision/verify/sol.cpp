#include <bits/stdc++.h>
using namespace std;

// Print a non-negative __int128 in decimal.
static string i128_to_string(__int128 x) {
    if (x == 0) return "0";
    bool neg = false;
    if (x < 0) { neg = true; x = -x; }
    string s;
    while (x > 0) { int d = (int)(x % 10); s.push_back(char('0' + d)); x /= 10; }
    if (neg) s.push_back('-');
    reverse(s.begin(), s.end());
    return s;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;                 // empty input -> treat as n = 0
    vector<long long> b(n);
    for (auto &x : b) cin >> x;
    vector<vector<long long>> m(n, vector<long long>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> m[i][j];

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << b[0] << "\n"; return 0; }

    const __int128 NEG = -1;                    // sentinel: state unreachable
    // dp[mask][last] = max product of a path visiting exactly the nodes in mask
    // and ending at node `last` (last must be in mask). Stored as __int128.
    int full = 1 << n;
    vector<vector<__int128>> dp(full, vector<__int128>(n, NEG));

    for (int v = 0; v < n; v++)
        dp[1 << v][v] = (__int128)b[v];         // a path of one node = its base value

    __int128 best = NEG;
    for (int mask = 1; mask < full; mask++) {
        for (int last = 0; last < n; last++) {
            __int128 cur = dp[mask][last];
            if (cur == NEG) continue;
            if (mask == full - 1) {             // a full Hamiltonian path
                if (best == NEG || cur > best) best = cur;
            }
            for (int nxt = 0; nxt < n; nxt++) {
                if (mask & (1 << nxt)) continue;
                __int128 cand = cur * (__int128)m[last][nxt];   // extend by one edge
                int nm = mask | (1 << nxt);
                if (dp[nm][nxt] == NEG || cand > dp[nm][nxt])
                    dp[nm][nxt] = cand;          // EXACT compare of products, no division/log
            }
        }
    }

    cout << i128_to_string(best) << "\n";
    return 0;
}
