#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S, p;
    if (!(cin >> n >> S >> p)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // A "way" is fixed by how many coins of each DISTINCT denomination value are
    // used, so duplicate denomination values in the input refer to the same coin
    // type and are collapsed first.
    sort(c.begin(), c.end());
    c.erase(unique(c.begin(), c.end()), c.end());

    // dp[s] = number of distinct multisets of coins summing to exactly s, mod p.
    // Order-independent counting: put the COIN loop OUTSIDE and the sum loop
    // INSIDE. Each coin is fully processed before the next, so every multiset is
    // counted once in a fixed canonical order of denominations (no permutations).
    vector<long long> dp(S + 1, 0);
    dp[0] = 1 % p;                       // exactly one way to make 0: the empty multiset
    for (size_t i = 0; i < c.size(); i++) {
        long long coin = c[i];
        if (coin > S) continue;          // a coin larger than S can never be used
        for (long long s = coin; s <= S; s++) {
            dp[s] = (dp[s] + dp[s - coin]) % p;
        }
    }

    cout << dp[S] % p << "\n";
    return 0;
}
