#include <bits/stdc++.h>
using namespace std;

// distinct prime factors of m
static vector<long long> primeFactors(long long m) {
    vector<long long> ps;
    for (long long p = 2; p * p <= m; ++p) {
        if (m % p == 0) {
            ps.push_back(p);
            while (m % p == 0) m /= p;
        }
    }
    if (m > 1) ps.push_back(m);
    return ps;
}

// Precompute the signed squarefree divisors of m from its distinct primes:
// each subset S contributes divisor prod(S) with sign (-1)^|S|. Built once,
// reused by every query, so a single query costs O(#subsets) and no bit loop.
static vector<pair<long long,int>> buildSigned(const vector<long long>& ps) {
    vector<pair<long long,int>> div; // (divisor, sign)
    div.push_back({1LL, +1});        // empty subset
    for (long long p : ps) {
        int cur = (int)div.size();
        for (int i = 0; i < cur; ++i)
            div.push_back({div[i].first * p, -div[i].second});
    }
    return div;
}

// count of integers in [1, N] coprime to m, via the precomputed signed divisors.
// N may be 0 -> returns 0 (handled by caller passing N = L-1 which can be 0).
static long long coprimeUpTo(long long N, const vector<pair<long long,int>>& div) {
    if (N <= 0) return 0;
    long long total = 0;
    for (const auto& d : div) total += d.second * (N / d.first);
    return total;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long m;
    int q;
    if (!(cin >> m >> q)) return 0;

    vector<long long> ps = primeFactors(m);    // empty when m == 1
    vector<pair<long long,int>> div = buildSigned(ps);

    while (q--) {
        long long L, R;
        cin >> L >> R;
        // safe positions in [L, R] inclusive = coprimeUpTo(R) - coprimeUpTo(L-1)
        long long ans = coprimeUpTo(R, div) - coprimeUpTo(L - 1, div);
        cout << ans << "\n";
    }
    return 0;
}
