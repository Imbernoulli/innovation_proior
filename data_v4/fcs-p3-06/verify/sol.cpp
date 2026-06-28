#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    long long p;
    if (!(cin >> t >> p)) return 0;

    vector<long long> ns(t);
    long long maxn = 0;
    for (int i = 0; i < t; i++) {
        cin >> ns[i];
        maxn = max(maxn, ns[i]);
    }

    // D(n) = number of derangements of n elements, modulo the prime p.
    // Recurrence: D(0) = 1, D(1) = 0, D(n) = (n-1) * (D(n-1) + D(n-2)) for n >= 2.
    // Compute D(k) mod p for every k up to maxn in one O(maxn) sweep.
    vector<long long> der(maxn + 1, 0);
    if (maxn >= 0) der[0] = 1 % p;
    if (maxn >= 1) der[1] = 0 % p;
    for (long long n = 2; n <= maxn; n++) {
        long long coeff = (n - 1) % p;
        long long inner = (der[n - 1] + der[n - 2]) % p;
        der[n] = (coeff * inner) % p;
    }

    string out;
    out.reserve((size_t)t * 12);
    for (int i = 0; i < t; i++) {
        out += to_string(der[ns[i]] % p);
        out += '\n';
    }
    cout << out;
    return 0;
}
