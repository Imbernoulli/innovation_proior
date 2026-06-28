#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // no piles -> First cannot move -> Second wins

    // Sieve primes up to sqrt(max a[i]) = sqrt(1e9) < 31623, so 31623 suffices.
    const int LIM = 31623;
    vector<int> spf(LIM + 1, 0);            // smallest prime factor; 0 marks "prime/unset"
    vector<int> primes;
    for (int i = 2; i <= LIM; i++) {
        if (spf[i] == 0) { spf[i] = i; primes.push_back(i); }
        for (int p : primes) {
            if ((long long)p * i > LIM) break;
            spf[p * i] = p;
            if (p == spf[i]) break;
        }
    }

    // For each pile x, the Grundy value of the "move to a proper divisor" game is
    // Omega(x) = number of prime factors of x counted with multiplicity.
    // Sprague-Grundy: the whole game's value is the XOR of per-pile Grundy values;
    // the first player wins iff that XOR is nonzero.
    int nimXor = 0;
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        int omega = 0;                      // Omega(x), prime-factor count with multiplicity
        for (int p : primes) {
            if ((long long)p * p > x) break;
            while (x % p == 0) { x /= p; omega++; }
        }
        if (x > 1) omega++;                 // remaining cofactor is a single prime
        nimXor ^= omega;                    // XOR the per-pile Grundy values
    }

    cout << (nimXor != 0 ? "First" : "Second") << "\n";
    return 0;
}
