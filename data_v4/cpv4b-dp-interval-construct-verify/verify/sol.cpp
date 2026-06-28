#include <bits/stdc++.h>
using namespace std;

// Smallest prime p with p >= n (n >= 1). Trial division is fine: the prime we need is < 2n,
// and we only test odd candidates up to sqrt(candidate).
static bool isPrime(long long v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (long long d = 3; d * d <= v; d += 2)
        if (v % d == 0) return false;
    return true;
}
static long long nextPrimeAtLeast(long long n) {
    long long p = max(n, 2LL);
    while (!isPrime(p)) p++;
    return p;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // n marks 0 = x_0 < x_1 < ... < x_{n-1} <= M with all pairwise differences distinct (Sidon set),
    // M = 8*n*n. Erdos-Turan: with a prime p, b[k] = 2*p*k + (k^2 mod p) (k = 0..p-1) is a Sidon set.
    // Take the first n elements (a subset of a Sidon set is Sidon) and shift so the smallest is 0.

    if (n == 1) {            // single mark sits at the origin; no pair, vacuously distinct
        cout << "0\n";
        return 0;
    }

    long long p = nextPrimeAtLeast(n);          // p in [n, 2n) by Bertrand's postulate
    vector<long long> b(n);
    for (int k = 0; k < n; k++)
        b[k] = 2 * p * (long long)k + ((long long)k * k) % p;   // already strictly increasing in k
    long long base = b[0];                       // b[0] = 0 here, but subtract defensively
    for (int k = 0; k < n; k++) b[k] -= base;

    // b is strictly increasing, so it is already sorted; emit as the mark coordinates.
    for (int k = 0; k < n; k++) {
        cout << b[k];
        cout << (k + 1 == n ? '\n' : ' ');
    }
    return 0;
}
