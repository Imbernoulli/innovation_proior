#include <bits/stdc++.h>
using namespace std;

// Smallest prime >= x (x small here: x <= ~3000, so trial division is fine).
static bool isPrime(long long x) {
    if (x < 2) return false;
    if (x % 2 == 0) return x == 2;
    for (long long d = 3; d * d <= x; d += 2)
        if (x % d == 0) return false;
    return true;
}
static long long smallestPrimeGE(long long x) {
    long long p = max(2LL, x);
    while (!isPrime(p)) p++;
    return p;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // Erdos-Turan B2 (Sidon) set: pick a prime p >= n, then
    //   a_k = 2*p*k + (k*k mod p),   k = 0 .. n-1.
    // The full set k=0..p-1 has all pairwise differences distinct; any subset
    // (here the first n) inherits that, so the n elements form a Sidon set.
    // Max element is at k=n-1: 2*p*(n-1)+((n-1)^2 mod p) < 2*p*n <= 2*(1.4n)*n < 4n^2.
    long long p = smallestPrimeGE(n);

    vector<long long> a(n);
    for (int k = 0; k < n; k++) {
        long long km = (long long)k % p;
        a[k] = 2 * p * (long long)k + (km * km) % p;
    }

    // ---- Two-pointer certification at the REQUIRED scale (not just tiny cases) ----
    // a[] is strictly increasing. Collect all positive pairwise differences,
    // sort them, then a single two-pointer (adjacent) pass detects any duplicate.
    // If a duplicate exists the set is NOT Sidon. This certifies the construction
    // for the actual n, instead of trusting it from small hand examples.
    {
        // strictly increasing check
        for (int i = 1; i < n; i++) assert(a[i] > a[i - 1]);
        // range check
        long long L = 4LL * n * n;
        for (int i = 0; i < n; i++) assert(a[i] >= 0 && a[i] <= L);
        // all pairwise differences distinct (Sidon) via sort + two-pointer
        vector<long long> diffs;
        diffs.reserve((size_t)n * (n - 1) / 2);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                diffs.push_back(a[j] - a[i]);
        sort(diffs.begin(), diffs.end());
        for (size_t lo = 0, hi = 1; hi < diffs.size(); lo++, hi++)
            assert(diffs[lo] != diffs[hi]);
    }

    // Output the constructed set, space-separated on one line.
    string out;
    out.reserve((size_t)n * 8);
    for (int i = 0; i < n; i++) {
        if (i) out.push_back(' ');
        out += to_string(a[i]);
    }
    out.push_back('\n');
    fputs(out.c_str(), stdout);
    return 0;
}
