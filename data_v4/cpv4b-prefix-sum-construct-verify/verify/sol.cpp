#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n;
    if (!(cin >> n)) return 0;

    // Distinct prefix sums mod n exist iff n == 1 or n is even.
    // (Z_n is sequenceable iff n is even; n == 1 is the trivial single-seat case.)
    if (n != 1 && (n & 1LL)) {
        cout << -1 << "\n";
        return 0;
    }

    // Construction for even n (and n == 1): interleave the descending evens
    // starting at n with the ascending odds starting at 1:
    //   p = n, 1, n-2, 3, n-4, 5, ...
    // Position 0 takes n, n-2, n-4, ...; the odd positions take 1, 3, 5, ...
    // This is a permutation of {1..n}, and its prefix sums are pairwise
    // distinct modulo n.
    string out;
    out.reserve((size_t)n * 7);
    long long big = n;      // descending evens: n, n-2, n-4, ...
    long long small = 1;    // ascending odds:   1, 3, 5, ...
    for (long long i = 0; i < n; i++) {
        long long v;
        if ((i & 1LL) == 0) { v = big; big -= 2; }
        else                { v = small; small += 2; }
        out += to_string(v);
        out += (i + 1 == n) ? '\n' : ' ';
    }
    cout << out;
    return 0;
}
