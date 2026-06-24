#include <bits/stdc++.h>
using namespace std;

// Count non-empty subsequences whose product is strictly positive, modulo m.
// A subset has a strictly positive product iff it contains NO zero element and an
// EVEN number (0, 2, 4, ...) of negative elements. Positives are unconstrained.
// Let P = #positives, N = #negatives. Zeros can never be part of such a subset.
//   E_N = number of even-sized subsets of the N negatives
//       = 2^(N-1)  if N >= 1   (exactly half of the 2^N subsets are even-sized),
//       = 1        if N == 0   (only the empty choice, which is even-sized).
//   total = 2^P * E_N          (subsets with no zero and even #negatives, empty allowed)
//   answer = (total - 1) mod m (remove the single empty subset), kept non-negative.

long long power_mod(long long base, long long exp, long long mod) {
    base %= mod;
    if (base < 0) base += mod;
    long long result = 1 % mod;            // 1 % mod handles mod == 1
    while (exp > 0) {
        if (exp & 1) result = (__int128)result * base % mod;
        base = (__int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, m;
    if (!(cin >> n >> m)) return 0;

    long long P = 0, N = 0; // zeros are counted by neither: they can never be chosen
    for (long long i = 0; i < n; i++) {
        long long x;
        cin >> x;
        if (x > 0) P++;
        else if (x < 0) N++;
        // x == 0 contributes to neither P nor N
    }

    long long posWays = power_mod(2, P, m);              // 2^P
    long long evenNeg;
    if (N == 0) evenNeg = 1 % m;                         // base case: NOT 2^(-1)
    else evenNeg = power_mod(2, N - 1, m);               // 2^(N-1)

    long long total = (__int128)posWays * evenNeg % m;   // includes the empty subset, in [0, m)
    long long answer = (total - 1 % m + m) % m;          // remove empty subset, keep non-negative

    cout << answer << "\n";
    return 0;
}
