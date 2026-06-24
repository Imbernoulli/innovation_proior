#include <bits/stdc++.h>
using namespace std;

// Minimum number of perfect squares (1,4,9,16,...) that sum to N.
// By Lagrange's four-square theorem the answer is always 1, 2, 3, or 4.
// We decide which using number-theoretic tests (Legendre's three-square theorem),
// not greedy and not an O(N*sqrt N) DP, so it works for N up to 1e9.

static bool isSquare(long long n) {
    if (n < 0) return false;
    long long r = (long long)sqrtl((long double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    return r * r == n;
}

int main() {
    long long n;
    if (!(cin >> n)) return 0;

    // Answer 1: N itself is a perfect square.
    if (isSquare(n)) { cout << 1 << "\n"; return 0; }

    // Answer 2: N = a^2 + b^2 for some a >= 1. Try every a up to sqrt(N).
    for (long long a = 1; a * a <= n; a++) {
        if (isSquare(n - a * a)) { cout << 2 << "\n"; return 0; }
    }

    // Answer 4: Legendre's three-square theorem. N is NOT a sum of three squares
    // iff N = 4^k * (8*m + 7) for non-negative integers k, m.
    {
        long long m = n;
        while (m % 4 == 0) m /= 4;
        if (m % 8 == 7) { cout << 4 << "\n"; return 0; }
    }

    // Otherwise three squares suffice (and we already ruled out 1 and 2).
    cout << 3 << "\n";
    return 0;
}
