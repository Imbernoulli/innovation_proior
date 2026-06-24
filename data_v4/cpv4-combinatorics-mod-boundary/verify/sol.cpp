#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k, c, S, M;
    if (!(cin >> k >> c >> S >> M)) return 0;

    // Count integer solutions to x_1 + ... + x_k = S with 0 <= x_i <= c, modulo prime M.
    // Inclusion-exclusion over the number j of children that "overflow" (hold >= c+1):
    //   answer = sum_{j>=0, S - j*(c+1) >= 0} (-1)^j * C(k, j) * C(S - j*(c+1) + k - 1, k - 1)
    // The boundary "S - j*(c+1) >= 0" is INCLUSIVE: j may drive the remainder to exactly 0,
    // which still corresponds to a valid (fully-removed-overflow) configuration.

    if (S < 0) { cout << 0 << "\n"; return 0; }
    if (k == 0) { cout << ((S == 0) ? 1 % M : 0) << "\n"; return 0; }

    // Binomials needed: top argument is S + k - 1; also C(k, j). M is a prime strictly
    // larger than every argument we touch (guaranteed by the constraints), so factorial
    // inverses are valid.
    long long maxN = S + k; // safe upper bound on any "top" argument
    vector<long long> fact(maxN + 1), inv(maxN + 1);
    fact[0] = 1 % M;
    for (long long i = 1; i <= maxN; i++) fact[i] = fact[i - 1] * (i % M) % M;
    auto power = [&](long long b, long long e) {
        long long r = 1 % M; b %= M; if (b < 0) b += M;
        while (e > 0) { if (e & 1) r = r * b % M; b = b * b % M; e >>= 1; }
        return r;
    };
    inv[maxN] = power(fact[maxN], M - 2);
    for (long long i = maxN; i >= 1; i--) inv[i - 1] = inv[i] * (i % M) % M;

    auto C = [&](long long N, long long r) -> long long {
        if (r < 0 || N < 0 || r > N) return 0;
        return fact[N] * inv[r] % M * inv[N - r] % M;
    };

    long long ans = 0;
    long long step = c + 1; // an overflowing child holds at least c+1
    for (long long j = 0; S - j * step >= 0 && j <= k; j++) {
        long long rem = S - j * step;          // candies left after removing j overflows
        long long ways = C(k, j) * C(rem + k - 1, k - 1) % M;
        if (j & 1) ans = (ans - ways + M) % M;
        else       ans = (ans + ways) % M;
    }

    cout << ans % M << "\n";
    return 0;
}
