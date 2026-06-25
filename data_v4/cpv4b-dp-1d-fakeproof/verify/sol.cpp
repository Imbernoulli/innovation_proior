#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // empty input -> nothing to print

    const long long MOD = 1000000007LL;

    // Prefix XOR P[0..n] with P[0] = 0. The segment a[j..i-1] has signature
    // P[i] XOR P[j]; it is "clean" when popcount(P[i] XOR P[j]) is EVEN.
    //
    // Identity (verified numerically before trusting it):
    //   parity(popcount(x XOR y)) = parity(popcount(x)) XOR parity(popcount(y)).
    // Hence popcount(P[i] XOR P[j]) is even  <=>  the popcount PARITIES of
    // P[i] and P[j] are EQUAL. So bucket previous dp by the parity of
    // popcount(P[j]) and, at step i, add the bucket whose parity matches P[i].
    //
    // dp[i] = number of clean partitions of the prefix of length i, with dp[0] = 1.

    long long bucket[2] = {0, 0};         // bucket[p] = sum of dp[j] over processed j with parity p
    int P = 0;                            // running prefix XOR, P[0] = 0
    // j = 0: dp[0] = 1, parity of popcount(P[0]=0) is 0.
    bucket[__builtin_parity((unsigned)P)] = 1; // = bucket[0] = 1

    long long dp = 1;                     // dp[0]; reassigned each step to dp[i]
    for (int i = 1; i <= n; i++) {
        int x;
        cin >> x;
        P ^= x;                           // now P == P[i]
        int par = __builtin_parity((unsigned)P);
        dp = bucket[par] % MOD;           // sum of dp[j], j<i, with matching popcount parity
        bucket[par] = (bucket[par] + dp) % MOD; // register dp[i] = dp under P[i]'s parity
    }

    cout << dp % MOD << "\n";
    return 0;
}
