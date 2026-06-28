#include <bits/stdc++.h>
using namespace std;

// Pell: P(0)=0, P(1)=1, P(n)=2P(n-1)+P(n-2).
// Fast doubling on the companion matrix M=[[2,1],[1,0]], M^n=[[P(n+1),P(n)],[P(n),P(n-1)]].
// With p(n)=P(n), q(n)=P(n-1) (so p(n+1)=2p(n)+q(n)... we track P(n) and P(n+1)):
//   P(2k)   = P(k) * (2*P(k+1) - 2*P(k))            ... derived below
//   P(2k+1) = P(k+1)^2 + P(k)^2
// All arithmetic modulo a prime p given on input.

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        unsigned long long N;     // 0 <= N <= 1e18
        unsigned long long p;     // 2 <= p <= 1e18, prime (used only as a modulus)
        cin >> N >> p;

        // Iterative fast doubling from the most significant set bit of N.
        // Invariant: (a, b) = (P(k) mod p, P(k+1) mod p) for the prefix k of N's bits read so far.
        unsigned long long a = 0 % p; // P(0)
        unsigned long long b = 1 % p; // P(1)

        // __int128 to multiply two values < p <= 1e18 without overflow.
        for (int i = 63; i >= 0; --i) {
            // c = P(2k)   = P(k) * (2*P(k+1) - 2*P(k))  (mod p)
            // d = P(2k+1) = P(k+1)^2 + P(k)^2           (mod p)
            unsigned long long two_b = ( (unsigned __int128)2 * b ) % p;
            unsigned long long two_a = ( (unsigned __int128)2 * a ) % p;
            unsigned long long t = (two_b + p - two_a) % p;       // 2*P(k+1) - 2*P(k)
            unsigned long long c = (unsigned long long)((unsigned __int128)a * t % p);
            unsigned long long d = (unsigned long long)(((unsigned __int128)b * b + (unsigned __int128)a * a) % p);

            if ((N >> i) & 1ULL) {
                // bit is 1: new index is 2k+1, so (P(2k+1), P(2k+2))
                // P(2k+2) = 2*P(2k+1) + P(2k) = 2*d + c
                a = d;
                b = (unsigned long long)(((unsigned __int128)2 * d + c) % p);
            } else {
                // bit is 0: new index is 2k, so (P(2k), P(2k+1)) = (c, d)
                a = c;
                b = d;
            }
        }

        cout << a % p << "\n";
    }
    return 0;
}
