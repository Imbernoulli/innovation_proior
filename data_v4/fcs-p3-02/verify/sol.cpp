#include <bits/stdc++.h>
using namespace std;

// Count tilings of a 2xN board with 1x2 dominoes, modulo p.
// T(0)=1, T(1)=1, T(N)=T(N-1)+T(N-2)  (so T(N) = Fibonacci(N+1)).
// N can be up to 1e18, so we raise the 2x2 transition matrix
//   M = [[1,1],[1,0]]  to the N-th power by binary exponentiation.
// [[T(N+1)],[T(N)]] = M^N * [[T(1)],[T(0)]] = M^N * [[1],[1]] ... but
// cleaner: M^N = [[F(N+1),F(N)],[F(N),F(N-1)]] with F(0)=0,F(1)=1, and
// T(N) = F(N+1). We extract T(N) = (M^N)[0][0] because (M^N)[0][0]=F(N+1).

struct Mat {
    long long a, b, c, d; // [[a,b],[c,d]]
};

static Mat mul(const Mat &x, const Mat &y, long long p) {
    // entries are in [0,p), p <= 1e9 so products fit in long long (< 1e18).
    Mat r;
    r.a = ((x.a * y.a) % p + (x.b * y.c) % p) % p;
    r.b = ((x.a * y.b) % p + (x.b * y.d) % p) % p;
    r.c = ((x.c * y.a) % p + (x.d * y.c) % p) % p;
    r.d = ((x.c * y.b) % p + (x.d * y.d) % p) % p;
    return r;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, p;
        cin >> n >> p;

        // Identity matrix reduced mod p (handles p == 1 gracefully: 1 % 1 == 0).
        Mat result = {1 % p, 0 % p, 0 % p, 1 % p};
        Mat base = {1 % p, 1 % p, 1 % p, 0 % p};

        long long e = n;
        while (e > 0) {
            if (e & 1LL) result = mul(result, base, p);
            base = mul(base, base, p);
            e >>= 1;
        }
        // (M^N)[0][0] = F(N+1) = T(N).
        long long ans = result.a % p;
        cout << ans << "\n";
    }
    return 0;
}
