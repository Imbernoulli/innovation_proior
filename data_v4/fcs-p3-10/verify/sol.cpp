#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __uint128_t u128;

// modulus is per-query; we keep it in a small struct of fixed-size 3x3 matrices.
struct Mat {
    u64 m[3][3];
};

static u64 MOD;

// (a*b) % MOD with 128-bit intermediate; a,b already reduced < MOD.
static inline u64 mulmod(u64 a, u64 b) {
    return (u64)((u128)a * b % MOD);
}

static Mat mul(const Mat &A, const Mat &B) {
    Mat C;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            u64 s = 0;
            for (int k = 0; k < 3; k++) {
                s += mulmod(A.m[i][k], B.m[k][j]);
                if (s >= MOD) s -= MOD;            // each term < MOD, so s stays < 2*MOD < 2^63
            }
            C.m[i][j] = s % MOD;
        }
    }
    return C;
}

static Mat identity() {
    Mat I;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            I.m[i][j] = (i == j) ? (1 % MOD) : 0;
    return I;
}

static Mat matpow(Mat base, u64 e) {
    Mat r = identity();
    while (e) {
        if (e & 1ULL) r = mul(r, base);
        base = mul(base, base);
        e >>= 1;
    }
    return r;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    while (q--) {
        long long a_in, b_in, c_in, d_in;
        unsigned long long N;
        long long p_in;
        cin >> a_in >> b_in >> c_in >> d_in >> N >> p_in;

        MOD = (u64)p_in;

        // S(N) = sum_{i=0}^{N-1} f(i)  (sum of first N terms)
        // f(0)=a, f(1)=b, f(i)=c*f(i-1)+d*f(i-2)
        if (MOD == 1) { cout << 0 << "\n"; continue; }

        // reduce inputs into [0, MOD)
        auto red = [](long long v, u64 mod) -> u64 {
            long long r = v % (long long)mod;
            if (r < 0) r += (long long)mod;
            return (u64)r;
        };
        u64 a = red(a_in, MOD);
        u64 b = red(b_in, MOD);
        u64 c = red(c_in, MOD);
        u64 d = red(d_in, MOD);

        if (N == 0ULL) { cout << 0 << "\n"; continue; }

        // N >= 1: S(0) = f(0) = a   (sum of first 1 term)
        if (N == 1ULL) { cout << a % MOD << "\n"; continue; }

        // State vector at index i (>=1): v_i = [ f(i), f(i-1), Spref(i) ]^T
        // where Spref(i) = sum_{j=0}^{i} f(j) = S(i+1) (sum of first i+1 terms).
        // Base i=1: v_1 = [ f(1), f(0), f(0)+f(1) ] = [ b, a, (a+b) mod ].
        // Transition v_{i+1} = M v_i with
        //   f(i+1)   = c*f(i) + d*f(i-1)
        //   f(i)     = f(i)
        //   Spref(i+1) = Spref(i) + f(i+1) = c*f(i) + d*f(i-1) + Spref(i)
        // M (rows = new, cols = [f(i), f(i-1), Spref(i)]):
        //   [ c d 0 ]
        //   [ 1 0 0 ]
        //   [ c d 1 ]
        Mat M;
        M.m[0][0] = c; M.m[0][1] = d; M.m[0][2] = 0;
        M.m[1][0] = 1 % MOD; M.m[1][1] = 0; M.m[1][2] = 0;
        M.m[2][0] = c; M.m[2][1] = d; M.m[2][2] = 1 % MOD;

        // We want S(N) = sum of first N terms = Spref(N-1).
        // v_1 corresponds to i=1; applying M^(N-2) reaches v_{N-1}, whose
        // third component is Spref(N-1) = S(N).  N >= 2 here.
        Mat P = matpow(M, (u64)(N - 2));

        u64 v0 = b;                 // f(1)
        u64 v1 = a;                 // f(0)
        u64 v2 = (a + b) % MOD;     // Spref(1) = f(0)+f(1)

        // result third row of P times v
        u64 res = ( mulmod(P.m[2][0], v0)
                  + mulmod(P.m[2][1], v1)
                  + mulmod(P.m[2][2], v2) ) % MOD;

        cout << res << "\n";
    }
    return 0;
}
