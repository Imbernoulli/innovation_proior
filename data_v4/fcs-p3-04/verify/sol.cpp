#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __uint128_t u128;

// Active modulus for the current test case (1 <= MOD <= 1e18 < 2^60).
static u64 MOD;

// (a * b) % MOD with a 128-bit intermediate. Requires a, b < MOD <= 1e18,
// so the product is < 1e36 < 2^128 and never overflows the u128.
static inline u64 mulmod(u64 a, u64 b) {
    return (u64)((u128)a * b % MOD);
}

// 2x2 matrix over Z_MOD.
struct Mat {
    u64 m[2][2];
};

static Mat mat_mul(const Mat &A, const Mat &B) {
    Mat C;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            u64 s = 0;
            for (int k = 0; k < 2; k++) {
                s += mulmod(A.m[i][k], B.m[k][j]);
                if (s >= MOD) s -= MOD; // s stays < 2*MOD <= 2e18 < 2^64
            }
            C.m[i][j] = s % MOD;
        }
    }
    return C;
}

static Mat mat_identity() {
    Mat I;
    I.m[0][0] = 1 % MOD; I.m[0][1] = 0;
    I.m[1][0] = 0;       I.m[1][1] = 1 % MOD;
    return I;
}

static Mat mat_pow(Mat base, u64 e) {
    Mat result = mat_identity();
    while (e > 0) {
        if (e & 1ULL) result = mat_mul(result, base);
        base = mat_mul(base, base);
        e >>= 1;
    }
    return result;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        u64 n, p;
        cin >> n >> p;
        MOD = p;

        // f(N) = number of binary strings of length N with no two adjacent ones.
        // f(0) = 1, f(1) = 2, f(N) = f(N-1) + f(N-2) for N >= 2.
        // Reduce the base values mod p up front (p may be 1, giving 0).
        u64 f0 = 1 % MOD;
        u64 f1 = 2 % MOD;

        u64 ans;
        if (n == 0) ans = f0;
        else if (n == 1) ans = f1;
        else {
            // State vector v_k = [f(k), f(k-1)]^T, with
            //   v_{k+1} = M * v_k,  M = [[1,1],[1,0]].
            // Then v_n = M^{n-1} * v_1, and f(n) = first component of v_n,
            // where v_1 = [f(1), f(0)]^T = [f1, f0]^T.
            Mat M;
            M.m[0][0] = 1 % MOD; M.m[0][1] = 1 % MOD;
            M.m[1][0] = 1 % MOD; M.m[1][1] = 0;

            Mat P = mat_pow(M, n - 1);
            // f(n) = P[0][0]*f1 + P[0][1]*f0   (mod p)
            u64 r = 0;
            r += mulmod(P.m[0][0], f1); if (r >= MOD) r -= MOD;
            r += mulmod(P.m[0][1], f0); if (r >= MOD) r -= MOD;
            ans = r % MOD;
        }

        cout << ans << "\n";
    }
    return 0;
}
