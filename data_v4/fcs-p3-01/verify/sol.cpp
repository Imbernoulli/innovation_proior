#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __uint128_t u128;

// modulus, set once per test
static u64 MOD;

// (a * b) % MOD using 128-bit intermediate (a, b already < MOD < 2^63)
static inline u64 mulmod(u64 a, u64 b) {
    return (u64)((u128)a * b % MOD);
}

// 3x3 matrix over Z_MOD
struct Mat {
    u64 m[3][3];
};

static Mat mat_mul(const Mat &A, const Mat &B) {
    Mat C;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            u64 s = 0;
            for (int k = 0; k < 3; k++) {
                s += mulmod(A.m[i][k], B.m[k][j]);
                if (s >= MOD) s -= MOD; // s < 3*MOD < 2^64 since MOD < 2^62
            }
            C.m[i][j] = s % MOD;
        }
    }
    return C;
}

static Mat mat_identity() {
    Mat I;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            I.m[i][j] = (i == j) ? (1 % MOD) : 0;
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
        u64 f0, f1, f2;
        cin >> n >> p >> f0 >> f1 >> f2;
        MOD = p;

        // reduce inputs mod p first
        u64 a0 = f0 % MOD;
        u64 a1 = f1 % MOD;
        u64 a2 = f2 % MOD;

        u64 ans;
        if (n == 0) ans = a0;
        else if (n == 1) ans = a1;
        else if (n == 2) ans = a2;
        else {
            // State vector v_k = [f(k), f(k-1), f(k-2)]^T.
            // v_{k+1} = M * v_k where
            //   M = [[1,1,1],[1,0,0],[0,1,0]]
            // Starting from v_2 = [f2, f1, f0]^T, we have v_n = M^{n-2} * v_2,
            // and f(n) = first component of v_n.
            Mat M;
            M.m[0][0] = 1 % MOD; M.m[0][1] = 1 % MOD; M.m[0][2] = 1 % MOD;
            M.m[1][0] = 1 % MOD; M.m[1][1] = 0;       M.m[1][2] = 0;
            M.m[2][0] = 0;       M.m[2][1] = 1 % MOD; M.m[2][2] = 0;

            Mat P = mat_pow(M, n - 2);
            // f(n) = P[0][0]*f2 + P[0][1]*f1 + P[0][2]*f0
            u64 r = 0;
            r += mulmod(P.m[0][0], a2); if (r >= MOD) r -= MOD;
            r += mulmod(P.m[0][1], a1); if (r >= MOD) r -= MOD;
            r += mulmod(P.m[0][2], a0); if (r >= MOD) r -= MOD;
            ans = r % MOD;
        }

        cout << ans << "\n";
    }
    return 0;
}
