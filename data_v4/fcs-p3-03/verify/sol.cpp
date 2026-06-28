#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;

int MOD; // prime modulus, given on input

// multiply two m x m matrices mod MOD
static vector<vector<ll>> matmul(const vector<vector<ll>>& A,
                                 const vector<vector<ll>>& B, int m) {
    vector<vector<ll>> C(m, vector<ll>(m, 0));
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < m; k++) {
            if (A[i][k] == 0) continue;
            ll aik = A[i][k];
            const vector<ll>& Bk = B[k];
            vector<ll>& Ci = C[i];
            for (int j = 0; j < m; j++) {
                Ci[j] = (Ci[j] + aik * Bk[j]) % MOD;
            }
        }
    }
    return C;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll N;          // number of stairs
    int k;         // size of the step set S
    int p;         // prime modulus
    if (!(cin >> N >> k >> p)) return 0;
    MOD = p;

    // Read the step set S; deduplicate and find the maximum step m.
    vector<int> raw(k);
    int m = 0;
    for (int i = 0; i < k; i++) {
        cin >> raw[i];
        m = max(m, raw[i]);
    }
    // mark[s] = true if step size s (1..m) is allowed
    vector<char> allowed(m + 1, 0);
    for (int i = 0; i < k; i++) allowed[raw[i]] = 1;

    // Recurrence: f(0) = 1, f(n) = sum_{s in S, s <= n} f(n - s) for n >= 1.
    // f(n) counts ordered compositions of n using parts from S.
    // Order of the linear recurrence is m (the largest allowed step):
    //   f(n) = sum_{s=1..m} allowed[s] * f(n - s),  valid for n >= m.

    if (N == 0) {
        // The empty climb: exactly one way (take no steps).
        cout << (1 % MOD) << "\n";
        return 0;
    }

    // Compute base values f(0..m-1) directly with the recurrence (small range).
    // f(n) = sum_{s=1..min(n,m)} allowed[s] * f(n-s)  (only s with s<=n contribute).
    vector<ll> base(m, 0);
    base[0] = 1 % MOD; // f(0)
    for (int n = 1; n < m; n++) {
        ll v = 0;
        for (int s = 1; s <= n && s <= m; s++) {
            if (allowed[s]) v += base[n - s];
        }
        base[n] = v % MOD;
    }

    if (N < m) {
        cout << base[(int)N] << "\n";
        return 0;
    }

    // For N >= m use matrix exponentiation on the state
    // vector v_n = [f(n), f(n-1), ..., f(n-m+1)]^T.
    // Transition T maps v_{n} -> v_{n+1}:
    //   row 0: coefficients allowed[1..m]   (f(n+1) = sum allowed[s] f(n+1-s))
    //   rows 1..m-1: shift down (identity on subdiagonal).
    // We start from v_{m-1} = [f(m-1),...,f(0)] and apply T^(N-(m-1)).
    vector<vector<ll>> T(m, vector<ll>(m, 0));
    for (int s = 1; s <= m; s++) T[0][s - 1] = allowed[s] ? 1 : 0;
    for (int r = 1; r < m; r++) T[r][r - 1] = 1;

    // Matrix power T^e  with e = N - (m-1).
    ll e = N - (m - 1);
    // identity
    vector<vector<ll>> R(m, vector<ll>(m, 0));
    for (int i = 0; i < m; i++) R[i][i] = 1 % MOD;
    vector<vector<ll>> P = T;
    while (e > 0) {
        if (e & 1) R = matmul(R, P, m);
        e >>= 1;
        if (e) P = matmul(P, P, m);
    }

    // v_{m-1} = [f(m-1), f(m-2), ..., f(0)]
    // result f(N) = (R * v_{m-1})[0]
    ll ans = 0;
    for (int j = 0; j < m; j++) {
        // base index: row j of v_{m-1} is f(m-1-j)
        ll fv = base[m - 1 - j];
        ans = (ans + R[0][j] * fv) % MOD;
    }
    cout << ans << "\n";
    return 0;
}
