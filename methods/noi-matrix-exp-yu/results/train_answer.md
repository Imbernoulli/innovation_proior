We are handed two superficially different counting problems that turn out to be the same problem. In the first, a directed graph on $d$ vertices carries an edge-count table $G$ where $G[i][j]$ is the number of edges $i \to j$, and we must count walks of length exactly $N$ from a source $s$ to a target $t$ — sequences $s = u_0 \to u_1 \to \dots \to u_N = t$, with revisits and edge reuse allowed and counted with multiplicity. In the second, a $d$-term linear recurrence $a_n = c_1 a_{n-1} + c_2 a_{n-2} + \dots + c_d a_{n-d}$ with fixed integer coefficients and $d$ given initial terms must be evaluated at index $N$. In both, $N$ can be as large as $10^{18}$ and the answer is reported modulo a given $m$. The honest first move on the recurrence is to run it forward: keep a sliding window of the last $d$ values, compute the next as the fixed weighted sum, slide, repeat $N$ times. Correct and trivial — and fatal. Each step is $O(d)$ multiply-adds and there are $N$ of them, so $O(dN)$, which at $N = 10^{18}$ is on the order of $10^{18}$ operations no matter how the constant is tuned; a machine doing $10^9$ simple operations a second needs decades. The enemy is not the cost per step but the sheer *count* of steps, the linear dependence on $N$. To survive $N = 10^{18}$ the work must grow like $\log N$ — sixty-odd units, not $10^{18}$ of them — so the real question is whether there is structure that lets us leap across many steps at once.

There is, and the method that exploits it is fast matrix exponentiation of the transition table. Stare at one step of the recurrence. From the window $(a_{n-1}, a_{n-2}, \dots, a_{n-d})$ we produce the next window $(a_n, a_{n-1}, \dots, a_{n-d+1})$: one freshly computed value $a_n = \sum_j c_j a_{n-j}$, and the old window shifted down by one with its oldest entry dropped. Two facts matter. Every entry of the new window is a *linear* combination of the old window — $a_n$ is the weighted sum, and each shifted entry is the trivial combination "copy this old entry." And the rule mapping old window to new is *identical at every index*: the coefficients $c_j$ are fixed and the shift is fixed, so step $5$ and step $5{,}000{,}000$ apply the same transformation. We are not doing $N$ different things; we are doing the *same* thing $N$ times. Naming the state $v_n = (a_n, a_{n-1}, \dots, a_{n-d+1})^\top$ (newest on top), there is a single linear map $T$ with $v_n = T\,v_{n-1}$. Reading off the components of the new window in terms of the old gives the companion matrix
$$T = \begin{pmatrix} c_1 & c_2 & c_3 & \cdots & c_d \\ 1 & 0 & 0 & \cdots & 0 \\ 0 & 1 & 0 & \cdots & 0 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & 0 \end{pmatrix},$$
whose top row is the dot product $(c_1, \dots, c_d)\cdot v_{n-1} = a_n$, and whose sub-diagonal of $1$s shifts each old entry down one slot — row $i$ for $i \ge 2$ picks out coordinate $i-1$. Iterating the same map, two steps are $T(T v_{n-2})$, and unrolling all the way to the given initial window $v_{d-1}$ gives $v_N = T^{\,N-(d-1)}\, v_{d-1}$, with $a_N$ the top entry. The forward scan has collapsed into a *power* of one fixed table: $N$ is no longer the length of a loop but an exponent on $T$. Walk counting is the same picture for free. The number of length-$1$ walks $i \to j$ is $G[i][j]$, and by choosing the midpoint $k$, multiplying the independent choices, and summing, the number of length-$2$ walks is $\sum_k G[i][k]\,G[k][j]$ — exactly the $(i,j)$ entry of $G$ times $G$. By the same midpoint identity $G^k[i][j] = \sum_l G^{k-1}[i][l]\,G[l][j]$, so the count of length-$N$ walks is the single entry $(G^N)[s][t]$. Here $T$ needs no companion bookkeeping at all; $T$ *is* the edge-count table. Both problems reduce to one task: compute the $N$-fold self-product of a fixed $d \times d$ table, mod $m$.

Forming $T^N$ as $T \cdot T \cdots T$ is still $N-1$ products, linear in $N$ — reorganized but not reduced. The way out is the same trick that computes a scalar power cheaply: the waste in $a \cdot a \cdots a$ is repeatedly multiplying by the same small thing, whereas $a^4 = (a^2)^2$ reaches exponent four in two multiplies, $a^8 = (a^4)^2$ in three, each squaring *doubling* the reachable exponent, so $a^N$ costs about $\log_2 N$ squarings. For general $N$, write it in binary, $a^N = \prod_{i:\,b_i=1} a^{2^i}$: walk the bits, keep a running square that starts at $a$ and is squared each bit, and fold it into an accumulator whenever the bit is set, for $O(\log N)$ multiplies. What licenses this regrouping for *numbers* is associativity — we are free to compute the cheap grouping because $(a\cdot a)(a\cdot a)$ and $((a\cdot a)\cdot a)\cdot a$ are the same value. To port it to tables I need the same freedom, so I must actually verify that table multiplication is associative rather than assume it. With the rule $(XY)[i,j] = \sum_k X[i,k]\,Y[k,j]$,
$$((AB)C)[i,j] = \sum_{l}\Big(\sum_{k} A[i,k]\,B[k,l]\Big) C[l,j] = \sum_{l}\sum_{k} A[i,k]\,B[k,l]\,C[l,j],$$
a finite double sum of ordinary products, which may be reordered freely:
$$= \sum_{k} A[i,k]\Big(\sum_{l} B[k,l]\,C[l,j]\Big) = (A(BC))[i,j].$$
Equal entrywise, so $(AB)C = A(BC)$ — table multiplication is associative precisely because the underlying number arithmetic is, $T^N$ is unambiguous, and squaring is licensed. So I compute the power by binary exponentiation: an accumulator $R$ starting at the identity table $I$ (the table analogue of the number $1$, which leaves any table unchanged under the product), a running power $B$ starting at $T$ and squared once per bit, folded into $R$ on each set bit. There are only $O(\log N)$ bits, hence $O(\log N)$ squarings and folds. One $d \times d$ table product costs $O(d^3)$ — each of $d^2$ output entries is a sum over $d$ terms — so the whole power is $O(d^3 \log N)$ time and $O(d^2)$ space; at $N = 10^{18}$ that is about sixty bit rounds instead of $10^{18}$ scan steps. The modulus slots in cleanly because reduction mod $m$ commutes with both $+$ and $\times$, so keeping every entry reduced into $[0,m)$ after each operation yields the true counts mod $m$ while the gigantic exact counts are never formed — the only reason the answer is even representable. A small Fibonacci check confirms the wiring: with $d=2$, $c_1=c_2=1$, $T = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$ acts as $T\binom{a_{n-1}}{a_{n-2}} = \binom{a_n}{a_{n-1}}$, and indeed $T^n = \begin{pmatrix} F_{n+1} & F_n \\ F_n & F_{n-1} \end{pmatrix}$. One boundary case remains: when $N \le d-1$ for the recurrence the answer is simply a given initial term and the exponent $N-(d-1)$ would go negative, so I guard it by returning $a_N \bmod m$ directly when $N < d$; for the walk problem $N=0$ is the empty walk, the identity table, which the loop returns correctly since $R$ starts at $I$ and the loop is never entered.

The deliverable is a single self-contained C++17 program reading from stdin: the first token selects the form — `W` then `d s t N m` followed by the $d\times d$ edge-count table prints the length-$N$ walk count $s\to t$; `L` then `d N m`, the $d$ coefficients and $d$ initial terms prints $a_N$ — every entry kept reduced mod $m$, with the one product before each reduction widened to `__int128` so it cannot overflow.

```cpp
// Reads from stdin either a walk-counting query or a linear-recurrence query and
// prints the single integer answer mod m to stdout.
//   First token = mode: "W" (walks) or "L" (linrec).
//   W:  d  s  t  N  m,  then a d x d edge-count table adj[i][j] (# edges i->j);
//       prints the number of length-N walks s -> t, mod m.
//   L:  d  N  m,  then d coefficients c_1..c_d, then d initial terms a_0..a_{d-1};
//       prints a_N for  a_n = c_1 a_{n-1} + ... + c_d a_{n-d},  mod m.
// N may be as large as 1e18; everything is kept reduced mod m via O(d^3 log N)
// fast matrix exponentiation (entries stay in [0,m), products fit in __int128).
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<vector<ll>> Mat;

static ll MOD;

// (p x q) by (q x r) product, every entry reduced mod MOD.
Mat mul(const Mat &A, const Mat &B) {
    int p = A.size(), q = B.size(), r = B[0].size();
    Mat C(p, vector<ll>(r, 0));
    for (int i = 0; i < p; i++) {
        for (int k = 0; k < q; k++) {
            ll a = A[i][k];
            if (a == 0) continue;
            const vector<ll> &Bk = B[k];
            vector<ll> &Ci = C[i];
            for (int j = 0; j < r; j++) {
                Ci[j] = (Ci[j] + (__int128)a * Bk[j]) % MOD;
            }
        }
    }
    return C;
}

Mat identity(int d) {
    Mat I(d, vector<ll>(d, 0));
    for (int i = 0; i < d; i++) I[i][i] = 1 % MOD;
    return I;
}

// M^N mod MOD by binary exponentiation; O(d^3 log N) ring multiplications.
Mat matpow(Mat M, ll N) {
    int d = M.size();
    Mat R = identity(d);
    for (auto &row : M) for (auto &x : row) x %= MOD;
    while (N > 0) {
        if (N & 1) R = mul(R, M);
        M = mul(M, M);
        N >>= 1;
    }
    return R;
}

// Number of length-N walks s -> t in a directed graph, mod MOD.
ll walks(const Mat &adj, int s, int t, ll N) {
    return matpow(adj, N)[s][t];
}

// Evaluate a_n = sum_j coeffs[j]*a_{n-1-j} at index N, given init = [a_0..a_{d-1}].
ll linrec(const vector<ll> &coeffs, const vector<ll> &init, ll N) {
    int d = coeffs.size();
    if (N < d) return ((init[N] % MOD) + MOD) % MOD;
    Mat T(d, vector<ll>(d, 0));
    for (int j = 0; j < d; j++) T[0][j] = ((coeffs[j] % MOD) + MOD) % MOD; // top row = coefficients
    for (int i = 1; i < d; i++) T[i][i - 1] = 1 % MOD;                     // sub-diagonal shift
    Mat v0(d, vector<ll>(1, 0));
    for (int i = 0; i < d; i++) v0[i][0] = ((init[d - 1 - i] % MOD) + MOD) % MOD; // (a_{d-1},...,a_0)^T
    Mat vN = mul(matpow(T, N - (d - 1)), v0);
    return vN[0][0];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string mode;
    if (!(cin >> mode)) return 0;
    if (mode == "W" || mode == "w") {
        int d, s, t; ll N;
        cin >> d >> s >> t >> N >> MOD;
        Mat adj(d, vector<ll>(d, 0));
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) cin >> adj[i][j];
        cout << walks(adj, s, t, N) << "\n";
    } else { // "L" linear recurrence
        int d; ll N;
        cin >> d >> N >> MOD;
        vector<ll> coeffs(d), init(d);
        for (int i = 0; i < d; i++) cin >> coeffs[i];
        for (int i = 0; i < d; i++) cin >> init[i];
        cout << linrec(coeffs, init, N) << "\n";
    }
    return 0;
}
```
