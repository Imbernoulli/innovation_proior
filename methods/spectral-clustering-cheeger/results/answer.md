# Spectral clustering and the discrete Cheeger inequality

## Problem

Partition the vertices of a graph $G=(V,E,w)$ into two well-separated clusters by minimizing the
**conductance** of a cut $S\subseteq V$,

$$\phi(S)=\frac{|\partial(S)|}{\min(d(S),\,d(V\setminus S))},\qquad
\phi_G=\min_{\varnothing\ne S\subsetneq V}\phi(S),$$

where $\partial(S)$ are the edges leaving $S$ and $d(S)=\sum_{u\in S}d(u)$ is its volume. Finding the
minimum-conductance / sparsest cut is NP-hard. We want a polynomial-time proxy with a *provable*
relationship to $\phi_G$.

## Key idea

The cut size of an indicator is a Laplacian quadratic form, $\chi_S^{T}L\chi_S=|\partial(S)|$ with
$L=D-A$ and $x^{T}Lx=\sum_{(u,v)\in E}w_{u,v}(x(u)-x(v))^2$. Relaxing the $0/1$ indicator to a real
vector orthogonal to the kernel $\mathbf 1$ (resp. to $d$, for the volume-normalized version) turns
conductance into a Rayleigh quotient whose minimum is the second-smallest eigenvalue of the
(normalized) Laplacian. That eigenvalue therefore brackets the sparsest cut from both sides, and
thresholding its eigenvector (the **Fiedler vector**) yields a cut achieving the upper side.

## The Cheeger inequality (discrete)

Let $N=D^{-1/2}LD^{-1/2}$ be the normalized Laplacian, $0=\nu_1\le\nu_2\le\cdots\le\nu_n$ its
eigenvalues. Then

$$\boxed{\ \frac{\nu_2}{2}\ \le\ \phi_G\ \le\ \sqrt{2\,\nu_2}\ }$$

where $\nu_2=\min_{y\perp d,\ y\ne0}\dfrac{y^{T}Ly}{y^{T}Dy}$, and a cut meeting the upper bound is
found by sweeping thresholds of the Fiedler vector.

**Easy direction $\nu_2/2\le\phi_G$.** For the optimal $S$, set $y=\chi_S-\sigma\mathbf 1$,
$\sigma=d(S)/d(V)$. Then $y\perp d$, $y^{T}Ly=|\partial(S)|$, and
$y^{T}Dy=d(S)d(V\setminus S)/d(V)$, so $\nu_2\le |\partial(S)|d(V)/(d(S)d(V\setminus S))\le 2\phi(S)$,
since $\max(d(S),d(V\setminus S))\ge d(V)/2$.

**Hard direction $\phi_G\le\sqrt{2\nu_2}$ (sweep rounding; Trevisan-style proof).** Let $y\perp d$
have Rayleigh quotient $\rho=y^{T}Ly/y^{T}Dy$ (take $y$ the eigenvector, $\rho=\nu_2$). Sort
$y(1)\le\cdots\le y(n)$, let $j$ be the half-volume median ($\sum_{u\le j}d(u)\ge d(V)/2$ least),
and center $z=y-y(j)\mathbf 1$. The numerator is unchanged, and since $y\perp d$ minimizes
$(y+t\mathbf 1)^TD(y+t\mathbf 1)$ over shifts, $z^{T}Dz\ge y^{T}Dy$, hence
$z^{T}Lz/z^{T}Dz\le\rho$. Scale $z(1)^2+z(n)^2=1$.
Consider sweep cuts $S_t=\{u:z(u)\le t\}$ with $t$ drawn at density $2|t|$ on $[z(1),z(n)]$
($\int 2|t|=z(1)^2+z(n)^2=1$). Then:

- $\displaystyle \mathbb E[|\partial(S_t)|]=\sum_{(u,v)\in E}w_{u,v}\Pr[(u,v)\in\partial(S_t)]
  \le\sum_{(u,v)\in E}w_{u,v}|z(u)-z(v)|(|z(u)|+|z(v)|)$, because in both the same-sign case
  $|z(u)^2-z(v)^2|$ and the opposite-sign case $z(u)^2+z(v)^2$ are $\le|z(u)-z(v)|(|z(u)|+|z(v)|)$.
- $\displaystyle \mathbb E[\min(d(S_t),d(V\setminus S_t))]=\sum_u z(u)^2 d(u)=z^{T}Dz$: the median
  choice gives $d(\{z<0\})<d(V)/2$ and $d(\{z\le0\})\ge d(V)/2$, so negative thresholds use $S_t$
  as the smaller side and nonnegative thresholds use $V\setminus S_t$.
- Cauchy–Schwarz: $\sum_{(u,v)}w_{u,v}|z(u)-z(v)|(|z(u)|+|z(v)|)\le\sqrt{z^{T}Lz}\,\sqrt{\sum_{(u,v)}w_{u,v}(|z(u)|+|z(v)|)^2}
  \le\sqrt{\rho\,z^{T}Dz}\,\sqrt{2\,z^{T}Dz}=\sqrt{2\rho}\,z^{T}Dz$,
  using $z^{T}Lz\le\rho z^{T}Dz$ and
  $\sum_{(u,v)}w_{u,v}(|z(u)|+|z(v)|)^2\le 2\sum_u z(u)^2d(u)=2z^{T}Dz$.

So $\mathbb E[|\partial(S_t)|]\le\sqrt{2\rho}\,\mathbb E[\min(d(S_t),d(V\setminus S_t))]$, whence some
threshold $t$ gives $\phi(S_t)\le\sqrt{2\rho}=\sqrt{2\nu_2}$. $\qquad\blacksquare$

The rounding used only $y\perp d$, not that $y$ is an eigenvector. The $\sqrt{\cdot}$ gap is tight:
a cycle $C_n$ has $\nu_2=1-\cos(2\pi/n)=\Theta(1/n^2)$ but $\phi_G=\Theta(1/n)$, so
$\phi_G\asymp\sqrt{\nu_2}$.

## Algorithm

A self-contained C++17 program. It reads a weighted undirected graph from stdin
(`n m`, then `m` lines `u v w`), forms the normalized Laplacian
$N=D^{-1/2}LD^{-1/2}$, computes the Fiedler vector with a built-in symmetric
Jacobi eigensolver, sweeps the $n-1$ threshold cuts keeping the least-conductance
one, and writes $\nu_2$, the cut conductance, the Cheeger bracket, and the smaller
side $S$ to stdout.

```cpp
// Spectral clustering via the discrete Cheeger inequality.
// Reads a weighted undirected graph from stdin and prints a low-conductance cut
// (the sweep over the Fiedler vector) together with the Cheeger bracket
// nu2/2 <= phi_G <= phi(cut) <= sqrt(2*nu2).
//
// Input (stdin):
//   n m                      n = #vertices (0-indexed 0..n-1), m = #edges
//   u v w   (m lines)        undirected edge {u,v} with positive weight w
// Output (stdout):
//   nu2 best_phi  lower=nu2/2 upper=sqrt(2*nu2)
//   k                        size of the returned cut S (smaller side listed)
//   the k vertices of S, ascending

#include <bits/stdc++.h>
using namespace std;

// Symmetric eigendecomposition by the cyclic Jacobi rotation method.
// A (n x n, row-major) is overwritten; eigenvalues -> eval, eigenvectors -> columns of evec.
static void jacobiEigen(vector<double>& A, int n, vector<double>& eval, vector<double>& evec) {
    evec.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) evec[(size_t)i * n + i] = 1.0;
    const int maxSweeps = 100;
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double apq = A[(size_t)p * n + q];
                if (fabs(apq) < 1e-300) continue;
                double app = A[(size_t)p * n + p];
                double aqq = A[(size_t)q * n + q];
                double phi = 0.5 * atan2(2.0 * apq, aqq - app);
                double c = cos(phi), s = sin(phi);
                for (int k = 0; k < n; ++k) {
                    double akp = A[(size_t)k * n + p];
                    double akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - s * akq;
                    A[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    double apk = A[(size_t)p * n + k];
                    double aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - s * aqk;
                    A[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    double vkp = evec[(size_t)k * n + p];
                    double vkq = evec[(size_t)k * n + q];
                    evec[(size_t)k * n + p] = c * vkp - s * vkq;
                    evec[(size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    eval.resize(n);
    for (int i = 0; i < n; ++i) eval[i] = A[(size_t)i * n + i];
}

static bool lexLessVec(const vector<int>& a, const vector<int>& b) {
    return lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

static vector<int> listedSide(const vector<char>& inS, const vector<double>& deg,
                              double dV, double volS) {
    vector<int> side, comp;
    side.reserve(inS.size());
    comp.reserve(inS.size());
    for (int i = 0; i < (int)inS.size(); ++i) {
        if (inS[i]) side.push_back(i);
        else comp.push_back(i);
    }
    double volC = dV - volS;
    const double EPS = 1e-12;
    if (volS < volC - EPS) return side;
    if (volC < volS - EPS) return comp;
    return lexLessVec(side, comp) ? side : comp;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // Dense weighted adjacency and degrees.
    vector<double> adj((size_t)n * n, 0.0);
    vector<double> deg(n, 0.0);
    vector<array<int, 2>> edges;       // store endpoints for boundary scoring
    vector<double> ew;                 // edge weights
    edges.reserve(m);
    ew.reserve(m);
    for (int e = 0; e < m; ++e) {
        int u, v; double w;
        cin >> u >> v >> w;
        adj[(size_t)u * n + v] += w;
        adj[(size_t)v * n + u] += w;
        deg[u] += w;
        deg[v] += w;
        edges.push_back({u, v});
        ew.push_back(w);
    }

    double dV = 0.0;
    for (int i = 0; i < n; ++i) dV += deg[i];

    // Normalized Laplacian N = D^{-1/2} L D^{-1/2}, L = D - A.
    // N(u,u) = 1 (if deg>0), N(u,v) = -A(u,v)/sqrt(d(u) d(v)).
    bool posdeg = true;
    for (int i = 0; i < n; ++i) if (deg[i] <= 0.0) posdeg = false;

    vector<double> y(n, 0.0);   // Fiedler vector y = D^{-1/2} x2, orthogonal to d
    double nu2 = 0.0;

    if (posdeg && n >= 2) {
        vector<double> N((size_t)n * n, 0.0);
        vector<double> dinv(n);
        for (int i = 0; i < n; ++i) dinv[i] = 1.0 / sqrt(deg[i]);
        for (int i = 0; i < n; ++i) N[(size_t)i * n + i] = 1.0;
        for (int u = 0; u < n; ++u)
            for (int v = 0; v < n; ++v)
                if (u != v && adj[(size_t)u * n + v] != 0.0)
                    N[(size_t)u * n + v] = -adj[(size_t)u * n + v] * dinv[u] * dinv[v];

        vector<double> eval, evec;
        jacobiEigen(N, n, eval, evec);

        // Order eigenvalues ascending; the 2nd smallest is nu2, its eigenvector x2.
        vector<int> ord(n);
        iota(ord.begin(), ord.end(), 0);
        sort(ord.begin(), ord.end(), [&](int a, int b) { return eval[a] < eval[b]; });
        int idx2 = ord[1];
        nu2 = max(eval[idx2], 0.0);
        for (int i = 0; i < n; ++i)
            y[i] = evec[(size_t)i * n + idx2] * dinv[i];   // map back: y = D^{-1/2} x2
    }

    // Fix the arbitrary eigenvector sign so equal-quality complementary sweeps
    // print a deterministic side.
    for (double val : y) {
        if (fabs(val) > 1e-12) {
            if (val > 0.0)
                for (double& z : y) z = -z;
            break;
        }
    }

    // Sweep: sort vertices by y, try the n-1 prefix cuts, keep least conductance.
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (fabs(y[a] - y[b]) > 1e-12) return y[a] < y[b];
        return a < b;
    });

    vector<char> inS(n, 0);
    vector<int> bestS;
    double bestPhi = numeric_limits<double>::infinity();
    double volS = 0.0;
    bool haveBest = false;

    // Boundary weight is maintained incrementally as vertices enter S.
    double boundary = 0.0;
    for (int k = 0; k < n - 1; ++k) {
        int u = order[k];
        inS[u] = 1;
        volS += deg[u];
        // update boundary: edges from u to outside add, edges from u to inside subtract
        for (int v = 0; v < n; ++v) {
            double w = adj[(size_t)u * n + v];
            if (w == 0.0) continue;
            if (inS[v]) boundary -= w;   // v already in S: this edge no longer crosses
            else        boundary += w;   // v outside S: this edge now crosses
        }
        double volC = dV - volS;
        double mn = min(volS, volC);
        if (mn <= 0.0) continue;
        double phi = boundary / mn;
        vector<int> candidate = listedSide(inS, deg, dV, volS);
        const double EPS = 1e-12;
        if (!haveBest || phi < bestPhi - EPS ||
            (fabs(phi - bestPhi) <= EPS && lexLessVec(candidate, bestS))) {
            bestPhi = phi;
            bestS = move(candidate);
            haveBest = true;
        }
    }

    double lower = nu2 / 2.0;
    double upper = sqrt(2.0 * nu2);

    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    cout << nu2 << " " << bestPhi << "  lower=" << lower << " upper=" << upper << "\n";
    cout << bestS.size() << "\n";
    for (size_t i = 0; i < bestS.size(); ++i)
        cout << bestS[i] << (i + 1 < bestS.size() ? ' ' : '\n');
    if (bestS.empty()) cout << "\n";
    return 0;
}
```

This is **spectral clustering / spectral bisection** by the Fiedler vector: a continuous
eigenvector relaxation of an NP-hard combinatorial cut, rounded by a sweep, provably within a
square-root factor of optimal via the discrete Cheeger inequality.
