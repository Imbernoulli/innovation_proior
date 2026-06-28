# Electrical Flows for Approximate Maximum Flow

## Problem

Compute a `(1−ε)`-approximate maximum `s-t` flow (and a `(1+ε)`-approximate minimum
`s-t` cut) in an undirected graph `G = (V, E)` with `n` vertices, `m` edges, integer
capacities `u_e`. The yardstick to beat is the `O(m^{3/2})` / `Õ(m√n)`-style running time of
path-augmentation and blocking-flow methods, which sit inside the `Ω(mn)` flow-decomposition
barrier.

## Key idea

Replace path-based augmentation with a non-combinatorial primitive: the **electrical flow**,
the minimum-energy (`ℓ_2`) `s-t` flow, computed by a single Laplacian linear-system solve in
nearly-linear time. Max flow is an `ℓ_∞` problem (bound the worst congestion); electrical flow
is the `ℓ_2` relaxation, which can overload an edge by up to `√m`. **Multiplicative weights**
turns this capacity-oblivious oracle into a feasible flow, with iteration count proportional to
the oracle's **width** (worst congestion). The width is `√m` for a plain electrical flow, but
the overloaded edges are *fragile*: removing any edge that exceeds a target width `ρ` and
recomputing — while tracking the **effective resistance** as a monotone potential that jumps
each time such an edge is cut — keeps the number of removals small. Balancing iterations against
removals gives `ρ ≈ m^{1/3}` and running time `Õ(m^{4/3} ε^{-3})`.

## The electrical-flow primitive

Assign resistance `r_e > 0`, conductance `c_e = 1/r_e`, `C = diag(c_e)`, incidence matrix `B`,
Laplacian `L = B C Bᵀ`. The electrical `s-t` flow of value `F` minimizes the energy
`E_r(f) = Σ_e r_e f(e)²` over flows with `Bf = Fχ_{s,t}`; it is a potential flow `f = C Bᵀφ`
with `φ = L⁺(Fχ_{s,t})`, and `E_r(f) = F² R_eff(r)` where `R_eff(r) = χᵀL⁺χ` is the effective
`s-t` resistance. `L` is symmetric diagonally dominant, so `Lφ = Fχ` is solved approximately in
`Õ(m log(1/δ))` time (Koutis-Miller-Peng; Spielman-Teng), returning `φ̂` with
`‖φ̂ − φ‖_L ≤ δ'‖φ‖_L`.

Two facts that drive the analysis:

- **Effective conductance / Thomson's principle.**
  `C_eff(r) = 1/R_eff(r) = min_{φ: φ_s=1, φ_t=0} Σ_{(u,v)} (φ_u − φ_v)²/r_{uv}`,
  minimized by the electrical potentials. Hence **Rayleigh monotonicity**: `r' ≥ r ⇒ R_eff(r') ≥ R_eff(r)`.

- **Effect of a resistance increase.** If edge `h` carries a `β` fraction of the energy
  (`f(h)² r_h = β E_r(f)`) and `r_h → γ r_h`, then
  `R_eff(r') ≥ (γ/(β + γ(1−β))) R_eff(r)`. In particular cutting `h` (`γ = ∞`) gives
  `R_eff(r') ≥ R_eff(r)/(1−β)`; a bump `γ = 1+ε` gives `R_eff(r') ≥ (1 + εβ/2) R_eff(r)`.

  *Proof.* Normalize `f` to the unit-conductance flow, `φ_s = 1, φ_t = 0`; then
  `C_eff(r) = Σ (φ_u−φ_v)²/r_{uv}`, the `h`-term is `βC_eff(r)`, the rest `(1−β)C_eff(r)`.
  Plugging this same `φ` into the min for `r'` (an upper bound):
  `C_eff(r') ≤ (β/γ)C_eff(r) + (1−β)C_eff(r) = C_eff(r)(β + γ(1−β))/γ`. Invert. ∎

## Multiplicative-weights oracle (width `3√(m/ε)`)

Maintain weights `w_e ≥ 1`. Given `w` and target value `F`, set
`r_e = (1/u_e²)(w_e + ε‖w‖_1/(3m))`, compute the `(ε/3)`-approximate electrical flow `f̃` of
value `F`, fail if `E_r(f̃) > (1+ε)‖w‖_1`, else return `f̃`. The `w_e` term ties energy to the
*weighted-average* congestion; the additive floor `ε‖w‖_1/3m` caps the *worst* congestion.

**Lemma (oracle).** This is an `(ε, 3√(m/ε))`-oracle: when `F ≤ F*` it never fails, and any
returned `f̃` satisfies `Σ_e w_e cong(f̃,e) ≤ (1+ε)‖w‖_1` and `max_e cong(f̃,e) ≤ 3√(m/ε)`.

*Proof.* For feasible `f*`, `cong ≤ 1` so `E_r(f*) ≤ (1+ε/3)‖w‖_1`; since electrical flow
minimizes energy and the approximation costs `(1+ε/3)`, `E_r(f̃) ≤ (1+ε/3)²‖w‖_1 ≤ (1+ε)‖w‖_1`
when `F ≤ F*`. Given `E_r(f̃) ≤ (1+ε)‖w‖_1`: dropping the floor, `Σ w_e cong² ≤ (1+ε)‖w‖_1`, and
Cauchy-Schwarz gives `Σ w_e cong ≤ √(1+ε)‖w‖_1`. Keeping only the floor,
`(ε‖w‖_1/3m)cong_e² ≤ (1+ε)‖w‖_1`, so `cong_e ≤ √(3m(1+ε)/ε) ≤ 3√(m/ε)`. ∎

## Multiplicative-weights theorem

```
MaxFlowMW(G, F, oracle O with width ρ, ε):
  w_e ← 1 for all e;  N ← 2 ρ ln m / ε²
  for i = 1..N:
    f^i ← O(w, F);  if fail: return FAIL
    w_e ← w_e · (1 + (ε/ρ) cong(f^i, e))   for all e
  return  f̄ ← ((1−ε)² / ((1+ε) N)) · Σ_i f^i
```

**Theorem.** Given an `(ε, ρ)`-oracle of running time `T`, `MaxFlowMW` computes a
`(1−O(ε))`-approximate maximum flow in `Õ(ρ ε^{-2} · T)` time.

*Proof.* Potential `μ_i = ‖w^i‖_1`. From the average bound,
`μ_{i+1} ≤ μ_i(1 + ε(1+ε)/ρ) ≤ μ_i exp(ε(1+ε)/ρ)`, so `μ_N ≤ m exp(ε(1+ε)N/ρ)`. Per edge,
using `1 + εx ≥ exp((1−ε)εx)` for `x = cong/ρ ∈ [0,1]`,
`w_e^N ≥ exp((1−ε)(ε/ρ) Σ_i cong(f^i,e)) ≥ exp((1−ε)(ε/ρ) N cong(f̄_+,e))`, where `f̄_+` is the
average flow. Since `w_e^N ≤ μ_N`, taking logs and using `N = 2ρ ln m/ε²`, the per-edge
congestion of the scaled return `f̄` satisfies `cong(f̄,e) ≤ 1 − ε + ε(1−ε)/(2(1+ε)) ≤ 1`, so
`f̄` is feasible. Each `f^i` has value `F`, so `f̄` has value `(1−ε)²/(1+ε)·F ≥ (1−O(ε))F`. ∎

Combining with the `√(m/ε)`-oracle gives a **`(1−ε)`-approximate max flow in
`Õ(m^{3/2} ε^{-5/2})`** time.

## Width reduction → `Õ(m^{4/3} ε^{-3})`

The bad graph (`k` parallel length-`k` paths plus one direct `s-t` edge, `m = Θ(k²)`) forces
`Θ(√m)` flow on the direct edge — `ρ = Θ(√m)` is tight for a single electrical flow. But that
edge is fragile. Use the modified oracle:

```
ImprovedOracle(w, F, H, ρ):                      # ρ = 8 m^{1/3} ln^{1/3}m / ε
  r_e ← (1/u_e²)(w_e + ε|w|₁/(3m))   for e ∈ E∖H
  f̃ ← approx electrical flow on G_H = (V, E∖H), value F, accuracy δ=ε/3
  if E_r(f̃) > (1+ε)|w|₁  or  s,t disconnected in G_H:  return FAIL
  if ∃ e with cong(f̃,e) > ρ:  add e to H;  restart
  return f̃, H
```

**Lemma (removals).** Throughout the algorithm `|H| ≤ 30 m ln m/(ε²ρ²)` and
`u(H) ≤ 30 m F ln m/(ε²ρ³)`. With `ρ = 8 m^{1/3}ln^{1/3}m/ε`: `|H| ≤ (15/32)(m ln m)^{1/3}` and
`u(H) < εF/12`.

*Proof sketch.* Potential `Φ(j) = R_eff(r^j)` (removed edges at `r = ∞`). (1) `Φ` never
decreases (Rayleigh). (2) `Φ(1) ≥ m^{-4}F^{-2}` (the unit flow sends `≥ 1/m` across the min cut,
whose edges have `r ≥ 1/F*²`, and `F* ≤ mF`). (3) Each removed edge has `cong > ρ`, so via the
floor it carries `> ερ²/(5m)` of the energy (transferred to the exact flow by the solver's
per-edge guarantee), and cutting it multiplies `Φ` by `≥ 1/(1−ερ²/5m)`. Combining with
`Φ(j) ≤ (1+ε)‖w‖_1/F² ≤ 2m⁵ exp(3ε^{-1}ln m)·Φ(1)` and `ln(1−c) < −c` yields the cardinality
bound; `u_e < F/ρ` (since `> ρu_e` units flow over a removed edge but never `> F`) gives the
capacity bound. ∎

Because `u(H) < εF/12`, a feasible flow of value `(1−ε/12)F` always survives — the oracle never
wrongly fails. Total electrical solves `≤ N + |H| = Õ(ρ/ε²) + Õ(m^{1/3}) = Õ(m^{1/3}ε^{-3})`,
each `Õ(m)`:

**Theorem.** A `(1−ε)`-approximate maximum `s-t` flow is computable in `Õ(m^{4/3} ε^{-3})` time;
with Karger's graph smoothing, in **`Õ(m n^{1/3} ε^{-11/3})`** time. The min `s-t` cut value is
`(1+ε)`-approximable in `Õ(m + n^{4/3} ε^{-8/3})` time.

## Dual cut algorithm (`Õ(m + n^{4/3} ε^{-8/3})`)

No oracle/averaging: repeatedly solve an electrical flow, raise resistances by congestion, and
read a cut from the potentials by threshold sweep.

```
MinCut(G, F, ε≤1/7):                              # ρ = 3 m^{1/3} ε^{-2/3}, N = 5 ε^{-8/3} m^{1/3} ln m, δ=ε²
  w_e ← 1
  for i = 1..N:
    f̃, φ̃ ← approx electrical flow & potentials with r_e = w_e/u_e², value F, accuracy δ
    μ ← Σ_e w_e
    w_e ← w_e + (ε/ρ) cong(f̃,e) w_e + (ε²/(mρ)) μ        # extra floor term keeps w_e ≥ (ε/m)μ
    rescale φ̃ so φ̃_s = 1, φ̃_t = 0;  S_x = {v : φ̃_v > x}
    S ← arg min_x cap(S_x, V∖S_x)
    if cap(S) < F/(1−7ε):  return S
  return FAIL
```

A random threshold cut has expected capacity `Σ_e |φ_u−φ_v| u_e`, and by Cauchy-Schwarz with
`μ = Σ_e u_e² r_e` this is `≤ √(μ/R_eff)`. The contradiction argument (total weight bounded; the
geometric mean `ν` of min-cut weights grows on low-congestion steps; `R_eff` grows on
high-congestion steps; both sums `< N`) shows `R_eff` reaches `(1−7ε)μ/F²` within `N` steps, so
the best threshold cut has capacity `≤ F/(1−7ε)`. On a Benczúr-Karger sparsifier with
`O(n log n/ε²)` edges, this gives a `(1+ε)`-cut in `Õ(m + n^{4/3} ε^{-8/3})`.

## Reference implementation

Single self-contained C++17 program. It reads `n m s t F eps` and then `m` lines `a b u`
from stdin and prints the feasibility-scaled flow value, the per-edge flow `f(e)`, and the
maximum congestion (or `FAIL` when the energy fail-test certifies `F > F*`). The Laplacian
solve is a dense Cholesky factorization of the grounded system, standing in for the
nearly-linear-time SDD solver the analysis assumes.

```cpp
// Electrical-flow + multiplicative-weights approximate maximum s-t flow.
// Reads from stdin:  n m s t F eps
//                    then m lines:  a b u   (undirected edge a--b, capacity u)
// Writes to stdout:  one line "FAIL" if F > F* (oracle certifies infeasibility),
//                    otherwise "value <V>" then m lines of per-edge flow f(e),
//                    then "maxcong <c>".  (0-indexed vertices.)
//
// The electrical flow of value F minimizes the energy sum_e r_e f(e)^2 over s-t
// flows with B f = F*chi; it is the potential flow f = C B^T phi where
// L phi = F*chi, L = B C B^T the weighted Laplacian.  L is SDD; here we ground
// one vertex and solve the dense reduced system directly (the algorithm's
// intended regime replaces this with a nearly-linear-time SDD solver).  The
// multiplicative-weights outer loop turns this capacity-oblivious oracle into a
// feasible flow, reweighting by congestion each round.
//
// long long is unused for the numeric core (flows are real), but capacities and
// counts are read as long long to avoid overflow on large inputs.

#include <bits/stdc++.h>
using namespace std;

struct Edge { int a, b; double u; };

// Solve the symmetric positive-definite reduced Laplacian system A x = rhs by
// Cholesky factorization (A is the Laplacian with the grounded vertex removed).
// A is given as a dense (k x k) row-major matrix; solves in place.
static vector<double> cholesky_solve(vector<vector<double>>& A, vector<double> rhs) {
    int k = (int)A.size();
    // Cholesky: A = L L^T (store L in lower triangle of A).
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (int p = 0; p < j; ++p) sum -= A[i][p] * A[j][p];
            if (i == j) {
                if (sum <= 0) sum = 1e-12;        // guard tiny/round-off pivots
                A[i][j] = sqrt(sum);
            } else {
                A[i][j] = sum / A[j][j];
            }
        }
    }
    // Forward solve L y = rhs.
    vector<double> y(k);
    for (int i = 0; i < k; ++i) {
        double sum = rhs[i];
        for (int p = 0; p < i; ++p) sum -= A[i][p] * y[p];
        y[i] = sum / A[i][i];
    }
    // Back solve L^T x = y.
    vector<double> x(k);
    for (int i = k - 1; i >= 0; --i) {
        double sum = y[i];
        for (int p = i + 1; p < k; ++p) sum -= A[p][i] * x[p];
        x[i] = sum / A[i][i];
    }
    return x;
}

// Electrical s-t flow of value F with the given per-edge conductances.
// Returns the flow vector f (length m); potentials phi returned via out-param.
static vector<double> electrical_flow(int n, const vector<Edge>& edges,
                                      const vector<double>& conduct,
                                      int s, int t, double F,
                                      vector<double>& phi_out) {
    int m = (int)edges.size();
    // Build dense Laplacian L = B C B^T (n x n).
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    for (int e = 0; e < m; ++e) {
        if (conduct[e] == 0.0) continue;
        int a = edges[e].a, b = edges[e].b;
        double c = conduct[e];
        L[a][a] += c; L[b][b] += c;
        L[a][b] -= c; L[b][a] -= c;
    }
    // Ground vertex 0: solve on indices 1..n-1.
    int k = n - 1;
    vector<vector<double>> A(k, vector<double>(k, 0.0));
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            A[i][j] = L[i + 1][j + 1];
    vector<double> rhs(k, 0.0);
    // chi: +1 at s, -1 at t (scaled by F); drop the grounded row 0.
    auto add_chi = [&](int v, double val) {
        if (v == 0) return;        // grounded row removed
        rhs[v - 1] += val;
    };
    add_chi(s, F);
    add_chi(t, -F);

    vector<double> xk = cholesky_solve(A, rhs);
    vector<double> phi(n, 0.0);
    for (int i = 0; i < k; ++i) phi[i + 1] = xk[i];

    // Ohm's law: f = C B^T phi, with B^T phi on edge (a,b) = phi[a]-phi[b].
    vector<double> f(m, 0.0);
    for (int e = 0; e < m; ++e) {
        if (conduct[e] == 0.0) { f[e] = 0.0; continue; }
        f[e] = conduct[e] * (phi[edges[e].a] - phi[edges[e].b]);
    }
    phi_out = phi;
    return f;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, s, t;
    long long m_ll;
    double F, eps;
    if (!(cin >> n >> m_ll >> s >> t >> F >> eps)) return 0;
    int m = (int)m_ll;

    vector<Edge> edges(m);
    vector<double> u(m);
    for (int e = 0; e < m; ++e) {
        cin >> edges[e].a >> edges[e].b >> edges[e].u;
        u[e] = edges[e].u;
    }

    // Multiplicative-weights outer loop (the plain (eps, 3 sqrt(m/eps)) oracle).
    double rho = 3.0 * sqrt((double)m / eps);             // width of the plain oracle
    long long N = (long long)ceil(2.0 * rho * log((double)max(m, 2)) / (eps * eps));
    // Cap iterations so the dense O(N * n^3) demo stays bounded on big inputs.
    const long long N_CAP = 20000;
    if (N > N_CAP) N = N_CAP;

    vector<double> w(m, 1.0);
    vector<double> acc(m, 0.0);
    bool failed = false;

    for (long long it = 0; it < N; ++it) {
        double w1 = 0.0;
        for (int e = 0; e < m; ++e) w1 += w[e];

        // r_e = (1/u_e^2)(w_e + eps*|w|_1/(3m)): w_e term for the average,
        // floor term eps*|w|_1/(3m) caps the worst congestion.
        vector<double> conduct(m, 0.0);
        vector<double> res(m, 0.0);
        double floor_term = eps * w1 / (3.0 * m);
        for (int e = 0; e < m; ++e) {
            res[e] = (w[e] + floor_term) / (u[e] * u[e]);
            conduct[e] = 1.0 / res[e];
        }

        vector<double> phi;
        vector<double> f = electrical_flow(n, edges, conduct, s, t, F, phi);

        // Energy E_r(f); fail-test certifies F > F* when energy too large.
        double E = 0.0;
        for (int e = 0; e < m; ++e) E += res[e] * f[e] * f[e];
        if (E > (1.0 + eps) * w1) { failed = true; break; }

        // Reweight by congestion; accumulate the per-round flow.
        for (int e = 0; e < m; ++e) {
            double cong = fabs(f[e]) / u[e];
            w[e] *= (1.0 + (eps / rho) * cong);
            acc[e] += f[e];
        }
    }

    cout.setf(std::ios::fixed);
    cout << setprecision(6);

    if (failed) {
        cout << "FAIL\n";
        return 0;
    }

    // Feasibility-scaled average of the per-round flows.
    double scale = (1.0 - eps) * (1.0 - eps) / ((1.0 + eps) * (double)N);
    vector<double> fbar(m);
    for (int e = 0; e < m; ++e) fbar[e] = scale * acc[e];

    // Flow value = net flow out of s.
    double value = 0.0;
    for (int e = 0; e < m; ++e) {
        if (edges[e].a == s) value += fbar[e];
        else if (edges[e].b == s) value -= fbar[e];
    }

    double maxcong = 0.0;
    for (int e = 0; e < m; ++e) maxcong = max(maxcong, fabs(fbar[e]) / u[e]);

    cout << "value " << fabs(value) << "\n";
    for (int e = 0; e < m; ++e) cout << fbar[e] << "\n";
    cout << "maxcong " << maxcong << "\n";
    return 0;
}
```

Running this on two parallel unit-capacity edges (stdin `2 2 0 1 2 0.2` then `0 1 1` twice)
prints `value 1.066667`, each edge near `0.53` (`maxcong 0.533333 < 1`, feasible); the value
gap from `F` is the conservative `(1−ε)²/(1+ε)` feasibility-scaling, which the analysis only
needs up to `1 − O(ε)`. Pushing `F` past `F*` (`2 1 0 1 10 0.2` then `0 1 1`) prints `FAIL`.
