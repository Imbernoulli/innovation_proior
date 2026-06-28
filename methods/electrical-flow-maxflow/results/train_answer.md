For thirty-five years the fastest way to compute a maximum $s$-$t$ flow in an undirected unit-capacity graph with $m = O(n)$ edges has been blocking flows, $O(n^{3/2}) = O(m^{3/2})$, and nobody has moved it. Goldberg and Rao squeezed the general capacitated case to $\tilde O(m\sqrt n)$ with a binary length function on the arcs and blocking-flow computations on the induced DAG, which is genuinely beautiful, but the $\sqrt n$ remains. And they name the reason honestly: a flow-decomposition barrier. Any algorithm that builds an explicit flow decomposition and augments path by path, one arc at a time, can be forced into $\Theta(mn)$ total work, because the total length of the paths in a worst-case decomposition is $\Theta(mn)$. So the $\sqrt n$, or the $\sqrt m$, is not an accident of one algorithm — it is baked into the *paradigm* of path augmentation, layered graphs, and blocking flows. Daitch and Spielman avoid path augmentation entirely with an interior-point method whose Newton step is a Laplacian solve, but they pay $\sqrt m$ interior-point iterations and land at $\tilde O(m^{3/2}\log U)$; the Laplacian solver speeds the inner step without reducing the outer count. The plain multiplicative-weights framework reduces flow to repeated single-constraint oracle calls, but its iteration count scales with the oracle's width, and a generic shortest-path oracle has large width. To break $m^{3/2}$ I have to stop making path augmentation faster and find a primitive that produces a whole flow vector in one shot and that I can compute fast.

The one thing that has genuinely changed is that we can now solve symmetric diagonally dominant linear systems — Laplacian systems — in nearly linear time (Spielman-Teng, then Koutis-Miller-Peng). The method I propose, then, is to build approximate maximum flow on top of the *electrical flow*: put a resistance $r_e > 0$ on each edge with conductance $c_e = 1/r_e$, form the Laplacian $L = B C B^\top$ from the incidence matrix $B$ and $C = \operatorname{diag}(c_e)$, solve $L\varphi = F\chi_{s,t}$ for vertex potentials, and read off the flow by Ohm's law $f = C B^\top\varphi$, i.e. $f(u,v) = (\varphi_v - \varphi_u)/r_{uv}$. This $f$ satisfies $Bf = F\chi_{s,t}$ — an honest $s$-$t$ flow of value $F$ — and among all such flows it uniquely minimizes the energy $E_r(f) = \sum_e r_e f(e)^2$. (Minimum norm under a linear constraint forces $f = C B^\top\varphi$; substituting into $Bf = F\chi$ gives $L\varphi = F\chi$, so $\varphi = L^+(F\chi)$ and $E_r(f) = F^2 R_\text{eff}(r)$ with $R_\text{eff}(r) = \chi^\top L^+\chi$ the effective $s$-$t$ resistance.) Because $L$ is SDD, this solve runs in $\tilde O(m\log(1/\delta))$ time.

The catch is that maximum flow, written as "push as much value $F$ as possible while keeping $\max_e |f(e)|/u_e \le 1$," is an $\ell_\infty$ problem on the congestion vector, whereas electrical flow minimizes a sum of squares — an $\ell_2$ problem. The electrical flow will happily overload one edge to lower the total energy. How badly? Take $k$ parallel paths of length $k$ from $s$ to $t$ plus one direct $s$-$t$ edge, so $m = \Theta(k^2)$. The max flow is $k+1$, but with unit resistances each path is $k$ resistors in series (resistance $k$) while the direct edge has resistance $1$, so current splits inversely to resistance and the direct edge carries $\approx k/2 \approx \sqrt m/2$ against its capacity of one. The $\ell_2$ flow overloads by $\Theta(\sqrt m)$. That is exactly the kind of factor I want to start at and then beat down.

The wrapper that converts this capacity-oblivious primitive into a feasible flow is multiplicative weights. Maintain a weight $w_e \ge 1$ per edge. An $(\varepsilon,\rho)$-oracle, given $w$ and target $F$, returns an $s$-$t$ flow of value $F$ that satisfies the capacity constraints *on weighted average*, $\sum_e w_e\,\mathrm{cong}(f,e) \le (1+\varepsilon)\|w\|_1$, and whose *worst* congestion is bounded by the width, $\max_e \mathrm{cong}(f,e) \le \rho$ (and may fail if $F > F^*$). The outer loop starts $w_e = 1$, runs $N = 2\rho\ln m/\varepsilon^2$ rounds, and after each oracle call reweights $w_e \leftarrow w_e(1 + (\varepsilon/\rho)\,\mathrm{cong}(f^i,e))$, piling weight onto edges that keep overloading so the next solve relieves them; it returns a feasibility-scaled average of the per-round flows. Tracking the potential $\mu_i = \|w^i\|_1$, the average bound gives $\mu_{i+1} \le \mu_i\exp(\varepsilon(1+\varepsilon)/\rho)$, hence $\mu_N \le m\exp(\varepsilon(1+\varepsilon)N/\rho)$; per edge, using $1 + \varepsilon x \ge \exp((1-\varepsilon)\varepsilon x)$ for $x = \mathrm{cong}/\rho \in [0,1]$, the weight grows at least exponentially in the summed congestion. Squeezing the single-edge weight against $\mu_N$ and choosing $N = 2\rho\ln m/\varepsilon^2$ shows the scaled return has $\mathrm{cong}(\bar f,e) \le 1-\varepsilon+\varepsilon(1-\varepsilon)/(2(1+\varepsilon)) \le 1$ everywhere — feasible — with value $(1-\varepsilon)^2/(1+\varepsilon)\,F \ge (1-O(\varepsilon))F$. So the cost is $\tilde O(\rho\,\varepsilon^{-2})$ oracle calls, and the width $\rho$ is the lever.

To build the oracle from one electrical solve I set
$$r_e = \frac{1}{u_e^2}\Big(w_e + \frac{\varepsilon\|w\|_1}{3m}\Big),$$
so that $E_r(f) = \sum_e\big(w_e + \varepsilon\|w\|_1/3m\big)\mathrm{cong}(f,e)^2$ — putting $1/u_e^2$ into the resistance makes each energy term a congestion-squared. The $w_e$ term ties energy to the weighted-average congestion; the additive floor $\varepsilon\|w\|_1/3m$ is essential because without it an edge whose weight had decayed toward zero would have near-zero resistance and could absorb unbounded current at no energy cost, leaving the width uncontrolled — the floor forces a large congestion on any single edge to pay a large energy contribution. The oracle computes the $(\varepsilon/3)$-approximate electrical flow of value $F$, declares fail if $E_r(\tilde f) > (1+\varepsilon)\|w\|_1$, else returns it. It never wrongly fails when $F \le F^*$: a feasible $f^*$ has $\mathrm{cong} \le 1$, so $E_r(f^*) \le (1+\varepsilon/3)\|w\|_1$, and since the electrical flow minimizes energy the approximation gives $E_r(\tilde f) \le (1+\varepsilon/3)^2\|w\|_1 \le (1+\varepsilon)\|w\|_1$. Conversely $E_r(\tilde f) \le (1+\varepsilon)\|w\|_1$ implies both bounds: dropping the floor and applying Cauchy-Schwarz, $\sum_e w_e\,\mathrm{cong} \le \sqrt{1+\varepsilon}\,\|w\|_1$; keeping only the floor, $\mathrm{cong}_e \le \sqrt{3m(1+\varepsilon)/\varepsilon} \le 3\sqrt{m/\varepsilon}$. So this is an $(\varepsilon, 3\sqrt{m/\varepsilon})$-oracle, each call one $\tilde O(m)$ Laplacian solve, and the multiplicative-weights theorem yields a $(1-\varepsilon)$-approximate flow in $\tilde O(m^{3/2}\varepsilon^{-5/2})$ — matching, not yet beating, $m^{3/2}$.

The win comes from width reduction, and it rests on the observation that the overloaded edges are *fragile*. In the bad graph, delete the single direct edge and the electrical flow on the remaining parallel paths is perfectly balanced, while the max flow only drops from $k+1$ to $k$. So I modify the oracle: pick a target width $\rho$ below $\sqrt m$, compute the electrical flow, and whenever some edge exceeds congestion $\rho$, *remove it permanently* (add it to a forbidden set $H$) and recompute, repeating until every edge is within $\rho$ or the oracle genuinely fails. Two things must hold — the removed capacity stays tiny so the oracle never wrongly fails, and the total number of removals stays small — and both follow from a single potential: the effective resistance $\Phi = R_\text{eff}$. By Thomson's principle, $C_\text{eff}(r) = 1/R_\text{eff}(r) = \min_{\varphi:\varphi_s=1,\varphi_t=0}\sum_{(u,v)}(\varphi_u-\varphi_v)^2/r_{uv}$, from which Rayleigh monotonicity is immediate (raising any resistance can only raise $R_\text{eff}$), so $\Phi$ ratchets upward through the whole run. The quantitative engine is the resistance-increase lemma: if an edge $h$ carries a $\beta$ fraction of the energy and $r_h \to \gamma r_h$, then $R_\text{eff}(r') \ge \big(\gamma/(\beta + \gamma(1-\beta))\big)R_\text{eff}(r)$, so cutting it ($\gamma = \infty$) gives $R_\text{eff}(r') \ge R_\text{eff}(r)/(1-\beta)$ and a gentle bump ($\gamma = 1+\varepsilon$) gives $R_\text{eff}(r') \ge (1+\varepsilon\beta/2)R_\text{eff}(r)$; the proof plugs the unnormalized potentials of $r$ as a feasible point into the min for $r'$. A removed edge has $\mathrm{cong} > \rho$, so through the floor it carries more than an $\varepsilon\rho^2/(5m)$ fraction of the energy (transferred from the approximate to the exact flow by the solver's per-edge guarantee), and each removal multiplies $\Phi$ by at least $1/(1-\varepsilon\rho^2/5m)$. Bounding $\Phi$ below at the first solve by $m^{-4}F^{-2}$ and above before the last removal by $(1+\varepsilon)\|w\|_1/F^2$, the ratio caps the removals at $|H| \le 30\,m\ln m/(\varepsilon^2\rho^2)$, and since a removed edge carries more than $\rho u_e$ but never more than $F$ units, $u_e < F/\rho$ and $u(H) \le 30\,mF\ln m/(\varepsilon^2\rho^3)$. The iterations $\tilde O(\rho/\varepsilon^2)$ and the removals $\tilde O(m/(\varepsilon^2\rho^2))$ balance at $\rho^3 \approx m$; setting $\rho = 8m^{1/3}\ln^{1/3}m/\varepsilon$ gives $|H| = \tilde O(m^{1/3})$ and $u(H) < \varepsilon F/12$, so a feasible flow of value $(1-\varepsilon/12)F$ always survives and the oracle remains legitimate. Total solves $N + |H| = \tilde O(m^{1/3}\varepsilon^{-3})$, each $\tilde O(m)$, for a running time of $\tilde O(m^{4/3}\varepsilon^{-3})$ — past the $m^{3/2}$ barrier. Feeding this into Karger's graph smoothing gives $\tilde O(mn^{1/3}\varepsilon^{-11/3})$, and running on a Benczúr-Karger cut sparsifier gives a $(1+\varepsilon)$-approximate cut value in $\tilde O(m + n^{4/3}\varepsilon^{-3})$.

The cut admits a cleaner dual that needs no oracle abstraction, no averaging, and no forbidden set: repeatedly solve an electrical flow, raise resistances by congestion, and read a cut directly from the potentials by a threshold sweep. Scaling $\varphi_s = 1, \varphi_t = 0$ and cutting at a uniform random threshold $x$, an edge $(u,v)$ is cut with probability $|\varphi_u - \varphi_v|$, so some threshold achieves capacity at most $\sum_e |\varphi_u-\varphi_v|u_e$, which by Cauchy-Schwarz with $\mu = \sum_e u_e^2 r_e$ is $\le \sqrt{\mu/R_\text{eff}}$. Driving $R_\text{eff}$ up to $\approx\mu/F^2$ — its ceiling, attained when the resistance has concentrated onto the minimum cut — makes this $\approx F$. The update $w_e \leftarrow w_e + (\varepsilon/\rho)\mathrm{cong}(\tilde f,e)w_e + (\varepsilon^2/(m\rho))\mu$ carries a new additive floor term keeping every $w_e \ge (\varepsilon/m)\mu$, which the jump lemma needs so a reweighted edge carries meaningful absolute weight. A contradiction argument — total weight bounded, the capacity-weighted geometric mean of min-cut edge weights growing on low-congestion steps, $R_\text{eff}$ growing on high-congestion steps, both step-counts summing to less than $N$ — shows $R_\text{eff}$ reaches $(1-7\varepsilon)\mu/F^2$ within $N = 5\varepsilon^{-8/3}m^{1/3}\ln m$ steps with $\rho = 3m^{1/3}\varepsilon^{-2/3}$, yielding a cut of capacity at most $F/(1-7\varepsilon)$ in $\tilde O(m^{4/3}\varepsilon^{-8/3})$, and $\tilde O(m + n^{4/3}\varepsilon^{-8/3})$ on a sparsifier.

Here is the algorithm — the electrical-flow primitive, the oracle, and the multiplicative-weights loop — as a single self-contained C++17 program. It reads `n m s t F eps` and then `m` lines `a b u` from stdin and prints the feasibility-scaled flow value, the per-edge flow, and the maximum congestion (or `FAIL` when the energy fail-test certifies `F > F*`). The Laplacian solve is a dense Cholesky factorization of the grounded system, standing in for the nearly-linear-time SDD solver the analysis assumes:

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

static double output_number(double x) {
    double y = round(x * 1000000.0) / 1000000.0;
    if (fabs(y) < 0.5e-6) return 0.0;
    return y;
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
    vector<double> printed_flow(m);
    for (int e = 0; e < m; ++e) printed_flow[e] = output_number(fbar[e]);

    // Flow value = net flow out of s in the printed certificate.
    double value = 0.0;
    for (int e = 0; e < m; ++e) {
        if (edges[e].a == s) value += printed_flow[e];
        else if (edges[e].b == s) value -= printed_flow[e];
    }

    double maxcong = 0.0;
    for (int e = 0; e < m; ++e) maxcong = max(maxcong, fabs(printed_flow[e]) / u[e]);

    cout << "value " << output_number(fabs(value)) << "\n";
    for (int e = 0; e < m; ++e) cout << printed_flow[e] << "\n";
    cout << "maxcong " << output_number(maxcong) << "\n";
    return 0;
}
```
