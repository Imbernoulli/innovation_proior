// TIER: strong
// The Lam-Leung insight, done properly: the achievable re-voicings of a chord
// are NOT just "subtract full cycles you can already see" -- they are every
// integer (signed!) combination of rotated p-cycles and q-cycles. That means
// a smaller chord can require *borrowing*: temporarily adding a q-cycle you
// don't fully see yet so that several p-cycles become removable elsewhere,
// netting a bigger reduction than any pure-subtraction pass could find (e.g.
// n=6, only bells 3 and 5 struck once each: no full cycle is even present,
// yet the whole chord collapses to a single strike on bell 4).
//
// Because n has exactly two distinct prime factors p, q, every index i lies
// in exactly one p-cycle-orbit j(i) and one q-cycle-orbit k(i), and the
// achievable strike counts are b_i = a_i - x_{j(i)} - y_{k(i)} for INTEGER
// (any sign) potentials x_j, y_k, minimizing sum(b) subject to b >= 0 --
// i.e. maximizing sum(x)*p + sum(y)*q subject to x_j + y_k <= a_i for every
// index i. This is exactly the LP dual of a balanced transportation problem
// (n/p sources each supplying p units, n/q sinks each demanding q units,
// edge (j(i),k(i)) costing a_i) -- so we solve it with a textbook min-cost
// flow, then read the potentials off the tight (positive-flow) edges. The
// result is verified against the SAME exact cyclotomic-integer arithmetic
// the checker uses before being trusted; on any doubt we fall back to the
// safe alternating-exhaustion cycle-subtraction heuristic, which is always
// feasible.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ---- exact cyclotomic reduction (for internal self-verification only) ----
static vector<ll> polyDiv(vector<ll> a, const vector<ll> &b) {
    int da = (int)a.size() - 1, db = (int)b.size() - 1;
    vector<ll> q(da - db + 1, 0);
    for (int i = da; i >= db; i--) {
        ll coef = a[i];
        q[i - db] = coef;
        if (coef != 0) for (int j = 0; j <= db; j++) a[i - db + j] -= coef * b[j];
    }
    return q;
}
static map<int, vector<ll>> philib;
static vector<ll> getPhi(int n) {
    auto it = philib.find(n);
    if (it != philib.end()) return it->second;
    vector<ll> poly(n + 1, 0);
    poly[0] = -1; poly[n] = 1;
    for (int d = 1; d < n; d++) if (n % d == 0) poly = polyDiv(poly, getPhi(d));
    philib[n] = poly;
    return poly;
}
static vector<ll> reduceModPhi(vector<ll> c, int n) {
    vector<ll> poly = getPhi(n);
    int m = (int)poly.size() - 1;
    c.resize(n, 0);
    for (int i = n - 1; i >= m; i--) {
        if (c[i] != 0) {
            ll coef = c[i];
            for (int j = 0; j < m; j++) c[i - m + j] -= coef * poly[j];
            c[i] = 0;
        }
    }
    c.resize(m);
    return c;
}

static vector<int> primeFactors(int n) {
    vector<int> ps; int m = n;
    for (int p = 2; (ll)p * p <= m; p++) if (m % p == 0) { ps.push_back(p); while (m % p == 0) m /= p; }
    if (m > 1) ps.push_back(m);
    return ps;
}

// ---- fallback: exhaust one prime's cycles fully & independently, alternate ----
static void exhaustPrime(vector<ll> &a, int n, int p) {
    int step = n / p;
    for (int j = 0; j < step; j++) {
        ll mn = LLONG_MAX;
        for (int k = 0; k < p; k++) mn = min(mn, a[j + (ll)k * step]);
        if (mn > 0) for (int k = 0; k < p; k++) a[j + (ll)k * step] -= mn;
    }
}
static vector<ll> alternate(vector<ll> a, int n, int first, int second) {
    ll prevSum = -1;
    while (true) {
        ll before = accumulate(a.begin(), a.end(), 0LL);
        if (before == prevSum) break;
        prevSum = before;
        exhaustPrime(a, n, first);
        exhaustPrime(a, n, second);
    }
    return a;
}

// ---- min cost flow (SPFA-based successive shortest augmenting paths) ----
struct MCMF {
    struct E { int to; ll cap, cost; };
    vector<E> edges;
    vector<vector<int>> g;
    int N;
    MCMF(int n) : N(n), g(n) {}
    void add(int u, int v, ll cap, ll cost) {
        g[u].push_back((int)edges.size()); edges.push_back({v, cap, cost});
        g[v].push_back((int)edges.size()); edges.push_back({u, 0, -cost});
    }
    // returns total flow actually pushed, stops once `want` units are pushed
    ll run(int s, int t, ll want) {
        ll pushed = 0;
        while (pushed < want) {
            vector<ll> dist(N, LLONG_MAX / 2);
            vector<int> pe(N, -1);
            vector<char> inq(N, 0);
            deque<int> q;
            dist[s] = 0; q.push_back(s); inq[s] = 1;
            while (!q.empty()) {
                int u = q.front(); q.pop_front(); inq[u] = 0;
                for (int id : g[u]) {
                    auto &e = edges[id];
                    if (e.cap > 0 && dist[u] + e.cost < dist[e.to]) {
                        dist[e.to] = dist[u] + e.cost;
                        pe[e.to] = id;
                        if (!inq[e.to]) { inq[e.to] = 1; q.push_back(e.to); }
                    }
                }
            }
            if (pe[t] == -1) break; // no augmenting path (shouldn't happen before `want`)
            ll aug = want - pushed;
            for (int v = t; v != s; ) {
                int id = pe[v];
                aug = min(aug, edges[id].cap);
                v = edges[id ^ 1].to;
            }
            for (int v = t; v != s; ) {
                int id = pe[v];
                edges[id].cap -= aug;
                edges[id ^ 1].cap += aug;
                v = edges[id ^ 1].to;
            }
            pushed += aug;
        }
        return pushed;
    }
};

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<ll> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    vector<int> primes = primeFactors(n); // exactly two, by problem constraints
    int P = primes[0], Q = primes.size() > 1 ? primes[1] : primes[0];
    int stepP = n / P, stepQ = n / Q;
    int numJ = stepP, numK = stepQ; // number of p-orbits, q-orbits

    vector<ll> best = alternate(a, n, P, Q);
    vector<ll> alt = alternate(a, n, Q, P);
    if (accumulate(alt.begin(), alt.end(), 0LL) < accumulate(best.begin(), best.end(), 0LL))
        best = alt;

    // ---- try the full lattice (signed combinations) via min-cost flow ----
    int S = 0, T = numJ + numK + 1;
    // node 1..numJ = p-orbits, numJ+1..numJ+numK = q-orbits
    MCMF mc(numJ + numK + 2);
    vector<int> edgeOfIndex(n, -1); // id of the forward (p_j -> q_k) edge for index i
    ll BIG = (ll)P + Q + 5;
    for (int j = 0; j < numJ; j++) mc.add(S, 1 + j, P, 0);
    for (int k = 0; k < numK; k++) mc.add(1 + numJ + k, T, Q, 0);
    for (int i = 0; i < n; i++) {
        int j = i % stepP;      // p-orbit of i
        int k = i % stepQ;      // q-orbit of i
        edgeOfIndex[i] = (int)mc.edges.size();
        mc.add(1 + j, 1 + numJ + k, BIG, a[i]);
    }
    ll pushed = mc.run(S, T, (ll)n);

    if (pushed == (ll)n) {
        // recover flow on each index edge: original cap BIG minus remaining cap
        vector<ll> flowOfIndex(n, 0);
        for (int i = 0; i < n; i++) flowOfIndex[i] = BIG - mc.edges[edgeOfIndex[i]].cap;

        // build positive-flow graph over the numJ+numK orbit nodes, assign
        // potentials per connected component via BFS (gauge-fixed at 0).
        vector<vector<pair<int, int>>> padj(numJ + numK); // node -> (other node, index i)
        for (int i = 0; i < n; i++) if (flowOfIndex[i] > 0) {
            int j = i % stepP, k = numJ + (i % stepQ);
            padj[j].push_back({k, i});
            padj[k].push_back({j, i});
        }
        vector<ll> pot(numJ + numK, 0);
        vector<char> seen(numJ + numK, 0);
        for (int s0 = 0; s0 < numJ + numK; s0++) if (!seen[s0]) {
            seen[s0] = 1; pot[s0] = 0;
            deque<int> bfs; bfs.push_back(s0);
            while (!bfs.empty()) {
                int u = bfs.front(); bfs.pop_front();
                for (auto &pr : padj[u]) {
                    int v = pr.first, idx = pr.second;
                    if (seen[v]) continue;
                    seen[v] = 1;
                    pot[v] = a[idx] - pot[u]; // tight edge: pot[u]+pot[v] = a_idx
                    bfs.push_back(v);
                }
            }
        }
        vector<ll> flowB(n);
        bool ok = true;
        for (int i = 0; i < n; i++) {
            int j = i % stepP, k = numJ + (i % stepQ);
            ll bi = a[i] - pot[j] - pot[k];
            if (bi < 0) { ok = false; break; }
            flowB[i] = bi;
        }
        if (ok) {
            vector<ll> ra = reduceModPhi(a, n);
            vector<ll> rb = reduceModPhi(flowB, n);
            if (ra == rb && accumulate(flowB.begin(), flowB.end(), 0LL) <
                             accumulate(best.begin(), best.end(), 0LL)) {
                best = flowB;
            }
        }
    }

    ll t = accumulate(best.begin(), best.end(), 0LL);
    cout << t << "\n";
    for (int i = 0; i < n; i++) for (ll k = 0; k < best[i]; k++) cout << i << ' ';
    cout << "\n";
    return 0;
}
