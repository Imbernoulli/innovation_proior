// TIER: strong
// Electrical local search. Inject a unit s->t current and solve the network for bus
// potentials (dense Cholesky on the reduced Laplacian, grounding t). Rank lines by the power
// they dissipate, g_e*(V_u - V_v)^2 -- the lines actually carrying the s->t current. Cut the
// highest-power, affordable lines that keep s-t connected, then re-solve so the attack adapts
// as current re-routes onto longer detours. This targets the electrical objective directly and
// beats the cost-first greedy, with different per-test behaviour.
#include <bits/stdc++.h>
using namespace std;

int n, m, s, t;
long long B;
vector<int> eu, ev, er, ec;

bool connected(const vector<char>& alive) {
    vector<vector<int>> g(n + 1);
    for (int e = 0; e < m; e++) if (alive[e]) { g[eu[e]].push_back(ev[e]); g[ev[e]].push_back(eu[e]); }
    vector<char> seen(n + 1, 0); vector<int> st = {s}; seen[s] = 1;
    while (!st.empty()) { int u = st.back(); st.pop_back(); for (int w : g[u]) if (!seen[w]) { seen[w] = 1; st.push_back(w); } }
    return seen[t];
}

// Solve for potentials on the alive subgraph (component of s, t grounded). Fills V[1..n]
// (0 outside comp / at t). Returns false if solve degenerate.
bool potentials(const vector<char>& alive, vector<double>& V) {
    vector<vector<int>> g(n + 1);
    for (int e = 0; e < m; e++) if (alive[e]) { g[eu[e]].push_back(ev[e]); g[ev[e]].push_back(eu[e]); }
    vector<int> loc(n + 1, -1), comp; comp.push_back(s); loc[s] = 0;
    for (size_t h = 0; h < comp.size(); h++) { int u = comp[h]; for (int w : g[u]) if (loc[w] < 0) { loc[w] = (int)comp.size(); comp.push_back(w); } }
    int p = (int)comp.size();
    if (loc[t] < 0) return false;
    // dense Laplacian on comp
    vector<double> L((size_t)p * p, 0.0);
    auto AT = [&](int i, int j) -> double& { return L[(size_t)i * p + j]; };
    for (int e = 0; e < m; e++) if (alive[e]) {
        int a = loc[eu[e]], b = loc[ev[e]]; if (a < 0 || b < 0) continue;
        double gc = 1.0 / (double)er[e];
        AT(a, b) -= gc; AT(b, a) -= gc; AT(a, a) += gc; AT(b, b) += gc;
    }
    // reduced system on all comp nodes except t
    int ti = loc[t];
    vector<int> keep; for (int i = 0; i < p; i++) if (i != ti) keep.push_back(i);
    int q = (int)keep.size();
    if (q == 0) { V.assign(n + 1, 0.0); return true; }
    vector<double> A((size_t)q * q), b(q, 0.0);
    for (int i = 0; i < q; i++) for (int j = 0; j < q; j++) A[(size_t)i * q + j] = AT(keep[i], keep[j]);
    b[/*local index of s among keep*/ (loc[s] < ti ? loc[s] : loc[s] - 1)] = 1.0; // s != t so present
    // Cholesky factor A = R R^T (lower), solve A x = b
    for (int i = 0; i < q; i++) {
        double d = A[(size_t)i * q + i];
        for (int k = 0; k < i; k++) { double v = A[(size_t)i * q + k]; d -= v * v; }
        if (d <= 1e-300) d = 1e-300;
        double di = sqrt(d); A[(size_t)i * q + i] = di;
        for (int j = i + 1; j < q; j++) {
            double sum = A[(size_t)j * q + i];
            for (int k = 0; k < i; k++) sum -= A[(size_t)j * q + k] * A[(size_t)i * q + k];
            A[(size_t)j * q + i] = sum / di;
        }
    }
    // forward solve L y = b
    vector<double> y(q);
    for (int i = 0; i < q; i++) {
        double sum = b[i];
        for (int k = 0; k < i; k++) sum -= A[(size_t)i * q + k] * y[k];
        y[i] = sum / A[(size_t)i * q + i];
    }
    // back solve L^T x = y
    vector<double> x(q);
    for (int i = q - 1; i >= 0; i--) {
        double sum = y[i];
        for (int k = i + 1; k < q; k++) sum -= A[(size_t)k * q + i] * x[k];
        x[i] = sum / A[(size_t)i * q + i];
    }
    V.assign(n + 1, 0.0);
    for (int i = 0; i < q; i++) V[comp[keep[i]]] = x[i];
    // t and non-comp stay 0
    return true;
}

int main() {
    scanf("%d %d %d %d %lld", &n, &m, &s, &t, &B);
    eu.resize(m); ev.resize(m); er.resize(m); ec.resize(m);
    for (int e = 0; e < m; e++) scanf("%d %d %d %d", &eu[e], &ev[e], &er[e], &ec[e]);

    vector<char> alive(m, 1), cut(m, 0);
    long long spent = 0;
    int rounds = 6;
    for (int rd = 0; rd < rounds; rd++) {
        if (spent >= B) break;
        vector<double> V;
        if (!potentials(alive, V)) break;
        // rank alive, uncut edges by dissipated power
        vector<pair<double, int>> imp;
        imp.reserve(m);
        for (int e = 0; e < m; e++) if (alive[e]) {
            double gc = 1.0 / (double)er[e];
            double dv = V[eu[e]] - V[ev[e]];
            imp.push_back({gc * dv * dv, e});
        }
        sort(imp.begin(), imp.end(), [](const pair<double,int>& a, const pair<double,int>& b) { return a.first > b.first; });
        long long roundCap = spent + (B - spent) / (rounds - rd); // spend a slice this round
        bool progressed = false;
        for (auto& pr : imp) {
            int e = pr.second;
            if (pr.first <= 0.0) break;                 // no current beyond here
            if (spent >= roundCap) break;
            if (spent + ec[e] > B) continue;
            alive[e] = 0;
            if (connected(alive)) { spent += ec[e]; cut[e] = 1; progressed = true; }
            else alive[e] = 1;
        }
        if (!progressed) break;
    }
    // final pass: spend any leftover budget on remaining current-carrying lines
    {
        vector<double> V;
        if (potentials(alive, V)) {
            vector<pair<double,int>> imp;
            for (int e = 0; e < m; e++) if (alive[e]) {
                double gc = 1.0 / (double)er[e]; double dv = V[eu[e]] - V[ev[e]];
                imp.push_back({gc * dv * dv, e});
            }
            sort(imp.begin(), imp.end(), [](const pair<double,int>& a, const pair<double,int>& b) { return a.first > b.first; });
            for (auto& pr : imp) {
                int e = pr.second;
                if (spent + ec[e] > B) continue;
                alive[e] = 0;
                if (connected(alive)) { spent += ec[e]; cut[e] = 1; }
                else alive[e] = 1;
            }
        }
    }

    vector<int> out;
    for (int e = 0; e < m; e++) if (cut[e]) out.push_back(e + 1);
    printf("%d\n", (int)out.size());
    for (size_t i = 0; i < out.size(); i++) printf("%d%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    if (out.empty()) printf("\n");
    return 0;
}
