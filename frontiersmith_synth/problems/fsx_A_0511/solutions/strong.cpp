// TIER: strong
// INSIGHT: reformulate "cover the most allowed k-mers with one forbidden-free tape"
// as a MAX-EDGE walk on the mutilated de Bruijn graph.  A single greedy trail
// stalls at the first sink; the win is to take the largest STRONGLY-CONNECTED
// component (all of whose edges are jointly coverable), EULERIZE it by repeating a
// few allowed edges -- repeats cost nothing because coverage counts DISTINCT
// factors -- and sweep every one of its edges with one Eulerian circuit.  On dense
// mutilated de Bruijn graphs this giant-SCC circuit dominates; on very sparse ones a
// long DAG-path can win, so we also keep a dead-end-avoiding trail and output
// whichever tape covers more distinct allowed factors.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int A, K, V;
static ll E;
static vector<char> forb;
static inline int tgt(int v, int s) { return (int)(((ll)v * A + s) % V); }

// distinct allowed length-K factors of S (0 if any forbidden factor appears)
static ll coverage(const string& S) {
    if ((ll)S.size() < K) return -1;
    vector<ll> pw(K, 1);
    for (int i = 1; i < K; i++) pw[i] = pw[i - 1] * A;
    ll high = pw[K - 1];
    vector<char> seen(E, 0);
    ll code = 0;
    for (int i = 0; i < K; i++) code = code * A + (S[i] - '0');
    ll cov = 0;
    if (forb[code]) return -1;
    seen[code] = 1; cov = 1;
    for (size_t i = K; i < S.size(); i++) {
        code = (code - (ll)(S[i - K] - '0') * high) * A + (S[i] - '0');
        if (forb[code]) return -1;
        if (!seen[code]) { seen[code] = 1; cov++; }
    }
    return cov;
}

static string spell(int v) {
    string s; vector<int> d(K - 1, 0); int x = v;
    for (int i = K - 2; i >= 0; i--) { d[i] = x % A; x /= A; }
    for (int i = 0; i < K - 1; i++) s.push_back((char)('0' + d[i]));
    return s;
}

// ---------- candidate 1: largest-SCC Eulerian circuit ----------
static string sccCircuit() {
    vector<vector<int>> out(V), radj(V);
    for (int v = 0; v < V; v++)
        for (int s = 0; s < A; s++)
            if (!forb[(ll)v * A + s]) { int w = tgt(v, s); out[v].push_back(s); radj[w].push_back(v); }

    // Kosaraju
    vector<int> order; order.reserve(V);
    vector<char> vis(V, 0); vector<int> st, pi(V, 0);
    for (int s0 = 0; s0 < V; s0++) {
        if (vis[s0]) continue;
        st.push_back(s0); vis[s0] = 1; pi[s0] = 0;
        while (!st.empty()) {
            int v = st.back();
            if (pi[v] < (int)out[v].size()) {
                int w = tgt(v, out[v][pi[v]++]);
                if (!vis[w]) { vis[w] = 1; pi[w] = 0; st.push_back(w); }
            } else { order.push_back(v); st.pop_back(); }
        }
    }
    vector<int> comp(V, -1); int nc = 0;
    for (int i = (int)order.size() - 1; i >= 0; i--) {
        int r = order[i]; if (comp[r] != -1) continue;
        st.clear(); st.push_back(r); comp[r] = nc;
        while (!st.empty()) { int v = st.back(); st.pop_back();
            for (int u : radj[v]) if (comp[u] == -1) { comp[u] = nc; st.push_back(u); } }
        nc++;
    }
    vector<ll> ecnt(nc, 0);
    for (int v = 0; v < V; v++)
        for (int s : out[v]) if (comp[tgt(v, s)] == comp[v]) ecnt[comp[v]]++;
    int S = 0; for (int c = 1; c < nc; c++) if (ecnt[c] > ecnt[S]) S = c;
    if (ecnt[S] <= 0) return "";

    vector<vector<int>> adj(V);
    vector<int> outdeg(V, 0), indeg(V, 0);
    for (int v = 0; v < V; v++) if (comp[v] == S)
        for (int s : out[v]) if (comp[tgt(v, s)] == S) {
            adj[v].push_back(s); outdeg[v]++; indeg[tgt(v, s)]++;
        }
    // Eulerize: pair excess-in (needs outgoing) with excess-out (needs incoming)
    vector<int> needIn, needOut;
    for (int v = 0; v < V; v++) if (comp[v] == S) {
        int d = outdeg[v] - indeg[v];
        if (d > 0) for (int i = 0; i < d; i++) needIn.push_back(v);
        else if (d < 0) for (int i = 0; i < -d; i++) needOut.push_back(v);
    }
    vector<int> par(V), parsym(V), q; q.reserve(V);
    for (size_t idx = 0; idx < needOut.size() && idx < needIn.size(); idx++) {
        int a = needOut[idx], b = needIn[idx]; if (a == b) continue;
        fill(par.begin(), par.end(), -1);
        q.clear(); q.push_back(a); par[a] = a; size_t head = 0; bool found = false;
        while (head < q.size() && !found) {
            int v = q[head++];
            for (int s : adj[v]) { int w = tgt(v, s);
                if (par[w] == -1) { par[w] = v; parsym[w] = s; q.push_back(w); if (w == b) { found = true; break; } } }
        }
        if (par[b] == -1) continue;
        int cur = b; while (cur != a) { int pv = par[cur]; adj[pv].push_back(parsym[cur]); cur = pv; }
    }
    int start = -1;
    for (int v = 0; v < V; v++) if (comp[v] == S && !adj[v].empty()) { start = v; break; }
    if (start < 0) return "";
    vector<int> ptr(V, 0), sv, ss, rev;
    sv.push_back(start); ss.push_back(-1);
    while (!sv.empty()) {
        int v = sv.back();
        if (ptr[v] < (int)adj[v].size()) { int s = adj[v][ptr[v]++]; sv.push_back(tgt(v, s)); ss.push_back(s); }
        else { int s = ss.back(); sv.pop_back(); ss.pop_back(); if (s != -1) rev.push_back(s); }
    }
    reverse(rev.begin(), rev.end());
    string s = spell(start);
    for (int x : rev) s.push_back((char)('0' + x));
    return s;
}

// ---------- candidate 2: dead-end-avoiding best-of-starts trail ----------
static string greedyTrail() {
    vector<int> clean;
    for (int v = 0; v < V; v++)
        for (int s = 0; s < A; s++) if (!forb[(ll)v * A + s]) { clean.push_back(v); break; }
    if (clean.empty()) return "";
    vector<int> starts; int want = 64;
    if ((int)clean.size() <= want) starts = clean;
    else { int stride = (int)clean.size() / want;
        for (size_t i = 0; i < clean.size() && (int)starts.size() < want; i += stride) starts.push_back(clean[i]); }
    string best; ll bestCov = -1; vector<char> used(E, 0);
    for (int stt : starts) {
        fill(used.begin(), used.end(), 0);
        string s = spell(stt); int cur = stt; ll cov = 0;
        while (true) {
            int bestS = -1, bestScore = -1;
            for (int a = 0; a < A; a++) { ll ec = (ll)cur * A + a;
                if (forb[ec] || used[ec]) continue; int nx = (int)(ec % V); int sc = 0;
                for (int b = 0; b < A; b++) { ll e2 = (ll)nx * A + b; if (!forb[e2] && !used[e2]) sc++; }
                if (sc > bestScore) { bestScore = sc; bestS = a; } }
            if (bestS < 0) break;
            ll ec = (ll)cur * A + bestS; used[ec] = 1; cov++; s.push_back((char)('0' + bestS)); cur = (int)(ec % V);
        }
        if (cov > bestCov) { bestCov = cov; best.swap(s); }
    }
    return best;
}

int main() {
    int F;
    scanf("%d %d %d", &A, &K, &F);
    V = 1; for (int i = 0; i < K - 1; i++) V *= A;
    E = (ll)V * A;
    forb.assign(E, 0);
    static char buf[64];
    for (int t = 0; t < F; t++) { scanf("%s", buf); ll code = 0;
        for (int i = 0; i < K; i++) code = code * A + (buf[i] - '0'); forb[code] = 1; }

    string c1 = sccCircuit();
    string c2 = greedyTrail();
    ll v1 = c1.empty() ? -1 : coverage(c1);
    ll v2 = c2.empty() ? -1 : coverage(c2);
    const string& best = (v1 >= v2) ? c1 : c2;
    if (best.empty()) { printf("0\n"); return 0; }
    printf("%s\n", best.c_str());
    return 0;
}
