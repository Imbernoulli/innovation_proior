// TIER: strong
#include <bits/stdc++.h>
using namespace std;
// Insight: interdependent cascades are worsened by degree-correlated coupling.
// P is built redundant (a cycle backbone survives losing a hub); C is built
// fragile (star clusters collapse when their relay hub dies). So a power hub's
// failure must be routed onto an UNIMPORTANT comm node, never onto a comm hub.
// Base construction: sort P descending / C ascending by degree and pair by rank
// (anti-correlated coupling) -- this alone raises the mutual-percolation
// threshold. On top of that we run a bounded local search restricted to the
// highest-degree P-nodes (the only ones whose partner choice can matter under
// the given attack scenarios): repeatedly try swapping the C-partners of two
// such nodes and keep the swap only if it improves the TOTAL survivor count
// summed over every attack scenario, re-simulating the real cascade each time
// (an exchange-argument refinement, not just "greedy plus more iterations").

static int N;
static vector<vector<int>> adjP, adjC;

static long long runCascade(const vector<int>& attacked, const vector<int>& matchPtoC,
                             const vector<int>& matchCtoP) {
    vector<char> aliveP(N, 1), aliveC(N, 1);
    for (int v : attacked) aliveP[v] = 0;
    vector<int> comp(N);
    int cap = N + 5;
    for (int round = 0; round < cap; round++) {
        bool changed = false;
        fill(comp.begin(), comp.end(), -1);
        vector<int> sizeOf;
        for (int s = 0; s < N; s++) {
            if (!aliveP[s] || comp[s] != -1) continue;
            int cid = (int)sizeOf.size(); int sz = 0;
            vector<int> st{s}; comp[s] = cid;
            while (!st.empty()) { int u = st.back(); st.pop_back(); sz++;
                for (int w : adjP[u]) if (aliveP[w] && comp[w] == -1) { comp[w] = cid; st.push_back(w); } }
            sizeOf.push_back(sz);
        }
        int bestC = -1, bestSz = -1;
        for (int i = 0; i < (int)sizeOf.size(); i++) if (sizeOf[i] > bestSz) { bestSz = sizeOf[i]; bestC = i; }
        for (int v = 0; v < N; v++) if (aliveP[v] && comp[v] != bestC) { aliveP[v] = 0; changed = true; }

        for (int i = 0; i < N; i++)
            if (!aliveP[i] && aliveC[matchPtoC[i]]) { aliveC[matchPtoC[i]] = 0; changed = true; }

        fill(comp.begin(), comp.end(), -1);
        sizeOf.clear();
        for (int s = 0; s < N; s++) {
            if (!aliveC[s] || comp[s] != -1) continue;
            int cid = (int)sizeOf.size(); int sz = 0;
            vector<int> st{s}; comp[s] = cid;
            while (!st.empty()) { int u = st.back(); st.pop_back(); sz++;
                for (int w : adjC[u]) if (aliveC[w] && comp[w] == -1) { comp[w] = cid; st.push_back(w); } }
            sizeOf.push_back(sz);
        }
        bestC = -1; bestSz = -1;
        for (int i = 0; i < (int)sizeOf.size(); i++) if (sizeOf[i] > bestSz) { bestSz = sizeOf[i]; bestC = i; }
        for (int v = 0; v < N; v++) if (aliveC[v] && comp[v] != bestC) { aliveC[v] = 0; changed = true; }

        for (int j = 0; j < N; j++)
            if (!aliveC[j] && aliveP[matchCtoP[j]]) { aliveP[matchCtoP[j]] = 0; changed = true; }

        if (!changed) break;
    }
    long long surv = 0;
    for (int i = 0; i < N; i++) if (aliveP[i]) surv++;
    return surv;
}

static long long totalSurvivors(const vector<vector<int>>& attacks, const vector<int>& matchPtoC) {
    vector<int> matchCtoP(N);
    for (int i = 0; i < N; i++) matchCtoP[matchPtoC[i]] = i;
    long long tot = 0;
    for (auto& a : attacks) tot += runCascade(a, matchPtoC, matchCtoP);
    return tot;
}

int main() {
    int Mp, Mc;
    if (!(cin >> N >> Mp >> Mc)) return 0;
    adjP.assign(N, {}); adjC.assign(N, {});
    vector<int> degP(N, 0), degC(N, 0);
    for (int e = 0; e < Mp; e++) {
        int a, b; cin >> a >> b; a--; b--;
        adjP[a].push_back(b); adjP[b].push_back(a);
        degP[a]++; degP[b]++;
    }
    for (int e = 0; e < Mc; e++) {
        int a, b; cin >> a >> b; a--; b--;
        adjC[a].push_back(b); adjC[b].push_back(a);
        degC[a]++; degC[b]++;
    }
    int K; cin >> K;
    vector<vector<int>> attacks(K);
    for (int t = 0; t < K; t++) {
        int A; cin >> A;
        attacks[t].resize(A);
        for (int i = 0; i < A; i++) { cin >> attacks[t][i]; attacks[t][i]--; }
    }

    vector<int> pOrder(N), cDesc(N);
    for (int i = 0; i < N; i++) pOrder[i] = i, cDesc[i] = i;
    sort(pOrder.begin(), pOrder.end(), [&](int a, int b) {
        if (degP[a] != degP[b]) return degP[a] > degP[b]; return a < b; });
    sort(cDesc.begin(), cDesc.end(), [&](int a, int b) {
        if (degC[a] != degC[b]) return degC[a] > degC[b]; return a < b; });

    // Read the attack scenarios themselves to tell whether ANY of them looks
    // like a deliberate strike on the high-degree power sites, as opposed to
    // broad/indiscriminate damage. Compare, for each scenario, the fraction of
    // the top-quintile-by-degree power sites it removes against what a
    // same-size UNIFORM random draw would remove in expectation; a ratio well
    // above 1 means that scenario is deliberately hub-seeking.
    int topQ = max(1, N / 5);
    vector<char> inTopQ(N, 0);
    for (int i = 0; i < topQ; i++) inTopQ[pOrder[i]] = 1;
    double worstRisk = 0.0;
    for (auto& a : attacks) {
        if (a.empty()) continue;
        int hit = 0;
        for (int v : a) if (inTopQ[v]) hit++;
        double expected = (double)a.size() * topQ / (double)N;
        double risk = hit / max(1e-9, expected);
        worstRisk = max(worstRisk, risk);
    }
    bool targetedRegime = worstRisk > 2.5;

    // Initial construction: a THREE-BAND coupling, not a flat full reversal.
    //   - P's top band (the likely targeted power hubs) -> C's bottom band
    //     (the cheapest, least important comm leaves): a targeted strike on a
    //     power hub then only ever costs an unimportant comm node.
    //   - C's top band (the fragile comm relay hubs) -> P's SECOND band (well-
    //     connected but not top-band power sites): protects a comm hub against
    //     broad random/storm damage without parking it on the exact sites a
    //     deliberate strike would hit.
    //   - everything else: leftover bands paired up (order doesn't matter much
    //     here; local search below can still touch it).
    vector<int> matchPtoC(N, -1);
    if (!targetedRegime) {
        // No scenario deliberately seeks out the power hubs: pairing degree
        // with degree (same-direction correlation) is actually the RIGHT
        // call here -- a comm hub parked on the most storm-resistant power
        // site there is, with nothing around to exploit that choice.
        for (int r = 0; r < N; r++) matchPtoC[pOrder[r]] = cDesc[r];
    } else {
        int B = max(2, min(N / 2, N / 10 + 2));
        vector<char> pUsed(N, 0), cUsed(N, 0);
        for (int i = 0; i < B; i++) {
            int p = pOrder[i], c = cDesc[N - 1 - i];
            matchPtoC[p] = c; pUsed[p] = 1; cUsed[c] = 1;
        }
        int placed = 0;
        for (int i = B; i < N && placed < B; i++) {
            int p = pOrder[i];
            if (pUsed[p]) continue;
            int c = cDesc[placed]; // C's top band (real hubs)
            matchPtoC[p] = c; pUsed[p] = 1; cUsed[c] = 1;
            placed++;
        }
        // leftover: any unused P paired with any unused C, in degree-rank
        // order (not correlated in a meaningful way -- these are all fairly
        // generic nodes on both sides).
        vector<int> leftP, leftC;
        for (int p : pOrder) if (!pUsed[p]) leftP.push_back(p);
        for (int c : cDesc) if (!cUsed[c]) leftC.push_back(c);
        for (size_t i = 0; i < leftP.size(); i++) matchPtoC[leftP[i]] = leftC[i];
    }

    // Local search pool spans BOTH ends of the degree spectrum: the highest-
    // degree P-nodes (currently paired with the weakest comm nodes) AND the
    // lowest-degree P-nodes (currently paired with the comm hubs). A useful
    // hedge -- trading some comm hub's weak-leaf partner for a stronger one,
    // at the cost of exposing it a little to a targeted strike -- requires
    // swapping across exactly these two groups, so both must be reachable by
    // the same local search.
    int half = min(N / 2, max(6, min(100, N / 2)));
    vector<int> pool(pOrder.begin(), pOrder.begin() + half);
    for (int i = max(0, N - half); i < N; i++) pool.push_back(pOrder[i]);
    sort(pool.begin(), pool.end());
    pool.erase(unique(pool.begin(), pool.end()), pool.end());
    int poolSz = (int)pool.size();

    mt19937 rng(12345);
    long long curScore = totalSurvivors(attacks, matchPtoC);
    int iters = min(3000, max(120, 50000 / max(1, N)) * 8);
    for (int it = 0; it < iters && poolSz >= 2; it++) {
        int ia = pool[rng() % poolSz];
        int ib = pool[rng() % poolSz];
        if (ia == ib) continue;
        swap(matchPtoC[ia], matchPtoC[ib]);
        long long ns = totalSurvivors(attacks, matchPtoC);
        if (ns >= curScore) {
            curScore = ns;
        } else {
            swap(matchPtoC[ia], matchPtoC[ib]);  // revert
        }
    }

    for (int i = 0; i < N; i++) printf("%d\n", matchPtoC[i] + 1);
    return 0;
}
