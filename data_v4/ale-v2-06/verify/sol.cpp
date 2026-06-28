#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------------------------------
// Dense Weighted Independent Set  (ALE-Bench heuristic optimization)
//
// Read an undirected weighted graph from stdin, output an INDEPENDENT SET of vertex
// ids maximizing total weight. We must ALWAYS print a feasible independent set within
// the time budget; the empty set is the trivial safety net (weight 0), so we only ever
// replace it with something strictly better.
//
// Strongest standard heuristic for MWIS: a TIGHTNESS-based local search
// (Andrade-Resende-Werneck): keep, for every vertex, tight[v] = number of its
// neighbours currently in the solution. A non-solution vertex is FREE iff tight[v]==0,
// i.e. it can be added without breaking independence. Inserting / removing a vertex
// only changes its neighbours' tightness, so every move is O(deg(v)) -- never O(n).
// On top of greedy construction we run:
//   * (0,1) add  : insert any free vertex (always increases weight),
//   * (1,2) swap : remove one solution vertex x, then add the (>=1) vertices that
//                  became free, choosing a non-adjacent improving pair -- the move
//                  that classic "remove-blocking-vertex" search cannot see,
// wrapped in simulated annealing with random perturbation (force a vertex in, evict
// its solution-neighbours) so we escape the deep local optima dense MWIS is full of.
// The incremental weight delta of every move is O(deg), evaluated without recomputing
// the objective from scratch.
// ------------------------------------------------------------------------------------

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// xorshift128+ style fast RNG
struct RNG {
    uint64_t s;
    RNG(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    inline uint64_t next() {
        uint64_t x = s;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        s = x;
        return x;
    }
    inline uint32_t u32() { return (uint32_t)(next() >> 32); }
    inline int randint(int n) { return (int)(u32() % (uint32_t)n); }      // [0,n)
    inline double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N;
long long M;
vector<int> W;
vector<int> adjStart;          // CSR adjacency
vector<int> adjList;
vector<int> deg;

// CSR neighbour range of v: [adjStart[v], adjStart[v+1])
static inline int beg(int v) { return adjStart[v]; }
static inline int end_(int v) { return adjStart[v + 1]; }

int main() {
    double T0 = now_sec();
    const double TIME_LIMIT = 1.85;   // seconds wall-clock; keep a feasible set always

    // ---- read instance ----
    if (scanf("%d %lld", &N, &M) != 2) return 0;
    if (N <= 0) { printf("0\n"); return 0; }
    W.resize(N);
    long long totW = 0;
    for (int i = 0; i < N; i++) { scanf("%d", &W[i]); totW += W[i]; }
    deg.assign(N, 0);
    vector<int> ea(M), eb(M);
    for (long long e = 0; e < M; e++) {
        int a, b; scanf("%d %d", &a, &b);
        ea[e] = a; eb[e] = b;
        deg[a]++; deg[b]++;
    }
    adjStart.assign(N + 1, 0);
    for (int i = 0; i < N; i++) adjStart[i + 1] = adjStart[i] + deg[i];
    adjList.assign(adjStart[N], 0);
    {
        vector<int> cur(adjStart.begin(), adjStart.begin() + N);
        for (long long e = 0; e < M; e++) {
            int a = ea[e], b = eb[e];
            adjList[cur[a]++] = b;
            adjList[cur[b]++] = a;
        }
    }
    ea.clear(); ea.shrink_to_fit();
    eb.clear(); eb.shrink_to_fit();

    // ---- solution state ----
    vector<char> inSol(N, 0);
    vector<int> tight(N, 0);       // # solution-neighbours of v
    long long curW = 0;            // weight of current solution

    // toggling helpers keep tight[] / inSol[] / curW consistent; O(deg)
    auto addVertex = [&](int v) {
        // precondition: v not in solution and tight[v]==0 (free)
        inSol[v] = 1;
        curW += W[v];
        for (int p = beg(v); p < end_(v); p++) tight[adjList[p]]++;
    };
    auto removeVertex = [&](int v) {
        inSol[v] = 0;
        curW -= W[v];
        for (int p = beg(v); p < end_(v); p++) tight[adjList[p]]--;
    };

    RNG rng(0xC0FFEEULL ^ (uint64_t)(N) ^ ((uint64_t)M << 17));

    // ---------- construction: GWMIN greedy (by weight, ties smaller id) ----------
    // Take the heaviest still-free vertex, forbid its neighbours, repeat. Feasible by
    // construction. This is the scorer's baseline, so the local search starts already
    // at >= baseline and only climbs.
    {
        vector<int> order(N);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            if (W[a] != W[b]) return W[a] > W[b];
            return a < b;
        });
        for (int v : order) {
            if (tight[v] == 0 && !inSol[v]) addVertex(v);
        }
    }

    // best-so-far snapshot (always feasible)
    vector<char> bestInSol = inSol;
    long long bestW = curW;

    // free-vertex add to local optimum w.r.t. (0,1): add any free vertex.
    auto addAllFree = [&]() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (int v = 0; v < N; v++) {
                if (!inSol[v] && tight[v] == 0) { addVertex(v); changed = true; }
            }
        }
    };

    // ---------- (1,2)-swap local search (the key MWIS move) ----------
    // For a solution vertex x, the vertices that would become FREE if x were removed
    // are exactly x's neighbours whose only solution-neighbour is x (tight==1 and
    // adjacent to x). Removing x (weight w_x) and adding two NON-ADJACENT such
    // candidates u,v improves iff w_u + w_v > w_x. We scan candidates of x and look
    // for the best improving pair. This is the canonical one-improvement step that
    // pure add/remove search cannot reach.
    // We collect, per pass, candidate vertices and try improving (1,2) swaps.
    auto try12swaps = [&]() -> bool {
        bool improvedAny = false;
        // iterate over solution vertices in current order
        // (snapshot solution to avoid iterator issues while we mutate)
        static vector<int> sol;
        sol.clear();
        for (int v = 0; v < N; v++) if (inSol[v]) sol.push_back(v);
        for (int x : sol) {
            if (!inSol[x]) continue;       // may have been removed by a prior swap
            // candidates: neighbours of x with tight==1 (x is their only sol-neighbour)
            // Find best single candidate and best improving pair.
            int wx = W[x];
            // gather candidates
            static vector<int> cand;
            cand.clear();
            for (int p = beg(x); p < end_(x); p++) {
                int u = adjList[p];
                if (!inSol[u] && tight[u] == 1) cand.push_back(u);
            }
            if (cand.empty()) continue;
            // best single (for a (1,1) improving move: w_u > w_x)
            int bestSingle = -1, bestSingleW = -1;
            for (int u : cand) if (W[u] > bestSingleW) { bestSingleW = W[u]; bestSingle = u; }

            // best improving non-adjacent pair: try heaviest candidate against the
            // rest. To keep it cheap we sort candidates by weight desc and probe pairs
            // greedily; candidate lists are small in dense graphs (tight==1 is rare).
            sort(cand.begin(), cand.end(), [&](int a, int b) { return W[a] > W[b]; });
            int pairU = -1, pairV = -1; long long pairW = -1;
            int CS = (int)cand.size();
            // adjacency test via a small hash set of x's neighbourhood is overkill;
            // candidates are few, so O(cand^2) with a direct adjacency probe is fine.
            for (int i = 0; i < CS; i++) {
                int u = cand[i];
                // early stop: if W[u] + W[cand[i+1]] <= wx, no pair starting at i can help
                if (i + 1 < CS && (long long)W[u] + W[cand[i + 1]] <= wx) break;
                for (int j = i + 1; j < CS; j++) {
                    int v = cand[j];
                    long long pw = (long long)W[u] + W[v];
                    if (pw <= wx) break;       // sorted desc: rest only smaller
                    if (pw <= pairW) break;
                    // u,v must be non-adjacent
                    bool adjacent = false;
                    // probe the shorter adjacency list
                    int du = end_(u) - beg(u), dv = end_(v) - beg(v);
                    if (du <= dv) {
                        for (int q = beg(u); q < end_(u); q++) if (adjList[q] == v) { adjacent = true; break; }
                    } else {
                        for (int q = beg(v); q < end_(v); q++) if (adjList[q] == u) { adjacent = true; break; }
                    }
                    if (!adjacent) { pairU = u; pairV = v; pairW = pw; break; }
                }
            }

            // Apply the best improving move at x.
            if (pairU >= 0 && pairW > wx) {
                removeVertex(x);
                // after removing x, both u and v are free (tight became 0) since x was
                // their only solution-neighbour and they are non-adjacent.
                if (tight[pairU] == 0 && !inSol[pairU]) addVertex(pairU);
                if (tight[pairV] == 0 && !inSol[pairV]) addVertex(pairV);
                improvedAny = true;
            } else if (bestSingle >= 0 && bestSingleW > wx) {
                // (1,1) improving swap
                removeVertex(x);
                if (tight[bestSingle] == 0 && !inSol[bestSingle]) addVertex(bestSingle);
                improvedAny = true;
            }
        }
        return improvedAny;
    };

    auto localSearch = [&]() {
        addAllFree();
        // alternate (1,2)-swaps and free-adds until no improvement
        for (int it = 0; it < 1000; it++) {
            bool a = try12swaps();
            addAllFree();
            if (!a) break;
        }
    };

    localSearch();
    if (curW > bestW) { bestW = curW; bestInSol = inSol; }

    // ---------- iterated local search with SA acceptance ----------
    // Perturbation: FORCE a random vertex v into the solution (evicting its
    // solution-neighbours), then re-run local search. SA accepts non-improving kicks
    // with a cooling probability so we traverse plateaus and escape local optima.
    double Tstart = 0.0;
    // scale temperature to typical vertex weight
    {
        long long mx = 1;
        for (int i = 0; i < N; i++) mx = max(mx, (long long)W[i]);
        Tstart = (double)mx * 0.5 + 1.0;
    }
    double Tend = 1.0;

    long long iter = 0;
    // snapshot state we can roll back to (the accepted incumbent of the ILS)
    vector<char> accInSol = inSol;
    long long accW = curW;

    while (true) {
        if ((iter & 63) == 0) {
            double t = now_sec() - T0;
            if (t > TIME_LIMIT) break;
        }
        iter++;

        // forced insertion perturbation
        int v = rng.randint(N);
        if (!inSol[v]) {
            // evict its solution neighbours, then add v
            for (int p = beg(v); p < end_(v); p++) {
                int u = adjList[p];
                if (inSol[u]) removeVertex(u);
            }
            addVertex(v);
        }
        // occasionally a second kick to escape harder traps
        if (rng.uni() < 0.35) {
            int v2 = rng.randint(N);
            if (!inSol[v2]) {
                for (int p = beg(v2); p < end_(v2); p++) {
                    int u = adjList[p];
                    if (inSol[u]) removeVertex(u);
                }
                addVertex(v2);
            }
        }

        localSearch();

        // SA acceptance vs the accepted incumbent
        double frac = (now_sec() - T0) / TIME_LIMIT;
        if (frac > 1.0) frac = 1.0;
        double Temp = Tstart * pow(Tend / Tstart, frac);
        long long delta = curW - accW;
        bool accept;
        if (delta >= 0) accept = true;
        else accept = (rng.uni() < exp((double)delta / Temp));

        if (accept) {
            accInSol = inSol;
            accW = curW;
        } else {
            // roll back to the accepted incumbent
            // rebuild state from accInSol
            // (full O(N+M) rebuild; amortized fine since localSearch dominates)
            curW = 0;
            fill(tight.begin(), tight.end(), 0);
            for (int i = 0; i < N; i++) inSol[i] = accInSol[i];
            for (int i = 0; i < N; i++) if (inSol[i]) {
                curW += W[i];
                for (int p = beg(i); p < end_(i); p++) tight[adjList[p]]++;
            }
        }

        if (curW > bestW) { bestW = curW; bestInSol = inSol; }
    }

    // ---------- output the best feasible independent set ----------
    // bestInSol is always a valid independent set (every state we ever recorded was
    // produced by feasibility-preserving moves). As a final guard we re-derive the
    // list directly from bestInSol.
    vector<int> ans;
    ans.reserve(N);
    for (int i = 0; i < N; i++) if (bestInSol[i]) ans.push_back(i);

    // safety net: if somehow empty, the empty set (k=0) is still feasible (weight 0).
    string out;
    out.reserve(ans.size() * 7 + 16);
    out += to_string((int)ans.size());
    out += "\n";
    for (int v : ans) { out += to_string(v); out += "\n"; }
    fputs(out.c_str(), stdout);
    return 0;
}
