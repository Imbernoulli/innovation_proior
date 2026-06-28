// Knapsack with Synergies -- Quadratic Knapsack Problem (QKP) heuristic solver.
//
// Objective: choose a subset S of items to MAXIMIZE
//     sum_{i in S} v_i  +  sum_{ {i,j}: i,j in S } b_{ij}
// subject to  sum_{i in S} w_i <= W.  Read the instance from stdin, print the
// chosen subset (count then indices) to stdout. An over-budget / malformed
// output scores 0, so we keep a feasible subset at every moment.
//
// Method (the innovation):
//   1. Per-item INCIDENT-SYNERGY LISTS adj[i] = {(j, b_ij)} and a running
//      maintained array g[i] = "synergy item i would gain/lose against the
//      CURRENTLY selected set" = sum of b_ij over selected neighbours j. With g
//      maintained, the exact objective delta of flipping item i is
//          add i :  +v_i + g[i]
//          drop i: -v_i - g[i]
//      a single O(1) read. Flipping i then updates g only along i's incident
//      edges -- O(deg(i)) -- never an O(n^2) / O(p) re-evaluation. This is the
//      whole engine: O(degree) incremental evaluation of the quadratic term.
//   2. Construction: a SYNERGY-AWARE density greedy. Repeatedly add the
//      feasible item maximising (v_i + g[i]) / w_i (marginal value INCLUDING the
//      synergy it forms with the already-chosen set), beating the synergy-blind
//      ratio greedy that the scorer uses as its reference.
//   3. Simulated annealing over flips: add / drop / swap (drop one selected,
//      add one unselected) moves, each evaluated in O(1)+O(deg) via g, accepting
//      worsening moves with the usual Metropolis rule to escape local optima.
//      The incumbent best feasible subset is always retained.
//   4. Fill-up-and-exchange post-pass (Yang/Billionnet style): greedily fill any
//      leftover budget by best marginal density, then 1-1 exchanges, repeated to
//      a local optimum. Always leaves a feasible subset; any early time cutoff
//      still prints the best feasible subset found.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t nextu(uint32_t m) { return m ? (uint32_t)(next() % m) : 0u; }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N;
long long W;
vector<long long> w, v;
// incident synergy lists: adj[i] = list of (neighbour, bonus)
vector<vector<pair<int,long long>>> adj;

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.90;

    if (scanf("%d %lld", &N, &W) != 2) return 0;
    if (N <= 0) { printf("0\n"); return 0; }
    w.assign(N, 0); v.assign(N, 0);
    for (int i = 0; i < N; i++) {
        long long wi, vi;
        if (scanf("%lld %lld", &wi, &vi) != 2) { wi = 1; vi = 0; }
        if (wi < 1) wi = 1;               // guard: weights are positive
        w[i] = wi; v[i] = vi;
    }
    long long p = 0;
    if (scanf("%lld", &p) != 1) p = 0;
    adj.assign(N, {});
    for (long long e = 0; e < p; e++) {
        int a, b; long long bonus;
        if (scanf("%d %d %lld", &a, &b, &bonus) != 3) { a = 0; b = 0; bonus = 0; }
        if (a < 0 || a >= N || b < 0 || b >= N || a == b || bonus == 0) continue;
        adj[a].push_back({b, bonus});
        adj[b].push_back({a, bonus});
    }

    // ---------- state ----------
    vector<char> sel(N, 0);            // membership of current subset
    vector<long long> g(N, 0);         // g[i] = sum of b_ij over selected neighbours j
    long long curW = 0;                // total selected weight
    long long curObj = 0;              // current objective

    // flip item i (toggle membership); maintains sel,g,curW,curObj in O(deg(i))
    auto flip = [&](int i) {
        if (!sel[i]) {                 // ADD i
            curObj += v[i] + g[i];     // delta uses g BEFORE update
            curW += w[i];
            sel[i] = 1;
            for (auto &e : adj[i]) g[e.first] += e.second;
        } else {                        // DROP i
            curObj -= v[i] + g[i];
            curW -= w[i];
            sel[i] = 0;
            for (auto &e : adj[i]) g[e.first] -= e.second;
        }
    };

    Rng rng(0xC0FFEEULL ^ (uint64_t)N * 1000003ULL ^ (uint64_t)p * 2654435761ULL ^ (uint64_t)W);

    // ---------- 2. synergy-aware density-greedy construction ----------
    // Add the feasible item maximising (v_i + g[i]) / w_i, repeat. g already
    // reflects synergy to the chosen set, so this captures the quadratic term.
    {
        // Simple repeated scan; N <= 900 so O(N^2) construction is fine.
        while (true) {
            int best = -1;
            double bestKey = -1e18;
            for (int i = 0; i < N; i++) {
                if (sel[i]) continue;
                if (curW + w[i] > W) continue;
                double key = (double)(v[i] + g[i]) / (double)w[i];
                if (key > bestKey) { bestKey = key; best = i; }
            }
            if (best < 0) break;
            // only add if it does not strictly hurt (marginal value >= 0), else stop:
            if (v[best] + g[best] < 0) break;
            flip(best);
        }
    }

    // incumbent best
    vector<char> bestSel = sel;
    long long bestObj = curObj;

    auto saveIfBetter = [&]() {
        if (curObj > bestObj && curW <= W) { bestObj = curObj; bestSel = sel; }
    };
    saveIfBetter();

    // lists of currently in/out items for sampling swap partners
    // (rebuilt lazily; cheap to scan since N is small)
    auto pickIn = [&]() -> int {
        // reservoir pick a selected item
        int chosen = -1, cnt = 0;
        for (int i = 0; i < N; i++) if (sel[i]) { cnt++; if (rng.nextu((uint32_t)cnt) == 0) chosen = i; }
        return chosen;
    };
    auto pickOut = [&]() -> int {
        int chosen = -1, cnt = 0;
        for (int i = 0; i < N; i++) if (!sel[i]) { cnt++; if (rng.nextu((uint32_t)cnt) == 0) chosen = i; }
        return chosen;
    };

    // ---------- 3. simulated annealing over add / drop / swap flips ----------
    // delta of a candidate move is read from g in O(1); a swap is two coupled
    // flips so its delta accounts for the synergy edge between the two items.
    double Tcur = 0.0;
    // scale the temperature to typical bonus magnitudes
    {
        long long mx = 1;
        for (int i = 0; i < N; i++) mx = max(mx, v[i]);
        for (int i = 0; i < N; i++) for (auto &e : adj[i]) mx = max(mx, e.second);
        Tcur = (double)mx * 2.0 + 1.0;
    }
    double Tstart = Tcur, Tend = 0.05;
    long long iter = 0;
    const double SA_FRAC = 0.86;   // fraction of budget for SA, rest for post-pass
    while (true) {
        if ((iter & 511) == 0) {
            double el = now_sec() - T0;
            if (el > TIME_LIMIT * SA_FRAC) break;
            double frac = el / (TIME_LIMIT * SA_FRAC);
            if (frac > 1) frac = 1;
            // geometric cooling
            Tcur = Tstart * pow(Tend / Tstart, frac);
            if (Tcur < 1e-6) Tcur = 1e-6;
        }
        iter++;
        int moveType = (int)rng.nextu(3);
        if (moveType == 0) {
            // ADD a random unselected item that fits
            int i = pickOut();
            if (i < 0 || curW + w[i] > W) continue;
            long long delta = v[i] + g[i];          // O(1)
            if (delta >= 0 || rng.nextd() < exp((double)delta / Tcur)) {
                flip(i);
                saveIfBetter();
            }
        } else if (moveType == 1) {
            // DROP a random selected item
            int i = pickIn();
            if (i < 0) continue;
            long long delta = -(v[i] + g[i]);        // O(1)
            if (delta >= 0 || rng.nextd() < exp((double)delta / Tcur)) {
                flip(i);
                saveIfBetter();
            }
        } else {
            // SWAP: drop selected i, add unselected j (must keep budget feasible)
            int i = pickIn();
            int j = pickOut();
            if (i < 0 || j < 0) continue;
            if (curW - w[i] + w[j] > W) continue;
            // delta of dropping i then adding j. The synergy edge i-j (if any)
            // is handled correctly because after dropping i, g[j] no longer
            // counts b_ij; we evaluate sequentially to stay exact.
            long long dDrop = -(v[i] + g[i]);
            // tentatively account: after dropping i, g[j] would lose b_ij if edge exists.
            // Easiest exact route: do the two flips and compare objective.
            long long before = curObj;
            flip(i);                                  // drop i  (O(deg i))
            long long dAdd = v[j] + g[j];             // now g[j] reflects i removed
            flip(j);                                  // add j   (O(deg j))
            long long delta = curObj - before;        // exact two-flip delta
            (void)dDrop; (void)dAdd;
            if (delta >= 0 || rng.nextd() < exp((double)delta / Tcur)) {
                saveIfBetter();
            } else {
                // revert: drop j, add i
                flip(j);
                flip(i);
            }
        }
    }

    // restore incumbent best before the deterministic post-pass
    {
        // rebuild state to bestSel
        for (int i = 0; i < N; i++) { sel[i] = 0; g[i] = 0; }
        curW = 0; curObj = 0;
        for (int i = 0; i < N; i++) if (bestSel[i]) flip(i);
    }

    // ---------- 4. fill-up-and-exchange post-pass to a local optimum ----------
    // Repeat: (a) FILL UP -- add the best positive-marginal-density item that
    // fits; (b) EXCHANGE -- for each selected i and unselected j, if swapping
    // them stays feasible and strictly improves, do it. Loop until no change or
    // time out. Pure hill-climbing, so it only ever improves the incumbent.
    {
        bool improved = true;
        while (improved) {
            if (now_sec() - T0 > TIME_LIMIT) break;
            improved = false;

            // (a) fill up
            while (true) {
                int best = -1; double bestKey = 0.0; bool any = false;
                for (int i = 0; i < N; i++) {
                    if (sel[i] || curW + w[i] > W) continue;
                    long long marg = v[i] + g[i];
                    if (marg <= 0) continue;
                    double key = (double)marg / (double)w[i];
                    if (!any || key > bestKey) { bestKey = key; best = i; any = true; }
                }
                if (best < 0) break;
                flip(best);
                improved = true;
                saveIfBetter();
                if (now_sec() - T0 > TIME_LIMIT) break;
            }

            // (b) 1-1 exchanges (first-improvement). Bound the work by scanning
            // selected x unselected; N small so this is affordable per sweep.
            for (int i = 0; i < N && (now_sec() - T0 <= TIME_LIMIT); i++) {
                if (!sel[i]) continue;
                for (int j = 0; j < N; j++) {
                    if (sel[j]) continue;
                    if (curW - w[i] + w[j] > W) continue;
                    long long before = curObj;
                    flip(i);                          // drop i
                    long long after = curObj + v[j] + g[j];  // if we then add j
                    if (after > before) {
                        flip(j);                      // commit add j
                        improved = true;
                        saveIfBetter();
                        break;                        // i is gone; move to next i
                    } else {
                        flip(i);                      // revert: re-add i
                    }
                }
            }
        }
        saveIfBetter();
    }

    // ---------- emit best feasible subset ----------
    vector<int> outIdx;
    long long checkW = 0;
    for (int i = 0; i < N; i++) if (bestSel[i]) { outIdx.push_back(i); checkW += w[i]; }
    // final safety: if (impossibly) over budget, fall back to empty set.
    if (checkW > W) outIdx.clear();

    string buf;
    buf.reserve(outIdx.size() * 7 + 16);
    buf += to_string((long long)outIdx.size());
    buf += "\n";
    for (size_t k = 0; k < outIdx.size(); k++) {
        buf += to_string(outIdx[k]);
        buf += (k + 1 == outIdx.size()) ? '\n' : ' ';
    }
    if (outIdx.empty()) buf += "\n";
    fputs(buf.c_str(), stdout);
    return 0;
}
