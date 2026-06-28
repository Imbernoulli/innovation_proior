// Prize-Collecting Patrol -- heuristic solver.
//
// Objective: starting and ending at a fixed DEPOT, choose a SUBSET of the n
// optional nodes and a visiting order for them, forming a closed tour
//     depot -> p[0] -> ... -> p[k-1] -> depot,
// to MAXIMIZE   profit = (sum of prizes of visited nodes) - (Euclidean travel).
// Read the instance from stdin; write k then the k chosen node ids (visiting
// order), one per line, to stdout. k = 0 (visit nothing, profit 0) is a legal,
// always-feasible answer and is our safety net.
//
// Method (the innovation):
//   * Tour kept as a doubly linked list over node ids plus a virtual DEPOT, so
//     prev/next of every visited node is O(1). `inTour[v]` flags membership.
//   * Construction: greedy cheapest-insertion of profitable nodes (insert the
//     node whose prize minus insertion-detour is largest, while positive).
//   * ONE fused local-search neighbourhood, all moves with an O(1) gain test from
//     cached prev/next:
//       - ADD v after u:   gain = prize[v] - (d(u,v)+d(v,next[u]) - d(u,next[u]))
//       - DROP v:          gain = -prize[v] + (d(prev[v],v)+d(v,next[v])
//                                              - d(prev[v],next[v]))
//       - RELOCATE v (Or-opt-1): drop v then re-add it at its best candidate slot
//         -- this simultaneously reorders the visited set AND can be combined with
//         toggles; the in/out toggle is the move competitors miss when they fix
//         the visited set first.
//       - 2-opt (un-cross two tour edges) restricted to candidate neighbours,
//         done on an array snapshot, to clean up route crossings.
//   * Candidate lists: each node's K nearest other nodes (and the depot), built
//     once from a uniform spatial grid, so ADD/RELOCATE only try good slots.
//   * The inner loop is a deterministic descent that sweeps ADD/DROP/RELOCATE +
//     2-opt to a local optimum. The outer loop is iterated local search: kick a
//     few nodes in/out, re-descend, and accept the new local optimum by an
//     SA-style rule (so small valleys can be crossed). We always remember the
//     best feasible tour seen and print THAT.
// The linked list is a valid tour at all times, so any early stop (time limit)
// still yields a feasible solution.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }  // [0, m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N;                 // number of optional nodes
int DEPOT;             // virtual id == N
vector<double> X, Y;   // coords, size N+1 (index N = depot)
vector<double> P;      // prize, size N+1 (depot prize = 0)

static inline double dist(int a, int b) {
    double dx = X[a] - X[b];
    double dy = Y[a] - Y[b];
    return sqrt(dx * dx + dy * dy);
}

// ---- tour state (doubly linked list over visited ids + depot) ----
vector<int> nxt, prv;     // size N+1
vector<char> inTour;      // size N+1 (depot always "in")
double curProfit = 0.0;   // profit of the current tour

// candidate neighbours (nearest nodes) per node
vector<vector<int>> cand;

// insert v immediately after u in the linked list; updates curProfit by delta
static inline void linkAfter(int u, int v) {
    int w = nxt[u];
    nxt[u] = v; prv[v] = u;
    nxt[v] = w; prv[w] = v;
    inTour[v] = 1;
    curProfit += P[v] - (dist(u, v) + dist(v, w) - dist(u, w));
}

// remove v from the linked list; updates curProfit by delta
static inline void unlink(int v) {
    int u = prv[v], w = nxt[v];
    nxt[u] = w; prv[w] = u;
    inTour[v] = 0;
    curProfit += -P[v] + (dist(u, v) + dist(v, w) - dist(u, w));
}

// gain of ADDING v after u (without doing it)
static inline double addGain(int u, int v) {
    int w = nxt[u];
    return P[v] - (dist(u, v) + dist(v, w) - dist(u, w));
}
// gain of DROPPING v (without doing it)
static inline double dropGain(int v) {
    int u = prv[v], w = nxt[v];
    return -P[v] + (dist(u, v) + dist(v, w) - dist(u, w));
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;  // wall-clock budget (seconds)

    // ---- read instance ----
    if (scanf("%d", &N) != 1) return 0;
    if (N < 0) N = 0;
    DEPOT = N;
    X.assign(N + 1, 0.0); Y.assign(N + 1, 0.0); P.assign(N + 1, 0.0);
    {
        double dx = 0, dy = 0;
        if (scanf("%lf %lf", &dx, &dy) != 2) { dx = 0; dy = 0; }
        X[DEPOT] = dx; Y[DEPOT] = dy; P[DEPOT] = 0.0;
    }
    for (int i = 0; i < N; i++) {
        double xi, yi, pi;
        if (scanf("%lf %lf %lf", &xi, &yi, &pi) != 3) { xi = 0; yi = 0; pi = 0; }
        X[i] = xi; Y[i] = yi; P[i] = pi;
    }

    // Degenerate: nothing to choose.
    if (N == 0) { printf("0\n"); return 0; }

    Rng rng(0xC0FFEEULL ^ (uint64_t)N * 0x9E3779B97F4A7C15ULL);

    // ---- candidate lists: K nearest nodes per node, via a uniform grid ----
    const int K = min(N - 1 >= 0 ? N - 1 : 0, N <= 600 ? 12 : 8);
    cand.assign(N, {});
    if (K > 0) {
        // grid over the bounding box of nodes (depot included for safety)
        double minx = X[DEPOT], maxx = X[DEPOT], miny = Y[DEPOT], maxy = Y[DEPOT];
        for (int i = 0; i < N; i++) {
            minx = min(minx, X[i]); maxx = max(maxx, X[i]);
            miny = min(miny, Y[i]); maxy = max(maxy, Y[i]);
        }
        double w = max(1.0, maxx - minx), h = max(1.0, maxy - miny);
        int G = max(1, (int)floor(sqrt((double)N / 2.0)));
        double cw = w / G, ch = h / G;
        auto cellOf = [&](int i, int &cx, int &cy) {
            cx = (int)((X[i] - minx) / cw); if (cx < 0) cx = 0; if (cx >= G) cx = G - 1;
            cy = (int)((Y[i] - miny) / ch); if (cy < 0) cy = 0; if (cy >= G) cy = G - 1;
        };
        vector<vector<int>> grid(G * G);
        for (int i = 0; i < N; i++) {
            int cx, cy; cellOf(i, cx, cy);
            grid[cy * G + cx].push_back(i);
        }
        vector<pair<double,int>> buf;
        for (int i = 0; i < N; i++) {
            int cx, cy; cellOf(i, cx, cy);
            buf.clear();
            // expand rings until we have comfortably more than K candidates
            for (int r = 0; ; r++) {
                int x0 = max(0, cx - r), x1 = min(G - 1, cx + r);
                int y0 = max(0, cy - r), y1 = min(G - 1, cy + r);
                buf.clear();
                for (int gy = y0; gy <= y1; gy++)
                    for (int gx = x0; gx <= x1; gx++)
                        for (int j : grid[gy * G + gx])
                            if (j != i) buf.push_back({dist(i, j), j});
                if ((int)buf.size() >= K + 2 || (x0 == 0 && y0 == 0 && x1 == G - 1 && y1 == G - 1))
                    break;
            }
            int kk = min((int)buf.size(), K);
            nth_element(buf.begin(), buf.begin() + kk, buf.end());
            sort(buf.begin(), buf.begin() + kk);
            cand[i].reserve(kk);
            for (int t = 0; t < kk; t++) cand[i].push_back(buf[t].second);
        }
    }

    // ---- linked-list init: empty tour (just the depot looping to itself) ----
    nxt.assign(N + 1, -1); prv.assign(N + 1, -1); inTour.assign(N + 1, 0);
    nxt[DEPOT] = DEPOT; prv[DEPOT] = DEPOT; inTour[DEPOT] = 1;
    curProfit = 0.0;

    // ---- greedy cheapest-insertion construction ----
    // Repeatedly insert the (node, slot) with the largest positive add-gain.
    // To stay cheap, scan each not-in-tour node's best slot among: the depot,
    // and the tour-neighbours of its candidate nodes that are already in tour.
    {
        // seed with the single most profitable node placed at the depot, if any
        int bestSeed = -1; double bestSeedGain = 0.0;
        for (int v = 0; v < N; v++) {
            double g = P[v] - 2.0 * dist(DEPOT, v); // insert between depot and depot
            if (g > bestSeedGain) { bestSeedGain = g; bestSeed = v; }
        }
        if (bestSeed >= 0) linkAfter(DEPOT, bestSeed);

        // iterative greedy insertion
        bool progress = true;
        int guard = 0;
        while (progress && guard++ < N + 5) {
            progress = false;
            double bestG = 1e-9; int bestV = -1, bestU = -1;
            for (int v = 0; v < N; v++) {
                if (inTour[v]) continue;
                // try inserting after the depot and after each in-tour candidate
                double localBest = -1e18; int localU = -1;
                double g0 = addGain(DEPOT, v);
                if (g0 > localBest) { localBest = g0; localU = DEPOT; }
                for (int c : cand[v]) {
                    if (inTour[c]) {
                        double g = addGain(c, v);
                        if (g > localBest) { localBest = g; localU = c; }
                        int pc = prv[c];
                        double g2 = addGain(pc, v);
                        if (g2 > localBest) { localBest = g2; localU = pc; }
                    }
                }
                if (localBest > bestG) { bestG = localBest; bestV = v; bestU = localU; }
            }
            if (bestV >= 0) { linkAfter(bestU, bestV); progress = true; }
            if (now_sec() - T0 > TIME_LIMIT * 0.4) break; // leave time for local search
        }
    }

    // snapshot the best tour seen
    auto snapshot = [&](vector<int> &out) {
        out.clear();
        for (int v = nxt[DEPOT]; v != DEPOT; v = nxt[v]) out.push_back(v);
    };
    vector<int> bestTour; snapshot(bestTour);
    double bestProfit = curProfit;

    // ---- helper local moves ----
    // try the best ADD for node v (not in tour): scan candidate slots, return gain & u
    auto bestAddSlot = [&](int v, int &outU) -> double {
        double best = -1e18; int bu = -1;
        double g0 = addGain(DEPOT, v);
        best = g0; bu = DEPOT;
        for (int c : cand[v]) {
            if (!inTour[c]) continue;
            double g = addGain(c, v);
            if (g > best) { best = g; bu = c; }
            int pc = prv[c];
            double g2 = addGain(pc, v);
            if (g2 > best) { best = g2; bu = pc; }
        }
        outU = bu;
        return best;
    };

    // 2-opt pass on an array snapshot of the tour: un-cross edges using candidate
    // lists. Operates on the sequence depot, t[0..m-1], back to depot.
    auto twoOptPass = [&]() {
        vector<int> t; snapshot(t);
        int m = (int)t.size();
        if (m < 3) return;
        // sequence with depot at both ends: seq[0]=DEPOT, seq[1..m]=t, seq[m+1]=DEPOT
        vector<int> seq; seq.reserve(m + 2);
        seq.push_back(DEPOT);
        for (int v : t) seq.push_back(v);
        seq.push_back(DEPOT);
        vector<int> posInSeq(N + 1, -1);
        for (int i = 1; i <= m; i++) posInSeq[seq[i]] = i;
        bool improved = true; int rounds = 0;
        while (improved && rounds++ < 6) {
            improved = false;
            for (int i = 1; i <= m; i++) {
                int a = seq[i - 1], b = seq[i];
                double dab = dist(a, b);
                int base = (b == DEPOT) ? DEPOT : b;
                if (base == DEPOT) continue;
                for (int c : cand[base]) {
                    int j = posInSeq[c];
                    if (j <= i) continue;            // need j > i, c after b
                    int cc = seq[j], dd = seq[j + 1];
                    double dcd = dist(cc, dd);
                    double dac = dist(a, cc), dbd = dist(b, dd);
                    if (dac + dbd + 1e-9 < dab + dcd) {
                        // reverse seq[i..j]
                        int lo = i, hi = j;
                        while (lo < hi) {
                            swap(seq[lo], seq[hi]);
                            posInSeq[seq[lo]] = lo; posInSeq[seq[hi]] = hi;
                            lo++; hi--;
                        }
                        if (lo == hi) posInSeq[seq[lo]] = lo;
                        improved = true;
                        b = seq[i]; dab = dist(a, b);
                        if (b == DEPOT) break;
                    }
                }
                if (now_sec() - T0 > TIME_LIMIT) break;
            }
            if (now_sec() - T0 > TIME_LIMIT) break;
        }
        // rebuild linked list from seq and recompute profit exactly
        int prevNode = DEPOT;
        double dtot = 0.0, prizetot = 0.0;
        for (int i = 1; i <= m; i++) {
            int v = seq[i];
            nxt[prevNode] = v; prv[v] = prevNode;
            dtot += dist(prevNode, v);
            prizetot += P[v];
            prevNode = v;
        }
        nxt[prevNode] = DEPOT; prv[DEPOT] = prevNode;
        dtot += dist(prevNode, DEPOT);
        curProfit = prizetot - dtot;
    };

    // restore the linked list (and inTour/curProfit) from an ordered id vector
    auto restoreFromVec = [&](const vector<int> &t) {
        for (int v = 0; v < N; v++) inTour[v] = 0;
        inTour[DEPOT] = 1;
        int pn = DEPOT; double dtot = 0.0, prizetot = 0.0;
        for (int v : t) {
            nxt[pn] = v; prv[v] = pn; inTour[v] = 1;
            dtot += dist(pn, v); prizetot += P[v]; pn = v;
        }
        nxt[pn] = DEPOT; prv[DEPOT] = pn;
        dtot += dist(pn, DEPOT);
        curProfit = prizetot - dtot;
    };

    // One full deterministic local-search descent over the FUSED neighbourhood:
    //   ADD every profitable not-in-tour node at its best slot,
    //   DROP every loser (positive drop-gain),
    //   RELOCATE (Or-opt-1) every node whose drop+best-readd is a net gain,
    //   then a 2-opt pass to un-cross edges.
    // Repeat until a sweep makes no improvement or time runs out. Every accepted
    // move is a strict improvement, so curProfit climbs monotonically here.
    auto localDescent = [&]() {
        bool improvedAny = true;
        int sweep = 0;
        while (improvedAny && (now_sec() - T0) < TIME_LIMIT) {
            improvedAny = false;
            sweep++;
            // ADD pass
            for (int v = 0; v < N; v++) {
                if (inTour[v]) continue;
                int u; double g = bestAddSlot(v, u);
                if (g > 1e-7) { linkAfter(u, v); improvedAny = true; }
                if ((v & 1023) == 0 && (now_sec() - T0) > TIME_LIMIT) return;
            }
            // DROP pass
            for (int v = 0; v < N; v++) {
                if (!inTour[v]) continue;
                if (dropGain(v) > 1e-7) { unlink(v); improvedAny = true; }
            }
            // RELOCATE pass (drop then best re-add; commit only if net gain)
            for (int v = 0; v < N; v++) {
                if (!inTour[v]) continue;
                int u0 = prv[v];
                double dg = dropGain(v);
                unlink(v);
                int u; double ag = bestAddSlot(v, u);
                if (dg + ag > 1e-7 && u != u0) {
                    linkAfter(u, v); improvedAny = true;
                } else {
                    linkAfter(u0, v);  // exact revert to original slot
                }
                if ((v & 1023) == 0 && (now_sec() - T0) > TIME_LIMIT) return;
            }
            // 2-opt cleanup (reorders the kept set)
            double before = curProfit;
            twoOptPass();
            if (curProfit > before + 1e-7) improvedAny = true;
        }
    };

    // Initial descent from the greedy construction.
    localDescent();
    if (curProfit > bestProfit) { bestProfit = curProfit; snapshot(bestTour); }

    // ---- iterated local search with a perturbation kick ----
    // Kick = randomly toggle a handful of nodes (force a few in and a few out),
    // then re-descend; accept by an SA-style rule so we can cross small valleys,
    // always tracking the best tour seen.
    double T_start;
    {
        double s = 0; int cnt = 0;
        for (int i = 0; i < N && cnt < 2000; i++)
            for (int c : cand[i]) { s += dist(i, c); if (++cnt >= 2000) break; }
        T_start = (cnt ? s / cnt : 1000.0);
        if (T_start < 1.0) T_start = 1.0;
    }
    double accProfit = bestProfit;       // profit of the "current" ILS state
    vector<int> accTour = bestTour;      // current ILS state tour
    long long kicks = 0;
    while ((now_sec() - T0) < TIME_LIMIT) {
        kicks++;
        double prog = min(1.0, (now_sec() - T0) / TIME_LIMIT);
        double T = T_start * pow(1e-2, prog);  // cooling
        // perturb from the current accepted state
        restoreFromVec(accTour);
        int kickSize = 1 + (int)rng.nextu(4);  // 1..4 forced toggles
        for (int t = 0; t < kickSize; t++) {
            int v = rng.nextu(N);
            if (inTour[v]) {
                unlink(v);                       // force OUT
            } else {
                int u; bestAddSlot(v, u);        // pick a good slot
                linkAfter(u, v);                 // force IN (even if a small loss)
            }
        }
        localDescent();
        double newProfit = curProfit;
        if (newProfit > bestProfit) {
            bestProfit = newProfit; snapshot(bestTour);
        }
        // SA acceptance of the new local optimum as the next ILS state
        if (newProfit >= accProfit || rng.nextd() < exp((newProfit - accProfit) / T)) {
            accProfit = newProfit; snapshot(accTour);
        }
        if ((kicks & 63) == 0 && (now_sec() - T0) > TIME_LIMIT) break;
    }

    // final cleanup descent on the best tour
    restoreFromVec(bestTour);
    localDescent();
    if (curProfit > bestProfit) { bestProfit = curProfit; snapshot(bestTour); }

    // The empty tour (profit 0) is always available; if our best is negative,
    // emit the empty tour instead (never worse than 0).
    if (bestProfit < 0.0) bestTour.clear();

    // ---- output: k then the ids in visiting order ----
    string out;
    out += to_string((int)bestTour.size()); out += "\n";
    for (int v : bestTour) { out += to_string(v); out += "\n"; }
    fputs(out.c_str(), stdout);
    return 0;
}
