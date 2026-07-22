#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Derandomized Dust: Sculpting a Rotor-Router Aggregate".
//
// Input:  L N ; sx sy ; then N lines "x y" -- the target cell coordinates on an
//         L x L torus.
// Output: L lines of L integers -- phase[y][x] in {0,1,2,3}, the INITIAL rotor
//         state of every cell (fixed cyclic exit order E=0,N=1,W=2,S=3).
//
// EMISSION PORTS (fixed, NOT chosen by the solver): particle k (k=1..N) always
// enters at port[k], the k-th cell (1-indexed) discovered by a breadth-first
// search outward from the source that skips the source itself and every
// target cell -- i.e. the N torus cells closest to the source, in BFS order,
// excluding source/target. This is a pure function of L, sx, sy and the target
// set, so the checker and every solution compute the identical port list
// without it needing to be transmitted.
//
// Simulation (rotor-router walk per particle, starting at its own port):
//   repeatedly read the current cell's rotor value, advance that rotor by 1
//   (mod 4), step onto the neighbor it pointed to (torus wraparound); if that
//   neighbor is unoccupied, occupy it and the particle is done; otherwise keep
//   walking from there (firing that cell's rotor again).
//
// ODOMETER BUDGET (feasibility, not scoring): the total number of rotor firings
// across all N particles (S, the ODOMETER) may not exceed BUDGET = 6 * D0,
// where D0 is the sum, over the N fixed ports, of that port's BFS distance to
// the nearest target cell (a fixed, instance-only quantity -- not dependent on
// the participant's phases). Any phase choice whose induced walk wastes the
// odometer beyond that budget is INFEASIBLE (scores 0): the derandomized walk
// must actually settle efficiently, not merely eventually.
//
// Objective (MAX), computed once for the participant's phases (-> F) and once
// for the textbook default phases, all zero (-> B, the checker's own baseline
// construction -- what "trivial" reproduces exactly):
//   J = |occupied ∩ target| / |occupied ∪ target|   (shape match; since
//       |occupied| = |target| = N, this is inter/(2N-inter), 1 iff exact match)
//   quality = J   (subject to the odometer budget above being met)
// Score: sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
//   -> phase=0-everywhere reproduces B exactly, so it scores ratio == 0.1.
// -----------------------------------------------------------------------------

static int L, N, sx, sy;
static int DX[4] = {1, 0, -1, 0};
static int DY[4] = {0, 1, 0, -1};
static vector<int> portArr;   // portArr[k] = cell index particle k (0-indexed) enters at

inline int wrapc(int v) { v %= L; if (v < 0) v += L; return v; }
inline int idxOf(int x, int y) { return y * L + x; }
inline int torusDist(int x1, int y1, int x2, int y2) {
    int dx = abs(x1 - x2); dx = min(dx, L - dx);
    int dy = abs(y1 - y2); dy = min(dy, L - dy);
    return dx + dy;
}

void buildPorts(const vector<char>& target) {
    vector<char> seen(L * (size_t)L, 0);
    vector<int> disc;
    queue<int> q;
    int srcIdx = idxOf(sx, sy);
    seen[srcIdx] = 1;
    q.push(srcIdx);
    while (!q.empty() && (int)disc.size() < N) {
        int cur = q.front(); q.pop();
        int cx = cur % L, cy = cur / L;
        if (cur != srcIdx && !target[cur]) disc.push_back(cur);
        for (int d = 0; d < 4; d++) {
            int nx = wrapc(cx + DX[d]), ny = wrapc(cy + DY[d]);
            int ni = idxOf(nx, ny);
            if (!seen[ni]) { seen[ni] = 1; q.push(ni); }
        }
    }
    if (disc.empty()) disc.push_back(srcIdx);   // pathological fallback, never used in practice
    portArr.resize(N);
    for (int i = 0; i < N; i++) portArr[i] = disc[i % disc.size()];
}

struct SimResult { vector<char> occ; ll S; ll D; bool ok; };
static const ll PER_PARTICLE_CAP = 20000;

SimResult simulate(vector<int> rotor) {
    vector<char> occ(L * (ll)L, 0);
    ll S = 0, D = 0;
    for (int k = 0; k < N; k++) {
        int start = portArr[k];
        int cx = start % L, cy = start / L;
        ll steps = 0;
        int finalIdx = -1;
        while (true) {
            int cur = idxOf(cx, cy);
            int dir = rotor[cur];
            rotor[cur] = (dir + 1) & 3;
            cx = wrapc(cx + DX[dir]);
            cy = wrapc(cy + DY[dir]);
            steps++; S++;
            if (steps > PER_PARTICLE_CAP) return {occ, S, D, false};
            int nxt = idxOf(cx, cy);
            if (!occ[nxt]) { occ[nxt] = 1; finalIdx = nxt; break; }
        }
        int fx = finalIdx % L, fy = finalIdx / L;
        int stx = start % L, sty = start / L;
        D += torusDist(stx, sty, fx, fy);
    }
    return {occ, S, D, true};
}

double computeQuality(const SimResult& r, const vector<char>& target) {
    ll inter = 0;
    for (int i = 0; i < L * L; i++) if (r.occ[i] && target[i]) inter++;
    double uni = 2.0 * N - (double)inter;
    double J = uni > 0 ? (double)inter / uni : 0.0;
    return J;
}

// D0 = sum over the N fixed ports of that port's BFS distance to the nearest
// target cell (multi-source BFS from the target set). A fixed, phase-independent
// quantity used only to size the odometer budget.
ll computeD0(const vector<char>& target) {
    vector<int> dist(L * (size_t)L, -1);
    queue<int> q;
    for (int i = 0; i < L * L; i++) if (target[i]) { dist[i] = 0; q.push(i); }
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        int cx = cur % L, cy = cur / L;
        for (int d = 0; d < 4; d++) {
            int nx = wrapc(cx + DX[d]), ny = wrapc(cy + DY[d]);
            int ni = idxOf(nx, ny);
            if (dist[ni] == -1) { dist[ni] = dist[cur] + 1; q.push(ni); }
        }
    }
    ll D0 = 0;
    for (int p : portArr) D0 += dist[p];
    return D0;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    L = inf.readInt();
    N = inf.readInt();
    sx = inf.readInt();
    sy = inf.readInt();
    vector<char> target(L * (ll)L, 0);
    for (int i = 0; i < N; i++) {
        int x = inf.readInt(0, L - 1);
        int y = inf.readInt(0, L - 1);
        target[idxOf(x, y)] = 1;
    }
    buildPorts(target);

    vector<int> phase(L * (ll)L);
    for (int y = 0; y < L; y++)
        for (int x = 0; x < L; x++)
            phase[idxOf(x, y)] = ouf.readInt(0, 3, "phase");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the L x L phase grid");

    SimResult res = simulate(phase);
    if (!res.ok)
        quitf(_wa, "a particle exceeded the routing step cap (this phase configuration never settles)");

    ll D0 = computeD0(target);
    ll BUDGET = 6 * max((ll)1, D0);
    if (res.S > BUDGET)
        quitf(_wa, "odometer budget exceeded: spent %lld rotor firings > budget %lld (6x the ports' distance-to-target sum)",
              res.S, BUDGET);

    double quality = computeQuality(res, target);
    ll F = (ll)llround(quality * 1000000.0);
    if (F < 0) F = 0;

    vector<int> defPhase(L * (ll)L, 0);
    SimResult resB = simulate(defPhase);
    double qualityB = resB.ok ? computeQuality(resB, target) : 0.0;
    ll B = (ll)llround(qualityB * 1000000.0);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
