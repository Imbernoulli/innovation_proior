// TIER: strong
// Insight: the whole process is a deterministic function of the phases, and
// with N distinct emission ports the ODOMETER argument becomes concrete --
// any intermediate cell used by MORE THAN ONE port's path has its rotor
// dilute across visits (each visit fires a different one of the four
// directions), so the real lever is building port->target ROUTES that avoid
// sharing cells wherever possible, not just picking short routes in
// isolation. Construction: (1) assign each port to its nearest still-free
// target (greedy nearest-pair matching, processed closest-pair-first);
// (2) route each pair with a 0-1 BFS that strongly prefers cells no earlier
// path has already used (a cell already "spent" by another path costs 1 extra,
// an untouched cell costs 0) -- an explicit anti-contention construction;
// (3) polish with a bounded, fully deterministic (fixed iteration count, fixed
// seed -- never wall-clock) local search that resimulates the exact process
// and keeps any phase perturbation that improves the true objective.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int L, N, sx, sy;
static int DX[4] = {1, 0, -1, 0};
static int DY[4] = {0, 1, 0, -1};
static vector<char> target;
static vector<int> portArr;

inline int wrapc(int v) { v %= L; if (v < 0) v += L; return v; }
inline int idxOf(int x, int y) { return y * L + x; }
inline int torusDist(int x1, int y1, int x2, int y2) {
    int dx = abs(x1 - x2); dx = min(dx, L - dx);
    int dy = abs(y1 - y2); dy = min(dy, L - dy);
    return dx + dy;
}

struct SimResult { vector<char> occ; ll S; ll D; bool ok; };
static const ll PER_PARTICLE_CAP = 20000;

SimResult simulate(const vector<int>& phase0) {
    vector<int> rotor = phase0;
    vector<char> occ(L * (size_t)L, 0);
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
double quality(const SimResult& r) {
    if (!r.ok) return -1.0;
    ll inter = 0;
    for (int i = 0; i < L * L; i++) if (r.occ[i] && target[i]) inter++;
    double uni = 2.0 * N - inter;
    return uni > 0 ? (double)inter / uni : 0.0;
}

int main() {
    scanf("%d %d", &L, &N);
    scanf("%d %d", &sx, &sy);
    target.assign(L * (size_t)L, 0);
    vector<int> targetList(N);
    for (int i = 0; i < N; i++) {
        int x, y; scanf("%d %d", &x, &y);
        target[idxOf(x, y)] = 1;
        targetList[i] = idxOf(x, y);
    }

    // fixed emission ports: BFS from source, skip source & target cells
    vector<char> seen(L * (size_t)L, 0);
    vector<int> disc;
    queue<int> q;
    int srcIdx = idxOf(sx, sy);
    seen[srcIdx] = 1; q.push(srcIdx);
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
    if (disc.empty()) disc.push_back(srcIdx);
    portArr.resize(N);
    for (int i = 0; i < N; i++) portArr[i] = disc[i % disc.size()];

    // (1) nearest-pair greedy matching: process ports in BFS-discovery order,
    // each claims its nearest still-free target.
    vector<char> taken(L * (size_t)L, 0);
    vector<pair<int,int>> pairs;
    for (int p : disc) {
        if ((int)pairs.size() >= N) break;
        int px = p % L, py = p / L;
        int best = -1, bestd = INT_MAX;
        for (int t : targetList) {
            if (taken[t]) continue;
            int tx = t % L, ty = t / L;
            int d = torusDist(px, py, tx, ty);
            if (d < bestd) { bestd = d; best = t; }
        }
        if (best == -1) break;
        taken[best] = 1;
        pairs.push_back({p, best});
    }
    for (int t : targetList)
        if (!taken[t]) { pairs.push_back({disc[(int)pairs.size() % disc.size()], t}); taken[t] = 1; }
    sort(pairs.begin(), pairs.end(), [&](const pair<int,int>& a, const pair<int,int>& b) {
        int ax = a.first % L, ay = a.first / L, atx = a.second % L, aty = a.second / L;
        int bx = b.first % L, by = b.first / L, btx = b.second % L, bty = b.second / L;
        return torusDist(ax, ay, atx, aty) < torusDist(bx, by, btx, bty);
    });

    // (2) contention-aware routing: 0-1 BFS preferring cells no earlier path used
    vector<int> phase(L * (size_t)L, 0);
    vector<int> usedCount(L * (size_t)L, 0);
    for (auto& pr : pairs) {
        int s = pr.first, t = pr.second;
        vector<int> pdist(L * (size_t)L, INT_MAX);
        vector<int> parent(L * (size_t)L, -1);
        deque<int> dq;
        pdist[s] = 0; dq.push_back(s);
        while (!dq.empty()) {
            int cur = dq.front(); dq.pop_front();
            if (cur == t) break;
            int cx = cur % L, cy = cur / L;
            for (int d = 0; d < 4; d++) {
                int nx = wrapc(cx + DX[d]), ny = wrapc(cy + DY[d]);
                int ni = idxOf(nx, ny);
                int w = usedCount[ni] > 0 ? 1 : 0;
                if (pdist[cur] + w < pdist[ni]) {
                    pdist[ni] = pdist[cur] + w; parent[ni] = cur;
                    if (w == 0) dq.push_front(ni); else dq.push_back(ni);
                }
            }
        }
        if (parent[t] != -1 || s == t) {
            vector<int> path;
            int cur = t;
            while (cur != s) { path.push_back(cur); cur = parent[cur]; }
            path.push_back(s);
            reverse(path.begin(), path.end());
            for (size_t i = 0; i + 1 < path.size(); i++) {
                int a = path[i], b = path[i + 1];
                int ax = a % L, ay = a / L, bx = b % L, by = b / L;
                for (int d = 0; d < 4; d++)
                    if (wrapc(ax + DX[d]) == bx && wrapc(ay + DY[d]) == by) { phase[a] = d; break; }
                usedCount[a]++;
            }
        }
    }

    // (3) bounded deterministic local search polish (fixed seed, fixed
    // iteration budget -- a function of L only, never wall-clock)
    mt19937 rng(1234567u);
    SimResult cur = simulate(phase);
    double curQ = quality(cur);
    double bestQ = curQ;
    vector<int> bestPhase = phase;
    int ITERS = min(300000, 60000 + 3000 * L);
    for (int it = 0; it < ITERS; it++) {
        vector<int> trial = phase;
        int nflip = 1 + rng() % 6;
        for (int f = 0; f < nflip; f++) {
            int cell = rng() % (L * L);
            trial[cell] = rng() % 4;
        }
        SimResult r = simulate(trial);
        double q = quality(r);
        if (q >= curQ) { phase = trial; curQ = q; }
        if (q > bestQ) { bestQ = q; bestPhase = trial; }
    }

    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) printf("%d%c", bestPhase[idxOf(x, y)], x + 1 < L ? ' ' : '\n');
    }
    return 0;
}
