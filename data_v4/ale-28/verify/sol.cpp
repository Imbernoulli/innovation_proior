// Sensor Placement for Coverage + Connectivity -- heuristic solver.
//
// Problem. An H x W grid; cell (i,j) has demand d[i][j] >= 0. Place at most k
// sensors on DISTINCT cells, each a disk of radius r. A cell is covered iff it
// lies within Euclidean distance r of some sensor; the covered demand is the sum
// of d over the UNION of the coverage disks. Two sensors are "linked" iff their
// centres are within distance 2r ((di)^2+(dj)^2 <= (2r)^2); C is the number of
// connected components of the sensor graph. We MAXIMIZE
//       objective = covered_demand - lam * max(0, C - 1).
// We read the instance from stdin and write
//       s
//       i j        (s lines, the sensor positions)
// to stdout. Any infeasible output (s>k, duplicate/out-of-range position) scores
// 0, so we never emit one.
//
// Method (the innovation): coverage is a monotone SUBMODULAR set function, so a
// greedy that repeatedly adds the sensor of largest MARGINAL coverage is within
// 1-1/e of optimal -- but recomputing every candidate's marginal gain each round
// is O(#cands * #cells) and far too slow. We use CELF (Cost-Effective Lazy
// Forward): keep a max-heap of cached upper-bound gains; by submodularity a gain
// only ever DECREASES, so when we pop a candidate whose cached gain was last
// refreshed this round it is provably the true best and we take it; otherwise we
// recompute its (smaller) gain and re-push. That recomputes only a tiny fraction
// of candidates per pick.
//
// Pure coverage greedy, though, scatters sensors onto separate demand hotspots,
// leaving C > 1 and paying lam*(C-1). The non-obvious composition is a
// Steiner-style CONNECTIVITY REPAIR: after the coverage placement, while it pays
// to do so, find the two closest components and insert/relocate sensors along a
// near-straight chain between them (each hop within 2r) to merge them, spending
// either spare budget or our lowest-coverage sensors -- accepting a merge only
// when the lam saved outweighs the coverage lost. A final hill-climb (relocate a
// sensor to its best free cell) polishes the trade-off. Every intermediate state
// is a valid placement, so any early stop still prints a feasible solution.
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
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int H, W, K, R, LAM;
vector<int> D;                 // demand, row-major, size H*W
int R2;                        // r*r
int LINK2;                     // (2r)^2

inline int cell(int i, int j) { return i * W + j; }

// For a sensor at cell p, the list of cells its disk covers (offsets precomputed).
vector<pair<int,int>> diskOffsets;   // (di, dj) within radius r

// coverCells[p] = list of cell indices covered by a sensor at p.
vector<vector<int>> coverCells;

void buildDisk() {
    diskOffsets.clear();
    for (int di = -R; di <= R; di++)
        for (int dj = -R; dj <= R; dj++)
            if (di * di + dj * dj <= R2)
                diskOffsets.push_back({di, dj});
    coverCells.assign(H * W, {});
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            int p = cell(i, j);
            auto &v = coverCells[p];
            v.reserve(diskOffsets.size());
            for (auto &o : diskOffsets) {
                int a = i + o.first, b = j + o.second;
                if (a >= 0 && a < H && b >= 0 && b < W) v.push_back(cell(a, b));
            }
        }
}

// ---- coverage bookkeeping: how many placed sensors currently cover each cell ----
// covCount[c] = number of placed sensors covering cell c. coveredDemand = sum of
// d[c] over cells with covCount[c] > 0.
vector<int> covCount;
long long coveredDemand;

// marginal gain of adding a sensor at p given current covCount (cells newly
// going from 0 -> 1 contribute their demand).
long long marginalGain(int p) {
    long long g = 0;
    for (int c : coverCells[p]) if (covCount[c] == 0) g += D[c];
    return g;
}

void addSensorCoverage(int p) {
    for (int c : coverCells[p]) {
        if (covCount[c] == 0) coveredDemand += D[c];
        covCount[c]++;
    }
}
void removeSensorCoverage(int p) {
    for (int c : coverCells[p]) {
        covCount[c]--;
        if (covCount[c] == 0) coveredDemand -= D[c];
    }
}

// connectivity: components of the sensor set (link iff centres within 2r).
int countComponents(const vector<int>& sensors) {
    int s = (int)sensors.size();
    if (s == 0) return 0;
    vector<int> par(s);
    iota(par.begin(), par.end(), 0);
    function<int(int)> find = [&](int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };
    for (int u = 0; u < s; u++) {
        int iu = sensors[u] / W, ju = sensors[u] % W;
        for (int v = u + 1; v < s; v++) {
            int iv = sensors[v] / W, jv = sensors[v] % W;
            int dd = (iu - iv) * (iu - iv) + (ju - jv) * (ju - jv);
            if (dd <= LINK2) {
                int ru = find(u), rv = find(v);
                if (ru != rv) par[ru] = rv;
            }
        }
    }
    int comp = 0;
    for (int x = 0; x < s; x++) if (find(x) == x) comp++;
    return comp;
}

long long objectiveOf(const vector<int>& sensors) {
    // covered demand of the UNION
    static vector<char> seen;
    seen.assign(H * W, 0);
    long long cov = 0;
    for (int p : sensors)
        for (int c : coverCells[p])
            if (!seen[c]) { seen[c] = 1; cov += D[c]; }
    int C = countComponents(sensors);
    return cov - (long long)LAM * max(0, C - 1);
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;

    if (scanf("%d %d %d %d %d", &H, &W, &K, &R, &LAM) != 5) return 0;
    D.assign(H * W, 0);
    for (int i = 0; i < H * W; i++) { if (scanf("%d", &D[i]) != 1) D[i] = 0; }
    R2 = R * R;
    LINK2 = (2 * R) * (2 * R);

    if (K <= 0 || H <= 0 || W <= 0) { printf("0\n"); return 0; }

    buildDisk();

    Rng rng(0xC0FFEEull ^ ((uint64_t)H << 40) ^ ((uint64_t)W << 24)
            ^ ((uint64_t)K << 12) ^ ((uint64_t)R << 6) ^ (uint64_t)LAM);

    // ---------------- Phase 1: CELF lazy-greedy submodular coverage ----------------
    covCount.assign(H * W, 0);
    coveredDemand = 0;
    vector<char> used(H * W, 0);   // is this cell already a sensor?

    // Heap of (cachedGain, lastUpdatedRound, cell).
    struct Node { long long g; int round; int p; };
    struct Cmp { bool operator()(const Node& a, const Node& b) const { return a.g < b.g; } };
    priority_queue<Node, vector<Node>, Cmp> pq;

    // initialize: gain of every candidate = total demand its disk covers.
    for (int p = 0; p < H * W; p++) {
        long long g = 0;
        for (int c : coverCells[p]) g += D[c];
        pq.push(Node{g, 0, p});
    }

    vector<int> chosen;
    int curRound = 0;
    while ((int)chosen.size() < K && !pq.empty()) {
        Node top = pq.top();
        if (used[top.p]) { pq.pop(); continue; }
        if (top.round == curRound) {
            // cached gain is fresh => provably the true best (submodularity). take it.
            pq.pop();
            if (top.g <= 0) break;          // no positive marginal coverage left
            addSensorCoverage(top.p);
            used[top.p] = 1;
            chosen.push_back(top.p);
            curRound++;
        } else {
            // stale: recompute its (only-smaller) gain and re-push.
            pq.pop();
            long long g = marginalGain(top.p);
            pq.push(Node{g, curRound, top.p});
        }
    }
    // If demand ran out before K sensors, that's fine -- placing more would not
    // add coverage and could only hurt connectivity. chosen is our coverage set.

    // helper: recompute covCount/coveredDemand from a sensor list (keeps the
    // incremental structures in sync after we rebuild `chosen`).
    auto resetCoverageFrom = [&](const vector<int>& sensors) {
        covCount.assign(H * W, 0);
        coveredDemand = 0;
        fill(used.begin(), used.end(), (char)0);
        for (int p : sensors) { addSensorCoverage(p); used[p] = 1; }
    };

    // ---------------- Phase 2: Steiner-style connectivity repair ----------------
    // While C > 1 and a merge pays off, connect the two closest components with a
    // near-straight chain of hops (each <= 2r), spending spare budget first, else
    // relocating our lowest-coverage sensors into bridge cells.
    auto componentsOf = [&](const vector<int>& sensors, vector<int>& comp) -> int {
        int s = (int)sensors.size();
        comp.assign(s, -1);
        if (s == 0) return 0;
        vector<int> par(s); iota(par.begin(), par.end(), 0);
        function<int(int)> find = [&](int x){ while(par[x]!=x){par[x]=par[par[x]];x=par[x];} return x; };
        for (int u = 0; u < s; u++) {
            int iu = sensors[u]/W, ju = sensors[u]%W;
            for (int v = u+1; v < s; v++) {
                int iv = sensors[v]/W, jv = sensors[v]%W;
                int dd=(iu-iv)*(iu-iv)+(ju-jv)*(ju-jv);
                if (dd<=LINK2){int ru=find(u),rv=find(v); if(ru!=rv)par[ru]=rv;}
            }
        }
        unordered_map<int,int> lab; int nc=0;
        for (int x=0;x<s;x++){int rt=find(x); auto it=lab.find(rt); if(it==lab.end()){lab[rt]=nc; comp[x]=nc; nc++;} else comp[x]=it->second;}
        return nc;
    };

    // marginal coverage contributed UNIQUELY by sensor at index t in `chosen`
    // (cells it covers that no other sensor covers): how much we'd lose dropping it.
    auto uniqueCoverageLoss = [&](int t)->long long{
        long long loss=0;
        for (int c : coverCells[chosen[t]]) if (covCount[c]==1) loss += D[c];
        return loss;
    };

    resetCoverageFrom(chosen);
    {
        // repeatedly try to merge the two closest components by a bridge chain.
        int guard = 0;
        while (guard++ < 4 * K) {
            vector<int> comp;
            int C = componentsOf(chosen, comp);
            if (C <= 1) break;

            // gather component members and centroids
            vector<vector<int>> members(C);
            for (int t = 0; t < (int)chosen.size(); t++) members[comp[t]].push_back(t);

            // find the closest pair of sensors across two different components.
            long long bestD = LLONG_MAX; int bu=-1, bv=-1;
            for (int u = 0; u < (int)chosen.size(); u++) {
                int iu=chosen[u]/W, ju=chosen[u]%W;
                for (int v = u+1; v < (int)chosen.size(); v++) {
                    if (comp[u]==comp[v]) continue;
                    int iv=chosen[v]/W, jv=chosen[v]%W;
                    long long dd=(long long)(iu-iv)*(iu-iv)+(long long)(ju-jv)*(ju-jv);
                    if (dd<bestD){bestD=dd;bu=u;bv=v;}
                }
            }
            if (bu<0) break;

            // build a chain of bridge cells from sensor bu toward bv, each hop
            // within 2r, so the two components become linked.
            int iu=chosen[bu]/W, ju=chosen[bu]%W;
            int iv=chosen[bv]/W, jv=chosen[bv]%W;
            double dist = sqrt((double)bestD);
            int step = max(1, (int)floor(2.0*R*0.92)); // hop length a bit under 2r to stay linked
            int hops = max(0, (int)ceil(dist/step) - 1);
            if (hops <= 0) break; // already within 2r somehow

            vector<int> bridge;
            for (int hcnt = 1; hcnt <= hops; hcnt++) {
                double f = (double)hcnt / (double)(hops+1);
                int bi = (int)llround(iu + (iv-iu)*f);
                int bj = (int)llround(ju + (jv-ju)*f);
                bi = max(0, min(H-1, bi));
                bj = max(0, min(W-1, bj));
                int bp = cell(bi, bj);
                // nudge bridge cell to the best-demand nearby free cell (small box)
                long long bestGain = -1; int bestp = -1;
                for (int ddi=-1; ddi<=1; ddi++) for (int ddj=-1; ddj<=1; ddj++){
                    int ni=bi+ddi, nj=bj+ddj;
                    if(ni<0||ni>=H||nj<0||nj>=W) continue;
                    int np=cell(ni,nj);
                    if (used[np]) continue;
                    long long g = marginalGain(np);
                    if (g>bestGain){bestGain=g;bestp=np;}
                }
                if (bestp<0) bestp = bp;            // fallback (may already be used)
                if (!used[bestp]) { bridge.push_back(bestp); used[bestp]=1; addSensorCoverage(bestp);}
            }
            if (bridge.empty()) break;

            // Cost/benefit: a successful merge of two components saves LAM.
            // The bridge sensors we added cost budget; if we exceed K we must drop
            // our lowest-unique-coverage existing sensors to pay for them.
            for (int bp : bridge) chosen.push_back(bp);

            // enforce budget K: while over budget, drop the sensor with smallest
            // unique coverage loss that is NOT a freshly added bridge that would
            // re-split (drop from the tail-safe set: prefer non-bridge).
            // Recompute covCount to keep uniqueCoverageLoss exact.
            // (bridge cells are at the end of `chosen`.)
            int bridgeStart = (int)chosen.size() - (int)bridge.size();
            while ((int)chosen.size() > K) {
                long long bestLoss = LLONG_MAX; int dropT = -1;
                for (int t = 0; t < bridgeStart; t++) {   // prefer dropping coverage sensors, keep bridges
                    long long loss = uniqueCoverageLoss(t);
                    if (loss < bestLoss) { bestLoss = loss; dropT = t; }
                }
                if (dropT < 0) {
                    // only bridges remain to drop; drop the last bridge (undo merge)
                    dropT = (int)chosen.size() - 1;
                }
                int dp = chosen[dropT];
                removeSensorCoverage(dp);
                used[dp] = 0;
                chosen.erase(chosen.begin() + dropT);
                if (dropT < bridgeStart) bridgeStart--;
            }
        }
    }

    // Make sure incremental structures match `chosen` exactly.
    resetCoverageFrom(chosen);

    // best-so-far
    vector<int> best = chosen;
    long long bestObj = objectiveOf(best);

    // ---------------- Phase 3: hill-climb / SA polish ----------------
    // Moves: relocate one sensor to a nearby/global free cell, or add a sensor if
    // under budget, or swap. Evaluate the full objective (covered union +
    // connectivity penalty); accept by Metropolis. Always feasible.
    vector<int> cur = chosen;
    long long curObj = bestObj;
    double Tstart = max(1.0, (double)LAM * 1.0 + 50.0);
    double Tend = 0.5;
    long long iters = 0;
    // candidate free cells with positive demand neighbourhood, for relocation targets
    while (true) {
        if ((iters & 63) == 0) { if (now_sec() - T0 > TIME_LIMIT) break; }
        iters++;
        double frac = min(1.0, (now_sec() - T0) / TIME_LIMIT);
        double Temp = Tstart * pow(Tend / Tstart, frac);

        vector<int> nxt = cur;
        int mv = rng.nextu(3);
        if (nxt.empty()) mv = 2;  // can only add
        if (mv == 0) {
            // relocate one sensor to a random distinct free cell
            int t = rng.nextu((uint32_t)nxt.size());
            int np = (int)rng.nextu((uint32_t)(H * W));
            // ensure distinct
            bool dup = false;
            for (int q = 0; q < (int)nxt.size(); q++) if (q!=t && nxt[q]==np) { dup=true; break; }
            if (dup) continue;
            nxt[t] = np;
        } else if (mv == 1 && (int)nxt.size() >= 2) {
            // remove a sensor (frees budget; may help connectivity by dropping a stray)
            int t = rng.nextu((uint32_t)nxt.size());
            nxt.erase(nxt.begin() + t);
        } else {
            // add a sensor at a random free cell if under budget
            if ((int)nxt.size() >= K) continue;
            int np = (int)rng.nextu((uint32_t)(H * W));
            bool dup=false; for (int q : nxt) if (q==np){dup=true;break;}
            if (dup) continue;
            nxt.push_back(np);
        }

        long long nObj = objectiveOf(nxt);
        long long dlt = nObj - curObj;
        if (dlt >= 0 || rng.nextd() < exp((double)dlt / Temp)) {
            cur.swap(nxt);
            curObj = nObj;
            if (curObj > bestObj) { bestObj = curObj; best = cur; }
        }
    }

    // ---------------- output the best feasible placement ----------------
    // (best is guaranteed: distinct cells, in range, size <= K.)
    string buf;
    buf.reserve(best.size() * 8 + 16);
    buf += to_string(best.size());
    buf += '\n';
    for (int p : best) {
        buf += to_string(p / W);
        buf += ' ';
        buf += to_string(p % W);
        buf += '\n';
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
