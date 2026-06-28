// Lattice Antenna Coverage (ale-12) -- budgeted monotone-submodular maximum
// coverage solved with cost-benefit LAZY GREEDY (CELF) + single-best guard,
// then incremental-eval SWAP local search. Reads stdin, writes a feasible
// subset of antenna-site indices to stdout. Never crashes, never infeasible.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

int G, M;
long long B;
vector<int> demand;            // demand[y*G+x]
struct Site { int sx, sy, r; long long c; int x0, y0, x1, y1; };
vector<Site> site;

// covered cell counts (how many CHOSEN antennas cover each cell)
vector<int> cov;
long long curScore = 0;        // total demand of the covered union
long long curCost  = 0;

// add/remove a site, maintaining cov[], curScore, curCost in O(footprint).
inline void addSite(int i) {
    const Site& s = site[i];
    for (int y = s.y0; y <= s.y1; ++y) {
        int base = y * G;
        for (int x = s.x0; x <= s.x1; ++x) {
            int idx = base + x;
            if (cov[idx]++ == 0) curScore += demand[idx]; // newly covered
        }
    }
    curCost += s.c;
}
inline void removeSite(int i) {
    const Site& s = site[i];
    for (int y = s.y0; y <= s.y1; ++y) {
        int base = y * G;
        for (int x = s.x0; x <= s.x1; ++x) {
            int idx = base + x;
            if (--cov[idx] == 0) curScore -= demand[idx]; // last cover gone
        }
    }
    curCost -= s.c;
}

// marginal gain of adding site i to the CURRENT cov[] (cells with cov==0).
inline long long marginalGain(int i) {
    const Site& s = site[i];
    long long g = 0;
    for (int y = s.y0; y <= s.y1; ++y) {
        int base = y * G;
        for (int x = s.x0; x <= s.x1; ++x) {
            int idx = base + x;
            if (cov[idx] == 0) g += demand[idx];
        }
    }
    return g;
}

int main() {
    double t_start = now_sec();
    const double TIME_LIMIT = 1.8; // seconds (budget); local search uses the rest

    // ---- read instance ----
    {
        int m; long long b;
        if (scanf("%d %d %lld", &G, &m, &b) != 3) { printf("0\n"); return 0; }
        M = m; B = b;
        demand.assign((size_t)G * G, 0);
        for (size_t i = 0; i < (size_t)G * G; ++i) scanf("%d", &demand[i]);
        site.resize(M);
        for (int i = 0; i < M; ++i) {
            int sx, sy, r; long long c;
            scanf("%d %d %d %lld", &sx, &sy, &r, &c);
            Site s; s.sx = sx; s.sy = sy; s.r = r; s.c = c;
            s.x0 = max(0, sx - r); s.y0 = max(0, sy - r);
            s.x1 = min(G - 1, sx + r); s.y1 = min(G - 1, sy + r);
            site[i] = s;
        }
    }
    cov.assign((size_t)G * G, 0);

    // ---- CELF: cost-benefit lazy greedy ----
    // Priority queue keyed on an UPPER BOUND of gain-per-cost. We store the
    // gain computed when an entry was (re)evaluated together with the size of
    // the covered set at that time (`stamp`). Because coverage only grows as we
    // add antennas, a previously computed marginal gain is an upper bound on the
    // current one (submodularity), so a stale top entry can be re-evaluated
    // lazily and re-inserted; once the top entry's stamp is current it is the
    // true best and we commit it. This skips recomputing stale gains.
    struct Entry { double ratio; long long gain; int idx; long long stamp; };
    struct Cmp { bool operator()(const Entry& a, const Entry& b) const { return a.ratio < b.ratio; } };
    priority_queue<Entry, vector<Entry>, Cmp> pq;

    vector<char> inSet(M, 0);
    long long stampNow = 0; // increments whenever cov[] changes (an antenna is added)

    // initial evaluation of every affordable site
    for (int i = 0; i < M; ++i) {
        if (site[i].c > B) continue;            // can never fit alone -> skip
        long long g = marginalGain(i);
        double ratio = (g <= 0) ? -1.0 : (double)g / (double)site[i].c;
        pq.push({ratio, g, i, stampNow});
    }

    vector<int> greedyPick;
    while (!pq.empty()) {
        if (now_sec() - t_start > TIME_LIMIT * 0.55) break; // leave time for local search
        Entry e = pq.top(); pq.pop();
        int i = e.idx;
        if (inSet[i]) continue;
        if (site[i].c > B - curCost) continue;  // does not fit anymore
        if (e.stamp != stampNow) {
            // stale: re-evaluate against current coverage and re-insert
            long long g = marginalGain(i);
            if (g <= 0) continue;                // adds nothing new -> drop
            double ratio = (double)g / (double)site[i].c;
            pq.push({ratio, g, i, stampNow});
            continue;
        }
        // top entry is current and feasible -> commit it
        if (e.gain <= 0) break;                  // best remaining adds nothing
        addSite(i);
        inSet[i] = 1;
        greedyPick.push_back(i);
        ++stampNow;
    }

    // ---- single-best guard (Khuller-Moss-Naor) ----
    // The cost-benefit greedy alone can be arbitrarily bad on a single huge-gain
    // expensive element; comparing against the best single feasible antenna and
    // keeping the better of the two restores the (1-1/e)/2 guarantee.
    long long greedyScore = curScore;
    vector<int> greedySet = greedyPick;
    {
        long long bestSingle = -1; int bestI = -1;
        // current cov[] still reflects greedySet; compute single best from empty.
        for (int i = 0; i < M; ++i) {
            if (site[i].c > B) continue;
            // gain from empty == total demand of footprint (cov-independent upper part).
            const Site& s = site[i];
            long long g = 0;
            for (int y = s.y0; y <= s.y1; ++y) {
                int base = y * G;
                for (int x = s.x0; x <= s.x1; ++x) g += demand[base + x];
            }
            if (g > bestSingle) { bestSingle = g; bestI = i; }
        }
        if (bestSingle > greedyScore && bestI >= 0) {
            // rebuild cov[] for the single-best solution
            for (int i : greedySet) removeSite(i);
            for (int i = 0; i < M; ++i) inSet[i] = 0;
            greedySet.clear();
            addSite(bestI); inSet[bestI] = 1; greedySet.push_back(bestI);
        }
    }

    // ---- incremental-eval SWAP / add local search ----
    // cov[], curScore, curCost reflect the current chosen set `greedySet`.
    // Repeatedly try (a) add a feasible site with positive gain, or (b) remove
    // one site and add one or two cheaper ones that more than compensate. Each
    // move is evaluated and applied in O(footprint) via cov[]. Keep the best.
    vector<int> cur = greedySet;
    auto inCur = [&](int idx) { for (int v : cur) if (v == idx) return true; return false; };

    long long bestScore = curScore;
    vector<int> bestSet = cur;

    std::mt19937 rng(123456789u);
    while (now_sec() - t_start < TIME_LIMIT) {
        bool improved = false;

        // (a) greedy ADD pass: any positive-gain site that fits
        for (int i = 0; i < M && (now_sec() - t_start < TIME_LIMIT); ++i) {
            if (inSet[i]) continue;
            if (site[i].c > B - curCost) continue;
            if (marginalGain(i) > 0) {
                addSite(i); inSet[i] = 1; cur.push_back(i);
                improved = true;
            }
        }
        if (curScore > bestScore) { bestScore = curScore; bestSet = cur; }

        // (b) SWAP pass: try removing each chosen site, then greedily refill the
        // freed budget with best-ratio additions; accept if it helps.
        for (size_t pi = 0; pi < cur.size() && (now_sec() - t_start < TIME_LIMIT); ++pi) {
            int out = cur[pi];
            // snapshot to allow rollback
            long long beforeScore = curScore;
            removeSite(out); inSet[out] = 0;

            // greedily add best-ratio feasible sites (a few passes)
            vector<int> added;
            for (int rep = 0; rep < 6; ++rep) {
                int bi = -1; double bratio = 0.0; long long bgain = 0;
                for (int i = 0; i < M; ++i) {
                    if (inSet[i] || i == out) continue;
                    if (site[i].c > B - curCost) continue;
                    long long g = marginalGain(i);
                    if (g <= 0) continue;
                    double ratio = (double)g / (double)site[i].c;
                    if (ratio > bratio) { bratio = ratio; bi = i; bgain = g; }
                }
                if (bi < 0) break;
                addSite(bi); inSet[bi] = 1; added.push_back(bi);
                (void)bgain;
            }

            if (curScore > beforeScore) {
                // accept the swap: rebuild `cur`
                cur.erase(cur.begin() + pi);
                for (int a : added) cur.push_back(a);
                improved = true;
                if (curScore > bestScore) { bestScore = curScore; bestSet = cur; }
                break; // cur mutated; restart the swap pass next loop
            } else {
                // rollback
                for (int a : added) { removeSite(a); inSet[a] = 0; }
                addSite(out); inSet[out] = 1;
            }
        }

        if (curScore > bestScore) { bestScore = curScore; bestSet = cur; }
        if (!improved) {
            // random perturbation: drop a random chosen site to escape a plateau
            if (cur.empty()) break;
            int pi = rng() % cur.size();
            int out = cur[pi];
            removeSite(out); inSet[out] = 0;
            cur.erase(cur.begin() + pi);
        }
    }

    // ---- emit the best set found (guaranteed feasible) ----
    // validate cost once more defensively; if somehow over budget, drop sites.
    {
        long long cost = 0;
        for (int i : bestSet) cost += site[i].c;
        while (cost > B && !bestSet.empty()) { cost -= site[bestSet.back()].c; bestSet.pop_back(); }
    }
    printf("%d", (int)bestSet.size());
    for (int i : bestSet) printf(" %d", i);
    printf("\n");
    return 0;
}
