// TIER: strong
// Insight: the schedule AUTHORS the reuse-distance histogram. A globally-hot page
// stays resident iff its reuse distance stays below C. So instead of only chasing
// immediate locality (clustering), we WEAVE: run clustering but, whenever a hot page
// is resident and about to fall off the LRU tail, interrupt to schedule a ready task
// that refreshes it -- keeping the hot page's reuse distance < C. We try several
// weave aggressivenesses plus plain clustering and identity, and emit the order that
// actually minimizes simulated misses (the model is deterministic, so we can score
// our own candidates).
#include <bits/stdc++.h>
using namespace std;

int N, M, C, E;
vector<vector<int>> pg;
vector<vector<int>> succ;
vector<int> indeg0;
vector<int> gcount;   // global #tasks touching each page

struct LRU {
    int M, C, head, tail, sz, clk;
    vector<char> inC;
    vector<int> prv, nx, lastUse;
    void init(int M_, int C_) {
        M = M_; C = C_; inC.assign(M, 0); prv.assign(M, -1); nx.assign(M, -1);
        lastUse.assign(M, -1); head = tail = -1; sz = 0; clk = 0;
    }
    void unlink(int p) {
        int a = prv[p], b = nx[p];
        if (a != -1) nx[a] = b; else head = b;
        if (b != -1) prv[b] = a; else tail = a;
        prv[p] = nx[p] = -1;
    }
    void pushFront(int p) {
        prv[p] = -1; nx[p] = head;
        if (head != -1) prv[head] = p; else tail = p;
        head = p;
    }
    void access(int p) {
        clk++;
        if (inC[p]) { unlink(p); pushFront(p); lastUse[p] = clk; return; }
        if (sz == C) { int ev = tail; unlink(ev); inC[ev] = 0; sz--; }
        inC[p] = 1; pushFront(p); sz++; lastUse[p] = clk;
    }
    int age(int p) { return clk - lastUse[p]; }
};

long long simulate(const vector<int>& order) {
    LRU l; l.init(M, C); long long m = 0;
    for (int t : order) for (int p : pg[t]) { if (!l.inC[p]) m++; l.access(p); }
    return m;
}

vector<int> greedySched() {
    vector<int> indeg = indeg0; LRU l; l.init(M, C);
    vector<int> ready;
    for (int i = 1; i <= N; i++) if (indeg[i] == 0) ready.push_back(i);
    vector<int> order; order.reserve(N);
    while ((int)order.size() < N) {
        int bi = -1, bMiss = INT_MAX, bPg = INT_MAX, bId = INT_MAX;
        for (int k = 0; k < (int)ready.size(); k++) {
            int id = ready[k], miss = 0;
            for (int p : pg[id]) if (!l.inC[p]) miss++;
            int pgs = (int)pg[id].size();
            if (miss < bMiss || (miss == bMiss && (pgs < bPg || (pgs == bPg && id < bId)))) {
                bMiss = miss; bPg = pgs; bId = id; bi = k;
            }
        }
        int id = ready[bi];
        for (int p : pg[id]) l.access(p);
        order.push_back(id);
        ready[bi] = ready.back(); ready.pop_back();
        for (int v : succ[id]) if (--indeg[v] == 0) ready.push_back(v);
    }
    return order;
}

// Histogram-shaping: pack the hot-page touches together (schedule ready hot-touching
// tasks first) so the few hot pages stay resident, then run the cluster bodies. Among
// equally-hot choices, keep signature locality (fewest immediate misses).
vector<int> hotFirstSched(int hotMin) {
    vector<int> indeg = indeg0; LRU l; l.init(M, C);
    vector<char> isHot(M, 0);
    for (int p = 0; p < M; p++) if (gcount[p] >= hotMin) isHot[p] = 1;
    vector<int> ready;
    for (int i = 1; i <= N; i++) if (indeg[i] == 0) ready.push_back(i);
    vector<int> order; order.reserve(N);
    while ((int)order.size() < N) {
        int bi = -1, bKey = INT_MAX, bMiss = INT_MAX, bPg = INT_MAX, bId = INT_MAX;
        for (int k = 0; k < (int)ready.size(); k++) {
            int id = ready[k], miss = 0, hot = 0;
            for (int p : pg[id]) { if (!l.inC[p]) miss++; if (isHot[p]) hot++; }
            int key = hot > 0 ? 0 : 1;             // prefer hot-touching tasks
            int pgs = (int)pg[id].size();
            if (key < bKey || (key == bKey && (miss < bMiss ||
                (miss == bMiss && (pgs < bPg || (pgs == bPg && id < bId)))))) {
                bKey = key; bMiss = miss; bPg = pgs; bId = id; bi = k;
            }
        }
        int id = ready[bi];
        for (int p : pg[id]) l.access(p);
        order.push_back(id);
        ready[bi] = ready.back(); ready.pop_back();
        for (int v : succ[id]) if (--indeg[v] == 0) ready.push_back(v);
    }
    return order;
}

vector<int> weaveSched(int ageThresh, int hotMin) {
    vector<int> indeg = indeg0; LRU l; l.init(M, C);
    vector<int> remaining = gcount;
    vector<char> isHot(M, 0), endanger(M, 0);
    vector<int> hotlist;
    for (int p = 0; p < M; p++) if (gcount[p] >= hotMin) { isHot[p] = 1; hotlist.push_back(p); }
    vector<int> ready;
    for (int i = 1; i <= N; i++) if (indeg[i] == 0) ready.push_back(i);
    vector<int> order; order.reserve(N);
    vector<int> endList;
    while ((int)order.size() < N) {
        endList.clear();
        for (int p : hotlist)
            if (l.inC[p] && remaining[p] >= 1 && l.age(p) >= ageThresh) { endanger[p] = 1; endList.push_back(p); }
        int bi = -1;
        if (!endList.empty()) {                 // refresh: prefer a ready task touching an endangered hot page
            int bMiss = INT_MAX, bId = INT_MAX;
            for (int k = 0; k < (int)ready.size(); k++) {
                int id = ready[k], miss = 0; bool tch = false;
                for (int p : pg[id]) { if (!l.inC[p]) miss++; if (endanger[p]) tch = true; }
                if (tch && (miss < bMiss || (miss == bMiss && id < bId))) { bMiss = miss; bId = id; bi = k; }
            }
        }
        if (bi == -1) {                          // fall back to clustering
            int bMiss = INT_MAX, bPg = INT_MAX, bId = INT_MAX;
            for (int k = 0; k < (int)ready.size(); k++) {
                int id = ready[k], miss = 0;
                for (int p : pg[id]) if (!l.inC[p]) miss++;
                int pgs = (int)pg[id].size();
                if (miss < bMiss || (miss == bMiss && (pgs < bPg || (pgs == bPg && id < bId)))) {
                    bMiss = miss; bPg = pgs; bId = id; bi = k;
                }
            }
        }
        for (int p : endList) endanger[p] = 0;
        int id = ready[bi];
        for (int p : pg[id]) { l.access(p); remaining[p]--; }
        order.push_back(id);
        ready[bi] = ready.back(); ready.pop_back();
        for (int v : succ[id]) if (--indeg[v] == 0) ready.push_back(v);
    }
    return order;
}

int main() {
    if (scanf("%d %d %d %d", &N, &M, &C, &E) != 4) return 0;
    pg.assign(N + 1, {});
    gcount.assign(M, 0);
    for (int i = 1; i <= N; i++) {
        int k; scanf("%d", &k);
        pg[i].resize(k);
        for (int j = 0; j < k; j++) { scanf("%d", &pg[i][j]); }
        sort(pg[i].begin(), pg[i].end());
        for (int p : pg[i]) gcount[p]++;
    }
    succ.assign(N + 1, {});
    indeg0.assign(N + 1, 0);
    for (int i = 0; i < E; i++) {
        int u, v; scanf("%d %d", &u, &v);
        succ[u].push_back(v); indeg0[v]++;
    }

    vector<vector<int>> cands;
    { vector<int> id(N); for (int i = 0; i < N; i++) id[i] = i + 1; cands.push_back(id); } // identity
    cands.push_back(greedySched());
    int thr[] = { max(1, C - 1), max(1, C - 3), max(1, C - 6), max(1, C / 2), C + 2 };
    int hm[] = { max(2, C), max(2, C / 2), 3 };
    for (int b : hm) cands.push_back(hotFirstSched(b));
    for (int a : thr) for (int b : hm) cands.push_back(weaveSched(a, b));

    long long best = LLONG_MAX; int bidx = 0;
    for (int i = 0; i < (int)cands.size(); i++) {
        long long m = simulate(cands[i]);
        if (m < best) { best = m; bidx = i; }
    }
    const vector<int>& o = cands[bidx];
    for (int i = 0; i < N; i++) printf("%d%c", o[i], i == N - 1 ? '\n' : ' ');
    return 0;
}
