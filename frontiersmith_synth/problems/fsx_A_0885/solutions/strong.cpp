// TIER: strong
// Insight: reformulate the carve as SAME-RECIPE-COMPONENT selection, not
// geometric partitioning. Every maximal same-recipe edge-connected component
// is already a 100%-pure territory candidate (isolating it captures its full
// spend outright, since its own recipe is trivially its own plurality) -- this
// includes both the town's big background regions AND any thin, scattered
// pocket of a minority recipe hiding inside one. Starting from "one territory
// per component" (globally optimal if it fits the budget), repeatedly merge
// the globally WEAKEST-value component into whichever adjacent territory
// maximizes the resulting merged score, until at most K territories remain.
// This keeps the highest-value pockets carved out and only sacrifices the
// components that were never going to swing a plurality anyway -- exactly the
// family's pack/crack insight, and something a geometry-only partitioner can
// never discover because it never looks at the recipe data.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef array<int,7> Cnt;
typedef array<ll,7> Spd;

int R, C, K;
vector<vector<int>> f;
vector<vector<ll>> s;
vector<vector<int>> label;
vector<int> parent_;
vector<Cnt> cnt;
vector<Spd> spd;
vector<vector<int>> members;
vector<vector<int>> rawAdj;

int find_(int x) { while (parent_[x] != x) { parent_[x] = parent_[parent_[x]]; x = parent_[x]; } return x; }

ll scoreOf(const Cnt& c, const Spd& d) {
    int best = 1;
    for (int r = 2; r <= 6; r++) if (c[r] > c[best]) best = r;
    return d[best];
}

int main() {
    scanf("%d %d %d", &R, &C, &K);
    f.assign(R, vector<int>(C));
    s.assign(R, vector<ll>(C));
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) scanf("%d", &f[r][c]);
    for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) scanf("%lld", &s[r][c]);

    // ---- same-recipe connected components ----
    label.assign(R, vector<int>(C, -1));
    static const int dr[4] = {1, -1, 0, 0};
    static const int dc[4] = {0, 0, 1, -1};
    int numComp = 0;
    for (int r0 = 0; r0 < R; r0++)
        for (int c0 = 0; c0 < C; c0++) {
            if (label[r0][c0] != -1) continue;
            int id = numComp++;
            int rec = f[r0][c0];
            long long size = 0; ll tot = 0;
            vector<pair<int,int>> st = {{r0, c0}};
            label[r0][c0] = id;
            while (!st.empty()) {
                auto [r, c] = st.back(); st.pop_back();
                size++; tot += s[r][c];
                for (int d = 0; d < 4; d++) {
                    int nr = r + dr[d], nc = c + dc[d];
                    if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                    if (label[nr][nc] != -1) continue;
                    if (f[nr][nc] != rec) continue;
                    label[nr][nc] = id;
                    st.push_back({nr, nc});
                }
            }
            cnt.push_back(Cnt{0,0,0,0,0,0,0});
            spd.push_back(Spd{0,0,0,0,0,0,0});
            cnt.back()[rec] = (int)size;
            spd.back()[rec] = tot;
        }

    parent_.resize(numComp);
    for (int i = 0; i < numComp; i++) parent_[i] = i;
    members.assign(numComp, {});
    for (int i = 0; i < numComp; i++) members[i] = {i};
    rawAdj.assign(numComp, {});
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            int u = label[r][c];
            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                int v = label[nr][nc];
                if (v != u) rawAdj[u].push_back(v);
            }
        }

    set<int> alive;
    for (int i = 0; i < numComp; i++) alive.insert(i);

    while ((int)alive.size() > K) {
        // find globally weakest alive component
        int weak = -1; ll weakScore = -1;
        for (int r : alive) {
            ll sc = scoreOf(cnt[r], spd[r]);
            if (weak == -1 || sc < weakScore) { weak = r; weakScore = sc; }
        }
        // gather distinct alive neighbor roots
        set<int> candSet;
        for (int m : members[weak])
            for (int nb : rawAdj[m]) {
                int root = find_(nb);
                if (root != weak) candSet.insert(root);
            }
        int best = -1; ll bestScore = -1;
        for (int cand : candSet) {
            Cnt mc = cnt[weak]; Spd md = spd[weak];
            for (int rec = 1; rec <= 6; rec++) { mc[rec] += cnt[cand][rec]; md[rec] += spd[cand][rec]; }
            ll sc = scoreOf(mc, md);
            if (best == -1 || sc > bestScore) { best = cand; bestScore = sc; }
        }
        if (best == -1) {
            // grid is connected, so this should not happen while alive.size()>1;
            // safety fallback: merge into any other alive root.
            for (int r : alive) if (r != weak) { best = r; break; }
        }
        // commit merge: weak -> best
        for (int rec = 1; rec <= 6; rec++) { cnt[best][rec] += cnt[weak][rec]; spd[best][rec] += spd[weak][rec]; }
        for (int m : members[weak]) parent_[m] = best;
        members[best].insert(members[best].end(), members[weak].begin(), members[weak].end());
        alive.erase(weak);
    }

    // compact output ids
    map<int,int> outId;
    int next = 1;
    for (int r : alive) outId[r] = next++;

    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            int root = find_(label[r][c]);
            printf("%d%c", outId[root], c + 1 == C ? '\n' : ' ');
        }
    }
    return 0;
}
