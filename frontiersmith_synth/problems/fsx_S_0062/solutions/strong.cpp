// TIER: strong
// Greedy weighted set cover, then two improvement phases:
//   (1) redundancy pruning: drop, cost-descending, any beacon whose recorded
//       artifacts are all still covered by the remaining beacons;
//   (2) replacement: for each artifact left uniquely covered by an expensive
//       beacon, try to swap in a cheaper candidate that still covers everything
//       that beacon uniquely provides.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, r;
    if (scanf("%d %d", &N, &r) != 2) return 0;
    vector<int> row(N + 1), col(N + 1), art(N + 1);
    vector<ll>  cost(N + 1);
    int maxc = 0;
    for (int i = 1; i <= N; i++) {
        scanf("%d %d %lld %d", &row[i], &col[i], &cost[i], &art[i]);
        maxc = max(maxc, max(row[i], col[i]));
    }
    int W = maxc + 1;

    vector<int> artId(N + 1, -1);
    vector<int> arts;
    unordered_map<ll, vector<int>> cell;
    cell.reserve(N * 2);
    auto key = [&](int rr, int cc){ return (ll)rr * (ll)(W + 1) + cc; };
    for (int i = 1; i <= N; i++) {
        if (art[i] == 1) {
            int id = arts.size();
            artId[i] = id; arts.push_back(i);
            cell[key(row[i], col[i])].push_back(id);
        }
    }
    int A = arts.size();

    vector<vector<int>> cover(N + 1);
    for (int s = 1; s <= N; s++) {
        int r0 = row[s], c0 = col[s];
        for (int dr = -r; dr <= r; dr++) {
            int rr = r0 + dr; if (rr < 0 || rr > maxc) continue;
            for (int dc = -r; dc <= r; dc++) {
                int cc = c0 + dc; if (cc < 0 || cc > maxc) continue;
                auto it = cell.find(key(rr, cc));
                if (it == cell.end()) continue;
                for (int id : it->second) cover[s].push_back(id);
            }
        }
    }

    // ---- phase 0: greedy ----
    vector<char> done(A, 0);
    int remaining = A;
    vector<int> chosen;
    auto gain = [&](int s)->int { int g = 0; for (int id : cover[s]) if (!done[id]) g++; return g; };
    priority_queue<pair<double,int>> pq;
    for (int s = 1; s <= N; s++) if (!cover[s].empty())
        pq.push({(double)cover[s].size() / (double)cost[s], s});
    while (remaining > 0 && !pq.empty()) {
        auto top = pq.top(); pq.pop();
        int s = top.second; int g = gain(s);
        if (g == 0) continue;
        double cur = (double)g / (double)cost[s];
        if (cur < top.first - 1e-12) { pq.push({cur, s}); continue; }
        chosen.push_back(s);
        for (int id : cover[s]) if (!done[id]) { done[id] = 1; remaining--; }
    }
    for (int id = 0; id < A; id++) if (!done[id]) { chosen.push_back(arts[id]); done[id] = 1; }

    // coverage multiplicity over the chosen set
    vector<int> cnt(A, 0);
    for (int s : chosen) for (int id : cover[s]) cnt[id]++;

    // ---- phase 1: redundancy pruning, cost-descending ----
    sort(chosen.begin(), chosen.end(), [&](int a, int b){ return cost[a] > cost[b]; });
    vector<char> keep(chosen.size(), 1);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t k = 0; k < chosen.size(); k++) {
            if (!keep[k]) continue;
            int s = chosen[k];
            bool redundant = true;
            for (int id : cover[s]) if (cnt[id] <= 1) { redundant = false; break; }
            if (redundant) {
                keep[k] = 0; changed = true;
                for (int id : cover[s]) cnt[id]--;
            }
        }
    }
    vector<int> cur;
    for (size_t k = 0; k < chosen.size(); k++) if (keep[k]) cur.push_back(chosen[k]);

    // ---- phase 2: replacement of expensive beacons by cheaper coverers ----
    // Build, per artifact, the set of candidate sites covering it (for lookups).
    vector<vector<int>> coveredBy(A);
    for (int s = 1; s <= N; s++) for (int id : cover[s]) coveredBy[id].push_back(s);

    vector<int> curCnt(A, 0);
    vector<char> inSet(N + 1, 0);
    for (int s : cur) { inSet[s] = 1; for (int id : cover[s]) curCnt[id]++; }

    // try, cost-descending, to replace each beacon by the single cheapest
    // candidate that covers all artifacts it uniquely provides.
    sort(cur.begin(), cur.end(), [&](int a, int b){ return cost[a] > cost[b]; });
    for (size_t k = 0; k < cur.size(); k++) {
        int s = cur[k];
        if (!inSet[s]) continue;
        // artifacts uniquely covered by s
        vector<int> uniq;
        for (int id : cover[s]) if (curCnt[id] == 1) uniq.push_back(id);
        if (uniq.empty()) continue; // pruning already handles fully redundant
        // find cheapest candidate (other than s) covering all of uniq
        int best = -1; ll bestCost = cost[s];
        // candidates that cover uniq[0] are the only viable ones
        for (int cand : coveredBy[uniq[0]]) {
            if (cand == s) continue;
            if (cost[cand] >= bestCost) continue;
            // check it covers all of uniq
            // (cover[cand] as a set)
            bool okAll = true;
            // build a hash set of cover[cand] lazily only when promising
            static vector<char> mark; // reused
            // fallback: linear membership test using per-call small set
            // Use unordered check:
            // (uniq is small on average)
            // Build set of cand coverage
            // To keep it simple and correct, use a boolean over artifacts:
            // but A can be large; use a local hash set instead.
            {
                unordered_set<int> cs(cover[cand].begin(), cover[cand].end());
                for (int id : uniq) if (!cs.count(id)) { okAll = false; break; }
            }
            if (okAll) { best = cand; bestCost = cost[cand]; }
        }
        if (best != -1) {
            // swap s -> best
            inSet[s] = 0; for (int id : cover[s]) curCnt[id]--;
            if (!inSet[best]) { inSet[best] = 1; for (int id : cover[best]) curCnt[id]++; }
        }
    }

    vector<int> out;
    for (int s = 1; s <= N; s++) if (inSet[s]) out.push_back(s);
    if (out.empty()) out = cur; // safety

    printf("%d\n", (int)out.size());
    for (size_t i = 0; i < out.size(); i++)
        printf("%d%c", out[i], i + 1 == out.size() ? '\n' : ' ');
    return 0;
}
