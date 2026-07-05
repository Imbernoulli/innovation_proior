// TIER: greedy
// Greedy weighted set cover: repeatedly install the candidate maximizing
// (newly-recorded artifacts / cost) until every artifact is recorded.
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

    // artifact ids + spatial bucket by cell
    vector<int> artId(N + 1, -1);
    vector<int> arts;
    unordered_map<ll, vector<int>> cell; // encoded cell -> artifact-ids
    cell.reserve(N * 2);
    auto key = [&](int rr, int cc){ return (ll)rr * (ll)(W + 1) + cc; };
    for (int i = 1; i <= N; i++) {
        if (art[i] == 1) {
            int id = arts.size();
            artId[i] = id;
            arts.push_back(i);
            cell[key(row[i], col[i])].push_back(id);
        }
    }
    int A = arts.size();

    // coverage list: for each candidate, the artifact-ids it records
    vector<vector<int>> cover(N + 1);
    for (int s = 1; s <= N; s++) {
        int r0 = row[s], c0 = col[s];
        for (int dr = -r; dr <= r; dr++) {
            int rr = r0 + dr;
            if (rr < 0 || rr > maxc) continue;
            for (int dc = -r; dc <= r; dc++) {
                int cc = c0 + dc;
                if (cc < 0 || cc > maxc) continue;
                auto it = cell.find(key(rr, cc));
                if (it == cell.end()) continue;
                for (int id : it->second) cover[s].push_back(id);
            }
        }
    }

    vector<char> done(A, 0);
    int remaining = A;
    vector<int> chosen;

    // lazy max-heap keyed by (newly-covered / cost)
    priority_queue<pair<double,int>> pq;
    auto gain = [&](int s)->int {
        int g = 0;
        for (int id : cover[s]) if (!done[id]) g++;
        return g;
    };
    for (int s = 1; s <= N; s++) {
        if (cover[s].empty()) continue;
        double ratio = (double)cover[s].size() / (double)cost[s];
        pq.push({ratio, s});
    }

    while (remaining > 0 && !pq.empty()) {
        auto top = pq.top(); pq.pop();
        int s = top.second;
        int g = gain(s);
        if (g == 0) continue;
        double cur = (double)g / (double)cost[s];
        if (cur < top.first - 1e-12) { pq.push({cur, s}); continue; }
        // commit
        chosen.push_back(s);
        for (int id : cover[s]) if (!done[id]) { done[id] = 1; remaining--; }
    }

    // safety: cover any leftover artifact by itself (should not trigger)
    for (int id = 0; id < A; id++) if (!done[id]) { chosen.push_back(arts[id]); done[id] = 1; }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    return 0;
}
