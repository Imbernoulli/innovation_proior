// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The obvious first approach: the standard batching rule is "cluster jobs of
// similar temperature together" (minimizes wasted temperature spread inside a
// batch), then process the clusters in ascending temperature order so the
// retained-heat carryover is exploited between consecutive batches. This
// captures the directionality insight but is completely DEADLINE-BLIND: due
// dates never influence which batch a job lands in or when it runs, so urgent
// jobs whose required temperature sits far from the ascending sweep's current
// position get pushed to whatever time the pure temperature order gives them.

struct Job { ll T, s, d, w; int idx; };

int main() {
    int n; ll V, LAMBDA;
    cin >> n >> V >> LAMBDA;
    ll decay, r0; cin >> decay >> r0;
    ll cheat, ccool, theat, tcool, base, ppu;
    cin >> cheat >> ccool >> theat >> tcool >> base >> ppu;
    vector<Job> jobs(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> jobs[i].T >> jobs[i].s >> jobs[i].d >> jobs[i].w;
        jobs[i].idx = i;
    }

    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (jobs[a].T != jobs[b].T) return jobs[a].T < jobs[b].T;
        return a < b;
    });

    vector<vector<int>> batches;
    vector<int> cur;
    ll curSize = 0;
    for (int id : order) {
        if (curSize + jobs[id].s > V && !cur.empty()) {
            batches.push_back(cur);
            cur.clear();
            curSize = 0;
        }
        cur.push_back(id);
        curSize += jobs[id].s;
    }
    if (!cur.empty()) batches.push_back(cur);

    // batches were formed from a temperature-sorted list, so they are already
    // (weakly) ascending in max-temperature; process in that formation order.
    printf("%d\n", (int)batches.size());
    for (auto &b : batches) {
        printf("%d", (int)b.size());
        for (int id : b) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
