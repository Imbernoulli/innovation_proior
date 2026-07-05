// TIER: greedy
// Value-greedy: consider tasks in order of penalty v_j descending.  For each task grab the
// p_j cheapest still-free steps inside its window (solar preferred over diesel via step cost)
// and commit it only if the power cost is strictly less than the penalty it saves.  Big tasks
// grab the scarce solar steps first, which is often wasteful -- a decent but improvable pass.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, N;
    if (scanf("%d %d", &T, &N) != 2) return 0;
    vector<int> a(T + 1); vector<long long> cs(T + 1), cd(T + 1);
    for (int t = 1; t <= T; t++) scanf("%d %lld %lld", &a[t], &cs[t], &cd[t]);
    vector<int> r(N + 1), d(N + 1), p(N + 1); vector<long long> v(N + 1);
    for (int j = 1; j <= N; j++) scanf("%d %d %d %lld", &r[j], &d[j], &p[j], &v[j]);

    auto stepCost = [&](int t) -> long long { return a[t] ? cs[t] : cd[t]; };

    vector<int> job(T + 1, 0), mode(T + 1, 0);
    vector<char> occ(T + 1, 0);

    vector<int> ord(N);
    for (int j = 0; j < N; j++) ord[j] = j + 1;
    sort(ord.begin(), ord.end(), [&](int x, int y){ return v[x] > v[y]; });

    vector<pair<long long,int>> cand;    // (cost, step)
    for (int j : ord) {
        cand.clear();
        for (int t = r[j]; t <= d[j]; t++)
            if (!occ[t]) cand.push_back({stepCost(t), t});
        if ((int)cand.size() < p[j]) continue;
        nth_element(cand.begin(), cand.begin() + p[j], cand.end());
        long long cost = 0;
        for (int k = 0; k < p[j]; k++) cost += cand[k].first;
        if (cost >= v[j]) continue;      // not worth clearing
        for (int k = 0; k < p[j]; k++) {
            int t = cand[k].second;
            occ[t] = 1; job[t] = j; mode[t] = a[t] ? 0 : 1;
        }
    }

    string out; out.reserve((size_t)T * 4);
    char buf[32];
    for (int t = 1; t <= T; t++) {
        int len = sprintf(buf, "%d %d\n", job[t], mode[t]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
