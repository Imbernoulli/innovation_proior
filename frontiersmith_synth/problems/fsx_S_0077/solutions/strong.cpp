// TIER: strong
// Multi-restart active list scheduling with gap-filling insertion. One deterministic
// min-completion pass plus many randomized job-priority restarts; keeps the schedule
// with the shortest makespan. Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> os_;
vector<vector<vector<pair<int,int>>>> opt; // opt[j][k] = list of (dam,dur)

// earliest feasible start >= ready such that [start,start+dur) fits a gap on 'iv'
// (iv is sorted, non-overlapping).
ll earliestFit(const vector<pair<ll,ll>>& iv, ll ready, ll dur) {
    ll cand = ready;
    for (auto& itv : iv) {
        if (itv.second <= cand) continue;          // gap already past
        if (cand + dur <= itv.first) return cand;  // fits before this interval
        cand = max(cand, itv.second);              // pushed past this interval
    }
    return cand;
}

// Run one schedule; rule=0 -> select available op with min achievable completion,
// rule=1 -> select by (earliest ready, random priority). Returns makespan and fills asg.
ll runSchedule(int rule, const vector<int>& prio,
               vector<vector<pair<int,ll>>>& asg) {
    vector<vector<pair<ll,ll>>> iv(m);
    vector<int> frontier(n, 0);
    vector<ll> jobReady(n, 0);
    asg.assign(n, {});
    for (int j = 0; j < n; j++) asg[j].assign(os_[j], {-1, 0});

    int remaining = 0;
    for (int j = 0; j < n; j++) remaining += os_[j];

    ll makespan = 0;
    while (remaining > 0) {
        int selJob = -1, selDam = -1; ll selStart = 0, selComp = LLONG_MAX;
        int selRank = INT_MAX; ll selReady = LLONG_MAX;
        for (int j = 0; j < n; j++) {
            int k = frontier[j];
            if (k >= os_[j]) continue;
            // best dam for this op (min completion via gap fill)
            int bDam = -1; ll bStart = 0, bComp = LLONG_MAX;
            for (auto& pr : opt[j][k]) {
                int dam = pr.first, dur = pr.second;
                ll st = earliestFit(iv[dam], jobReady[j], dur);
                ll comp = st + dur;
                if (comp < bComp) { bComp = comp; bStart = st; bDam = dam; }
            }
            bool take = false;
            if (rule == 0) {
                if (bComp < selComp || (bComp == selComp && prio[j] < selRank))
                    take = true;
            } else {
                if (jobReady[j] < selReady ||
                    (jobReady[j] == selReady && prio[j] < selRank))
                    take = true;
            }
            if (take) {
                selJob = j; selDam = bDam; selStart = bStart; selComp = bComp;
                selRank = prio[j]; selReady = jobReady[j];
            }
        }
        // commit
        int j = selJob, k = frontier[j];
        // find duration for chosen dam
        int dur = 0;
        for (auto& pr : opt[j][k]) if (pr.first == selDam) { dur = pr.second; break; }
        iv[selDam].push_back({selStart, selStart + dur});
        // keep dam intervals sorted
        auto& vv = iv[selDam];
        for (int i = (int)vv.size() - 1; i > 0 && vv[i].first < vv[i-1].first; --i)
            swap(vv[i], vv[i-1]);
        asg[j][k] = {selDam, selStart};
        jobReady[j] = selStart + dur;
        makespan = max(makespan, jobReady[j]);
        frontier[j]++;
        remaining--;
    }
    return makespan;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    os_.resize(n); opt.resize(n);
    int ops = 0;
    for (int j = 0; j < n; j++) {
        int o; scanf("%d", &o); os_[j] = o; opt[j].resize(o); ops += o;
        for (int k = 0; k < o; k++) {
            int c; scanf("%d", &c);
            for (int e = 0; e < c; e++) {
                int dam, dur; scanf("%d %d", &dam, &dur);
                opt[j][k].push_back({dam, dur});
            }
        }
    }

    mt19937 rng(9876543u);
    vector<int> prio(n);
    for (int j = 0; j < n; j++) prio[j] = j;

    vector<vector<pair<int,ll>>> best, cur;
    ll bestMk = LLONG_MAX;

    // deterministic min-completion pass
    ll mk = runSchedule(0, prio, cur);
    if (mk < bestMk) { bestMk = mk; best = cur; }

    int R = max(8, 4000 / max(1, ops));
    if (R > 80) R = 80;
    for (int r = 0; r < R; r++) {
        shuffle(prio.begin(), prio.end(), rng);
        int rule = (r % 3 == 0) ? 0 : 1;
        ll mk2 = runSchedule(rule, prio, cur);
        if (mk2 < bestMk) { bestMk = mk2; best = cur; }
    }

    for (int j = 0; j < n; j++) {
        string line;
        for (int k = 0; k < os_[j]; k++) {
            line += to_string(best[j][k].first) + " " + to_string(best[j][k].second);
            if (k + 1 < os_[j]) line += " ";
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
