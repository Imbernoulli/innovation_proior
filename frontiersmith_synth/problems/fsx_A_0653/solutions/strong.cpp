// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: the checker's dispatcher is a FIXED, deterministic algorithm --
// replaying it once (with no ropes at all) directly reveals every maximal
// "stall run": a stretch [runStart, landing) on one lane, inside one convoy,
// where that lane's pointer is strictly behind and must walk one marker at a
// time until its value catches up to the (constant, for that run) pivot
// value. That landing position is a property of the ACTUAL paired lanes'
// values (found once the run closes, or by binary search on the lane's own
// sorted markers for the largest pivot ever observed at that exact start --
// landing-point-alignment), not of index proportion. Runs starting at the
// same (lane, position) across many different convoys are aggregated
// (cross-list-query-coupling: a start shared by many convoys is worth more),
// and a single global ranking then spends the shared rope budget on the
// highest-payoff run starts across ALL lanes at once (skip-budget-allocation).

int M, Q, B;
vector<vector<ll>> lane;
vector<vector<int>> qch;

int main() {
    if (!(cin >> M >> Q >> B)) return 0;
    lane.assign(M + 1, {});
    for (int c = 1; c <= M; c++) {
        int n; cin >> n;
        lane[c].resize(n);
        for (int j = 0; j < n; j++) cin >> lane[c][j];
    }
    qch.assign(Q, {});
    for (int q = 0; q < Q; q++) {
        int k; cin >> k;
        qch[q].resize(k);
        for (int i = 0; i < k; i++) cin >> qch[q][i];
    }

    // aggregated per (lane, run-start position): summed estimated savings,
    // and the LARGEST pivot value any occurrence needed to reach (safe to
    // aim for: an occurrence with a smaller pivot simply won't trigger the
    // rope and falls back to a plain step -- never worse than no rope).
    vector<vector<ll>> savings(M + 1), maxV(M + 1);
    for (int c = 1; c <= M; c++) {
        savings[c].assign(lane[c].size(), 0);
        maxV[c].assign(lane[c].size(), -1);
    }

    auto closeRun = [&](int c, ll runStart, ll runV, ll landing) {
        if (landing > runStart) {
            savings[c][runStart] += (landing - runStart - 1);
            if (runV > maxV[c][runStart]) maxV[c][runStart] = runV;
        }
    };

    for (auto &ch : qch) {
        int k = (int)ch.size();
        vector<ll> p(k, 0), n(k);
        for (int i = 0; i < k; i++) n[i] = (ll)lane[ch[i]].size();
        vector<char> wasBehind(k, 0);
        vector<ll> runStart(k, 0), runV(k, 0);
        ll roundCap = 1;
        for (int i = 0; i < k; i++) roundCap += n[i];
        ll rounds = 0;
        while (true) {
            bool exhausted = false;
            for (int i = 0; i < k; i++) if (p[i] >= n[i]) { exhausted = true; break; }
            if (exhausted) {
                for (int i = 0; i < k; i++)
                    if (wasBehind[i]) closeRun(ch[i], runStart[i], runV[i], n[i] - 1);
                break;
            }
            if (++rounds > roundCap + 5) break; // defensive; cannot actually happen
            ll v = -1;
            for (int i = 0; i < k; i++) v = max(v, lane[ch[i]][p[i]]);
            for (int i = 0; i < k; i++) {
                if (lane[ch[i]][p[i]] == v && wasBehind[i]) {
                    closeRun(ch[i], runStart[i], runV[i], p[i]);
                    wasBehind[i] = 0;
                }
            }
            bool allEq = true;
            for (int i = 0; i < k; i++) if (lane[ch[i]][p[i]] != v) { allEq = false; break; }
            if (allEq) {
                for (int i = 0; i < k; i++) p[i]++;
                continue;
            }
            for (int i = 0; i < k; i++) {
                if (lane[ch[i]][p[i]] == v) continue;
                if (!wasBehind[i]) { wasBehind[i] = 1; runStart[i] = p[i]; runV[i] = v; }
                p[i] += 1; // baseline replay: no ropes exist yet
            }
        }
    }

    // build candidates: for every run-start, the landing index for the
    // largest observed pivot value, via binary search on the lane.
    struct Cand { ll gain; int lane; ll pos, target; };
    vector<Cand> cand;
    for (int c = 1; c <= M; c++) {
        int len = (int)lane[c].size();
        for (int pos = 0; pos < len - 1; pos++) {
            if (savings[c][pos] <= 0 || maxV[c][pos] < 0) continue;
            ll v = maxV[c][pos];
            auto it = lower_bound(lane[c].begin() + pos + 1, lane[c].end(), v);
            ll target;
            if (it == lane[c].end()) target = len - 1;
            else target = (ll)(it - lane[c].begin());
            if (target <= pos) target = min((ll)len - 1, (ll)pos + 1);
            if (target <= pos) continue;
            cand.push_back({savings[c][pos], c, (ll)pos, target});
        }
    }
    sort(cand.begin(), cand.end(), [](const Cand &a, const Cand &b) { return a.gain > b.gain; });

    vector<vector<pair<ll,ll>>> chosen(M + 1);
    int spent = 0;
    for (auto &cd : cand) {
        if (spent >= B) break;
        chosen[cd.lane].push_back({cd.pos, cd.target});
        spent++;
    }
    for (int c = 1; c <= M; c++) {
        cout << chosen[c].size() << "\n";
        for (auto &pr : chosen[c]) cout << pr.first << " " << pr.second << "\n";
    }
    return 0;
}
