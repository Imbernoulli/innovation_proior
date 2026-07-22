// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: carryover makes batch cost DIRECTIONAL, so the design object is a set
// of ascending chains braided against the deadline structure, not a similarity
// clustering. Build the batch SKELETON in due-date order first (so urgent jobs
// land early regardless of temperature), then run a deterministic local search
// that (a) swaps the PROCESSING ORDER of adjacent batches and (b) moves/exchanges
// jobs between adjacent batches -- both moves are only kept when they strictly
// reduce the true objective. This recovers most of the ascending-chain carryover
// savings the deadline-blind greedy gets, while keeping urgent jobs on time by
// letting a batch break the ascending run early (a chain reset) exactly where a
// deadline forces it.

ll DECAY, R0, CHEAT, CCOOL, THEAT, TCOOL, BASE, PPU, LAMBDA;

struct Job { ll T, s, d, w; };
vector<Job> J;

ll evalSchedule(const vector<vector<int>> &batches) {
    ll F = 0, prevTemp = R0, Cprev = 0;
    for (auto &b : batches) {
        ll temp = 0, size = 0;
        for (int idx : b) { temp = max(temp, J[idx].T); size += J[idx].s; }
        ll R = max((ll)0, prevTemp - DECAY);
        ll htime, cost;
        if (temp >= R) { htime = THEAT * (temp - R); cost = CHEAT * (temp - R); }
        else           { htime = TCOOL * (R - temp); cost = CCOOL * (R - temp); }
        F += cost;
        ll dur = BASE + PPU * size + htime;
        ll Ck = Cprev + dur;
        for (int idx : b) {
            ll tardi = Ck - J[idx].d;
            if (tardi > 0) F += LAMBDA * J[idx].w * tardi;
        }
        Cprev = Ck;
        prevTemp = temp;
    }
    return F;
}

ll batchSize(const vector<int> &b) {
    ll s = 0;
    for (int idx : b) s += J[idx].s;
    return s;
}

int main() {
    int n; ll V;
    cin >> n >> V >> LAMBDA;
    cin >> DECAY >> R0;
    cin >> CHEAT >> CCOOL >> THEAT >> TCOOL >> BASE >> PPU;
    J.assign(n + 1, Job());
    for (int i = 1; i <= n; i++) cin >> J[i].T >> J[i].s >> J[i].d >> J[i].w;

    // ---- skeleton: due-date order, first-fit packing (deadline-aware order) ----
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (J[a].d != J[b].d) return J[a].d < J[b].d;
        if (J[a].T != J[b].T) return J[a].T < J[b].T;
        return a < b;
    });

    // A tight (binding-looking) due date must not get diluted by sharing a
    // batch with unrelated filler: only bulk jobs (all tied at the loose
    // ceiling) may be freely capacity-packed together; anything with a due
    // date below that ceiling starts its own batch whenever the due date
    // value changes, so an urgent job's own completion time isn't pushed out
    // by riders that don't need to be there.
    ll looseCeil = 0;
    for (int i = 1; i <= n; i++) looseCeil = max(looseCeil, J[i].d);
    vector<vector<int>> batches;
    vector<int> cur;
    ll curSize = 0;
    ll curDue = -1;
    bool curTight = false;
    for (int id : order) {
        bool tight = J[id].d < looseCeil;
        bool mustSplit = !cur.empty() && ((curSize + J[id].s > V) ||
                          (tight != curTight) ||
                          (tight && curTight && J[id].d != curDue));
        if (mustSplit) {
            batches.push_back(cur);
            cur.clear();
            curSize = 0;
        }
        cur.push_back(id);
        curSize += J[id].s;
        curDue = J[id].d;
        curTight = tight;
    }
    if (!cur.empty()) batches.push_back(cur);

    // ---- deterministic local search: adjacent batch-order swaps + job moves ----
    int m = (int)batches.size();
    ll curF = evalSchedule(batches);
    const int PASSES = 6;
    for (int pass = 0; pass < PASSES; pass++) {
        bool improved = false;

        // (a) swap the whole processing order of adjacent batches k, k+1
        for (int k = 0; k + 1 < m; k++) {
            swap(batches[k], batches[k + 1]);
            ll nf = evalSchedule(batches);
            if (nf < curF) { curF = nf; improved = true; }
            else swap(batches[k], batches[k + 1]);   // revert
        }

        // (b) move a single job from batch k to adjacent batch k+1 (or back),
        //     trying every job in either side, keeping capacity feasible.
        for (int k = 0; k + 1 < m; k++) {
            bool moved = true;
            while (moved) {
                moved = false;
                for (int t = 0; t < (int)batches[k].size(); t++) {
                    int id = batches[k][t];
                    if (batchSize(batches[k + 1]) + J[id].s > V) continue;
                    vector<int> savedA = batches[k], savedB = batches[k + 1];
                    batches[k].erase(batches[k].begin() + t);
                    batches[k + 1].push_back(id);
                    if (batches[k].empty()) { batches[k] = savedA; batches[k + 1] = savedB; continue; }
                    ll nf = evalSchedule(batches);
                    if (nf < curF) { curF = nf; moved = true; improved = true; break; }
                    batches[k] = savedA; batches[k + 1] = savedB;
                }
                if (moved) continue;
                for (int t = 0; t < (int)batches[k + 1].size(); t++) {
                    int id = batches[k + 1][t];
                    if (batchSize(batches[k]) + J[id].s > V) continue;
                    vector<int> savedA = batches[k], savedB = batches[k + 1];
                    batches[k + 1].erase(batches[k + 1].begin() + t);
                    batches[k].push_back(id);
                    if (batches[k + 1].empty()) { batches[k] = savedA; batches[k + 1] = savedB; continue; }
                    ll nf = evalSchedule(batches);
                    if (nf < curF) { curF = nf; moved = true; improved = true; break; }
                    batches[k] = savedA; batches[k + 1] = savedB;
                }
            }
        }

        // drop any batch that emptied out (shouldn't normally happen, defensive)
        vector<vector<int>> compact;
        for (auto &b : batches) if (!b.empty()) compact.push_back(b);
        batches.swap(compact);
        m = (int)batches.size();

        if (!improved) break;
    }

    printf("%d\n", (int)batches.size());
    for (auto &b : batches) {
        printf("%d", (int)b.size());
        for (int id : b) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
