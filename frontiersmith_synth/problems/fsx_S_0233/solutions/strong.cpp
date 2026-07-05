// TIER: strong
// Multi-restart randomized list scheduling with GAP INSERTION. Each dispatched step is
// placed in the EARLIEST feasible idle gap of its resource (>= the step's ready time),
// not merely after the resource's last op. Dispatch priority is ECT with randomized
// tie-breaking / occasional random picks; many seeds are tried and the least-makespan
// schedule is kept. Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;

static int n, m, total;
static vector<long long> r;
static vector<int> o;
static vector<vector<int>> mac;
static vector<vector<long long>> dur;

// earliest start >= est on resource c for length d, gap-aware
static long long earliestStart(const vector<pair<long long,long long>>& busy,
                               long long est, long long d) {
    long long t = est;
    for (size_t i = 0; i < busy.size(); i++) {
        long long bs = busy[i].first, be = busy[i].second;
        if (t + d <= bs) return t;        // fits before this interval
        if (t < be) t = be;               // pushed past it
    }
    return t;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    r.resize(n); o.resize(n); mac.resize(n); dur.resize(n);
    total = 0;
    for (int j = 0; j < n; j++) {
        scanf("%lld %d", &r[j], &o[j]);
        mac[j].resize(o[j]); dur[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) scanf("%d %lld", &mac[j][k], &dur[j][k]);
        total += o[j];
    }

    mt19937 rng(987654321u);
    long long bestMake = LLONG_MAX;
    vector<vector<long long>> bestSt(n);
    for (int j = 0; j < n; j++) bestSt[j].resize(o[j]);

    const int RESTARTS = 240;
    for (int rs = 0; rs < RESTARTS; rs++) {
        vector<vector<pair<long long,long long>>> busy(m); // sorted by start
        vector<long long> jobReady(n);
        vector<int> nxt(n, 0);
        for (int j = 0; j < n; j++) jobReady[j] = r[j];
        vector<vector<long long>> st(n);
        for (int j = 0; j < n; j++) st[j].resize(o[j]);
        long long make = 0;

        // randomization strength grows across restarts
        double randPick = (rs == 0) ? 0.0 : 0.10 + 0.30 * ((double)(rs % 8) / 8.0);

        for (int done = 0; done < total; done++) {
            // gather ready telescopes and their earliest completion
            int chosen = -1;
            long long chStart = 0, bestComp = LLONG_MAX;
            vector<int> ready;
            for (int j = 0; j < n; j++) if (nxt[j] < o[j]) ready.push_back(j);

            if ((double)(rng() % 1000) / 1000.0 < randPick) {
                // random ready telescope
                int j = ready[rng() % ready.size()];
                int k = nxt[j];
                long long est = jobReady[j];
                chStart = earliestStart(busy[mac[j][k]], est, dur[j][k]);
                chosen = j;
            } else {
                for (int j : ready) {
                    int k = nxt[j];
                    long long est = jobReady[j];
                    long long start = earliestStart(busy[mac[j][k]], est, dur[j][k]);
                    long long comp = start + dur[j][k];
                    // random tie-break
                    if (comp < bestComp || (comp == bestComp && (rng() & 1))) {
                        bestComp = comp; chStart = start; chosen = j;
                    }
                }
            }

            int j = chosen, k = nxt[j];
            long long start = chStart, d = dur[j][k];
            st[j][k] = start;
            // insert into busy list keeping sorted
            auto& bl = busy[mac[j][k]];
            bl.insert(lower_bound(bl.begin(), bl.end(), make_pair(start, start + d)),
                      make_pair(start, start + d));
            jobReady[j] = start + d;
            nxt[j]++;
            make = max(make, start + d);
        }

        if (make < bestMake) {
            bestMake = make;
            bestSt = st;
        }
    }

    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%lld%c", bestSt[j][k], k + 1 < o[j] ? ' ' : '\n');
    return 0;
}
