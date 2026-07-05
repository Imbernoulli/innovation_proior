// TIER: strong
// Start from the feasible OD-EDF baseline, then re-cover each section (earliest deadline
// first) using the cheapest available slots (cost-per-unit greedy over spot vs on-demand),
// drawing only from that section's own slots plus free/pause slots before its deadline.
// A feasibility guard only takes a low-capacity spot window when the remaining slots can
// still cover the residual with on-demand, so every section stays feasible while cheap
// sub-D spot windows and idle slots the plain greedy ignores get exploited.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, T; long long D, OD;
    scanf("%d %d %lld %lld", &N, &T, &D, &OD);
    vector<long long> cap(T + 1), sc(T + 1);
    for (int s = 1; s <= T; s++) scanf("%lld %lld", &cap[s], &sc[s]);
    vector<long long> W(N + 1), d(N + 1), k(N + 1);
    for (int i = 1; i <= N; i++) {
        scanf("%lld %lld", &W[i], &d[i]);
        k[i] = (W[i] + D - 1) / D;
    }

    vector<int> mode(T + 1, 0), sec(T + 1, 0);
    int st = 1;
    for (int i = 1; i <= N; i++)
        for (long long c = 0; c < k[i]; c++) { mode[st] = 2; sec[st] = i; st++; }

    for (int i = 1; i <= N; i++) {
        int dl = (int)d[i];
        vector<int> S;
        for (int t = 1; t <= dl; t++) {
            if (sec[t] == i) { mode[t] = 0; sec[t] = 0; }   // release own OD slots
            if (mode[t] == 0) S.push_back(t);               // now free
        }
        struct Off { double cpu; long long deliver, cost; int t, m; };
        vector<Off> offs; offs.reserve(S.size());
        for (int t : S) {
            long long bd = D, bc = OD; int bm = 2; double bcpu = (double)OD / (double)D;
            if (cap[t] > 0) {
                double cpuS = (double)sc[t] / (double)cap[t];
                if (cpuS < bcpu || (cpuS == bcpu && sc[t] < bc)) {
                    bd = cap[t]; bc = sc[t]; bm = 1; bcpu = cpuS;
                }
            }
            offs.push_back({bcpu, bd, bc, t, bm});
        }
        sort(offs.begin(), offs.end(), [](const Off& a, const Off& b) {
            if (a.cpu != b.cpu) return a.cpu < b.cpu;
            return a.t < b.t;
        });
        long long got = 0;
        int M = (int)offs.size();
        for (int p = 0; p < M; p++) {
            if (got >= W[i]) break;
            long long deliver = offs[p].deliver, cost = offs[p].cost; int m = offs[p].m;
            long long need = W[i] - got;
            if (m == 1) {
                // will the remaining slots (all as OD) still cover the residual?
                long long restCap = (long long)(M - p - 1) * D;
                if (need - deliver > restCap) { deliver = D; cost = OD; m = 2; }
            }
            mode[offs[p].t] = m; sec[offs[p].t] = i;
            got += deliver;
        }
    }

    for (int s = 1; s <= T; s++) printf("%d %d\n", mode[s], sec[s]);
    return 0;
}
