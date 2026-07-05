// TIER: strong
// Regret-ordered greedy construction + local search (relocate & swap moves).
#include <bits/stdc++.h>
using namespace std;

int N, P;
vector<long long> C, K, remVol, remCnt;
vector<vector<long long>> val, vol;
vector<int> a; // a[i] in [0,P], 0 = dry

int main() {
    if (scanf("%d %d", &N, &P) != 2) return 0;
    C.assign(P, 0); K.assign(P, 0);
    for (int j = 0; j < P; j++) scanf("%lld %lld", &C[j], &K[j]);
    val.assign(N, vector<long long>(P));
    vol.assign(N, vector<long long>(P));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            scanf("%lld %lld", &val[i][j], &vol[i][j]);

    remVol = C; remCnt = K;
    a.assign(N, 0);

    // ---- regret ordering: prioritise zones with a big gap between best and 2nd-best gate ----
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    vector<long long> regret(N, 0), bestv(N, 0);
    for (int i = 0; i < N; i++) {
        long long b1 = -1, b2 = -1;
        for (int j = 0; j < P; j++) {
            long long v = val[i][j];
            if (v > b1) { b2 = b1; b1 = v; }
            else if (v > b2) b2 = v;
        }
        if (b2 < 0) b2 = 0;
        bestv[i] = b1;
        regret[i] = b1 - b2;
    }
    sort(order.begin(), order.end(), [&](int x, int y) {
        if (regret[x] != regret[y]) return regret[x] > regret[y];
        return bestv[x] > bestv[y];
    });

    // ---- constructive assignment: each zone to feasible gate with max value (tie: min vol) ----
    for (int idx = 0; idx < N; idx++) {
        int i = order[idx];
        int best = -1; long long bV = -1, bW = 0;
        for (int j = 0; j < P; j++) {
            if (remCnt[j] >= 1 && vol[i][j] <= remVol[j]) {
                if (val[i][j] > bV || (val[i][j] == bV && vol[i][j] < bW)) {
                    bV = val[i][j]; bW = vol[i][j]; best = j;
                }
            }
        }
        if (best >= 0) {
            remVol[best] -= vol[i][best];
            remCnt[best] -= 1;
            a[i] = best + 1;
        }
    }

    // ---- local search: relocate moves ----
    auto cur = [&](int i) -> long long { return a[i] ? val[i][a[i] - 1] : 0LL; };
    for (int pass = 0; pass < 80; pass++) {
        bool improved = false;
        // relocate: move zone i to a better feasible gate (freeing its current one)
        for (int i = 0; i < N; i++) {
            int ci = a[i] - 1; // -1 if dry
            long long curV = cur(i);
            // temporarily free
            if (ci >= 0) { remVol[ci] += vol[i][ci]; remCnt[ci] += 1; }
            int best = ci; long long bestGain = 0, bW = (ci >= 0 ? vol[i][ci] : 0);
            for (int j = 0; j < P; j++) {
                if (remCnt[j] >= 1 && vol[i][j] <= remVol[j]) {
                    long long gain = val[i][j] - curV;
                    if (gain > bestGain || (gain == bestGain && gain > 0 && vol[i][j] < bW)) {
                        bestGain = gain; best = j; bW = vol[i][j];
                    }
                }
            }
            if (best != ci && bestGain > 0) {
                remVol[best] -= vol[i][best];
                remCnt[best] -= 1;
                a[i] = best + 1;
                improved = true;
            } else if (ci >= 0) {
                // restore original
                remVol[ci] -= vol[i][ci];
                remCnt[ci] -= 1;
            }
        }
        // swap: exchange gates of two zones if total value improves and stays feasible
        for (int i = 0; i < N; i++) {
            int ci = a[i] - 1;
            if (ci < 0) continue;
            for (int k = i + 1; k < N; k++) {
                int ck = a[k] - 1;
                if (ck < 0 || ck == ci) continue;
                long long before = val[i][ci] + val[k][ck];
                long long after = val[i][ck] + val[k][ci];
                if (after <= before) continue;
                // feasibility after swap
                long long nvi = remVol[ci] + vol[i][ci] - vol[k][ci];
                long long nvk = remVol[ck] + vol[k][ck] - vol[i][ck];
                if (nvi < 0 || nvk < 0) continue; // counts unchanged, so cnt ok
                remVol[ci] = nvi; remVol[ck] = nvk;
                a[i] = ck + 1; a[k] = ci + 1;
                improved = true;
                break; // a[i] changed -> cached ci is now stale; refresh on next outer i
            }
        }
        if (!improved) break;
    }

    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
