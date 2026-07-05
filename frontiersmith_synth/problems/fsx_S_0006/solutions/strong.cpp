// TIER: strong
// Value greedy seed + local search: moves, admissions, and swaps until stable.
#include <bits/stdc++.h>
using namespace std;

int N, P;
vector<long long> C;
vector<vector<long long>> v, w;

int main() {
    scanf("%d %d", &N, &P);
    C.assign(P, 0);
    for (int j = 0; j < P; j++) scanf("%lld", &C[j]);
    v.assign(N, vector<long long>(P));
    w.assign(N, vector<long long>(P));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++) scanf("%lld %lld", &v[i][j], &w[i][j]);

    // seed: value-first greedy over (school,pool) pairs
    struct Cand { double key; int i, j; };
    vector<Cand> cand;
    cand.reserve((size_t)N * P);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            cand.push_back({(double)v[i][j] / (double)w[i][j] + 0.001 * v[i][j], i, j});
    sort(cand.begin(), cand.end(), [](const Cand& x, const Cand& y) { return x.key > y.key; });

    vector<int> a(N, 0);          // a[i] = pool index (1..P) or 0
    vector<long long> rem = C;
    for (auto& c : cand) {
        if (a[c.i] != 0) continue;
        if (w[c.i][c.j] <= rem[c.j]) { rem[c.j] -= w[c.i][c.j]; a[c.i] = c.j + 1; }
    }

    auto placeCost = [&](int i, int cur) -> long long { // value if currently assigned
        return cur == 0 ? 0 : v[i][cur - 1];
    };

    bool improved = true;
    int guard = 0;
    while (improved && guard++ < 60) {
        improved = false;

        // 1) move / admit: put school i into the pool that maximizes value gain
        for (int i = 0; i < N; i++) {
            int cur = a[i];
            long long curVal = placeCost(i, cur);
            long long remCur = cur == 0 ? 0 : rem[cur - 1] + w[i][cur - 1];
            int bestJ = cur;
            long long bestGain = 0;
            for (int j = 1; j <= P; j++) {
                long long avail = (cur == j) ? remCur : rem[j - 1];
                if (w[i][j - 1] <= avail) {
                    long long gain = v[i][j - 1] - curVal;
                    if (gain > bestGain) { bestGain = gain; bestJ = j; }
                }
            }
            if (bestJ != cur) {
                if (cur != 0) rem[cur - 1] += w[i][cur - 1];
                rem[bestJ - 1] -= w[i][bestJ - 1];
                a[i] = bestJ;
                improved = true;
            }
        }

        // 2) swaps: exchange two assigned schools between their pools if total value rises
        for (int i = 0; i < N; i++) {
            if (a[i] == 0) continue;
            for (int k = i + 1; k < N; k++) {
                if (a[k] == 0 || a[k] == a[i]) continue;
                int ji = a[i] - 1, jk = a[k] - 1;
                // try moving i->jk and k->ji
                long long remJi = rem[ji] + w[i][ji];
                long long remJk = rem[jk] + w[k][jk];
                if (w[k][ji] <= remJi && w[i][jk] <= remJk) {
                    long long before = v[i][ji] + v[k][jk];
                    long long after  = v[i][jk] + v[k][ji];
                    if (after > before) {
                        rem[ji] = remJi - w[k][ji];
                        rem[jk] = remJk - w[i][jk];
                        int tmp = a[i]; a[i] = a[k]; a[k] = tmp;
                        improved = true;
                    }
                }
            }
        }
    }

    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
