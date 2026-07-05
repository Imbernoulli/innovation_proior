// TIER: strong
// Two constructions (value-first greedy and value/footprint-density greedy), each refined
// by local search (best-value reassignment passes + capacity-aware pairwise swaps),
// keeping the better of the two total values.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, R;
vector<ll> C;
vector<vector<int>> V, D;

ll objective(const vector<int>& a) {
    ll F = 0;
    for (int i = 1; i <= H; i++) if (a[i]) F += V[i][a[i]];
    return F;
}

void localSearch(vector<int>& a) {
    vector<ll> used(R + 1, 0);
    for (int i = 1; i <= H; i++) if (a[i]) used[a[i]] += D[i][a[i]];

    bool improved = true;
    int passes = 0;
    const int maxPasses = 30;
    while (improved && passes < maxPasses) {
        improved = false;
        passes++;

        // best-value reassignment for each herd (including unassign option)
        for (int i = 1; i <= H; i++) {
            int cur = a[i];
            ll curVal = cur ? V[i][cur] : 0;
            int bestJ = 0; ll bestVal = 0;   // unassigned baseline
            for (int j = 1; j <= R; j++) {
                ll avail = C[j] - used[j] + (j == cur ? (ll)D[i][cur] : 0);
                if ((ll)D[i][j] <= avail && V[i][j] > bestVal) { bestVal = V[i][j]; bestJ = j; }
            }
            if (bestVal > curVal && bestJ != cur) {
                if (cur) used[cur] -= D[i][cur];
                if (bestJ) used[bestJ] += D[i][bestJ];
                a[i] = bestJ;
                improved = true;
            }
        }

        // capacity-aware pairwise swaps (both herds currently assigned); limited to
        // moderate H to stay within the time limit.
        if (H <= 600) {
            for (int i = 1; i <= H; i++) {
                int ci = a[i];
                if (!ci) continue;
                for (int k = i + 1; k <= H; k++) {
                    int ck = a[k];
                    if (!ck || ck == ci) continue;
                    ll gain = (ll)V[i][ck] + V[k][ci] - V[i][ci] - V[k][ck];
                    if (gain <= 0) continue;
                    ll nUsedCi = used[ci] - D[i][ci] + D[k][ci];
                    ll nUsedCk = used[ck] - D[k][ck] + D[i][ck];
                    if (nUsedCi > C[ci] || nUsedCk > C[ck]) continue;
                    used[ci] = nUsedCi; used[ck] = nUsedCk;
                    a[i] = ck; a[k] = ci;
                    improved = true;
                    ci = a[i];   // herd i now sits on the old route of k; keep local state consistent
                }
            }
        }
    }
}

vector<int> constructValue() {
    vector<ll> rem = C;
    vector<array<int,3>> pairs;
    pairs.reserve((size_t)H * R);
    for (int i = 1; i <= H; i++)
        for (int j = 1; j <= R; j++)
            pairs.push_back({V[i][j], i, j});
    sort(pairs.begin(), pairs.end(), [](const array<int,3>& a, const array<int,3>& b){ return a[0] > b[0]; });
    vector<int> a(H + 1, 0);
    for (auto& p : pairs) {
        int i = p[1], j = p[2];
        if (a[i]) continue;
        if ((ll)D[i][j] <= rem[j]) { a[i] = j; rem[j] -= D[i][j]; }
    }
    return a;
}

vector<int> constructDensity() {
    vector<ll> rem = C;
    vector<pair<double,pair<int,int>>> pairs;
    pairs.reserve((size_t)H * R);
    for (int i = 1; i <= H; i++)
        for (int j = 1; j <= R; j++)
            pairs.push_back({(double)V[i][j] / (double)D[i][j], {i, j}});
    sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
    vector<int> a(H + 1, 0);
    for (auto& p : pairs) {
        int i = p.second.first, j = p.second.second;
        if (a[i]) continue;
        if ((ll)D[i][j] <= rem[j]) { a[i] = j; rem[j] -= D[i][j]; }
    }
    return a;
}

int main() {
    scanf("%d %d", &H, &R);
    C.assign(R + 1, 0);
    for (int j = 1; j <= R; j++) scanf("%lld", &C[j]);
    V.assign(H + 1, vector<int>(R + 1));
    D.assign(H + 1, vector<int>(R + 1));
    for (int i = 1; i <= H; i++)
        for (int j = 1; j <= R; j++) scanf("%d %d", &V[i][j], &D[i][j]);

    vector<int> a1 = constructValue();   localSearch(a1);
    vector<int> a2 = constructDensity(); localSearch(a2);
    vector<int>& best = (objective(a1) >= objective(a2)) ? a1 : a2;

    for (int i = 1; i <= H; i++) printf("%d\n", best[i]);
    return 0;
}
