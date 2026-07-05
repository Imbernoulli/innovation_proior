// TIER: greedy
// Nearest-neighbor path construction on the peak-coupling metric.
// Start at observation 0, repeatedly append the unvisited observation with
// the smallest interference to the current tail.
#include <bits/stdc++.h>
using namespace std;

int n, L;
vector<vector<int>> pat;
vector<int> diffCnt, touched;

int coupling(const vector<int>& A, const vector<int>& B) {
    touched.clear();
    int best = 0;
    for (int a : A) for (int b : B) {
        int idx = a - b + L;
        int v = ++diffCnt[idx];
        if (v == 1) touched.push_back(idx);
        if (v > best) best = v;
    }
    for (int idx : touched) diffCnt[idx] = 0;
    return best;
}

int main() {
    if (scanf("%d %d", &n, &L) != 2) return 0;
    pat.assign(n, {});
    diffCnt.assign(2 * L + 2, 0);
    for (int i = 0; i < n; i++) {
        int s; scanf("%d", &s);
        pat[i].resize(s);
        for (int j = 0; j < s; j++) scanf("%d", &pat[i][j]);
    }

    // full pairwise coupling matrix
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            int v = coupling(pat[i], pat[j]);
            C[i][j] = C[j][i] = v;
        }

    vector<char> used(n, 0);
    vector<int> order;
    int cur = 0;
    used[0] = 1; order.push_back(0);
    for (int step = 1; step < n; step++) {
        int best = -1, bestv = INT_MAX;
        for (int j = 0; j < n; j++) {
            if (used[j]) continue;
            if (C[cur][j] < bestv) { bestv = C[cur][j]; best = j; }
        }
        used[best] = 1; order.push_back(best); cur = best;
    }

    for (int i = 0; i < n; i++) printf("%d%c", order[i] + 1, i == n - 1 ? '\n' : ' ');
    return 0;
}
