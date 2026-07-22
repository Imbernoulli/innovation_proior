// TIER: invalid
// Deliberately infeasible: for the LAST region of the LAST layer, emits a
// color id == C (one past the valid range [0, C-1]), which the checker must
// reject with score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int L, C;
    scanf("%d %d", &L, &C);
    vector<vector<long long>> P(C, vector<long long>(C));
    for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) scanf("%lld", &P[i][j]);

    vector<int> Rs(L);
    vector<vector<vector<int>>> cls(L);
    for (int l = 0; l < L; l++) {
        int R; scanf("%d", &R);
        Rs[l] = R;
        cls[l].resize(R);
        for (int r = 0; r < R; r++) {
            int k; scanf("%d", &k);
            cls[l][r].resize(k);
            for (int i = 0; i < k; i++) scanf("%d", &cls[l][r][i]);
        }
    }

    for (int l = 0; l < L; l++) {
        int R = Rs[l];
        for (int r = 0; r < R; r++) printf("%d%c", r, r + 1 == R ? '\n' : ' ');
        for (int r = 0; r < R; r++) {
            int col = cls[l][r][0];
            if (l == L - 1 && r == R - 1) col = C; // out of range: triggers _wa
            printf("%d%c", col, r + 1 == R ? '\n' : ' ');
        }
    }
    return 0;
}
