// TIER: trivial
// Do-nothing baseline: print each layer's regions in the LISTED (input)
// order, and for every region use the FIRST color in its tolerance class.
// Never reorders, never recolors. This matches the checker's internal
// baseline B exactly.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int L, C;
    scanf("%d %d", &L, &C);
    vector<vector<long long>> P(C, vector<long long>(C));
    for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) scanf("%lld", &P[i][j]);

    for (int l = 0; l < L; l++) {
        int R; scanf("%d", &R);
        vector<vector<int>> cls(R);
        for (int r = 0; r < R; r++) {
            int k; scanf("%d", &k);
            cls[r].resize(k);
            for (int i = 0; i < k; i++) scanf("%d", &cls[r][i]);
        }
        for (int r = 0; r < R; r++) printf("%d%c", r, r + 1 == R ? '\n' : ' ');
        for (int r = 0; r < R; r++) printf("%d%c", cls[r][0], r + 1 == R ? '\n' : ' ');
    }
    return 0;
}
