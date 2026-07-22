// TIER: greedy
// The obvious approach: scanline (row-major); at each cell place the tile that
// maximizes agreement with the already-placed LEFT and TOP neighbors, tie-break by
// lowest index. This locks in local matches -- but at a forced-defect cell the unique
// best-matching tile is the poison DECOY, which desynchronizes the rest of the row and
// column from the phase field: a cascade of downstream losses.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int R, C, T;
    if (scanf("%d %d %d", &R, &C, &T) != 3) return 0;
    vector<int> W(T), E(T), N(T), S(T);
    for (int i = 0; i < T; i++)
        scanf("%d %d %d %d", &W[i], &E[i], &N[i], &S[i]);

    vector<vector<int>> g(R, vector<int>(C, 0));
    for (int r = 0; r < R; r++){
        for (int c = 0; c < C; c++){
            int bestScore = -1, bestIdx = 0;
            int leftE = (c > 0) ? E[g[r][c - 1]] : INT_MIN;
            int topS  = (r > 0) ? S[g[r - 1][c]] : INT_MIN;
            for (int t = 0; t < T; t++){
                int s = 0;
                if (c > 0 && W[t] == leftE) s++;
                if (r > 0 && N[t] == topS)  s++;
                if (s > bestScore){ bestScore = s; bestIdx = t; }
            }
            g[r][c] = bestIdx;
        }
    }
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            printf("%d%c", g[r][c], c + 1 < C ? ' ' : '\n');
    return 0;
}
