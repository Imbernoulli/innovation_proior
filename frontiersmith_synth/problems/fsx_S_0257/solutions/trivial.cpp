// TIER: trivial
// Serial overhaul: run every stage of every loop back-to-back on one timeline.
// Makespan = total work = the checker's baseline B, so ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<vector<int>> mach(N, vector<int>(M)), dur(N, vector<int>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            scanf("%d %d", &mach[i][j], &dur[i][j]);

    long long cursor = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (j) printf(" ");
            printf("%lld", cursor);
            cursor += dur[i][j];
        }
        printf("\n");
    }
    return 0;
}
