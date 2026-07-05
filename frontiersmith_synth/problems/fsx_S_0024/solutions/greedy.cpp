// TIER: greedy
// Per-tower spot greedy: staff spot at every available step (in time order) until R_i met.
// Ignores the spot price and never uses rangers.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, T, K, D;
    scanf("%d %d %d %d", &N, &T, &K, &D);
    vector<int> spot(T);
    for (int j = 0; j < T; j++) scanf("%d", &spot[j]);
    vector<int> R(N), lam(N);
    for (int i = 0; i < N; i++) scanf("%d", &R[i]);
    for (int i = 0; i < N; i++) scanf("%d", &lam[i]);
    vector<vector<char>> avail(N, vector<char>(T));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) { int a; scanf("%d", &a); avail[i][j] = (char)a; }

    vector<vector<int>> m(N, vector<int>(T, 0));
    for (int i = 0; i < N; i++) {
        int work = 0;
        for (int j = 0; j < T && work < R[i]; j++) {
            if (avail[i][j]) { m[i][j] = 1; work++; }
        }
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) printf("%d%c", m[i][j], j + 1 == T ? '\n' : ' ');
    return 0;
}
