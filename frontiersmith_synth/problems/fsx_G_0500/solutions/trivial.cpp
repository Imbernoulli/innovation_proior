// TIER: trivial
// Canonical baseline schedule == exactly the checker's internal baseline B.
// For each pair {i,j} (i<j) meeting in rounds r1<r2: the earlier game is hosted
// by the lower-indexed team i, the later game by the higher-indexed team j.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, R, G;
ll lambda;

int main() {
    if (scanf("%d %d %lld", &n, &R, &lambda) != 3) return 0;
    G = n / 2;
    vector<ll> X(n + 1), Y(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld %lld", &X[i], &Y[i]);
    vector<array<int,2>> gameTeams;
    vector<vector<int>> roundPos(R);
    for (int r = 0; r < R; r++)
        for (int g = 0; g < G; g++) {
            int a, b; scanf("%d %d", &a, &b);
            int pos = (int)gameTeams.size();
            gameTeams.push_back({a, b});
            roundPos[r].push_back(pos);
        }
    int P = (int)gameTeams.size();

    map<pair<int,int>, vector<pair<int,int>>> meet;
    for (int r = 0; r < R; r++)
        for (int pos : roundPos[r]) {
            int a = gameTeams[pos][0], b = gameTeams[pos][1];
            if (a > b) swap(a, b);
            meet[{a, b}].push_back({r, pos});
        }
    vector<int> home(P, 0);
    for (auto& kv : meet) {
        int i = kv.first.first, j = kv.first.second;
        auto lst = kv.second;
        sort(lst.begin(), lst.end());
        home[lst[0].second] = i;
        home[lst[1].second] = j;
    }
    for (int r = 0; r < R; r++) {
        for (int g = 0; g < G; g++) printf("%d%c", home[roundPos[r][g]], g + 1 < G ? ' ' : '\n');
    }
    return 0;
}
