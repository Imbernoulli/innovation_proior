// TIER: trivial
// Do-nothing: seat guests 1,2,...,N in input order. Reproduces the checker's
// own baseline construction exactly (ratio ~= 0.1).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K, M;
    if (!(cin >> N >> K >> M)) return 0;
    for (int i = 0; i < N; i++) { int p; cin >> p; }
    for (int i = 0; i < M; i++) { int u, v, c; cin >> u >> v >> c; }
    for (int g = 1; g <= N; g++) cout << g << (g < N ? ' ' : '\n');
    return 0;
}
