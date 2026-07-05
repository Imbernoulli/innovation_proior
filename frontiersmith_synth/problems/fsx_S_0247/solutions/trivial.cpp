// TIER: trivial
// Pole-chain baseline: wire the light poles in input index order into one path,
// ignoring junction cabinets. This is exactly the checker's baseline construction,
// so it scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    vector<int> poles;
    for (int i = 0; i < N; i++) {
        long long x, y, t, cap;
        scanf("%lld %lld %lld %lld", &x, &y, &t, &cap);
        if (t) poles.push_back(i + 1);
    }
    int M = (int)poles.size();
    printf("%d\n", max(0, M - 1));
    for (int j = 0; j + 1 < M; j++)
        printf("%d %d\n", poles[j], poles[j + 1]);
    return 0;
}
