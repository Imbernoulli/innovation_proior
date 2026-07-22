// TIER: invalid
// Deliberately infeasible: every aircraft "crosses" at tick P-1, whose offset
// (P-1) >= W is never inside an open slot (W < P), so the checker must reject it
// and score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, P, W, d;
    scanf("%d %d %d %d", &N, &P, &W, &d);
    vector<ll> rel(N), tau(N), w(N);
    vector<int> alley(N);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %lld %d", &rel[i], &tau[i], &w[i], &alley[i]);
    for (int i = 0; i < N; i++)
        printf("%lld %d\n", rel[i], P - 1); // offset P-1 >= W => not in any slot
    return 0;
}
