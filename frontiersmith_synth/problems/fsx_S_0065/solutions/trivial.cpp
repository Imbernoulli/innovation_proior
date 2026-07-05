// TIER: trivial
// Fully-serialized schedule: run every operation one at a time in a global
// order that respects route precedence. Makespan == sum of all durations == T,
// which is exactly the checker's baseline B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    vector<vector<ll>> dur(J, vector<ll>(M));
    for (int j = 0; j < J; j++)
        for (int o = 0; o < M; o++) {
            int b; ll d; scanf("%d %lld", &b, &d); dur[j][o] = d;
        }
    ll cursor = 0;
    for (int j = 0; j < J; j++) {
        for (int o = 0; o < M; o++) {
            printf("%lld", cursor);
            cursor += dur[j][o];
            if (o + 1 < M) printf(" ");
        }
        printf("\n");
    }
    return 0;
}
