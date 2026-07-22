// TIER: greedy
// The "obvious" turret-minimizing recipe: batch all punches by ascending tool
// id so the turret almost never rotates. Completely ignores the sheet's
// geometry -- if a single-cell corridor's tool comes up early, it gets
// punched immediately, stranding whatever room sits behind it.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, N, T;
    scanf("%d %d %d %d", &R, &C, &N, &T);
    for (int r = 0; r < R; r++) { char buf[64]; scanf("%s", buf); }
    vector<int> tool(N);
    for (int i = 0; i < N; i++) { int a, b; scanf("%d %d %d", &a, &b, &tool[i]); }

    vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i;
    stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return tool[a] < tool[b]; });

    for (int i : idx) printf("%d\n", i + 1);
    return 0;
}
