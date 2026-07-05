// TIER: trivial
// Install a beacon on every artifact cell -> exactly the checker's baseline B, ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, r;
    if (scanf("%d %d", &N, &r) != 2) return 0;
    vector<int> art;
    for (int i = 1; i <= N; i++) {
        long long a, b, c, d;
        scanf("%lld %lld %lld %lld", &a, &b, &c, &d);
        if (d == 1) art.push_back(i);
    }
    printf("%d\n", (int)art.size());
    for (size_t i = 0; i < art.size(); i++)
        printf("%d%c", art[i], i + 1 == art.size() ? '\n' : ' ');
    if (art.empty()) printf("\n");
    return 0;
}
