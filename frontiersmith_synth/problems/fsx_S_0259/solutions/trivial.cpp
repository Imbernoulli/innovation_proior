// TIER: trivial
// Lay the pre-surveyed backbone path 1-2-...-n using the cheapest direct segment
// for each consecutive pair. Always feasible (degree <= 2). Matches baseline B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, m, D;
    scanf("%d %d %d", &n, &m, &D);
    vector<int> bestIdx(n + 1, -1);
    vector<long long> bestW(n + 1, LLONG_MAX);
    for (int i = 1; i <= m; i++){
        int u, v; long long w;
        scanf("%d %d %lld", &u, &v, &w);
        if (abs(u - v) == 1){
            int lo = min(u, v);
            if (w < bestW[lo]){ bestW[lo] = w; bestIdx[lo] = i; }
        }
    }
    printf("%d\n", n - 1);
    for (int i = 1; i < n; i++) printf("%d\n", bestIdx[i]);
    return 0;
}
