// TIER: invalid
// Deliberately infeasible: negative tolls (out of range) -> checker scores 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N, M, D, R; long long T;
    if(!(cin >> N >> M >> D >> T >> R)) return 0;
    for(int e = 0; e < M; e++){ if(e) putchar(' '); printf("-1"); }
    printf("\n");
    return 0;
}
